"""Authenticated loopback FastAPI worker with serialized in-place LoRA replacement."""

import argparse
import hmac
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from swarm_common import (
    ADAPTERS, CONTEXT, MAX_CODE_TOKENS, MODEL, MODEL_REVISION, PORTS, adapter_fingerprint,
)


class InferRequest(BaseModel):
    system: str = Field(min_length=1, max_length=2048)
    user: str = Field(min_length=1, max_length=2500)
    seed: int = Field(ge=0, lt=2**32)
    temperature: float = Field(ge=0, le=1)
    max_tokens: int = Field(ge=8, le=MAX_CODE_TOKENS)


Version = Literal["seed_v0", "gen1"]


class SwapRequest(BaseModel):
    version: Version
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Engine:
    def __init__(self, port: int, successor: Path):
        self.port = port
        self.paths = {"seed_v0": ADAPTERS / "seed_v0", "gen1": successor}
        self.lock = threading.Lock()
        self.active = ("seed_v0", "")
        self.status, self.calls = "LOADING", 0

    def load(self):
        import torch
        from peft import PeftModel
        from peft.utils.hotswap import hotswap_adapter
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.cuda.set_per_process_memory_fraction(0.18, device=0)
        self.torch, self.hotswap = torch, hotswap_adapter
        version = "seed_v0"
        self.active = (version, adapter_fingerprint(self.paths[version]))
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=MODEL_REVISION, local_files_only=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=MODEL_REVISION, local_files_only=True,
            device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        if not getattr(base, "is_loaded_in_4bit", False):
            raise RuntimeError("Worker base was not loaded in 4-bit mode.")
        self.model = PeftModel.from_pretrained(base, str(self.paths[version]), is_trainable=False)
        self.model.eval()
        self.status = "ONLINE"

    def health(self):
        version, fingerprint = self.active
        return {
            "port": self.port, "pid": os.getpid(), "status": self.status,
            "version": version, "sha256": fingerprint, "calls": self.calls,
        }

    def infer(self, request):
        with self.lock:
            self.status = "INFERRING"
            try:
                tokens = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": request.system},
                     {"role": "user", "content": request.user}],
                    tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                ).to("cuda:0")
                length = tokens["input_ids"].shape[-1]
                if length + request.max_tokens > CONTEXT:
                    raise HTTPException(422, "Prompt plus completion exceeds the worker context.")
                self.torch.manual_seed(request.seed)
                sampling = ({"do_sample": True, "temperature": request.temperature, "top_p": 0.95}
                            if request.temperature else {"do_sample": False, "temperature": None, "top_p": None,
                                                         "top_k": None})
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **tokens, max_new_tokens=request.max_tokens, **sampling,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                text = self.tokenizer.decode(generated[0, length:], skip_special_tokens=True)
                self.calls += 1
                return {**self.health(), "text": text}
            finally:
                self.status = "ONLINE"

    def swap(self, request):
        path = self.paths[request.version]
        fingerprint = adapter_fingerprint(path)
        if fingerprint != request.sha256:
            raise HTTPException(409, "Checkpoint SHA-256 mismatch.")
        with self.lock:
            previous, previous_hash = self.active
            if request.version == previous and fingerprint == previous_hash:
                return self.health()
            self.status = "SYNCING"
            try:
                self.hotswap(self.model, str(path), adapter_name="default", torch_device="cuda:0")
                self.torch.cuda.synchronize()
                self.active = (request.version, fingerprint)
            except Exception as error:
                self.hotswap(self.model, str(self.paths[previous]), adapter_name="default", torch_device="cuda:0")
                self.torch.cuda.synchronize()
                raise HTTPException(500, "Adapter update failed and was rolled back.") from error
            finally:
                self.status = "ONLINE"
            return self.health()


def create_app(engine, token):
    @asynccontextmanager
    async def lifespan(_):
        engine.load()
        yield

    def authorized(x_swarm_token: Annotated[str | None, Header()] = None):
        if not x_swarm_token or not hmac.compare_digest(x_swarm_token.encode(), token.encode()):
            raise HTTPException(401, "Invalid swarm token.")

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None,
                  dependencies=[Depends(authorized)])
    app.get("/health")(engine.health)

    @app.post("/infer")
    def infer(request: InferRequest):
        return engine.infer(request)

    @app.post("/hot_swap")
    def swap(request: SwapRequest):
        return engine.swap(request)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=PORTS, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    from swarm_common import WORKSPACE
    successor = WORKSPACE / "smoke_adapter" if args.smoke else ADAPTERS / "gen1"
    app = create_app(Engine(args.port, successor), os.environ["SWARM_TOKEN"])
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False, log_level="warning")
