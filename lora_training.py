"""Canonical Unsloth QLoRA implementation for both training stages."""

import os
import shutil
import tempfile
from pathlib import Path

from swarm_common import (
    CONTEXT, CORE_MODULES, MODEL, MODEL_REVISION, SEED, adapter_fingerprint, write_json,
)


def training_rows(tokenizer, examples):
    for system, user, answer in examples:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True,
        )
        completion = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = prompt + completion
        if len(ids) > CONTEXT:
            raise ValueError(f"Training example exceeds {CONTEXT} tokens.")
        yield {"input_ids": ids, "attention_mask": [1] * len(ids),
               "labels": [-100] * len(prompt) + completion}


def train(examples, output, *, steps, learning_rate, fraction, parent, provenance):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}. Use a new run directory.")

    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    from transformers import DataCollatorForSeq2Seq, set_seed
    from trl import SFTConfig, SFTTrainer

    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    torch.cuda.reset_peak_memory_stats()
    set_seed(SEED)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, revision=MODEL_REVISION, max_seq_length=CONTEXT,
        dtype=torch.bfloat16, load_in_4bit=True, device_map={"": 0},
        local_files_only=parent is not None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=list(CORE_MODULES), use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    if parent:
        adapter_fingerprint(parent)
        loaded = set_peft_model_state_dict(model, load_peft_weights(str(parent), device="cpu"))
        if loaded.unexpected_keys or any(".lora_" in key for key in loaded.missing_keys):
            raise RuntimeError("Seed LoRA tensors did not match the continuation model.")
    FastLanguageModel.for_training(model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = Dataset.from_list(list(training_rows(tokenizer, examples)))
    staging = Path(tempfile.mkdtemp(prefix=".training-", dir=output.parent))
    try:
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=data,
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
            args=SFTConfig(
                output_dir=str(staging / "scratch"),
                dataset_kwargs={"skip_prepare_dataset": True},
                completion_only_loss=False,
                max_length=CONTEXT, packing=False,
                per_device_train_batch_size=1, gradient_accumulation_steps=4 if steps > 1 else 1,
                max_steps=steps, learning_rate=learning_rate, warmup_steps=2 if steps > 2 else 0,
                lr_scheduler_type="cosine", optim="adamw_8bit", bf16=True,
                gradient_checkpointing=True, logging_steps=1, save_strategy="no",
                report_to="none", disable_tqdm=True, seed=SEED, data_seed=SEED,
            ),
        )
        result = trainer.train()
        import math
        if result.global_step != steps or not math.isfinite(result.training_loss):
            raise RuntimeError("Training did not produce the requested finite update.")
        model.save_pretrained(staging / "adapter", safe_serialization=True)
        adapter_path = staging / "adapter"
        weight_hash = adapter_fingerprint(adapter_path)
        write_json(adapter_path / "training.json", {
            **provenance, "base_model": MODEL, "base_revision": MODEL_REVISION,
            "pid": os.getpid(), "steps": result.global_step, "rows": len(data),
            "training_loss": result.training_loss, "adapter_sha256": weight_hash,
            "peak_allocator_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
            "memory_fraction": fraction, "learning_rate": learning_rate,
        })
        if output.exists():
            raise FileExistsError(output)
        adapter_path.rename(output)
    finally:
        shutil.rmtree(staging)


def verify_checkpoint(folder, expected):
    from swarm_common import read_json

    info = read_json(folder / "training.json")
    if info["adapter_sha256"] != adapter_fingerprint(folder) or any(
        info.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"{folder} does not match its inputs. Choose a fresh run directory.")
    return info
