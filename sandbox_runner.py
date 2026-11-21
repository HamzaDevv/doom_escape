"""Restricted Linux subprocess entry point. Input contains one case and no answer key."""

import bisect
import collections
import copy
import ctypes
import decimal
import errno
import fractions
import functools
import heapq
import itertools
import json
import math
import operator
import os
import random
import re
import resource
import socket
import statistics
import string
import sys
import typing

MODULES = {
    mod.__name__: mod for mod in (
        bisect, collections, copy, decimal, fractions, functools, heapq, itertools,
        math, operator, re, statistics, string, typing,
    )
}

# Fresh, isolated Python loads its libraries before the filter. No syscall for opening
# files, creating processes, networking, inspecting other processes, or devices is allowed.
ALLOWED_SYSCALLS = (
    "read", "write", "close", "fstat", "lseek", "mmap", "mprotect", "munmap",
    "mremap", "brk", "madvise", "futex", "rt_sigaction", "rt_sigprocmask",
    "rt_sigreturn", "clock_gettime", "exit", "exit_group",
)


def restrict():
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 << 10, 32 << 10))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = (ctypes.c_uint32,)
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint)
    lib.seccomp_load.argtypes = (ctypes.c_void_p,)
    lib.seccomp_release.argtypes = (ctypes.c_void_p,)
    allow = 0x7FFF0000
    deny = 0x00050000 | errno.EPERM
    context = lib.seccomp_init(deny)
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        for name in ALLOWED_SYSCALLS:
            syscall = lib.seccomp_syscall_resolve_name(name.encode())
            if syscall < 0 or lib.seccomp_rule_add(context, allow, syscall, 0) != 0:
                raise RuntimeError(f"Cannot allow syscall {name}")
        if lib.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        lib.seccomp_release(context)


def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in MODULES:
        raise ImportError("module unavailable in the verifier")
    return MODULES[name]


def isolation_probe():
    for operation in (
        lambda: os.open("/proc/self/environ", os.O_RDONLY),
        lambda: os.fork(),
        lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM),
    ):
        try:
            operation()
        except PermissionError:
            continue
        os._exit(78)
    print("isolated", flush=True)


def main():
    request = json.loads(sys.stdin.buffer.read(24_000))
    random.seed(0)
    try:
        restrict()
    except Exception:
        os._exit(78)
    if request.get("probe"):
        isolation_probe()
        return
    builtins = dict(vars(__import__("builtins")))
    builtins.pop("open", None)
    builtins["__import__"] = restricted_import
    namespace = {"__name__": "candidate", "__builtins__": builtins}
    exec(compile(request["code"], "<candidate>", "exec"), namespace)
    result = namespace["solve"](*request["args"])
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
