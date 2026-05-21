#!/usr/bin/env python3
"""Pre-launch smoke: exercise the exact paths that `train.py` will hit
on epoch 0, in <30 seconds, so a broken image fails before parse/
tokenize burn ~35 min per (arm, seed).
"""

from __future__ import annotations

import sys
import traceback


def _step(label, fn):
    """Run a smoke step. On exception, print 1-line diagnostic to
    stderr + full traceback to stdout, then re-raise."""
    print(f"[preflight] {label} ...", flush=True)
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL [{label}]: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stdout)
        raise
    print(f"[preflight] {label} OK", flush=True)


def check_inductor_cpu_vec_isa():
    """The specific failure point that crashed la1/seed0 (2026-05-11).
    ``pick_vec_isa()`` shells out to `g++` -- if missing, raises
    ``InvalidCxxCompiler``."""
    from torch._inductor import cpu_vec_isa

    cpu_vec_isa.pick_vec_isa()


def check_torch_imports():
    """Top-level imports the train path depends on. Catches version
    drift, missing wheels, broken extensions."""
    # pylint: disable=import-outside-toplevel,unused-import
    import torch  # noqa: F401  pylint: disable=unused-import
    import pytorch_lightning  # noqa: F401  pylint: disable=unused-import


def check_cuda():
    """Container must see at least one CUDA device if --gpus is passed.
    Skip silently if no GPU is requested (smoke is still useful on
    CPU-only hosts)."""
    import torch

    if not torch.cuda.is_available():
        import os

        if os.environ.get("SMOKE_REQUIRE_CUDA", "0") == "1":
            raise RuntimeError("CUDA unavailable but SMOKE_REQUIRE_CUDA=1")
        print("[preflight] CUDA unavailable (CPU-only); skipping GPU compile step")
        return
    x = torch.zeros(8, device="cuda")
    _ = (x + 1).sum().item()


def check_torch_compile():
    """The actual compile path Lightning will hit on epoch 0. Exercises
    Inductor codegen + (if CUDA available) Triton."""
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)).to(device)
    compiled = torch.compile(model)
    x = torch.randn(4, 8, device=device, requires_grad=False)
    target = torch.zeros(4, 8, device=device)
    y = compiled(x)
    loss = (y - target).pow(2).mean()
    loss.backward()


def check_preframr_imports():
    """The model wiring lives in ``preframr.model``; mount over /preframr
    is what makes new args work, so this catches mount-vs-image-drift."""
    # pylint: disable=import-outside-toplevel,unused-import
    sys.path.insert(0, "/")
    import preframr.args  # noqa: F401  pylint: disable=unused-import
    import preframr.train.model  # noqa: F401  pylint: disable=unused-import


def main():
    print("[preflight] python", sys.version.split()[0], flush=True)
    try:
        _step("torch imports", check_torch_imports)
        _step("inductor cpu_vec_isa", check_inductor_cpu_vec_isa)
        _step("cuda visibility", check_cuda)
        _step("torch.compile + backward", check_torch_compile)
        _step("preframr imports", check_preframr_imports)
    except Exception:
        print("[preflight] FAILED", file=sys.stderr, flush=True)
        return 1
    print("[preflight] all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
