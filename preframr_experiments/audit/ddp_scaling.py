#!/usr/bin/env python3
"""DDP scaling microbenchmark at prodlike body."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager

_PRODLIKE_LAYERS = 16
_PRODLIKE_HEADS = 12
_PRODLIKE_KV_HEADS = 4
_PRODLIKE_EMBED = 768
_PRODLIKE_INTERMEDIATE = 2048

_PRODLIKE_SEQ_LEN = 8192
_PRODLIKE_TKVOCAB = 131072

_RENTAL_GATE_WORLD_SIZE = 4
_RENTAL_GATE_EFFICIENCY = 0.60


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--world-size",
        type=int,
        default=None,
        help=(
            "DDP world size. Defaults to env LOCAL_WORLD_SIZE / "
            "WORLD_SIZE (set by torchrun) or 1 for direct invocation."
        ),
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Steps to measure after warmup. Default 50.",
    )
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help=(
            "Warmup steps to discard before timing. Default 10. Covers "
            "torch.compile + Inductor warmup + DDP first-iter bucket "
            "rebalancing."
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-rank batch size. Default 4 (mirrors prodlike train).",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=_PRODLIKE_SEQ_LEN,
        help=f"Sequence length. Default {_PRODLIKE_SEQ_LEN}.",
    )
    ap.add_argument(
        "--tkvocab",
        type=int,
        default=_PRODLIKE_TKVOCAB,
        help=f"Token vocab size. Default {_PRODLIKE_TKVOCAB}.",
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help=(
            "Disable torch.compile. Compile is on by default; use "
            "--no-compile if Inductor is broken in the image."
        ),
    )
    ap.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "If set, rank 0 writes the per-world-size summary JSON to "
            "this path. Useful for piping into a decision script."
        ),
    )
    return ap.parse_args()


def _resolve_world_size(cli_world_size):
    """Resolve the effective world size."""
    torchrun_ws = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("WORLD_SIZE")
    if cli_world_size is not None:
        if torchrun_ws and int(torchrun_ws) != cli_world_size:
            print(
                f"[ddp_scaling] WARN: --world-size={cli_world_size} differs from "
                f"torchrun env ({torchrun_ws}); using CLI value.",
                file=sys.stderr,
            )
        return cli_world_size
    if torchrun_ws:
        return int(torchrun_ws)
    return 1


def _rank():
    """Best-effort current rank (0 if not under torchrun)."""
    return int(os.environ.get("RANK", "0"))


def _build_model(args):
    """Construct the prodlike-shaped model. Skeleton -- the real
    implementation imports preframr.model (or wraps torchtune.llama3_2)
    with the body flags above. Kept as a stub here so the script is
    runnable for argparse smoke; flesh out before measurement.
    """
    raise NotImplementedError(
        "Skeleton: build the prodlike body via preframr.model. See "
        "train_prodlike_oom_smoke.py for the construction pattern."
    )


def _build_optimizer(model, args):
    """AdamW with prodlike defaults (lr=2e-4, wd=0.01). Skeleton."""
    raise NotImplementedError("Skeleton: AdamW(lr=2e-4, weight_decay=0.01).")


def _build_synthetic_batch(args, device):
    """One (x, y) batch of int64 token ids at the prodlike shape."""
    import torch

    batch = (args.batch_size, args.seq_len)
    x = torch.randint(0, args.tkvocab, batch, dtype=torch.long, device=device)
    y = torch.randint(0, args.tkvocab, batch, dtype=torch.long, device=device)
    return x, y


def _init_ddp(world_size):
    """Init `torch.distributed` if world_size > 1."""
    if world_size <= 1:
        return None

    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank


@contextmanager
def _timed():
    """Context manager: record wall and CUDA event times."""
    import torch

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    t_wall = time.monotonic()
    start_evt.record()
    yield_obj = {"start_evt": start_evt, "end_evt": end_evt}
    yield yield_obj
    end_evt.record()
    torch.cuda.synchronize()
    yield_obj["wall_s"] = time.monotonic() - t_wall
    yield_obj["cuda_ms"] = start_evt.elapsed_time(end_evt)


def _measure_one_world_size(args, world_size):
    """Run warmup + measured steps; return a summary dict."""
    raise NotImplementedError(
        "Skeleton: implement measurement loop. See module docstring "
        "for the contract."
    )


def _extrapolate(baseline_steps_per_s, measured_steps_per_s, world_size):
    """Extrapolate DDP efficiency to larger world sizes."""
    measured_efficiency = measured_steps_per_s / (baseline_steps_per_s * world_size)
    eff_drop = (1.0 - measured_efficiency) / (world_size - 1) if world_size > 1 else 0.0
    out = {}
    for ws in (1, 2, 4, 8):
        if ws == 1:
            out[ws] = (baseline_steps_per_s, 1.0)
            continue
        if ws == world_size:
            out[ws] = (measured_steps_per_s, measured_efficiency)
            continue
        predicted_eff = max(0.0, 1.0 - (ws - 1) * eff_drop)
        out[ws] = (baseline_steps_per_s * ws * predicted_eff, predicted_eff)
    return out


def _render_decision(extrapolated):
    """Render the rental-gate decision line. Returns (pass: bool, msg: str)."""
    target_steps_per_s, target_eff = extrapolated.get(
        _RENTAL_GATE_WORLD_SIZE, (None, None)
    )
    if target_eff is None:
        return False, "world_size=4 not extrapolated"
    passed = target_eff >= _RENTAL_GATE_EFFICIENCY
    return passed, (
        f"world_size={_RENTAL_GATE_WORLD_SIZE}: "
        f"{target_steps_per_s:.2f} steps/s, "
        f"{target_eff:.1%} efficient "
        f"(gate >= {_RENTAL_GATE_EFFICIENCY:.0%}): "
        f"{'PASS' if passed else 'FAIL'}"
    )


def main():
    args = _parse_args()
    world_size = _resolve_world_size(args.world_size)
    rank = _rank()

    print(
        f"[ddp_scaling] rank={rank} world_size={world_size} steps={args.steps} "
        f"batch_size={args.batch_size} seq_len={args.seq_len} "
        f"tkvocab={args.tkvocab} compile={not args.no_compile}",
        flush=True,
    )

    raise NotImplementedError(
        "Skeleton: real measurement loop not implemented yet. "
        "Flesh out _build_model / _build_optimizer / _measure_one_world_size "
        "before running for real."
    )


if __name__ == "__main__":
    main()
