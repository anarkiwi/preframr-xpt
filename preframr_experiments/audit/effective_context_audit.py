#!/usr/bin/env python3
"""Effective-context probe: how much history does the checkpoint actually use?

For probe positions across held-out blocks, measures teacher-forced top-1 accuracy when
the context is truncated to the last k tokens (k in a sweep). The k at which accuracy
saturates IS the model's effective context length. This settles the gap probe's
`short_context_or_bug` branch: if accuracy saturates at small k, long-horizon
free-running is doomed by construction (the model can't use long context at all) and the
horizon/lane fixes won't help until that changes — operationalizing the causal-state /
dependency-horizon view of `learnability_token_ordering_theory.md`.

Pure measurement (curve, saturation point) is unit-tested in CI; the truncated-context
forwards (batched per k) reuse the checkpoint forward path and run on a GPU host. See
`design/generation/free_running_pathology_remediation_design.md` (Tier 0).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

DEFAULT_CTX_LENS = (16, 64, 256, 1024, 4096)


# --------------------------------------------------------------------------- #
# Pure measurement (torch-free; unit-tested)
# --------------------------------------------------------------------------- #
def context_curve(acc: dict) -> list:
    """{k:[hits,tot]} -> sorted [{k, acc, n}] (k = context length in tokens)."""
    return [
        {"k": k, "acc": acc[k][0] / acc[k][1], "n": acc[k][1]}
        for k in sorted(acc)
        if acc[k][1]
    ]


def saturation_k(curve: list, tol: float = 0.02):
    """Smallest k whose accuracy is within ``tol`` of the curve's max — the effective
    context length. None on an empty curve."""
    if not curve:
        return None
    best = max(p["acc"] for p in curve)
    for p in curve:
        if p["acc"] >= best - tol:
            return p["k"]
    return curve[-1]["k"]


def summarize(acc: dict, tol: float = 0.02) -> dict:
    curve = context_curve(acc)
    sat = saturation_k(curve, tol)
    near = curve[0]["acc"] if curve else None
    far = curve[-1]["acc"] if curve else None
    return {
        "curve": curve,
        "saturation_k": sat,
        "acc_at_min_k": near,
        "acc_at_max_k": far,
        "long_context_gain": (
            (far - near) if (near is not None and far is not None) else None
        ),
    }


# --------------------------------------------------------------------------- #
# Truncated-context forwards (torch; GPU host) — lazy-imported
# --------------------------------------------------------------------------- #
def _iter_blocks(blocks_glob: str, n_blocks: int, min_len: int, logger):
    import numpy as np  # pylint: disable=import-outside-toplevel

    out = []
    for path in sorted(glob.glob(blocks_glob, recursive=True)):
        arr = np.load(path)
        for row in np.atleast_2d(arr):
            nz = row[row > 0]
            if len(nz) < min_len:
                continue
            out.append((path, [int(t) for t in nz]))
            break
        if len(out) >= n_blocks:
            break
    logger.info("effective_context: %u blocks from %s", len(out), blocks_glob)
    return out


def _probe_positions(block_len: int, k: int, max_probes: int) -> list:
    """Evenly-spaced positions p with a full k-token context available (k <= p < len)."""
    if block_len <= k:
        return []
    lo, hi = k, block_len - 1
    if max_probes <= 0 or hi - lo + 1 <= max_probes:
        return list(range(lo, hi + 1))
    stride = (hi - lo) / (max_probes - 1)
    return sorted({lo + int(round(i * stride)) for i in range(max_probes)})


def run_audit(args, logger) -> int:
    import torch  # pylint: disable=import-outside-toplevel
    from torchtune.modules.common_utils import (
        disable_kv_cache,
    )  # pylint: disable=import-outside-toplevel

    from preframr.inference.predict import (
        load_model,
    )  # pylint: disable=import-outside-toplevel

    ctx_lens = (
        tuple(int(k) for k in args.ctx_lens.split(","))
        if args.ctx_lens
        else DEFAULT_CTX_LENS
    )
    _dataset, model, device, _ = load_model(
        args, logger
    )  # pylint: disable=unused-variable
    model = model.to(device)
    model.eval()
    blocks = _iter_blocks(args.blocks_glob, args.n_blocks, min(ctx_lens) + 2, logger)

    acc: dict = {}
    for _path, block in blocks:
        for k in ctx_lens:
            positions = _probe_positions(len(block), k, args.max_probes)
            if not positions:
                continue
            windows = [block[p - k : p] for p in positions]
            x = torch.tensor(windows, dtype=torch.long).to(device)
            with torch.inference_mode(), disable_kv_cache(model.model):
                logits = model.model(x)
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
            pred_last = logits[:, -1, :].argmax(dim=-1).cpu().tolist()
            targets = [block[p] for p in positions]
            e = acc.setdefault(k, [0, 0])
            e[0] += sum(1 for pr, tg in zip(pred_last, targets) if pr == tg)
            e[1] += len(targets)

    result = summarize(acc, args.tol)
    result["config"] = {
        "blocks_glob": args.blocks_glob,
        "n_blocks": len(blocks),
        "ctx_lens": list(ctx_lens),
        "max_probes": args.max_probes,
    }
    text = json.dumps(result, indent=2, default=str)
    if args.out is not None:
        Path(args.out).write_text(text)
    print(text)
    logger.info(
        "effective_context: saturation_k=%s long_context_gain=%s",
        result["saturation_k"],
        result["long_context_gain"],
    )
    return 0


def add_audit_args(parser):
    parser.add_argument(
        "--blocks-glob", type=str, default="/scratch/preframr/train/**/*.blocks.npy"
    )
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument(
        "--ctx-lens",
        type=str,
        default="",
        help="Comma-sep k's (default 16,64,256,1024,4096).",
    )
    parser.add_argument(
        "--max-probes", type=int, default=32, help="Positions per (block,k); 0 = all."
    )
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main():
    from preframr.args import add_args  # pylint: disable=import-outside-toplevel
    from preframr.utils import get_logger  # pylint: disable=import-outside-toplevel

    parser = add_audit_args(
        add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    )
    args = parser.parse_args()
    logger = get_logger("INFO")
    sys.exit(run_audit(args, logger))


if __name__ == "__main__":
    main()
