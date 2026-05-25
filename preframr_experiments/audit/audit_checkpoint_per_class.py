#!/usr/bin/env python3
"""Post-hoc per-tier accuracy audit on a saved Lightning checkpoint."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

from preframr_tokens import tier_accuracy
from preframr.train.model import Model, build_tier_map


def _iter_eval_blocks(work_dir: Path, max_blocks_per_subset: int):
    """Yield (subset_name, block_array) from each eval_*/*/*.0.blocks.npy in work_dir."""
    subsets = sorted(p for p in work_dir.iterdir() if p.name.startswith("eval"))
    for subset_dir in subsets:
        files = sorted(glob.glob(str(subset_dir / "*" / "*.0.blocks.npy")))
        if not files:
            continue
        emitted = 0
        for f in files:
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr[None, :]
            for row in arr:
                yield subset_dir.name, row
                emitted += 1
                if max_blocks_per_subset and emitted >= max_blocks_per_subset:
                    break
            if max_blocks_per_subset and emitted >= max_blocks_per_subset:
                break


def audit(ckpt_path: Path, work_dir: Path, max_blocks_per_subset: int, device: str):
    """Load ckpt, forward eval blocks, bucket predictions by tier, return per-tier accuracy."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    args = copy.deepcopy(hparams["args"])
    args.compile = False
    model = Model(
        args,
        hparams["n_vocab"],
        hparams["tokens"],
        hparams["tkmodel"],
        hparams.get("metadata"),
        reg_widths=hparams.get("reg_widths"),
    )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model = model.to(device)
    tier_map = build_tier_map(args, model.n_vocab, model.tokens, model.tkmodel)
    per_subset_preds: dict[str, list] = {}
    per_subset_gt: dict[str, list] = {}
    with torch.inference_mode():
        for name, block in _iter_eval_blocks(work_dir, max_blocks_per_subset):
            x = torch.from_numpy(block[:-1]).long().unsqueeze(0).to(device)
            y = block[1:]
            logits = model.model(x)
            if isinstance(logits, list):
                pred_ids = torch.cat([c.argmax(dim=-1) for c in logits], dim=1)
            else:
                pred_ids = logits.argmax(dim=-1)
            per_subset_preds.setdefault(name, []).extend(
                int(t) for t in pred_ids.flatten().tolist()
            )
            per_subset_gt.setdefault(name, []).extend(int(t) for t in y.tolist())
    out: dict = {
        "ckpt": str(ckpt_path),
        "subsets": {},
    }
    for name in sorted(per_subset_preds):
        out["subsets"][name] = tier_accuracy(
            per_subset_preds[name], per_subset_gt[name], tier_map
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="The arm-seed work_dir containing dataset.csv.zst + eval_*/ subdirs.",
    )
    ap.add_argument(
        "--max-blocks",
        type=int,
        default=8,
        help="Stop after N blocks per subset (8 default; 0 = whole eval set).",
    )
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--out", type=Path, default=None)
    cli = ap.parse_args()
    result = audit(cli.ckpt, cli.work_dir, cli.max_blocks, cli.device)
    text = json.dumps(result, indent=2, default=str)
    if cli.out is not None:
        cli.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
