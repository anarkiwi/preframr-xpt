#!/usr/bin/env python3
"""Per-class + per-tier accuracy audit; tracks the apush4x content-vs-structural signature."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def audit(predicted, ground_truth, tier_map):
    """Per-class + per-tier next-token accuracy keyed by the ground-truth token's tier (``tier_map``: token_id -> tier_name; unmapped ids bucket as ``_unknown``). Returns per-class + per-tier {n, correct, acc}, total ``n_positions`` (min of the two lengths), and ``content_over_structural`` (content acc / structural acc, the apush4x collapse signature; 0.0 when structural acc is 0)."""
    n = min(len(predicted), len(ground_truth))
    per_class = {}
    per_tier = {}
    for i in range(n):
        gt = ground_truth[i]
        correct = int(predicted[i] == gt)
        cls = per_class.setdefault(gt, {"n": 0, "correct": 0})
        cls["n"] += 1
        cls["correct"] += correct
        tier = tier_map.get(gt, "_unknown")
        bucket = per_tier.setdefault(tier, {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += correct
    for bucket in (*per_class.values(), *per_tier.values()):
        bucket["acc"] = bucket["correct"] / bucket["n"] if bucket["n"] else 0.0
    struct_acc = per_tier.get("structural", {}).get("acc", 0.0)
    content_acc = per_tier.get("content", {}).get("acc", 0.0)
    cos = content_acc / struct_acc if struct_acc > 0 else 0.0
    return {
        "n_positions": n,
        "per_class": {str(k): v for k, v in per_class.items()},
        "per_tier": per_tier,
        "content_over_structural": cos,
    }


def _load_int_csv(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    start = 0 if rows[0][0].lstrip("-").isdigit() else 1
    return [int(r[0]) for r in rows[start:] if r and r[0].strip()]


def _load_tier_map(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    start = 0 if rows[0][0].lstrip("-").isdigit() else 1
    return {int(r[0]): r[1] for r in rows[start:] if r and len(r) >= 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predicted", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument(
        "--tier-map",
        type=Path,
        required=True,
        help="CSV: token_id,tier_name. Tier names: structural, mid, content, zero.",
    )
    ap.add_argument("--out", type=Path, default=None)
    cli = ap.parse_args()
    predicted = _load_int_csv(cli.predicted)
    ground_truth = _load_int_csv(cli.ground_truth)
    tier_map = _load_tier_map(cli.tier_map)
    result = audit(predicted, ground_truth, tier_map)
    text = json.dumps(result, indent=2)
    if cli.out is not None:
        cli.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
