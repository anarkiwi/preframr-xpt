#!/usr/bin/env python3
"""Per-class + per-tier accuracy audit; tracks the apush4x content-vs-structural signature."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from preframr_tokens import tier_accuracy as audit


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
