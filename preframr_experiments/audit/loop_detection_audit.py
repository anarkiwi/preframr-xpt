#!/usr/bin/env python3
"""Detect greedy-decode collapse to short cycles in generated token streams."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def detect_tail_cycle(tokens, tail_window=128, max_period=32, min_repeats=3):
    """Detect whether the tail of ``tokens`` collapsed into a short repeating cycle. Examines the last ``tail_window`` ids and returns ``{"period", "repeats", "cycle"}`` for the smallest period (<= ``max_period``) under which the whole window is exactly periodic with at least ``min_repeats`` repetitions, else None."""
    window = list(tokens[-tail_window:]) if tail_window else list(tokens)
    w = len(window)
    for period in range(1, max_period + 1):
        if w < period * min_repeats:
            break
        if all(window[i] == window[i - period] for i in range(period, w)):
            return {
                "period": period,
                "repeats": w // period,
                "cycle": window[:period],
            }
    return None


def _load_tokens(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        import numpy as np

        arr = np.load(path)
        return [int(x) for x in arr.flatten()]
    if suffix == ".csv":
        with open(path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return []
        first = rows[0][0]
        start = 1 if not first.lstrip("-").isdigit() else 0
        return [int(r[0]) for r in rows[start:] if r and r[0].strip()]
    text = path.read_text().split()
    return [int(x) for x in text]


def audit(paths, tail_window, max_period, min_repeats):
    """Run detection over each path; return per-file + per-arm aggregates."""
    per_file = []
    arm_buckets: dict[str, dict] = {}
    for raw in paths:
        if ":" in raw:
            arm, file_str = raw.split(":", 1)
        else:
            arm, file_str = "_default", raw
        path = Path(file_str)
        tokens = _load_tokens(path)
        verdict = detect_tail_cycle(
            tokens,
            tail_window=tail_window,
            max_period=max_period,
            min_repeats=min_repeats,
        )
        per_file.append(
            {
                "arm": arm,
                "path": str(path),
                "n_tokens": len(tokens),
                "collapsed": verdict is not None,
                "cycle": verdict,
            }
        )
        bucket = arm_buckets.setdefault(arm, {"n": 0, "collapsed": 0, "periods": []})
        bucket["n"] += 1
        if verdict is not None:
            bucket["collapsed"] += 1
            bucket["periods"].append(verdict["period"])
    per_arm = {
        arm: {
            "n_samples": b["n"],
            "n_collapsed": b["collapsed"],
            "collapse_rate": b["collapsed"] / b["n"] if b["n"] else 0.0,
            "periods": sorted(b["periods"]),
        }
        for arm, b in arm_buckets.items()
    }
    return {"per_file": per_file, "per_arm": per_arm}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "tokens",
        nargs="+",
        help="Token files (.csv|.npy|.txt). Prefix with 'arm_label:' to bucket.",
    )
    ap.add_argument("--tail-window", type=int, default=128)
    ap.add_argument("--max-period", type=int, default=32)
    ap.add_argument("--min-repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    cli = ap.parse_args()
    result = audit(cli.tokens, cli.tail_window, cli.max_period, cli.min_repeats)
    text = json.dumps(result, indent=2)
    if cli.out is not None:
        cli.out.write_text(text)
    print(text)
    any_collapsed = any(f["collapsed"] for f in result["per_file"])
    return 1 if any_collapsed else 0


if __name__ == "__main__":
    sys.exit(main())
