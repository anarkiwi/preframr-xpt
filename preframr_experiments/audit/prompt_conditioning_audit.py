#!/usr/bin/env python3
"""Detect prompt-ignoring pathology: compare output diversity across prompt groups."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _load_int_csv(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    start = 0 if rows[0][0].lstrip("-").isdigit() else 1
    return [int(r[0]) for r in rows[start:] if r and r[0].strip()]


def jaccard(a, b):
    """Token-set Jaccard similarity on two streams."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def ngram_overlap(a, b, n: int = 4):
    """N-gram set overlap (Jaccard on n-grams)."""
    if len(a) < n or len(b) < n:
        return 0.0
    na = {tuple(a[i : i + n]) for i in range(len(a) - n + 1)}
    nb = {tuple(b[i : i + n]) for i in range(len(b) - n + 1)}
    union = na | nb
    return len(na & nb) / len(union) if union else 0.0


def audit_group(streams):
    """Pairwise jaccard + ngram-overlap within a group; return mean + std."""
    import math

    pairs_j, pairs_n = [], []
    for i in range(len(streams)):
        for j in range(i + 1, len(streams)):
            pairs_j.append(jaccard(streams[i], streams[j]))
            pairs_n.append(ngram_overlap(streams[i], streams[j], n=4))
    if not pairs_j:
        return {"n_pairs": 0, "mean_jaccard": 0.0, "mean_ngram4": 0.0}
    mean_j = sum(pairs_j) / len(pairs_j)
    mean_n = sum(pairs_n) / len(pairs_n)
    var_j = sum((p - mean_j) ** 2 for p in pairs_j) / len(pairs_j)
    return {
        "n_pairs": len(pairs_j),
        "mean_jaccard": mean_j,
        "std_jaccard": math.sqrt(var_j),
        "mean_ngram4": mean_n,
    }


def audit(real_paths, random_paths, perturbed_paths=None):
    """Compare diversity across prompt groups. Low ratio = model ignores prompt."""
    real_streams = [_load_int_csv(p) for p in real_paths]
    random_streams = [_load_int_csv(p) for p in random_paths]
    groups = {
        "real": audit_group(real_streams),
        "random": audit_group(random_streams),
    }
    if perturbed_paths:
        groups["perturbed"] = audit_group([_load_int_csv(p) for p in perturbed_paths])
    real_div = 1.0 - groups["real"]["mean_jaccard"]
    random_div = 1.0 - groups["random"]["mean_jaccard"]
    diversity_ratio = real_div / random_div if random_div > 1e-9 else 0.0
    return {
        "groups": groups,
        "diversity_ratio": diversity_ratio,
        "verdict": "prompt_ignored" if diversity_ratio < 1.2 else "prompt_used",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--real",
        nargs="+",
        type=Path,
        required=True,
        help="Generated token CSVs from real eval prompts.",
    )
    ap.add_argument(
        "--random",
        nargs="+",
        type=Path,
        required=True,
        help="Generated token CSVs from random-token prompts.",
    )
    ap.add_argument(
        "--perturbed",
        nargs="*",
        type=Path,
        default=None,
        help="Optional: generated CSVs from token-perturbed real prompts.",
    )
    ap.add_argument("--out", type=Path, default=None)
    cli = ap.parse_args()
    result = audit(cli.real, cli.random, cli.perturbed)
    text = json.dumps(result, indent=2)
    if cli.out is not None:
        cli.out.write_text(text)
    print(text)
    return 1 if result["verdict"] == "prompt_ignored" else 0


if __name__ == "__main__":
    sys.exit(main())
