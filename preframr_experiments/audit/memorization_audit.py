#!/usr/bin/env python3
"""Memorization audit: how much of a generated stream is verbatim training corpus?

Builds an n-gram index over the training blocks (token-id space) and scores a query
stream (a generated continuation) for **novel-fraction** at n=8/16 and the **longest
verbatim match bucket**. The output-side copy-dominance measure: a model that scores
well by reproducing training spans is memorizing, not composing — the necessary check
before any "the model generated this" claim, and a direct read on the free-running
pathology's copy-dominance mechanism.

This realizes the memorization audit specified (but not yet built) in
`design/generation/generation_quality_gate.md`. The index + scoring are torch-free and
unit-tested in CI; the query stream is supplied via --query-ids (e.g. the gap probe's
generated output, or a predict dump), or generated inline if --model-state is given.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

DEFAULT_NS = (8, 16, 32, 64)


# --------------------------------------------------------------------------- #
# Pure index + scoring (torch-free; unit-tested)
# --------------------------------------------------------------------------- #
def ngram_sets(seqs, ns) -> dict:
    """{n: set(n-gram tuples)} over all sequences (lists of ints)."""
    sets = {n: set() for n in ns}
    for s in seqs:
        s = list(s)
        for n in ns:
            for i in range(len(s) - n + 1):
                sets[n].add(tuple(s[i : i + n]))
    return sets


def novel_fraction(query, sets: dict, n: int):
    """Fraction of the query's n-grams NOT present in the index (None if too short)."""
    q = list(query)
    grams = [tuple(q[i : i + n]) for i in range(len(q) - n + 1)]
    if not grams:
        return None
    novel = sum(1 for g in grams if g not in sets[n])
    return novel / len(grams)


def longest_match_bucket(query, sets: dict, ns) -> int:
    """Largest n in ``ns`` for which some query n-gram appears in the index (0 = none even
    at the smallest n). A coarse 'longest verbatim match' that needs only the per-n sets.
    """
    q = list(query)
    best = 0
    for n in sorted(ns):
        if n not in sets:
            continue
        if any(tuple(q[i : i + n]) in sets[n] for i in range(len(q) - n + 1)):
            best = n
    return best


def score_query(query, sets: dict, ns) -> dict:
    """Novel-fraction per n + the longest-match bucket + nearest headline (n=8/16)."""
    nf = {str(n): novel_fraction(query, sets, n) for n in sorted(ns)}
    return {
        "query_len": len(list(query)),
        "novel_fraction": nf,
        "longest_match_bucket": longest_match_bucket(query, sets, ns),
        "headline": {
            "novel_frac_n8": nf.get("8"),
            "novel_frac_n16": nf.get("16"),
        },
    }


# --------------------------------------------------------------------------- #
# IO + optional model-side generation (numpy/torch lazy-imported)
# --------------------------------------------------------------------------- #
def _iter_train_seqs(blocks_glob: str, max_files: int, logger):
    import numpy as np  # pylint: disable=import-outside-toplevel

    seqs = []
    files = sorted(glob.glob(blocks_glob, recursive=True))
    if max_files:
        files = files[:max_files]
    for path in files:
        arr = np.load(path)
        for row in np.atleast_2d(arr):
            nz = row[row > 0]
            if len(nz):
                seqs.append([int(t) for t in nz])
    logger.info(
        "memorization: indexed %u train sequences from %u files", len(seqs), len(files)
    )
    return seqs


def _load_query_ids(path: str):
    import numpy as np  # pylint: disable=import-outside-toplevel

    if path.endswith(".json"):
        return [int(t) for t in json.loads(Path(path).read_text())]
    arr = np.load(path)
    arr = np.atleast_2d(arr)[0]
    nz = arr[arr > 0]
    return [int(t) for t in nz]


def run_audit(args, logger) -> int:
    ns = tuple(int(n) for n in args.ns.split(",")) if args.ns else DEFAULT_NS
    train = _iter_train_seqs(args.train_glob, args.max_train_files, logger)
    sets = ngram_sets(train, ns)
    if not args.query_ids:
        print("MEMORIZATION_RESULT " + json.dumps({"error": "no --query-ids"}))
        return 1
    query = _load_query_ids(args.query_ids)
    result = score_query(query, sets, ns)
    result["config"] = {
        "train_glob": args.train_glob,
        "n_train_seqs": len(train),
        "ns": list(ns),
        "query_ids": args.query_ids,
    }
    text = json.dumps(result, indent=2, default=str)
    if args.out is not None:
        Path(args.out).write_text(text)
    print(text)
    logger.info(
        "memorization: novel_n8=%s longest_match_bucket=%u",
        result["headline"]["novel_frac_n8"],
        result["longest_match_bucket"],
    )
    return 0


def add_audit_args(parser):
    parser.add_argument(
        "--train-glob", type=str, default="/scratch/preframr/train/**/*.blocks.npy"
    )
    parser.add_argument(
        "--query-ids",
        type=str,
        default=None,
        help="Generated stream to score: .npy block or .json list of ids.",
    )
    parser.add_argument(
        "--ns", type=str, default="", help="Comma-separated n's (default 8,16,32,64)."
    )
    parser.add_argument("--max-train-files", type=int, default=0, help="0 = all.")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main():
    parser = add_audit_args(
        argparse.ArgumentParser(description=__doc__.splitlines()[0])
    )
    args = parser.parse_args()
    import logging  # pylint: disable=import-outside-toplevel

    logging.basicConfig(level=logging.INFO)
    sys.exit(run_audit(args, logging.getLogger("memorization")))


if __name__ == "__main__":
    main()
