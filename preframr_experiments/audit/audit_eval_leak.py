#!/usr/bin/env python3
"""Eval-set leak audit across train / eval-* .list files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

FINGERPRINT_WRITES = 2000
NGRAM_WINDOW = 128
NGRAM_HASH_BYTES = 8


def _read_list(path: Path) -> list[str]:
    """Parse a .list file: one HVSC-relative path per line; ``# ...``
    comments stripped; blank lines ignored. Same convention as the
    experiment harness's ``read_list``.
    """
    out = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out


def _read_writes(parquet_path: Path) -> np.ndarray:
    """Return shape ``(N, 4)`` int64 array of (clock, irq, reg, val)
    for the dump. Missing / unreadable parquets return shape (0, 4).
    """
    try:
        pf = pq.ParquetFile(parquet_path)
        if pf.num_row_groups == 0:
            return np.zeros((0, 4), dtype=np.int64)
        table = pf.read(columns=["clock", "irq", "reg", "val"])
    except (OSError, KeyError) as e:
        logging.warning("%s: skip (read failed: %s)", parquet_path.name, e)
        return np.zeros((0, 4), dtype=np.int64)
    cols = [
        table.column(c).to_numpy().astype(np.int64)
        for c in ("clock", "irq", "reg", "val")
    ]
    return np.stack(cols, axis=1)


def _fingerprint(writes: np.ndarray, n: int = FINGERPRINT_WRITES) -> str:
    """SHA-256 of the first ``n`` writes -- same construction the
    picker uses for engine-fingerprint dedup.
    """
    n = min(n, writes.shape[0])
    if n == 0:
        return ""
    h = hashlib.sha256()
    for col in range(4):
        h.update(writes[:n, col].tobytes())
    return h.hexdigest()


def _ngram_hashes(writes: np.ndarray, window: int = NGRAM_WINDOW) -> set[bytes]:
    """Return the set of truncated Blake2b digests over each
    contiguous ``window``-row slice of ``writes``. Compares against
    structural content, not start position, so a remix that shifts
    the same body still hashes the same way once its prefix is past.
    """
    n = writes.shape[0]
    if n < window:
        return set()
    out = set()
    flat = writes.tobytes()
    row_bytes = 4 * 8
    win_bytes = window * row_bytes
    for start in range(0, (n - window + 1) * row_bytes, row_bytes):
        h = hashlib.blake2b(
            flat[start : start + win_bytes], digest_size=NGRAM_HASH_BYTES
        )
        out.add(h.digest())
    return out


def _resolve_paths(hvsc_root: Path, rels: Iterable[str]) -> list[tuple[str, Path]]:
    """Map .list entries to (rel, abs) pairs. Missing files are
    skipped with a warning so a partially-cached corpus doesn't
    abort the audit; the JSON summary reports the skip count.
    """
    out = []
    skipped = 0
    for rel in rels:
        p = hvsc_root / rel
        if not p.exists():
            skipped += 1
            continue
        out.append((rel, p))
    if skipped:
        logging.warning("skipped %u missing dumps under %s", skipped, hvsc_root)
    return out


def _audit_fingerprints(
    sets: dict[str, list[tuple[str, Path]]],
) -> tuple[list[dict], dict[str, int]]:
    """First-pass fingerprint audit: hash each dump's first 2000
    writes, group by hash, flag any hash whose membership spans
    >= 2 sets.
    """
    fp_to_members = defaultdict(lambda: defaultdict(list))
    counts = {}
    for set_name, entries in sets.items():
        counts[set_name] = len(entries)
        for rel, abs_path in entries:
            writes = _read_writes(abs_path)
            fp = _fingerprint(writes)
            if fp:
                fp_to_members[fp][set_name].append(rel)
    violations = []
    for fp, members_by_set in fp_to_members.items():
        if len(members_by_set) > 1:
            violations.append(
                {
                    "fingerprint": fp,
                    "members": {k: list(v) for k, v in members_by_set.items()},
                }
            )
    return violations, counts


def _audit_ngram_overlap(
    train_entries: list[tuple[str, Path]],
    eval_sets: dict[str, list[tuple[str, Path]]],
    window: int,
    warn_frac: float,
    abort_frac: float,
) -> tuple[list[dict], list[dict]]:
    """Build the train-side window-hash set, then for each eval SID
    compute (overlapping_windows / total_windows).
    """
    logging.info(
        "building train n-gram set: %u SIDs, window=%u", len(train_entries), window
    )
    train_hashes: set[bytes] = set()
    for i, (rel, abs_path) in enumerate(train_entries):
        if i % 50 == 0:
            logging.info(
                "  train %u/%u (%u windows so far)",
                i,
                len(train_entries),
                len(train_hashes),
            )
        writes = _read_writes(abs_path)
        train_hashes.update(_ngram_hashes(writes, window=window))
    logging.info("train set: %u distinct windows", len(train_hashes))

    warnings: list[dict] = []
    aborts: list[dict] = []
    for set_name, entries in eval_sets.items():
        logging.info("auditing %s: %u SIDs", set_name, len(entries))
        for rel, abs_path in entries:
            writes = _read_writes(abs_path)
            hashes = _ngram_hashes(writes, window=window)
            if not hashes:
                continue
            overlap = len(hashes & train_hashes)
            frac = overlap / len(hashes)
            record = {
                "set": set_name,
                "rel": rel,
                "n_windows": len(hashes),
                "n_overlap": overlap,
                "overlap_frac": frac,
            }
            if frac >= abort_frac:
                aborts.append(record)
            elif frac >= warn_frac:
                warnings.append(record)
    return warnings, aborts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hvsc-root", type=Path, required=True)
    ap.add_argument("--train-list", type=Path, required=True)
    ap.add_argument(
        "--eval-list",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each eval split (e.g. eval-A.list, eval-B-*.list).",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=NGRAM_WINDOW)
    ap.add_argument(
        "--warn-threshold",
        type=float,
        default=0.01,
        help="Fraction of eval windows present in train -- log a warning above this.",
    )
    ap.add_argument(
        "--abort-threshold",
        type=float,
        default=0.05,
        help="Fraction above which the audit exits non-zero.",
    )
    ap.add_argument(
        "--skip-ngram",
        action="store_true",
        help="Run only the fingerprint pass (much faster; the n-gram pass is the slow one).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("audit_eval_leak")

    train_rels = _read_list(args.train_list)
    train_entries = _resolve_paths(args.hvsc_root, train_rels)
    logger.info("train: %u rels, %u resolved", len(train_rels), len(train_entries))

    eval_sets: dict[str, list[tuple[str, Path]]] = {}
    for p in args.eval_list:
        name = p.stem
        rels = _read_list(p)
        entries = _resolve_paths(args.hvsc_root, rels)
        eval_sets[name] = entries
        logger.info("%s: %u rels, %u resolved", name, len(rels), len(entries))

    fp_sets = {"train": train_entries, **eval_sets}
    fp_violations, fp_counts = _audit_fingerprints(fp_sets)
    if fp_violations:
        logger.warning(
            "fingerprint violations: %u cross-set duplicates", len(fp_violations)
        )
        for v in fp_violations[:5]:
            logger.warning("  fp=%s... %s", v["fingerprint"][:8], v["members"])

    ngram_warnings: list[dict] = []
    ngram_aborts: list[dict] = []
    if not args.skip_ngram:
        ngram_warnings, ngram_aborts = _audit_ngram_overlap(
            train_entries,
            eval_sets,
            window=args.window,
            warn_frac=args.warn_threshold,
            abort_frac=args.abort_threshold,
        )
    if ngram_warnings:
        logger.warning(
            "n-gram overlap warnings (>= %.1f%%): %u eval SIDs",
            args.warn_threshold * 100,
            len(ngram_warnings),
        )
    if ngram_aborts:
        logger.error(
            "n-gram overlap aborts (>= %.1f%%): %u eval SIDs",
            args.abort_threshold * 100,
            len(ngram_aborts),
        )
        for r in ngram_aborts[:5]:
            logger.error(
                "  %s/%s overlap=%.1f%% (n=%u)",
                r["set"],
                r["rel"],
                r["overlap_frac"] * 100,
                r["n_windows"],
            )

    summary = {
        "hvsc_root": str(args.hvsc_root),
        "set_sizes": fp_counts,
        "fingerprint_violations": fp_violations,
        "ngram": {
            "window": args.window,
            "warn_threshold": args.warn_threshold,
            "abort_threshold": args.abort_threshold,
            "warnings": ngram_warnings,
            "aborts": ngram_aborts,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", args.out)

    if fp_violations or ngram_aborts:
        logger.error("audit FAILED -- see %s", args.out)
        return 1
    logger.info("audit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
