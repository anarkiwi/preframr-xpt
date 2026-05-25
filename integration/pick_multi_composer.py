#!/usr/bin/env python3
"""Pick a multi-composer train + dual-eval corpus from HVSC."""

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PAL_CLOCK_HZ = 985248
DURATION_FLOOR_S = 30.0
EVAL_PER_COMPOSER_DEFAULT = 16
TRAIN_CAP_PER_COMPOSER_DEFAULT = 200
FINGERPRINT_WRITES = 2000

DEFAULT_TRAIN_COMPOSERS = [
    ("Goto80", "MUSICIANS/G/Goto80"),
    ("Hubbard_Rob", "MUSICIANS/H/Hubbard_Rob"),
    ("Galway_Martin", "MUSICIANS/G/Galway_Martin"),
    ("Tel_Jeroen", "MUSICIANS/T/Tel_Jeroen"),
    ("DRAX", "MUSICIANS/D/DRAX"),
]
DEFAULT_EVAL_B_COMPOSERS = [
    ("Daglish_Ben", "MUSICIANS/D/Daglish_Ben"),
    ("Follin_Tim", "MUSICIANS/F/Follin_Tim"),
]


def discover_top_composers(hvsc_root, top_n, exclude_names):
    """Walk HVSC's MUSICIANS/<letter>/<composer>/ tree, count
    ``.dump.parquet`` files per composer, and return the top-N by
    count. ``exclude_names`` (set of composer names) are dropped before
    ranking so Eval-B composers don't accidentally land in the train
    pool. Skips composer dirs with zero pre-cached dumps.
    """
    musicians = Path(hvsc_root) / "MUSICIANS"
    counts = []
    for letter_dir in sorted(musicians.iterdir()):
        if not letter_dir.is_dir():
            continue
        for composer_dir in sorted(letter_dir.iterdir()):
            if not composer_dir.is_dir():
                continue
            name = composer_dir.name
            if name in exclude_names:
                continue
            n_dumps = sum(1 for _ in composer_dir.rglob("*.dump.parquet"))
            if n_dumps == 0:
                continue
            rel = str(composer_dir.relative_to(hvsc_root))
            counts.append((n_dumps, name, rel))
    counts.sort(reverse=True)
    return [(name, rel) for _, name, rel in counts[:top_n]]


def dump_duration_s(parquet_path):
    """Total dump duration in seconds via the ``clock`` column."""
    pf = pq.ParquetFile(parquet_path)
    if pf.num_row_groups == 0:
        return 0.0
    table = pf.read(columns=["clock"])
    arr = table.column("clock").to_numpy()
    if arr.size < 2:
        return 0.0
    return float((arr[-1] - arr[0]) / PAL_CLOCK_HZ)


def dump_fingerprint(parquet_path, n_writes=FINGERPRINT_WRITES):
    """SHA-256 over the first n raw register writes (clock, irq, reg, val)."""
    pf = pq.ParquetFile(parquet_path)
    if pf.num_row_groups == 0:
        return None
    table = pf.read(columns=["clock", "irq", "reg", "val"])
    n = min(n_writes, table.num_rows)
    if n == 0:
        return None
    head = table.slice(0, n)
    h = hashlib.sha256()
    for col in ("clock", "irq", "reg", "val"):
        h.update(head.column(col).to_numpy().tobytes())
    return h.hexdigest()


def list_composer_dumps(hvsc_root, composer_path):
    p = Path(hvsc_root) / composer_path
    if not p.is_dir():
        return []
    return sorted(p.rglob("*.dump.parquet"))


def stratified_eval_pick(records, n):
    """Pick ``n`` records spaced evenly across the duration distribution."""
    if not records:
        return []
    if n >= len(records):
        return list(records)
    sorted_r = sorted(records, key=lambda r: r["duration_s"])
    step = len(sorted_r) / n
    return [sorted_r[int(i * step)] for i in range(n)]


def gather_composer(name, hvsc_root, composer_path, logger):
    paths = list_composer_dumps(hvsc_root, composer_path)
    out = []
    for p in paths:
        try:
            dur = dump_duration_s(p)
        except (OSError, KeyError, IndexError) as e:
            logger.warning("%s: skip (duration read failed: %s)", p.name, e)
            continue
        if dur < DURATION_FLOOR_S:
            continue
        try:
            fp = dump_fingerprint(p)
        except (OSError, KeyError, IndexError) as e:
            logger.warning("%s: skip (fingerprint failed: %s)", p.name, e)
            continue
        if fp is None:
            continue
        rel = str(p.relative_to(hvsc_root))
        out.append(
            {
                "composer": name,
                "rel_path": rel,
                "duration_s": dur,
                "fp": fp,
            }
        )
    logger.info(
        "%s: %u dumps, %u >= %.0fs gate",
        name,
        len(paths),
        len(out),
        DURATION_FLOOR_S,
    )
    return out


def cross_composer_dedup(records_by_composer, logger):
    """Drop fingerprint duplicates that span composer boundaries."""
    fp_to_composers = defaultdict(list)
    for composer, records in records_by_composer.items():
        for r in records:
            fp_to_composers[r["fp"]].append(composer)

    drop_targets = set()
    leak_log = []
    for fp, composers in fp_to_composers.items():
        unique = set(composers)
        if len(unique) <= 1:
            continue
        sizes = {c: len(records_by_composer[c]) for c in unique}
        keeper = min(sizes, key=sizes.get)
        for c in unique:
            if c != keeper:
                drop_targets.add((c, fp))
        leak_log.append((fp, sorted(unique), keeper))

    if leak_log:
        logger.info(
            "cross-composer fingerprint matches: %u distinct hashes",
            len(leak_log),
        )
        for fp, composers, keeper in leak_log[:10]:
            logger.info("  fp=%s... composers=%s keep=%s", fp[:8], composers, keeper)
        if len(leak_log) > 10:
            logger.info("  (+ %u more)", len(leak_log) - 10)

    out = {}
    for composer, records in records_by_composer.items():
        kept = [r for r in records if (composer, r["fp"]) not in drop_targets]
        if len(kept) != len(records):
            logger.info(
                "%s: %u -> %u after cross-composer dedup",
                composer,
                len(records),
                len(kept),
            )
        out[composer] = kept
    return out


def _song_base(rel_path):
    """Strip the trailing ``.<tune>.dump.parquet`` so sibling
    subtunes of one .sid file share a key. Used by ``split_composer``
    to keep all subtunes of a single song on the same side of the
    train / eval split -- without this, ``Foo.1`` could land in
    eval while ``Foo.4`` lands in train, and the n-gram leak
    """
    base, _, tail = rel_path.rpartition(".")
    if tail != "parquet":
        return rel_path
    base, _, tail = base.rpartition(".")
    if tail != "dump":
        return rel_path
    base, _, tail = base.rpartition(".")
    if not tail.isdigit():
        return rel_path
    return base


def split_composer(records, eval_n, train_cap, _rng):
    """Eval-A pick stratified by duration; train cap applies to the
    leftover. Returns (eval_records, train_records).
    """
    if not records:
        return [], []
    groups: dict[str, list] = {}
    for r in records:
        groups.setdefault(_song_base(r["rel_path"]), []).append(r)
    reps = [max(g, key=lambda r: r["duration_s"]) for g in groups.values()]

    eval_reps = stratified_eval_pick(reps, eval_n)
    eval_bases = {_song_base(r["rel_path"]) for r in eval_reps}
    eval_picks = [r for r in records if _song_base(r["rel_path"]) in eval_bases]
    rest = [r for r in records if _song_base(r["rel_path"]) not in eval_bases]

    if len(rest) > train_cap:
        rest_sorted = sorted(rest, key=lambda r: r["duration_s"])
        step = len(rest_sorted) / train_cap
        rest = [rest_sorted[int(i * step)] for i in range(train_cap)]
    return eval_picks, rest


def write_list(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(f"{r['rel_path']}  # {r['composer']}, {r['duration_s']:.1f}s\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hvsc-root",
        default="/scratch/preframr/hvsc",
        help="Root containing MUSICIANS/...",
    )
    ap.add_argument(
        "--out-dir",
        default="/scratch/preframr/multi_composer_lists",
        help="Where to write *.list and summary.json",
    )
    ap.add_argument(
        "--train-top-n",
        type=int,
        default=0,
        help=(
            "If > 0, auto-pick the top-N composers by .dump.parquet count "
            "from HVSC instead of the default 5-composer set. Eval-B "
            "composers are excluded from the ranking."
        ),
    )
    ap.add_argument(
        "--eval-b-composers",
        default="Daglish_Ben,Follin_Tim",
        help="Comma-separated list of composer names to hold out for Eval-B.",
    )
    ap.add_argument(
        "--train-cap",
        type=int,
        default=TRAIN_CAP_PER_COMPOSER_DEFAULT,
        help="Per-composer train cap after duration gate + cross-composer dedup.",
    )
    ap.add_argument(
        "--eval-per-composer",
        type=int,
        default=EVAL_PER_COMPOSER_DEFAULT,
        help="Per-composer Eval-A holdout count (also Eval-B per composer).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("pick_mc")

    eval_b_names = [s.strip() for s in args.eval_b_composers.split(",") if s.strip()]

    eval_b_pairs = []
    for name in eval_b_names:
        rel = f"MUSICIANS/{name[0].upper()}/{name}"
        if not (Path(args.hvsc_root) / rel).is_dir():
            matches = list(Path(args.hvsc_root, "MUSICIANS").rglob(name))
            matches = [m for m in matches if m.is_dir()]
            if matches:
                rel = str(matches[0].relative_to(args.hvsc_root))
            else:
                logger.warning("Eval-B composer %s not found under HVSC root", name)
                continue
        eval_b_pairs.append((name, rel))

    if args.train_top_n > 0:
        train_pairs = discover_top_composers(
            args.hvsc_root, args.train_top_n, exclude_names=set(eval_b_names)
        )
        logger.info(
            "discovered top-%u train composers (excluding Eval-B): %s",
            args.train_top_n,
            ", ".join(name for name, _ in train_pairs[:10])
            + (", ..." if len(train_pairs) > 10 else ""),
        )
    else:
        train_pairs = list(DEFAULT_TRAIN_COMPOSERS)

    rng = np.random.default_rng(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    train_records = {}
    eval_b_records = {}
    for name, path in train_pairs:
        train_records[name] = gather_composer(name, args.hvsc_root, path, logger)
    for name, path in eval_b_pairs:
        eval_b_records[name] = gather_composer(name, args.hvsc_root, path, logger)

    union = {**train_records, **eval_b_records}
    union = cross_composer_dedup(union, logger)
    train_records = {k: union[k] for k, _ in train_pairs}
    eval_b_records = {k: union[k] for k, _ in eval_b_pairs}

    eval_a_picks = []
    train_picks = []
    per_composer_summary = {}
    for name, _ in train_pairs:
        records = train_records[name]
        eval_picks, train_set = split_composer(
            records, args.eval_per_composer, args.train_cap, rng
        )
        eval_a_picks.extend(eval_picks)
        train_picks.extend(train_set)
        per_composer_summary[name] = {
            "post_filter_dumps": len(records),
            "train": len(train_set),
            "eval_A": len(eval_picks),
        }
        logger.info(
            "%s: train=%u eval_A=%u (post-filter pool=%u)",
            name,
            len(train_set),
            len(eval_picks),
            len(records),
        )

    eval_b_summary = {}
    eval_b_files = {}
    for name, _ in eval_b_pairs:
        records = eval_b_records[name]
        picks = stratified_eval_pick(records, args.eval_per_composer)
        eval_b_summary[name] = {
            "post_filter_dumps": len(records),
            "eval_B": len(picks),
        }
        eval_b_files[name] = picks
        logger.info(
            "%s [Eval-B]: picked %u from %u post-filter pool",
            name,
            len(picks),
            len(records),
        )

    train_picks.sort(key=lambda r: (r["composer"], r["rel_path"]))
    eval_a_picks.sort(key=lambda r: (r["composer"], r["rel_path"]))

    write_list(Path(args.out_dir) / "train.list", train_picks)
    write_list(Path(args.out_dir) / "eval-A.list", eval_a_picks)
    for name, picks in eval_b_files.items():
        suffix = name.split("_")[0].lower()
        write_list(Path(args.out_dir) / f"eval-B-{suffix}.list", picks)

    summary = {
        "hvsc_root": args.hvsc_root,
        "duration_floor_s": DURATION_FLOOR_S,
        "fingerprint_writes": FINGERPRINT_WRITES,
        "train_top_n": args.train_top_n,
        "train_cap_per_composer": args.train_cap,
        "eval_per_composer": args.eval_per_composer,
        "train": {
            "total_dumps": len(train_picks),
            "by_composer": per_composer_summary,
        },
        "eval_A": {
            "total_dumps": len(eval_a_picks),
            "per_composer": args.eval_per_composer,
        },
        "eval_B": eval_b_summary,
    }
    summary_path = Path(args.out_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %s", summary_path)
    logger.info(
        "TOTAL train=%u eval_A=%u eval_B=%u",
        len(train_picks),
        len(eval_a_picks),
        sum(len(v) for v in eval_b_files.values()),
    )


if __name__ == "__main__":
    main()
