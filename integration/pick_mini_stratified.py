#!/usr/bin/env python3
"""Pick a cluster-stratified mini-tier pin from the existing pinned
canonical / prodlike pools.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

TRAIN_PICKS = [
    ("DRAX", 1, "prodlike/train.list"),
    ("Goto80", 2, "canonical/train.list"),
    ("Whittaker_David", 3, "prodlike/train.list"),
    ("Jammer", 4, "prodlike/train.list"),
    ("Galway_Martin", 6, "prodlike/train.list"),
    ("Hubbard_Rob", 7, "prodlike/train.list"),
]
TRAIN_PER_COMPOSER = 25
EVAL_A_PER_COMPOSER = 5
EVAL_B_PER_SUBSET = 8

EVAL_B_PICKS = [
    ("daglish", 4),
    ("follin", 4),
    ("crisps", 1),
    ("mibri", 2),
    ("marquis", 3),
    ("winterberg", 6),
    ("wilson", 7),
]


def _load_list(path: Path) -> list[str]:
    out = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        out.append(line)
    return out


def _filter_composer(rows: list[str], composer: str) -> list[str]:
    return [r for r in rows if f"/{composer}/" in r]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to <repo-root>/preframr_experiments/data/mini",
    )
    args = ap.parse_args(argv)

    data_dir = args.repo_root / "preframr_experiments" / "data"
    out_dir = args.out_dir or (data_dir / "mini")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "pick_version": "mini_stratified_v1",
        "covered_clusters_train": sorted({c for _, c, _ in TRAIN_PICKS}),
        "covered_clusters_eval_b": sorted({c for _, c in EVAL_B_PICKS}),
        "train_per_composer": TRAIN_PER_COMPOSER,
        "eval_a_per_composer": EVAL_A_PER_COMPOSER,
        "eval_b_per_subset": EVAL_B_PER_SUBSET,
        "train_composers": [],
        "eval_b_subsets": [],
    }

    train_lines: list[str] = []
    eval_a_lines: list[str] = []
    by_composer: dict[str, dict] = defaultdict(dict)
    for composer, cluster, source in TRAIN_PICKS:
        source_path = data_dir / source
        rows = _filter_composer(_load_list(source_path), composer)
        rows = sorted(rows)
        need = TRAIN_PER_COMPOSER + EVAL_A_PER_COMPOSER
        if len(rows) < need:
            print(
                f"ERROR: {composer} has {len(rows)} entries in {source}, "
                f"need {need}",
                file=sys.stderr,
            )
            return 1
        picked = rows[:need]
        train_picked = picked[:TRAIN_PER_COMPOSER]
        eval_a_picked = picked[TRAIN_PER_COMPOSER:need]
        train_lines.extend(train_picked)
        eval_a_lines.extend(eval_a_picked)
        by_composer[composer] = {
            "cluster": cluster,
            "source_list": source,
            "n_train": len(train_picked),
            "n_eval_a": len(eval_a_picked),
        }
        summary["train_composers"].append(
            {
                "name": composer,
                "cluster": cluster,
                "n_train": len(train_picked),
                "n_eval_a": len(eval_a_picked),
            }
        )

    train_lines.sort()
    eval_a_lines.sort()
    (out_dir / "train.list").write_text("\n".join(train_lines) + "\n")
    (out_dir / "eval-A.list").write_text("\n".join(eval_a_lines) + "\n")

    for stem, cluster in EVAL_B_PICKS:
        source_path = data_dir / "prodlike" / f"eval-B-{stem}.list"
        rows = _load_list(source_path)
        picked = sorted(rows)[:EVAL_B_PER_SUBSET]
        if len(picked) < EVAL_B_PER_SUBSET:
            print(
                f"WARN: eval-B-{stem} short ({len(picked)} < " f"{EVAL_B_PER_SUBSET})",
                file=sys.stderr,
            )
        (out_dir / f"eval-B-{stem}.list").write_text("\n".join(picked) + "\n")
        summary["eval_b_subsets"].append(
            {
                "stem": stem,
                "cluster": cluster,
                "n_dumps": len(picked),
                "source_list": f"prodlike/eval-B-{stem}.list",
            }
        )

    (out_dir / "picker_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"wrote {len(train_lines)} train / {len(eval_a_lines)} eval-A / "
        f"{sum(s['n_dumps'] for s in summary['eval_b_subsets'])} eval-B "
        f"across {len(summary['eval_b_subsets'])} subsets into {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
