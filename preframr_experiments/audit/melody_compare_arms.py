"""Compare model output musicality across arms of an experiment.

Walks a directory of prediction dumps shaped:
  <root>/<arm>/seed<N>/val_<I>.pred.parquet

Scores each dump via melody_features + the prodlike baseline, then emits
per-arm aggregates and a head-to-head comparison table. Saves
``compare_arms.json`` so downstream analysis can pick up the raw rows.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from preframr_experiments.audit.melody_features import FEATURES
from preframr_experiments.audit.melody_score_generation import score_one


def discover(root: Path) -> dict[str, list[Path]]:
    arms: dict[str, list[Path]] = defaultdict(list)
    for dump in sorted(root.glob("*/seed*/val_*.pred.parquet")):
        arm = dump.parts[-3]
        arms[arm].append(dump)
    return dict(arms)


def aggregate_per_arm(arm_reports: list[dict]) -> dict:
    feats: dict[str, list[float]] = defaultdict(list)
    headlines: list[float] = []
    verdicts: dict[str, int] = defaultdict(int)
    for r in arm_reports:
        headlines.append(r["headline"])
        verdicts[r["verdict"]] += 1
        for row in r["rows"]:
            if row["value"] is not None:
                feats[row["feature"]].append(float(row["value"]))
    out = {
        "n": len(arm_reports),
        "headline_mean": float(statistics.mean(headlines)) if headlines else 0.0,
        "headline_std": (
            float(statistics.stdev(headlines)) if len(headlines) > 1 else 0.0
        ),
        "verdicts": dict(verdicts),
        "features": {
            f: {
                "mean": float(statistics.mean(vs)),
                "std": float(statistics.stdev(vs)) if len(vs) > 1 else 0.0,
                "n": len(vs),
            }
            for f, vs in feats.items()
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "root", type=Path, help="audition root holding <arm>/seedN/val_I.pred.parquet"
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path("/scratch/tmp/preframr_substrate/baseline"),
    )
    ap.add_argument("--out", type=Path, default=None)
    cli = ap.parse_args()
    scaler = json.loads((cli.baseline / "baseline_scaler.json").read_text())
    arms = discover(cli.root)
    if not arms:
        print(f"no <arm>/seed*/val_*.pred.parquet under {cli.root}")
        return 1
    print(f"arms found: {list(arms.keys())}")
    per_arm_reports: dict[str, list[dict]] = {}
    for arm, dumps in arms.items():
        rs = []
        for d in dumps:
            try:
                rs.append(score_one(str(d), scaler, {}))
            except Exception as e:
                print(f"  SKIP {d}: {e}", file=sys.stderr)
        per_arm_reports[arm] = rs
    summary = {arm: aggregate_per_arm(rs) for arm, rs in per_arm_reports.items()}
    print("\n=== per-arm summary ===")
    fmt = "{:24s} n={:2d}  headline={:5.2f}±{:4.2f}  verdicts={}"
    for arm, s in summary.items():
        print(
            fmt.format(
                arm, s["n"], s["headline_mean"], s["headline_std"], s["verdicts"]
            )
        )
    print("\n=== feature means per arm ===")
    feature_order = list(FEATURES)
    header = "{:30s}".format("feature") + "".join(f"{a[:14]:>16s}" for a in summary)
    print(header)
    for f in feature_order:
        row = "{:30s}".format(f)
        for arm in summary:
            ent = summary[arm]["features"].get(f)
            cell = "n/a" if ent is None else f"{ent['mean']:.3f}±{ent['std']:.3f}"
            row += f"{cell:>16s}"
        print(row)
    if cli.out:
        cli.out.write_text(
            json.dumps(
                {
                    "per_arm_reports": {
                        k: [{**r, "rows": r["rows"]} for r in v]
                        for k, v in per_arm_reports.items()
                    },
                    "summary": summary,
                },
                indent=2,
                default=float,
            )
        )
        print(f"\nwrote {cli.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
