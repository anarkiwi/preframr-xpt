"""Build the prodlike training-set baseline distribution of melody features.

Iterates a paths list (relative to HVSC root), runs ``melody_features.analyze``
on each, writes:
  - per-tune feature rows -> baseline_per_tune.csv
  - summary mean / std / percentiles per feature -> baseline_summary.csv
  - per-feature scaler params for z-scoring -> baseline_scaler.json

Multiprocesses by default. ~10 min on the prodlike train list (4437 tunes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from preframr_experiments.audit.melody_features import FEATURES, analyze


def _worker(path: str) -> dict[str, float | None]:
    try:
        return analyze(path)
    except Exception as e:
        return {"_path": path, "_error": str(e)}


def read_dump_list(list_path: Path, hvsc_root: Path) -> list[str]:
    out: list[str] = []
    for raw in list_path.read_text().splitlines():
        rel = raw.split("#", 1)[0].strip()
        if not rel:
            continue
        p = hvsc_root / rel
        if p.exists():
            out.append(str(p))
    return out


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    feats = [c for c in FEATURES if c in df.columns]
    summary = {}
    for c in feats:
        s = df[c].dropna().astype(float)
        if len(s) == 0:
            continue
        summary[c] = {
            "mean": float(s.mean()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "median": float(s.median()),
            "p10": float(s.quantile(0.10)),
            "p90": float(s.quantile(0.90)),
            "n": int(len(s)),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--list",
        type=Path,
        default=Path(
            "/scratch/anarkiwi/preframr-xpt/preframr_experiments/data/prodlike/train.list"
        ),
    )
    ap.add_argument("--hvsc-root", type=Path, default=Path("/scratch/preframr/hvsc"))
    ap.add_argument(
        "--out-dir", type=Path, default=Path("/scratch/tmp/preframr_substrate/baseline")
    )
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    cli = ap.parse_args()
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    paths = read_dump_list(cli.list, cli.hvsc_root)
    if cli.limit > 0:
        paths = paths[: cli.limit]
    print(f"baseline corpus: {len(paths)} tunes, workers={cli.workers}")
    rows: list[dict] = []
    err = 0
    with ProcessPoolExecutor(max_workers=cli.workers) as ex:
        futures = {ex.submit(_worker, p): p for p in paths}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            if row.get("_error"):
                err += 1
            else:
                rows.append(row)
            if i % 200 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)} done (errors: {err})")
    df = pd.DataFrame(rows)
    per_tune = cli.out_dir / "baseline_per_tune.csv"
    df.to_csv(per_tune, index=False)
    summary = summarize(rows)
    summary_csv = cli.out_dir / "baseline_summary.csv"
    pd.DataFrame(summary).T.to_csv(summary_csv)
    scaler = {
        c: {"mean": summary[c]["mean"], "std": max(summary[c]["std"], 1e-6)}
        for c in summary
    }
    (cli.out_dir / "baseline_scaler.json").write_text(json.dumps(scaler, indent=2))
    print(f"wrote {per_tune}\nwrote {summary_csv}")
    print("\n=== feature summary (prodlike train) ===")
    for c, s in summary.items():
        print(
            f"  {c:30s} mean={s['mean']:8.3f} std={s['std']:7.3f} "
            f"median={s['median']:8.3f} [p10..p90: {s['p10']:.2f}..{s['p90']:.2f}] "
            f"n={s['n']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
