#!/usr/bin/env python3
"""Re-render an experiment's report.md against an existing results dir."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from preframr_experiments.base import ArmArtefacts
from preframr_experiments.metrics import compute_metrics, validate_metric_names
from preframr_experiments.report import render
from preframr_experiments.run import load_spec


def _reconstruct_artefacts(_spec, arm, seed: int, work_dir: Path) -> ArmArtefacts:
    log_dir = work_dir / "logs"
    return ArmArtefacts(
        arm=arm,
        seed=seed,
        work_dir=work_dir,
        tokens_csv=work_dir / "tokens.csv",
        df_map_csv=work_dir / "df-map.csv",
        tb_logs=work_dir / "tb_logs",
        train_log=log_dir / "train.log",
        parse_log=log_dir / "parse.log",
        tokenize_log=log_dir / "tokenize.log",
        metrics_json=work_dir / "metrics.json",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment", help="spec module name")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/scratch/tmp/preframr_experiments"),
        help="experiment workspace root (matches run.py --root)",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logger = logging.getLogger("experiments.rerender_report")

    spec = load_spec(args.experiment)
    validate_metric_names(spec)
    results_dir = args.root / "results" / spec.name
    if not results_dir.exists():
        logger.error("no results dir at %s", results_dir)
        return 1

    results: dict[str, list[dict]] = {arm.label: [] for arm in spec.arms}
    for arm in spec.arms:
        for seed in range(spec.seeds):
            work_dir = results_dir / arm.label / f"seed{seed}"
            if not (work_dir / "tb_logs").exists():
                logger.info("skip %s/seed%d (no tb_logs)", arm.label, seed)
                continue
            art = _reconstruct_artefacts(spec, arm, seed, work_dir)
            metrics = compute_metrics(spec, art)
            results[arm.label].append(metrics)
            logger.info("%s/seed%d: %d metrics", arm.label, seed, len(metrics))

    md_path = render(spec, results, results_dir)
    logger.info("report: %s", md_path)
    print(md_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
