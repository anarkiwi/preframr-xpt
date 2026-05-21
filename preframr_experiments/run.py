#!/usr/bin/env python3
"""Experiment runner CLI."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

from preframr_experiments.base import (
    ExperimentSpec,
    preflight_check,
    resolve_data_layout,
    run_arm,
)
from preframr_experiments.metrics import compute_metrics, validate_metric_names
from preframr_experiments.report import render


def load_spec(name: str) -> ExperimentSpec:
    """Resolve the named spec module and return its ``spec`` global."""
    mod_name = f"preframr_experiments.specs.{name}"
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as exc:
        raise SystemExit(f"no such experiment: {name} (import error: {exc})") from exc
    spec = getattr(mod, "spec", None)
    if not isinstance(spec, ExperimentSpec):
        raise SystemExit(f"{mod_name} must expose a top-level ``spec: ExperimentSpec``")
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", help="spec module name (e.g. ``memorize``)")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/scratch/tmp/preframr_experiments"),
        help="Workspace root for arm dirs and the report.",
    )
    ap.add_argument(
        "--src-root",
        type=Path,
        default=Path("/scratch/preframr/training-dumps"),
        help=(
            "Where the .dump.parquet cache lives. Each list entry "
            "resolves against this root."
        ),
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override spec.seeds. Useful for fast smoke runs.",
    )
    ap.add_argument(
        "--only-arm",
        type=str,
        default=None,
        help=(
            "If set, run only the arm with this label. Other arms are "
            "skipped; the report still renders against whatever data "
            "is on disk for the un-skipped arms."
        ),
    )
    ap.add_argument(
        "--tb-scope",
        choices=("spec", "root"),
        default="spec",
        help=(
            "Tensorboard mount scope for the preflight tb restart. "
            "'spec' (default) mounts only this experiment's results "
            "subdir; 'root' mounts the full experiments root (legacy "
            "behaviour, useful for overnight batches that want all "
            "specs in one TB)."
        ),
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logger = logging.getLogger("experiments.run")

    spec = load_spec(args.experiment)
    if args.seeds is not None:
        spec.seeds = args.seeds
    validate_metric_names(spec)

    data_layout = resolve_data_layout(spec)
    results_dir = args.root / "results" / spec.name
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        preflight_check(spec, results_dir, logger, tb_scope=args.tb_scope)
    except RuntimeError as exc:
        logger.error("preflight failed: %s", exc)
        return 1

    results: dict[str, list[dict]] = {arm.label: [] for arm in spec.arms}

    for arm in spec.arms:
        if args.only_arm and arm.label != args.only_arm:
            logger.info("skipping arm %s (--only-arm=%s)", arm.label, args.only_arm)
            continue
        for seed in range(spec.seeds):
            work_dir = results_dir / arm.label / f"seed{seed}"
            try:
                artefacts = run_arm(
                    spec=spec,
                    arm=arm,
                    seed=seed,
                    work_dir=work_dir,
                    data_layout=data_layout,
                    src_root=args.src_root,
                    logger=logger,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("arm=%s seed=%d failed: %s", arm.label, seed, exc)
                continue

            metrics = compute_metrics(spec, artefacts)
            results[arm.label].append(metrics)

            if spec.predict_gate is not None:
                passed, msg = spec.predict_gate(artefacts)
                if not passed:
                    logger.error(
                        "arm=%s seed=%d predict-gate failed: %s",
                        arm.label,
                        seed,
                        msg,
                    )
                    return 1

    md_path = render(spec, results, results_dir)
    logger.info("report: %s", md_path)
    print(md_path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
