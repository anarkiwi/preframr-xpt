#!/usr/bin/env python3
"""Experiment runner CLI."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
import sys
from pathlib import Path

from preframr_experiments.base import (
    _PREFRAMR_BIND_SRC_ENV,
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


def _override_train_arg(train_args: str, flag: str, value) -> str:
    """Replace ``<flag> <n>`` (or ``<flag>=<n>``) in a train-arg string; append if absent. No-op when value is None."""
    if value is None:
        return train_args
    new, n = re.subn(
        rf"({re.escape(flag)})(=|\s+)\S+", rf"\g<1>\g<2>{value}", train_args
    )
    return new if n else f"{train_args} {flag} {value}"


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
        "--tkvocab",
        type=int,
        default=None,
        help=(
            "Override spec.tkvocab (e.g. vocab-trim re-runs). Folds into the "
            "dataset-cache key, so a trimmed cap re-tokenizes rather than "
            "serving a stale tokenization."
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override --batch-size (micro-batch) in spec.train_args. Pair with "
            "--accumulate-grad-batches to hold the effective batch constant."
        ),
    )
    ap.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=None,
        help="Override --accumulate-grad-batches in spec.train_args.",
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
    ap.add_argument(
        "--bind-src",
        action="store_true",
        help=(
            "Bind-mount the working-tree preframr/ over the baked image so "
            "containers run current code without a rebake. OFF by default: "
            "runs use the frozen, tested baked image, so a long run can't be "
            "perturbed by edits and the working tree stays free to change. "
            "Use only for short iterate-without-rebake loops."
        ),
    )
    args = ap.parse_args()

    if args.bind_src:
        os.environ[_PREFRAMR_BIND_SRC_ENV] = "1"

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logger = logging.getLogger("experiments.run")
    if args.bind_src:
        logger.warning(
            "--bind-src: bind-mounting working-tree preframr/ over the baked "
            "image; this run reflects uncommitted/un-gated code."
        )

    spec = load_spec(args.experiment)
    if args.seeds is not None:
        spec.seeds = args.seeds
    if args.tkvocab is not None:
        spec.tkvocab = args.tkvocab
    spec.train_args = _override_train_arg(
        spec.train_args, "--batch-size", args.batch_size
    )
    spec.train_args = _override_train_arg(
        spec.train_args, "--accumulate-grad-batches", args.accumulate_grad_batches
    )
    if (
        args.tkvocab is not None
        or args.batch_size is not None
        or (args.accumulate_grad_batches is not None)
    ):
        logger.info(
            "overrides: tkvocab=%s batch_size=%s accumulate_grad_batches=%s",
            args.tkvocab,
            args.batch_size,
            args.accumulate_grad_batches,
        )
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

    for seed in range(spec.seeds):
        for arm in spec.arms:
            if args.only_arm and arm.label != args.only_arm:
                logger.info("skipping arm %s (--only-arm=%s)", arm.label, args.only_arm)
                continue
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
