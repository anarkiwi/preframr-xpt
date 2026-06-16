"""Augmentation dosage A/B (Tier-3, attacks free-running copy-dominance M4).

Atoms-only baseline vs the same train set plus write-domain augmented dumps from preframr-aug
(reduce = melody-only-prefix; instrument = same-role timbre transplant). Each augmented arm copies its
extra ``.dump.parquet`` files into the staged train dir via :func:`_stage_aug` (run after the canonical
corpus is staged). The augmented dumps are NOT in the cache key (it hashes the data_layout lists, not the
hook output), so this run MUST set ``PREFRAMR_DATASET_CACHE_DISABLE=1`` -- otherwise the augmented arms
would silently reuse the baseline tokenisation. Matches the v2 atoms-only baseline (tkvocab 0, seq_len
8192, ep100); decide on the event-native Tier-0 audits (copy_novel novel-fraction, free_running_gap) run
on each arm's ckpt, plus eval-B held-out-composer content tier.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

from preframr_experiments.base import Arm, ExperimentSpec
from preframr_experiments.specs.generalize import _TRAIN_ARGS, _generalize_gate

_BASELINE_TRAIN = 789
_DOSE_25 = _BASELINE_TRAIN // 4

_AUG = {
    "reduce_full": ("/scratch/tmp/aug_corpora/reduce_full", None),
    "reduce_25": ("/scratch/tmp/aug_corpora/reduce_full", _DOSE_25),
    "instrument_full": ("/scratch/tmp/aug_corpora/instrument_full", None),
    "instrument_25": ("/scratch/tmp/aug_corpora/instrument_full", _DOSE_25),
}


def _stage_aug(arm, work_dir):
    """Copy this arm's augmented dumps into the staged train dir (a new composer bucket the train glob
    picks up). Deterministic subset by sorted name for the dose arms; baseline gets nothing.
    """
    entry = _AUG.get(arm.label)
    if entry is None:
        return
    src, limit = entry
    dumps = sorted(glob.glob(os.path.join(src, "*.dump.parquet")))
    if limit is not None:
        dumps = dumps[:limit]
    dst = Path(work_dir) / "train" / f"aug_{arm.label}"
    dst.mkdir(parents=True, exist_ok=True)
    for dump in dumps:
        shutil.copy(dump, dst / os.path.basename(dump))


spec = ExperimentSpec(
    name="generalize_aug_ab",
    doc=(
        "Augmentation dosage A/B: atoms-only baseline vs +reduce / +instrument write-domain "
        "augmentation from preframr-aug. Tier-3 of the free-running pathology ladder (breaks the "
        "copy reward). Decide on event-native copy_novel + free_running_gap audits per arm."
    ),
    tier="canonical",
    arms=[
        Arm(label="baseline", baseline=True),
        Arm(label="reduce_full"),
        Arm(label="instrument_full"),
        Arm(label="reduce_25"),
        Arm(label="instrument_25"),
    ],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=1,
    seq_len=8192,
    tkvocab=0,
    train_args=_TRAIN_ARGS,
    predict_gate=_generalize_gate,
    pre_run_hook=_stage_aug,
)
