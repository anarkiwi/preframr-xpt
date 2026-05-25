"""Phase 1 mini A/B for voice-permutation augmentation. Target: voice_permutation_K5 (each train dump emits 5 non-identity voice permutations alongside, parser re-encodes macros on each variant). Baseline: no augmentation. 3 seeds, mini body=large, 60 epochs. Pass: val_acc on eval_a >= baseline + 0.005 AND no structural-tier regression > 1sigma. Sibling to melody_transfer_augmentation cross-song splice (see integration_tests/design/melody_transfer_augmentation_design.md 'Voice permutation variant'). Requires the pre_run_hook to bake permuted dumps into work_dir before the parse stage; PREFRAMR_DATASET_CACHE_DISABLE=1 because the augmented dumps are spec-dependent (different than the baseline's cache key would suggest)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from preframr_experiments.base import (
    Arm,
    ExperimentSpec,
    mini_train_args,
)

_BASE_TRANSFORMS = [
    {"name": "freq_trajectory"},
    {"name": "preset"},
    {"name": "hard_restart"},
    {"name": "legato_per_cluster", "params": {"clusters": [2, 4]}},
    {"name": "voice_block_order"},
    {"name": "ctrl_bigram"},
    {"name": "loop"},
]

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)

_PREFRAMR_SRC_DIR = os.environ.get(
    "PREFRAMR_SRC_DIR", "/scratch/anarkiwi/preframr/preframr"
)
_AUG_SCRIPT = Path(_PREFRAMR_SRC_DIR).parent / (
    "integration_tests/profile/augment_voice_permutation.py"
)


def _augment_voice_permutation(arm: Arm, work_dir: Path) -> None:
    """For the voice_permutation_K5 arm, generate 5 non-identity voice permutations per train dump under work_dir/train/. Skipped for the baseline arm."""
    if arm.label != "voice_permutation_K5":
        return
    train_root = Path(work_dir) / "train"
    if not train_root.exists():
        raise FileNotFoundError(
            f"train dumps not staged at {train_root}; "
            "voice permutation must run after stage_dumps"
        )
    subprocess.run(
        [
            "python3",
            str(_AUG_SCRIPT),
            "--input-glob",
            str(train_root / "*" / "*.dump.parquet"),
            "--permutations",
            "all",
            "--in-place",
        ],
        check=True,
    )


spec = ExperimentSpec(
    name="voice_permutation_mini_body_large",
    doc=(
        "Phase 1 mini A/B for within-song voice permutation. Target arm "
        "augments 196 train SIDs with all 5 non-identity voice permutations "
        "(196 × 6 = 1176 effective). Baseline arm runs unaugmented. 3 seeds. "
        "Pass: val_acc on eval_a >= baseline + 0.005 AND no structural-tier "
        "regression > 1sigma. Requires PREFRAMR_DATASET_CACHE_DISABLE=1 "
        "(pre_run_hook must bake augmented dumps every seed; dataset cache "
        "would skip the hook)."
    ),
    tier="mini",
    arms=[
        Arm(
            label="voice_permutation_K5",
            extra_cargs="",
        ),
        Arm(
            label="baseline",
            extra_cargs="",
            baseline=True,
        ),
    ],
    metrics=[
        "alphabet_size",
        "val_loss_best",
        "val_acc_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=3,
    seq_len=4096,
    tkvocab=32768,
    max_perm=1,
    train_args=_TRAIN_ARGS,
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
    pre_run_hook=_augment_voice_permutation,
)
