"""Path A: re-test InfoNCE at body=large mini where content acc is measurable."""

from __future__ import annotations

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


spec = ExperimentSpec(
    name="contrastive_mini_body_large",
    doc=(
        "Re-test InfoNCE at body=large mini after content_floor_check "
        "showed body=small was the floor (content acc literally 0). "
        "Body=large baseline content acc 0.0063; this sweep tests "
        "whether InfoNCE lifts that. Gate off (0.015 baseline c/s ratio "
        "is well under default gate threshold). 60 max-epochs to "
        "saturate content learning per floor_check trajectory."
    ),
    tier="mini",
    arms=[
        Arm(
            label="L0.1_K32",
            extra_cargs="--infonce-content-loss-weight 0.1 --infonce-distractors 32",
        ),
        Arm(
            label="L0.05_K64",
            extra_cargs="--infonce-content-loss-weight 0.05 --infonce-distractors 64",
        ),
        Arm(label="baseline", baseline=True),
    ],
    metrics=[
        "alphabet_size",
        "val_loss_best",
        "val_acc_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seq_len=4096,
    tkvocab=32768,
    max_perm=1,
    train_args=_TRAIN_ARGS,
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
)
