"""Floor check: does content acc > 0 at body=large mini? Decides B-vs-capacity bottleneck."""

from __future__ import annotations

from preframr_experiments.base import (
    Arm,
    ExperimentSpec,
    mini_train_args,
)

_BASE_TRANSFORMS = [
    {"name": "slope"},
    {"name": "preset"},
    {"name": "hard_restart"},
    {"name": "legato_per_cluster", "params": {"clusters": [2, 4]}},
    {"name": "voice_block_order"},
    {"name": "ctrl_bigram"},
    {"name": "loop"},
]


spec = ExperimentSpec(
    name="content_floor_check",
    doc=(
        "Single-arm body=large baseline at mini scale (196 SIDs, 160 "
        "epochs / early-stop patience 6). Audits per-tier accuracy on "
        "the best ckpt: if content acc > 0 cleanly, body=small was the "
        "floor and Approach C is justified. If content acc still 0, the "
        "bottleneck is data scale or fundamental capacity and we need "
        "melody-transfer augmentation or different intervention."
    ),
    tier="mini",
    arms=[
        Arm(label="baseline_large", baseline=True),
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
    train_args=mini_train_args(body="large"),
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
)
