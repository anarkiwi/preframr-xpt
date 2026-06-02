"""Floor check: does content acc > 0 at body=large mini? Decides B-vs-capacity bottleneck."""

from __future__ import annotations

from preframr_experiments.base import (
    Arm,
    ExperimentSpec,
    mini_train_args,
)

_BASE_MACROS = (
    "freq_trajectory_pass",
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c4",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
)


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
        Arm(label="baseline_large", macro_flags=_BASE_MACROS, baseline=True),
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
)
