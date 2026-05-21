"""Decisive InfoNCE viability test at prodlike scale (4854 SIDs / canonical body)."""

from __future__ import annotations

from preframr_experiments.base import (
    Arm,
    ExperimentSpec,
    prodlike_train_args,
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
    name="contrastive_prodlike",
    doc=(
        "Decisive InfoNCE viability A/B at prodlike scale. Body=large "
        "mini sweep showed L0.05_K64 lifts content acc 5x baseline "
        "(0.0008 -> 0.0043) -- direction real but magnitude tiny. "
        "This run tests whether the lift survives prodlike scale "
        "(4854 SIDs, 125M params, 60 epochs). Gate off (default "
        "thresholds misfire at prodlike subset granularity; first "
        "prodlike with tokenizer fix from 2026-05-21). Pass: content "
        "acc on eval_a >= 14% AND >= 4 of 8 eval-B families show "
        "non-zero lift vs baseline AND no greedy-decode collapse."
    ),
    tier="prodlike",
    arms=[
        Arm(
            label="contrastive",
            extra_cargs="--infonce-content-loss-weight 0.05 --infonce-distractors 64",
        ),
        Arm(label="baseline", baseline=True),
    ],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "val_loss_macro_best",
        "val_acc_macro_at_best_loss",
        "val_loss_eval_a_best",
        "val_acc_eval_a_at_best_loss",
        "val_loss_eval_b_daglish_best",
        "val_acc_eval_b_daglish_at_best_loss",
        "val_loss_eval_b_follin_best",
        "val_acc_eval_b_follin_at_best_loss",
        "val_loss_eval_b_crisps_best",
        "val_acc_eval_b_crisps_at_best_loss",
        "val_loss_eval_b_mibri_best",
        "val_acc_eval_b_mibri_at_best_loss",
        "val_loss_eval_b_marquis_best",
        "val_acc_eval_b_marquis_at_best_loss",
        "val_loss_eval_b_dobek_best",
        "val_acc_eval_b_dobek_at_best_loss",
        "val_loss_eval_b_winterberg_best",
        "val_acc_eval_b_winterberg_at_best_loss",
        "val_loss_eval_b_wilson_best",
        "val_acc_eval_b_wilson_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=1,
    seq_len=8192,
    tkvocab=32768,
    max_perm=1,
    train_args=prodlike_train_args(),
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
)
