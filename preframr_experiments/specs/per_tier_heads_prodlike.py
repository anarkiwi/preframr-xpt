"""Phase 3 prodlike A/B for per-tier heads + MoS K=4 (Approach C); 1 seed, canonical body, 60 epochs. Spec details in `integration_tests/design/per_tier_heads_design.md`. Revisited evidence in `integration_tests/data/refuted/per_tier_heads_mos_revisited.md` (Phase 2 PASS at sampling T>=0.5)."""

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
    name="per_tier_heads_prodlike",
    doc=(
        "Phase 3 prodlike A/B for per-tier heads + MoS K=4. Phase 2 "
        "mini Pass: val_acc 0.1494 +/- 0.0028 vs baseline 0.0832 +/- "
        "0.0074 (+1.80x, 3-seed). Phase 2 generation audits initially "
        "refuted at T=0 (greedy router saturates structural -> loop "
        "collapse), then re-opened at T in {0.5, 0.7} (both criteria "
        "PASS). Phase 3 tests whether the mini lift survives prodlike "
        "scale (4854 SIDs, 125M params, 60 epochs, single seed). "
        "Target arm first per convention; baseline last. Pass: (1) "
        "eval_a content acc >= 2x baseline AND >= 0.14; (2) >= 5 of 8 "
        "eval_b_* families show non-zero content acc lift; (3) "
        "loop_collapse_rate at T=0.5 <= baseline; (4) diversity_ratio "
        "at T=0.5 > 1.2 on real-vs-random; (5) no structural "
        "regression > 1 sigma."
    ),
    tier="prodlike",
    arms=[
        Arm(
            label="per_tier_heads_mos4",
            extra_cargs="--per-tier-heads --per-tier-content-mos-k 4",
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
