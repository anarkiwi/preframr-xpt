"""Phase 3 prodlike retest of per-tier heads + MoS K=4 with --per-tier-mos-entropy-lambda 0.01. Trigger: the original per_tier_heads_prodlike (v10) refuted on criteria 3+4 (router saturation); the entropy retest at mini (per_tier_heads_entropy_retest.py) PASSED all three criteria (val_acc 0.1483 +/- 0.0016 vs lambda=0 0.1494 +/- 0.0028, collapse_rate 0/12, diversity_ratio 1.535 vs lambda=0 1.220). Companion mask_structural_loss probe at mini REFUTED (diversity_ratio 0.863 < 1.0 — structural supervision was load-bearing for prompt-conditioning), confirming the router-architecture framing over the gradient-dominance framing. Target arm only (--only-arm at runtime); compare against the existing v10 baseline ckpt for the gate."""

from __future__ import annotations

from preframr_experiments.base import (
    Arm,
    ExperimentSpec,
    prodlike_train_args,
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


spec = ExperimentSpec(
    name="per_tier_heads_entropy_prodlike",
    doc=(
        "Phase 3 prodlike retest of per-tier heads + MoS K=4 with "
        "--per-tier-mos-entropy-lambda 0.01. v10 (lambda=0) refuted on "
        "criteria 3+4. Mini retest at lambda=0.01 PASSED all criteria; "
        "this experiment tests whether the mini lift transfers to "
        "prodlike capacity. Target arm only; baseline comparator is "
        "the existing prodlike baseline ckpt at "
        "/scratch/tmp/preframr_experiments/results/per_tier_heads_prodlike"
        "/baseline/seed0/.../best-epoch=59-val_loss=5.4574.ckpt. Pass: "
        "(1) eval_a content acc >= 2x baseline AND >= prodlike-baseline-"
        "calibrated absolute floor (0.0618 baseline -> 0.12 reasonable "
        "target; not the 0.14 mini-tier floor); (2) >= 5 of 8 eval_b_* "
        "families show non-zero content acc lift; (3) loop_collapse_rate "
        "at T=0.5 <= baseline; (4) diversity_ratio at T=0.5 > 1.2; (5) "
        "no structural regression > 1 sigma."
    ),
    tier="prodlike",
    arms=[
        Arm(
            label="per_tier_heads_mos4_entropy",
            extra_cargs=(
                "--per-tier-heads --per-tier-content-mos-k 4 "
                "--per-tier-mos-entropy-lambda 0.01"
            ),
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
