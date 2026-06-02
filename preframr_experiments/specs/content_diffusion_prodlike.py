"""Phase 3 prodlike for content diffusion (Approach A). Target: DiffusionContentHead (D3PM absorbing-state, T=8 cosine schedule) at prodlike capacity. Single seed. Baseline comparator is the existing v10 prodlike baseline ckpt at /scratch/tmp/preframr_experiments/results/per_tier_heads_prodlike/baseline/seed0/.../best-epoch=59-val_loss=5.4574.ckpt (saves ~6 hr by not re-running baseline). Run with --only-arm diffusion_T8."""

from __future__ import annotations

from preframr_experiments.base import (
    Arm,
    ExperimentSpec,
    prodlike_train_args,
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
    name="content_diffusion_prodlike",
    doc=(
        "Phase 3 prodlike for DiffusionContentHead (D3PM absorbing-state, "
        "T=8). Single arm + single seed. Pre-condition: mini Phase 2 PASSED "
        "(val_acc within 1sigma + diversity_ratio > 1.65 + collapse <= "
        "baseline). Pass criteria: (1) eval_a content acc >= 2x baseline "
        "AND >= 0.12 (recalibrated floor from prodlike baseline 0.0618, "
        "see refuted/per_tier_heads_mos_prodlike.md 'What this changes' #3); "
        "(2) >= 5 of 8 eval_b_* families show non-zero content acc lift; "
        "(3) loop_collapse_rate at T=0.5 <= baseline (v10 baseline 8%); "
        "(4) diversity_ratio at T=0.5 > 1.2 on real-vs-random; (5) no "
        "structural regression > 1 sigma."
    ),
    tier="prodlike",
    arms=[
        Arm(
            label="diffusion_T8",
            macro_flags=_BASE_MACROS,
            extra_cargs=(
                "--per-tier-heads --content-diffusion --content-diffusion-t 8"
            ),
        ),
        Arm(label="baseline", macro_flags=_BASE_MACROS, baseline=True),
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
)
