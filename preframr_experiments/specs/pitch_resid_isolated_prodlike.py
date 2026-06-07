"""Pitch-model ISOLATION prodlike A/B/C (tokens 0.46.3) -- identifies the *marginal* generalization effect of
the universal pitch model, which pitch_resid_prodlike's 2-arm (target vs atomic) design cannot.

The problem with pitch_resid_prodlike: target = full_macros + universal_pitch + table_resid_split, baseline =
atomic (all passes OFF). full_macros does NOT contain the pitch flags, so that contrast bundles the entire
full_macros stack AND the pitch model against bare atomic -- it mostly re-measures the already-known
full_macros>>atomic win (prior canonical eval_a 0.324 vs 0.219, x3 seeds) and CANNOT attribute any delta to
the pitch model, the one open question.

This spec adds the missing isolating arm. Three arms, decisive comparison = arm0 vs arm1:
  - pitch_resid  = full_macros + universal_pitch + table_resid_split  (the pitch model on top of the stack)
  - full_macros  = full_macros only                                   (the isolating reference: pitch marginal
                                                                        = pitch_resid - full_macros)
  - atomic       = all passes OFF (baseline)                          (the floor: re-confirms full_macros win,
                                                                        contextualizes the pitch delta's size)
The decisive pair (pitch_resid, full_macros) runs first under the seed-major runner, so the 1-seed marginal is
available before the atomic floor finishes. 1 seed, canonical body (tkvocab 32768, seq_len 8192, 60 epochs),
on anarkiwi/preframr:0.2.25.

Reads: PRIMARY for the generalization claim = eval_b_* held-out composer families (pitch_resid vs full_macros);
the universal pitch model's claim IS cross-composer transfer. SECONDARY/diagnostic = audit.content_tier_report
(per-tier content_over_structural + per-op op_acc), read with the caveat that universal_pitch changes the
tokenization (new NOTE_UNIV ops; per-op op_acc only compares ops present in both arms).

Launch (after / instead of the 2-arm pitch_resid_prodlike; reuses no cache from it -- each arm's macro set is a
distinct dataset-cache key):
  preframr-experiments-run pitch_resid_isolated_prodlike   # on the GPU host (defroster, RTX 4090 -- NOT fogbank)
"""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, prodlike_train_args

_IMAGE = "anarkiwi/preframr:0.2.25"


spec = ExperimentSpec(
    name="pitch_resid_isolated_prodlike",
    doc=(
        "Prodlike A/B/C isolating the universal pitch model's marginal effect: "
        "full_macros + universal_pitch + table_resid_split (target) vs full_macros "
        "(isolating reference) vs atomic (floor baseline), on preframr-tokens 0.46.3. "
        "1 seed, canonical body (tkvocab 32768, seq_len 8192, 60 epochs). Decisive = "
        "pitch_resid vs full_macros; PRIMARY read = eval_b_* held-out composer "
        "generalization; SECONDARY = content_tier_report (per-tier "
        "content_over_structural + per-op op_acc). Decisive pair first; floor last."
    ),
    tier="prodlike",
    image=_IMAGE,
    arms=[
        Arm(
            label="pitch_resid",
            macro_config="full_macros",
            macro_flags=("universal_pitch", "table_resid_split"),
        ),
        Arm(label="full_macros", macro_config="full_macros"),
        Arm(label="atomic", baseline=True),
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
