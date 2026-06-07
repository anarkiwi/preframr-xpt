"""Generator-era canonical prodlike A/B (tokens 0.46.1) -- the decisive cross-tune generalization read.

Target = the most-learnable-by-construction encoding: full_macros + universal_pitch + table_resid_split
(the per-voice universal note-index for melody onsets AND for GEN_TABLE arps via the NOTE_UNIV residual-
split / item #4 keying). Baseline = the atomic control (all passes OFF). 1 seed, canonical body, on
anarkiwi/preframr:0.2.23 (preframr-tokens 0.46.1).

WHY THIS A/B, not within-block triage: the within-block induction-copy triage is exhausted -- it is FLAT
across every generator encoding (atomic copy 0.930 beats the generator family ~0.916; measured for 0.45.1,
universal_pitch, universal_freq, table_resid_split, and the full item #4). The triage is structurally blind
to the lever these encodings target -- cross-tune transfer (universal_pitch lifts absolute transfer +67%,
0.094->0.157, measured separately). Only training settles whether that transfer translates to
generalization. Decisive read = audit.content_tier_report (per-tier content_over_structural + per-op
op_acc, on the corrected 0.46.1 tier map); secondary = eval_b_* held-out composer families.

Replaces melody_skeleton_prodlike.py (0.45.1 / no residual-split). Launch:
  preframr-experiments-run pitch_resid_prodlike   # on the GPU host (defroster, RTX 4090 -- NOT fogbank)
"""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, prodlike_train_args

_IMAGE = "anarkiwi/preframr:0.2.23"


spec = ExperimentSpec(
    name="pitch_resid_prodlike",
    doc=(
        "Prodlike A/B: full_macros + universal_pitch + table_resid_split (per-voice "
        "note-index + GEN_TABLE NOTE_UNIV residual-split, item #4) vs the atomic "
        "baseline, on preframr-tokens 0.46.1. 1 seed, canonical body (tkvocab 32768, "
        "seq_len 8192, 60 epochs). Decisive gate = content_tier_report (per-tier "
        "content_over_structural + per-op op_acc); secondary = eval_b_* held-out "
        "composer generalization. Target arm first; baseline last."
    ),
    tier="prodlike",
    image=_IMAGE,
    arms=[
        Arm(
            label="pitch_resid",
            macro_config="full_macros",
            macro_flags=("universal_pitch", "table_resid_split"),
        ),
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
