"""Phase 3 prodlike v12: per-tier heads + MoS K=4 with --per-tier-mos-entropy-lambda 0.02 (raised from v11's 0.01). Trigger: v11 (lambda=0.01) landed criteria 1,2,3 PASS but criterion 4 borderline FAIL at diversity_ratio 1.123 (gate 1.2). Mini sweep showed inverted-U with peak at lambda=0.02 (diversity_ratio 1.596 at mini, predicted ~1.140 at prodlike under v11's 29% mini-to-prodlike lift attenuation). This run verifies the extrapolation; if it clears 1.2, we ship; if it lands ~1.14 as predicted, pivot to model_loss_queue.md item 2 (cluster-conditional content head)."""

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
    name="per_tier_heads_entropy_prodlike_v12",
    doc=(
        "Phase 3 prodlike v12: mos4 + --per-tier-mos-entropy-lambda 0.02. "
        "v11 (lambda=0.01) verdict was 3-of-4 PASS, criterion 4 FAIL by 0.077 "
        "(diversity_ratio 1.123 vs gate 1.2). Mini sweep at {0.01, 0.02, 0.05} "
        "shows peak at 0.02 (mini diversity_ratio 1.596, predicted prodlike "
        "1.140). Target arm only; baseline ckpt from per_tier_heads_prodlike "
        "is the comparator."
    ),
    tier="prodlike",
    arms=[
        Arm(
            label="per_tier_heads_mos4_entropy_0p02",
            extra_cargs=(
                "--per-tier-heads --per-tier-content-mos-k 4 "
                "--per-tier-mos-entropy-lambda 0.02"
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
