"""Router-entropy retest at mini for per-tier heads + MoS K=4 mos4 arm. Trigger: per_tier_heads_mos_prodlike refute on criteria 3+4 (diversity_ratio 1.03, loop_collapse_rate 33%). Hypothesis: positive --per-tier-mos-entropy-lambda re-distributes the router posterior across tiers, restoring prompt-sensitivity. Compare against existing phase2 mos4 (lambda=0) audit JSONs at T=0.5."""

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
    name="per_tier_heads_entropy_retest",
    doc=(
        "Single-arm retest of mos4 at mini, body=large, 3 seeds with "
        "--per-tier-mos-entropy-lambda 0.01. Compare T=0.5 streams + audits "
        "against existing data/audit/per_tier_heads_phase2/loop_detection_"
        "per_tier_heads_mos4_T0.5.json + prompt_conditioning_per_tier_heads_"
        "mos4_T0.5.json (both at lambda=0). Pass: diversity_ratio > 1.2 AND "
        "val_acc within 1sigma of the lambda=0 mos4 mean."
    ),
    tier="mini",
    arms=[
        Arm(
            label="mos4_entropy_lambda_0p01",
            extra_cargs=(
                "--per-tier-heads --per-tier-content-mos-k 4 "
                "--per-tier-mos-entropy-lambda 0.01"
            ),
        ),
    ],
    metrics=[
        "alphabet_size",
        "val_loss_best",
        "val_acc_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=3,
    seq_len=4096,
    tkvocab=32768,
    max_perm=1,
    train_args=_TRAIN_ARGS,
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
)
