"""Entropy-lambda sweep at mini to pick the lambda for the next prodlike retest. v11 prodlike (lambda=0.01) hit diversity_ratio 1.123 — just below the 1.2 gate. Mini hit 1.535 at the same lambda, so prodlike attenuates diversity_ratio by ~1.37x. To clear 1.2 at prodlike we want approximately 1.65 at mini, which likely needs higher lambda. Single seed per arm; result is a trend probe, not a verdict. Pair with the existing per_tier_heads_entropy_retest (lambda=0.01) for a 3-point curve."""

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
    name="per_tier_heads_entropy_sweep_mini",
    doc=(
        "Two-arm entropy lambda sweep at mini, body=large, single seed. "
        "Target: pick lambda that gives diversity_ratio >= 1.65 at mini "
        "(proxy for >= 1.2 at prodlike, per the v11 attenuation ratio). "
        "Comparator: existing lambda=0.01 mini (diversity_ratio 1.535) "
        "from per_tier_heads_entropy_retest. Pair gives 3-point curve."
    ),
    tier="mini",
    arms=[
        Arm(
            label="mos4_entropy_lambda_0p02",
            extra_cargs=(
                "--per-tier-heads --per-tier-content-mos-k 4 "
                "--per-tier-mos-entropy-lambda 0.02"
            ),
        ),
        Arm(
            label="mos4_entropy_lambda_0p05",
            extra_cargs=(
                "--per-tier-heads --per-tier-content-mos-k 4 "
                "--per-tier-mos-entropy-lambda 0.05"
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
    seeds=1,
    seq_len=4096,
    tkvocab=32768,
    max_perm=1,
    train_args=_TRAIN_ARGS,
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
)
