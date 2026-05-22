"""Option-1 probe at mini for the per_tier_heads refute. Single arm: plain-CE single-head (no --per-tier-heads) + --mask-structural-tier-loss, mini body=large, 3 seeds, 60-ep. Tests whether hard-masking structural-tier targets on the simpler architecture closes the diversity gap, as a separator vs the router-architecture framing that the entropy retest tested. If diversity_ratio > 1.2 at T=0.5 here AND content acc is not regressed vs the existing phase2 baseline, the gradient-dominance framing is load-bearing and per-tier-heads is not needed. Compare against data/audit/per_tier_heads_phase2/loop_detection_baseline_T0.5.json + prompt_conditioning_baseline_T0.5.json (the lambda=0 plain-CE baseline)."""

from __future__ import annotations

from preframr_experiments.base import (
    Arm,
    ExperimentSpec,
    mini_train_args,
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

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)


spec = ExperimentSpec(
    name="mask_structural_loss_mini",
    doc=(
        "Single-arm: plain-CE single-head + --mask-structural-tier-loss, "
        "mini body=large, 3 seeds. Pass: diversity_ratio > 1.2 at T=0.5 AND "
        "content acc (from audit_checkpoint_per_class.py) within 1sigma of "
        "the phase2 baseline content acc. Comparator streams + audits at "
        "data/audit/per_tier_heads_phase2/{loop_detection,prompt_conditioning}"
        "_baseline_T0.5.json (lambda=0 plain CE)."
    ),
    tier="mini",
    arms=[
        Arm(
            label="mask_structural",
            extra_cargs="--mask-structural-tier-loss",
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
