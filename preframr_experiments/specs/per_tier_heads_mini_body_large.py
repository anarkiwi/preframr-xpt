"""Phase 2 mini A/B for per-tier heads + MoS content head (Approach C); 3 seeds, body=large mini, 60 epochs. Spec details in `integration_tests/design/per_tier_heads_design.md`."""

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
    name="per_tier_heads_mini_body_large",
    doc=(
        "Phase 2 mini A/B for per-tier heads + MoS content head. Target arm "
        "first per convention; baseline last. 3 seeds for variance bound. "
        "Pass: (1) content acc not regressed vs baseline (3sigma floor); "
        "(2) router_acc > 0.7; (3) loop_collapse_rate <= baseline at "
        "temperature 0 and 1.0; (4) prompt_conditioning.diversity_ratio "
        "not regressed; (5) content_over_structural ratio >= baseline + "
        "1sigma."
    ),
    tier="mini",
    arms=[
        Arm(
            label="per_tier_heads_mos4",
            extra_cargs=(
                "--per-tier-heads --per-tier-content-mos-k 4 "
                "--per-tier-mos-entropy-lambda 0.0"
            ),
        ),
        Arm(label="baseline", baseline=True),
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
