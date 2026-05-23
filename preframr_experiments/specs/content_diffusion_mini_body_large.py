"""Phase 2 mini A/B for content diffusion (Approach A). Target: DiffusionContentHead (D3PM absorbing-state, T=8 cosine schedule). Baseline: mos4 + entropy lambda=0.02 (best-known mini config, same comparator as the cluster head Phase 2). 3 seeds. Pass: (1) val_acc within 1sigma of baseline; (2) diversity_ratio at T=0.5 > 1.65 (proxy for >1.2 at prodlike); (3) loop_collapse_rate at T=0.5 <= baseline. Tests whether non-autoregressive content modeling closes the prompt-conditioning gap that every per-token architecture (mos, entropy, mask, cluster) has missed at prodlike."""

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
    name="content_diffusion_mini_body_large",
    doc=(
        "Phase 2 mini A/B for DiffusionContentHead. Target: diffusion_T8 "
        "(D3PM absorbing-state, T=8). Baseline: mos4 + entropy lambda=0.02. "
        "3 seeds. Pass: (1) val_acc within 1sigma of baseline; (2) "
        "diversity_ratio at T=0.5 > 1.65 (proxy for >1.2 at prodlike); "
        "(3) loop_collapse_rate at T=0.5 <= baseline. Per-token-head "
        "framing has been refuted four times (mos / entropy / mask / "
        "cluster); diffusion tests the non-autoregressive content "
        "modeling framing instead."
    ),
    tier="mini",
    arms=[
        Arm(
            label="diffusion_T8",
            extra_cargs=(
                "--per-tier-heads --content-diffusion --content-diffusion-t 8"
            ),
        ),
        Arm(
            label="mos4_entropy_0p02",
            extra_cargs=(
                "--per-tier-heads --per-tier-content-mos-k 4 "
                "--per-tier-mos-entropy-lambda 0.02"
            ),
            baseline=True,
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
