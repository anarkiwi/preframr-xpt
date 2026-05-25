"""Phase 2 mini A/B for the cluster-conditional content head (queue item 2). Target arm = ClusterContentHead (C=256, structural-feature cluster index from Phase 0a). Baseline arm = mos4 + entropy lambda=0.02 (best-known mini config from the entropy sweep). 3 seeds; pass criteria: diversity_ratio > 1.65 at T=0.5 mini (proxy for >1.2 at prodlike per v11's 29% attenuation) AND val_acc within 1sigma of baseline. The pre_run_hook copies the cluster_assignments.json from the main repo's integration_tests/data/content_clusters/ tree into work_dir so the train container sees it under /scratch/preframr/cluster_assignments.json. Cache MUST be disabled for this spec (PREFRAMR_DATASET_CACHE_DISABLE=1) so the hook runs every seed."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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

_CLUSTER_INDEX_SRC = Path(
    os.environ.get(
        "PREFRAMR_CONTENT_CLUSTER_INDEX",
        "/scratch/anarkiwi/preframr-xpt/data/content_clusters/prodlike_991929_structural_c256.json",
    )
)


def _copy_cluster_index(_arm, work_dir: Path) -> None:
    """Stage cluster_assignments.json into work_dir so the train container sees it at /scratch/preframr/cluster_assignments.json."""
    if not _CLUSTER_INDEX_SRC.exists():
        raise FileNotFoundError(
            f"cluster index not found at {_CLUSTER_INDEX_SRC}; "
            f"set PREFRAMR_CONTENT_CLUSTER_INDEX or build via "
            f"preframr_experiments/audit/build_content_clusters.py"
        )
    dst = Path(work_dir) / "cluster_assignments.json"
    shutil.copy2(_CLUSTER_INDEX_SRC, dst)


spec = ExperimentSpec(
    name="cluster_content_mini_body_large",
    doc=(
        "Phase 2 mini A/B for ClusterContentHead. Target: cluster_C256 + "
        "structural-feature cluster index (Phase 0a). Baseline: mos4 + entropy "
        "lambda=0.02 (best-known mini config). 3 seeds. Pass: (1) val_acc "
        "within 1sigma of baseline; (2) diversity_ratio at T=0.5 > 1.65 (proxy "
        "for >1.2 at prodlike); (3) loop_collapse_rate at T=0.5 <= baseline; "
        "(4) cluster_log_p entropy histogram non-uniform (sanity). Requires "
        "PREFRAMR_DATASET_CACHE_DISABLE=1 (pre_run_hook must run every seed)."
    ),
    tier="mini",
    arms=[
        Arm(
            label="cluster_C256",
            extra_cargs=(
                "--per-tier-heads --content-cluster-head "
                "--content-cluster-c 256 "
                "--content-cluster-index /scratch/preframr/cluster_assignments.json"
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
    pre_run_hook=_copy_cluster_index,
)
