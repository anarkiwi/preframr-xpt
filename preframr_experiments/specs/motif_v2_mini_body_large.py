"""Phase 2 mini 3-arm A/B comparing the corpus-mined motif pass: no-motif vs v1
exact-match vs v2 value-slotted templates (preframr-tokens 0.22.0). All arms run
the confirmed full_macros representation. The two motif arms add --motif-pass with
a dict a pre_run_hook mines from the staged train dumps — v1 (exact (op,reg,subreg,
val,diff) BPE) vs v2 (--motif-mine-version 2: shape templates + content-tier value
slots). 3 seeds, mini body=large, 60 epochs.

Decisive read: per_class CONTENT-tier val_acc (run post-hoc in the xpt image, which
must layer on :0.2.3 so MOTIF_ARG classifies as content) + loop_collapse /
prompt-conditioning. The v1 A/B already showed exact-match doesn't help and is
confounded (it hides content in zero-tier MOTIF_OP); v2 exposes the slot as content
and de-fragments — this arm tests whether that helps vs v1 and no-motif.

Requires PREFRAMR_DATASET_CACHE_DISABLE=1 (the hook mines a per-arm dict every seed)
and image anarkiwi/preframr:0.2.3 (the --motif-mine-version flag + v2 tokenizer +
the MOTIF_ARG->content tier fix)."""

from __future__ import annotations

import shlex
import subprocess
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

_FULL_MACRO_CARGS = (
    "--ctrl-triple-pass "
    "--freq-nudge-pass "
    "--release-update-pass "
    "--lonely-catch-all"
)

_MOTIF_DICT_CONTAINER_PATH = "/scratch/preframr/motif_dict.json"
_IMAGE = "anarkiwi/preframr:0.2.3"
_MOTIF_K = 256
_MOTIF_MIN_COUNT = 3
_MOTIF_MIN_COMPOSERS = 6
_ARM_MINE_VERSION = {"full_macros_motif_v2": 2, "full_macros_motif_v1": 1}

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)

_MOTIF_CARGS = (
    f"{_FULL_MACRO_CARGS} --motif-pass --motif-dict {_MOTIF_DICT_CONTAINER_PATH}"
)


def _mine_motif_dict(arm: Arm, work_dir: Path) -> None:
    """Mine the per-arm motif dict (v1 or v2 by arm label) from the staged train
    dumps into work_dir/motif_dict.json. No-op for the no-motif baseline arm."""
    version = _ARM_MINE_VERSION.get(arm.label)
    if version is None:
        return
    train_root = Path(work_dir) / "train"
    if not train_root.exists():
        raise FileNotFoundError(
            f"train dumps not staged at {train_root}; "
            "motif mining must run after stage_dumps"
        )
    cmd = [
        "docker",
        "run",
        "--rm",
        "--memory=16g",
        "-v",
        f"{Path(work_dir).resolve()}:/scratch/preframr",
        _IMAGE,
        "python3",
        "/preframr/mine_motifs.py",
        "--no-require-pq",
        "--pipeline-spec",
        "@/scratch/preframr/pipeline_spec.json",
        *shlex.split(_FULL_MACRO_CARGS),
        "--reglogs",
        "/scratch/preframr/train/*/*.dump.parquet",
        "--max-files",
        "999999",
        "--motif-out",
        _MOTIF_DICT_CONTAINER_PATH,
        "--motif-k",
        str(_MOTIF_K),
        "--motif-min-count",
        str(_MOTIF_MIN_COUNT),
        "--motif-min-composers",
        str(_MOTIF_MIN_COMPOSERS),
        "--motif-mine-version",
        str(version),
    ]
    subprocess.run(cmd, check=True)


spec = ExperimentSpec(
    name="motif_v2_mini_body_large",
    doc=(
        "Mini 3-arm: full_macros + v2 templated motif (target) vs + v1 exact "
        "motif vs no-motif. 3 seeds, mini body=large, 60 epochs. Decisive gate = "
        "per_class content-tier val_acc + loop/prompt; all-tier is CONFOUNDED "
        "(each arm tokenizes differently). Requires PREFRAMR_DATASET_CACHE_DISABLE=1 "
        "and image anarkiwi/preframr:0.2.3."
    ),
    tier="mini",
    arms=[
        Arm(label="full_macros_motif_v2", extra_cargs=_MOTIF_CARGS),
        Arm(label="full_macros_motif_v1", extra_cargs=_MOTIF_CARGS),
        Arm(label="full_macros", extra_cargs=_FULL_MACRO_CARGS, baseline=True),
    ],
    metrics=[
        "alphabet_size",
        "longtail_frac",
        "worst_family_longtail_frac",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=3,
    seq_len=4096,
    tkvocab=32768,
    max_perm=1,
    image=_IMAGE,
    train_args=_TRAIN_ARGS,
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
    pre_run_hook=_mine_motif_dict,
)
