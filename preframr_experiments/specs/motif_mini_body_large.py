"""Phase 1 mini A/B for the corpus-mined motif pass (preframr-tokens 0.20.0). Both
arms run the full_macros representation (the confirmed prodlike win: base pipeline
+ CTRL_TRIPLE / FREQ_NUDGE / RELEASE_UPDATE / lonely_catch_all). The target arm
adds the motif pass: a pre_run_hook mines a cross-composer motif dictionary from
the staged train dumps (motif OFF) and the train/tokenize parse then collapses
those motifs into MOTIF_OP atoms (lossless; expanded byte-exact on decode). 3
seeds, mini body=large, 60 epochs.

Measures (1) compression: encoded_tokens_per_song must DROP vs baseline (the
~24% atom-level reduction the prodlike probe found should carry to tokens); (2)
learnability: eval_a content-tier val_acc must not regress > 1 sigma vs baseline,
and the per_class content audit must hold (the motif tokens are LOSS_TIER zero, so
the content tier is measured on the un-collapsed atoms); (3) generalization:
loop_collapse / prompt-conditioning must not worsen -- motifs absorb content
(melodic figures) more than structure, so the open risk is amplified replay, not
just compression.

Requires PREFRAMR_DATASET_CACHE_DISABLE=1 (the hook mines a spec-dependent dict
every seed; the dataset cache would skip it). Requires an image with
preframr-tokens >= 0.20.0 baked (the motif pass + mine_motifs CLI); the runner's
narrow train/-only src-bind does not carry these, so this spec runs on a rebaked
image, not --bind-src."""

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
_MOTIF_ARM = "full_macros_motif"
_IMAGE = "anarkiwi/preframr:0.2.0"
_MOTIF_K = 256
_MOTIF_MIN_COUNT = 3
_MOTIF_MIN_COMPOSERS = 6

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)


def _mine_motif_dict(arm: Arm, work_dir: Path) -> None:
    """For the motif arm, mine a motif dict from the staged train dumps (motif
    OFF, same full_macros pipeline) into work_dir/motif_dict.json so the parse /
    tokenize / train containers read it at the container path. No-op for the
    baseline arm."""
    if arm.label != _MOTIF_ARM:
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
    ]
    subprocess.run(cmd, check=True)


spec = ExperimentSpec(
    name="motif_mini_body_large",
    doc=(
        "Phase 1 mini A/B for the corpus-mined motif pass. Target: "
        "full_macros + motif (k=256, min_composers=6). Baseline: full_macros. "
        "3 seeds, mini body=large, 60 epochs. Pass: (1) encoded_tokens_per_song "
        "DROPS vs baseline; (2) eval_a content-tier val_acc within 1 sigma of "
        "baseline (per_class content audit decisive); (3) loop_collapse / "
        "prompt-conditioning not worse. Requires PREFRAMR_DATASET_CACHE_DISABLE=1 "
        "(hook mines a spec-dependent dict every seed) and a rebaked image with "
        "preframr-tokens >= 0.20.0."
    ),
    tier="mini",
    arms=[
        Arm(
            label=_MOTIF_ARM,
            extra_cargs=(
                f"{_FULL_MACRO_CARGS} "
                f"--motif-pass --motif-dict {_MOTIF_DICT_CONTAINER_PATH}"
            ),
        ),
        Arm(
            label="full_macros",
            extra_cargs=_FULL_MACRO_CARGS,
            baseline=True,
        ),
    ],
    metrics=[
        "alphabet_size",
        "longtail_frac",
        "worst_family_longtail_frac",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "val_loss_eval_a_best",
        "val_acc_eval_a_at_best_loss",
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
