"""Mini run of the Stage-1 skeleton encoding: each clean held freq note is ONE
atomic SKEL atom (per-note token), on the de-merged (--tkvocab 0) PW+filter-ablated
substrate. Tests the thesis that a per-note token generates COHERENT melody where
the de-merged base-atom stream fragmented into "rapid clicks" (and where the merged
Unigram regime pinned V0~0).

Skeleton owns the freq channel: --skeleton-pass on, --no-freq-trajectory-pass,
freq-onset-pass off (so op48 is not emitted). Anchor pass on (skeleton consumes
traj_anchor for note segmentation). Notes with intra-note motion / off-semitone
held values stay raw op0 SET (RESID pass-through) at Stage 1.

Decisive gate: the constrained-decode WAV audition (does it sound like coherent
notes+rhythm, not clicks) + content_tier_report --onset (SKEL learnable; op48 gone).
Requires PREFRAMR_DATASET_CACHE_DISABLE=1.
"""

from pathlib import Path

from preframr_experiments.audit.ablate_pwfilter import ablate_staged_dumps
from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.14"

_BASE_MACROS = (
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c4",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
)

# Skeleton owns the freq channel (anchor for note segmentation + skeleton);
# freq_trajectory is left off (it conflicts with skeleton_pass).
_SKEL_MACROS = ("trajectory_anchor_pass", "skeleton_pass")

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)


def _ablate_hook(arm: Arm, work_dir: Path) -> None:
    train_root = Path(work_dir) / "train"
    if not train_root.exists():
        raise FileNotFoundError(
            f"train dumps not staged at {train_root}; ablation must run after stage_dumps"
        )
    n = ablate_staged_dumps(Path(work_dir))
    print(f"{arm.label} ablation: rewrote {n} staged dumps (PW+filter dropped)")


spec = ExperimentSpec(
    name="melody_skeleton_mini",
    doc=(
        "Mini Stage-1 skeleton encoding: per-note SKEL atom on the de-merged "
        "PW+filter-ablated substrate. Tests per-note-token generation coherence. "
        "--skeleton-pass owns the freq channel (no freq-trajectory/onset). 1 arm, "
        "mini body=large, 60 epochs, tkvocab 0. Requires "
        "PREFRAMR_DATASET_CACHE_DISABLE=1."
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(
            label="skeleton",
            macro_flags=_BASE_MACROS + _SKEL_MACROS,
            extra_cargs="--tkvocab 0",
            baseline=True,
        ),
    ],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
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
    pre_run_hook=_ablate_hook,
)
