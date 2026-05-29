"""Mini 2-arm A/B isolating the Unigram merge as the V0-onset blocker.

Phase 0 on `melody_substrate_iter_mini` localised V0-onset=0 to the tokenizer:
on the clean substrate the pitch onset atom (op45 V0_HI) is a ~2-value target
(mode-ceiling 0.646) yet scores 0.013, because the Unigram tokenizer welds it
forward into 9489-distinct compound tokens (merge-len 7.37) that bundle the
pitch with V0_LO + the per-frame DELTA shape samples. The model nails the
compound SHAPE but cannot isolate the PITCH. Rather than de-weld surgically
(melody_merge_split), this arm removes the confound wholesale: `--tkvocab 0`
trains directly on the base-atom alphabet (~2050), so V0_HI is a standalone
2-value atom.

Both arms run the same clean substrate (PW+filter ablation) + freq stack
(anchor + interval-V0 + freq-onset-pass). The ONLY difference is the tokenizer:

- no_unigram: `--tkvocab 0` -- no merging; every position is one base atom.
- merged: spec tkvocab=32768 -- the existing substrate_no_macros baseline
  (reproduces V0_HI=0.013 as a setup sanity check).

Decisive read: `content_tier_report --onset` + the op x subreg split on op45
sr1 (V0_HI). If V0_HI moves off ~0 with merging gone, the blocker is the
tokenizer (structural), not capacity/scale. If it stays ~0 on a standalone
2-value target, the structural-merge story is wrong.

Requires PREFRAMR_DATASET_CACHE_DISABLE=1 (arms tokenize differently).
"""

from pathlib import Path

from preframr_experiments.audit.ablate_pwfilter import ablate_staged_dumps
from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.12"

_BASE_TRANSFORMS = [
    {"name": "freq_trajectory"},
    {"name": "preset"},
    {"name": "hard_restart"},
    {"name": "legato_per_cluster", "params": {"clusters": [2, 4]}},
    {"name": "voice_block_order"},
    {"name": "ctrl_bigram"},
    {"name": "loop"},
]

_FREQ_STACK = "--trajectory-anchor-pass --freq-v0-interval --freq-onset-pass"

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
    name="melody_no_unigram_mini",
    doc=(
        "Mini 2-arm A/B: disable the Unigram tokenizer (--tkvocab 0) vs the "
        "merged baseline, both on the clean PW+filter-ablated substrate + freq "
        "stack. Tests whether the Unigram weld of pitch onset into compound "
        "shape tokens is the V0-onset=0 blocker. Decisive: op45 sr1 (V0_HI) "
        "acc via content_tier_report --onset. 3 seeds, mini body=large, 60 "
        "epochs. Requires PREFRAMR_DATASET_CACHE_DISABLE=1."
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(label="no_unigram", extra_cargs=f"{_FREQ_STACK} --tkvocab 0"),
        Arm(label="merged", extra_cargs=_FREQ_STACK, baseline=True),
    ],
    metrics=[
        "alphabet_size",
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
    train_args=_TRAIN_ARGS,
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
    pre_run_hook=_ablate_hook,
)
