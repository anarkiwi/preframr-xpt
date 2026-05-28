"""Mini 3-arm A/B on the PW+filter-ablated melodic substrate (extends
`freq_core_ablation_mini`, which was 2-arm and concluded "mini capacity, not
melody"; that conclusion used the broken row-index op reader -- re-read via
content_tier_report after the vocab_atom fix).

Hypothesis: melody is failing to learn not because the model lacks capacity
(freq_onset_channel mini hit SET 0.83) but because filter+PW writes dominate
the stream and the absorber macros' wins are mostly re-typing THOSE SETs. With
PW=50% constant and filter regs dropped, the surviving substrate is
freq+ctrl/gate+ADSR+frame -- the tracker-model anchor for melody.

Arms (same ablation, different encoding layered on top):
- substrate_no_macros: anchored + interval-V0 + freq-onset-pass, NO absorber
  macros. Pure freq-stack on the cleaned substrate. The diagnostic baseline:
  do op45/op48 move off 0 when the noise dominators are gone AND there are no
  absorbers to confound the per-op read?
- substrate_full_macros: anchored + interval-V0 + freq-onset-pass + the four
  absorbers (ctrl_triple / freq_nudge / release_update / lonely_catch_all).
  Tests whether absorbers still pay on the cleaned substrate.
- baseline_full_stack: un-ablated, full stack (the `freq_onset_channel_mini`
  onset_chan target). Reference point for what the full signal gives at mini.

Decisive read: `content_tier_report --onset` on the unified V0-onset bucket
(op45 V0 + op48 FREQ_ONSET + op47 NUDGE pitch on freq regs), with the FIXED
vocab_atom-based id->op map. The substrate_no_macros vs substrate_full_macros
delta also tells us whether absorber wins survive when filter+PW SETs are gone.

Requires PREFRAMR_DATASET_CACHE_DISABLE=1 (pre_run_hook mutates staged dumps
every seed). The audit_checkpoint_per_class fix (vocab_atom emission) must be
in the xpt image; rebake or run with `--bind-src` (with OK)."""

from __future__ import annotations

from pathlib import Path

from preframr_experiments.audit.ablate_pwfilter import ablate_staged_dumps
from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.11"

_BASE_TRANSFORMS = [
    {"name": "freq_trajectory"},
    {"name": "preset"},
    {"name": "hard_restart"},
    {"name": "legato_per_cluster", "params": {"clusters": [2, 4]}},
    {"name": "voice_block_order"},
    {"name": "ctrl_bigram"},
    {"name": "loop"},
]

_FREQ_STACK = (
    "--trajectory-anchor-pass --freq-v0-interval --freq-onset-pass"
)
_ABSORBERS = (
    "--ctrl-triple-pass --freq-nudge-pass --release-update-pass --lonely-catch-all"
)

_ABLATE_ARMS = frozenset({"substrate_no_macros", "substrate_full_macros"})

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)


def _ablate_hook(arm: Arm, work_dir: Path) -> None:
    if arm.label not in _ABLATE_ARMS:
        return
    train_root = Path(work_dir) / "train"
    if not train_root.exists():
        raise FileNotFoundError(
            f"train dumps not staged at {train_root}; ablation must run after stage_dumps"
        )
    n = ablate_staged_dumps(Path(work_dir))
    print(f"{arm.label} ablation: rewrote {n} staged dumps (PW+filter dropped)")


spec = ExperimentSpec(
    name="melody_substrate_iter_mini",
    doc=(
        "Mini 3-arm A/B on the PW+filter-ablated melodic substrate. "
        "substrate_no_macros (freq-stack only) vs substrate_full_macros vs "
        "baseline_full_stack (un-ablated). Decisive gate = content_tier_report "
        "--onset (V0-onset unified bucket) with the vocab_atom fix. 3 seeds, "
        "mini body=large, 60 epochs. Requires PREFRAMR_DATASET_CACHE_DISABLE=1."
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(label="substrate_no_macros", extra_cargs=_FREQ_STACK),
        Arm(label="substrate_full_macros", extra_cargs=f"{_FREQ_STACK} {_ABSORBERS}"),
        Arm(
            label="baseline_full_stack",
            extra_cargs=f"{_FREQ_STACK} {_ABSORBERS}",
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
