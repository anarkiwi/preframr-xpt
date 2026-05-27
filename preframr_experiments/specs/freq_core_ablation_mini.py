"""Diagnostic A/B: is the musical CORE (frequency + control/gate + ADSR) learnable
once the high-entropy timbral modulation (pulse-width + filter) is removed, or is the
core itself aleatoric? Motivated by the FREQ_TRAJ probes: the full_macros content win
is SET register content, not melody (FREQ_TRAJ ~0.026 acc); the model declines to emit
FREQ_TRAJ at all, betting on the dominant SET stream. The tracker models (defMON /
SID Wizard) confirm melody is anchored at the CONTROL gate (note-on at pattern rows) +
base note, with per-frame frequency = base + arpeggio + vibrato -- so the learnable
melodic anchor is the gate, which this ablation RETAINS.

Target arm `freq_core`: a pre_run_hook ablates every staged train+eval dump (drop PW +
filter writes, inject one PW=midpoint per voice; the parser's DELAY mechanism
consolidates the emptied frames, timing preserved -- verified by render). Baseline arm
`full`: the un-ablated full signal. Both run the confirmed full_macros representation.

Decisive read: per_class CONTENT-tier val_acc split by op (audit_checkpoint_per_class
+ the by-op parser) -- does FREQ_TRAJ (op 45) and CONTROL/gate accuracy RISE once PW/
filter stop competing for capacity and shortening context? all-tier is CONFOUNDED (the
arms tokenize different content). Outcome either way is informative: a rise => timbral
modulation was the noise drowning the core (representation lever: tier it off); flat ~0
=> the melodic core is aleatoric (pivot to a distributional/perceptual metric).

Requires PREFRAMR_DATASET_CACHE_DISABLE=1 (the hook mutates staged dumps every seed)."""

from __future__ import annotations

from pathlib import Path

from preframr_experiments.audit.ablate_pwfilter import ablate_staged_dumps
from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

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
    "--ctrl-triple-pass --freq-nudge-pass --release-update-pass --lonely-catch-all"
)

_ABLATE_ARM = "freq_core"

_TRAIN_ARGS = mini_train_args(body="large").replace("--max-epochs 160", "--max-epochs 60")


def _ablate_hook(arm: Arm, work_dir: Path) -> None:
    """Ablate PW+filter from the staged dumps for the freq_core arm; no-op for the
    full baseline. Runs after stage_dumps, before parse/tokenize."""
    if arm.label != _ABLATE_ARM:
        return
    train_root = Path(work_dir) / "train"
    if not train_root.exists():
        raise FileNotFoundError(
            f"train dumps not staged at {train_root}; ablation must run after stage_dumps"
        )
    n = ablate_staged_dumps(Path(work_dir))
    print(f"freq_core ablation: rewrote {n} staged dumps (PW+filter dropped)")


spec = ExperimentSpec(
    name="freq_core_ablation_mini",
    doc=(
        "Mini 2-arm diagnostic: full_macros on the freq+ctrl+ADSR CORE (PW+filter "
        "ablated, PW set to midpoint) vs the full signal. 3 seeds, mini body=large, "
        "60 epochs. Decisive gate = per_class CONTENT-tier val_acc split by op "
        "(FREQ_TRAJ op45 + CONTROL/gate) -- does removing timbral modulation make the "
        "melodic core learnable? all-tier is CONFOUNDED. Requires "
        "PREFRAMR_DATASET_CACHE_DISABLE=1."
    ),
    tier="mini",
    arms=[
        Arm(label=_ABLATE_ARM, extra_cargs=_FULL_MACRO_CARGS),
        Arm(label="full", extra_cargs=_FULL_MACRO_CARGS, baseline=True),
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
