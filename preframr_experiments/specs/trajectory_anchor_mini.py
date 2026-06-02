"""Mini 2-arm A/B for the trajectory-anchoring tokenizer fix (preframr-tokens 0.25.0):
does anchoring FREQ_TRAJ at its real note/sweep origin make melody learnable? The
confirmed full_macros content win is SET register scaffolding, NOT melody -- FREQ_TRAJ
(op 45) sits at ~0.026 content acc because FreqTrajectoryPass segments on value-runs and
discards the gate/sweep anchor, so the note-onset pitch reaches the model as un-anchored
noise. TrajectoryAnchorPass recovers the per-register origin (sustained-departure ∪ gate
retrigger, with ramp/oscillator collapse) and forces FreqTrajectoryPass to start each
trajectory there. See design/trajectory_anchoring.md (research framing) +
preframr-tokens/design/freq_trajectory_anchoring.md (impl).

Target arm `anchored`: full_macros + --trajectory-anchor-pass (the pass is opt-in,
default OFF in the framework; toggled per-arm via extra_cargs like the absorber macros
and --motif-pass, so apply_pipeline_spec_to_args does not clobber it). Baseline arm
`unanchored`: the confirmed full_macros encoding, anchor off. The ONLY arm difference is
the anchor pass; both ride the shared base pipeline + full_macros absorbers. 3 seeds,
mini body=large, 60 epochs.

Decisive read (design step 2): per_class CONTENT-tier val_acc split by op
(audit_checkpoint_per_class + the by-op parser) -- (1) does FREQ_TRAJ (op 45) acc RISE
from its ~0.026 floor, and (2) does overall content-tier acc lift BEYOND the
SET-scaffolding plateau (baseline content ~0.32)? all-tier val_acc is CONFOUNDED (the
arms tokenize FREQ_TRAJ differently). A rise => anchoring makes melody learnable,
supersedes the SET-only win, re-opens melodic augmentation (preframr-aug) on a learnable
substrate. Flat => the anchor is not the lever (freq_core_ablation_mini is the
complementary tiebreaker: core aleatoric vs drowned by PW/filter noise).

No pre_run_hook and no PREFRAMR_DATASET_CACHE_DISABLE needed: the arms differ only by a
parse flag, which _dataset_affecting_cargs folds into the dataset cache key, so the two
arms key apart automatically and re-tokenize fresh.

Requires image anarkiwi/preframr:0.2.6 (the --trajectory-anchor-pass wiring + tokens
0.25.0 TrajectoryAnchorPass)."""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.6"

# Both arms ride the registered full_macros encoding; the only difference is the
# opt-in trajectory_anchor_pass on the target arm.

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)


spec = ExperimentSpec(
    name="trajectory_anchor_mini",
    doc=(
        "Mini 2-arm A/B: full_macros + trajectory anchoring (FREQ_TRAJ origins "
        "recovered at note/sweep anchors) vs the confirmed full_macros encoding. "
        "3 seeds, mini body=large, 60 epochs. Decisive gate = per_class CONTENT-tier "
        "val_acc split by op -- does FREQ_TRAJ (op45) acc rise from ~0.026 and does "
        "content lift beyond the SET-scaffolding plateau? all-tier is CONFOUNDED. "
        "Requires anarkiwi/preframr:0.2.6."
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(
            label="anchored",
            macro_config="full_macros",
            macro_flags=("trajectory_anchor_pass",),
        ),
        Arm(label="unanchored", macro_config="full_macros", baseline=True),
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
)
