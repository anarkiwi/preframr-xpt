"""Mini 2-arm A/B for interval-coded freq V0 (preframr-tokens 0.26.0): does encoding the
melodic onset as a transposition-invariant interval (instead of absolute pitch) finally make
the V0 onset learnable? Motivated by trajectory_anchor_mini: the onset line is trigram-0.79
predictable yet the model gets V0-onset acc 0.000 -- because absolute-pitch onsets don't
transfer across keys. Interval-coding (--freq-v0-interval) makes the same motif identical
tokens at any pitch. See design/trajectory_anchoring.md + preframr-tokens/design/freq_v0_interval.md.

Both arms run full_macros + trajectory anchoring; the ONLY difference is --freq-v0-interval,
toggled per-arm via extra_cargs (opt-in flag, not clobbered by apply_pipeline_spec_to_args;
dataset-affecting so the cache keys the arms apart -- no hook/cache-disable needed). 3 seeds,
mini body=large, 60 epochs, :0.2.7.

Decisive read: the V0-ONSET subreg split (op45 subreg 1/2), NOT aggregate op45 -- does V0-onset
acc rise off 0.000? Plus audit.melody_predictability on the interval onset line (should be
low-entropy AND now captured by the model). Mini is data-starved for melody, so a clear onset-acc
lift here is strong; flat would push to the interval-vs-absolute prodlike for the real test.
all-tier val_acc is CONFOUNDED (the arms tokenize the onset differently)."""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.7"

_BASE_MACROS = (
    "freq_trajectory_pass",
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c4",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
)

_ANCHOR_MACROS = _BASE_MACROS + (
    "ctrl_triple_pass",
    "freq_nudge_pass",
    "release_update_pass",
    "lonely_catch_all",
    "trajectory_anchor_pass",
)

_INTERVAL_MACROS = _ANCHOR_MACROS + ("freq_v0_interval",)

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)


spec = ExperimentSpec(
    name="freq_v0_interval_mini",
    doc=(
        "Mini 2-arm A/B: anchored full_macros + interval-coded freq V0 vs anchored-absolute. "
        "3 seeds, mini body=large, 60 epochs, :0.2.7. Decisive gate = V0-onset subreg split "
        "(op45 subreg 1/2) -- does onset acc rise off 0.000? -- + melody_predictability on the "
        "interval onset line. all-tier is CONFOUNDED."
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(label="interval", macro_flags=_INTERVAL_MACROS),
        Arm(label="absolute", macro_flags=_ANCHOR_MACROS, baseline=True),
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
