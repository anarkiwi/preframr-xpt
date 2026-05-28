"""Mini 2-arm A/B for the FREQ_ONSET channel (preframr 0.2.9 + preframr-tokens 0.27.0): does
re-tagging residual op0 SET on TRAJ_REGS into the onset channel (op48) shift the unified
freq-pitch acc, on top of anchored+interval? Both arms full_macros + anchor + interval; the
ONLY arm difference is --freq-onset-pass (a parse/tokenize transform, so the arms WILL
tokenize differently -- not stripped from the cache key).

Decisive read: `content_tier_report --onset` -- the onset bucket now unifies op45 V0 +
op48 FREQ_ONSET + op47 FREQ_NUDGE pitch on freq regs (the full freq-pitch denominator).
The split-encoding baseline has melody fragmented across op0-SET-freq (12% of stream, acc
0.0013) + op45 V0 (0.9%, 0) + op47 nudge (0.4%, 0); the onset-channel arm consolidates the
12% SET fragment into op48. Expected at mini: onset count in op48 channel rises; whether
unified V0-onset acc moves off 0.000 is the (scale-bound) open question -- mini caveat
still applies. all-tier and SET share are CONFOUNDED (different tokenizations); the clean
read is the unified-melody acc."""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.9"

_BASE_TRANSFORMS = [
    {"name": "freq_trajectory"},
    {"name": "preset"},
    {"name": "hard_restart"},
    {"name": "legato_per_cluster", "params": {"clusters": [2, 4]}},
    {"name": "voice_block_order"},
    {"name": "ctrl_bigram"},
    {"name": "loop"},
]

_INTERVAL_CARGS = (
    "--ctrl-triple-pass --freq-nudge-pass --release-update-pass --lonely-catch-all "
    "--trajectory-anchor-pass --freq-v0-interval"
)

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)


spec = ExperimentSpec(
    name="freq_onset_channel_mini",
    doc=(
        "Mini 2-arm A/B: anchored+interval full_macros with --freq-onset-pass (residual "
        "TRAJ_REG SETs re-tagged to FREQ_ONSET op48) vs without. 3 seeds, mini body=large, "
        ":0.2.9. Decisive gate = content_tier_report --onset on the unified V0-onset bucket "
        "(op45 V0 + op48 FREQ_ONSET + op47 nudge pitch, freq regs). The arms tokenize "
        "differently (parse-time transform), so cache keys apart correctly."
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(label="onset_chan", extra_cargs=f"{_INTERVAL_CARGS} --freq-onset-pass"),
        Arm(label="split", extra_cargs=_INTERVAL_CARGS, baseline=True),
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
)
