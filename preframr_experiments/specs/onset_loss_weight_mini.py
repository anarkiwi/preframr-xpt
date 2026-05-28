"""Mini 2-arm A/B for onset loss prioritization (preframr 0.2.8 --onset-loss-weight): does
up-weighting the rare FREQ V0-onset in the CE loss force the model to learn it? Converged
diagnosis (design/onset_loss_prioritization.md + trajectory_anchoring.md): across four mini
conditions the model's V0-onset acc is 0.000 despite the onset line being trigram-0.82
predictable and the model having ample mini capacity (freq_core lifted SET to 0.42) -- the
onset is rare and the model won't spend capacity on it.

Both arms run the best onset encoding (full_macros + anchoring + interval-coded V0); the ONLY
difference is --onset-loss-weight (train-only, so the arms share tokenization -- it is in the
runner's _TRAIN_ONLY_CARG_FLAGS denylist). Target `onset_w10` boosts onset CE 10x; baseline
`onset_w1` is W=1 (off). 3 seeds, mini body=large, 60 epochs, :0.2.8.

Decisive gate: `content_tier_report --onset` -- does V0-onset acc move off 0.000, AND does
all-tier val_acc not collapse (over-focusing onset would regress structural/content)? A clear
move at mini -> sweep W + fold into a prodlike interval+onset-weight run. Flat at W=10 -> try
higher / the onset needs scale. W previously refuted as generic token-weighting, but never
onset-targeted on the anchored+interval encoding."""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.8"

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
    name="onset_loss_weight_mini",
    doc=(
        "Mini 2-arm A/B: anchored+interval full_macros with FREQ V0-onset CE up-weighted "
        "10x (--onset-loss-weight 10) vs W=1. Identical tokenization (train-only knob). "
        "3 seeds, mini body=large, 60 epochs, :0.2.8. Decisive gate = content_tier_report "
        "--onset: does V0-onset acc move off 0.000 without all-tier collapsing?"
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(label="onset_w10", extra_cargs=f"{_INTERVAL_CARGS} --onset-loss-weight 10"),
        Arm(label="onset_w1", extra_cargs=_INTERVAL_CARGS, baseline=True),
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
