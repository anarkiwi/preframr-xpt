"""Re-baseline of freq_onset_channel_mini on :0.2.11 (preframr-tokens 0.29.0) which fixes
``iter_self_contained_row_blocks`` to actually re-run the freq passes after
``expand_to_literal_form``. Every prior mini A/B trained on a stream with **zero**
op=45/47/48 atoms; with the fix the encoded blocks finally contain them. Same arms +
seeds + training args as the original; the diff is just the image.

Decisive read: ``content_tier_report --onset`` on the unified V0-onset bucket. If
V0-onset acc finally moves off 0.000 on this re-run, every flat mini result in the
melody arc was an artifact of the encoder bug, and the structural levers worked all
along -- we just couldn't see them. If V0-onset stays 0.000 even with the atoms
present, the scale-bound diagnosis stands and we pivot to the distributional metric."""

from __future__ import annotations

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

_INTERVAL_CARGS = (
    "--ctrl-triple-pass --freq-nudge-pass --release-update-pass --lonely-catch-all "
    "--trajectory-anchor-pass --freq-v0-interval"
)

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)


spec = ExperimentSpec(
    name="freq_passes_fix_rebaseline_mini",
    doc=(
        "Re-baseline of freq_onset_channel_mini on :0.2.11 (the freq passes now actually "
        "re-fire inside iter_self_contained_row_blocks). 2 arms x 3 seeds, mini body=large. "
        "Decisive gate = content_tier_report --onset; first chance to measure V0-onset acc "
        "on a stream that actually contains op=45/47/48."
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
