"""Mini 2-arm A/B for melody-merge-split (preframr 0.2.10 + preframr-tokens 0.28.0):
does forcing Unigram NOT to merge melody-pitch with non-melody atoms finally surface pitch
as a learnable target at mini? freq_onset_channel_mini showed the FREQ_ONSET re-tag let
Unigram absorb most freq-pitch into merged compounds (merged share 33%->54%) -- the model
nails the compound shapes (content +0.173) but never has to predict pitch as a separable
class, so V0-onset stayed 0. The bet (per design/melody_learnability.md): if mini has
capacity (proven: SET 0.83 in onset_chan) and pitch is predictable (trigram 0.82), the
remaining cause for V0-onset=0 is that pitch is hidden inside compound vocab classes.
De-merging directly tests that.

Both arms run the best landed stack (anchored + interval + freq-onset-pass +
--onset-loss-weight 10); the ONLY arm difference is --melody-merge-split. The arms tokenize
differently (the split changes the stream), so the dataset cache keys them apart correctly.
3 seeds, mini body=large, 60 epochs, :0.2.10.

Decisive gate: `content_tier_report --onset` on the unified `melodic_onset_bucket` (op45 V0
+ op48 FREQ_ONSET + op47 NUDGE pitch on freq regs 0/7/14). Does V0-onset acc move off
0.000 at mini? If yes -> the merge-hiding diagnosis is correct, sweep at prodlike. If no
-> the structural-blocker story is wrong; pivot the melody success metric (distributional/
perceptual; music_llm_landscape_and_fail_fast_plan.md territory)."""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.10"

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

_STACK_MACROS = _BASE_MACROS + (
    "ctrl_triple_pass",
    "freq_nudge_pass",
    "release_update_pass",
    "lonely_catch_all",
    "trajectory_anchor_pass",
    "freq_v0_interval",
    "freq_onset_pass",
)

_STACK_CARGS = "--onset-loss-weight 10"

_TRAIN_ARGS = mini_train_args(body="large").replace(
    "--max-epochs 160", "--max-epochs 60"
)


spec = ExperimentSpec(
    name="melody_merge_split_mini",
    doc=(
        "Mini 2-arm A/B: the best stack (anchor + interval + freq-onset-pass + "
        "--onset-loss-weight 10) + `--melody-merge-split` vs the same stack without. "
        "3 seeds, mini body=large, 60 epochs, :0.2.10. Decisive gate = "
        "content_tier_report --onset (unified V0-onset bucket). Does forcing pitch out "
        "of compound Unigram tokens move V0-onset off 0.000 at mini?"
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(
            label="merge_split",
            extra_cargs=f"{_STACK_CARGS} --melody-merge-split",
            macro_flags=_STACK_MACROS,
        ),
        Arm(
            label="merged",
            extra_cargs=_STACK_CARGS,
            macro_flags=_STACK_MACROS,
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
)
