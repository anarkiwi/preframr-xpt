"""Prodlike full-stack melody arbiter -- the deferred decisive test. Five mini A/Bs have
exhausted cheap diligence; mini provably can't learn melody (V0-onset = 0.000 across
full_macros / anchored / interval / freq_core / onset_loss_weight / freq_onset_channel).
Only scale moves it (prodlike absolute op45 = 0.067). This A/B tests the FULL landed
encoding/loss stack (anchor + interval V0 + FREQ_ONSET channel + onset-loss-weight) vs
the plain-full_macros deployment baseline, at the scale where melody has signal.

Dual-purpose A/B (the same data answers two questions):
1. **Melody:** does the unified V0-onset acc (op45 V0 + op48 FREQ_ONSET + op47 NUDGE pitch)
   rise above the absolute baseline's 0.067 -> the stack works, sweep W + augmentation
   re-opens. Flat -> the encoding axis is exhausted; pivot to a distributional/perceptual
   melody metric (music_llm_landscape_and_fail_fast_plan.md territory).
2. **Content (SET) scale-confirmation:** freq_onset_channel_mini gave a massive SET-cleanup
   lift (content-tier 0.076->0.249, op0 SET 0.154->0.831). Prodlike confirms whether this
   holds at scale -- a deployment win regardless of the melody read.

Deployment config (matches the STAGE 2 full_macros_prodlike re-run): tkvocab 8192,
batch-size 4 / accumulate-grad-batches 8 (B=8 OOMs at prodlike), 3 seeds, seq_len 8192,
on anarkiwi/preframr:0.2.9. Read via `audit.content_tier_report --onset` (unified
`melodic_onset_bucket`). Seed-major runner -> 1-seed cross-arm signal ~6-11h in, not 30h.
"""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, prodlike_train_args

_IMAGE = "anarkiwi/preframr:0.2.9"

# base pipeline + the four absorbers == REGISTERED_MACROS (the full_macros preset).
_STACK_MACROS = ("trajectory_anchor_pass", "freq_v0_interval", "freq_onset_pass")

_TRAIN_ARGS = (
    prodlike_train_args()
    .replace("--accumulate-grad-batches 16", "--accumulate-grad-batches 8")
    .replace("--batch-size 2", "--batch-size 4")
)


spec = ExperimentSpec(
    name="melody_stack_prodlike",
    doc=(
        "Prodlike A/B: full_macros + (anchor + interval V0 + FREQ_ONSET channel + "
        "--onset-loss-weight 10) vs plain full_macros. 3 seeds, deployment config "
        "(tkvocab 8192, B=4/accum=8), :0.2.9. Decisive gate = content_tier_report "
        "--onset (unified V0-onset bucket: op45 V0 + op48 FREQ_ONSET + op47 NUDGE pitch); "
        "secondary = content-tier acc (SET-cleanup scale confirmation)."
    ),
    tier="prodlike",
    image=_IMAGE,
    arms=[
        Arm(
            label="full_stack",
            macro_config="full_macros",
            macro_flags=_STACK_MACROS,
            extra_cargs="--onset-loss-weight 10",
        ),
        Arm(label="full_macros", macro_config="full_macros", baseline=True),
    ],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "val_loss_eval_a_best",
        "val_acc_eval_a_at_best_loss",
        "val_loss_eval_b_daglish_best",
        "val_acc_eval_b_daglish_at_best_loss",
        "val_loss_eval_b_follin_best",
        "val_acc_eval_b_follin_at_best_loss",
        "val_loss_eval_b_crisps_best",
        "val_acc_eval_b_crisps_at_best_loss",
        "val_loss_eval_b_mibri_best",
        "val_acc_eval_b_mibri_at_best_loss",
        "val_loss_eval_b_marquis_best",
        "val_acc_eval_b_marquis_at_best_loss",
        "val_loss_eval_b_dobek_best",
        "val_acc_eval_b_dobek_at_best_loss",
        "val_loss_eval_b_winterberg_best",
        "val_acc_eval_b_winterberg_at_best_loss",
        "val_loss_eval_b_wilson_best",
        "val_acc_eval_b_wilson_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=3,
    seq_len=8192,
    tkvocab=8192,
    max_perm=1,
    train_args=_TRAIN_ARGS,
)
