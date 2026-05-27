"""Prodlike A/B for trajectory anchoring -- the only regime that can test the melodic
hypothesis. The mini A/B (`trajectory_anchor_mini`, 2026-05-27) was a seed-stable content
win (content-tier 0.036->0.080, val_acc 0.113->0.137) but SET-carried (op0 0.063->0.175);
FREQ_TRAJ op45 stayed at the floor (~0.002) AND op45 was ~0 in both arms, so mini cannot
test melody. Prodlike baseline op45 was 0.067 (~30-60x higher), so this is where the claim
is decidable: does the content win hold AND does op45 rise where it has signal?

Both arms run the confirmed full_macros representation on the shared base pipeline; the ONLY
arm difference is `--trajectory-anchor-pass` (opt-in flag, toggled per-arm via extra_cargs
like the absorber macros, so apply_pipeline_spec_to_args does not clobber it; the flag is
dataset-affecting so the cache keys the arms apart -- no hook / no cache-disable needed).

Deployment config (matches the STAGE 2 full_macros_prodlike re-run that confirmed the win):
tkvocab 8192, batch-size 4 / accumulate-grad-batches 8 (effective batch 32; B=8 OOMs at
prodlike), 3 seeds, seq_len 8192, on anarkiwi/preframr:0.2.6.

Decisive read: per arm-seed run audit_checkpoint_per_class (whole eval set, cuda) ->
audit_per_class.json, then `python3 -m preframr_experiments.audit.content_tier_report
--results-root <dir>`: content-tier acc Δ + the by-op breakdown (op45 first-class). all-tier
val_acc is CONFOUNDED (the arms tokenize FREQ_TRAJ differently)."""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, prodlike_train_args

_IMAGE = "anarkiwi/preframr:0.2.6"

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

_ANCHOR_CARGS = f"{_FULL_MACRO_CARGS} --trajectory-anchor-pass"

_TRAIN_ARGS = (
    prodlike_train_args()
    .replace("--accumulate-grad-batches 16", "--accumulate-grad-batches 8")
    .replace("--batch-size 2", "--batch-size 4")
)


spec = ExperimentSpec(
    name="trajectory_anchor_prodlike",
    doc=(
        "Prodlike A/B: full_macros + trajectory anchoring vs full_macros. The only "
        "regime that tests melody (prodlike baseline op45 ~0.067 vs mini ~0). 3 seeds, "
        "deployment config (tkvocab 8192, batch 4 / accum 8), :0.2.6. Decisive gate = "
        "content_tier_report by-op (op45 rise + content lift); all-tier CONFOUNDED."
    ),
    tier="prodlike",
    image=_IMAGE,
    arms=[
        Arm(label="anchored", extra_cargs=_ANCHOR_CARGS),
        Arm(label="unanchored", extra_cargs=_FULL_MACRO_CARGS, baseline=True),
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
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
)
