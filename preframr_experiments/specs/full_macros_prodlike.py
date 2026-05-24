"""Prodlike A/B for the full current macro set (post preframr-tokens 0.14.1 multi-frame decode fix) vs the prior registered baseline. The fix is decode-only, so training tokenization is unchanged; this measures whether the newer macros (ctrl_triple / freq_run / freq_vibrato / freq_nudge / release_update / lonely_catch_all) help compression + learnability at scale, now that they also render correctly on generation. The newer macros ride in extra_cargs because they are not yet registered pipeline-spec transform names (only legacy CLI flags); oscillate_env + loop_transposed already default on."""

from __future__ import annotations

from preframr_experiments.base import (
    Arm,
    ExperimentSpec,
    prodlike_train_args,
)

_BASE_TRANSFORMS = [
    {"name": "slope"},
    {"name": "preset"},
    {"name": "hard_restart"},
    {"name": "legato_per_cluster", "params": {"clusters": [2, 4]}},
    {"name": "voice_block_order"},
    {"name": "ctrl_bigram"},
    {"name": "loop"},
]

_NEW_MACRO_CARGS = (
    "--ctrl-triple-pass "
    "--freq-run-pass "
    "--freq-nudge-pass "
    "--release-update-pass "
    "--vibrato-env-pass "
    "--lonely-catch-all"
)


spec = ExperimentSpec(
    name="full_macros_prodlike",
    doc=(
        "Prodlike A/B: full current macro set vs prior registered "
        "baseline. The target arm adds the newer collapse/absorber "
        "macros (CTRL_TRIPLE, FREQ_RUN, FREQ_VIBRATO, FREQ_NUDGE, "
        "RELEASE_UPDATE, plus lonely_catch_all to drive the strict-no-"
        "diff residual toward zero) on top of the shared registered "
        "pipeline (slope, preset, hard_restart, legato c2/c4, "
        "voice_block_order, ctrl_bigram, loop+transposed, oscillate_env). "
        "preframr-tokens 0.14.1 fixed the multi-frame decode off-by-one "
        "in those macros; the fix is decode-only so the train alphabet "
        "is unchanged vs a pre-fix tokenize, but generated output now "
        "renders correctly. Measures (1) compression: "
        "encoded_tokens_per_song should DROP vs baseline; (2) cost: "
        "alphabet_size grows with the added ops; (3) learnability: "
        "eval_a val_acc must not regress > 1 sigma vs baseline, and "
        ">= 5 of 8 eval_b_* families must hold non-negative content "
        "lift. Target arm first per convention; baseline last. Single "
        "seed, canonical body, prodlike train args."
    ),
    tier="prodlike",
    arms=[
        Arm(
            label="full_macros",
            extra_cargs=_NEW_MACRO_CARGS,
        ),
        Arm(label="baseline", baseline=True),
    ],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "val_loss_macro_best",
        "val_acc_macro_at_best_loss",
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
    seeds=1,
    seq_len=8192,
    tkvocab=32768,
    max_perm=1,
    train_args=prodlike_train_args(),
    pipeline_spec={"transforms": list(_BASE_TRANSFORMS)},
)
