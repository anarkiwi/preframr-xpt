"""Learnability audit, part 1/2: production ``full_macros`` vs the atomic
baseline on the BYTE-EXACT rebake (preframr-tokens 0.40.0), Unigram OFF.

Re-confirms the 2026-05-27 full_macros content win (eval_a content 0.324 vs
0.219, x3 seeds) on CORRECTED tokens -- the prior result was on partly-wrong
pre-0.40.0 tokens. The open question is learnability, not correctness: does the
production compressing vocab's PAYLOAD actually learn. Both arms train directly
on the base-atom alphabet (``--tkvocab 0``) so the per-op signal is not obscured
by Unigram merges and the constrained-decode mask stays aligned to the atomic
op-grammar.

Arms (pipeline_spec=None so per-arm CLI flags fully drive the passes; the
``apply_pipeline_spec_to_args`` override path only fires when a pipeline_spec is
present):

- full_macros: every flag in ``tokenizer_config.REGISTERED_MACROS`` ON. Emits
  the production compressing tokens (DIFF delta + BACK_REF distance via
  freq_trajectory + loop, plus the ctrl/freq/release absorbers).
- baseline: no macro passes -- the raw atomic op stream. ``baseline=True``.

Decisive reads (preframr 0.2.15, gate ON): per-tier ``gate/content_over_structural``
+ the per-op ``gate/op_acc/{op}`` (which compressing token learns: DIFF vs
BACK_REF), and the post-hoc content-tier read (audit_checkpoint_per_class ->
content_tier_report) for full_macros vs baseline content_acc. NO PW+filter
ablation -- this is the full byte-exact corpus, not the melody substrate line.

Part 2/2 is ``learnability_codebook_mini`` (the shipped skeleton+codebook stack).
"""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.15"

# tokenizer_config.REGISTERED_MACROS, as CLI flags (underscore -> hyphen).
_REGISTERED_MACRO_FLAGS = (
    "--freq-trajectory-pass "
    "--preset-pass "
    "--hard-restart-pass "
    "--legato-pass-c2 "
    "--legato-pass-c4 "
    "--voice-canonical-block-order "
    "--ctrl-bigram-pass "
    "--loop-pass "
    "--loop-transposed "
    "--freq-nudge-pass "
    "--release-update-pass "
    "--ctrl-triple-pass "
    "--lonely-catch-all"
)

# Gate ON for the learnability readout (content_over_structural + per-op acc).
_TRAIN_ARGS = (
    mini_train_args(body="large").replace("--max-epochs 160", "--max-epochs 60")
    + " --generalization-gate"
)


spec = ExperimentSpec(
    name="learnability_full_macros_mini",
    doc=(
        "Learnability audit 1/2: production full_macros (REGISTERED_MACROS) vs "
        "atomic baseline on the byte-exact (tokens 0.40.0) rebake, --tkvocab 0 "
        "(Unigram OFF), --generalization-gate. Re-confirms the 05-27 content win "
        "on corrected tokens and reads per-op gate/op_acc (DIFF vs BACK_REF "
        "learnability). 3 seeds, mini body=large, 60 epochs, no PW+filter "
        "ablation."
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(label="full_macros", extra_cargs=_REGISTERED_MACRO_FLAGS),
        Arm(label="baseline", baseline=True),
    ],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=3,
    seq_len=4096,
    tkvocab=0,
    max_perm=1,
    train_args=_TRAIN_ARGS,
    pipeline_spec=None,
)
