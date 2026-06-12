"""Generalize harness at prodlike body + unigram (BPE-dictionary) tokenization.

Same corpus + holdouts as ``generalize`` (canonical tier; identical
eval_a/eval_b_* dirs), so results are directly comparable to the tkvocab=0
atoms-only baseline. Only the model body (prodlike ~107M) and the tokenization
(unigram, tkvocab>0) differ. Needs RUST_MIN_STACK (set in base._docker_run) so
the UnigramTrainer survives long SID sentences.
"""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec
from preframr_experiments.specs.generalize import _generalize_gate

_TRAIN_ARGS = (
    "--model=llama3_2 "
    "--shuffle 0.4 --accumulate-grad-batches 8 --batch-size 4 "
    "--learning-rate 1e-4 --weight-decay 0.01 "
    "--layers 16 --heads 12 --kv-heads 4 --embed 768 --intermediate 2048 "
    "--attn-dropout 0.1 --max-epochs 200 "
    "--early-stop-patience 5 --early-stop-min-delta 0.01 "
    "--val-check-every 1"
)

spec = ExperimentSpec(
    name="generalize_prodlike_unigram",
    doc=(
        "Prodlike body (~107M, 16L/d768) + unigram tkvocab=2048 on the "
        "canonical generalize corpus/holdouts. Directly comparable to the "
        "tkvocab=0 atoms-only baseline; isolates body + tokenization."
    ),
    tier="canonical",
    arms=[Arm(label="default", baseline=True)],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=1,
    seq_len=8192,
    tkvocab=2048,
    train_args=_TRAIN_ARGS,
    predict_gate=_generalize_gate,
)
