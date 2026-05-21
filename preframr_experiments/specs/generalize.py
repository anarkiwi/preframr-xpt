"""Generalize harness spec."""

from __future__ import annotations

import os

from preframr_experiments.base import Arm, ArmArtefacts, ExperimentSpec
from preframr_experiments.metrics import _val_acc_at_best_loss

_TRAIN_ARGS = (
    "--model=llama3_2 "
    "--shuffle 0.4 --accumulate-grad-batches 8 --batch-size 4 "
    "--learning-rate 1e-4 --weight-decay 0.01 "
    "--layers 8 --heads 8 --kv-heads 4 --embed 320 --intermediate 896 "
    "--attn-dropout 0.1 --max-epochs 200 "
    "--early-stop-patience 5 --early-stop-min-delta 0.01 "
    "--val-check-every 1"
)


def _generalize_gate(art: ArmArtefacts):
    """Pass = val_acc at best val_loss >= GENERALIZE_MIN_VAL_ACC.
    Default 0.0 (report-only); operators set the env var to enforce
    a floor once a baseline is established (TODO §2)."""
    floor = float(os.environ.get("GENERALIZE_MIN_VAL_ACC", "0"))
    val_acc = _val_acc_at_best_loss(art)
    if val_acc != val_acc:
        return False, "no val_acc series in TB events"
    if val_acc < floor:
        return False, f"val_acc {val_acc:.4f} < min {floor:.4f}"
    return True, f"val_acc {val_acc:.4f} >= min {floor:.4f}"


spec = ExperimentSpec(
    name="generalize",
    doc=(
        "Held-out generalisation gate over the canonical 5-composer "
        "train set + Eval-A in-distribution holdouts. Calibration "
        "mode by default; set GENERALIZE_MIN_VAL_ACC env to enforce "
        "a floor."
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
    tkvocab=131072,
    train_args=_TRAIN_ARGS,
    predict_gate=_generalize_gate,
)
