"""Memorize-back smoke spec."""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, event_decode_gate

_TRAIN_ARGS = (
    "--model=llama3_2 "
    "--shuffle 32 --accumulate-grad-batches 2 --batch-size 16 "
    "--learning-rate 5e-4 --weight-decay 0.01 "
    "--layers 10 --heads 8 --kv-heads 4 --embed 384 --intermediate 1024 "
    "--attn-dropout 0.0 --max-epochs 500 "
    "--stop-loss 0.001 --stop-delta 0.0001"
)


spec = ExperimentSpec(
    name="memorize",
    doc=(
        "Memorize-back smoke test. Train on 7 fixed SIDs covering every "
        "macro class until train_loss < 0.001, then verify predict "
        "reproduces block 0 with greedy accuracy >= 0.2. Build-gate."
    ),
    tier="smoke",
    arms=[Arm(label="default", baseline=True)],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
        "train_loss_final",
        "wallclock_train_min",
    ],
    seeds=1,
    seq_len=1024,
    block_stride=256,
    train_args=_TRAIN_ARGS,
    predict_gate=event_decode_gate,
)
