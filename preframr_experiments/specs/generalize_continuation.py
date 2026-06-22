"""Continue-from-prompt generalisation spec (atoms-only, new decompiler codec).

SCAFFOLD pending the codec port (LIVE ARC in AGENTS.md). Three knobs must be
finalised once preframr-tokens ships the decompiler codec: the image/tokens
floor, CONTINUATION_TKVOCAB (= the codec's atom count), and the decisive gate
threshold from the first baseline. Runs unchanged otherwise.
"""

from __future__ import annotations

import json
import os

from preframr_experiments.base import Arm, ArmArtefacts, ExperimentSpec
from preframr_experiments.metrics import _val_acc_at_best_loss

_GAP_JSON = "free_running_gap.json"
_GAP_FAIL = {"exposure_bias", "short_context_or_bug"}

_TRAIN_ARGS = (
    "--model=llama3_2 "
    "--shuffle 0.4 --accumulate-grad-batches 8 --batch-size 4 "
    "--learning-rate 1e-4 --weight-decay 0.01 "
    "--layers 8 --heads 8 --kv-heads 4 --embed 320 --intermediate 896 "
    "--attn-dropout 0.1 --max-epochs 200 "
    "--early-stop-patience 5 --early-stop-min-delta 0.01 "
    "--val-check-every 1"
)


def _continuation_verdict(art: ArmArtefacts):
    """Read the free-running-gap audit JSON if present in the arm work_dir.
    Returns (verdict, reason) or (None, None) when the audit has not run."""
    path = art.work_dir / _GAP_JSON
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())
    return data.get("verdict"), data.get("verdict_reason")


def _continuation_gate(art: ArmArtefacts):
    """Pass = val_acc >= CONTINUATION_MIN_VAL_ACC and (when the gap audit is
    present and CONTINUATION_ENFORCE_GAP=1) its verdict is not pathological.
    Report-only by default, matching the generalize baseline."""
    floor = float(os.environ.get("CONTINUATION_MIN_VAL_ACC", "0"))
    val_acc = _val_acc_at_best_loss(art)
    if val_acc != val_acc:
        return False, "no val_acc series in TB events"
    verdict, reason = _continuation_verdict(art)
    tag = f"val_acc {val_acc:.4f}" + (f"; gap={verdict} ({reason})" if verdict else "")
    if val_acc < floor:
        return False, f"{tag} < min {floor:.4f}"
    enforce = os.environ.get("CONTINUATION_ENFORCE_GAP", "0") == "1"
    if enforce and verdict in _GAP_FAIL:
        return False, f"{tag}: free-running gap pathological ({verdict})"
    return True, tag


spec = ExperimentSpec(
    name="generalize_continuation",
    doc=(
        "Continue-from-prompt generalisation over the cross-composer canonical "
        "set, atoms-only on the decompiler codec. Primary signal val_acc; the "
        "decisive gate is the free-running-gap + copy/novel continuation audit "
        "(set CONTINUATION_MIN_VAL_ACC / CONTINUATION_ENFORCE_GAP to enforce)."
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
    train_args=_TRAIN_ARGS,
    predict_gate=_continuation_gate,
)
