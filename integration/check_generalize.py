#!/usr/bin/env python3
"""Generalize integration test gate."""

import argparse
import glob
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def find_event_file(tb_logs):
    """Pick the newest ``events.out.tfevents.*`` under ``tb_logs``."""
    pattern = os.path.join(tb_logs, "**", "events.out.tfevents.*")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"no TB events under {tb_logs}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def read_scalars(event_path):
    """Return ``{tag: [(step, value), ...]}`` for every scalar tag."""
    acc = EventAccumulator(event_path, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("scalars", []):
        out[tag] = [(ev.step, ev.value) for ev in acc.Scalars(tag)]
    return out


def best_val_acc_at_min_loss(scalars):
    """Find the val_acc at the epoch that had the lowest val_loss."""
    val_loss = scalars.get("val_loss", [])
    val_acc = scalars.get("val_acc", [])
    if not val_loss or not val_acc:
        raise ValueError(
            f"missing val_loss / val_acc series " f"(found {sorted(scalars.keys())})"
        )
    best_step, best_loss = min(val_loss, key=lambda sv: sv[1])
    matching = [v for s, v in val_acc if s == best_step]
    if not matching:
        raise ValueError(f"no val_acc logged at step {best_step} (best val_loss step)")
    return best_step, best_loss, matching[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tb-logs", required=True)
    ap.add_argument("--min-val-acc", type=float, required=True)
    ap.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help=(
            "Reject runs that converged in fewer than this many "
            "validated epochs. Catches collapse-at-epoch-0 bugs."
        ),
    )
    args = ap.parse_args()

    event_path = find_event_file(args.tb_logs)
    print(f"reading {event_path}", flush=True)
    scalars = read_scalars(event_path)
    step, loss, acc = best_val_acc_at_min_loss(scalars)
    epochs_run = len(scalars.get("val_loss", []))
    print(
        f"best val_loss {loss:.4f} at step {step}, "
        f"val_acc there = {acc:.4f}, "
        f"epochs evaluated = {epochs_run}",
        flush=True,
    )

    if epochs_run < args.min_epochs:
        print(
            f"FAIL: only {epochs_run} val epochs run "
            f"(min_epochs={args.min_epochs})",
            file=sys.stderr,
        )
        return 2
    if acc < args.min_val_acc:
        print(
            f"FAIL: val_acc {acc:.4f} < min_val_acc {args.min_val_acc}",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: val_acc {acc:.4f} >= {args.min_val_acc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
