"""Enumerate digi / out-of-bounds dumps in the corpus by re-running the
parser's built-in skip filters. Writes a CSV (`composer/basename,reason`)
that the parser can ingest via `--exclude-list` to short-circuit per-file
filter re-runs. Captures the canonical 'skipped <path>, ...' log lines
from RegLogParser so reasons stay in lockstep with the production code."""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import re
from pathlib import Path

from preframr.args import add_args
from preframr_tokens import RegLogParser

_SKIPPED_PREFIX = "skipped "


class SkipRecorder(logging.Handler):
    """Capture every `skipped <path>, <reason>` line emitted by the parser."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        msg = record.getMessage()
        if not msg.startswith(_SKIPPED_PREFIX):
            return
        body = msg[len(_SKIPPED_PREFIX) :]
        path, _, reason = body.partition(",")
        self.records.append((path.strip(), reason.strip()))


_DIGI_VOL = re.compile(r"digi-like vol density \(max (\d+) writes per frame\)")
_CTRL_BURST = re.compile(r"too many \((\d+)\) control reg changes per frame")
_IRQ_OUT = re.compile(r"irq (\d+) \(outside IRQ range\)")
_TOO_SHORT = re.compile(r"length (\d+) \(< (\d+)\)")


def categorise(reason):
    if (m := _DIGI_VOL.search(reason)) is not None:
        return "digi_vol_density", m.group(1)
    if (m := _CTRL_BURST.search(reason)) is not None:
        return "ctrl_reg_burst", m.group(1)
    if (m := _IRQ_OUT.search(reason)) is not None:
        return "irq_out_of_range", m.group(1)
    if (m := _TOO_SHORT.search(reason)) is not None:
        return "too_short", f"{m.group(1)}<{m.group(2)}"
    if reason.startswith("no frames"):
        return "no_frames", ""
    if reason.startswith("no irq"):
        return "no_irq", ""
    return "other", reason


def key_for(path, _root=None):
    p = Path(path)
    composer = p.parent.name
    base = p.name
    if not composer:
        return base
    return f"{composer}/{base}"


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--audit-glob", required=True, action="append")
    ap.add_argument("--audit-csv", required=True)
    args = ap.parse_args()

    recorder = SkipRecorder()
    audit_logger = logging.getLogger("digi_audit_parser")
    audit_logger.handlers.clear()
    audit_logger.addHandler(recorder)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False

    parser = RegLogParser(args=args, logger=audit_logger)

    dumps = []
    for g in args.audit_glob:
        dumps.extend(glob.glob(g))
    dumps = sorted(set(dumps))
    print(f"audit: {len(dumps)} dumps from globs={args.audit_glob}")

    yielded = 0
    for i, dump in enumerate(dumps):
        try:
            for _ in parser.parse(dump, max_perm=1, require_pq=False, reparse=True):
                yielded += 1
                break
        except Exception as e:
            recorder.records.append(
                (dump, f"parser_exception: {type(e).__name__}: {e}")
            )
        if (i + 1) % 200 == 0:
            print(
                f"  audited {i + 1}/{len(dumps)} dumps; {len(recorder.records)} skips"
            )

    seen = set()
    rows = []
    for path, reason in recorder.records:
        key = key_for(path)
        if key in seen:
            continue
        seen.add(key)
        cat, detail = categorise(reason)
        rows.append((key, cat, detail, reason))

    out_path = Path(args.audit_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "category", "detail", "raw_reason"])
        for row in sorted(rows):
            w.writerow(row)

    print(f"audited {len(dumps)} dumps; skipped {len(rows)} unique paths")
    print(f"yielded (parsed cleanly): {yielded}")
    print(f"CSV: {out_path}")
    counts = {}
    for _, cat, _, _ in rows:
        counts[cat] = counts.get(cat, 0) + 1
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
