"""Audit the parser's IRQ filter on the corpus: records raw irq
distribution per dump, the parser's detected IRQ, and whether
`_filter_irq` would skip the file. Writes a CSV."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import pandas as pd

from preframr.args import add_args
from preframr_tokens import RegLogParser
from preframr_tokens.stfconstants import MAX_REG, VAL_PDTYPE


def key_for(path):
    p = Path(path)
    composer = p.parent.name
    base = p.name
    if not composer:
        return base
    return f"{composer}/{base}"


def read_raw(path):
    df = pd.read_parquet(path)
    df = df[df["reg"] <= MAX_REG]
    df["val"] = df["val"].astype(VAL_PDTYPE)
    chips = df["chipno"].nunique() if "chipno" in df.columns else 1
    df = df[["clock", "irq", "reg", "val"]]
    if chips > 1:
        df = df[df["clock"] < 0]
    return df, int(chips)


def detect_irq(parser, raw_df):
    df = parser._squeeze_changes(raw_df.copy())
    df = parser._combine_regs(df)
    df = parser._quantize_freq_to_cents(df)
    df = parser._simplify_ctrl(df)
    df = parser._simplify_pcm(df)
    df = parser._squeeze_changes(df)
    if df.empty:
        return 0
    irq, _ = parser._add_frame_reg(df, diffmax=2048)
    return int(irq)


def audit_dump(parser, path, min_irq, max_irq):
    try:
        raw_df, chips = read_raw(path)
    except Exception as e:
        return None, f"read_error: {type(e).__name__}: {e}"
    if raw_df.empty:
        return None, "empty"
    irq_counts = raw_df["irq"].value_counts().sort_values(ascending=False)
    raw_first_irq = int(raw_df["irq"].iloc[0])
    raw_top_irq = int(irq_counts.index[0])
    raw_top_share = float(irq_counts.iloc[0]) / float(len(raw_df))
    raw_distinct = int(raw_df["irq"].nunique())
    try:
        parser_irq = detect_irq(parser, raw_df)
    except Exception as e:
        return None, f"detect_error: {type(e).__name__}: {e}"

    raw_in_bounds = min_irq <= raw_top_irq <= max_irq
    parser_in_bounds = min_irq <= parser_irq <= max_irq if parser_irq else False
    detected_zero = parser_irq == 0
    disagree = parser_irq != raw_top_irq and parser_irq != 0
    near_lower = raw_top_irq < min_irq and raw_top_irq >= min_irq * 0.5
    near_upper = raw_top_irq > max_irq and raw_top_irq <= max_irq * 2.0

    return (
        {
            "chips": chips,
            "raw_first_irq": raw_first_irq,
            "raw_top_irq": raw_top_irq,
            "raw_top_share": round(raw_top_share, 4),
            "raw_distinct_irqs": raw_distinct,
            "parser_detected_irq": parser_irq,
            "raw_in_bounds": int(raw_in_bounds),
            "parser_in_bounds": int(parser_in_bounds),
            "detected_zero": int(detected_zero),
            "parser_disagrees_with_raw": int(disagree),
            "near_lower_bound": int(near_lower),
            "near_upper_bound": int(near_upper),
        },
        None,
    )


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--audit-glob", required=True, action="append")
    ap.add_argument("--audit-csv", required=True)
    args = ap.parse_args()

    parser = RegLogParser(args=args)

    dumps = []
    for g in args.audit_glob:
        dumps.extend(glob.glob(g))
    dumps = sorted(set(dumps))
    print(f"audit: {len(dumps)} dumps; min_irq={args.min_irq} max_irq={args.max_irq}")

    rows = []
    errors = []
    for i, dump in enumerate(dumps):
        info, err = audit_dump(parser, dump, args.min_irq, args.max_irq)
        if err is not None:
            errors.append((key_for(dump), err))
            continue
        rows.append((key_for(dump), info))
        if (i + 1) % 200 == 0:
            print(f"  audited {i + 1}/{len(dumps)} dumps; errors={len(errors)}")

    out_path = Path(args.audit_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "key",
        "chips",
        "raw_first_irq",
        "raw_top_irq",
        "raw_top_share",
        "raw_distinct_irqs",
        "parser_detected_irq",
        "raw_in_bounds",
        "parser_in_bounds",
        "detected_zero",
        "parser_disagrees_with_raw",
        "near_lower_bound",
        "near_upper_bound",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key, info in sorted(rows):
            w.writerow([key] + [info[c] for c in cols[1:]])
    err_path = out_path.with_suffix(".errors.csv")
    with open(err_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "error"])
        for key, err in sorted(errors):
            w.writerow([key, err])

    print(f"audited {len(rows)} dumps; errors {len(errors)}")
    print(f"CSV: {out_path}")
    print(f"Errors CSV: {err_path}")

    summary = {}
    for _, info in rows:
        for k in (
            "raw_in_bounds",
            "parser_in_bounds",
            "detected_zero",
            "parser_disagrees_with_raw",
            "near_lower_bound",
            "near_upper_bound",
        ):
            summary[k] = summary.get(k, 0) + info[k]
    print("\nflags:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    bin_edges = [
        ("<7500", lambda x: x < 7500),
        ("7500-9999", lambda x: 7500 <= x < 10000),
        ("10000-14999", lambda x: 10000 <= x < 15000),
        ("15000-17499", lambda x: 15000 <= x < 17500),
        ("17500-19999", lambda x: 17500 <= x < 20000),
        ("20000-22499", lambda x: 20000 <= x < 22500),
        ("22500-24999", lambda x: 22500 <= x < 25000),
        ("25000-29999", lambda x: 25000 <= x < 30000),
        (">=30000", lambda x: x >= 30000),
    ]
    print("\nraw_top_irq histogram:")
    for label, pred in bin_edges:
        n = sum(1 for _, info in rows if pred(info["raw_top_irq"]))
        if n:
            print(f"  {label}: {n}")

    print("\nparser_detected_irq histogram:")
    for label, pred in bin_edges:
        n = sum(1 for _, info in rows if pred(info["parser_detected_irq"]))
        if n:
            print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
