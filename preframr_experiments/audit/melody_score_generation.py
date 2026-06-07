"""Score a SID dump (real or model-generated) against the prodlike training
baseline. Emits per-feature z-score, a robust-distance summary, and PASS/WARN/
FAIL verdicts. Intended as the auto-audition replacement.

Verdict policy (each feature):
- |z| <= 1.5 -> PASS
- 1.5 < |z| <= 3.0 -> WARN
- |z| > 3.0 -> FAIL

Headline score (0..1, higher = more in-distribution):
- fraction of features with |z| <= 1.5

Overall verdict:
- COLLAPSE: pitched_gates_per_sec < 1.0 (model is genuinely silent), OR
            3+ features each with z < -3 (multi-axis distribution collapse)
- FAIL: any single feature with |z| > 4 (extreme outlier on a single axis)
- WARN: more WARN than PASS features
- PASS: otherwise

The single-threshold collapse guards (hardcoded "median_note_frames < 1.0
= clicks only") are removed because they false-positive on legitimate fast-
note SID styles (Hubbard, etc.). The z-distance + multi-axis check catches
genuine degenerate output without flagging real music.

Usage:
  score_generation.py <dump.parquet> [<dump.parquet> ...] \\
    [--baseline /scratch/tmp/preframr_substrate/baseline] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from preframr_experiments.audit.melody_features import FEATURES, analyze

_SILENCE_THRESHOLD = 1.0  # pitched_gates_per_sec below this = silent
_EXTREME_Z = 4.0
_MULTI_AXIS_NEG_Z = -3.0
_MULTI_AXIS_MIN_COUNT = 3


def _z(value: float | None, mean: float, std: float) -> float | None:
    if value is None:
        return None
    return (value - mean) / max(std, 1e-6)


def _verdict_for_z(z: float | None) -> str:
    if z is None:
        return "MISSING"
    a = abs(z)
    if a <= 1.5:
        return "PASS"
    if a <= 3.0:
        return "WARN"
    return "FAIL"


def collapse_failures(features: dict, z_by_feature: dict[str, float]) -> list[str]:
    fails = []
    pg = features.get("pitched_gates_per_sec")
    if pg is None or pg < _SILENCE_THRESHOLD:
        fails.append(f"pitched_gates_per_sec={pg} < {_SILENCE_THRESHOLD} (silent)")
    severely_low = [
        (f, z)
        for f, z in z_by_feature.items()
        if z is not None and z < _MULTI_AXIS_NEG_Z
    ]
    if len(severely_low) >= _MULTI_AXIS_MIN_COUNT:
        fails.append(
            f"multi-axis collapse: {len(severely_low)} features with z < "
            f"{_MULTI_AXIS_NEG_Z}: "
            + ", ".join(f"{f}({z:+.1f})" for f, z in severely_low)
        )
    return fails


def score_one(path: str, scaler: dict, ref_summary: dict) -> dict:
    features = analyze(path)
    rows = []
    z_by_feature: dict[str, float] = {}
    n_pass = n_warn = n_fail = n_missing = 0
    extreme_outlier_features = []
    for c in FEATURES:
        if c not in scaler:
            continue
        v = features.get(c)
        mean, std = scaler[c]["mean"], scaler[c]["std"]
        z = _z(v, mean, std)
        z_by_feature[c] = z
        verdict = _verdict_for_z(z)
        if verdict == "PASS":
            n_pass += 1
        elif verdict == "WARN":
            n_warn += 1
        elif verdict == "FAIL":
            n_fail += 1
        else:
            n_missing += 1
        if z is not None and abs(z) > _EXTREME_Z:
            extreme_outlier_features.append((c, z))
        rows.append(
            {
                "feature": c,
                "value": v,
                "ref_mean": mean,
                "ref_std": std,
                "z": z,
                "verdict": verdict,
            }
        )
    collapse = collapse_failures(features, z_by_feature)
    n_scored = max(1, n_pass + n_warn + n_fail)
    headline = n_pass / n_scored
    if collapse:
        verdict = "COLLAPSE"
    elif extreme_outlier_features:
        verdict = "FAIL"
    elif n_warn > n_pass:
        verdict = "WARN"
    else:
        verdict = "PASS"
    return {
        "path": path,
        "duration_s": features.get("_duration_s"),
        "n_notes": features.get("_n_notes"),
        "rows": rows,
        "headline": headline,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "n_missing": n_missing,
        "collapse_reasons": collapse,
        "verdict": verdict,
    }


def format_report(report: dict) -> str:
    lines = [
        f"=== {report['path']} ===",
        f"  duration: {report['duration_s']:.1f}s  notes: {int(report['n_notes'])}  "
        f"verdict: {report['verdict']}  "
        f"headline: {report['headline']:.2f}  "
        f"(PASS {report['n_pass']} / WARN {report['n_warn']} / "
        f"FAIL {report['n_fail']} / MISSING {report['n_missing']})",
    ]
    if report["collapse_reasons"]:
        lines.append("  COLLAPSE GUARDS:")
        for c in report["collapse_reasons"]:
            lines.append(f"    - {c}")
    lines.append(
        f"  {'feature':30s} {'value':>10s} {'ref mean':>10s} {'ref std':>9s} "
        f"{'z':>7s}  verdict"
    )
    for r in report["rows"]:
        val = "n/a" if r["value"] is None else f"{r['value']:10.3f}"
        z = "n/a" if r["z"] is None else f"{r['z']:+7.2f}"
        lines.append(
            f"  {r['feature']:30s} {val} {r['ref_mean']:10.3f} {r['ref_std']:9.3f} "
            f"{z}  {r['verdict']}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dumps", nargs="+", help="dump.parquet path(s) to score")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path("/scratch/tmp/preframr_substrate/baseline"),
        help="dir holding baseline_scaler.json + baseline_summary.csv",
    )
    ap.add_argument("--json", action="store_true")
    cli = ap.parse_args()
    scaler = json.loads((cli.baseline / "baseline_scaler.json").read_text())
    ref_summary = {}
    reports = [score_one(d, scaler, ref_summary) for d in cli.dumps]
    if cli.json:
        print(json.dumps(reports, indent=2, default=float))
        return 0
    for r in reports:
        print(format_report(r))
        print()
    summary = {
        "PASS": sum(1 for r in reports if r["verdict"] == "PASS"),
        "WARN": sum(1 for r in reports if r["verdict"] == "WARN"),
        "FAIL": sum(1 for r in reports if r["verdict"] == "FAIL"),
        "COLLAPSE": sum(1 for r in reports if r["verdict"] == "COLLAPSE"),
    }
    print("=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
