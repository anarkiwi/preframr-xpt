"""Translate a (prompt-seq-len, max-seq-len) token budget into seconds
of rendered SID audio at sparse / median / dense corpus density.
Reports the song-duration distribution so the deployed budget can be
matched against how much of the corpus a single context can cover."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PAL_CLOCK_HZ = 985248.0

DEFAULT_MANIFEST = "/scratch/preframr/corpus_index_full/manifest.parquet"
DEFAULT_ATOMS_PER_TOKEN = 1.502
DEFAULT_MACRO_COMPRESSION = 0.512
DEFAULT_PROMPT = 2048
DEFAULT_MAX = 8192
PCT_KEYS = (10, 25, 50, 75, 90, 95, 99)


def _fmt_sec(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}m{s % 60:02d}s"


def _atoms_per_token_from_log(log_path: Path) -> float | None:
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    m = re.search(r"atoms/token[^0-9]*([0-9]+\.[0-9]+)", text)
    if m:
        return float(m.group(1))
    counts = re.search(
        r"(\d[\d,]*)\s+atoms[^0-9]+(\d[\d,]*)\s+tokens", text, flags=re.I
    )
    if counts:
        a = int(counts.group(1).replace(",", ""))
        t = int(counts.group(2).replace(",", ""))
        if t > 0:
            return a / t
    return None


def _audit_macro_compression(audit_json: Path, manifest: pd.DataFrame) -> float | None:
    try:
        rows = json.loads(audit_json.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    total_post_macro = sum(int(r["count"]) for r in rows)
    if total_post_macro <= 0:
        return None
    adm = manifest[~manifest["skipped_by_filter"]]
    median_atoms_per_sid_raw = adm["n_atoms_raw"].median()
    n_audit_sids = sum(1 for r in rows if r["op_name"] == "SET") and 190
    if not n_audit_sids:
        return None
    avg_post_per_sid = total_post_macro / n_audit_sids
    return avg_post_per_sid / median_atoms_per_sid_raw


def compute_coverage(
    manifest: pd.DataFrame,
    atoms_per_token: float,
    macro_compression: float,
    prompt: int,
    max_seq: int,
) -> dict:
    adm = manifest[~manifest["skipped_by_filter"]].copy()
    adm = adm[adm["n_frames"] > 0]
    adm["seconds"] = adm["n_frames"] / (PAL_CLOCK_HZ / adm["irq"])
    adm["atoms_per_frame_raw"] = adm["n_atoms_raw"] / adm["n_frames"]

    duration_pcts = {
        f"p{pct:02d}_seconds": float(adm["seconds"].quantile(pct / 100))
        for pct in PCT_KEYS
    }
    frame_pcts = {
        f"p{pct:02d}_frames": float(adm["n_frames"].quantile(pct / 100))
        for pct in PCT_KEYS
    }
    apf_pcts = {
        f"p{pct:02d}_atoms_per_frame_raw": float(
            adm["atoms_per_frame_raw"].quantile(pct / 100)
        )
        for pct in PCT_KEYS
    }

    density_rows = []
    for label, apf in (
        ("sparse_p10", float(adm["atoms_per_frame_raw"].quantile(0.10))),
        ("median_p50", float(adm["atoms_per_frame_raw"].quantile(0.50))),
        ("dense_p90", float(adm["atoms_per_frame_raw"].quantile(0.90))),
    ):
        atoms_per_frame_post = apf * macro_compression
        tokens_per_frame = atoms_per_frame_post / atoms_per_token
        tokens_per_sec = tokens_per_frame * 50.04
        prompt_sec = prompt / tokens_per_sec
        max_sec = max_seq / tokens_per_sec
        density_rows.append(
            {
                "density": label,
                "atoms_per_frame_raw": apf,
                "atoms_per_frame_post_macro": atoms_per_frame_post,
                "tokens_per_frame": tokens_per_frame,
                "tokens_per_sec": tokens_per_sec,
                "prompt_seconds": prompt_sec,
                "max_seconds": max_sec,
            }
        )

    return {
        "n_admitted": int(len(adm)),
        "n_total": int(len(manifest)),
        "atoms_per_token": atoms_per_token,
        "macro_compression": macro_compression,
        "prompt_tokens": prompt,
        "max_tokens": max_seq,
        "duration_seconds": duration_pcts,
        "frames": frame_pcts,
        "atoms_per_frame_raw": apf_pcts,
        "density_table": density_rows,
    }


def print_report(result: dict) -> None:
    print(
        f"=== corpus duration coverage ({result['n_admitted']:,} of "
        f"{result['n_total']:,} SIDs admitted; PAL 50.04 Hz) ==="
    )
    print()
    print("Song duration distribution:")
    for pct in PCT_KEYS:
        s = result["duration_seconds"][f"p{pct:02d}_seconds"]
        f = result["frames"][f"p{pct:02d}_frames"]
        print(f"  p{pct:02d}: {int(f):>7,} frames = {s:>6.1f} sec = {_fmt_sec(s)}")
    print()
    print(
        f"Encoder constants: atoms/token={result['atoms_per_token']:.3f} "
        f"(BPE), macro_compression={result['macro_compression']:.2f} "
        f"(post-pass atoms / raw atoms)"
    )
    print()
    print(
        f"Audio per token budget (PROMPT={result['prompt_tokens']}, "
        f"MAX={result['max_tokens']}):"
    )
    header = (
        f"  {'density':<12} {'atoms/frame':>12} {'tokens/sec':>11} "
        f"{'PROMPT sec':>12} {'MAX sec':>10}"
    )
    print(header)
    for row in result["density_table"]:
        print(
            f"  {row['density']:<12} "
            f"{row['atoms_per_frame_raw']:>12.2f} "
            f"{row['tokens_per_sec']:>11.1f} "
            f"{row['prompt_seconds']:>9.0f} ({_fmt_sec(row['prompt_seconds'])}) "
            f"{row['max_seconds']:>6.0f} ({_fmt_sec(row['max_seconds'])})"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--atoms-per-token",
        type=float,
        default=DEFAULT_ATOMS_PER_TOKEN,
        help="BPE compression. Override or pass --tokenize-log to derive.",
    )
    ap.add_argument(
        "--tokenize-log",
        default=None,
        help="Path to tokenize.log; if it carries atoms/token, overrides default.",
    )
    ap.add_argument(
        "--macro-compression",
        type=float,
        default=DEFAULT_MACRO_COMPRESSION,
        help="Post-pass atoms / raw atoms. Default from 2026-05-19 prodlike audit.",
    )
    ap.add_argument("--prompt", type=int, default=DEFAULT_PROMPT)
    ap.add_argument("--max", dest="max_seq", type=int, default=DEFAULT_MAX)
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = ap.parse_args()

    manifest = pd.read_parquet(args.manifest)
    atoms_per_token = args.atoms_per_token
    if args.tokenize_log:
        derived = _atoms_per_token_from_log(Path(args.tokenize_log))
        if derived is not None:
            atoms_per_token = derived

    result = compute_coverage(
        manifest,
        atoms_per_token=atoms_per_token,
        macro_compression=args.macro_compression,
        prompt=args.prompt,
        max_seq=args.max_seq,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
