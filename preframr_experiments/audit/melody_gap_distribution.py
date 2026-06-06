#!/usr/bin/env python3
"""Distance between consecutive melody-onset unigram positions in the encoded stream,
versus model context length. Tests whether the trigram predictability ceiling (k=2
lookback ~0.80 acc on the collapsed onset line) is reachable by the model at all,
or whether the 2 prior onsets sit outside the attention window.

Per encoded block, finds positions where the unigram decodes to (or contains, in the
merged case) any melody pitch atom -- op45 V0_HI/LO + op48 FREQ_ONSET + op47 FREQ_NUDGE
pitch on freq regs (0,7,14) -- then reports gap-to-prev and k-back distance percentiles
and the fraction within --ctx-len. Pure helpers are unit-tested; parquet/tokenizer
loads are lazy."""

from __future__ import annotations

import argparse
import glob as _glob
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np

_PCTS = (50, 75, 90, 95, 99)


def melody_positions(seq: np.ndarray, is_melody_uid: np.ndarray) -> np.ndarray:
    """Indices in ``seq`` where the unigram id is a melody-bearing token. Vectorised."""
    return np.flatnonzero(is_melody_uid[seq])


def gap_stats(positions: np.ndarray, kbacks=(1, 2, 3)) -> dict:
    """gap-to-prev (k=1) and k-back distance distributions, in unigram-position units.
    Returns {'n_positions', 'gaps': {k: {'mean','median','pct': {p: v}, 'max'}}}."""
    out: dict = {"n_positions": int(positions.size), "gaps": {}}
    for k in kbacks:
        if positions.size <= k:
            out["gaps"][k] = None
            continue
        diffs = positions[k:] - positions[:-k]
        out["gaps"][k] = {
            "n": int(diffs.size),
            "mean": float(diffs.mean()),
            "median": float(np.median(diffs)),
            "pct": {p: float(np.percentile(diffs, p)) for p in _PCTS},
            "max": int(diffs.max()),
        }
    return out


def within_context(positions: np.ndarray, ctx_len: int, k: int = 2) -> dict:
    """Fraction of positions for which the k-th prior melody position sits within
    ``ctx_len`` tokens back (i.e. inside the attention window when predicting position i).
    """
    if positions.size <= k:
        return {"n": 0, "frac_in_ctx": None, "frac_in_half_ctx": None}
    diffs = positions[k:] - positions[:-k]
    return {
        "n": int(diffs.size),
        "frac_in_ctx": float((diffs <= ctx_len).mean()),
        "frac_in_half_ctx": float((diffs <= ctx_len // 2).mean()),
    }


def pool_gap_diffs(per_block_positions: Iterable[np.ndarray], k: int) -> np.ndarray:
    """Concatenate k-back diffs across blocks (gaps don't span block boundaries)."""
    chunks: list[np.ndarray] = []
    for pos in per_block_positions:
        if pos.size > k:
            chunks.append(pos[k:] - pos[:-k])
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def summarise_diffs(diffs: np.ndarray) -> Optional[dict]:
    if diffs.size == 0:
        return None
    return {
        "n": int(diffs.size),
        "mean": float(diffs.mean()),
        "median": float(np.median(diffs)),
        "pct": {p: float(np.percentile(diffs, p)) for p in _PCTS},
        "max": int(diffs.max()),
    }


def _build_is_melody_uid(work_dir: Path) -> tuple[np.ndarray, int]:
    """Load tokens.csv + tkmodel.json from a seed work_dir; return (is_melody_uid[vocab],
    n_atoms). A unigram is "melody" if its decode contains ANY melody pitch atom."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    from preframr_tokens.regtokenizer import (  # pylint: disable=import-outside-toplevel
        RegTokenizer,
        is_melody_pitch_atom,
    )

    tokens_df = pd.read_csv(work_dir / "tokens.csv")
    with open(work_dir / "tkmodel.json", encoding="utf-8") as fh:
        tkmodel_json = fh.read()
    rt = RegTokenizer(SimpleNamespace(), tokens_df)
    rt.load(tkmodel_json, tokens_df)
    n_atoms = len(tokens_df)
    op = tokens_df["op"].to_numpy()
    reg = tokens_df["reg"].to_numpy()
    sr = tokens_df["subreg"].to_numpy()
    atom_is_melody = np.array(
        [
            is_melody_pitch_atom(int(op[i]), int(reg[i]), int(sr[i]))
            for i in range(n_atoms)
        ],
        dtype=bool,
    )
    vocab_size = rt.tkmodel.get_vocab_size()
    is_melody_uid = np.zeros(vocab_size, dtype=bool)
    for uid in range(vocab_size):
        base = rt.decode(np.array([uid], dtype=np.uint32))
        for b in base:
            bi = int(b)
            if 0 <= bi < n_atoms and atom_is_melody[bi]:
                is_melody_uid[uid] = True
                break
    return is_melody_uid, n_atoms


def _iter_block_files(work_dir: Path, subset: str) -> list[str]:
    return sorted(_glob.glob(str(work_dir / subset / "*" / "*.0.blocks.npy")))


def analyse_arm(
    work_dir: Path,
    subset: str = "train",
    ctx_len: int = 1024,
    max_blocks: int = 0,
    kbacks=(1, 2, 3),
) -> dict:
    """Pool melody-gap stats across all encoded blocks under ``work_dir/<subset>``."""
    is_melody_uid, n_atoms = _build_is_melody_uid(work_dir)
    files = _iter_block_files(work_dir, subset)
    per_block_positions: list[np.ndarray] = []
    n_blocks = 0
    n_melody_tokens = 0
    n_total_tokens = 0
    block_len = None
    for f in files:
        arr = np.load(f)
        if arr.ndim == 1:
            arr = arr[None, :]
        for row in arr:
            if block_len is None:
                block_len = int(row.size)
            pos = melody_positions(row.astype(np.int64), is_melody_uid)
            per_block_positions.append(pos)
            n_melody_tokens += int(pos.size)
            n_total_tokens += int(row.size)
            n_blocks += 1
            if max_blocks and n_blocks >= max_blocks:
                break
        if max_blocks and n_blocks >= max_blocks:
            break
    out: dict = {
        "work_dir": str(work_dir),
        "subset": subset,
        "ctx_len_arg": int(ctx_len),
        "block_len": block_len,
        "n_blocks": n_blocks,
        "n_total_tokens": n_total_tokens,
        "n_melody_tokens": n_melody_tokens,
        "melody_token_frac": (
            n_melody_tokens / n_total_tokens if n_total_tokens else 0.0
        ),
        "vocab_size": int(is_melody_uid.size),
        "n_melody_unigrams": int(is_melody_uid.sum()),
        "n_atoms": int(n_atoms),
        "gaps": {},
        "within_context": {},
    }
    for k in kbacks:
        diffs = pool_gap_diffs(per_block_positions, k)
        out["gaps"][k] = summarise_diffs(diffs)
        if diffs.size:
            out["within_context"][k] = {
                "frac_in_ctx": float((diffs <= ctx_len).mean()),
                "frac_in_half_ctx": float((diffs <= ctx_len // 2).mean()),
                "frac_in_quarter_ctx": float((diffs <= ctx_len // 4).mean()),
            }
        else:
            out["within_context"][k] = None
    return out


def format_report(res: dict) -> str:
    lines = [
        f"work_dir: {res['work_dir']}  subset={res['subset']}  blocks={res['n_blocks']}",
        f"block_len={res['block_len']}  ctx_len(arg)={res['ctx_len_arg']}  "
        f"vocab={res['vocab_size']}  melody_unigrams={res['n_melody_unigrams']}  "
        f"atoms={res['n_atoms']}",
        f"melody tokens: {res['n_melody_tokens']:,} / {res['n_total_tokens']:,} "
        f"({100*res['melody_token_frac']:.2f}%)",
    ]
    for k, g in res["gaps"].items():
        if g is None:
            lines.append(f"k={k}: (insufficient positions)")
            continue
        pct = g["pct"]
        lines.append(
            f"k={k} look-back gap (unigram positions): "
            f"n={g['n']:,}  median={g['median']:.0f}  "
            f"p75={pct[75]:.0f} p90={pct[90]:.0f} p95={pct[95]:.0f} p99={pct[99]:.0f}  "
            f"max={g['max']}"
        )
        wc = res["within_context"].get(k)
        if wc is not None:
            lines.append(
                f"    within ctx={res['ctx_len_arg']}: {100*wc['frac_in_ctx']:.1f}%  "
                f"within ctx/2={100*wc['frac_in_half_ctx']:.1f}%  "
                f"within ctx/4={100*wc['frac_in_quarter_ctx']:.1f}%"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Seed work_dir holding tokens.csv + tkmodel.json + train/eval block dirs.",
    )
    ap.add_argument(
        "--subset",
        default="train",
        help="Which encoded subset to scan (default: train).",
    )
    ap.add_argument(
        "--ctx-len",
        type=int,
        default=1024,
        help="Model context length to compare gaps against (default 1024).",
    )
    ap.add_argument(
        "--max-blocks",
        type=int,
        default=0,
        help="Stop after N blocks (0 = all).",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON only.")
    cli = ap.parse_args()
    res = analyse_arm(
        cli.work_dir, subset=cli.subset, ctx_len=cli.ctx_len, max_blocks=cli.max_blocks
    )
    if cli.json:
        print(json.dumps(res, indent=2))
    else:
        print(format_report(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
