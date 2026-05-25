"""Aggregator over the corpus_structural_index parquet shards.
Emits per-corpus aggregates under ``<index-root>/aggregates/``. See
``integration_tests/design/corpus_structural_index_design.md`` for the
schema and audit→column mapping. Pure pandas/pyarrow; no extra deps."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("aggregate_corpus_index")

CLUSTERS_FALLBACK = (-1, 1, 2, 3, 4, 5, 6, 7)
VOICES = 3
NON_VOICE_MARKER = 255


def _shards(index_root: Path, kind: str, cluster: int) -> list[Path]:
    pat = str(index_root / kind / f"cluster={cluster}" / "*" / "*.parquet")
    return [Path(p) for p in glob.glob(pat)]


def _list_clusters(index_root: Path, kind: str = "per_frame") -> list[int]:
    paths = glob.glob(str(index_root / kind / "cluster=*"))
    out = []
    for p in paths:
        try:
            out.append(int(os.path.basename(p).split("=", 1)[1]))
        except (ValueError, IndexError):
            continue
    return sorted(out) or list(CLUSTERS_FALLBACK)


def manifest_summary(index_root: Path) -> pd.DataFrame:
    """One-row-per-cluster manifest roll-up: counts, admit rate, skip mix."""
    mf = pd.read_parquet(index_root / "manifest.parquet")
    g = mf.groupby("engine_fp_cluster", as_index=False).agg(
        n_sids=("sid_path", "count"),
        n_admitted=("skipped_by_filter", lambda s: int((~s).sum())),
        n_frames_sum=("n_frames", "sum"),
        n_atoms_raw_sum=("n_atoms_raw", "sum"),
        max_writes_per_frame_max=("max_writes_per_frame", "max"),
        max_ctrl_pvf_max=("max_ctrl_writes_per_voice_frame", "max"),
        has_pcm_voice_count=("has_pcm_voice", "sum"),
    )
    g["admit_rate"] = g["n_admitted"] / g["n_sids"].clip(lower=1)
    return g


def _decode_atoms_blob(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.uint8)
    if arr.size % 4 != 0:
        return np.empty((0, 4), dtype=np.uint8)
    return arr.reshape(-1, 4)


def _shard_atom_counts(per_frame_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(per_frame_path, columns=["intra_frame_op_hist"])
    if df.empty:
        return pd.DataFrame(columns=["voice", "sub", "val", "count"])
    blobs = df["intra_frame_op_hist"].to_numpy()
    rows = [_decode_atoms_blob(b) for b in blobs if b]
    if not rows:
        return pd.DataFrame(columns=["voice", "sub", "val", "count"])
    cat = np.concatenate(rows, axis=0) if rows else np.empty((0, 4), dtype=np.uint8)
    if cat.size == 0:
        return pd.DataFrame(columns=["voice", "sub", "val", "count"])
    val16 = cat[:, 2].astype(np.uint16) | (cat[:, 3].astype(np.uint16) << 8)
    flat = pd.DataFrame(
        {
            "voice": cat[:, 0],
            "sub": cat[:, 1],
            "val": val16,
        }
    )
    out = flat.value_counts().reset_index(name="count")
    return out


def reg_op_val_freq(index_root: Path) -> pd.DataFrame:
    """Global per-(voice, sub, val) atom frequency from intra_frame_op_hist
    blobs. Each row is one (voice|255, sub_reg, value) tuple. Used by
    digi-detection threshold tuning + macro-bit-budget audits."""
    clusters = _list_clusters(index_root, "per_frame")
    parts = []
    for c in clusters:
        shards = _shards(index_root, "per_frame", c)
        if not shards:
            continue
        per_cluster = []
        for i, sh in enumerate(shards):
            per_cluster.append(_shard_atom_counts(sh))
            if (i + 1) % 1000 == 0:
                LOG.info("reg_op_val_freq: cluster=%d %u/%u", c, i + 1, len(shards))
        if not per_cluster:
            continue
        merged = (
            pd.concat(per_cluster, ignore_index=True)
            .groupby(["voice", "sub", "val"], as_index=False)["count"]
            .sum()
        )
        merged["cluster"] = c
        parts.append(merged)
        LOG.info("reg_op_val_freq: cluster=%d done (%u rows)", c, len(merged))
    if not parts:
        return pd.DataFrame(columns=["voice", "sub", "val", "cluster", "count"])
    all_rows = pd.concat(parts, ignore_index=True)
    global_counts = (
        all_rows.groupby(["voice", "sub", "val"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "count_global"})
    )
    pivot = all_rows.pivot_table(
        index=["voice", "sub", "val"],
        columns="cluster",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    pivot.columns = [
        f"count_cluster_{c}" if isinstance(c, (int, np.integer)) else c
        for c in pivot.columns
    ]
    return global_counts.merge(pivot, on=["voice", "sub", "val"], how="left")


def voice_state_reuse(index_root: Path, top_n: int = 200_000) -> pd.DataFrame:
    """Per-voice_state_fp roll-up: how often each (ctrl|ad|sr|pwm|freq16)
    fingerprint recurs across the corpus. The long tail is sparse; truncate
    to ``top_n`` rows by frame_count. Inputs the global_instr_ids decision."""
    clusters = _list_clusters(index_root, "per_voice_frame")
    parts = []
    for c in clusters:
        shards = _shards(index_root, "per_voice_frame", c)
        if not shards:
            continue
        per_fp: dict[int, int] = {}
        per_fp_sids: dict[int, set] = {}
        for i, sh in enumerate(shards):
            df = pd.read_parquet(sh, columns=["voice_state_fp"])
            counts = df["voice_state_fp"].value_counts()
            for fp, cnt in counts.items():
                per_fp[fp] = per_fp.get(fp, 0) + int(cnt)
                per_fp_sids.setdefault(fp, set()).add(sh.stem)
            if (i + 1) % 2000 == 0:
                LOG.info(
                    "voice_state_reuse: cluster=%d %u/%u (%u uniq fps so far)",
                    c,
                    i + 1,
                    len(shards),
                    len(per_fp),
                )
        if not per_fp:
            continue
        df = pd.DataFrame(
            {
                "voice_state_fp": list(per_fp.keys()),
                "frame_count": list(per_fp.values()),
                "sid_count": [len(per_fp_sids[fp]) for fp in per_fp.keys()],
                "cluster": c,
            }
        )
        df = df.nlargest(top_n // max(1, len(clusters)), "frame_count")
        parts.append(df)
        LOG.info("voice_state_reuse: cluster=%d done (%u rows kept)", c, len(df))
    if not parts:
        return pd.DataFrame(
            columns=["voice_state_fp", "frame_count", "sid_count", "cluster"]
        )
    return pd.concat(parts, ignore_index=True)


def _detect_pwm_preset_b2b(pvf: pd.DataFrame) -> tuple[int, int]:
    if pvf.empty:
        return 0, 0
    sites = 0
    savings = 0
    for v in range(VOICES):
        sub = pvf[pvf["voice"] == v].sort_values("frame_idx")
        if sub.empty:
            continue
        lonely = (
            (sub["pwm_writes_this_frame"] == 1)
            & (sub["ctrl_writes_this_frame"] == 0)
            & (sub["freq_writes_this_frame"] == 0)
        )
        run = (lonely.shift(fill_value=False) & lonely).to_numpy().sum()
        sites += int(run)
        savings += int(run)
    return sites, savings


def _detect_last_write_wins_ctrl(pvf: pd.DataFrame) -> tuple[int, int]:
    if pvf.empty:
        return 0, 0
    hot = pvf["ctrl_writes_this_frame"] >= 3
    if not hot.any():
        return 0, 0
    sites = int(hot.sum())
    savings = int((pvf.loc[hot, "ctrl_writes_this_frame"] - 1).sum())
    return sites, savings


def _detect_arpeggio_window(pvf: pd.DataFrame, window: int = 16) -> tuple[int, int]:
    if pvf.empty or len(pvf) < window * VOICES:
        return 0, 0
    sites = 0
    for v in range(VOICES):
        freq = (
            pvf[pvf["voice"] == v]
            .sort_values("frame_idx")["freq16"]
            .to_numpy(dtype=np.uint16)
        )
        if freq.size < window:
            continue
        ptr = 0
        while ptr + window <= freq.size:
            seg = freq[ptr : ptr + window]
            uniq = np.unique(seg).size
            if 2 <= uniq <= 4:
                sites += 1
                ptr += window
            else:
                ptr += 1
    return sites, sites * (window - 1) // 2


_MACRO_DETECTORS = (
    ("pwm_preset_b2b", _detect_pwm_preset_b2b),
    ("last_write_wins_ctrl", _detect_last_write_wins_ctrl),
    ("arpeggio_window_16", _detect_arpeggio_window),
)


def macro_potential(index_root: Path) -> pd.DataFrame:
    """Per (SID, macro_name) site count + estimated atom savings."""
    rows = []
    clusters = _list_clusters(index_root, "per_voice_frame")
    for c in clusters:
        shards = _shards(index_root, "per_voice_frame", c)
        for i, sh in enumerate(shards):
            try:
                pvf = pd.read_parquet(
                    sh,
                    columns=[
                        "frame_idx",
                        "voice",
                        "ctrl_writes_this_frame",
                        "freq_writes_this_frame",
                        "pwm_writes_this_frame",
                        "freq16",
                    ],
                )
            except Exception as exc:
                LOG.warning("macro_potential: bad shard %s: %s", sh, exc)
                continue
            composer = sh.parent.name.split("=", 1)[1] if "=" in sh.parent.name else ""
            sid_key = sh.stem
            for name, fn in _MACRO_DETECTORS:
                sites, savings = fn(pvf)
                if sites == 0:
                    continue
                rows.append(
                    {
                        "cluster": c,
                        "composer": composer,
                        "sid_key": sid_key,
                        "macro_name": name,
                        "sites_count": sites,
                        "est_atom_savings": savings,
                    }
                )
            if (i + 1) % 2000 == 0:
                LOG.info("macro_potential: cluster=%d %u/%u", c, i + 1, len(shards))
        LOG.info("macro_potential: cluster=%d done", c)
    if not rows:
        return pd.DataFrame(
            columns=[
                "cluster",
                "composer",
                "sid_key",
                "macro_name",
                "sites_count",
                "est_atom_savings",
            ]
        )
    return pd.DataFrame(rows)


_REGISTRY = {
    "manifest_summary": manifest_summary,
    "reg_op_val_freq": reg_op_val_freq,
    "voice_state_reuse": voice_state_reuse,
    "macro_potential": macro_potential,
}


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", default="/scratch/preframr/corpus_index_full")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Defaults to <index-root>/aggregates",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=sorted(_REGISTRY.keys()),
        help="Aggregate(s) to compute; defaults to all.",
    )
    args = ap.parse_args()
    index_root = Path(args.index_root)
    out_dir = Path(args.out_dir) if args.out_dir else index_root / "aggregates"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.only or list(_REGISTRY.keys())
    for name in names:
        fn = _REGISTRY[name]
        t0 = time.time()
        LOG.info("starting %s", name)
        df = fn(index_root)
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False, compression="zstd")
        LOG.info(
            "wrote %s (%u rows) in %.1fs",
            out_path,
            len(df),
            time.time() - t0,
        )


if __name__ == "__main__":
    main()
