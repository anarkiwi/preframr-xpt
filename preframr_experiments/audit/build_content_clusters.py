#!/usr/bin/env python3
"""Offline content-tier cluster index builder for the cluster-conditional content head (Phase 0). See integration_tests/design/cluster_conditional_content_head_design.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from preframr.utils import get_logger
from preframr_tokens import RegTokenizer
from preframr_tokens import CONTENT_TIER, vocab_id_tier

_FEATURE_MODES = ("structural", "engine_fp", "mel")


def _content_vocab_ids(tokens_df: pd.DataFrame, rt: RegTokenizer) -> list[int]:
    n_vocab = len(tokens_df)
    return [
        vid
        for vid in range(n_vocab)
        if vocab_id_tier(vid, rt, tokens_df) == CONTENT_TIER
    ]


def _structural_features(tokens_df: pd.DataFrame, vids: list[int]) -> np.ndarray:
    """One-hot(op) | one-hot(reg) | one-hot(subreg) | normalised(val)."""
    rows = tokens_df.iloc[vids]
    ops = sorted(tokens_df["op"].unique())
    regs = sorted(tokens_df["reg"].unique())
    subregs = sorted(tokens_df["subreg"].unique())
    op_idx = {v: i for i, v in enumerate(ops)}
    reg_idx = {v: i for i, v in enumerate(regs)}
    subreg_idx = {v: i for i, v in enumerate(subregs)}
    n = len(vids)
    feat = np.zeros((n, len(ops) + len(regs) + len(subregs) + 1), dtype=np.float64)
    val_max = float(tokens_df["val"].abs().max()) or 1.0
    for i, row in enumerate(rows.itertuples(index=False)):
        feat[i, op_idx[row.op]] = 1.0
        feat[i, len(ops) + reg_idx[row.reg]] = 1.0
        feat[i, len(ops) + len(regs) + subreg_idx[row.subreg]] = 1.0
        feat[i, -1] = float(row.val) / val_max
    return feat


def _mel_features(
    tokens_df: pd.DataFrame, vids: list[int], n_workers: int = -1
) -> np.ndarray:
    """Per-vocab-id mel feature via canonical-context render. Op-agnostic: treats (reg, val) as a direct SID register write regardless of the token's op-class. Macros (CTRL_BIGRAM, SLOPE, etc.) thus cluster on their (reg, val) signature, not on their full multi-write expansion -- acceptable first-pass approximation; proper macro decoding is a Phase 0b iteration if Phase 2 looks promising."""
    from preframr_audio.fingerprint import fingerprint_batch

    rows = tokens_df.iloc[vids]
    sequences = [
        [(0, int(row.reg), int(row.val))] for row in rows.itertuples(index=False)
    ]
    return fingerprint_batch(sequences, n_workers=n_workers, feature="mel")


def _engine_fp_features(*_args, **_kwargs) -> np.ndarray:
    raise NotImplementedError(
        "engine_fp feature mode is reserved for the preframr-tokens-side "
        "fingerprint that works on full dump parquets; not applicable to "
        "per-vocab-id clustering. Use --feature mel or --feature structural."
    )


_FEATURE_FNS = {
    "structural": _structural_features,
    "mel": _mel_features,
    "engine_fp": _engine_fp_features,
}


def _silhouette_sample(
    feat: np.ndarray, labels: np.ndarray, sample_size: int, rng: np.random.Generator
) -> float:
    """Compute mean silhouette score on a random subsample of points. Pure numpy; avoids the sklearn dependency. Returns 0.0 if there's only one cluster in the sample."""
    n = len(feat)
    if sample_size >= n:
        idx = np.arange(n)
    else:
        idx = rng.choice(n, size=sample_size, replace=False)
    sub_feat = feat[idx]
    sub_labels = labels[idx]
    unique = np.unique(sub_labels)
    if len(unique) < 2:
        return 0.0
    dists = np.sqrt(((sub_feat[:, None, :] - sub_feat[None, :, :]) ** 2).sum(axis=-1))
    scores = []
    for i in range(len(idx)):
        same = sub_labels == sub_labels[i]
        same[i] = False
        if not same.any():
            continue
        a = dists[i, same].mean()
        b_per_cluster = []
        for c in unique:
            if c == sub_labels[i]:
                continue
            mask = sub_labels == c
            if mask.any():
                b_per_cluster.append(dists[i, mask].mean())
        if not b_per_cluster:
            continue
        b = min(b_per_cluster)
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores)) if scores else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens-csv", type=Path, required=True)
    ap.add_argument("--tkmodel-json", type=Path, required=True)
    ap.add_argument(
        "--feature",
        choices=_FEATURE_MODES,
        default="structural",
        help="structural=op/reg/subreg/val one-hot (Phase 0a, cheap); "
        "engine_fp / mel require canonical-context synthesis (Phase 0b).",
    )
    ap.add_argument("--c", type=int, default=256, help="K-means cluster count.")
    ap.add_argument(
        "--n-workers",
        type=int,
        default=-1,
        help="multiprocessing.Pool size for mel rendering (-1 = cpu_count, "
        "1 = in-process sequential). Ignored for non-mel feature modes.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--silhouette-sample",
        type=int,
        default=512,
        help="Subsample size for silhouette score (pairwise; quadratic cost).",
    )
    ap.add_argument("--out", type=Path, required=True)
    cli = ap.parse_args()

    logger = get_logger("INFO")
    tokens_df = pd.read_csv(cli.tokens_csv)
    args = argparse.Namespace(tkvocab=32768)
    rt = RegTokenizer(args, tokens=tokens_df)
    rt.load(cli.tkmodel_json.read_text(), tokens_df)

    content_vids = _content_vocab_ids(tokens_df, rt)
    logger.info(
        "tokens.csv=%s n_vocab=%u n_content=%u (%.1f%%)",
        cli.tokens_csv,
        len(tokens_df),
        len(content_vids),
        100 * len(content_vids) / len(tokens_df),
    )

    if cli.feature == "mel":
        feat = _mel_features(tokens_df, content_vids, n_workers=cli.n_workers)
    else:
        feat = _FEATURE_FNS[cli.feature](tokens_df, content_vids)
    logger.info("feature mode=%s feat.shape=%s", cli.feature, feat.shape)

    rng = np.random.default_rng(cli.seed)
    _centroids, labels = kmeans2(feat, cli.c, minit="++", rng=rng, iter=50)
    logger.info("k-means done: %u clusters", cli.c)

    sizes = np.bincount(labels, minlength=cli.c)
    size_min = int(sizes.min())
    size_max = int(sizes.max())
    size_max_frac = size_max / max(len(content_vids), 1)
    n_singletons = int((sizes == 1).sum())
    singleton_vocab_frac = n_singletons / max(len(content_vids), 1)
    silhouette = _silhouette_sample(feat, labels, cli.silhouette_sample, rng)

    gate_silhouette = silhouette > 0.3
    gate_size_max = size_max_frac <= 0.3
    gate_singletons = singleton_vocab_frac <= 0.05
    gates_pass = gate_silhouette and gate_size_max and gate_singletons

    logger.info(
        "silhouette (n=%u) = %.4f (gate >0.3: %s)",
        cli.silhouette_sample,
        silhouette,
        "PASS" if gate_silhouette else "FAIL",
    )
    logger.info(
        "cluster size min=%u max=%u (max_frac=%.3f, gate <=30%%: %s)",
        size_min,
        size_max,
        size_max_frac,
        "PASS" if gate_size_max else "FAIL",
    )
    logger.info(
        "singletons=%u/%u (%.2f%% of vocab; gate <=5%%: %s)",
        n_singletons,
        cli.c,
        100 * singleton_vocab_frac,
        "PASS" if gate_singletons else "FAIL",
    )
    logger.info("OVERALL: %s", "PASS" if gates_pass else "FAIL")

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "tokens_csv": str(cli.tokens_csv),
        "tkmodel_json": str(cli.tkmodel_json),
        "feature_mode": cli.feature,
        "c": cli.c,
        "seed": cli.seed,
        "n_content_vocab": len(content_vids),
        "silhouette": silhouette,
        "silhouette_sample": cli.silhouette_sample,
        "cluster_sizes": {
            "min": size_min,
            "max": size_max,
            "max_frac": size_max_frac,
            "n_singletons": n_singletons,
            "singleton_vocab_frac": singleton_vocab_frac,
            "all": [int(s) for s in sizes],
        },
        "gates": {
            "silhouette": gate_silhouette,
            "size_max": gate_size_max,
            "singletons": gate_singletons,
            "overall": gates_pass,
        },
        "cluster_assignments": {int(v): int(c) for v, c in zip(content_vids, labels)},
    }
    cli.out.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", cli.out)
    return 0 if gates_pass else 1


if __name__ == "__main__":
    sys.exit(main())
