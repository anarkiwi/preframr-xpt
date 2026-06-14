#!/usr/bin/env python3
"""Predictive-entropy & calibration probe on content positions.

Measures whether the checkpoint's next-token distribution is well-calibrated:
expected calibration error (ECE), mean top-1 confidence vs accuracy (the
over/under-confidence gap), and the predictive-entropy distribution — restricted to
content positions, where it matters. Overconfidence (confidence >> accuracy, low
entropy) predicts the sampling mode-collapse to the near-silent DELAY drone; chronic
underconfidence predicts free-running drift. Nothing else in the suite measures
calibration, and it is the read that informs the Tier-1 structure-aware sampling regime.

Assumes the default single-softmax head (the per-tier-heads arm is refuted/off). Pure
measurement (ECE, summary) is unit-tested in CI; the forward + softmax reuse the
checkpoint forward path and run on a GPU host. See
`design/generation/free_running_pathology_remediation_design.md` (Tier 1).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Pure measurement (torch-free; unit-tested)
# --------------------------------------------------------------------------- #
def ece(confidences, corrects, n_bins: int = 10):
    """Expected calibration error: bin by top-1 confidence, sum |acc - conf| weighted by
    bin population. None if empty."""
    if not confidences:
        return None
    bins = [[0.0, 0, 0] for _ in range(n_bins)]  # [conf_sum, correct_sum, count]
    for c, hit in zip(confidences, corrects):
        b = min(int(c * n_bins), n_bins - 1)
        bins[b][0] += c
        bins[b][1] += hit
        bins[b][2] += 1
    n = len(confidences)
    e = 0.0
    for conf_sum, corr, cnt in bins:
        if not cnt:
            continue
        e += (cnt / n) * abs(corr / cnt - conf_sum / cnt)
    return e


def summarize(confidences, corrects, entropies, n_bins: int = 10) -> dict:
    """ECE + confidence/accuracy gap + entropy stats over the supplied content positions."""
    n = len(confidences)
    if not n:
        return {"n": 0}
    mean_conf = sum(confidences) / n
    acc = sum(corrects) / n
    ents = sorted(entropies)
    return {
        "n": n,
        "ece": ece(confidences, corrects, n_bins),
        "mean_confidence": mean_conf,
        "accuracy": acc,
        "overconfidence": mean_conf - acc,  # >0 overconfident, <0 underconfident
        "mean_entropy_nats": sum(entropies) / len(entropies) if entropies else None,
        "median_entropy_nats": ents[len(ents) // 2] if ents else None,
    }


# --------------------------------------------------------------------------- #
# Forward + softmax (torch; GPU host) — lazy-imported
# --------------------------------------------------------------------------- #
def _iter_blocks(blocks_glob: str, n_blocks: int, logger):
    import numpy as np  # pylint: disable=import-outside-toplevel

    out = []
    for path in sorted(glob.glob(blocks_glob, recursive=True)):
        arr = np.load(path)
        for row in np.atleast_2d(arr):
            nz = row[row > 0]
            if len(nz) >= 2:
                out.append((path, [int(t) for t in nz]))
                break
        if len(out) >= n_blocks:
            break
    logger.info("calibration: %u blocks from %s", len(out), blocks_glob)
    return out


def run_audit(args, logger) -> int:
    import torch  # pylint: disable=import-outside-toplevel
    from torchtune.modules.common_utils import (
        disable_kv_cache,
    )  # pylint: disable=import-outside-toplevel

    from preframr.inference.predict import (
        load_model,
    )  # pylint: disable=import-outside-toplevel

    _dataset, model, device, _ = load_model(
        args, logger
    )  # pylint: disable=unused-variable
    model = model.to(device)
    model.eval()
    content_ids = None
    try:
        from preframr.train.model import (
            build_tier_map,
        )  # pylint: disable=import-outside-toplevel

        tmap = build_tier_map(args, model.n_vocab, model.tokens, model.tkmodel)
        content_ids = {i for i, t in tmap.items() if t == "content"}
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("tier map unavailable (scoring all positions): %s", exc)

    blocks = _iter_blocks(args.blocks_glob, args.n_blocks, logger)
    confidences: list = []
    corrects: list = []
    entropies: list = []
    for _path, block in blocks:
        x = torch.tensor(block[:-1], dtype=torch.long).unsqueeze(0).to(device)
        with torch.inference_mode(), disable_kv_cache(model.model):
            logits = model.model(x)
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
            logp = torch.log_softmax(logits, dim=-1)
            p = logp.exp()
            conf, pred = p.max(dim=-1)
            ent = -(p * logp).sum(dim=-1)
        conf = conf.squeeze(0).cpu().tolist()
        pred = pred.squeeze(0).cpu().tolist()
        ent = ent.squeeze(0).cpu().tolist()
        for i in range(len(pred)):  # pred[i] predicts block[i+1]
            tgt = block[i + 1]
            if content_ids is not None and tgt not in content_ids:
                continue
            confidences.append(float(conf[i]))
            corrects.append(1 if pred[i] == tgt else 0)
            entropies.append(float(ent[i]))

    result = summarize(confidences, corrects, entropies, args.n_bins)
    result["config"] = {
        "blocks_glob": args.blocks_glob,
        "n_blocks": len(blocks),
        "scored": "content" if content_ids is not None else "all",
    }
    text = json.dumps(result, indent=2, default=str)
    if args.out is not None:
        Path(args.out).write_text(text)
    print(text)
    logger.info(
        "calibration: ece=%s overconfidence=%s mean_entropy=%s",
        result.get("ece"),
        result.get("overconfidence"),
        result.get("mean_entropy_nats"),
    )
    return 0


def add_audit_args(parser):
    parser.add_argument(
        "--blocks-glob", type=str, default="/scratch/preframr/train/**/*.blocks.npy"
    )
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main():
    from preframr.args import add_args  # pylint: disable=import-outside-toplevel
    from preframr.utils import get_logger  # pylint: disable=import-outside-toplevel

    parser = add_audit_args(
        add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    )
    args = parser.parse_args()
    logger = get_logger("INFO")
    sys.exit(run_audit(args, logger))


if __name__ == "__main__":
    main()
