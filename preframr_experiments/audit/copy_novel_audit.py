#!/usr/bin/env python3
"""Copy-vs-novel accuracy decomposition: is the checkpoint's teacher-forced accuracy
*real modeling* or just induction-head copying?

For each predicted token block[j], the position is **copyable** if the bigram the model
is completing — (block[j-1] -> block[j]) — has already occurred earlier in the same
window (an induction head can copy "what followed block[j-1] last time"); otherwise it
is **novel**. We then split teacher-forced top-1 accuracy by copyable vs novel, per
content/structural tier. If content accuracy lives almost entirely on copyable positions
and novel-content accuracy is at floor, the model learned to copy, not to generate — the
static smoking gun for the free-running pathology (it predicts continuations of seen
material but cannot sustain novel structure).

Distinct from `learnability_triage` (corpus-only induction-copy *rate*, no model) — this
is **model-conditioned**: it measures where the checkpoint's correctness concentrates.

Pure measurement (copyability, bucketing) is unit-tested in CI; the checkpoint forward
reuses the `audit_checkpoint_per_class` path and runs on a GPU host. See
`design/generation/free_running_pathology_remediation_design.md` (Tier 0).
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
def copyable_positions(block, pad_id: int = 0) -> dict:
    """Per target position ``j`` (1..len-1, non-pad): is ``block[j]`` induction-copyable —
    has the bigram ``(block[j-1], block[j])`` occurred earlier in ``block[:j]``? Returns
    ``{j: bool}``."""
    seen: set = set()
    out: dict = {}
    prev = None
    for j in range(1, len(block)):
        if prev is not None:
            seen.add(prev)
        a, b = block[j - 1], block[j]
        valid = a != pad_id and b != pad_id
        if valid:
            out[j] = (a, b) in seen
            prev = (a, b)
        else:
            prev = None
    return out


def _bump(acc: dict, key, hit: int) -> None:
    e = acc.setdefault(key, [0, 0])
    e[0] += hit
    e[1] += 1


def copy_novel_hits(block, tf_pred, tier_of=None, pad_id: int = 0):
    """Accumulate hits into ``{"copyable": {key:[hits,tot]}, "novel": {...}}`` where key is
    ``"all"`` plus the tier of ``block[j]`` when ``tier_of`` is given. ``tf_pred[j-1]`` is
    the teacher-forced argmax for ``block[j]``."""
    out = {"copyable": {}, "novel": {}}
    for j, is_copy in copyable_positions(block, pad_id).items():
        if j - 1 >= len(tf_pred):
            continue
        hit = 1 if tf_pred[j - 1] == block[j] else 0
        grp = out["copyable"] if is_copy else out["novel"]
        _bump(grp, "all", hit)
        if tier_of is not None and j < len(tier_of):
            _bump(grp, tier_of[j], hit)
    return out


def merge_hits(dst: dict, src: dict) -> None:
    """Fold one block's ``copy_novel_hits`` into a running accumulator."""
    for grp in ("copyable", "novel"):
        for key, (hits, tot) in src[grp].items():
            e = dst.setdefault(grp, {}).setdefault(key, [0, 0])
            e[0] += hits
            e[1] += tot


def finalize(acc: dict) -> dict:
    """Accumulator -> {group: {key: {acc, n}}} + the headline copyable-minus-novel gaps."""
    out: dict = {}
    for grp in ("copyable", "novel"):
        out[grp] = {
            key: {"acc": hits / tot, "n": tot}
            for key, (hits, tot) in sorted(acc.get(grp, {}).items())
            if tot
        }
    gaps = {}
    for key in sorted(set(out["copyable"]) & set(out["novel"])):
        gaps[key] = out["copyable"][key]["acc"] - out["novel"][key]["acc"]
    out["copy_minus_novel"] = gaps
    src = "content" if "content" in out.get("novel", {}) else "all"
    out["headline"] = {
        "novel_source": src,
        "novel_acc": out["novel"].get(src, {}).get("acc"),
        "copyable_acc": out["copyable"].get(src, {}).get("acc"),
        "copy_minus_novel": gaps.get(src),
    }
    return out


# --------------------------------------------------------------------------- #
# Checkpoint forward (torch; GPU host) — lazy-imported
# --------------------------------------------------------------------------- #
def _iter_blocks(blocks_glob: str, n_blocks: int, min_len: int, logger):
    import numpy as np  # pylint: disable=import-outside-toplevel

    out = []
    for path in sorted(glob.glob(blocks_glob, recursive=True)):
        arr = np.load(path)
        for row in np.atleast_2d(arr):
            nz = row[row > 0]
            if len(nz) < min_len:
                continue
            out.append((path, [int(t) for t in nz]))
            break
        if len(out) >= n_blocks:
            break
    logger.info("copy_novel: %u blocks from %s", len(out), blocks_glob)
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
    tier_of_id = None
    try:
        from preframr.train.model import (
            build_tier_map,
        )  # pylint: disable=import-outside-toplevel

        tier_of_id = build_tier_map(args, model.n_vocab, model.tokens, model.tkmodel)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("tier map unavailable (all-tier only): %s", exc)

    blocks = _iter_blocks(args.blocks_glob, args.n_blocks, 2, logger)
    acc: dict = {}
    for _path, block in blocks:
        x = torch.tensor(block[:-1], dtype=torch.long).unsqueeze(0).to(device)
        with torch.inference_mode(), disable_kv_cache(model.model):
            logits = model.model(x)
        if isinstance(logits, list):
            tf_pred = torch.cat([c.argmax(dim=-1) for c in logits], dim=1)
        else:
            tf_pred = logits.argmax(dim=-1)
        tf_pred = tf_pred.squeeze(0).cpu().tolist()
        tier_of = (
            [tier_of_id.get(int(t), "_unknown") for t in block]
            if tier_of_id is not None
            else None
        )
        merge_hits(acc, copy_novel_hits(block, tf_pred, tier_of=tier_of))

    result = finalize(acc)
    result["config"] = {"blocks_glob": args.blocks_glob, "n_blocks": len(blocks)}
    text = json.dumps(result, indent=2, default=str)
    if args.out is not None:
        Path(args.out).write_text(text)
    print(text)
    h = result["headline"]
    logger.info(
        "copy_novel: novel(%s)=%.3f copyable=%.3f gap=%.3f",
        h["novel_source"],
        h["novel_acc"] or float("nan"),
        h["copyable_acc"] or float("nan"),
        h["copy_minus_novel"] or float("nan"),
    )
    return 0


def add_audit_args(parser):
    parser.add_argument(
        "--blocks-glob", type=str, default="/scratch/preframr/train/**/*.blocks.npy"
    )
    parser.add_argument("--n-blocks", type=int, default=8)
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
