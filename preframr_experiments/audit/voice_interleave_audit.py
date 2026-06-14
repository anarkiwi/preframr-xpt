#!/usr/bin/env python3
"""Per-voice interleave accuracy: localize content failure to the cross-voice multiplex.

The event stream is frame-major — within a frame group consecutive events belong to
different voices — so a voice's line is chopped into slices and the next same-voice token
is a long-range, position-unstable dependency. This probe labels each atom's voice (by
replaying the event grammar) and buckets teacher-forced accuracy by voice and by the
**interleave gap** (atoms since the previous same-voice token). If accuracy falls sharply
as the gap grows, the multiplex is the culprit — which is the trigger condition
`lane_demux_hypothesis.md` and the AGENTS "IF CONTENT NOT LEARNED" branch gate on, and it
arms Tier-2 (lane de-mux) before paying to build it.

Atoms-only (tkvocab=0): block ids are atoms. The voice annotation drives
`preframr_tokens.events.constrained.EventStreamState` (CI); the gap/bucketing math is
pure and unit-tested locally. See
`design/generation/free_running_pathology_remediation_design.md` (Tier 2).
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
def _gap_bucket(g: int):
    b = max(g, 1).bit_length() - 1
    return (1 << b, (1 << (b + 1)) - 1)


def voice_gaps(voices) -> list:
    """Per position, atoms since the previous token of the SAME voice (None for the first
    occurrence of a voice, or for non-voice positions where voice < 0)."""
    last: dict = {}
    out = []
    for p, v in enumerate(voices):
        if v is None or v < 0:
            out.append(None)
            continue
        out.append(p - last[v] if v in last else None)
        last[v] = p
    return out


def _bump(acc: dict, key, hit: int) -> None:
    e = acc.setdefault(key, [0, 0])
    e[0] += hit
    e[1] += 1


def interleave_hits(block, tf_pred, voices, gaps, pad_id: int = 0) -> dict:
    """Accumulate hits keyed by voice and by interleave-gap bucket. ``tf_pred[j-1]`` is the
    teacher-forced argmax for ``block[j]``."""
    out = {"voice": {}, "gap": {}}
    for j in range(1, len(block)):
        if block[j] == pad_id or j - 1 >= len(tf_pred):
            continue
        v = voices[j] if j < len(voices) else -1
        if v is None or v < 0:
            continue
        hit = 1 if tf_pred[j - 1] == block[j] else 0
        _bump(out["voice"], v, hit)
        g = gaps[j] if j < len(gaps) else None
        if g is not None:
            _bump(out["gap"], _gap_bucket(g), hit)
    return out


def merge_hits(dst: dict, src: dict) -> None:
    for axis in ("voice", "gap"):
        for key, (hits, tot) in src[axis].items():
            e = dst.setdefault(axis, {}).setdefault(key, [0, 0])
            e[0] += hits
            e[1] += tot


def finalize(acc: dict) -> dict:
    voice = {
        str(v): {"acc": hits / tot, "n": tot}
        for v, (hits, tot) in sorted(acc.get("voice", {}).items())
        if tot
    }
    gap = [
        {"gap_lo": lo, "gap_hi": hi, "acc": hits / tot, "n": tot}
        for (lo, hi), (hits, tot) in sorted(acc.get("gap", {}).items())
        if tot
    ]
    penalty = (gap[0]["acc"] - gap[-1]["acc"]) if len(gap) >= 2 else None
    return {
        "by_voice": voice,
        "by_gap": gap,
        "interleave_penalty": penalty,  # near-gap acc minus far-gap acc; >0 = multiplex hurts
    }


def annotate_voices(block) -> list:
    """Voice id per atom by replaying the event grammar: voice-tag atoms carry their own
    voice; event/value/nibble atoms inherit the current voice; frame-delay digits, headers
    and preamble are -1. Resets on an ungrammatical atom (KEYFRAME / mid-stream cut)."""
    from preframr_tokens.events import (
        stream as s,
    )  # pylint: disable=import-outside-toplevel
    from preframr_tokens.events.constrained import (  # pylint: disable=import-outside-toplevel
        EventStreamState,
    )

    st = EventStreamState()
    voices = []
    for tok in block:
        tok = int(tok)
        v = -1
        if s.VOICE_BASE <= tok < s.VOICE_BASE + 4:
            v = tok - s.VOICE_BASE
        elif getattr(st, "phase", None) == "BODY":
            top = st.stack[-1] if st.stack else None
            is_dt = top is not None and top[0] == "V" and top[2] == "dt"
            if not is_dt and 0 <= st.cur_voice <= 3:
                v = st.cur_voice
        voices.append(v)
        try:
            st.push(tok)
        except Exception:  # pylint: disable=broad-except
            st = EventStreamState()
    return voices


# --------------------------------------------------------------------------- #
# Checkpoint forward (torch; GPU host) — lazy-imported
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
    logger.info("voice_interleave: %u blocks from %s", len(out), blocks_glob)
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
    if getattr(model, "tkmodel", None):
        logger.warning(
            "tkmodel set (BPE) — voice annotation needs atoms; run atoms-only"
        )
    blocks = _iter_blocks(args.blocks_glob, args.n_blocks, logger)
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
        voices = annotate_voices(block)
        gaps = voice_gaps(voices)
        merge_hits(acc, interleave_hits(block, tf_pred, voices, gaps))

    result = finalize(acc)
    result["config"] = {"blocks_glob": args.blocks_glob, "n_blocks": len(blocks)}
    text = json.dumps(result, indent=2, default=str)
    if args.out is not None:
        Path(args.out).write_text(text)
    print(text)
    logger.info("voice_interleave: interleave_penalty=%s", result["interleave_penalty"])
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
