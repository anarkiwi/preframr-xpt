#!/usr/bin/env python3
"""Position-within-event accuracy: separate the by-design sawtooth from pathology.

Buckets teacher-forced accuracy by each atom's grammatical **role** (voice-tag / kind /
value digit / waveform-articulation-envelope nibble / shape / reg / header / keyframe)
and by its **offset since the last event marker**. The v3 encoding front-loads the
most-determined atom (Principle 4.2), so a "high at the marker, lower in the fields"
sawtooth is the *expected* shape even for a healthy model — top-1 accuracy alone can't
tell that apart from a pathological collapse. This read makes the structure explicit, so
a low content-field accuracy can be attributed (front-loading vs genuine miss) rather
than guessed.

Atoms-only (tkvocab=0): block ids are atoms. Pure measurement (role classification,
offset, bucketing) is unit-tested in CI; the forward reuses the checkpoint forward path
on a GPU host. See `design/generation/free_running_pathology_remediation_design.md`
(Tier 0).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_MARKER_ROLES = ("voice_tag", "kind", "header", "keyframe")
_OFFSET_CAP = 8  # offsets >= this fold into the "8+" bucket


# --------------------------------------------------------------------------- #
# Pure measurement (torch-free; unit-tested)
# --------------------------------------------------------------------------- #
def atom_role(tok: int, r: dict) -> str:
    """Coarse grammatical role of an atom id, from the event alphabet's range boundaries
    ``r`` (see :func:`event_atom_ranges`)."""
    if r["var_base"] <= tok < r["var_end"]:
        return "digit"
    if r["reg_base"] <= tok < r["reg_end"]:
        return "reg"
    if r["voice_base"] <= tok < r["voice_end"]:
        return "voice_tag"
    if tok in r["headers"]:
        return "header"
    if r["kinds_lo"] <= tok <= r["kinds_hi"]:
        return "kind"
    if r["shape_lo"] <= tok <= r["shape_hi"]:
        return "shape"
    if r["nib_wave"] <= tok < r["nib_wave"] + 16:
        return "nib_wave"
    if r["nib_art"] <= tok < r["nib_art"] + 16:
        return "nib_art"
    if r["nib_env"] <= tok < r["nib_env"] + 16:
        return "nib_env"
    if tok == r["keyframe"]:
        return "keyframe"
    return "other"


def event_offsets(roles, markers=_MARKER_ROLES) -> list:
    """Atoms since the most recent event-marker role (marker = 0, following fields 1, 2, …;
    positions before the first marker = -1)."""
    offs = []
    o = -1
    for role in roles:
        if role in markers:
            o = 0
        elif o >= 0:
            o += 1
        offs.append(o)
    return offs


def _bump(acc: dict, key, hit: int) -> None:
    e = acc.setdefault(key, [0, 0])
    e[0] += hit
    e[1] += 1


def position_hits(block, tf_pred, r: dict, pad_id: int = 0) -> dict:
    """Accumulate hits keyed by role and by offset bucket. ``tf_pred[j-1]`` is the
    teacher-forced argmax for ``block[j]``."""
    roles = [atom_role(t, r) for t in block]
    offs = event_offsets(roles)
    out = {"role": {}, "offset": {}}
    for j in range(1, len(block)):
        if block[j] == pad_id or j - 1 >= len(tf_pred):
            continue
        hit = 1 if tf_pred[j - 1] == block[j] else 0
        _bump(out["role"], roles[j], hit)
        o = offs[j]
        key = "pre" if o < 0 else (f"{_OFFSET_CAP}+" if o >= _OFFSET_CAP else str(o))
        _bump(out["offset"], key, hit)
    return out


def merge_hits(dst: dict, src: dict) -> None:
    for axis in ("role", "offset"):
        for key, (hits, tot) in src[axis].items():
            e = dst.setdefault(axis, {}).setdefault(key, [0, 0])
            e[0] += hits
            e[1] += tot


def finalize(acc: dict) -> dict:
    return {
        axis: {
            key: {"acc": hits / tot, "n": tot}
            for key, (hits, tot) in sorted(acc.get(axis, {}).items())
            if tot
        }
        for axis in ("role", "offset")
    }


def event_atom_ranges() -> dict:
    """Range boundaries of the v3 event alphabet, from preframr_tokens (lazy import)."""
    from preframr_tokens.events import (
        stream as s,
    )  # pylint: disable=import-outside-toplevel

    return {
        "var_base": s.VAR_BASE,
        "var_end": s.REG_BASE,
        "reg_base": s.REG_BASE,
        "reg_end": s.VOICE_BASE,
        "voice_base": s.VOICE_BASE,
        "voice_end": s.VOICE_BASE + 4,
        "headers": frozenset((s.TUNING, s.NOTE_TABLE, s.TICK)),
        "kinds_lo": s.NI_STEP,
        "kinds_hi": s.G_RAMP,
        "shape_lo": s.SHAPE_POLY,
        "shape_hi": s.SHAPE_PERIOD,
        "nib_wave": s.NIB_WAVE,
        "nib_art": s.NIB_ART,
        "nib_env": s.NIB_ENV,
        "keyframe": s.KEYFRAME,
    }


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
    logger.info("event_position: %u blocks from %s", len(out), blocks_glob)
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
            "tkmodel set (BPE) — roles are id-space, not atom-space; run atoms-only (tkvocab=0)"
        )
    ranges = event_atom_ranges()
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
        merge_hits(acc, position_hits(block, tf_pred, ranges))

    result = finalize(acc)
    result["config"] = {"blocks_glob": args.blocks_glob, "n_blocks": len(blocks)}
    text = json.dumps(result, indent=2, default=str)
    if args.out is not None:
        Path(args.out).write_text(text)
    print(text)
    logger.info("event_position: roles=%s", sorted(result["role"]))
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
