"""Splice donor pitch + gate into host instrument program to produce an augmented SID dump."""

from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from preframr_tokens.stfconstants import (
    FC_LO_REG,
    MODE_VOL_REG,
    VOICES,
    VOICE_CTRL_REG,
    VOICE_REG_SIZE,
)

LOGGER = logging.getLogger(__name__)

_AD_REGS = frozenset(v * VOICE_REG_SIZE + 5 for v in range(VOICES))
_SR_REGS = frozenset(v * VOICE_REG_SIZE + 6 for v in range(VOICES))
_FREQ_HI_REGS = frozenset(v * VOICE_REG_SIZE + 1 for v in range(VOICES))


def _mean_nibble(vals: np.ndarray, shift: int) -> float:
    return float((vals >> shift & 0xF).mean()) if len(vals) else 0.0


def adsr_profile(df: pd.DataFrame) -> np.ndarray:
    """Mean (attack, decay, sustain, release) nibbles over the song's AD/SR writes."""
    ad = df.loc[df["reg"].isin(_AD_REGS), "val"].astype(int).to_numpy()
    sr = df.loc[df["reg"].isin(_SR_REGS), "val"].astype(int).to_numpy()
    return np.array(
        [
            _mean_nibble(ad, 4),
            _mean_nibble(ad, 0),
            _mean_nibble(sr, 4),
            _mean_nibble(sr, 0),
        ]
    )


def adsr_distance(donor_df: pd.DataFrame, host_df: pd.DataFrame) -> float:
    """Euclidean distance between donor/host ADSR profiles. Validated: large
    distance -> the splice (host ADSR + donor gate) decays to near-silence."""
    return float(np.linalg.norm(adsr_profile(donor_df) - adsr_profile(host_df)))


def freq_hi_overlap(donor_df: pd.DataFrame, host_df: pd.DataFrame) -> float:
    """Fraction of the donor's FREQ-HI range overlapping the host's -- a coarse
    pitch-range compatibility proxy (bass-into-lead -> ~0). Cheaper stand-in for
    the design's per-voice Bhattacharyya filter."""
    d = donor_df.loc[donor_df["reg"].isin(_FREQ_HI_REGS), "val"].astype(int)
    h = host_df.loc[host_df["reg"].isin(_FREQ_HI_REGS), "val"].astype(int)
    if d.empty or h.empty:
        return 0.0
    lo = max(int(d.min()), int(h.min()))
    hi = min(int(d.max()), int(h.max()))
    span = max(int(d.max()) - int(d.min()), 1)
    return max(0.0, hi - lo) / span


def is_eligible(
    donor_df: pd.DataFrame,
    host_df: pd.DataFrame,
    max_adsr_dist: float = 4.0,
    min_freq_overlap: float = 0.3,
) -> bool:
    """Melody-transfer eligibility gate: ADSR-similarity (validated) + coarse
    pitch-range overlap. Cuts the quiet/incoherent splices before rendering."""
    return (
        adsr_distance(donor_df, host_df) <= max_adsr_dist
        and freq_hi_overlap(donor_df, host_df) >= min_freq_overlap
    )


PITCH_OFFSETS = (0, 1)
PWM_OFFSETS = (2, 3)
ADSR_OFFSETS = (5, 6)
CTRL_OFFSET = 4
GATE_BIT_MASK = 0x01
WAVE_BITS_MASK = 0xFE

DONOR_REG_OFFSETS = PITCH_OFFSETS
HOST_REG_OFFSETS = PWM_OFFSETS + ADSR_OFFSETS

DONOR_REGS = frozenset(
    v * VOICE_REG_SIZE + off for v in range(VOICES) for off in DONOR_REG_OFFSETS
)
HOST_VOICE_REGS = frozenset(
    v * VOICE_REG_SIZE + off for v in range(VOICES) for off in HOST_REG_OFFSETS
)
CTRL_REGS = frozenset(VOICE_CTRL_REG.values())
FILTER_REGS = frozenset(range(FC_LO_REG, MODE_VOL_REG + 1))


def _frame_groups(df: pd.DataFrame) -> "OrderedDict[int, pd.DataFrame]":
    if df.empty:
        return OrderedDict()
    groups = OrderedDict()
    for irq_val, sub in df.groupby("irq", sort=False):
        groups[int(irq_val)] = sub
    return groups


def _last_val_per_reg(frame: pd.DataFrame, regs: Iterable[int]) -> dict[int, int]:
    out = {}
    sub = frame[frame["reg"].isin(list(regs))]
    if sub.empty:
        return out
    for reg, reg_sub in sub.groupby("reg", sort=False):
        out[int(reg)] = int(reg_sub["val"].iloc[-1])
    return out


def splice_dumps(donor_df: pd.DataFrame, host_df: pd.DataFrame) -> pd.DataFrame:
    donor_frames = _frame_groups(donor_df)
    host_frames = _frame_groups(host_df)
    if not donor_frames or not host_frames:
        raise ValueError("donor or host dump has no frames")

    n_pairs = min(len(donor_frames), len(host_frames))
    donor_items = list(donor_frames.items())[:n_pairs]
    host_items = list(host_frames.items())[:n_pairs]

    chipno = int(host_df["chipno"].iloc[0])
    prev_donor_pitch: dict[int, int] = {}
    prev_host_voice: dict[int, int] = {}
    prev_host_filter: dict[int, int] = {}
    prev_donor_ctrl: dict[int, int] = dict.fromkeys(CTRL_REGS, 0)
    prev_host_ctrl: dict[int, int] = dict.fromkeys(CTRL_REGS, 0)
    prev_merged_ctrl: dict[int, int] = dict.fromkeys(CTRL_REGS, 0)
    rows: list[tuple[int, int, int, int, int]] = []

    for (_, donor_frame), (host_irq, host_frame) in zip(donor_items, host_items):
        host_clock_min = int(host_frame["clock"].min())

        donor_pitch = _last_val_per_reg(donor_frame, DONOR_REGS)
        host_voice = _last_val_per_reg(host_frame, HOST_VOICE_REGS)
        host_filter = _last_val_per_reg(host_frame, FILTER_REGS)
        donor_ctrl_now = {
            **prev_donor_ctrl,
            **_last_val_per_reg(donor_frame, CTRL_REGS),
        }
        host_ctrl_now = {**prev_host_ctrl, **_last_val_per_reg(host_frame, CTRL_REGS)}
        merged_ctrl = {
            ctrl_reg: (donor_ctrl_now[ctrl_reg] & GATE_BIT_MASK)
            | (host_ctrl_now[ctrl_reg] & WAVE_BITS_MASK)
            for ctrl_reg in CTRL_REGS
        }

        for reg, val in donor_pitch.items():
            if prev_donor_pitch.get(reg) != val:
                rows.append((host_clock_min, host_irq, chipno, reg, val))
        for reg, val in host_voice.items():
            if prev_host_voice.get(reg) != val:
                rows.append((host_clock_min, host_irq, chipno, reg, val))
        for reg, val in host_filter.items():
            if prev_host_filter.get(reg) != val:
                rows.append((host_clock_min, host_irq, chipno, reg, val))
        for reg, val in merged_ctrl.items():
            if prev_merged_ctrl.get(reg) != val:
                rows.append((host_clock_min, host_irq, chipno, reg, val))

        prev_donor_pitch = {**prev_donor_pitch, **donor_pitch}
        prev_host_voice = {**prev_host_voice, **host_voice}
        prev_host_filter = {**prev_host_filter, **host_filter}
        prev_donor_ctrl = donor_ctrl_now
        prev_host_ctrl = host_ctrl_now
        prev_merged_ctrl = merged_ctrl

    out_df = pd.DataFrame(
        rows, columns=["clock", "irq", "chipno", "reg", "val"]
    ).astype(host_df.dtypes.to_dict())
    return out_df


def _load_dump(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"clock", "irq", "chipno", "reg", "val"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--skip-ineligible",
        action="store_true",
        help="Skip (exit 0, no output) if the pair fails the ADSR/pitch eligibility gate.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(message)s")

    donor_df = _load_dump(args.donor)
    host_df = _load_dump(args.host)
    dist = adsr_distance(donor_df, host_df)
    overlap = freq_hi_overlap(donor_df, host_df)
    eligible = is_eligible(donor_df, host_df)
    LOGGER.info(
        "donor=%s rows=%d host=%s rows=%d adsr_dist=%.2f freq_overlap=%.2f eligible=%s",
        args.donor.name,
        len(donor_df),
        args.host.name,
        len(host_df),
        dist,
        overlap,
        eligible,
    )
    if not eligible and args.skip_ineligible:
        LOGGER.info("skipping ineligible pair (no output written)")
        return 0

    spliced = splice_dumps(donor_df, host_df)
    LOGGER.info("spliced rows=%d", len(spliced))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    spliced.to_parquet(args.out, index=False)
    LOGGER.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
