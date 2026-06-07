"""Can the macro find the trajectory's TRUE origin anchor intrinsically (no gate)?
Model each voice's per-frame pitch as a slowly-varying BASE (the note/melody) +
fast modulation (vibrato/arp). Recover the base by a median filter (suppresses
vibrato), take changepoints as intrinsic note anchors, and VALIDATE against gate-on
(corroboration, not mechanism): overlap = it found real onsets; anchors away from
gate-on = legato/slide changes gate would miss. Also report base-pitch entropy."""

import sys, math
from collections import Counter
import numpy as np
import pandas as pd

VB = {0: 0, 1: 7, 2: 14}
MEDIAN_W = 7  # frames; > vibrato period, < note duration
STEP = 1  # semitone level-change threshold


def per_frame(df, base):
    """Return ordered (irq, semitone, gate_on) per frame for one voice."""
    flo, fhi, ctrl = base, base + 1, base + 4
    lo = hi = gate = 0
    frames = {}
    order = []
    for _, r in df[df["reg"].isin([flo, fhi, ctrl])].iterrows():
        irq, reg, val = int(r["irq"]), int(r["reg"]), int(r["val"])
        if irq not in frames:
            frames[irq] = [lo, hi, gate, False]
            order.append(irq)
        f = frames[irq]
        if reg == flo:
            lo = val
            f[0] = lo
        elif reg == fhi:
            hi = val
            f[1] = hi
        else:
            g = val & 1
            f[3] = bool(g and not gate)  # gate-on this frame
            gate = g
            f[2] = gate
    out = []
    for irq in order:
        lo_, hi_, g_, gon = frames[irq]
        fr = (hi_ << 8) | lo_
        sem = round(12 * math.log2(fr)) if fr > 0 else None
        out.append((irq, sem, gon))
    return out


def analyze(path):
    df = pd.read_parquet(path)
    df = df[df["chipno"] == 0].sort_values("clock")
    print(f"\n### {path.split('/')[-1]}")
    for v, base in VB.items():
        rows = per_frame(df, base)
        sem = np.array([s if s is not None else np.nan for _, s, _ in rows], float)
        gate_on = np.array([g for _, _, g in rows], bool)
        valid = ~np.isnan(sem)
        if valid.sum() < 50:
            continue
        # median-filter the pitch (vibrato suppression) over valid frames
        s = sem.copy()
        # forward-fill NaN for filtering
        idx = np.where(valid)[0]
        s = np.interp(np.arange(len(s)), idx, sem[idx])
        med = np.array(
            [
                np.median(s[max(0, i - MEDIAN_W // 2) : i + MEDIAN_W // 2 + 1])
                for i in range(len(s))
            ]
        )
        # hysteresis level tracker: a new base level must be HELD >= MIN_HOLD frames
        # within BAND, else it's vibrato/arp/slide modulation, not a note anchor.
        BAND, MIN_HOLD = 1.5, 4
        level = med[0]
        cand, cand_start, cand_run = None, 0, 0
        anchor_frames, anchor_vals = [], []
        for t in range(1, len(med)):
            p = med[t]
            if abs(p - level) <= BAND:
                cand = None
                continue
            if cand is None or abs(p - cand) > BAND:
                cand, cand_start, cand_run = p, t, 1
            else:
                cand_run += 1
            if cand_run >= MIN_HOLD:
                level = round(np.median(med[cand_start : t + 1]))
                anchor_frames.append(cand_start)
                anchor_vals.append(int(level))
                cand = None
        anchor_frames = np.array(anchor_frames)
        gate_on_frames = np.where(gate_on)[0]
        if len(anchor_frames) < 5 or len(gate_on_frames) < 5:
            continue
        # corroboration: anchor within +-1 frame of a gate-on
        gset = set(gate_on_frames)
        near = lambda fr: any((fr + d) in gset for d in (-1, 0, 1))
        anch_at_gate = sum(near(fr) for fr in anchor_frames)
        gate_with_anchor = sum(
            any(abs(int(fr) - int(a)) <= 1 for a in anchor_frames)
            for fr in gate_on_frames
        )
        # entropy of intrinsic base-pitch sequence (at anchors)
        c = Counter(anchor_vals)
        n = sum(c.values())
        H = -sum((x / n) * math.log2(x / n) for x in c.values())
        print(
            f"  voice {v}: intrinsic anchors={len(anchor_frames)}  gate-on={len(gate_on_frames)}"
        )
        print(
            f"    corroboration: {100*anch_at_gate/len(anchor_frames):.0f}% of intrinsic anchors are at a gate-on  "
            f"| {100*gate_with_anchor/len(gate_on_frames):.0f}% of gate-ons have an intrinsic anchor"
        )
        print(
            f"    legato (intrinsic anchor, NO gate retrigger): {len(anchor_frames)-anch_at_gate} "
            f"({100*(len(anchor_frames)-anch_at_gate)/len(anchor_frames):.0f}%)"
        )
        print(f"    base-pitch entropy {H:.2f}b ({len(c)} pitches)")


for p in sys.argv[1:]:
    analyze(p)
