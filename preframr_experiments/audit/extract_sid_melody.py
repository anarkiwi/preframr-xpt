"""Extract per-(dump,voice) melodic onset-pitch sequences from raw SID dumps for the
melody data-gap ladder (design/melody_data_gap_ladder.md). Host-side (pandas; no torch).

Ladder levels (onset definition):
  L1 all-freq   : every freq-register change is an onset (ornament-laden; the high
                  predictability here is a vibrato/sustain-repeat ARTIFACT, not melody).
  L2 gateanchor : onset only at control-reg gate-on transitions (the musical note-ons).
  L3 dearp      : L2 + minimum note duration (>=4 frames) — collapse fast arps/retriggers.
With --lead, keep only the most-melodic voice per dump (max distinct gate-on pitches).
With --composer NAME, restrict to that composer's subdir (homogeneity test, e.g. DRAX).

Pitch is MIDI (semitone) from SID Fn (equal temperament). Feed the output json to
`melody_ladder` (measure on intervals — key-invariant — via to-intervals, see the doc).

Usage:
  python3 -m preframr_experiments.audit.extract_sid_melody \
      --dumps '<root>/*/*.dump.parquet' --level L2 --lead --out mini_L2lead.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math

import pandas as pd

CLOCK_RATE = 985248


def fn_to_midi(fn: int) -> int:
    if fn < 8:
        return 0
    hz = fn * CLOCK_RATE / 16777216.0
    if hz < 16:
        return 0
    m = round(69 + 12 * math.log2(hz / 440.0))
    return int(m) if 24 <= m <= 108 else 0


def _voice_onsets(d, level):
    """Return list of 3 per-voice onset sequences [(frame, midi), ...] at the level."""
    fc = int(d["irq"][d["irq"] > 0].mode().iloc[0]) if (d["irq"] > 0).any() else 19592
    reg = d["reg"].to_numpy()
    val = d["val"].to_numpy()
    clk = d["clock"].to_numpy()
    out = []
    for v in range(3):
        lo, hi, ctrl = v * 7, v * 7 + 1, v * 7 + 4
        cl = ch = gate = 0
        l1, l2 = [], []
        for i in range(len(reg)):
            r, x, fr = reg[i], int(val[i]), clk[i] // fc
            if r == lo and x != cl:
                cl = x
                m = fn_to_midi((ch << 8) | cl)
                if m:
                    l1.append((fr, m))
            elif r == hi and x != ch:
                ch = x
                m = fn_to_midi((ch << 8) | cl)
                if m:
                    l1.append((fr, m))
            elif r == ctrl:
                ng = x & 1
                if ng and not gate:
                    m = fn_to_midi((ch << 8) | cl)
                    if m:
                        l2.append((fr, m))
                gate = ng
        if level == "L1":
            seq = l1
        elif level == "L2":
            seq = l2
        else:  # L3 de-arp
            seq, last = [], -999
            for fr, m in l2:
                if fr - last >= 4:
                    seq.append((fr, m))
                    last = fr
        out.append(seq)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dumps", required=True, help="glob for *.dump.parquet")
    ap.add_argument("--level", choices=("L1", "L2", "L3"), default="L2")
    ap.add_argument("--lead", action="store_true", help="keep only most-melodic voice/dump")
    ap.add_argument("--composer", default=None, help="restrict to this composer subdir")
    ap.add_argument("--min-notes", type=int, default=12)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    files = sorted(glob.glob(cli.dumps))
    if cli.composer:
        files = [f for f in files if f"/{cli.composer}/" in f]
    seqs = []
    for di, f in enumerate(files):
        d = pd.read_parquet(f).sort_values("clock")
        vseqs = _voice_onsets(d, cli.level)
        if cli.lead:
            cand = [(len({m for _, m in s}), s) for s in vseqs if len(s) >= cli.min_notes]
            if not cand:
                continue
            _, s = max(cand, key=lambda z: z[0])
            seqs.append({"dump": di, "pitch": [m for _, m in s]})
        else:
            for s in vseqs:
                if len(s) >= cli.min_notes:
                    seqs.append({"dump": di, "pitch": [m for _, m in s]})
    json.dump({"seqs": seqs}, open(cli.out, "w"))
    n = sum(len(s["pitch"]) for s in seqs)
    print(f"{cli.out}: {len(seqs)} seqs, {n} onsets ({len(files)} dumps, level {cli.level})")


if __name__ == "__main__":
    main()
