"""Test the hypothesis: melody is gate-anchored (base note written at gate-on),
while sustain-phase frequency writes are arp/vibrato modulation. For each voice,
track the SID gate bit (control reg bit0) and the 16-bit frequency; classify every
frequency write as gate-anchored (same frame as a gate-on) vs sustain modulation;
compare entropy of the gate-on base-note sequence vs the full frequency stream."""

import sys, math
from collections import Counter
import pandas as pd

VOICE_BASE = {0: 0, 1: 7, 2: 14}


def semitone(f):
    return round(12 * math.log2(f)) if f > 0 else None


def analyze(path):
    df = pd.read_parquet(path)
    df = df[df["chipno"] == 0]
    print(f"\n### {path.split('/')[-1]}  ({len(df)} raw writes)")
    for v, base in VOICE_BASE.items():
        flo, fhi, ctrl = base + 0, base + 1, base + 4
        lo = hi = 0
        gate = 0
        gate_on_frames = set()
        # pass 1: find gate-on frames (gate 0->1)
        for _, r in df[df["reg"] == ctrl].iterrows():
            g = int(r["val"]) & 1
            if g and not gate:
                gate_on_frames.add(int(r["irq"]))
            gate = g
        # pass 2: walk freq writes, classify, collect base notes at gate-on
        gate = 0
        anchored = sustain = 0
        all_notes, gate_notes = [], []
        last_irq_note = None
        rows = df[df["reg"].isin([flo, fhi, ctrl])].sort_values("clock")
        cur_gate_on = False
        for _, r in rows.iterrows():
            reg, val, irq = int(r["reg"]), int(r["val"]), int(r["irq"])
            if reg == ctrl:
                g = val & 1
                cur_gate_on = bool(g and not gate)
                gate = g
                continue
            if reg == flo:
                lo = val
            else:
                hi = val
            s = semitone((hi << 8) | lo)
            if s is None:
                continue
            all_notes.append(s)
            if irq in gate_on_frames:
                anchored += 1
                # first freq value seen at this gate-on frame = base note
                if irq != last_irq_note:
                    gate_notes.append(s)
                    last_irq_note = irq
            else:
                sustain += 1
        tot = anchored + sustain
        if tot < 20:
            continue
        # entropy of full freq stream (collapse consecutive dups) vs gate-on notes
        coll = [all_notes[0]]
        for s in all_notes[1:]:
            if s != coll[-1]:
                coll.append(s)

        def ent(seq):
            c = Counter(seq)
            n = sum(c.values())
            return -sum((x / n) * math.log2(x / n) for x in c.values()), len(c), n

        Hall, kall, nall = ent(coll)
        Hg, kg, ng = ent(gate_notes) if gate_notes else (0, 0, 0)
        print(
            f"  voice {v}: gate-on notes={len(gate_on_frames)}  freq writes={tot}  "
            f"({100*anchored/tot:.0f}% at gate-on, {100*sustain/tot:.0f}% sustain-mod)  "
            f"freq-writes/note={tot/max(1,len(gate_on_frames)):.1f}"
        )
        print(
            f"           ENTROPY  full-freq-stream {Hall:.2f}b ({kall} pitches) "
            f"vs gate-on base notes {Hg:.2f}b ({kg} pitches, n={ng})"
        )


for p in sys.argv[1:]:
    analyze(p)
