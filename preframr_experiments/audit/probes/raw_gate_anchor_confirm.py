"""Confirm in the ORIGINAL raw dump that frequency CHANGES are anchored to gate-on.
Per voice, at frame granularity: gate state (control bit0), end-of-frame 16-bit freq,
and |semitone delta| vs previous frame. Tests: (1) what fraction of gate-on events
carry a fresh freq write in the same frame; (2) are the large (note-level) freq
changes concentrated at gate-on vs sustain frames."""

import sys, math
from collections import defaultdict
import numpy as np
import pandas as pd

VB = {0: 0, 1: 7, 2: 14}


def analyze(path):
    df = pd.read_parquet(path)
    df = df[df["chipno"] == 0].sort_values("clock")
    print(f"\n### {path.split('/')[-1]}")
    for v, base in VB.items():
        flo, fhi, ctrl = base, base + 1, base + 4
        # per-frame (irq) end-of-frame state
        frames = {}  # irq -> dict(lo,hi,gate, freq_written)
        lo = hi = gate = 0
        order = []
        for _, r in df[df["reg"].isin([flo, fhi, ctrl])].iterrows():
            irq, reg, val = int(r["irq"]), int(r["reg"]), int(r["val"])
            if irq not in frames:
                frames[irq] = {"freq_written": False, "gate": gate, "lo": lo, "hi": hi}
                order.append(irq)
            f = frames[irq]
            if reg == flo:
                lo = val
                f["lo"] = val
                f["freq_written"] = True
            elif reg == fhi:
                hi = val
                f["hi"] = val
                f["freq_written"] = True
            else:
                gate = val & 1
                f["gate"] = gate
        if len(order) < 20:
            continue
        # walk frames in time order; detect gate-on, freq, semitone delta
        prev_gate = 0
        prev_sem = None
        gate_on_total = gate_on_with_freq = 0
        d_gateon, d_sustain = [], []
        for irq in order:
            f = frames[irq]
            g = f["gate"]
            freq = (f["hi"] << 8) | f["lo"]
            sem = round(12 * math.log2(freq)) if freq > 0 else None
            is_gate_on = g and not prev_gate
            if is_gate_on:
                gate_on_total += 1
                if f["freq_written"]:
                    gate_on_with_freq += 1
            if sem is not None and prev_sem is not None:
                d = abs(sem - prev_sem)
                (d_gateon if is_gate_on else d_sustain).append(d)
            prev_gate = g
            if sem is not None:
                prev_sem = sem
        dg, ds = np.array(d_gateon), np.array(d_sustain)
        if gate_on_total < 10:
            continue
        print(
            f"  voice {v}: gate-on events={gate_on_total}  "
            f"with same-frame freq write={100*gate_on_with_freq/gate_on_total:.0f}%"
        )
        print(
            f"           |Δsemitone| at gate-on: mean={dg.mean():.1f} median={np.median(dg):.0f} "
            f">0: {100*(dg>0).mean():.0f}%  (n={len(dg)})"
        )
        print(
            f"           |Δsemitone| sustain:    mean={ds.mean():.1f} median={np.median(ds):.0f} "
            f">0: {100*(ds>0).mean():.0f}%  (n={len(ds)})"
        )


for p in sys.argv[1:]:
    analyze(p)
