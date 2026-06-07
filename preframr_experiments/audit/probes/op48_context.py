"""Classify op48 FREQ_ONSET writes by trajectory context: why does the trajectory
pass leave them as residual instead of folding them into an op45 V0 trajectory?
Walks the op-tagged post-pass block dfs, frame-indexed, per voice-0 freq-lo reg."""

import glob
from collections import Counter

import pandas as pd

FRAME_REG, DELAY_REG = -128, -127
FREQLO = {0, 7, 14}  # voice freq low byte
TRAJ_OP, ONSET_OP, SET_OP = 45, 48, 0


def main():
    files = sorted(
        glob.glob(
            "/scratch/tmp/preframr_no_unigram_clean/results/melody_no_unigram_mini/"
            "no_unigram/seed0/eval/*/*.0.parquet"
        )
    )
    gap_prev = Counter()  # frames since previous freq event on same reg
    neighbor = Counter()  # op of nearest freq neighbor on same reg
    frame_multi = Counter()  # how many freq writes share the op48's frame on that reg
    singleton = Counter()  # is the op48 the only freq event within +/-2 frames?
    total = 0
    for f in files:
        df = pd.read_parquet(f)
        regs = df["reg"].to_numpy()
        ops = df["op"].to_numpy()
        vals = df["val"].to_numpy()
        frame = 0
        # per-reg list of (frame, op) freq events
        events = {r: [] for r in FREQLO}
        # also record per-row frame for op48 rows
        op48_rows = []
        for i in range(len(df)):
            r = int(regs[i])
            if r == FRAME_REG:
                frame += 1
                continue
            if r == DELAY_REG:
                frame += int(vals[i]) if int(vals[i]) > 0 else 1
                continue
            if r in FREQLO and int(ops[i]) in (TRAJ_OP, ONSET_OP, SET_OP):
                events[r].append((frame, int(ops[i])))
                if int(ops[i]) == ONSET_OP:
                    op48_rows.append((r, len(events[r]) - 1))
        for r, idx in op48_rows:
            total += 1
            ev = events[r]
            fr, _ = ev[idx]
            # neighbors on same reg
            prev_same = ev[idx - 1] if idx > 0 else None
            next_same = ev[idx + 1] if idx + 1 < len(ev) else None
            gp = (fr - prev_same[0]) if prev_same else 999
            gn = (next_same[0] - fr) if next_same else 999
            gap_prev[min(gp, 6) if gp < 999 else "none"] += 1
            # same-frame multiplicity on this reg
            same_frame = sum(1 for (ff, _o) in ev if ff == fr)
            frame_multi[min(same_frame, 4)] += 1
            # nearest neighbor op (whichever is closer in frames)
            cand = []
            if prev_same:
                cand.append((gp, prev_same[1]))
            if next_same:
                cand.append((gn, next_same[1]))
            if cand:
                cand.sort()
                neighbor[
                    {45: "op45_traj", 48: "op48_onset", 0: "op0_set"}.get(
                        cand[0][1], cand[0][1]
                    )
                ] += 1
            else:
                neighbor["alone_on_reg"] += 1
            # isolated within +/-2 frames (no other freq event on reg)?
            near = sum(1 for (ff, _o) in ev if ff != fr and abs(ff - fr) <= 2)
            singleton["isolated(+/-2f)" if near == 0 else "in_cluster"] += 1
    print(f"op48 freq-lo writes classified: {total}\n")
    print("gap to previous freq event on same reg (frames; capped at 6):")
    for k in sorted(gap_prev, key=lambda x: (x == "none", x)):
        print(f"  {k:>5}: {gap_prev[k]:5d} ({100*gap_prev[k]/total:.0f}%)")
    print("\nnearest same-reg freq neighbor op:")
    for k, v in neighbor.most_common():
        print(f"  {k:>12}: {v:5d} ({100*v/total:.0f}%)")
    print(
        "\nfreq writes sharing the op48's frame on that reg (sub-frame multiplicity):"
    )
    for k, v in sorted(frame_multi.items()):
        print(f"  {k}: {v:5d} ({100*v/total:.0f}%)")
    print("\nisolation within +/-2 frames:")
    for k, v in singleton.most_common():
        print(f"  {k:>14}: {v:5d} ({100*v/total:.0f}%)")


if __name__ == "__main__":
    main()
