"""Interval-reducibility probe: is the corpus's melodic pitch content more
concentrated as INTERVALS (relative) than as ABSOLUTE pitch? Parses dumps with the
full_macros pipeline (FREQ_TRAJ on), extracts per-register trajectory anchors
(V0/TERMINAL = pitch events) exactly as the model predicts them, converts to
semitones, and compares absolute-pitch vs successive-interval entropy. No training."""

from __future__ import annotations
import argparse, math, sys
from collections import Counter, defaultdict
import numpy as np

from preframr_tokens.blocks import glob_dumps, iter_voiced_blocks
from preframr_tokens.macros.motif_pass import _atoms_of
from preframr_tokens.reglogparser import RegLogParser
from preframr_tokens.stfconstants import (
    FREQ_TRAJ_OP, FT_SUBREG_FLAGS, FT_SUBREG_V0_HI, FT_SUBREG_V0_LO,
)
from preframr.args import add_args, apply_macro_flags_to_args


def anchors_per_reg(streams):
    """Per FREQ_TRAJ register, the ordered list of anchor freqs (HI<<8|LO after a
    FLAGS atom). Anchor = V0 (run/osc) or TERMINAL (ramp); both live at subreg 1/2."""
    per_reg = defaultdict(list)
    for atoms in streams:
        pend = {}  # reg -> {hi,lo} being assembled
        for a in atoms:
            op, reg, subreg, val = int(a[0]), int(a[1]), int(a[2]), int(a[3])
            if op != FREQ_TRAJ_OP:
                continue
            if subreg == FT_SUBREG_FLAGS:
                pend[reg] = {}
            elif subreg == FT_SUBREG_V0_HI:
                pend.setdefault(reg, {})["hi"] = val
            elif subreg == FT_SUBREG_V0_LO:
                p = pend.setdefault(reg, {})
                p["lo"] = val
                if "hi" in p:
                    per_reg[reg].append((p["hi"] << 8) | p["lo"])
                    pend[reg] = {}
    return per_reg


def to_semitones(freqs):
    """16-bit SID freq -> nearest-semitone index (const offset dropped; only the
    log-scale matters for interval entropy). Collapse consecutive duplicates (a held
    note is one event)."""
    sem = [round(12 * math.log2(f)) for f in freqs if f > 0]
    out = [sem[0]] if sem else []
    for s in sem[1:]:
        if s != out[-1]:
            out.append(s)
    return out


def stats(counter):
    N = sum(counter.values())
    ps = [c / N for c in counter.values()]
    H = -sum(p * math.log2(p) for p in ps if p > 0)
    return N, len(counter), H, 2 ** H, max(counter.values()) / N


def main():
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--motif-out"); ap.add_argument("--motif-k", type=int, default=0)
    ap.add_argument("--motif-min-count", type=int, default=0)
    ap.add_argument("--motif-min-composers", type=int, default=0)
    ap.add_argument("--motif-mine-version", type=int, default=1)
    args = ap.parse_args(sys.argv[1:])
    apply_macro_flags_to_args(args)
    args.motif_pass = False
    parser = RegLogParser(args, __import__("logging"))
    bp = RegLogParser(args, __import__("logging"))
    seq_len = getattr(args, "seq_len", 4096)
    streams = []
    for name in glob_dumps(args.reglogs, args.max_files, require_pq=False):
        try:
            for df in parser.parse(name, max_perm=1, require_pq=False, reparse=True):
                for v in iter_voiced_blocks(df, seq_len, bp, {}, stride=None):
                    if not v.empty:
                        streams.append(_atoms_of(v))
        except (AssertionError, ValueError, KeyError):
            pass

    per_reg = anchors_per_reg(streams)
    abs_c, int_c = Counter(), Counter()
    for reg, freqs in per_reg.items():
        sem = to_semitones(freqs)
        abs_c.update(sem)
        int_c.update(b - a for a, b in zip(sem, sem[1:]))

    print(f"blocks={len(streams)}  FREQ_TRAJ regs={len(per_reg)}  "
          f"anchor events={sum(len(v) for v in per_reg.values())}")
    for name, c in (("ABSOLUTE semitone", abs_c), ("INTERVAL (relative)", int_c)):
        N, k, H, eff, maj = stats(c)
        print(f"  {name:<22} N={N:>7} distinct={k:>4} entropy={H:>5.2f} bits  "
              f"eff_classes={eff:>5.0f}  majority_floor={maj:.3f}")
    if abs_c and int_c:
        _, _, Ha, ea, _ = stats(abs_c)
        _, _, Hi, ei, _ = stats(int_c)
        print(f"\n  interval encoding cuts entropy {Ha:.2f}->{Hi:.2f} bits "
              f"({Ha-Hi:+.2f}); effective vocab {ea:.0f}->{ei:.0f}")
        top = int_c.most_common(9)
        tot = sum(int_c.values())
        print("  top intervals (semitones): " +
              ", ".join(f"{k:+d}:{100*v/tot:.0f}%" for k, v in top))


if __name__ == "__main__":
    sys.exit(main())
