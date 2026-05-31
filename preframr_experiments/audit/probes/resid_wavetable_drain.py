"""Validation: does the WAVETABLE codebook pass (v0.38.0, gate ``wavetable_pass``)
actually DRAIN the codebook-able ORN-RESID notes, BYTE-EXACT?

For each dominant engine, sample a few tunes and parse twice through the full
RegLogParser.parse path: ``wavetable_pass`` OFF then ON. Measure:
  - ORN-RESID notes (op==ORN_OP, subreg==TYPE, val==RESID) before vs after,
  - WAVETABLE_DEF/REF atoms emitted ON,
  - byte-exact: register_state(OFF) == register_state(ON)  (the isolation oracle).
Drain% should approach the recurrence probe's codebook-able fraction; any
register_state mismatch is a corruption bug (must be zero).

Usage: resid_wavetable_drain.py [n_tunes_per_engine]"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
sys.path.insert(0, os.path.dirname(__file__))
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.audit_primitives import register_state  # noqa: E402
from preframr_tokens.stfconstants import (  # noqa: E402
    ORN_OP, ORN_SUBREG_TYPE, ORN_TYPE_RESID,
    WAVETABLE_DEF_OP, WAVETABLE_REF_OP,
)
from parse_probes import parse_args  # noqa: E402
import sidid_cache  # noqa: E402
import numpy as np  # noqa: E402

ENGINES = ["Hermit/SidWizard_V1.x", "Music_Assembler", "GoatTracker_V2.x",
           "DMC", "MoN/FutureComposer", "JCH_NewPlayer"]

BASE = dict(skeleton_pass=True, trajectory_anchor_pass=True,
            stamp_pass=True, sweep_pass=True, patch_pass=True, held_arp=True)


def tunes_for_engine(engine, limit):
    labels = sidid_cache.load()
    out = []
    for sidpath in sorted(p for p, e in labels.items() if e == engine):
        dump = sidpath[:-4] + ".1.dump.parquet"
        if os.path.exists(dump):
            out.append(dump)
        if len(out) >= limit:
            break
    return out


def parse_df(dump, **flags):
    a = parse_args(**{**BASE, **flags})
    return next(RegLogParser(args=a).parse(dump, max_perm=1, require_pq=False, reparse=True), None)


def count_resid(df):
    if df is None or "op" not in getattr(df, "columns", []):
        return None
    op = df["op"].to_numpy()
    sub = df["subreg"].to_numpy()
    val = df["val"].to_numpy()
    resid = int(((op == ORN_OP) & (sub == ORN_SUBREG_TYPE) & (val == ORN_TYPE_RESID)).sum())
    wdef = int((op == WAVETABLE_DEF_OP).sum())
    wref = int((op == WAVETABLE_REF_OP).sum())
    return resid, wdef, wref


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"=== WAVETABLE drain validation ({n} tunes/engine) ===")
    print(f"{'engine':28s} {'RESID_off':>9} {'RESID_on':>8} {'drain%':>6} "
          f"{'wDEF':>5} {'wREF':>5} {'byte-exact':>11}")
    grand = defaultdict(int)
    for engine in ENGINES:
        off_tot = on_tot = wdef_tot = wref_tot = 0
        exact_ok = exact_bad = 0
        for dump in tunes_for_engine(engine, n):
            try:
                d_off = parse_df(dump, wavetable_pass=False)
                d_on = parse_df(dump, wavetable_pass=True)
            except Exception as e:  # noqa: BLE001
                print(f"  ! parse fail {os.path.basename(dump)}: {e}", file=sys.stderr)
                continue
            c_off = count_resid(d_off)
            c_on = count_resid(d_on)
            if c_off is None or c_on is None:
                continue
            off_tot += c_off[0]
            on_tot += c_on[0]
            wdef_tot += c_on[1]
            wref_tot += c_on[2]
            try:
                ok = np.array_equal(register_state(d_off), register_state(d_on))
            except Exception as e:  # noqa: BLE001
                print(f"  ! oracle fail {os.path.basename(dump)}: {e}", file=sys.stderr)
                ok = False
            exact_ok += int(ok)
            exact_bad += int(not ok)
        drain = 100 * (off_tot - on_tot) // max(off_tot, 1)
        tag = f"{exact_ok}/{exact_ok + exact_bad} OK" + ("" if not exact_bad else "  CORRUPT!")
        print(f"{engine:28s} {off_tot:9d} {on_tot:8d} {drain:5d}% "
              f"{wdef_tot:5d} {wref_tot:5d} {tag:>11}")
        grand["off"] += off_tot
        grand["on"] += on_tot
        grand["bad"] += exact_bad
    gd = 100 * (grand["off"] - grand["on"]) // max(grand["off"], 1)
    print(f"\nTOTAL drain {gd}%  ({grand['off']}->{grand['on']})  "
          f"corruptions={grand['bad']}")


if __name__ == "__main__":
    main()
