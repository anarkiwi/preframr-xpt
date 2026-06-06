"""Root-cause one WAVETABLE byte-exact failure: parse a tune OFF and ON, find the first freq-register
frame where register_state diverges, and dump the token rows (OFF ORN-RESID vs ON WAVETABLE) for that
voice around that frame, plus the decoded per-frame freq on both sides, so the desync is visible.

Usage: resid_wt_trace.py <dump.parquet>"""

import os
import sys

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
sys.path.insert(0, os.path.dirname(__file__))
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.audit_primitives import register_state  # noqa: E402
from preframr_tokens.stfconstants import (  # noqa: E402
    ORN_OP,
    WAVETABLE_DEF_OP,
    WAVETABLE_STEP_OP,
    WAVETABLE_END_OP,
    WAVETABLE_REF_OP,
    SKEL_OP,
    FREQ_TRAJ_REGS,
)
from parse_probes import parse_args  # noqa: E402
import numpy as np  # noqa: E402

BASE = dict(
    skeleton_pass=True,
    trajectory_anchor_pass=True,
    stamp_pass=True,
    sweep_pass=True,
    patch_pass=True,
    held_arp=True,
)
OPNAME = {
    ORN_OP: "ORN",
    WAVETABLE_DEF_OP: "WT_DEF",
    WAVETABLE_STEP_OP: "WT_STEP",
    WAVETABLE_END_OP: "WT_END",
    WAVETABLE_REF_OP: "WT_REF",
    SKEL_OP: "SKEL",
}


def parse_df(dump, **flags):
    a = parse_args(**{**BASE, **flags})
    return next(
        RegLogParser(args=a).parse(dump, max_perm=1, require_pq=False, reparse=True),
        None,
    )


def dump_rows(df, label, lo, hi):
    print(f"\n--- {label} token rows [{lo}:{hi}] ---")
    op = df["op"].to_numpy()
    reg = df["reg"].to_numpy()
    sub = df["subreg"].to_numpy()
    val = df["val"].to_numpy()
    for i in range(max(0, lo), min(len(df), hi)):
        nm = OPNAME.get(int(op[i]), str(int(op[i])))
        print(
            f"  {i:6d} op={nm:7s} reg={int(reg[i]):3d} sub={int(sub[i]):3d} val={int(val[i])}"
        )


def main():
    dump = sys.argv[1]
    print(f"tune: {dump}")
    d_off = parse_df(dump, wavetable_pass=False)
    d_on = parse_df(dump, wavetable_pass=True)
    so, sn = register_state(d_off), register_state(d_on)
    print(f"off frames={so.shape[0]} on frames={sn.shape[0]}")
    n = min(so.shape[0], sn.shape[0])
    diff = np.where(np.any(so[:n] != sn[:n], axis=1))[0]
    if len(diff) == 0:
        print(
            "no diff in overlap; length differs only"
            if so.shape != sn.shape
            else "EQUAL"
        )
        return
    f0 = int(diff[0])
    cols = np.where(so[f0] != sn[f0])[0]
    print(f"first diff frame {f0}, regs {list(cols)}; total diff frames={len(diff)}")
    for r in list(cols)[:3]:
        print(
            f"  reg {r}: off={so[f0, r]} on={sn[f0, r]}  "
            f"(prev off={so[f0-1, r]} on={sn[f0-1, r]})"
        )
    # show the freq trajectory around the diff for the first diverging reg
    r = int(cols[0])
    print(f"\n=== reg {r} per-frame [{f0-3}:{f0+8}] OFF vs ON ===")
    for f in range(max(0, f0 - 3), min(n, f0 + 8)):
        mark = "  <-- DIFF" if so[f, r] != sn[f, r] else ""
        print(f"  frame {f:6d}  off={so[f, r]:6d}  on={sn[f, r]:6d}{mark}")
    # locate the ON token rows near this region: find WT_REF/WT_DEF rows for this voice
    print(f"\n=== ON wavetable tokens (reg {r} family) — first 6 WT_REF/DEF/END ===")
    op = d_on["op"].to_numpy()
    wt = np.where(
        np.isin(
            op,
            [WAVETABLE_DEF_OP, WAVETABLE_STEP_OP, WAVETABLE_END_OP, WAVETABLE_REF_OP],
        )
    )[0]
    if len(wt):
        dump_rows(d_on, "ON", int(wt[0]) - 2, int(wt[0]) + 30)


if __name__ == "__main__":
    main()
