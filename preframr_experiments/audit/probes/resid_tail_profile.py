"""Post-WAVETABLE residue map: parse a broad random sample twice (wavetable_pass OFF then ON)
through the FULL RegLogParser.parse, and for each engine report how much ORN-RESID drained,
byte-exactly, AND classify what SURVIVES as RESID with wavetable ON -- the genuine tail to close
for RESID=0. Survivor classes (offset-only, since the ORN atom stores note-relative offsets):
  SWEEP   -- constant-delta ramp (>=3 frames)            (SweepPass / wavetable sweep gap)
  PERIOD  -- period<=8 cycle after onset-strip           (held-ARP / codebook should catch)
  RECUR   -- exact offset seq appears >=2x in this tune  (codebook TUNING bug: should have drained)
  STRUCT  -- has a loop body (factorise loop<len) but unique -> inline one-shot SHOULD emit
  SHORT   -- core length < 2                             (too small for a program)
  FLAT    -- unique, non-periodic, no loop body          (genuine flat one-shot: needs inline emit)

Usage: resid_tail_profile.py <N|paths> [procs]"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob  # noqa: E402
import random  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
sys.path.insert(0, os.path.dirname(__file__))
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.audit_primitives import register_state  # noqa: E402
from preframr_tokens.macros.wavetable import factorise  # noqa: E402
from preframr_tokens.stfconstants import (  # noqa: E402
    ORN_OP, ORN_SUBREG_TYPE, ORN_SUBREG_P1, ORN_SUBREG_P2, ORN_TYPE_RESID,
    WAVETABLE_DEF_OP, WAVETABLE_REF_OP, FREQ_TRAJ_REGS,
)
from parse_probes import parse_args  # noqa: E402
import sidid_cache  # noqa: E402
import numpy as np  # noqa: E402

CORPUS = "/scratch/preframr/hvsc"
BASE = dict(skeleton_pass=True, trajectory_anchor_pass=True,
            stamp_pass=True, sweep_pass=True, patch_pass=True, held_arp=True)
_FREQ = {int(r) for r in FREQ_TRAJ_REGS}


def parse_df(dump, **flags):
    a = parse_args(**{**BASE, **flags})
    return next(RegLogParser(args=a).parse(dump, max_perm=1, require_pq=False, reparse=True), None)


def resid_offsets(df):
    """Each surviving ORN-RESID note's note-relative offset tuple from a parsed df."""
    if df is None or "op" not in getattr(df, "columns", []):
        return []
    op = df["op"].to_numpy(); reg = df["reg"].to_numpy()
    sub = df["subreg"].to_numpy(); val = df["val"].to_numpy()
    n = len(df); out = []; i = 0
    while i < n:
        if not (op[i] == ORN_OP and sub[i] == ORN_SUBREG_TYPE
                and val[i] == ORN_TYPE_RESID and int(reg[i]) in _FREQ):
            i += 1
            continue
        r = int(reg[i]); offs = []; length = None; j = i + 1
        while j < n and op[j] == ORN_OP and int(reg[j]) == r:
            if sub[j] == ORN_SUBREG_P2:
                length = int(val[j]) & 0xFFFF
            elif sub[j] == ORN_SUBREG_P1:
                v = int(val[j]) & 0xFF
                offs.append(v if v < 128 else v - 256)
            j += 1
        i = j
        if length is not None and len(offs) == length:
            out.append(tuple(offs))
    return out


def _const_delta(xs, tol=1):
    if len(xs) < 3:
        return False
    d = [b - a for a, b in zip(xs, xs[1:])]
    nz = [x for x in d if abs(x) > tol]
    return len(nz) >= 2 and (max(d) - min(d)) <= max(2, abs(sum(nz) // len(nz)) // 4)


def _period(xs, onset_strip=2, maxp=8):
    for strip in range(onset_strip + 1):
        s = xs[strip:]
        if len(s) < 6:
            continue
        n = len(s)
        for p in range(1, min(maxp, n // 2) + 1):
            if all(s[k] == s[k % p] for k in range(n)):
                return True
    return False


def classify(seq, tune_counts):
    if tune_counts[seq] >= 2:
        return "RECUR"
    if _const_delta(list(seq)):
        return "SWEEP"
    if _period(list(seq)):
        return "PERIOD"
    if len(seq) < 2:
        return "SHORT"
    steps, loop = factorise(list(seq))
    return "STRUCT" if loop < len(steps) else "FLAT"


def analyze(paths, dmap):
    drain = defaultdict(lambda: {"off": 0, "on": 0, "bad": 0, "tunes": 0})
    tail = defaultdict(Counter)
    for p in paths:
        tune = os.path.basename(p).split(".")[0].lower()
        eng = dmap.get(os.path.dirname(p), {}).get(tune, "?")
        try:
            d_off = parse_df(p, wavetable_pass=False)
            d_on = parse_df(p, wavetable_pass=True)
        except Exception:
            continue
        off = resid_offsets(d_off)
        on = resid_offsets(d_on)
        if not off and not on:
            continue
        drain[eng]["off"] += len(off)
        drain[eng]["on"] += len(on)
        drain[eng]["tunes"] += 1
        try:
            if not np.array_equal(register_state(d_off), register_state(d_on)):
                drain[eng]["bad"] += 1
        except Exception:
            drain[eng]["bad"] += 1
        tc = Counter(on)
        for seq in on:
            tail[eng][classify(seq, tc)] += 1
    return dict(drain), {k: dict(v) for k, v in tail.items()}


def _worker(a):
    return analyze(*a)


def main():
    import multiprocessing as mp
    spec = sys.argv[1] if len(sys.argv) > 1 else "150"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    random.seed(13)
    if "/" in spec:
        sample = [p for p in spec.split(",") if os.path.exists(p)]
    else:
        by_dir = defaultdict(list)
        for d in glob.glob(f"{CORPUS}/MUSICIANS/*/*/*.1.dump.parquet"):
            by_dir[os.path.dirname(d)].append(d)
        sample = [random.choice(v) for v in by_dir.values()]
        random.shuffle(sample)
        sample = sample[: int(spec)]
    print(f"tail-profiling {len(sample)} dumps", file=sys.stderr, flush=True)
    import sidid_cache as sc
    full = sc.by_dir()
    dmap = {os.path.dirname(p): full.get(os.path.dirname(p), {}) for p in sample}
    shards = [(sample[i::procs], dmap) for i in range(procs)]
    shards = [s for s in shards if s[0]]
    with mp.Pool(len(shards)) as pool:
        res = pool.map(_worker, shards)
    drain = defaultdict(lambda: {"off": 0, "on": 0, "bad": 0, "tunes": 0})
    tail = defaultdict(Counter)
    for dr, tl in res:
        for e, d in dr.items():
            for k in d:
                drain[e][k] += d[k]
        for e, t in tl.items():
            tail[e].update(t)

    print("\n=== WAVETABLE DRAIN + RESID TAIL (wavetable_pass ON) ===")
    classes = ["RECUR", "STRUCT", "PERIOD", "SWEEP", "SHORT", "FLAT"]
    hdr = f"{'engine':26s} {'off':>6} {'on':>6} {'drain%':>6} {'bad':>4}  " + \
          " ".join(f"{c:>6}" for c in classes)
    print(hdr)
    goff = gon = gbad = 0
    for e in sorted(drain, key=lambda e: -drain[e]["off"]):
        d = drain[e]
        if d["off"] < 20:
            continue
        dr = 100 * (d["off"] - d["on"]) // max(d["off"], 1)
        cells = " ".join(f"{tail[e].get(c, 0):>6}" for c in classes)
        print(f"{e:26s} {d['off']:>6} {d['on']:>6} {dr:>5}% {d['bad']:>4}  {cells}")
        goff += d["off"]; gon += d["on"]; gbad += d["bad"]
    gd = 100 * (goff - gon) // max(goff, 1)
    allt = Counter()
    for t in tail.values():
        allt.update(t)
    print(f"\nTOTAL off={goff} on={gon} drain={gd}% corruptions={gbad}")
    print("TAIL classes:", dict(allt.most_common()))


if __name__ == "__main__":
    main()
