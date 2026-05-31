"""Byte-exactness hunt for WAVETABLE: the survey found ~3% of verified tunes fail the isolation oracle
(register_state OFF != ON) -- a correctness bug that BLOCKS RESID=0 (W3 makes the pass total). This probe
FULL-verifies every sampled tune (parse OFF + ON, compare register_state) and LOGS each failing tune with
a divergence diagnostic (shapes + first differing frame/reg/value), so the failure can be root-caused.

Usage: resid_byte_exact_hunt.py <N|all|paths> [procs]"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
sys.path.insert(0, os.path.dirname(__file__))
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.audit_primitives import register_state  # noqa: E402
from parse_probes import parse_args  # noqa: E402
import sidid_cache  # noqa: E402
import numpy as np  # noqa: E402

CORPUS = "/scratch/preframr/hvsc"
BASE = dict(skeleton_pass=True, trajectory_anchor_pass=True,
            stamp_pass=True, sweep_pass=True, patch_pass=True, held_arp=True)
_DMAP = {}


def parse_df(dump, **flags):
    a = parse_args(**{**BASE, **flags})
    return next(RegLogParser(args=a).parse(dump, max_perm=1, require_pq=False, reparse=True), None)


def first_diff(a, b):
    if a.shape != b.shape:
        return f"shape {a.shape} != {b.shape}"
    rows = np.where(np.any(a != b, axis=1))[0]
    if len(rows) == 0:
        return "equal"
    r = int(rows[0])
    cols = np.where(a[r] != b[r])[0]
    c = int(cols[0])
    return f"frame {r} reg {c}: off={int(a[r, c])} on={int(b[r, c])} (+{len(rows)} diff frames)"


def analyze(args):
    paths, wid = args
    bad = []
    n_ok = n_bad = n_err = 0
    for k, p in enumerate(paths):
        if (k + 1) % 100 == 0:
            print(f"[w{wid}] {k+1}/{len(paths)} ok={n_ok} bad={n_bad} err={n_err}",
                  file=sys.stderr, flush=True)
        tune = os.path.basename(p).split(".")[0].lower()
        eng = _DMAP.get(os.path.dirname(p), {}).get(tune, "?")
        try:
            d_off = parse_df(p, wavetable_pass=False)
            d_on = parse_df(p, wavetable_pass=True)
            if d_off is None and d_on is None:
                continue
            if d_off is None or d_on is None:
                n_bad += 1
                bad.append((p, eng, f"one-None off={d_off is not None} on={d_on is not None}"))
                continue
            so, sn = register_state(d_off), register_state(d_on)
            if np.array_equal(so, sn):
                n_ok += 1
            else:
                n_bad += 1
                bad.append((p, eng, first_diff(so, sn)))
        except Exception as e:  # noqa: BLE001
            n_err += 1
            bad.append((p, eng, f"EXC {type(e).__name__}: {str(e)[:80]}"))
    return bad, n_ok, n_bad, n_err


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "3000"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 22
    random.seed(13)
    if "/" in spec:
        sample = [p for p in spec.split(",") if os.path.exists(p)]
    else:
        sample = []
        for top in ("MUSICIANS", "DEMOS", "GAMES"):
            sample += glob.glob(f"{CORPUS}/{top}/**/*.1.dump.parquet", recursive=True)
        sample.sort()
        random.shuffle(sample)
        if spec != "all":
            sample = sample[: int(spec)]
    print(f"byte-exact hunt over {len(sample)} tunes on {procs} procs", file=sys.stderr, flush=True)
    global _DMAP
    full = sidid_cache.by_dir()
    _DMAP = {os.path.dirname(p): full.get(os.path.dirname(p), {}) for p in sample}
    n_chunks = max(procs, (len(sample) + 29) // 30)
    shards = [(sample[i::n_chunks], i) for i in range(n_chunks)]
    shards = [s for s in shards if s[0]]

    import multiprocessing as mp
    t0 = time.time()
    allbad = []
    tot_ok = tot_bad = tot_err = done = 0
    by_eng = defaultdict(lambda: [0, 0])
    with mp.Pool(min(procs, len(shards))) as pool:
        for bad, n_ok, n_bad, n_err in pool.imap_unordered(analyze, shards):
            allbad.extend(bad)
            tot_ok += n_ok; tot_bad += n_bad; tot_err += n_err
            done += n_ok + n_bad + n_err
            for p, eng, _ in bad:
                by_eng[eng][1] += 1
            el = int(time.time() - t0)
            print(f"[parent] {done}/{len(sample)} ok={tot_ok} BAD={tot_bad} err={tot_err} {el}s",
                  file=sys.stderr, flush=True)
    print(f"\n=== BYTE-EXACT HUNT: {len(sample)} tunes, OK={tot_ok} BAD={tot_bad} ERR={tot_err} "
          f"({100*tot_bad/max(tot_ok+tot_bad,1):.1f}% non-exact) ===")
    eng_bad = defaultdict(int)
    for p, eng, _ in allbad:
        eng_bad[eng] += 1
    print("by engine (bad count):", dict(sorted(eng_bad.items(), key=lambda x: -x[1])))
    print("\n=== failing tunes ===")
    for p, eng, diag in allbad[:80]:
        print(f"  {eng:26s} {diag:48s} {p}")


if __name__ == "__main__":
    main()
