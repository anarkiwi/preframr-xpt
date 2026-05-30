"""AUTO reverse-engineering of player techniques. Generalises the manual loop (sidid label ->
resid_trace frame detail -> recognise the parametric mechanism) into one tool: fit every RESID note
to a small library of driver-native parametric models, pick the best, and AGGREGATE PER ENGINE
(sidid) -> an auto-generated per-engine technique profile that says which primitive each player's
RESID reduces to. One run profiles every engine in the sample at once. The fitters ARE the
driver-abstraction recognisers; new fitters = new recognised techniques.

Model library (driver-native, waveform/pitch-domain aware):
  SWEEP_loop  -- constant raw-freq delta per frame, repeating (SoundMonitor looping arp)
  SWEEP_once  -- constant raw-freq delta, one-shot (portamento / skydive; noise => swept drum)
  ARP         -- note-relative offset cycle period<=8, NOISE-INCLUSIVE + onset-strip + RLE-collapse
                 (System6581 chord-arp; wavetable arps on any waveform)
  ARP_accent  -- a pitched ARP with a PERIODIC non-pitched (gate-off/noise-tik) accent interleaved
  PERC_sweep  -- no pitched frame + a freq sweep (swept noise drum)
  PERC_other  -- no pitched frame, not a clean sweep (stamp/sample drum)
  UNRESOLVED  -- no fitter matched (a genuinely new technique -> add a fitter)

Usage:  resid_engine_profile.py <N|paths> [procs] [min_notes]"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, glob, random, subprocess  # noqa: E401,E402
from collections import defaultdict, Counter  # noqa: E402

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.macros.skeleton_pass import SkeletonPass  # noqa: E402
from parse_probes import parse_args  # noqa: E402

CORPUS = "/scratch/preframr/hvsc"
SIDID_CFG = "/scratch/anarkiwi/sidid/sidid.cfg"


def _args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


def _const_delta(xs, tol=1):
    """If xs is a ~constant-delta ramp (len>=3, real slope), return the delta, else None."""
    if len(xs) < 3:
        return None
    d = [b - a for a, b in zip(xs, xs[1:])]
    nz = [x for x in d if abs(x) > tol]
    if len(nz) < 2 or (max(d) - min(d)) > max(2, abs(sum(nz) // len(nz)) // 4):
        return None
    return round(sum(d) / len(d))


def _loop_period(xs, tol=2):
    """Shortest period the sequence repeats with (a looping sweep/cycle), else None."""
    n = len(xs)
    for p in range(2, n // 2 + 1):
        if all(abs(xs[i] - xs[i % p]) <= tol for i in range(n)):
            return p
    return None


def _rle(xs):
    out = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _arp_period(offs):
    for strip in (0, 1, 2):
        s = offs[strip:]
        if len(s) < 6:
            continue
        for seq in (s, _rle(s)):
            n = len(seq)
            for p in range(1, min(8, n // 2) + 1):
                if all(seq[i] == seq[i % p] for i in range(n)):
                    return p
    return None


def fit(rec):
    """Best-fit driver-native model for one RESID note. rec=[(offset,ctrl,is_pitched,fn)]."""
    fns = [fn for _o, _c, _m, fn in rec]
    offs = [o for o, _c, _m, _fn in rec]
    pitched = [m for _o, _c, m, _fn in rec]
    npitched = sum(pitched)
    non_pitched = len(rec) - npitched
    # looping/one-shot freq-domain sweep (SoundMonitor; portamento/skydive)
    fn_loop = _loop_period(fns)
    if fn_loop and _const_delta(fns[:fn_loop]) is not None:
        return "SWEEP_loop", f"d={_const_delta(fns[:fn_loop])},p={fn_loop}"
    delta = _const_delta(fns)
    if delta is not None:
        return ("PERC_sweep" if npitched == 0 else "SWEEP_once"), f"d={delta}"
    # note-relative arp (noise-inclusive)
    p = _arp_period(offs)
    if p is not None:
        # periodic non-pitched accent? (noise-tik / gate-off every cycle)
        if non_pitched and npitched >= 3:
            return "ARP_accent", f"p={p}"
        return "ARP", f"p={p}"
    if npitched == 0:
        return "PERC_other", ""
    # pitched-only arp after dropping non-pitched (control-aware)
    poff = [o for o, _c, m, _fn in rec if m]
    if _arp_period(poff) is not None:
        return "ARP_accent", f"p={_arp_period(poff)}"
    return "UNRESOLVED", ""


def driver_map(dirs):
    out = {}
    for d in sorted(set(dirs)):
        m = {}
        try:
            r = subprocess.run(["sidid", d], capture_output=True, text=True, timeout=120,
                               env={**os.environ, "SIDIDCFG": SIDID_CFG})
            for ln in r.stdout.splitlines():
                parts = ln.split()
                if len(parts) >= 2 and parts[0].lower().endswith(".sid"):
                    m[parts[0][:-4].lower()] = parts[-1]
        except Exception:
            pass
        out[d] = m
    return out


def analyze(paths, dmap):
    prof = defaultdict(Counter)          # engine -> mechanism -> count
    ex = defaultdict(lambda: defaultdict(list))  # engine -> mech -> param examples
    a = _args()
    for p in paths:
        tune = os.path.basename(p).split(".")[0]
        drv = dmap.get(os.path.dirname(p), {}).get(tune.lower(), "?")
        SkeletonPass._resid_diag = []
        try:
            parsed = next(RegLogParser(args=a).parse(p, max_perm=1, require_pq=False, reparse=True), None)
        except Exception:
            SkeletonPass._resid_diag = None
            continue
        recs = SkeletonPass._resid_diag or []
        SkeletonPass._resid_diag = None
        if parsed is None or "op" not in getattr(parsed, "columns", []):
            continue
        for _reg, isr, _n, _on, rec in recs:
            if not isr or not rec:
                continue
            mech, param = fit(rec)
            prof[drv][mech] += 1
            if param and len(ex[drv][mech]) < 5:
                ex[drv][mech].append(param)
    return prof, {k: {m: v for m, v in d.items()} for k, d in ex.items()}


def _worker(args):
    return analyze(*args)


if __name__ == "__main__":
    import multiprocessing as mp

    spec = sys.argv[1] if len(sys.argv) > 1 else "200"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    min_notes = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    random.seed(13)
    if "/" in spec:
        sample = [p for p in spec.split(",") if os.path.exists(p)]
    else:
        all_dumps = glob.glob(f"{CORPUS}/MUSICIANS/*/*/*.1.dump.parquet")
        by_dir = defaultdict(list)
        for d in all_dumps:
            by_dir[os.path.dirname(d)].append(d)
        sample = [random.choice(v) for v in by_dir.values()]
        random.shuffle(sample)
        sample = sample[: int(spec)]
    print(f"profiling engines across {len(sample)} dumps", file=sys.stderr, flush=True)
    dmap = driver_map([os.path.dirname(p) for p in sample])
    shards = [(sample[i::procs], dmap) for i in range(procs)]
    shards = [s for s in shards if s[0]]
    with mp.Pool(len(shards)) as pool:
        results = pool.map(_worker, shards)
    prof = defaultdict(Counter)
    ex = defaultdict(lambda: defaultdict(list))
    for pr, e in results:
        for drv, mc in pr.items():
            prof[drv].update(mc)
        for drv, md in e.items():
            for mech, ps in md.items():
                ex[drv][mech].extend(ps[: max(0, 5 - len(ex[drv][mech]))])

    print("\n=== PER-ENGINE TECHNIQUE PROFILE (auto reverse-engineered from RESID) ===")
    for drv in sorted(prof, key=lambda d: -sum(prof[d].values())):
        tot = sum(prof[drv].values())
        if tot < min_notes:
            continue
        un = prof[drv].get("UNRESOLVED", 0)
        print(f"\n--- {drv}  (RESID notes={tot}, UNRESOLVED {100*un//max(tot,1)}%) ---")
        for mech, c in prof[drv].most_common():
            exs = ",".join(ex[drv].get(mech, [])[:4])
            print(f"   {c:6d} ({100*c//max(tot,1):2d}%)  {mech:12s} {exs}")
