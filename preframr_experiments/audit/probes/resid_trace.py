"""Per-RESID-note MECHANISM tracer for the RESID=0 loop. For a small sample, account for EVERY
unmodelled write: label each RESID note's driver (sidid) and the precise driver mechanism it leaks
through (percussion / noise-accent / freq-sweep / fast-run / wide-overflow / glissando / irregular),
co-reading the control register and the RAW freq word. Drives the iterate->refine->measure loop;
when a sample hits RESID=0 it is expanded. Output -> /scratch/tmp.

Usage:  resid_trace.py <fixtures|N|path[,path...]> [procs] [worst_n]
  fixtures = the documented canonical tunes (Baggis/Camerock JCH, Commando Hubbard, Trap Crowther)."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, glob, random, subprocess  # noqa: E401,E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.macros.skeleton_pass import SkeletonPass, _OFFSET_LIMIT  # noqa: E402
from parse_probes import parse_args  # noqa: E402

CORPUS = "/scratch/preframr/hvsc"
SIDID_CFG = "/scratch/anarkiwi/sidid/sidid.cfg"
FIXTURES = [
    f"{CORPUS}/MUSICIANS/G/Goto80/Baggis.1.dump.parquet",
    f"{CORPUS}/MUSICIANS/D/DRAX/Camerock.1.dump.parquet",
    f"{CORPUS}/GAMES/G-L/Gridtrap.1.dump.parquet",
]


def _args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


def driver_map(dirs):
    """dir -> {sid_basename(lower, no ext): driver} via one sidid scan per dir."""
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


def wf(ctrl):
    """One-letter waveform/role from a ctrl byte (-1 = unknown)."""
    if ctrl < 0:
        return "?"
    if ctrl & 0x08:
        return "T"  # test/HR
    bits = ""
    if ctrl & 0x80:
        bits += "N"
    if ctrl & 0x40:
        bits += "P"
    if ctrl & 0x20:
        bits += "S"
    if ctrl & 0x10:
        bits += "t"
    if not bits:
        return "-"  # gate/idle, no waveform
    return bits


def _rle(xs):
    """Run-length collapse consecutive duplicates -> (values, holds). Reveals held/stretched arps
    (JCH wave-delay holds each chord step N frames)."""
    vals, holds = [], []
    for x in xs:
        if vals and vals[-1] == x:
            holds[-1] += 1
        else:
            vals.append(x)
            holds.append(1)
    return vals, holds


def _period(xs, cap):
    """Shortest genuinely-repeating period <= cap (seen >=2x), or None."""
    n = len(xs)
    for p in range(1, min(cap, n // 2) + 1):
        if all(xs[i] == xs[i % p] for i in range(n)):
            return p
    return None


def _uniform_ramp(xs, tol):
    """True if successive diffs are ~constant (a linear ramp) within tol, len>=3, real slope."""
    if len(xs) < 3:
        return False
    diffs = [b - a for a, b in zip(xs, xs[1:])]
    if all(abs(d) <= tol for d in diffs):
        return False  # flat, not a ramp
    return (max(diffs) - min(diffs)) <= tol


def classify(rec):
    """rec = [(offset, ctrl, is_pitched, fn)]. Returns a precise mechanism label."""
    pitched = [(o, fn) for o, _c, m, fn in rec if m]
    noise = [1 for _o, c, _m, _fn in rec if c >= 0 and (c & 0x80) and not (c & 0x08)]
    test = [1 for _o, c, _m, _fn in rec if c >= 0 and (c & 0x08)]
    npf = len(pitched)
    if npf == 0:
        if noise:
            return "percussion: noise (no pitched frame)"
        if test:
            return "test/HR only (absorbable)"
        return "percussion: non-noise (no pitched frame)"
    poff = [o for o, _fn in pitched]
    pfn = [fn for _o, fn in pitched]
    contam = (len(rec) - npf) > 0
    suffix = " +noise/test" if contam else ""
    if npf <= 2:
        return "noise-accent: pitched core <=2" + suffix
    if any(abs(o) > _OFFSET_LIMIT for o in poff):
        # wide -- periodic? (wide arp/octave) else wide-irregular
        for p in range(1, len(poff) // 2 + 1):
            if all(poff[i] == poff[i % p] for i in range(len(poff))):
                return "wide periodic (ARP/OCTAVE overflow >24)" + suffix
        if _uniform_ramp(pfn, 64):
            return "freq-sweep/skydive (linear-freq ramp)" + suffix
        return "wide irregular (>24, aperiodic)" + suffix
    diffs = [b - a for a, b in zip(poff, poff[1:])]
    monotone = bool(diffs) and (all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs))
    if monotone and abs(poff[-1] - poff[0]) >= 2:
        if _uniform_ramp(pfn, 8):
            return "freq-slide (linear-freq ramp, semitone-accel)" + suffix
        return ("glissando/slide (uniform, noise-poisoned)" + suffix) if contam \
            else "glissando/slide (non-uniform rate)"
    if _period(poff, 8) is not None:
        return "arp/octave (periodic<=8, noise-poisoned)" + suffix
    # held/stretched arp: collapse wave-delay holds, then look for a chord cycle
    rvals, rholds = _rle(poff)
    rp = _period(rvals, 8)
    if rp is not None and len(set(rholds)) <= 2:
        return f"HELD-ARP (chord cycle p={rp}, hold~{max(set(rholds), key=rholds.count)})" + suffix
    if _period(poff, 24) is not None:
        return "arp (period 9-24, ARP-cap overflow)" + suffix
    distinct = len(set(poff))
    span = max(poff) - min(poff)
    if distinct < 6 and span < 12:
        return "fast-melodic-run (should re-segment)" + suffix
    return "genuinely-irregular pitched" + suffix


def trace(paths, dmap):
    by_drv_mech = defaultdict(Counter)   # driver -> mech -> count
    worst = []                            # (len, driver, comp, tune, mech, rec)
    a = _args()
    for p in paths:
        seg = p.split("/MUSICIANS/", 1)
        comp = seg[-1].split("/")[1] if len(seg) > 1 else p.split("/")[-2]
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
        for _reg, is_resid, _note, _onset, rec in recs:
            if not is_resid or not rec:
                continue
            mech = classify(rec)
            by_drv_mech[drv][mech] += 1
            worst.append((len(rec), drv, comp, tune, mech, rec))
    worst.sort(key=lambda x: -x[0])
    return by_drv_mech, worst[:60]


def _worker(args):
    paths, dmap = args
    return trace(paths, dmap)


if __name__ == "__main__":
    import multiprocessing as mp

    spec = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    worst_n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    random.seed(13)
    if spec == "fixtures":
        sample = [p for p in FIXTURES if os.path.exists(p)]
    elif "/" in spec:
        sample = [p for p in spec.split(",") if os.path.exists(p)]
    else:
        all_dumps = glob.glob(f"{CORPUS}/MUSICIANS/*/*/*.1.dump.parquet")
        by_dir = defaultdict(list)
        for d in all_dumps:
            by_dir[os.path.dirname(d)].append(d)
        sample = [random.choice(v) for v in by_dir.values()]
        random.shuffle(sample)
        sample = sample[: int(spec)]
    print(f"tracing {len(sample)} dumps", file=sys.stderr, flush=True)
    dmap = driver_map([os.path.dirname(p) for p in sample])
    shards = [sample[i::procs] for i in range(procs)]
    shards = [(s, dmap) for s in shards if s]
    with mp.Pool(len(shards)) as pool:
        results = pool.map(_worker, shards)
    by_drv_mech = defaultdict(Counter)
    worst = []
    for bdm, w in results:
        for drv, mc in bdm.items():
            by_drv_mech[drv].update(mc)
        worst.extend(w)
    worst.sort(key=lambda x: -x[0])

    tot = sum(sum(mc.values()) for mc in by_drv_mech.values())
    print(f"\n=== RESID notes = {tot} across {len(by_drv_mech)} driver(s) ===")
    for drv in sorted(by_drv_mech, key=lambda d: -sum(by_drv_mech[d].values())):
        dtot = sum(by_drv_mech[drv].values())
        print(f"\n--- {drv}  (RESID notes={dtot}) ---")
        for mech, c in by_drv_mech[drv].most_common():
            print(f"   {c:5d} ({100*c//max(dtot,1):2d}%)  {mech}")

    print(f"\n=== {worst_n} worst (longest) RESID notes -- frame detail (off:wf) ===")
    for ln, drv, comp, tune, mech, rec in worst[:worst_n]:
        cells = " ".join(f"{o}:{wf(c)}" for o, c, _m, _fn in rec[:24])
        print(f"\n[{ln}f] {drv} {comp}/{tune}  <{mech}>")
        print(f"   {cells}")
        print(f"   fn: {[fn for _o, _c, _m, fn in rec[:24]]}")
