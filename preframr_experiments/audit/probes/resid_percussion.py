"""Algorithmic percussion detection from the token stream as a MUSICAL SCOPE. A drum is not a
waveform -- it is an EXACT series of register writes stamped down repeatedly in time (a drum
pattern). So detect percussion structurally: per voice, find RESID-note write-series that RECUR
(>=MINREP times), and check whether their onsets form a rhythmic GRID. Waveform-agnostic (catches
pitched/click/noise drums alike). Yields the learnable encoding directly: a stamp codebook + onset
pattern, NOT per-frame offsets. Validates the iter-2 percussion primitive before touching the
tokenizer.

Signature variants tested (which definition of 'same drum' best covers RESID):
  abs   = exact (fn, ctrl) per frame          -- a fixed sample/sweep at one absolute pitch
  rel   = (note-relative offset, ctrl)         -- same gesture re-pitched per trigger
  shape = (freq-delta sign/mag bucket, ctrl)   -- transposition + octave-invariant contour
  ctrl  = ctrl byte series only                -- the gate/waveform rhythm envelope

Usage:  resid_percussion.py <fixtures|N|paths> [procs] [minrep]"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, glob, random, statistics, subprocess  # noqa: E401,E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.macros.skeleton_pass import SkeletonPass  # noqa: E402
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


def _fdelta_bucket(d):
    if d == 0:
        return 0
    s = 1 if d > 0 else -1
    return s * (1 + (abs(d).bit_length()))  # coarse log-magnitude bucket


def signatures(rec):
    """rec = [(offset, ctrl, is_pitched, fn)]. Return the 4 candidate stamp signatures."""
    offs = [o for o, _c, _m, _fn in rec]
    ctrls = [c for _o, c, _m, _fn in rec]
    fns = [fn for _o, _c, _m, fn in rec]
    fdiffs = [0] + [fns[i] - fns[i - 1] for i in range(1, len(fns))]
    return {
        "abs": tuple(zip(fns, ctrls)),
        "rel": tuple(zip(offs, ctrls)),
        "shape": tuple((_fdelta_bucket(d), c) for d, c in zip(fdiffs, ctrls)),
        "ctrl": tuple(ctrls),
    }


def grid_score(onsets):
    """How rhythmic are these onsets? Returns (median_IOI, frac_on_grid). frac_on_grid = share of
    inter-onset intervals that are near-integer multiples of the smallest common pulse."""
    if len(onsets) < 3:
        return (0, 0.0)
    ois = sorted(onsets)
    iois = [b - a for a, b in zip(ois, ois[1:]) if b > a]
    if not iois:
        return (0, 0.0)
    base = min(iois)
    if base <= 0:
        return (0, 0.0)
    on = sum(1 for x in iois if abs(round(x / base) - x / base) <= 0.15)
    return (int(statistics.median(iois)), on / len(iois))


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


def scan(paths, dmap):
    # per (dump, reg, sigtype) -> sig -> list of (onset_fr, nframes)
    groups = defaultdict(lambda: defaultdict(list))
    resid_notes = defaultdict(int)   # driver -> count
    resid_frames = defaultdict(int)
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
        for reg, is_resid, _note, onset_fr, rec in recs:
            if not is_resid or not rec:
                continue
            resid_notes[drv] += 1
            resid_frames[drv] += len(rec)
            sigs = signatures(rec)
            for st, sig in sigs.items():
                # key WITHOUT reg: a stamp is voice-agnostic (same writes on whichever voice),
                # so occurrences on different voices MERGE; carry reg to count distinct voices.
                groups[(drv, p, st)][sig].append((onset_fr, len(rec), int(reg)))
    # plain picklable dicts (defaultdict(lambda) can't cross the mp.Pool boundary)
    return ({k: dict(v) for k, v in groups.items()}, dict(resid_notes), dict(resid_frames))


def _worker(args):
    paths, dmap = args
    return scan(paths, dmap)


if __name__ == "__main__":
    import multiprocessing as mp

    spec = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    minrep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
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
    print(f"scanning {len(sample)} dumps for recurring percussion stamps (minrep={minrep})",
          file=sys.stderr, flush=True)
    dmap = driver_map([os.path.dirname(p) for p in sample])
    shards = [sample[i::procs] for i in range(procs)]
    shards = [(s, dmap) for s in shards if s]
    with mp.Pool(len(shards)) as pool:
        results = pool.map(_worker, shards)
    groups = defaultdict(lambda: defaultdict(list))
    resid_notes, resid_frames = Counter(), Counter()
    for g, rn, rf in results:
        resid_notes.update(rn)
        resid_frames.update(rf)
        for key, sigmap in g.items():
            for sig, occ in sigmap.items():
                groups[key][sig].extend(occ)

    # Coverage per signature type: RESID notes/frames in a sig recurring >= minrep, per driver.
    print(f"\n=== RESID totals: {sum(resid_notes.values())} notes / "
          f"{sum(resid_frames.values())} frames ===")
    cover = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # sigtype -> driver -> [notes, frames]
    grids = defaultdict(list)  # sigtype -> [frac_on_grid for recurring stamps]
    multivoice = defaultdict(lambda: [0, 0])  # sigtype -> [stamps_on_>1_voice, notes_on_them]
    stamp_examples = defaultdict(list)
    for (drv, _p, st), sigmap in groups.items():
        for sig, occ in sigmap.items():
            if len(occ) >= minrep:
                cover[st][drv][0] += len(occ)
                cover[st][drv][1] += sum(n for _o, n, _r in occ)
                _med, frac = grid_score([o for o, _n, _r in occ])
                grids[st].append(frac)
                nvoices = len(set(r for _o, _n, r in occ))
                if nvoices > 1:
                    multivoice[st][0] += 1
                    multivoice[st][1] += len(occ)
                if len(stamp_examples[st]) < 10:
                    stamp_examples[st].append((drv, len(occ), _med, frac, nvoices, sig))

    for st in ("abs", "rel", "shape", "ctrl"):
        tn = sum(c[0] for c in cover[st].values())
        tf = sum(c[1] for c in cover[st].values())
        gl = grids[st]
        gridded = (sum(1 for f in gl if f >= 0.6) / len(gl)) if gl else 0
        mv = multivoice[st]
        print(f"\n--- signature='{st}' : recurring(>= {minrep}) covers "
              f"{tn} notes ({100*tn//max(sum(resid_notes.values()),1)}%) / "
              f"{tf} frames ({100*tf//max(sum(resid_frames.values()),1)}%) | "
              f"{len(gl)} stamps, {100*gridded:.0f}% gridded | "
              f"multi-voice: {mv[0]} stamps / {mv[1]} notes ---")
        for drv, c in sorted(cover[st].items(), key=lambda x: -x[1][0]):
            print(f"      {drv}: {c[0]} notes / {c[1]} frames")

    print("\n=== example recurring stamps (signature='abs', exact writes) ===")
    for drv, n, med, frac, nv, sig in sorted(stamp_examples["abs"], key=lambda x: -x[1])[:10]:
        print(f"  {drv}  x{n}  voices={nv}  IOI~{med}f grid={frac:.2f}  "
              f"(fn,ctrl)[:8]={list(sig[:8])}")
