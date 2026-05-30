"""End-to-end PROTOTYPE of footprint-mining + consistency-attribution for the percussion stamp
codebook (design/percussion_stamp_encoding.md), non-emitting -- measures the codebook economics
before building the tokenizer. Per tune:
  - mine recurring EXACT stamps (abs (fn,ctrl) series, >=MINREP) per voice  = the inline DEFS
  - cluster exact variants by SHAPE sig (freq-contour + ctrl)               = DRUMS (a drum redefined
    / retuned shows as multiple exact variants in one shape cluster)        = REDEFINITION count
  - consistency-attribution of the FULL footprint per exact stamp: fold PW/ADSR that recur every hit;
    attribute global filter (drum-scoped if identical every hit, else tune-global)
  - classify drum CHARACTER from the stamp's writes
Reports coverage (RESID notes/frames drained), codebook size (defs/tune), drums/tune, redefinitions,
drum-scoped-filter rate, folded-aux rate, and the character distribution -- across a big rung.

Usage:  resid_drum_codebook.py <fixtures|N|paths> [procs] [minrep]"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, glob, random, statistics  # noqa: E401,E402
from collections import defaultdict, Counter  # noqa: E402

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.macros.skeleton_pass import SkeletonPass, fn_to_note_resid  # noqa: E402
from preframr_tokens.stfconstants import SET_OP  # noqa: E402
from parse_probes import parse_args  # noqa: E402

CORPUS = "/scratch/preframr/hvsc"
FIXTURES = [
    f"{CORPUS}/MUSICIANS/G/Goto80/Baggis.1.dump.parquet",
    f"{CORPUS}/MUSICIANS/D/DRAX/Camerock.1.dump.parquet",
    f"{CORPUS}/GAMES/G-L/Gridtrap.1.dump.parquet",
]
FILT_REGS = {21, 22, 23, 24}


def _args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


def _fdbucket(d):
    if d == 0:
        return 0
    return (1 if d > 0 else -1) * (1 + abs(d).bit_length())


def shape_sig(rec):
    fns = [fn for _o, _c, _m, fn in rec]
    ctrls = [c for _o, c, _m, _fn in rec]
    fd = [0] + [fns[i] - fns[i - 1] for i in range(1, len(fns))]
    return tuple((_fdbucket(d), c) for d, c in zip(fd, ctrls))


def character(rec):
    """Coarse drum character from the stamp writes -> a small global vocabulary."""
    n = len(rec)
    noise = sum(1 for _o, c, _m, _fn in rec if c >= 0 and (c & 0x80) and not (c & 0x08))
    pitched = [(fn_to_note_resid(fn) or (0, 0))[0] for _o, _c, m, fn in rec if m]
    nfrac = noise / max(n, 1)
    pfrac = len(pitched) / max(n, 1)
    base = statistics.median(pitched) if pitched else 0
    sweep = (max(pitched) - min(pitched)) if len(pitched) >= 2 else 0
    down = pitched and pitched[0] - min(pitched) >= 6
    if nfrac >= 0.6 and pfrac < 0.3:
        return "HAT" if n <= 5 else ("CYMBAL" if n <= 16 else "NOISE_FX")
    if 0.2 <= nfrac < 0.8 and pfrac >= 0.2:
        return "SNARE"
    if pfrac >= 0.5 and down and sweep >= 12:
        return "KICK" if base <= 50 else "TOM"
    if pfrac >= 0.6 and sweep < 6:
        return "PITCH_FX"  # repeated tonal trigger -- recurring but not really a drum
    return "OTHER"


def analyze(paths, minrep):
    rows = []
    a = _args()
    for p in paths:
        tune = os.path.basename(p).split(".")[0]
        SkeletonPass._resid_diag = []
        SkeletonPass._df_sink = []
        try:
            next(RegLogParser(args=a).parse(p, max_perm=1, require_pq=False, reparse=True), None)
        except Exception:
            SkeletonPass._resid_diag = SkeletonPass._df_sink = None
            continue
        recs = SkeletonPass._resid_diag or []
        dfs = SkeletonPass._df_sink or []
        SkeletonPass._resid_diag = SkeletonPass._df_sink = None
        if not dfs:
            continue
        df = dfs[0]
        sets = df[df["op"] == SET_OP]
        byfr = defaultdict(list)
        for f, r, v in zip(sets["_fr"].to_numpy(), sets["reg"].to_numpy(), sets["val"].to_numpy()):
            byfr[int(f)].append((int(r), int(v)))

        resid_notes = sum(1 for _r, isr, _n, _o, rec in recs if isr and rec)
        resid_frames = sum(len(rec) for _r, isr, _n, _o, rec in recs if isr and rec)

        # group RESID notes by exact stamp (voice reg + abs (fn,ctrl) series)
        exact = defaultdict(list)  # (reg, abs_sig) -> [(onset, rec)]
        for reg, isr, _n, onset, rec in recs:
            if not isr or not rec:
                continue
            absig = tuple((fn, c) for _o, c, _m, fn in rec)
            exact[(int(reg), absig)].append((int(onset), rec))

        defs = 0
        cov_notes = cov_frames = 0
        clusters = defaultdict(set)  # shape_sig -> {abs variant ids}
        drum_scoped_filt = folded_aux = 0
        chars = Counter()
        for (reg, absig), occ in exact.items():
            if len(occ) < minrep:
                continue
            defs += 1
            cov_notes += len(occ)
            cov_frames += sum(len(r) for _o, r in occ)
            rec0 = occ[0][1]
            clusters[shape_sig(rec0)].add(absig)
            chars[character(rec0)] += 1
            # footprint consistency across this exact stamp's occurrences
            vregs = {reg + 2, reg + 5, reg + 6}
            fvoice, ffilt = [], []
            for onset, rec in occ:
                fv, ff = [], []
                for d in range(0, len(rec) + 1):
                    for r, v in byfr.get(onset + d, []):
                        if r in vregs:
                            fv.append((d, r, v))
                        elif r in FILT_REGS:
                            ff.append((d, r, v))
                fvoice.append(tuple(fv))
                ffilt.append(tuple(ff))
            if any(fvoice) and len(set(fvoice)) == 1:
                folded_aux += 1
            if any(ffilt) and len(set(ffilt)) == 1:
                drum_scoped_filt += 1
        drums = len(clusters)
        redefs = sum(len(v) - 1 for v in clusters.values())  # extra exact variants per drum
        rows.append((tune, resid_notes, resid_frames, cov_notes, cov_frames, defs, drums,
                     redefs, drum_scoped_filt, folded_aux, dict(chars)))
    return rows


def _worker(args):
    paths, minrep = args
    return analyze(paths, minrep)


if __name__ == "__main__":
    import multiprocessing as mp

    spec = sys.argv[1] if len(sys.argv) > 1 else "100"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
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
    print(f"mining drum codebook across {len(sample)} dumps (minrep={minrep})",
          file=sys.stderr, flush=True)
    shards = [(sample[i::procs], minrep) for i in range(procs)]
    shards = [s for s in shards if s[0]]
    with mp.Pool(len(shards)) as pool:
        results = pool.map(_worker, shards)
    allrows = [r for res in results for r in res]

    tot = Counter()
    chars = Counter()
    defs_per_tune, drums_per_tune, redef_per_drum, covpct = [], [], [], []
    tunes_with_drums = 0
    for (tune, rn, rf, cn, cf, defs, drums, redefs, dsf, fa, ch) in allrows:
        tot["resid_notes"] += rn
        tot["resid_frames"] += rf
        tot["cov_notes"] += cn
        tot["cov_frames"] += cf
        tot["defs"] += defs
        tot["drums"] += drums
        tot["redefs"] += redefs
        tot["drum_scoped_filt"] += dsf
        tot["folded_aux"] += fa
        chars.update(ch)
        if defs:
            tunes_with_drums += 1
            defs_per_tune.append(defs)
            drums_per_tune.append(drums)
            if drums:
                redef_per_drum.append(redefs / drums)
            covpct.append(100 * cn / max(rn, 1))

    def _stats(xs):
        if not xs:
            return "n/a"
        xs = sorted(xs)
        return f"median {statistics.median(xs):.1f}  p90 {xs[min(len(xs)-1, int(0.9*len(xs)))]:.1f}  max {max(xs):.1f}"

    print(f"\n=== DRUM CODEBOOK PROTOTYPE: {len(allrows)} tunes parsed, "
          f"{tunes_with_drums} with recurring stamps ===")
    print(f"RESID drained: {tot['cov_notes']}/{tot['resid_notes']} notes "
          f"({100*tot['cov_notes']//max(tot['resid_notes'],1)}%) | "
          f"{tot['cov_frames']}/{tot['resid_frames']} frames "
          f"({100*tot['cov_frames']//max(tot['resid_frames'],1)}%)")
    print(f"codebook size (DEFS/tune):     {_stats(defs_per_tune)}   total {tot['defs']}")
    print(f"distinct DRUMS/tune (clusters):{_stats(drums_per_tune)}   total {tot['drums']}")
    print(f"REDEFINITIONS per drum:        {_stats(redef_per_drum)}   total {tot['redefs']}")
    print(f"per-tune coverage %% of RESID notes: {_stats(covpct)}")
    print(f"folded PW/ADSR (consistent): {tot['folded_aux']}/{tot['defs']} defs "
          f"({100*tot['folded_aux']//max(tot['defs'],1)}%)")
    print(f"drum-scoped FILTER: {tot['drum_scoped_filt']}/{tot['defs']} defs "
          f"({100*tot['drum_scoped_filt']//max(tot['defs'],1)}%)")
    print("\n=== drum CHARACTER distribution (of defs) ===")
    for c, n in chars.most_common():
        print(f"   {n:5d} ({100*n//max(tot['defs'],1):2d}%)  {c}")
    perc = sum(n for c, n in chars.items() if c not in ("PITCH_FX", "OTHER"))
    print(f"   -> percussion-character defs: {perc}/{tot['defs']} "
          f"({100*perc//max(tot['defs'],1)}%); rest are repeated tonal/other stamps")
