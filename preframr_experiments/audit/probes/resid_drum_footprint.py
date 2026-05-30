"""Drum FULL-FOOTPRINT + filter-attribution probe. A drum stamp is not just freq+ctrl -- it is ALL
the register writes during its span (PW, ADSR), and possibly drum-scoped filter. The SID filter is
GLOBAL (one cutoff/res/mode shared by 3 voices), so filter writes overlapping a drum may be
(a) drum-scoped (the drum drives the filter; recurs identically every hit) or (b) tune-global
automation (independent sweep that merely overlaps). This probe attributes them.

Per recurring drum stamp (raw per-voice regs; freq R in {0,7,14} -> pw R+2, ad R+5, sr R+6; global
filter 21-24), across its occurrences:
  - PW/ADSR consistency: are the in-span voice writes IDENTICAL every hit? -> fold into the stamp def
  - filter in-span recurrence: identical every hit? -> drum-scoped, else coincidental
  - tune-wide: fraction of filter writes INSIDE any drum span vs OUTSIDE -> global vs drum

Usage:  resid_drum_footprint.py <fixtures|N|paths> [procs] [minrep]"""
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
from preframr_tokens.stfconstants import SET_OP  # noqa: E402
from parse_probes import parse_args  # noqa: E402

CORPUS = "/scratch/preframr/hvsc"
SIDID_CFG = "/scratch/anarkiwi/sidid/sidid.cfg"
FIXTURES = [
    f"{CORPUS}/MUSICIANS/G/Goto80/Baggis.1.dump.parquet",
    f"{CORPUS}/MUSICIANS/D/DRAX/Camerock.1.dump.parquet",
    f"{CORPUS}/GAMES/G-L/Gridtrap.1.dump.parquet",
]
FILT_REGS = {21, 22, 23, 24}


def _args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


def analyze(paths):
    out = []
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
        fr = sets["_fr"].to_numpy()
        rg = sets["reg"].to_numpy()
        vl = sets["val"].to_numpy()
        # frame -> list[(reg,val)]
        byfr = defaultdict(list)
        for i in range(len(sets)):
            byfr[int(fr[i])].append((int(rg[i]), int(vl[i])))
        filt_total = sum(1 for i in range(len(sets)) if int(rg[i]) in FILT_REGS)

        # recurring stamps by freq+ctrl signature, per voice reg
        sig2occ = defaultdict(list)
        for reg, isr, _note, onset, rec in recs:
            if not isr or not rec:
                continue
            sig = tuple((f, c) for _o, c, _m, f in rec)
            sig2occ[(int(reg), sig)].append((int(onset), len(rec)))

        filt_in = 0
        pwadsr_consistent = pwadsr_present = 0
        filt_inspan_consistent = filt_inspan_present = 0
        stamps = 0
        examples = []
        for (reg, _sig), occ in sig2occ.items():
            if len(occ) < MINREP:
                continue
            stamps += 1
            voice_regs = {reg + 2, reg + 5, reg + 6}  # pw, ad, sr (raw per-voice)
            foot_voice, foot_filt = [], []
            for onset, ln in occ:
                fv, ff = [], []
                for d in range(0, ln + 1):
                    for r, v in byfr.get(onset + d, []):
                        if r in voice_regs:
                            fv.append((d, r, v))
                        elif r in FILT_REGS:
                            ff.append((d, r, v))
                            filt_in += 1
                foot_voice.append(tuple(fv))
                foot_filt.append(tuple(ff))
            if any(foot_voice):
                pwadsr_present += 1
                if len(set(foot_voice)) == 1:
                    pwadsr_consistent += 1
            if any(foot_filt):
                filt_inspan_present += 1
                if len(set(foot_filt)) == 1:
                    filt_inspan_consistent += 1
            if len(examples) < 4 and (any(foot_voice) or any(foot_filt)):
                examples.append((reg, len(occ), foot_voice[0][:6], foot_filt[0][:6]))
        out.append((tune, stamps, filt_total, filt_in, pwadsr_present, pwadsr_consistent,
                    filt_inspan_present, filt_inspan_consistent, examples))
    return out


def _worker(paths):
    return analyze(paths)


if __name__ == "__main__":
    import multiprocessing as mp

    spec = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    MINREP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
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
    globals()["MINREP"] = MINREP
    print(f"footprint+filter-attribution on {len(sample)} dumps (minrep={MINREP})",
          file=sys.stderr, flush=True)
    shards = [sample[i::procs] for i in range(procs)]
    shards = [s for s in shards if s]
    with mp.Pool(len(shards)) as pool:
        results = pool.map(_worker, shards)

    print("\n=== per-tune: drum footprint + filter attribution ===")
    print(f"{'tune':22s} {'stamps':>6} {'filtTot':>7} {'filtIn':>6} {'filtIn%':>7} "
          f"{'pwadsr(cons/pres)':>17} {'filtSpan(cons/pres)':>19}")
    agg = Counter()
    for res in results:
        for (tune, st, ft, fin, pp, pc, fp, fc, ex) in res:
            pct = (100 * fin // ft) if ft else 0
            print(f"{tune[:22]:22s} {st:6d} {ft:7d} {fin:6d} {pct:6d}% "
                  f"{pc:>8}/{pp:<8} {fc:>9}/{fp:<9}")
            agg["filt_total"] += ft
            agg["filt_in"] += fin
            agg["pw_pres"] += pp
            agg["pw_cons"] += pc
            agg["filt_span_pres"] += fp
            agg["filt_span_cons"] += fc
            agg["stamps"] += st
            for (reg, n, fv, ff) in ex:
                if agg["ex_shown"] < 12:
                    agg["ex_shown"] += 1
                    print(f"      e.g. stamp reg={reg} x{n}  voice(pw/adsr)[:6]={list(fv)}  "
                          f"filt[:6]={list(ff)}")
    ft = agg["filt_total"]
    print(f"\n=== TOTALS: {agg['stamps']} stamps | filter writes={ft}, "
          f"{agg['filt_in']} inside drum spans ({100*agg['filt_in']//max(ft,1)}%) -> "
          f"{'mostly TUNE-GLOBAL' if ft and agg['filt_in']*2 < ft else 'check drum-scoped'} | "
          f"pw/adsr consistent-in-stamp {agg['pw_cons']}/{agg['pw_pres']} | "
          f"filter-in-span consistent {agg['filt_span_cons']}/{agg['filt_span_pres']} ===")
