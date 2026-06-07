"""PROTOTYPE of the melodic-PATCH preamble (the non-drum twin of the drum stamp
codebook). A melodic note's PITCH is the skeleton+ornament; its TIMBRE is a
reusable INSTRUMENT PATCH = the pitch-invariant non-pitch footprint: ADSR + the control-register
ARTICULATION sequence (HR/test -> attack waveform -> sustain -> release) + PW. Define the patch once
as a preamble, reference it per note; then a new melodic note = patch-ref + skeleton + ornament.
Unigram clusters similar patches (similar ctrl-progression / similar ADSR share sub-tokens).

This probe extracts, for EVERY claimed note (diag records onset_fr for all, RESID or not), the
per-note aux footprint from the raw df, clusters notes into patches, and measures codebook economics:
patches/tune, note coverage, reuse, and how much an exact codebook collapses under near-match
(ctrl-progression + ADSR only) -- the Unigram-clusterability proxy.

Usage:  resid_patch_codebook.py <fixtures|N|paths> [procs] [minrep]"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")
import sys, glob, random, statistics  # noqa: E401,E402
from collections import defaultdict, Counter  # noqa: E402

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.macros.skeleton_pass import SkeletonPass  # noqa: E402
from preframr_tokens.stfconstants import SET_OP  # noqa: E402
from parse_probes import parse_args  # noqa: E402

CORPUS = "/scratch/preframr/hvsc"
FIXTURES = [
    f"{CORPUS}/MUSICIANS/G/Goto80/Baggis.1.dump.parquet",
    f"{CORPUS}/MUSICIANS/D/DRAX/Camerock.1.dump.parquet",
    f"{CORPUS}/GAMES/G-L/Gridtrap.1.dump.parquet",
]


def _args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


def _rle(xs):
    out = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return tuple(out)


def patch_of(reg, onset, nxt, byfr):
    """Per-note aux footprint -> (ctrl_progression, AD, SR, PW). reg = voice freq reg (0/7/14);
    ctrl=reg+4, AD=reg+5, SR=reg+6, PW=reg+2 (raw per-voice)."""
    ctrl_r, ad_r, sr_r, pw_r = reg + 4, reg + 5, reg + 6, reg + 2
    ctrl_seq = []
    ad = sr = pw = None
    for d in range(0, max(1, nxt - onset)):
        for r, v in byfr.get(onset + d, []):
            if r == ctrl_r:
                ctrl_seq.append(v)
            elif r == ad_r and ad is None:
                ad = v
            elif r == sr_r and sr is None:
                sr = v
            elif r == pw_r and pw is None:
                pw = v
    return (_rle(ctrl_seq), ad, sr, pw)


def analyze(paths, minrep):
    rows = []
    a = _args()
    for p in paths:
        tune = os.path.basename(p).split(".")[0]
        SkeletonPass._resid_diag = []
        SkeletonPass._df_sink = []
        try:
            next(
                RegLogParser(args=a).parse(
                    p, max_perm=1, require_pq=False, reparse=True
                ),
                None,
            )
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
        for f, r, v in zip(
            sets["_fr"].to_numpy(), sets["reg"].to_numpy(), sets["val"].to_numpy()
        ):
            byfr[int(f)].append((int(r), int(v)))

        # all claimed note onsets per voice reg
        onsets = defaultdict(list)
        for reg, _isr, _n, onset, _rec in recs:
            onsets[int(reg)].append(int(onset))
        notes = 0
        exact = Counter()  # full patch sig -> count
        near = (
            Counter()
        )  # (ctrl_prog, AD, SR) -> count  (PW-invariant = Unigram cluster proxy)
        examples = {}
        for reg, ons in onsets.items():
            ons = sorted(ons)
            for i, onset in enumerate(ons):
                nxt = ons[i + 1] if i + 1 < len(ons) else onset + 32
                pat = patch_of(reg, onset, nxt, byfr)
                notes += 1
                exact[pat] += 1
                nk = (pat[0], pat[1], pat[2])
                near[nk] += 1
                if nk not in examples:
                    examples[nk] = pat
        rows.append((tune, notes, exact, near, examples))
    return rows


def _worker(args):
    paths, minrep = args
    return analyze(paths, minrep)


def _stats(xs):
    if not xs:
        return "n/a"
    xs = sorted(xs)
    return f"median {statistics.median(xs):.1f}  p90 {xs[min(len(xs)-1, int(0.9*len(xs)))]:.1f}  max {max(xs):.1f}"


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
    print(
        f"mining melodic patch codebook across {len(sample)} dumps",
        file=sys.stderr,
        flush=True,
    )
    shards = [(sample[i::procs], minrep) for i in range(procs)]
    shards = [s for s in shards if s[0]]
    with mp.Pool(len(shards)) as pool:
        results = pool.map(_worker, shards)
    allrows = [r for res in results for r in res]

    tot = Counter()
    exact_per_tune, near_per_tune, cov_top8, reuse = [], [], [], []
    all_near = Counter()
    for tune, notes, exact, near, examples in allrows:
        tot["notes"] += notes
        tot["exact"] += len(exact)
        tot["near"] += len(near)
        all_near.update({k: v for k, v in near.items()})
        if notes:
            exact_per_tune.append(len(exact))
            near_per_tune.append(len(near))
            top8 = sum(c for _k, c in near.most_common(8))
            cov_top8.append(100 * top8 / notes)
            reuse.append(notes / max(len(near), 1))

    print(
        f"\n=== MELODIC PATCH CODEBOOK PROTOTYPE: {len(allrows)} tunes, "
        f"{tot['notes']} claimed notes ==="
    )
    print(
        f"EXACT patches (ctrl-prog+ADSR+PW)/tune:   {_stats(exact_per_tune)}   total {tot['exact']}"
    )
    print(
        f"NEAR patches  (ctrl-prog+ADSR, PW-free)/tune: {_stats(near_per_tune)}   total {tot['near']}"
    )
    print(
        f"collapse exact->near (Unigram-cluster proxy): "
        f"{tot['exact']} -> {tot['near']} ({100*(tot['exact']-tot['near'])//max(tot['exact'],1)}% fewer)"
    )
    print(f"top-8 patches cover %% of a tune's notes: {_stats(cov_top8)}")
    print(f"reuse (notes per near-patch): {_stats(reuse)}")
    print("\n=== most common NEAR patches corpus-wide (ctrl-progression | AD | SR) ===")
    for (ctrl_prog, ad, sr), c in all_near.most_common(12):
        cp = [hex(x) for x in ctrl_prog[:8]]
        print(f"   x{c:6d}  ctrl={cp}  AD={ad} SR={sr}")
