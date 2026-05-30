"""Corpus survey of skeleton-encoding RESID-trajectory archetypes. Goal: RESID=0.
Every RESID note is a freq trajectory the current primitives miss -> classify the
archetype (in semitone AND freq-shape domains) so we can find the missing primitive,
and surface the composers/drivers that produce each so they can be traced.

CONTROL-AWARE (2026-05-30): co-read the control register via SkeletonPass._resid_diag,
which records each RESID note's frames as (offset, is_pitched). Splits RESID into
contamination (noise/test/HR frames -> reclaimable by control-aware segmentation; only
test-bit freq is safely discardable, the rest is a separate audible percussion/effect
channel) vs genuinely-irreducible PITCHED content (the real RESID=0 target). The freq-only
archetype is kept side-by-side so the reclassification is visible."""
import os

# Pin BLAS/parquet to 1 thread PER worker BEFORE numpy/pandas import -- else each of the N
# multiprocessing workers spawns its own thread pool and oversubscribes the cores (load >5x,
# the survey crawls). Must precede the heavy imports below.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, glob, random, statistics, traceback  # noqa: E401,E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.macros.skeleton_pass import SkeletonPass  # noqa: E402
from parse_probes import parse_args  # noqa: E402


def _args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


def archetype(offs):
    n = len(offs)
    if n == 0:
        return "empty"
    if n <= 2:
        return "short-note(<=2 frame ornament)"
    diffs = [b - a for a, b in zip(offs, offs[1:])]
    med = statistics.median(offs)
    reb = [o - med for o in offs]
    outl = sum(1 for o in reb if abs(o) > 12)
    core = [o for o in reb if abs(o) <= 12]
    core_span = (max(core) - min(core)) if core else 99
    if outl <= 2 and core_span <= 4:
        return "transient/attack(tight core + <=2 spikes)"
    mono_down = all(d <= 0 for d in diffs) and any(d < 0 for d in diffs)
    mono_up = all(d >= 0 for d in diffs) and any(d > 0 for d in diffs)
    if mono_down or mono_up:
        absd = [abs(d) for d in diffs if d != 0]
        if absd and max(absd) >= 3 * max(min(absd), 1):
            return "accelerating-sweep(linear-freq drop/skydive)"
        return "uniform-slide(log-freq, SLIDE-overflow)"
    pcs = set(o % 12 for o in offs)
    if len(pcs) <= 2:
        return "octave/ratio-oscillation(base-shift/wide)"
    for p in range(1, n // 2 + 1):
        if all(offs[i] == offs[i % p] for i in range(n)):
            return "periodic-arp(wide, OFFSET-overflow)"
    if outl <= 3 and core_span <= 12:
        return "rebased-melodic-run(span<=12 off median)"
    return "genuinely-irregular"


# Freq-only archetypes that BECOME clean once noise/test frames are stripped count as
# contamination reclaimable by control-aware segmentation, not irreducible content.
_CLEAN_AFTER_STRIP = {
    "short-note(<=2 frame ornament)",
    "transient/attack(tight core + <=2 spikes)",
}


def classify_ca(rec):
    """rec = [(offset, ctrl, is_pitched, fn), ...]. Returns (bucket, n_frames, n_nonpitched)."""
    n = len(rec)
    pitched = [o for o, _c, m, _fn in rec if m]
    nonp = n - len(pitched)
    if not pitched:
        return "percussion/timbre(no pitched frame)", n, nonp
    if len(pitched) <= 2:
        return "reclaimed: pitched core <=2 (contamination)", n, nonp
    base = archetype(pitched)
    if base in _CLEAN_AFTER_STRIP:
        return "reclaimed: " + base, n, nonp
    return "irreducible: " + base, n, nonp


def survey(paths, wid=0):
    arche = Counter()              # freq-only archetype (full offsets) -- baseline
    arche_ca = Counter()           # control-aware bucket (pitched core)
    flow = Counter()               # (freq_only -> control_aware) reclassification flow
    by_comp = defaultdict(lambda: [0, 0])  # composer -> [orn, resid]
    examples = defaultdict(list)
    frames_tot = 0
    frames_nonp = 0
    parsed_ok = 0
    a = _args()
    n_paths = len(paths)
    step = max(1, n_paths // 10)
    for pi, p in enumerate(paths):
        comp = p.split("/MUSICIANS/", 1)[-1].split("/")[1] if "/MUSICIANS/" in p else "?"
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
        parsed_ok += 1
        by_comp[comp][0] += len(recs)  # diag fires once per claimed note (= total ORN denominator)
        for _reg, is_resid, _note, _onset, rec in recs:
            if not is_resid or not rec:
                continue
            by_comp[comp][1] += 1
            offs = [o for o, _c, _m, _fn in rec]
            fa = archetype(offs)
            ca, nf, nnp = classify_ca(rec)
            arche[fa] += 1
            arche_ca[ca] += 1
            flow[(fa, ca)] += 1
            frames_tot += nf
            frames_nonp += nnp
            if len(examples[ca]) < 4:
                examples[ca].append(offs[:16])
        if (pi + 1) % step == 0 or pi + 1 == n_paths:
            print(f"  [w{wid}] {pi+1}/{n_paths} dumps  resid={sum(arche.values())}",
                  file=sys.stderr, flush=True)
    return (arche, arche_ca, flow, dict(by_comp), dict(examples),
            frames_tot, frames_nonp, parsed_ok)


def _worker(args):
    wid, paths = args
    try:
        return survey(paths, wid)
    except Exception:
        traceback.print_exc()
        return (Counter(), Counter(), Counter(), {}, {}, 0, 0, 0)


def _merge(results):
    arche, arche_ca, flow = Counter(), Counter(), Counter()
    by_comp = defaultdict(lambda: [0, 0])
    examples = defaultdict(list)
    frames_tot = frames_nonp = ok = 0
    for a, aca, fl, bc, ex, ft, fnp, o in results:
        arche.update(a); arche_ca.update(aca); flow.update(fl)
        frames_tot += ft; frames_nonp += fnp; ok += o
        for name, c in bc.items():
            by_comp[name][0] += c[0]; by_comp[name][1] += c[1]
        for k, v in ex.items():
            examples[k].extend(v[: max(0, 4 - len(examples[k]))])
    return arche, arche_ca, flow, by_comp, examples, frames_tot, frames_nonp, ok


if __name__ == "__main__":
    import multiprocessing as mp

    arg = sys.argv[1] if len(sys.argv) > 1 else "150"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    per_comp = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 = one per composer
    random.seed(13)
    all_dumps = glob.glob("/scratch/preframr/hvsc/MUSICIANS/*/*/*.1.dump.parquet")
    by_dir = defaultdict(list)
    for d in all_dumps:
        by_dir[os.path.dirname(d)].append(d)
    if arg == "full":
        sample = all_dumps
    else:
        n = int(arg)
        if per_comp:  # several dumps per composer for a deeper stratified sample
            sample = []
            for v in by_dir.values():
                random.shuffle(v)
                sample.extend(v[:per_comp])
        else:
            sample = [random.choice(v) for v in by_dir.values()]
        random.shuffle(sample)
        sample = sample[:n]
    print(f"sampling {len(sample)} dumps across {procs} workers (one-per-composer stratified)",
          file=sys.stderr, flush=True)
    shards = [sample[i::procs] for i in range(procs)]
    shards = [(i, s) for i, s in enumerate(shards) if s]
    with mp.Pool(len(shards)) as pool:
        results = pool.map(_worker, shards)
    (arche, arche_ca, flow, by_comp, examples,
     frames_tot, frames_nonp, ok) = _merge(results)
    tot_orn = sum(c[0] for c in by_comp.values())
    tot_resid = sum(c[1] for c in by_comp.values())
    print(f"\nparsed {ok}/{len(sample)} dumps | ORN notes={tot_orn} RESID={tot_resid} "
          f"share={tot_resid/max(tot_orn,1):.3f}")
    print(f"RESID frames: {frames_tot} | non-pitched (noise/test/HR) frames: {frames_nonp} "
          f"({100*frames_nonp//max(frames_tot,1)}% of RESID frames are control contamination)")

    print("\n=== RESID archetypes -- FREQ-ONLY (baseline, full offsets) ===")
    for k, v in arche.most_common():
        print(f"  {v:6d} ({100*v//max(tot_resid,1):2d}%)  {k}")

    print("\n=== RESID archetypes -- CONTROL-AWARE (pitched core) ===")
    for k, v in arche_ca.most_common():
        print(f"  {v:6d} ({100*v//max(tot_resid,1):2d}%)  {k}")
        for ex in examples[k][:2]:
            print(f"               e.g. {ex}")
    reclaimed = sum(v for k, v in arche_ca.items()
                    if k.startswith("reclaimed") or k.startswith("percussion"))
    irr = sum(v for k, v in arche_ca.items() if k.startswith("irreducible"))
    print(f"\n  RECLAIMABLE (contamination/percussion): {reclaimed} "
          f"({100*reclaimed//max(tot_resid,1)}%)  |  IRREDUCIBLE pitched: {irr} "
          f"({100*irr//max(tot_resid,1)}%)")

    print("\n=== reclassification flow (freq-only -> control-aware), top 12 ===")
    for (fa, ca), v in flow.most_common(12):
        print(f"  {v:6d}  {fa:46s} -> {ca}")

    print("\n=== worst composers by RESID share (>=20 notes) ===")
    worst = sorted(((c[1]/max(c[0],1), name, c[0], c[1])
                    for name, c in by_comp.items() if c[0] >= 20), reverse=True)
    for share, name, o, r in worst[:15]:
        print(f"  {share:.2f}  {name:24s} (orn={o} resid={r})")
