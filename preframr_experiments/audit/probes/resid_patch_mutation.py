"""Measure patch MUTATION granularity to design mutation primitives. A patch should be UPDATED inline
(change one ADSR param), not fully redefined, when it drifts minimally. So: identify an instrument by
its control-register articulation (ctrl-progression), track its parameters (attack/decay/sustain/
release nibbles + PW) FFILL'd (held) per voice, walk consecutive notes, and classify each transition:
  instrument-change (ctrl-prog differs)  vs  N-field parameter MUTATION (ctrl-prog same, params drift)
Reports how often a change is a 1-field tweak (a single ADSR nibble / PW) -> the mutation primitive's
payoff, and which fields mutate most.

Usage:  resid_patch_mutation.py <fixtures|N|paths> [procs]"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, glob, random, bisect  # noqa: E401,E402
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
FIELDS = ("attack", "decay", "sustain", "release", "pw")


def _args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


def _rle(xs):
    out = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return tuple(out)


def _ffill(series, frame):
    """Last (frame<=) value in a sorted [(frame,val)] series, or None."""
    if not series:
        return None
    i = bisect.bisect_right(series, (frame, 1 << 30)) - 1
    return series[i][1] if i >= 0 else None


def analyze(paths):
    out = Counter()
    field_hits = Counter()
    a = _args()
    for p in paths:
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
        # per-reg sorted (frame,val) for ffill; ctrl writes per frame
        reg_series = defaultdict(list)
        ctrl_byfr = defaultdict(lambda: defaultdict(list))  # reg -> frame -> [vals]
        for f, r, v in zip(sets["_fr"].to_numpy(), sets["reg"].to_numpy(), sets["val"].to_numpy()):
            reg_series[int(r)].append((int(f), int(v)))
            ctrl_byfr[int(r)][int(f)].append(int(v))
        for r in reg_series:
            reg_series[r].sort()

        onsets = defaultdict(list)
        for reg, _isr, _n, onset, _rec in recs:
            onsets[int(reg)].append(int(onset))

        for reg, ons in onsets.items():
            ons = sorted(ons)
            ad_r, sr_r, pw_r, ctrl_r = reg + 5, reg + 6, reg + 2, reg + 4
            prev = None
            for i, onset in enumerate(ons):
                nxt = ons[i + 1] if i + 1 < len(ons) else onset + 32
                ad = _ffill(reg_series.get(ad_r, []), onset)
                sr = _ffill(reg_series.get(sr_r, []), onset)
                pw = _ffill(reg_series.get(pw_r, []), onset)
                cseq = []
                for d in range(0, max(1, nxt - onset)):
                    # instrument identity = WAVEFORM progression; mask out per-note articulation
                    # (gate bit0, test/HR bit3) which is note-timing, not timbre.
                    cseq.extend(v & ~0x09 for v in ctrl_byfr[ctrl_r].get(onset + d, []))
                cprog = _rle(cseq)
                attack = (ad >> 4) if ad is not None else None
                decay = (ad & 0xF) if ad is not None else None
                sustain = (sr >> 4) if sr is not None else None
                release = (sr & 0xF) if sr is not None else None
                adsr = (attack, decay, sustain, release)
                cur = (adsr, cprog, pw)
                if prev is not None:
                    out["transitions"] += 1
                    padsr, pcprog, ppw = prev
                    wf_changed = pcprog != cprog
                    pw_changed = ppw != pw
                    adiff = [k for k in range(4) if padsr[k] != adsr[k]]
                    if not adiff:
                        out["adsr_same"] += 1
                    elif len(adiff) == 1:
                        out["adsr_1nibble"] += 1
                        field_hits[FIELDS[adiff[0]]] += 1
                        if not wf_changed:
                            out["adsr_1nibble_wf_stable"] += 1
                    elif len(adiff) == 2:
                        out["adsr_2nibble"] += 1
                        for k in adiff:
                            field_hits[FIELDS[k]] += 1
                    else:
                        out["adsr_3plus"] += 1
                    if adiff and not wf_changed:
                        out["adsr_change_wf_stable"] += 1
                    out["wf_changed"] += int(wf_changed)
                    out["pw_changed"] += int(pw_changed)
                prev = cur
    return out, field_hits


def _worker(paths):
    return analyze(paths)


if __name__ == "__main__":
    import multiprocessing as mp

    spec = sys.argv[1] if len(sys.argv) > 1 else "150"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 24
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
    print(f"measuring patch mutation granularity across {len(sample)} dumps", file=sys.stderr, flush=True)
    shards = [sample[i::procs] for i in range(procs)]
    shards = [s for s in shards if s]
    with mp.Pool(len(shards)) as pool:
        results = pool.map(_worker, shards)
    out = Counter()
    field_hits = Counter()
    for o, fh in results:
        out.update(o)
        field_hits.update(fh)

    t = max(out["transitions"], 1)
    adsr_changes = out["adsr_1nibble"] + out["adsr_2nibble"] + out["adsr_3plus"]
    ac = max(adsr_changes, 1)
    print(f"\n=== PATCH MUTATION GRANULARITY: {out['transitions']} note->note transitions "
          f"(same voice) ===")
    print("  -- ADSR (the stable patch core) change size:")
    print(f"  ADSR unchanged:        {out['adsr_same']:8d} ({100*out['adsr_same']//t}% of transitions)")
    print(f"  ADSR 1-nibble MUT:     {out['adsr_1nibble']:8d} ({100*out['adsr_1nibble']//ac}% of ADSR changes)")
    print(f"  ADSR 2-nibble MUT:     {out['adsr_2nibble']:8d} ({100*out['adsr_2nibble']//ac}%)")
    print(f"  ADSR 3-4 nibble (redef):{out['adsr_3plus']:8d} ({100*out['adsr_3plus']//ac}%)")
    print(f"  => ADSR changes with WAVEFORM STABLE (pure param tweak): "
          f"{out['adsr_change_wf_stable']}/{adsr_changes} "
          f"({100*out['adsr_change_wf_stable']//ac}%)")
    print(f"  => 1-nibble ADSR + waveform stable (ideal MUT primitive): {out['adsr_1nibble_wf_stable']}")
    print("\n  -- separate continuous channels (NOT patch identity):")
    print(f"  waveform-progression changed: {out['wf_changed']:8d} ({100*out['wf_changed']//t}% of transitions)")
    print(f"  PW changed:                   {out['pw_changed']:8d} ({100*out['pw_changed']//t}%)")
    print("\n=== which ADSR nibble mutates (1+2-nibble) ===")
    for f, n in field_hits.most_common():
        print(f"   {n:8d}  {f}")
