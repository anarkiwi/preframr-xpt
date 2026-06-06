"""FINAL RESID=0 accounting across the corpus. Apply the FULL designed mechanism stack to every
RESID note and report what (if anything) remains UNACCOUNTED -- the true RESID>0:
  1. STAMP    -- the note's exact (fn,ctrl) write-series RECURS >=MINREP in the tune (drum/effect
                 codebook; lossless define+backref).  [design/percussion_stamp_encoding.md]
  2. ARP      -- pitched-offset cycle (held-arp: RLE-collapse + minimal period; covers wave-delay
                 holds & period>8).                    [iter-1 landed + held-arp irregular-duration]
  3. SLIDE    -- pitched freq is a linear ramp (semitone uniform-rate OR freq-domain constant delta;
                 portamento / skydive).
  4. SEGMENT  -- fast-melodic-run (short, distinct<6, span<12) -> split into notes.
  5. UNACCOUNTED -> the residual the program must drive to 0; reported with examples + composers.

Usage:  resid_final_accounting.py <N|full> [procs] [minrep]   (1500 = the 10x rung)"""

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")
import sys, glob, random, traceback  # noqa: E401,E402
from collections import defaultdict, Counter  # noqa: E402

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.macros.skeleton_pass import SkeletonPass  # noqa: E402
from parse_probes import parse_args  # noqa: E402

CORPUS = "/scratch/preframr/hvsc"


def _args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


def _rle(xs):
    out = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _period(xs, cap):
    n = len(xs)
    for p in range(1, min(cap, n // 2) + 1):
        if all(xs[i] == xs[i % p] for i in range(n)):
            return p
    return None


def _const_delta(xs, jitter=2):
    """xs is a ~constant-delta ramp (>=3, real slope, allowing small quantisation jitter) -> the
    delta, else None. Engines sweep the raw freq register by a fixed step (SoundMonitor/DMC).
    """
    if len(xs) < 3:
        return None
    d = [b - a for a, b in zip(xs, xs[1:])]
    nz = [x for x in d if x != 0]
    if len(nz) < 2:
        return None
    mean = sum(d) / len(d)
    if abs(mean) < 1:
        return None
    if all(abs(x - mean) <= max(jitter, abs(mean) * 0.34) for x in d):
        return round(mean)
    return None


def _sweep(fns):
    """Freq-domain SWEEP: a const-Δfreq ramp, optionally LOOPING (SoundMonitor arp resets). True if a
    one-shot or looping constant-freq-delta is found."""
    if _const_delta(fns) is not None:
        return True
    n = len(fns)
    for p in range(3, n // 2 + 1):  # looping sweep: a const-delta run that repeats
        if (
            all(abs(fns[i] - fns[i % p]) <= 2 for i in range(n))
            and _const_delta(fns[:p]) is not None
        ):
            return True
    return False


def _arp_cycle(offs, cap=16):
    """Wavetable-arp offset cycle, onset-stripped + RLE-collapsed; waveform-AGNOSTIC (do NOT gate on
    is_pitched -- the engine's offset table drives any waveform incl. noise)."""
    for strip in (0, 1, 2, 3):
        s = offs[strip:]
        if len(s) >= 6 and (
            _period(s, cap) is not None or _period(_rle(s), 8) is not None
        ):
            return True
    return False


def _wildsig(rec):
    """Note-relative signature with WIDE (|offset|>24) / NOISE frames replaced by a wildcard 'W' --
    the reused gesture with its varying element masked (engine reuses the table, element drifts).
    """
    return tuple(
        "W" if ((c & 0x80) or abs(o) > 24) else (o, c) for o, c, _m, _fn in rec
    )


def _glide(fns):
    """Target+duration glide (MoN 'lands exact'): monotone toward a target with NON-INCREASING step
    magnitude (asymptotic approach), >=5 frames."""
    if len(fns) < 5:
        return False
    d = [b - a for a, b in zip(fns, fns[1:])]
    if not (all(x >= 0 for x in d) or all(x <= 0 for x in d)):
        return False
    mags = [abs(x) for x in d if x != 0]
    if len(mags) < 4:
        return False
    # second half's steps no larger than first half's (approaching the target)
    h = len(mags) // 2
    return max(mags[h:]) <= max(mags[:h]) + 1


def _arp_accent(rec):
    """ARP with a PERIODIC non-pitched accent interleaved (System6581 gate-off/noise-tik): the cycle
    is clean once the accent frames are carried -- detect over the pitched core too."""
    poff = [o for o, _c, m, _fn in rec if m]
    return len(poff) >= 4 and (
        _period(poff, 8) is not None or _period(_rle(poff), 8) is not None
    )


def classify(rec):
    """Driver-native mechanism accounting for a RESID note (excluding STAMP, handled per tune).
    Mirrors the documented engine abstractions; returns a mechanism name or None."""
    alloff = [o for o, _c, _m, _fn in rec]
    allfn = [fn for _o, _c, _m, fn in rec]
    npitched = sum(1 for _o, _c, m, _fn in rec if m)
    # ARP (noise-inclusive wavetable cycle)
    if _arp_cycle(alloff):
        return "ARP"
    # SWEEP (freq-domain const-delta, one-shot or looping) or target+duration GLIDE
    if _sweep(allfn) or _glide(allfn) or _glide([fn for _o, _c, m, fn in rec if m]):
        return "SWEEP"
    # ARP with periodic accent (control-aware)
    if _arp_accent(rec):
        return "ARP_accent"
    # percussion: no pitched frame -> representable as a (possibly one-off) drum stamp/sweep
    if npitched == 0:
        return "PERC"
    # segment-then-fit: strip onset/tail transients (non-pitched / outliers), re-fit the core
    core = [o for o, _c, m, _fn in rec if m]
    cfn = [fn for _o, _c, m, fn in rec if m]
    if len(core) >= 3 and (_arp_cycle(core) or _sweep(cfn)):
        return "SEGMENT_fit"
    distinct = len(set(core))
    if core and distinct < 6 and (max(core) - min(core)) < 12:
        return "SEGMENT"
    # decompose: a held-gate note concatenates several gestures -> cover it with fitting sub-windows
    # (each an arp/sweep), skipping <=2-frame transients. Bounded for cost.
    if 6 <= len(rec) <= 80 and _decomposes(alloff, allfn):
        return "DECOMP"
    return None


def _decomposes(offs, fns):
    """Greedy cover: walk the note, at each position consume the longest window that fits an
    arp-cycle or freq-sweep; skip a lone transient frame; succeed iff the whole note is covered by
    fitting segments (+ isolated transients)."""
    n = len(offs)
    i = 0
    skips = 0
    while i < n:
        best = 0
        for j in range(min(n, i + 16), i + 5, -1):  # prefer the longest window
            if _arp_cycle(offs[i:j], cap=8) or _sweep(fns[i:j]):
                best = j
                break
        if best > i:
            i = best
        elif n - i <= 2 or skips < max(3, n // 8):
            i += 1
            skips += 1
        else:
            return False
    return True


def analyze(paths, minrep):
    acct = Counter()
    by_comp_unacct = Counter()
    examples = []
    parsed_ok = 0
    a = _args()
    for p in paths:
        comp = (
            p.split("/MUSICIANS/", 1)[-1].split("/")[1] if "/MUSICIANS/" in p else "?"
        )
        SkeletonPass._resid_diag = []
        try:
            parsed = next(
                RegLogParser(args=a).parse(
                    p, max_perm=1, require_pq=False, reparse=True
                ),
                None,
            )
        except Exception:
            SkeletonPass._resid_diag = None
            continue
        recs = SkeletonPass._resid_diag or []
        SkeletonPass._resid_diag = None
        if parsed is None or "op" not in getattr(parsed, "columns", []):
            continue
        parsed_ok += 1
        resid = [(reg, onset, rec) for reg, isr, _n, onset, rec in recs if isr and rec]
        # STAMP set: a write-series recurring >=minrep within the tune -- ABS (fn,ctrl) = fixed-freq
        # drum/effect; REL (offset,ctrl) = a transposable pitched gesture (backref carries a base).
        abs_count, rel_count, wild_count = Counter(), Counter(), Counter()
        for _reg, _on, rec in resid:
            abs_count[tuple((fn, c) for _o, c, _m, fn in rec)] += 1
            rel_count[tuple((o, c) for o, c, _m, _fn in rec)] += 1
            wild_count[_wildsig(rec)] += 1
        for _reg, _on, rec in resid:
            absig = tuple((fn, c) for _o, c, _m, fn in rec)
            relsig = tuple((o, c) for o, c, _m, _fn in rec)
            if abs_count[absig] >= minrep:
                acct["STAMP_abs"] += 1
                continue
            if rel_count[relsig] >= minrep:
                acct["STAMP_rel"] += 1
                continue
            # wildcarded rel-stamp: a gesture reused with its WIDE/NOISE element varying
            # (Gilmore wide-arp drift, Danko noise-jitter) -- the engine reuses the table, the
            # backref carries the varying element. Require a non-wild majority to avoid spurious
            # all-noise matches.
            wsig = _wildsig(rec)
            nonwild = sum(1 for t in wsig if t != "W")
            if nonwild * 2 >= len(wsig) and wild_count[wsig] >= minrep:
                acct["STAMP_wild"] += 1
                continue
            mech = classify(rec)
            if mech:
                acct[mech] += 1
            else:
                acct["UNACCOUNTED"] += 1
                by_comp_unacct[comp] += 1
                if len(examples) < 60:
                    examples.append(
                        (
                            comp,
                            [o for o, _c, _m, _fn in rec][:14],
                            [hex(c) for _o, c, _m, _fn in rec][:14],
                        )
                    )
    return acct, by_comp_unacct, examples, parsed_ok


def _worker(args):
    paths, minrep = args
    try:
        return analyze(paths, minrep)
    except Exception:
        traceback.print_exc()
        return Counter(), Counter(), [], 0


if __name__ == "__main__":
    import multiprocessing as mp

    spec = sys.argv[1] if len(sys.argv) > 1 else "1500"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    minrep = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    random.seed(13)
    all_dumps = glob.glob(f"{CORPUS}/MUSICIANS/*/*/*.1.dump.parquet")
    by_dir = defaultdict(list)
    for d in all_dumps:
        by_dir[os.path.dirname(d)].append(d)
    if spec == "full":
        sample = all_dumps
    else:
        sample = [random.choice(v) for v in by_dir.values()]
        random.shuffle(sample)
        sample = sample[: int(spec)]
    print(
        f"final RESID accounting across {len(sample)} dumps (minrep={minrep})",
        file=sys.stderr,
        flush=True,
    )
    shards = [(sample[i::procs], minrep) for i in range(procs)]
    shards = [s for s in shards if s[0]]
    with mp.Pool(len(shards)) as pool:
        results = pool.map(_worker, shards)
    acct = Counter()
    by_comp = Counter()
    examples = []
    ok = 0
    for a, bc, ex, o in results:
        acct.update(a)
        by_comp.update(bc)
        ok += o
        if len(examples) < 60:
            examples.extend(ex[: 60 - len(examples)])

    total = sum(acct.values())
    print(
        f"\n=== FINAL RESID ACCOUNTING: {ok}/{len(sample)} dumps, {total} RESID notes ==="
    )
    for mech in (
        "STAMP_abs",
        "STAMP_rel",
        "STAMP_wild",
        "ARP",
        "ARP_accent",
        "SWEEP",
        "PERC",
        "SEGMENT_fit",
        "SEGMENT",
        "DECOMP",
        "UNACCOUNTED",
    ):
        n = acct[mech]
        print(f"   {mech:12s} {n:8d}  ({100*n/max(total,1):.2f}%)")
    un = acct["UNACCOUNTED"]
    print(
        f"\n   >>> RESID > 0 (unaccounted): {un} notes ({100*un/max(total,1):.3f}% of RESID, "
        f"{100*un/max(total,1):.4f}%) <<<"
    )
    print("\n=== worst composers by UNACCOUNTED count ===")
    for comp, n in by_comp.most_common(20):
        print(f"   {n:6d}  {comp}")
    print("\n=== UNACCOUNTED examples (offsets | ctrl) ===")
    for comp, offs, ctrls in examples[:30]:
        print(f"   {comp:18s} off={offs}  ctrl={ctrls}")
