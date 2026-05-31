"""Linchpin check for the wavetable-codebook primitive: do the post-stack RESID
offset-sequences RECUR across notes (codebook-able), or are they one-off?

`fit()`-UNRESOLVED only means "not a constant-delta ramp and not a period<=8 cycle".
A wavetable program is a one-shot/looping note-relative offset sequence replayed per
note -- non-periodic WITHIN a note but identical ACROSS notes. That cross-note
recurrence is what makes it a codebook DEF+REF (the pitched twin of STAMP), and it is
NOT what the per-note periodicity fitter measures. This probe measures it directly:
for the worst-residue engines, what fraction of RESID notes share an exact offset
sequence with another note (in the same tune), and how big is the per-tune codebook.

Usage: resid_wavetable_recurrence.py <engine> [n_tunes]"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
sys.path.insert(0, os.path.dirname(__file__))
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.macros.skeleton_pass import SkeletonPass  # noqa: E402
from parse_probes import parse_args  # noqa: E402
import sidid_cache  # noqa: E402

CORPUS = "/scratch/preframr/hvsc"


def _args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True,
                      stamp_pass=True, sweep_pass=True, patch_pass=True, held_arp=True)


def tunes_for_engine(engine, limit):
    labels = sidid_cache.load()
    hits = [p for p, e in labels.items() if e == engine]
    out = []
    for sidpath in sorted(hits):
        dump = sidpath[:-4] + ".1.dump.parquet"
        if os.path.exists(dump):
            out.append(dump)
        if len(out) >= limit:
            break
    return out


def resid_seqs(dump, a):
    SkeletonPass._resid_diag = []
    try:
        next(RegLogParser(args=a).parse(dump, max_perm=1, require_pq=False, reparse=True), None)
    except Exception:
        SkeletonPass._resid_diag = None
        return []
    recs = SkeletonPass._resid_diag or []
    SkeletonPass._resid_diag = None
    seqs = []
    for _reg, isr, _n, _on, rec in recs:
        if isr and rec:
            seqs.append(tuple(o for o, _c, _m, _fn in rec))
    return seqs


def main():
    engine = sys.argv[1] if len(sys.argv) > 1 else "GoatTracker_V2.x"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    a = _args()
    tot_notes = tot_recurring = 0
    codebook_sizes = []
    cover_top8 = []
    for dump in tunes_for_engine(engine, n):
        seqs = resid_seqs(dump, a)
        if not seqs:
            continue
        c = Counter(seqs)
        recurring = sum(v for v in c.values() if v >= 2)
        tot_notes += len(seqs)
        tot_recurring += recurring
        codebook_sizes.append(len(c))
        top8 = sum(v for _k, v in c.most_common(8))
        cover_top8.append(100 * top8 // max(len(seqs), 1))
    if not tot_notes:
        print(f"{engine}: no RESID notes in sample")
        return
    codebook_sizes.sort()
    med_cb = codebook_sizes[len(codebook_sizes) // 2] if codebook_sizes else 0
    print(f"=== {engine} ({len(codebook_sizes)} tunes, {tot_notes} RESID notes) ===")
    print(f"  RECUR>=2 (codebook-able): {100*tot_recurring//tot_notes}%  ({tot_recurring}/{tot_notes})")
    print(f"  distinct sequences/tune (codebook size): median {med_cb}, range {codebook_sizes[0]}-{codebook_sizes[-1]}")
    print(f"  top-8 sequences cover (per tune): {sorted(cover_top8)}")


if __name__ == "__main__":
    main()
