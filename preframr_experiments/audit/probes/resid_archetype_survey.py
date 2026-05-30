"""Corpus survey of skeleton-encoding RESID-trajectory archetypes. Goal: RESID=0.
Every RESID note is a freq trajectory the current primitives miss -> classify the
archetype (in semitone AND freq-shape domains) so we can find the missing primitive,
and surface the composers/drivers that produce each so they can be traced."""
import os, sys, glob, random, statistics, traceback
from collections import Counter, defaultdict

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.macros import iter_self_contained_row_blocks  # noqa: E402
from preframr_tokens.stfconstants import (  # noqa: E402
    SKEL_OP, ORN_OP, ORN_SUBREG_TYPE, ORN_TYPE_RESID,
)
from parse_probes import parse_args  # noqa: E402


def _args():
    return parse_args(skeleton_pass=True, trajectory_anchor_pass=True)


def archetype(offs):
    n = len(offs)
    if n == 0:
        return "empty"
    distinct = len(set(offs))
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


def survey(paths):
    arche = Counter()
    by_comp = defaultdict(lambda: [0, 0])  # composer -> [orn, resid]
    examples = defaultdict(list)
    parsed_ok = 0
    a = _args()
    for p in paths:
        comp = p.split("/MUSICIANS/", 1)[-1].split("/")[1] if "/MUSICIANS/" in p else "?"
        try:
            parsed = next(RegLogParser(args=a).parse(p, max_perm=1, require_pq=False, reparse=True), None)
        except Exception:
            continue
        if parsed is None or "op" not in getattr(parsed, "columns", []):
            continue
        parsed_ok += 1
        for block in iter_self_contained_row_blocks(parsed, 999999, args=a):
            ops = block["op"].to_numpy(); subs = block["subreg"].to_numpy(); vals = block["val"].to_numpy()
            nn = len(block); i = 0
            while i < nn:
                if int(ops[i]) == ORN_OP and int(subs[i]) == ORN_SUBREG_TYPE:
                    t = int(vals[i]); offs = []; j = i + 1
                    while j < nn and int(ops[j]) == ORN_OP and int(subs[j]) != ORN_SUBREG_TYPE:
                        if int(subs[j]) == 1:
                            v = int(vals[j]); offs.append(v - 256 if v > 127 else v)
                        j += 1
                    by_comp[comp][0] += 1
                    if t == ORN_TYPE_RESID and offs:
                        by_comp[comp][1] += 1
                        k = archetype(offs)
                        arche[k] += 1
                        if len(examples[k]) < 4:
                            examples[k].append(offs[:16])
                    i = j; continue
                i += 1
    return arche, by_comp, examples, parsed_ok


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    random.seed(13)
    all_dumps = glob.glob("/scratch/preframr/hvsc/MUSICIANS/*/*/*.1.dump.parquet")
    # one random dump per composer dir for driver diversity
    by_dir = defaultdict(list)
    for d in all_dumps:
        by_dir[os.path.dirname(d)].append(d)
    sample = [random.choice(v) for v in by_dir.values()]
    random.shuffle(sample)
    sample = sample[:n]
    arche, by_comp, examples, ok = survey(sample)
    tot_orn = sum(c[0] for c in by_comp.values())
    tot_resid = sum(c[1] for c in by_comp.values())
    print(f"parsed {ok}/{len(sample)} dumps | total ORN notes={tot_orn} RESID={tot_resid} share={tot_resid/max(tot_orn,1):.3f}")
    print("\n=== RESID archetypes (corpus-wide) ===")
    for k, v in arche.most_common():
        print(f"  {v:5d} ({100*v//max(tot_resid,1):2d}%)  {k}")
        for ex in examples[k][:2]:
            print(f"            e.g. {ex}")
    print("\n=== worst composers by RESID share (>=20 notes) ===")
    worst = sorted(((c[1]/max(c[0],1), name, c[0], c[1]) for name, c in by_comp.items() if c[0] >= 20), reverse=True)
    for share, name, o, r in worst[:15]:
        print(f"  {share:.2f}  {name:24s} (orn={o} resid={r})")
