"""Instrumented repro of the v1 mine_motifs spin. Caps the greedy loop at a few
iterations and reports, per iteration: stream size N, distinct pairs, % of pairs
that touch a frame atom (rejected by the 0.23.0 guard), how deep the candidate
scan goes, how many O(N) _ncomposers calls that costs, and wall time."""

import sys
import time
from collections import Counter

import preframr_tokens.macros.motif_pass as mp
import preframr_tokens.motif_mine as mm
from preframr.args import add_args, apply_pipeline_spec_to_args
import argparse

ITER_CAP = 256
_real_ncomp = mp._ncomposers
_ncomp_calls = [0]


def _counting_ncomp(*a, **k):
    _ncomp_calls[0] += 1
    return _real_ncomp(*a, **k)


def instrumented_mine(streams, composers, k=256, min_count=3, min_composers=3):
    seqs = [[mp._as_atom(a) for a in s] for s in streams]

    def has_frame(sym):
        return False if isinstance(sym, int) else mp._is_frame_advance(sym)

    expand, merges, next_id = {}, [], 0
    for it in range(min(k, ITER_CAP)):
        t0 = time.time()
        _ncomp_calls[0] = 0
        cnt = Counter()
        for s in seqs:
            cnt.update(zip(s, s[1:]))
        if not cnt:
            print(f"iter {it}: empty Counter -> would break")
            break
        mc = cnt.most_common()
        frame_pairs = sum(1 for (a, b), _ in mc if has_frame(a) or has_frame(b))
        depth, picked = 0, None
        for (sym_a, sym_b), count in mc:
            depth += 1
            if count < min_count:
                break
            if has_frame(sym_a) or has_frame(sym_b):
                continue
            if _counting_ncomp(seqs, composers, sym_a, sym_b) < min_composers:
                continue
            picked = (sym_a, sym_b)
            break
        n_atoms = sum(len(s) for s in seqs)
        print(
            f"iter {it}: N={n_atoms} pairs={len(mc)} "
            f"frame_pairs={frame_pairs} ({100*frame_pairs/max(1,len(mc)):.1f}%) "
            f"top_count={mc[0][1]} scan_depth={depth} "
            f"ncomp_calls={_ncomp_calls[0]} picked={'yes' if picked else 'NONE'} "
            f"dt={time.time()-t0:.2f}s",
            flush=True,
        )
        if picked is None:
            break
        sym_a, sym_b = picked
        mid = next_id
        next_id += 1
        exp_a = expand[sym_a] if isinstance(sym_a, int) else [sym_a]
        exp_b = expand[sym_b] if isinstance(sym_b, int) else [sym_b]
        expand[mid] = exp_a + exp_b
        merges.append((sym_a, sym_b, mid))
        seqs = [mp._merge_run(s, sym_a, sym_b, mid) for s in seqs]
    print(f"(capped at {ITER_CAP} iters; real run does up to k={k})")
    return mp.MotifDict(merges, {mid: expand[mid] for _, _, mid in merges})


mm.mine_motifs = instrumented_mine

ap = add_args(argparse.ArgumentParser())
ap.add_argument("--motif-out")
ap.add_argument("--motif-k", type=int, default=256)
ap.add_argument("--motif-min-count", type=int, default=3)
ap.add_argument("--motif-min-composers", type=int, default=3)
ap.add_argument("--motif-mine-version", type=int, default=1)
args = ap.parse_args(sys.argv[1:])
apply_pipeline_spec_to_args(args)
t = time.time()
mm.mine_dict_from_dumps(
    args, args.reglogs, max_files=args.max_files, k=args.motif_k,
    min_count=args.motif_min_count, min_composers=args.motif_min_composers,
    version=args.motif_mine_version,
)
print(f"total (parse + {ITER_CAP} mine iters): {time.time()-t:.1f}s")
