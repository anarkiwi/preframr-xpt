"""Full-corpus RESID=0 survey. Parses EVERY canonical .1.dump across MUSICIANS+DEMOS+GAMES once with
wavetable_pass ON, and per engine reports: residual ORN-RESID notes, drained notes (each WAVETABLE_REF
replaces one RESID note, so drain% = REF/(REF+RESID) -- no second parse needed), and a classification of
what SURVIVES as RESID (the tail to close). A 1-in-VERIFY_EVERY subsample additionally re-parses with the
pass OFF and asserts register_state OFF==ON (byte-exact corpus-wide corruption signal).

Survivor classes (offset-only; the ORN atom stores note-relative offsets):
  RECUR  -- exact offset seq appears >=2x in this tune (codebook TUNING bug: should have drained)
  STRUCT -- factorise finds a loop body but unique -> inline one-shot SHOULD emit
  PERIOD -- period<=8 after onset-strip (held-ARP/codebook should catch)
  SWEEP  -- constant-delta ramp (SweepPass / wavetable-sweep gap)
  ZERO   -- all offsets 0 (unresolvable/noise frames: timbre, not pitch)
  SHORT  -- core length < 2
  FLAT   -- unique, non-periodic, no loop body (genuine flat one-shot: needs inline emit)

Each worker prints a progress line every PROGRESS_EVERY tunes to stderr; the parent writes a partial
checkpoint to <out>.partial every CKPT_SECONDS. Usage: resid_corpus_survey.py <N|all|paths> [procs] [out]
"""

import os
import sys

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")
import glob  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens/tests")
sys.path.insert(0, "/scratch/anarkiwi/preframr-tokens")
sys.path.insert(0, os.path.dirname(__file__))
from preframr_tokens.reglogparser import RegLogParser  # noqa: E402
from preframr_tokens.audit_primitives import register_state  # noqa: E402
from preframr_tokens.macros.wavetable import factorise  # noqa: E402
from preframr_tokens.stfconstants import (  # noqa: E402
    ORN_OP,
    ORN_SUBREG_TYPE,
    ORN_SUBREG_P1,
    ORN_SUBREG_P2,
    ORN_TYPE_RESID,
    WAVETABLE_DEF_OP,
    WAVETABLE_REF_OP,
    WT_REF_SUBREG_ID,
    FREQ_TRAJ_REGS,
)
from parse_probes import parse_args  # noqa: E402
import sidid_cache  # noqa: E402
import numpy as np  # noqa: E402

CORPUS = "/scratch/preframr/hvsc"
BASE = dict(
    skeleton_pass=True,
    trajectory_anchor_pass=True,
    stamp_pass=True,
    sweep_pass=True,
    patch_pass=True,
    held_arp=True,
)
# Full Phase-4 deployed stack (W1-W5, the intended W7 default-ON set): wavetable codebook + ZERO->PLAIN
# (W1) + short literal codebook (W2) + inline one-shot RESID=0 backstop (W3) + wide-ramp SLIDE (W4) +
# exact-landing SLIDE2 (W5.1) + looping SWEEP (W5.3). The OFF baseline is BASE alone (no drain gates).
STACK = dict(
    wavetable_pass=True,
    zero_plain=True,
    wt_short=True,
    wt_oneshot=True,
    slide_wide=True,
    slide_landing=True,
    sweep_loop=True,
)
_FREQ = {int(r) for r in FREQ_TRAJ_REGS}
PROGRESS_EVERY = 200
VERIFY_EVERY = 50
CLASSES = ["RECUR", "STRUCT", "PERIOD", "SWEEP", "ZERO", "SHORT", "FLAT"]


def parse_df(dump, **flags):
    a = parse_args(**{**BASE, **flags})
    return next(
        RegLogParser(args=a).parse(dump, max_perm=1, require_pq=False, reparse=True),
        None,
    )


def resid_offsets(df):
    if df is None or "op" not in getattr(df, "columns", []):
        return [], 0
    op = df["op"].to_numpy()
    reg = df["reg"].to_numpy()
    sub = df["subreg"].to_numpy()
    val = df["val"].to_numpy()
    refs = int(((op == WAVETABLE_REF_OP) & (sub == WT_REF_SUBREG_ID)).sum())
    n = len(df)
    out = []
    i = 0
    while i < n:
        if not (
            op[i] == ORN_OP
            and sub[i] == ORN_SUBREG_TYPE
            and val[i] == ORN_TYPE_RESID
            and int(reg[i]) in _FREQ
        ):
            i += 1
            continue
        r = int(reg[i])
        offs = []
        length = None
        j = i + 1
        while j < n and op[j] == ORN_OP and int(reg[j]) == r:
            if sub[j] == ORN_SUBREG_P2:
                length = int(val[j]) & 0xFFFF
            elif sub[j] == ORN_SUBREG_P1:
                v = int(val[j]) & 0xFF
                offs.append(v if v < 128 else v - 256)
            j += 1
        i = j
        if length is not None and len(offs) == length:
            out.append(tuple(offs))
    return out, refs


def _const_delta(xs, tol=1):
    if len(xs) < 3:
        return False
    d = [b - a for a, b in zip(xs, xs[1:])]
    nz = [x for x in d if abs(x) > tol]
    return len(nz) >= 2 and (max(d) - min(d)) <= max(2, abs(sum(nz) // len(nz)) // 4)


def _period(xs, onset_strip=2, maxp=8):
    for strip in range(onset_strip + 1):
        s = xs[strip:]
        if len(s) < 6:
            continue
        n = len(s)
        for p in range(1, min(maxp, n // 2) + 1):
            if all(s[k] == s[k % p] for k in range(n)):
                return True
    return False


def classify(seq, tune_counts):
    if tune_counts[seq] >= 2:
        return "RECUR"
    if not any(seq):
        return "ZERO"
    if _const_delta(list(seq)):
        return "SWEEP"
    if _period(list(seq)):
        return "PERIOD"
    if len(seq) < 2:
        return "SHORT"
    steps, loop = factorise(list(seq))
    return "STRUCT" if loop < len(steps) else "FLAT"


_DMAP = {}


def analyze(args):
    paths, wid = args
    dmap = _DMAP
    drain = defaultdict(
        lambda: {"resid": 0, "drained": 0, "tunes": 0, "bad": 0, "verified": 0}
    )
    tail = defaultdict(Counter)
    examples = defaultdict(list)
    done = 0
    for p in paths:
        done += 1
        if done % PROGRESS_EVERY == 0:
            tot = sum(d["resid"] for d in drain.values())
            print(
                f"[w{wid}] {done}/{len(paths)} resid={tot}", file=sys.stderr, flush=True
            )
        tune = os.path.basename(p).split(".")[0].lower()
        eng = dmap.get(os.path.dirname(p), {}).get(tune, "?")
        try:
            d_on = parse_df(p, **STACK)
        except Exception:
            continue
        on, refs = resid_offsets(d_on)
        drain[eng]["tunes"] += 1
        drain[eng]["resid"] += len(on)
        drain[eng]["drained"] += refs
        tc = Counter(on)
        for seq in on:
            c = classify(seq, tc)
            tail[eng][c] += 1
            if len(examples[(eng, c)]) < 3:
                examples[(eng, c)].append(list(seq)[:16])
        if (hash(p) % VERIFY_EVERY) == 0:
            try:
                d_off = parse_df(p)
                drain[eng]["verified"] += 1
                if not np.array_equal(register_state(d_off), register_state(d_on)):
                    drain[eng]["bad"] += 1
            except Exception:
                drain[eng]["bad"] += 1
    return (
        dict(drain),
        {k: dict(v) for k, v in tail.items()},
        {f"{e}|{c}": v for (e, c), v in examples.items()},
    )


def merge(results):
    drain = defaultdict(
        lambda: {"resid": 0, "drained": 0, "tunes": 0, "bad": 0, "verified": 0}
    )
    tail = defaultdict(Counter)
    examples = {}
    for dr, tl, ex in results:
        for e, d in dr.items():
            for k in d:
                drain[e][k] += d[k]
        for e, t in tl.items():
            tail[e].update(t)
        for k, v in ex.items():
            examples.setdefault(k, v)
    return drain, tail, examples


def render(drain, tail, examples, sample_n):
    lines = []
    lines.append(
        f"\n=== FULL-CORPUS RESID=0 SURVEY ({sample_n} tunes, full Phase-4 stack ON) ==="
    )
    hdr = (
        f"{'engine':30s} {'tunes':>6} {'resid':>7} {'drained':>7} {'drain%':>6} "
        f"{'vrf':>5} {'bad':>4}  " + " ".join(f"{c:>6}" for c in CLASSES)
    )
    lines.append(hdr)
    gr = gd = gbad = gver = 0
    for e in sorted(drain, key=lambda e: -drain[e]["resid"]):
        d = drain[e]
        denom = d["resid"] + d["drained"]
        dr = 100 * d["drained"] // max(denom, 1)
        cells = " ".join(f"{tail[e].get(c, 0):>6}" for c in CLASSES)
        lines.append(
            f"{e:30s} {d['tunes']:>6} {d['resid']:>7} {d['drained']:>7} {dr:>5}% "
            f"{d['verified']:>5} {d['bad']:>4}  {cells}"
        )
        gr += d["resid"]
        gd += d["drained"]
        gbad += d["bad"]
        gver += d["verified"]
    gdenom = gr + gd
    lines.append(
        f"\nTOTAL residual_resid={gr} drained={gd} drain={100*gd//max(gdenom,1)}% "
        f"verified={gver} corruptions={gbad}"
    )
    allt = Counter()
    for t in tail.values():
        allt.update(t)
    lines.append(f"TAIL classes (all engines): {dict(allt.most_common())}")
    lines.append("\n=== tail examples (engine|class -> offset seqs) ===")
    for e in sorted(drain, key=lambda e: -drain[e]["resid"])[:12]:
        for c in CLASSES:
            k = f"{e}|{c}"
            if k in examples and tail[e].get(c, 0):
                lines.append(f"  {e:28s} {c:7s} n={tail[e][c]:<5} {examples[k]}")
    return "\n".join(lines)


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "all"
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 22
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/corpus_survey.txt"
    random.seed(13)
    if "/" in spec:
        sample = [p for p in spec.split(",") if os.path.exists(p)]
    else:
        sample = []
        for top in ("MUSICIANS", "DEMOS", "GAMES"):
            sample += glob.glob(f"{CORPUS}/{top}/**/*.1.dump.parquet", recursive=True)
        sample.sort()
        random.shuffle(sample)
        if spec != "all":
            sample = sample[: int(spec)]
    print(
        f"surveying {len(sample)} tunes on {procs} procs -> {out}",
        file=sys.stderr,
        flush=True,
    )
    full = sidid_cache.by_dir()
    global _DMAP
    _DMAP = {os.path.dirname(p): full.get(os.path.dirname(p), {}) for p in sample}
    chunk_target = 40
    n_chunks = max(procs, (len(sample) + chunk_target - 1) // chunk_target)
    shards = [(sample[i::n_chunks], i) for i in range(n_chunks)]
    shards = [s for s in shards if s[0]]

    import multiprocessing as mp

    t0 = time.time()
    results = []
    n_done_tunes = 0
    n_chunks = len(shards)
    with mp.Pool(min(procs, n_chunks)) as pool:
        for ci, res in enumerate(pool.imap_unordered(analyze, shards), 1):
            results.append(res)
            n_done_tunes += sum(d["tunes"] for d in res[0].values())
            el = int(time.time() - t0)
            rate = n_done_tunes / max(el, 1)
            eta = int((len(sample) - n_done_tunes) / max(rate, 0.01))
            drain, tail, examples = merge(results)
            gr = sum(d["resid"] for d in drain.values())
            gd = sum(d["drained"] for d in drain.values())
            gbad = sum(d["bad"] for d in drain.values())
            print(
                f"[parent] chunk {ci}/{n_chunks} tunes={n_done_tunes}/{len(sample)} "
                f"resid={gr} drained={gd} bad={gbad} {el}s rate={rate:.1f}/s eta={eta}s",
                file=sys.stderr,
                flush=True,
            )
            with open(out + ".partial", "w") as f:
                f.write(render(drain, tail, examples, n_done_tunes) + "\n")
    drain, tail, examples = merge(results)
    report = render(drain, tail, examples, len(sample))
    with open(out, "w") as f:
        f.write(report + "\n")
        f.write(
            "\n# raw per-engine json:\n# "
            + json.dumps({e: dict(d) for e, d in drain.items()})
            + "\n"
        )
    print(report)
    print(f"\n[wrote {out} in {int(time.time()-t0)}s]", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
