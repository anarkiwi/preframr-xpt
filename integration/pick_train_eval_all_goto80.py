#!/usr/bin/env python3
"""Pick train/eval Goto80 SIDs across the entire catalogue."""

import os
import re
import sys

HVSC = "/scratch/hvsc"
GOTO80 = f"{HVSC}/MUSICIANS/G/Goto80"
DB = f"{HVSC}/DOCUMENTS/Songlengths.md5"

PRIOR_EVAL = {"Robinson.sid", "Afternorm.sid", "Feddamys.sid", "Oj.sid"}

VARIANT_RE = re.compile(
    r"(_v[0-9]+|_[0-9]+|_Prv|_Preview|_intro|_title|_note|_tune_[0-9]+|"
    r"_preview|_tune_[a-z]+)\.sid$",
    re.IGNORECASE,
)

N_EVAL = 16


def parse_duration_db(path):
    out = {}
    with open(path, "rb") as f:
        lines = f.read().decode("latin-1").splitlines()
    for i, line in enumerate(lines):
        line = line.rstrip("\r")
        if not line.startswith("; "):
            continue
        sid_path = line[2:]
        if i + 1 >= len(lines):
            continue
        nxt = lines[i + 1].rstrip("\r")
        if "=" not in nxt:
            continue
        dur = nxt.split("=", 1)[1].split()[0]
        m, sep, rest = dur.partition(":")
        if not sep:
            continue
        s = rest.split(".")[0]
        try:
            out[sid_path] = int(m) * 60 + int(s)
        except ValueError:
            continue
    return out


def base_name(fn):
    """Strip variant suffix to get the canonical base name."""
    return VARIANT_RE.sub(".sid", fn)


def main():
    db = parse_duration_db(DB)
    canonicals = []
    variants = []
    for fn in sorted(os.listdir(GOTO80)):
        if not fn.endswith(".sid"):
            continue
        sec = db.get(f"/MUSICIANS/G/Goto80/{fn}")
        if sec is None or sec < 30:
            continue
        if VARIANT_RE.search(fn):
            variants.append((fn, sec))
        else:
            canonicals.append((fn, sec))

    print(
        f"canonicals={len(canonicals)} variants={len(variants)}",
        file=sys.stderr,
    )

    canonicals.sort(key=lambda fs: (fs[1], fs[0]))
    eval_picks = [(fn, s) for fn, s in canonicals if fn in PRIOR_EVAL]
    missing_priors = PRIOR_EVAL - {fn for fn, _ in eval_picks}
    if missing_priors:
        sys.exit(f"prior eval SIDs not in candidate set: {missing_priors}")
    rest = [(fn, s) for fn, s in canonicals if fn not in PRIOR_EVAL]
    n_more = N_EVAL - len(eval_picks)
    step = len(rest) / n_more
    extra_idxs = {int(i * step) for i in range(n_more)}
    extra = [rest[i] for i in sorted(extra_idxs)]
    eval_picks.extend(extra)
    eval_canonicals = {fn for fn, _ in eval_picks}

    train_canonicals = [(fn, s) for fn, s in canonicals if fn not in eval_canonicals]
    train_canonical_names = {fn for fn, _ in train_canonicals}

    train_variants = []
    dropped_variants = []
    for fn, s in variants:
        base = base_name(fn)
        if base in eval_canonicals:
            dropped_variants.append((fn, s, base))
        elif base in train_canonical_names:
            train_variants.append((fn, s))
        else:
            train_variants.append((fn, s))

    train_all = train_canonicals + train_variants
    train_all.sort(key=lambda fs: fs[0])
    eval_picks.sort(key=lambda fs: fs[0])

    print(
        f"train={len(train_all)} (canonical={len(train_canonicals)} "
        f"variant={len(train_variants)}) eval={len(eval_picks)} "
        f"dropped_variants={len(dropped_variants)}",
        file=sys.stderr,
    )
    if dropped_variants:
        print("# dropped (variants of eval canonicals):", file=sys.stderr)
        for fn, s, base in dropped_variants:
            print(f"#   {fn} ({s}s, base={base})", file=sys.stderr)

    print(f"# {len(train_all)} train SIDs", file=sys.stderr)
    print('TRAIN_SIDS="')
    for fn, sec in train_all:
        print(f"  MUSICIANS/G/Goto80/{fn}  # {sec}s")
    print('"')
    print()
    print(f"# {len(eval_picks)} eval SIDs", file=sys.stderr)
    print('EVAL_SIDS="')
    for fn, sec in eval_picks:
        print(f"  MUSICIANS/G/Goto80/{fn}  # {sec}s")
    print('"')


if __name__ == "__main__":
    main()
