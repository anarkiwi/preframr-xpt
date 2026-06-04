#!/usr/bin/env python3
"""Corpus-wide residual-SET census -- the acceptance gate for the residual-SET-elimination
work order: every raw SET surviving the residual macro arm is an unmodeled driver mechanism;
target is ZERO across the corpus (digi-excluded). Parallel driver around the same parse +
residual-arm flags as preframr-tokens' `residual_mechanism.py`, so the COUNT matches that
harness's definition of done; this tool adds corpus sampling, digi exclusion, per-item
progress, and a surviving-atom work-queue dump.

Run in the xpt image with the tokens source on the path, e.g.:
  docker run --rm --network host -v /scratch/anarkiwi/preframr-tokens:/tok \
    -v /scratch/preframr:/scratch/preframr -v <xpt>:/xpt -w /xpt \
    -e PYTHONPATH=/tok:/xpt anarkiwi/preframr-xpt:latest \
    python -m preframr_experiments.audit.residual_set_census --step 20 --workers 48

Acceptance: VERDICT line reports `residual_SETs=0` and `dirty_tunes=0` (no RARE survivors).
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
from collections import Counter
from multiprocessing import Pool

# The residual arm (mirror preframr-tokens residual_mechanism.py _BASE + _CODEBOOK). Kept here
# so the census is self-contained; if residual_mechanism.py's arm changes, update this list.
_BASE = (
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c4",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
)
_RESIDUAL_ARM = (
    "skeleton_pass",
    "held_arp",
    "zero_plain",
    "slide_wide",
    "slide_landing",
    "stamp_pass",
    "sweep_pass",
    "sweep_loop",
    "pw_sweep",
    "filter_sweep",
    "wavetable_pass",
    "wt_short",
    "wt_oneshot",
    "patch_pass",
    "ctrl_osc",
    "ctrl_wavetable",
    "env_wavetable",
    "filter_wavetable",
    "modevol_wavetable",
    "freq_wavetable",
    "pw_wavetable",
    "note_off",
    "envelope_osc",
    "init_preamble",
    "modevol_gradient",
    "env_gradient",
    "filter_gradient",
    "ctrl_gradient",
    "onset_instrument",
    "onset_def",
    "env_multiload",
    "pre_gate_freq",
    "nibble_wavetable",
)

_KW = None


def _kw():
    """Build the tokenizer kwargs once per worker; drop flags absent in this build (so the
    census runs both before and after the residual PRs ship)."""
    global _KW
    if _KW is None:
        from preframr_tokens.tokenizer_config import default_tokenizer_args

        want = list(_BASE) + list(_RESIDUAL_ARM)
        try:
            from preframr_tokens.macros.flag_registry import macro_flag_names

            known = set(macro_flag_names())
            dropped = [f for f in want if f not in known]
            if dropped:
                print(
                    f"  note: flags absent in this build (dropped): {sorted(dropped)}",
                    flush=True,
                )
            want = [f for f in want if f in known]
        except Exception:  # noqa: BLE001
            pass
        _KW = default_tokenizer_args(seq_len=4096, **{f: True for f in want})
    return _KW


def _is_digi(path):
    try:
        from preframr_tokens.dump_meta import meta_path_for, read_meta

        meta = read_meta(meta_path_for(path))
        return bool(getattr(meta, "is_digi", False))
    except Exception:  # noqa: BLE001
        return False


def check(path):
    """Return (name, kind, residual_count, sample_atoms). kind in {ok, digi, error}."""
    name = path.rsplit("/", 1)[-1]
    if _is_digi(path):
        return (name, "digi", 0, ())
    try:
        from preframr_tokens.reglogparser import RegLogParser
        from preframr_tokens.stfconstants import SET_OP

        df = next(RegLogParser(_kw()).parse(path, max_perm=1, require_pq=False))
    except StopIteration:
        return (name, "ok", 0, ())
    except Exception as e:  # noqa: BLE001
        return (name, "error", -1, (f"{type(e).__name__}: {e}",))
    ops = df["op"].to_numpy()
    regs = df["reg"].to_numpy()
    subs = df["subreg"].to_numpy() if "subreg" in df.columns else [-1] * len(df)
    vals = df["val"].to_numpy()
    count = 0
    sample = []
    for i in range(len(df)):
        if int(ops[i]) == SET_OP and 0 <= int(regs[i]) < 25:
            count += 1
            if len(sample) < 8:
                sample.append((int(regs[i]), int(subs[i]), int(vals[i])))
    return (name, "ok", count, tuple(sample))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="/scratch/preframr/hvsc/**/*.dump.parquet")
    ap.add_argument("--step", type=int, default=20, help="sample 1/step of the corpus")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument(
        "--outlier-mult",
        type=float,
        default=50.0,
        help="flag a tune as suspected-missed-digi if its residual > outlier_mult * median nonzero",
    )
    a = ap.parse_args()
    allf = sorted(glob.glob(a.glob, recursive=True))
    sample = allf[:: a.step]
    n = len(sample)
    print(
        f"residual-SET census: total={len(allf)} sample(1/{a.step})={n} workers={a.workers}",
        flush=True,
    )
    total = clean = digi = err = 0
    dirty = []  # (name, count, sample_atoms)
    errors = []
    with Pool(a.workers) as pool:
        for i, (name, kind, count, atoms) in enumerate(
            pool.imap_unordered(check, sample, chunksize=4), 1
        ):
            if kind == "digi":
                digi += 1
            elif kind == "error":
                err += 1
                errors.append((name, atoms[0] if atoms else "?"))
            else:
                if count == 0:
                    clean += 1
                else:
                    total += count
                    dirty.append((name, count, atoms))
            if i % 200 == 0:
                print(
                    f"[{i}/{n}] clean={clean} dirty={len(dirty)} digi={digi} err={err} residual_SETs={total}",
                    flush=True,
                )

    nonzero = [c for _, c, _ in dirty]
    med = statistics.median(nonzero) if nonzero else 0
    suspected = [(nm, c) for nm, c, _ in dirty if med and c > a.outlier_mult * med]
    print(
        f"\n===== VERDICT: residual_SETs={total} dirty_tunes={len(dirty)} "
        f"(clean={clean} digi_excluded={digi} errors={err} of n={n}) =====",
        flush=True,
    )
    if suspected:
        print(
            f"\nSUSPECTED MISSED DIGIS (residual > {a.outlier_mult}x median {med}; likely PWM digis "
            f"is_digi misses -- review/exclude, do not let them mask the real census):",
            flush=True,
        )
        for nm, c in sorted(suspected, key=lambda x: -x[1])[:20]:
            print(f"   {c:7d}  {nm}", flush=True)
    if dirty:
        print(
            f"\nWORK QUEUE -- top dirty tunes (name: residual_SETs; sample (reg,subreg,val)):",
            flush=True,
        )
        atom_hist = Counter()
        for nm, c, atoms in sorted(dirty, key=lambda x: -x[1])[:30]:
            print(f"   {c:7d}  {nm}  {list(atoms)}", flush=True)
        for _, _, atoms in dirty:
            for a_ in atoms:
                atom_hist[a_] += 1
        print(f"\nMost common surviving (reg,subreg,val):", flush=True)
        for atom, c in atom_hist.most_common(20):
            print(f"   {c:6d}  {atom}", flush=True)
    if errors:
        print(f"\nPARSE ERRORS ({len(errors)}):", flush=True)
        for nm, msg in errors[:20]:
            print(f"   {nm}: {msg}", flush=True)
    sys.exit(0 if (total == 0 and err == 0) else 1)


if __name__ == "__main__":
    main()
