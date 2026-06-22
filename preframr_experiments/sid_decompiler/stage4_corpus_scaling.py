"""Stage 4 -- corpus scaling test on depacker-less Goto80 drivers (design 6).

The DECISIVE metric this whole design exists to pass: on real depacker-less
drivers (JCH NewPlayer + DefMon), the recovered fixed-size ``{G_A}`` must stay
FLAT as the capture/fit window grows -- in direct contrast to the prior
output-shaped catalog that went 10.5k -> 34k tokens as the window grew.

There is NO depacker oracle for these drivers: the sole gate is residual-zero on
the byte-exact SID register stream (design 2.7). We measure, per tune:

  * residual(recovered subset) over the FULL stream -- must be 0 (the gate);
  * whole-tune residual -- the honestly-surfaced axes (design 5);
  * recovered-program size vs FIT WINDOW (N = 512..full) -- slope must be ~0;
  * SDST artifact (distill) size vs CAPTURE LENGTH N -- must be bounded/flat;
  * fraction of axes closed as recurrences vs surfaced.

This is a MEASUREMENT/REPORT harness only. It does NOT modify the recovery
algorithms (generalize.py / sdst.py) -- it drives them as a black box, exactly as
``recover_program`` exposes them. Run it against a directory of tracer artifacts
produced by ``preframr-sidtrace`` (``<prefix>.distill.bin`` + ``<prefix>.sidwr.bin``).

Usage:
    python -m preframr_experiments.sid_decompiler.stage4_corpus_scaling \
        <trace_dir> [--json out.json]

where ``<trace_dir>`` holds, per tune, ``<name>_N<frames>.distill.bin`` and
``<name>_N<frames>.sidwr.bin`` over several capture lengths N.
"""

import argparse
import glob
import json
import os
import re

import numpy as np

from . import sdst as S
from .generalize import recover_program_from, reference_from_sidwr, residual

# Fit windows for the host-side flat-slope test (design 6 "N=512..full").
FIT_WINDOWS = (512, 1024, 2048, 4096)

_NAME_RE = re.compile(r"^(?P<name>.+)_N(?P<n>\d+)\.distill\.bin$")


def discover(trace_dir):
    """Group artifacts by tune name -> {capture_N: prefix}."""
    tunes = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "*.distill.bin"))):
        m = _NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        name, n = m.group("name"), int(m.group("n"))
        prefix = path[: -len(".distill.bin")]
        if not os.path.exists(prefix + ".sidwr.bin"):
            continue
        tunes.setdefault(name, {})[n] = prefix
    return tunes


def artifact_size_vs_n(captures):
    """SDST distill size (bytes) per capture length N -- design 3 boundedness."""
    return {n: os.path.getsize(prefix + ".distill.bin")
            for n, prefix in sorted(captures.items())}


def recovered_size_vs_window(prefix):
    """Drive the host generalizer over a single (largest-N) artifact at several FIT
    windows. The artifact is FIXED; only how many frames we fit the recurrences
    from changes. Returns per-window measurements + the full-length verification.

    This is the anti-length-proportionality proof: the recovered fixed-size object
    is derived from a SHORT window and must reproduce the FULL stream residual-zero
    (recovered subset), with its byte-size NOT growing as the window grows."""
    art = S.parse_sdst(prefix + ".distill.bin")
    reference = reference_from_sidwr(prefix + ".sidwr.bin")
    nframes = len(reference)
    windows = [w for w in FIT_WINDOWS if w < nframes] + [nframes]
    rows = []
    for w in windows:
        prog, ref_a, phase, info = recover_program_from(
            art, reference, window=w, verify_full=True
        )
        rec = prog.recovered_regs()
        unrec = prog.unrecovered_regs()
        # residual of the recovered subset over the FULL aligned stream (the gate)
        rendered = prog.render(len(ref_a), phase)
        rows.append({
            "fit_window": w,
            "ref_frames": len(ref_a),
            "recovered_size_bytes": info["size_bytes"],
            "n_recovered_axes": len(rec),
            "n_surfaced_axes": len(unrec),
            "recovered_regs": list(rec),
            "surfaced_regs": list(unrec),
            "residual_recovered_subset_full": int(residual(rendered, ref_a, regs=rec)),
            "residual_whole_tune_full": int(residual(rendered, ref_a)),
            "residual_surfaced_subset_full": int(
                residual(rendered, ref_a, regs=unrec)) if unrec else 0,
        })
    return nframes, rows


def n_state_cells(prefix):
    art = S.parse_sdst(prefix + ".distill.bin")
    return len(art.siddf), len(art.sdcu), len(art.stateseq)


def analyze_tune(name, captures):
    largest_n = max(captures)
    prefix = captures[largest_n]
    art_sizes = artifact_size_vs_n(captures)
    nframes, win_rows = recovered_size_vs_window(prefix)
    n_siddf, n_sdcu, n_stsq = n_state_cells(prefix)
    sizes = [r["recovered_size_bytes"] for r in win_rows]
    full_row = win_rows[-1]
    short_row = win_rows[0]
    return {
        "name": name,
        "captures_N": sorted(captures),
        "largest_N_frames_rendered": nframes,
        "artifact_size_vs_N": art_sizes,
        "artifact_size_slope_bytes_per_frame": _slope(art_sizes),
        "n_siddf_sites": n_siddf,
        "n_sdcu_cells": n_sdcu,
        "n_stateseq_cells": n_stsq,
        "recovered_size_vs_window": win_rows,
        "recovered_size_min": min(sizes),
        "recovered_size_max": max(sizes),
        "recovered_size_flat": (max(sizes) - min(sizes)) == 0,
        # The headline: recovered subset residual-zero AND size flat from a short fit.
        "short_window_recovered_subset_residual": short_row[
            "residual_recovered_subset_full"],
        "full_recovered_subset_residual": full_row["residual_recovered_subset_full"],
        "whole_tune_residual": full_row["residual_whole_tune_full"],
        "n_recovered_axes": full_row["n_recovered_axes"],
        "n_surfaced_axes": full_row["n_surfaced_axes"],
        "frac_axes_closed": full_row["n_recovered_axes"]
        / max(1, full_row["n_recovered_axes"] + full_row["n_surfaced_axes"]),
    }


def _slope(size_by_n):
    """Least-squares slope of artifact bytes vs N (bytes/frame). ~0 == flat."""
    ns = np.array(sorted(size_by_n), dtype=float)
    ys = np.array([size_by_n[int(n)] for n in ns], dtype=float)
    if len(ns) < 2:
        return 0.0
    return float(np.polyfit(ns, ys, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tunes = discover(args.trace_dir)
    if not tunes:
        raise SystemExit(f"no <name>_N<frames>.distill.bin artifacts in {args.trace_dir}")

    results = [analyze_tune(name, caps) for name, caps in sorted(tunes.items())]

    # ---- Headline table 1: SDST artifact size vs capture length N (boundedness).
    print("=" * 78)
    print("TABLE 1: SDST artifact (distill) size vs CAPTURE LENGTH N  (must be flat)")
    print("=" * 78)
    all_n = sorted({n for r in results for n in r["artifact_size_vs_N"]})
    hdr = "tune".ljust(20) + "".join(f"N={n:>7}" for n in all_n) + "  slope(B/frame)"
    print(hdr)
    for r in results:
        row = r["name"].ljust(20)
        for n in all_n:
            v = r["artifact_size_vs_N"].get(n)
            row += f"{v:>9}" if v is not None else " " * 9
        row += f"   {r['artifact_size_slope_bytes_per_frame']:+.4f}"
        print(row)

    # ---- Headline table 2: recovered-program size vs FIT window (THE proof).
    print()
    print("=" * 78)
    print("TABLE 2: recovered {G_A} size (bytes) vs FIT WINDOW  (THE flat-slope proof)")
    print("    -- contrast the prior output catalog: 10.5k -> 34k tokens vs window")
    print("=" * 78)
    wins = sorted({r2["fit_window"]
                   for r in results for r2 in r["recovered_size_vs_window"]})
    print("tune".ljust(20) + "".join(f"w={w:>6}" for w in wins) + "   flat?")
    for r in results:
        by_w = {x["fit_window"]: x["recovered_size_bytes"]
                for x in r["recovered_size_vs_window"]}
        row = r["name"].ljust(20)
        for w in wins:
            v = by_w.get(w)
            row += f"{v:>8}" if v is not None else " " * 8
        row += f"    {'YES' if r['recovered_size_flat'] else 'NO'}"
        print(row)

    # ---- Headline table 3: residual-zero gate + axis closure.
    print()
    print("=" * 78)
    print("TABLE 3: residual-zero gate (the SOLE oracle) + axis closure per tune")
    print("=" * 78)
    print("tune".ljust(20) + "rec/tot  closed%  resid(rec)  resid(whole)  "
          "siddf/sdcu/stsq")
    for r in results:
        tot = r["n_recovered_axes"] + r["n_surfaced_axes"]
        print(
            r["name"].ljust(20)
            + f"{r['n_recovered_axes']:>2}/{tot:<2}    "
            + f"{100 * r['frac_axes_closed']:>5.1f}   "
            + f"{r['full_recovered_subset_residual']:>9}   "
            + f"{r['whole_tune_residual']:>11}    "
            + f"{r['n_siddf_sites']}/{r['n_sdcu_cells']}/{r['n_stateseq_cells']}"
        )

    # ---- Sanity: short-window fit reproduces full stream residual-zero (rec subset).
    print()
    print("residual-zero check: recovered subset from SHORT fit window vs FULL stream")
    for r in results:
        sw = r["recovered_size_vs_window"][0]
        ok = sw["residual_recovered_subset_full"] == 0
        print(f"  {r['name'].ljust(20)} fit_w={sw['fit_window']:>5} -> "
              f"resid(rec, full)={sw['residual_recovered_subset_full']} "
              f"{'PASS' if ok else 'FAIL'}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
