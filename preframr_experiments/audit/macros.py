#!/usr/bin/env python3
"""Per-macro-pass timing profiler."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preframr_tokens import macros as macros_mod  # noqa: E402
from preframr.args import add_args  # noqa: E402
from preframr_tokens import iter_voiced_blocks  # noqa: E402
from preframr_tokens import RegLogParser  # noqa: E402


class _Stats:
    def __init__(self):
        self.total = 0.0
        self.calls = 0


def _instrument(passes, stats):
    """Wrap each pass's ``apply`` so timing accumulates in ``stats``."""
    for p in passes:
        name = type(p).__name__
        stats.setdefault(name, _Stats())
        original = p.apply

        def wrapped(df, args=None, _orig=original, _name=name, _s=stats):
            t0 = time.perf_counter()
            try:
                return _orig(df, args=args)
            finally:
                _s[_name].total += time.perf_counter() - t0
                _s[_name].calls += 1

        p.apply = wrapped


def _instrument_fn(module, name, stats):
    """Wrap a module-level function (e.g. ``expand_to_literal_form``)."""
    stats.setdefault(name, _Stats())
    original = getattr(module, name)

    def wrapped(*a, _orig=original, _name=name, _s=stats, **kw):
        t0 = time.perf_counter()
        try:
            return _orig(*a, **kw)
        finally:
            _s[_name].total += time.perf_counter() - t0
            _s[_name].calls += 1

    setattr(module, name, wrapped)


def _instrument_method(cls, name, stats):
    """Wrap a class method (e.g. ``RegLogParser._consolidate_frames``)."""
    label = f"{cls.__name__}.{name}"
    stats.setdefault(label, _Stats())
    original = getattr(cls, name)

    def wrapped(self, *a, _orig=original, _name=label, _s=stats, **kw):
        t0 = time.perf_counter()
        try:
            return _orig(self, *a, **kw)
        finally:
            _s[_name].total += time.perf_counter() - t0
            _s[_name].calls += 1

    setattr(cls, name, wrapped)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.add_argument("dump_files", nargs="+")
    ap.add_argument(
        "--blocks",
        action="store_true",
        help=(
            "Also walk iter_voiced_blocks (which calls run_passes per "
            "block); without --blocks only the full-song parse is timed."
        ),
    )
    args = ap.parse_args()

    stats = {}
    _instrument(macros_mod.PASSES, stats)
    _instrument(macros_mod.POST_NORM_PRE_VOICE_PASSES, stats)
    _instrument_fn(macros_mod, "expand_to_literal_form", stats)
    _instrument_fn(macros_mod, "iter_self_contained_row_blocks", stats)
    _instrument_method(RegLogParser, "_consolidate_frames", stats)
    _instrument_method(RegLogParser, "_add_voice_reg", stats)
    _instrument_method(RegLogParser, "_remove_voice_reg", stats)

    parser = RegLogParser(args)
    block_parser = RegLogParser(args)
    seq_len = args.seq_len
    stride = getattr(args, "block_stride", None)

    n_rotations = 0
    n_blocks = 0
    parse_t = 0.0
    block_t = 0.0
    for path in args.dump_files:
        print(f"\n[{os.path.basename(path)}]", flush=True)
        t_parse = time.perf_counter()
        for df in parser.parse(path, max_perm=args.max_perm, require_pq=False):
            parse_t += time.perf_counter() - t_parse
            n_rotations += 1
            if args.blocks:
                t_blk = time.perf_counter()
                for _ in iter_voiced_blocks(
                    df, seq_len, block_parser, {}, stride=stride
                ):
                    n_blocks += 1
                block_t += time.perf_counter() - t_blk
            t_parse = time.perf_counter()

    rows = sorted(stats.items(), key=lambda kv: kv[1].total, reverse=True)
    grand_total = sum(s.total for s in stats.values())
    print(
        f"\n--- macro-pass timing: {n_rotations} rotation(s), "
        f"{n_blocks} block(s) ---",
        flush=True,
    )
    print(f"{'pass':30s} {'total(s)':>10s} {'%':>7s} {'calls':>10s} {'us/call':>10s}")
    print("-" * 75)
    for name, s in rows:
        if not s.calls:
            continue
        pct = 100.0 * s.total / grand_total if grand_total else 0
        per_us = (s.total / s.calls) * 1e6 if s.calls else 0
        print(f"{name:30s} {s.total:10.3f} {pct:6.1f}% {s.calls:10d} {per_us:10.0f}")
    print("-" * 75)
    print(f"{'TOTAL pass time':30s} {grand_total:10.3f}")
    print(f"  parse() wall total            {parse_t:10.3f}")
    if args.blocks:
        print(f"  iter_voiced_blocks total      {block_t:10.3f}")


if __name__ == "__main__":
    main()
