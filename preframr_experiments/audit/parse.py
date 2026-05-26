#!/usr/bin/env python3
"""cProfile a full RegLogParser.parse() + iter_voiced_blocks for one dump."""

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preframr.args import add_args  # noqa: E402
from preframr_tokens import iter_voiced_blocks  # noqa: E402
from preframr_tokens import RegLogParser  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("dump_file")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--seq-len-override", type=int, default=8192)
    ap.add_argument(
        "--reparse",
        action="store_true",
        help="Force re-parse from raw dump (bypass cached PARSED_SUFFIX parquets).",
    )
    args = ap.parse_args()

    parser = RegLogParser(args)
    block_parser = RegLogParser(args)
    seq_len = args.seq_len_override

    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    n_dfs = 0
    n_blocks = 0
    for df in parser.parse(
        args.dump_file, max_perm=3, require_pq=False, reparse=args.reparse
    ):
        t1 = time.perf_counter()
        rotation_blocks = list(
            iter_voiced_blocks(df, seq_len, block_parser, {}, stride=args.block_stride)
        )
        t2 = time.perf_counter()
        n_dfs += 1
        n_blocks += len(rotation_blocks)
        print(
            f"rotation {n_dfs}: parse=until-now {t1-t0:.2f}s, "
            f"blocks: {len(rotation_blocks)} in {t2-t1:.2f}s, df rows={len(df)}"
        )
        t0 = t2
    pr.disable()
    print(f"\ntotal rotations: {n_dfs}, total blocks: {n_blocks}")
    stats = pstats.Stats(pr).sort_stats("cumulative")
    stats.print_stats(args.top)
    print("\n--- by tottime ---")
    stats = pstats.Stats(pr).sort_stats("tottime")
    stats.print_stats(args.top)


if __name__ == "__main__":
    main()
