#!/usr/bin/env python3
"""CLI wrapper for the engine-fingerprint vector."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from preframr_tokens.engine_fingerprint import (  # pylint: disable=unused-import
    CTRL_2GRAM_DIM,
    CTRL_3GRAM_DIM,
    DEFAULT_FINGERPRINT_WRITES,
    DELTA_BUCKETS,
    DELTA_EDGES,
    FEATURE_DIM,
    FILTER_DIM,
    IDX_FILTER_TOUCH,
    REG_DENSITY_DIM,
    SLICE_CTRL_2GRAM,
    SLICE_CTRL_3GRAM,
    SLICE_DELTA,
    SLICE_REG_DENSITY,
    compute_fingerprint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parquet", type=Path, help="path to a .dump.parquet")
    parser.add_argument(
        "--n-writes",
        type=int,
        default=DEFAULT_FINGERPRINT_WRITES,
        help="number of leading register writes to fingerprint (default %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="dump JSON {path, n_writes, feature_dim, vector} to this path; "
        "stdout if omitted",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    vec = compute_fingerprint(args.parquet, n_writes=args.n_writes)
    if vec is None:
        logging.error("%s: fingerprint failed (empty / unreadable)", args.parquet)
        return 1
    payload = {
        "path": str(args.parquet),
        "n_writes": args.n_writes,
        "feature_dim": FEATURE_DIM,
        "vector": vec.tolist(),
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
