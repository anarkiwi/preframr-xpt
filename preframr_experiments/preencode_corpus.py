"""Fault-tolerant, in-place corpus pre-encoder for the event model.

Computes each in-scope dump's tkvocab-independent atom-id token stream and writes it to the
codec-version-keyed ``.atoms.zst`` sidecar next to the dump
(``preframr_tokens.events.dataset.dump_token_ids``). A later experiment run reuses the sidecar
instead of re-running ``stream.encode`` + its self-verify, so a tkvocab sweep skips the encode and
only retrains BPE.

Mirrors the runtime events scope filter (single-speed, non-digi) so only dumps the trainer would use
are encoded, and skips + logs per-dump failures (one bad dump never aborts the corpus). Re-run after
an HVSC upgrade; ``--only-missing`` skips dumps whose current-version sidecar already exists. Intended
to run inside the preframr image on fogbank, where the corpus physically lives.
"""

import argparse
import collections
import concurrent.futures
import glob as _glob
import logging
import multiprocessing
import os
import time

import pandas as pd

from preframr_tokens.dump_meta import raw_is_digi, read_meta
from preframr_tokens.events import oracle as events_oracle
from preframr_tokens.events import stream as events_stream
from preframr_tokens.events.dataset import ATOM_CACHE_VERSION, dump_token_ids
from preframr_tokens.stfconstants import DUMP_SUFFIX

_COLUMNS = ["clock", "irq", "chipno", "reg", "val"]


def _cache_path(df_file):
    return os.path.realpath(df_file).replace(
        DUMP_SUFFIX, f".{ATOM_CACHE_VERSION}.atoms.zst"
    )


def _cache_current(df_file):
    cache = _cache_path(df_file)
    return os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(
        df_file
    )


def _encode_one(df_file):
    """Encode one in-scope dump's atom stream into its sidecar; returns a status string."""
    df = pd.read_parquet(df_file, columns=_COLUMNS)
    ow = events_oracle.ordered_writes(df)
    if len(ow) == 0:
        return "empty"
    if not events_stream.single_speed(ow):
        return "multispeed"
    meta = read_meta(df_file)
    digi = meta.is_digi if (meta is not None and not meta.stale) else raw_is_digi(df)
    if digi:
        return "digi"
    dump_token_ids(df, df_file)
    return "ok"


def _glob_corpus(reglogs, reglogs_file, max_files):
    """Resolve the dump set: when ``reglogs_file`` is given (one glob/path per line) it is the corpus,
    else the comma-separated ``reglogs`` globs are. De-duped, sorted, capped at ``max_files``.
    """
    if reglogs_file:
        with open(reglogs_file, encoding="utf-8") as fh:
            patterns = [ln.strip() for ln in fh if ln.strip()]
    else:
        patterns = reglogs.split(",")
    dumps = set()
    for g in patterns:
        dumps.update(_glob.glob(g, recursive=True))
    return sorted(dumps)[:max_files]


def _encode_corpus(todo, workers, logger):
    """Encode each dump's atom stream in a process pool, logging progress; returns
    ``(status Counter, failed-dump list)``."""
    status = collections.Counter()
    failures = []
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_encode_one, d): d for d in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            df_file = futs[fut]
            exc = fut.exception()
            if exc is not None:
                failures.append(df_file)
                status["failed"] += 1
                logger.warning(
                    "FAILED %s: %s", df_file, str(exc).splitlines()[-1][:200]
                )
            else:
                status[fut.result()] += 1
            if i % 500 == 0 or i == len(todo):
                rate = i / max(time.monotonic() - start, 1e-9)
                logger.info(
                    "[%d/%d] %s (%.1f dumps/s)", i, len(todo), dict(status), rate
                )
    return status, failures


def main():
    """Glob + encode in-scope dumps into ``.atoms.zst`` sidecars (``--only-missing`` skips current
    ones), then log a status summary + any failures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reglogs", default="/dumps/**/*.dump.parquet")
    ap.add_argument(
        "--reglogs-file",
        default="",
        help="file of globs/paths (one per line); overrides --reglogs",
    )
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count")
    ap.add_argument("--max-files", type=int, default=10**9)
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--failures", default="", help="path to write failed-dump list")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    logger = logging.getLogger("preencode")

    dumps = _glob_corpus(a.reglogs, a.reglogs_file, a.max_files)
    todo = [d for d in dumps if not (a.only_missing and _cache_current(d))]
    workers = a.workers or multiprocessing.cpu_count()
    logger.info(
        "pre-encode: %d globbed, %d current skipped, %d to encode, %d workers",
        len(dumps),
        len(dumps) - len(todo),
        len(todo),
        workers,
    )

    status, failures = _encode_corpus(todo, workers, logger)

    if a.failures and failures:
        with open(a.failures, "w", encoding="utf-8") as fh:
            fh.write("\n".join(failures) + "\n")
        logger.info("wrote %d failures to %s", len(failures), a.failures)
    logger.info(
        "pre-encode done: %s skipped_current=%d total=%d",
        dict(status),
        len(dumps) - len(todo),
        len(dumps),
    )


if __name__ == "__main__":
    main()
