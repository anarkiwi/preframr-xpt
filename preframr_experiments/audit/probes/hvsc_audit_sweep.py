#!/usr/bin/env python3
"""Parallel streaming byte-exact audit over a representative 1/STEP sample of HVSC. Each worker parses
one tune on the byte-exact skeleton path with parse_audit="raise": a divergence (register_state
losslessness / loop-aware frame budget / loop-aware ref) raises AssertionError with the pinpoint; any
other exception is an error. Main streams DIRTY/ERROR the instant a worker returns one, plus a heartbeat
with rate / percent / ETA. Unbuffered (run with python3 -u). Args: STEP (default 10) WORKERS (default cpu-4).
"""

import glob
import os
import sys
import time
from multiprocessing import Pool


def check(path):
    name = path.rsplit("/", 1)[-1]
    try:
        from preframr_tokens.reglogparser import RegLogParser
        from preframr_tokens.tokenizer_config import default_tokenizer_args

        p = RegLogParser(
            args=default_tokenizer_args(
                skeleton_pass=True,
                loop_pass=True,
                loop_transposed=True,
                lonely_catch_all=True,
                parse_audit="raise",
            )
        )
        streams = 0
        for _ in p.parse(path, max_perm=1, require_pq=False, reparse=True):
            streams += 1
        return (name, "empty" if streams == 0 else "clean", "")
    except AssertionError as e:
        return (name, "dirty", str(e).split("PARSE AUDIT:", 1)[-1].strip()[:170])
    except Exception as e:  # noqa: BLE001
        return (name, "error", f"{type(e).__name__}: {str(e)[:140]}")


if __name__ == "__main__":
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    workers = (
        int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 4) - 4)
    )
    allf = sorted(glob.glob("/scratch/preframr/hvsc/**/*.dump.parquet", recursive=True))
    sample = allf[::step]
    n = len(sample)
    print(
        f"HVSC total={len(allf)} sample(1/{step})={n} workers={workers} "
        f"config=skeleton byte-exact",
        flush=True,
    )
    t0 = time.time()
    clean = empty = dirty = err = 0
    dirty_list = []
    err_list = []
    heartbeat = 50
    with Pool(workers) as pool:
        for i, (name, kind, detail) in enumerate(
            pool.imap_unordered(check, sample, chunksize=4), 1
        ):
            if kind == "dirty":
                dirty += 1
                dirty_list.append(name)
                print(f"  DIRTY [{i}/{n}] {name}: {detail}", flush=True)
            elif kind == "error":
                err += 1
                err_list.append(name)
                print(f"  ERROR [{i}/{n}] {name}: {detail}", flush=True)
            elif kind == "empty":
                empty += 1
            else:
                clean += 1
            if i % heartbeat == 0 or i == n:
                el = time.time() - t0
                rate = i / el if el else 0.0
                eta = (n - i) / rate if rate else 0.0
                print(
                    f"[{i}/{n} {100*i/n:.1f}%] rate={rate:.1f}/s elapsed={el/60:.1f}m "
                    f"ETA={eta/60:.1f}m | clean={clean} empty={empty} DIRTY={dirty} ERR={err}",
                    flush=True,
                )
    print(
        f"\nDONE sample={n} clean={clean} empty={empty} DIRTY={dirty} ERR={err} "
        f"in {(time.time()-t0)/60:.1f}m",
        flush=True,
    )
    if dirty_list:
        print(f"DIRTY tunes ({len(dirty_list)}): {dirty_list[:100]}", flush=True)
    if err_list:
        print(f"ERROR tunes ({len(err_list)}): {err_list[:100]}", flush=True)
    sys.exit(1 if (dirty or err) else 0)
