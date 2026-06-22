"""Full-HVSC codec-coverage census: the single source of truth for corpus
selection. For every ``.sid`` x subtune it runs the codec's sid-only
``recover_from_sid`` (no pre-rendered dump), checks residual-zero, and measures
the whole-song token count, joined to the ground-truth tracker catalog.

Output is a directory of parquet shards (resumable: a re-run skips
``(relpath, subtune)`` keys already present) plus a merged ``census.parquet``.
The recovery is sandboxed per task (own temp ``out_prefix``, cleaned up) and the
pool is capped for NFS hygiene; a per-tune failure is recorded in ``err``, never
crashes the sweep.

Run:
  PYTHONPATH=. python3 -m preframr_experiments.corpus.census \
    --hvsc /scratch/preframr/hvsc/C64Music \
    --catalog /scratch/anarkiwi/cbm/hvsc-tracker-catalog/data/results.csv \
    --songlengths /scratch/preframr/hvsc/C64Music/DOCUMENTS/Songlengths.md5 \
    --tokens-src /scratch/anarkiwi/preframr/preframr-tokens \
    --out /scratch/preframr/census --workers 20
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from preframr_experiments.corpus import hvsc

DEFAULT_TOKENS_SRC = "/scratch/anarkiwi/preframr/preframr-tokens"
DEFAULT_SIDTRACE_BIN = "/scratch/anarkiwi/preframr/preframr-sidtrace/build/sidtrace"
SHARDS_SUBDIR = "shards"
MERGED_NAME = "census.parquet"

COLUMNS = [
    "relpath",
    "subtune",
    "player",
    "family",
    "driver",
    "residual_ok",
    "resid_sum",
    "n_tokens",
    "n_frames",
    "tok_per_frame",
    "err",
]


def _worker_init(tokens_src: str, sidtrace_bin: str) -> None:
    """Make preframr_tokens importable + point at the sidtrace binary per worker."""
    if tokens_src and tokens_src not in sys.path:
        sys.path.insert(0, tokens_src)
    if sidtrace_bin:
        os.environ["SIDTRACE_BIN"] = sidtrace_bin


def _recover_one(task: tuple[str, int, str, str]) -> dict:
    """Census one (relpath, subtune): recover sid-only, residual + token measure.

    Sandboxed in its own temp dir (cleaned up); any failure is captured into
    ``err`` so the pool keeps going. Returns one census row dict.
    """
    relpath, subtune, player, hvsc_root = task
    row = {
        "relpath": relpath,
        "subtune": subtune,
        "player": player,
        "family": hvsc.backend_family(player),
        "driver": None,
        "residual_ok": False,
        "resid_sum": None,
        "n_tokens": None,
        "n_frames": None,
        "tok_per_frame": None,
        "err": None,
    }
    sid_path = os.path.join(hvsc_root, relpath)
    workdir = tempfile.mkdtemp(prefix="preframr_census_")
    try:
        # Imported lazily inside the worker so an import/codec error is recorded,
        # not fatal to the sweep.
        from preframr_tokens import measure, program_to_ids
        from preframr_tokens.bacc.generic import recover_from_sid

        songlengths = os.path.join(hvsc_root, "DOCUMENTS", "Songlengths.md5")
        nframes = hvsc.subtune_frames(sid_path, subtune, songlengths)
        program, resid, _ = recover_from_sid(
            sid_path,
            subtune=subtune,
            nframes=nframes,
            out_prefix=os.path.join(workdir, "trace"),
        )
        resid_sum = int(sum(resid.values()))
        breakdown, frames = measure(program)
        n_tokens = int(breakdown["total"])
        # program_to_ids must round-trip the same length the breakdown reports.
        if len(program_to_ids(program)) != n_tokens:
            raise ValueError("program_to_ids length != measure total")
        row.update(
            driver=program.driver,
            residual_ok=resid_sum == 0,
            resid_sum=resid_sum,
            n_tokens=n_tokens,
            n_frames=int(frames),
            tok_per_frame=round(n_tokens / frames, 6) if frames else None,
        )
    except Exception as err:  # pylint: disable=broad-except
        # Keep the row terse (first line only); a full trace is reproducible on
        # demand by re-running the single tune.
        row["err"] = f"{type(err).__name__}: {err}".splitlines()[0][:300]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return row


def _done_keys(shards_dir: Path) -> set[tuple[str, int]]:
    """(relpath, subtune) keys already recorded across existing shards."""
    done: set[tuple[str, int]] = set()
    for shard in shards_dir.glob("shard-*.parquet"):
        try:
            df = pd.read_parquet(shard, columns=["relpath", "subtune"])
        except (OSError, ValueError):
            continue
        done.update(zip(df["relpath"], df["subtune"].astype(int)))
    return done


def _build_tasks(args, done: set[tuple[str, int]]) -> list[tuple[str, int, str, str]]:
    """Enumerate (relpath, subtune, player, hvsc_root), skipping done keys and
    optional player/limit filters. Subtune count comes from Songlengths."""
    tracker = hvsc.load_tracker_map(args.catalog)
    tasks: list[tuple[str, int, str, str]] = []
    relpaths = sorted(tracker)
    if args.filter:
        needle = args.filter.lower()
        relpaths = [r for r in relpaths if needle in (tracker[r] or "").lower()]
    for relpath in relpaths:
        sid_path = os.path.join(args.hvsc, relpath)
        if not os.path.exists(sid_path):
            continue
        nsub = hvsc.subtune_count(sid_path, args.songlengths)
        for subtune in range(1, max(nsub, 1) + 1):
            if (relpath, subtune) in done:
                continue
            tasks.append((relpath, subtune, tracker[relpath], args.hvsc))
        if args.limit and len(tasks) >= args.limit:
            return tasks[: args.limit]
    return tasks


def _flush(rows: list[dict], shards_dir: Path, shard_idx: int) -> None:
    """Write a shard parquet atomically (tmp then rename)."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    final = shards_dir / f"shard-{shard_idx:06d}.parquet"
    tmp = shards_dir / f".shard-{shard_idx:06d}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, final)


def _next_shard_idx(shards_dir: Path) -> int:
    existing = [int(p.stem.split("-")[1]) for p in shards_dir.glob("shard-*.parquet")]
    return max(existing, default=-1) + 1


def merge(out_dir: Path) -> Path:
    """Concatenate all shards into ``census.parquet`` and return its path."""
    shards_dir = out_dir / SHARDS_SUBDIR
    frames = [pd.read_parquet(p) for p in sorted(shards_dir.glob("shard-*.parquet"))]
    merged = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=COLUMNS)
    )
    merged = merged.drop_duplicates(["relpath", "subtune"], keep="last")
    out = out_dir / MERGED_NAME
    merged.to_parquet(out, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hvsc", required=True, help="HVSC C64Music root")
    parser.add_argument("--catalog", required=True, help="tracker results.csv")
    parser.add_argument("--songlengths", required=True, help="Songlengths.md5")
    parser.add_argument("--out", required=True, help="census output directory")
    parser.add_argument("--tokens-src", default=DEFAULT_TOKENS_SRC)
    parser.add_argument("--sidtrace-bin", default=DEFAULT_SIDTRACE_BIN)
    parser.add_argument("--workers", type=int, default=min(20, (os.cpu_count() or 4) - 2))
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--filter", default=None, help="only players matching substring")
    parser.add_argument("--limit", type=int, default=0, help="cap tasks (smoke test)")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    shards_dir = out_dir / SHARDS_SUBDIR
    shards_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        print(f"merged -> {merge(out_dir)}")
        return 0

    done = _done_keys(shards_dir)
    tasks = _build_tasks(args, done)
    print(f"census: {len(tasks)} tasks ({len(done)} already done), workers={args.workers}")
    if not tasks:
        print(f"nothing to do; merged -> {merge(out_dir)}")
        return 0

    workers = max(1, min(args.workers, (os.cpu_count() or 4) - 2))
    shard_idx = _next_shard_idx(shards_dir)
    buf: list[dict] = []
    completed = 0
    started = time.time()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(args.tokens_src, args.sidtrace_bin),
    ) as pool:
        futures = [pool.submit(_recover_one, t) for t in tasks]
        for fut in as_completed(futures):
            buf.append(fut.result())
            completed += 1
            if len(buf) >= args.shard_size:
                _flush(buf, shards_dir, shard_idx)
                shard_idx += 1
                buf = []
            if completed % 500 == 0:
                rate = completed / max(time.time() - started, 1e-9)
                print(
                    f"  {completed}/{len(tasks)} ({rate:.1f}/s, shard {shard_idx})",
                    flush=True,
                )
    if buf:
        _flush(buf, shards_dir, shard_idx)
    out = merge(out_dir)
    print(f"done: {completed} tasks; merged -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
