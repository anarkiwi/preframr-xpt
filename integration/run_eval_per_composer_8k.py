"""Per-composer val_loss + val_acc breakdown for the 8K MC train."""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from collections import defaultdict


def parse_list(path):
    """Return list of (rel_path, composer) tuples from a picker .list file."""
    out = []
    with open(path) as f:
        for line in f:
            rel = line.split("#", 1)[0].strip()
            if not rel:
                continue
            parts = rel.split("/")
            if len(parts) < 4 or parts[0] != "MUSICIANS":
                raise ValueError(f"unexpected list path: {rel}")
            composer = parts[2]
            out.append((rel, composer))
    return out


def group_by_composer(entries):
    by = defaultdict(list)
    for rel, composer in entries:
        by[composer].append(rel)
    return by


def make_symlink_dir(symlink_root, label, eval_dir, basenames):
    """Build ``symlink_root/<label>/`` with symlinks to ``eval_dir/<basename>``
    for each basename. Returns the absolute symlink dir.
    """
    target = os.path.join(symlink_root, label)
    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target)
    n = 0
    for bn in basenames:
        src = os.path.join(eval_dir, bn)
        dst = os.path.join(target, bn)
        if not os.path.exists(src):
            print(f"  WARN: missing {src}", file=sys.stderr)
            continue
        os.symlink(src, dst)
        n += 1
    return target, n


def run_one(args, label):
    """Run eval_per_composer.py inside docker for one composer."""
    cmd = [
        "docker",
        "run",
        "--rm",
        f"--memory={args.memory}",
        "--gpus",
        "all",
        "-v",
        f"{args.root}:/scratch/preframr",
        "-v",
        f"{args.symlink_root}:/symlinks",
        "-v",
        "/scratch/anarkiwi/preframr/preframr:/preframr",
        "-v",
        "/scratch/anarkiwi/preframr/integration_tests:/integration_tests",
        args.image,
        "python3",
        "/integration_tests/eval_per_composer.py",
        "--no-require-pq",
        "--no-max-autotune",
        "--seq-len",
        str(args.seq_len),
        "--tkvocab",
        str(args.tkvocab),
        "--df-map-csv",
        "/scratch/preframr/df-map.csv",
        "--dataset-csv",
        "/scratch/preframr/dataset.csv.zst",
        "--token-csv",
        "/scratch/preframr/tokens.csv",
        "--reglogs",
        "/scratch/preframr/train/*.dump.parquet",
        "--ckpt",
        args.ckpt_in_container,
        "--label",
        label,
        "--eval-basename-dir",
        f"/symlinks/{label}",
        "--min-song-tokens",
        str(args.min_song_tokens),
        "--block-stride",
        str(args.block_stride),
        "--max-perm",
        "2",
        "--model",
        args.model,
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--kv-heads",
        str(args.kv_heads),
        "--embed",
        str(args.embed),
        "--intermediate",
        str(args.intermediate),
    ]
    print(f"  invoking docker for {label}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(
            f"  ERROR [{label}] rc={proc.returncode}:\n{proc.stderr[-2000:]}",
            file=sys.stderr,
        )
        return None
    last = [l for l in proc.stdout.strip().splitlines() if l.startswith(label + "\t")]
    if not last:
        print(f"  ERROR [{label}] no result line in stdout", file=sys.stderr)
        print(proc.stdout[-1000:], file=sys.stderr)
        return None
    return last[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="absolute ckpt path on host")
    ap.add_argument("--lists-dir", required=True)
    ap.add_argument(
        "--root", required=True, help="MC root, mounted as /scratch/preframr"
    )
    ap.add_argument("--symlink-root", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--image", default="anarkiwi/preframr")
    ap.add_argument("--memory", default="32g")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--tkvocab", type=int, default=131072)
    ap.add_argument("--min-song-tokens", type=int, default=128)
    ap.add_argument("--block-stride", type=int, default=2048)
    ap.add_argument("--model", default="llama3_2")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--embed", type=int, default=320)
    ap.add_argument("--intermediate", type=int, default=896)
    ap.add_argument("--only", default=None, help="run only this composer label (debug)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)
    if not args.ckpt.startswith(args.root):
        raise ValueError(
            f"--ckpt {args.ckpt} must be under --root {args.root} for the "
            "in-container mount to resolve"
        )
    args.ckpt_in_container = args.ckpt.replace(args.root, "/scratch/preframr", 1)

    eval_a_path = os.path.join(args.lists_dir, "eval-A.list")
    eval_a = parse_list(eval_a_path)
    by_composer_a = group_by_composer(eval_a)
    print(f"Eval-A: {len(by_composer_a)} composers, {len(eval_a)} dumps")

    eval_b_lists = sorted(glob.glob(os.path.join(args.lists_dir, "eval-B-*.list")))
    eval_b_jobs = []
    for p in eval_b_lists:
        composer = None
        entries = parse_list(p)
        if entries:
            composer = entries[0][1]
        if composer:
            eval_b_jobs.append((composer, [r for (r, _) in entries]))
    print(f"Eval-B: {len(eval_b_jobs)} composers")

    os.makedirs(args.symlink_root, exist_ok=True)
    eval_dir = os.path.join(args.root, "eval")
    if not os.path.isdir(eval_dir):
        raise RuntimeError(f"eval dir not found: {eval_dir}")

    jobs = []
    for composer, rels in by_composer_a.items():
        if args.only and composer != args.only:
            continue
        bns = [os.path.basename(r) for r in rels]
        _, n = make_symlink_dir(args.symlink_root, composer, eval_dir, bns)
        if n == 0:
            print(f"  SKIP {composer}: no dumps materialised in eval_dir", flush=True)
            continue
        jobs.append((composer, "Eval-A", n))
    for composer, rels in eval_b_jobs:
        if args.only and composer != args.only:
            continue
        bns = [os.path.basename(r) for r in rels]
        _, n = make_symlink_dir(args.symlink_root, composer, eval_dir, bns)
        if n == 0:
            print(f"  SKIP {composer}: no dumps materialised in eval_dir", flush=True)
            continue
        jobs.append((composer, "Eval-B", n))

    print(f"Running {len(jobs)} composer evals")
    results = []
    for label, role, n_dumps in jobs:
        line = run_one(args, label)
        if line is None:
            results.append((label, role, n_dumps, None, None))
            continue
        parts = line.split("\t")
        try:
            val_loss = float(parts[1])
            val_acc = float(parts[2])
        except (IndexError, ValueError):
            print(f"  malformed result for {label}: {line!r}", file=sys.stderr)
            results.append((label, role, n_dumps, None, None))
            continue
        results.append((label, role, n_dumps, val_loss, val_acc))
        print(
            f"  {label}\t{role}\tn={n_dumps}\tval_loss={val_loss:.4f}\tval_acc={val_acc:.4f}"
        )

    results.sort(key=lambda r: -(r[4] if r[4] is not None else -1))
    with open(args.out_tsv, "w") as f:
        f.write("composer\trole\tn_dumps\tval_loss\tval_acc\n")
        for label, role, n, vl, va in results:
            vl_s = "" if vl is None else f"{vl:.4f}"
            va_s = "" if va is None else f"{va:.4f}"
            f.write(f"{label}\t{role}\t{n}\t{vl_s}\t{va_s}\n")
    print(f"\nWrote {args.out_tsv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
