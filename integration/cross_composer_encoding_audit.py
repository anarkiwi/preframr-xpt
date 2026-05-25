"""Encoding-level OOD audit for the trained constrained-Unigram."""

import argparse
import glob
import logging
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preframr_tokens import RegTokenizer  # noqa: E402
from preframr_tokens import RegLogParser  # noqa: E402
from preframr_tokens.stfconstants import FRAME_REG, UNICODE_BASE  # noqa: E402

SPLITCHS = [ord(c) for c in string.punctuation]


class _Args:
    """Stand-in for parser args -- only the fields the parser actually
    reads. Keep this list tight; new ones surface as AttributeError so
    we know what to plug in."""

    cents = 1
    diffq = 64
    tk = None
    block_stride = 2048
    seq_len = 8192
    min_song_tokens = 128
    min_irq = 16000
    max_irq = 22000
    require_pq = False
    max_perm = 1
    no_max_autotune = True
    no_loop_transposed = False
    loop_pass = True
    gate_macro_pass = True
    instrument_pass = True


def _atomic_id_from_char(ch, splitters):
    c = ord(ch)
    if c in SPLITCHS:
        idx = SPLITCHS.index(c)
        if idx < splitters:
            return idx
    if c >= 0xE000:
        c -= 0x800
    return c - UNICODE_BASE


def _decomposition_lengths(tkmodel, _splitters):
    """For each sub-token id, how many atomic ids it expands to."""
    n = tkmodel.get_vocab_size()
    out = np.zeros(n, dtype=np.int32)
    for sub_id in range(n):
        s = tkmodel.id_to_token(sub_id)
        if s is None:
            out[sub_id] = 0
            continue
        if s.startswith("<") and s.endswith(">"):
            out[sub_id] = 0
            continue
        out[sub_id] = len(s)
    return out


def encode_song_atoms(rtok, atoms_arr):
    """Encode a 1-D ndarray of atomic ids via the trained Unigram."""
    enc = rtok.encode(atoms_arr.astype(np.uint32), dtype=np.int64)
    return np.asarray(enc, dtype=np.int64)


def parse_song_to_atoms(parser, dump_path, args):
    """Run the parse -> per-rotation atomic-token-id stream pipeline."""
    raise NotImplementedError(
        "Direct dump.parquet -> atom-id encoding requires a tokens df. "
        "Use --uni-glob mode instead, which reads pre-encoded .uni.zst "
        "files written by the trainer."
    )


def encode_uni_file(rtok, uni_path):
    """Read a .uni.zst (Unicode-encoded atomic stream) and run it
    through the Unigram tokenizer. Returns (atom_count, sub_ids).
    """
    import zstandard as zstd

    with zstd.open(uni_path, "r") as f:
        encoded = f.read()
    if isinstance(encoded, bytes):
        encoded = encoded.decode("utf-8")
    atom_count = len(encoded)
    enc = rtok.tkmodel.encode(encoded)
    return atom_count, np.asarray(enc.ids, dtype=np.int64)


def parse_dump_to_uni(parser, dump_path, rtok, args, _logger):
    """Live-parse a dump.parquet to a Unicode-encoded atom stream the
    way train_tokenizer's write_uni() does. For dumps that have no
    pre-baked .uni.zst (e.g. cross-composer SIDs that never went
    through the trainer), this is the path.
    """
    out = []
    for df in parser.parse(
        dump_path, max_perm=args.max_perm, require_pq=False, reparse=True
    ):
        merged = rtok.merge_token_df(rtok.tokens, df)
        if merged is None or merged.empty:
            continue
        atoms = merged["n"].to_numpy(dtype=np.uint32)
        if atoms.size == 0:
            continue
        unicode_str = rtok.encode_unicode(atoms)
        out.append(unicode_str)
    return out


def audit_corpus(label, encoded_strs, rtok, decomp_lens, logger):
    total_atoms = 0
    total_subs = 0
    total_singletons = 0
    seen_subs = set()
    per_song = []
    t0 = time.perf_counter()
    for i, s in enumerate(encoded_strs):
        atom_count = len(s)
        enc = rtok.tkmodel.encode(s)
        sub_ids = np.asarray(enc.ids, dtype=np.int64)
        if sub_ids.size == 0:
            continue
        decomp_per_sub = decomp_lens[sub_ids]
        singletons = int((decomp_per_sub == 1).sum())
        total_atoms += atom_count
        total_subs += int(sub_ids.size)
        total_singletons += singletons
        seen_subs.update(int(x) for x in sub_ids)
        per_song.append(
            {
                "label": label,
                "i": i,
                "atoms": atom_count,
                "subs": int(sub_ids.size),
                "singleton_rate": singletons / max(int(sub_ids.size), 1),
                "mean_enc_len_per_atom": int(sub_ids.size) / max(atom_count, 1),
            }
        )
    dt = time.perf_counter() - t0
    summary = {
        "label": label,
        "n_songs": len(encoded_strs),
        "atoms": total_atoms,
        "subs": total_subs,
        "singleton_rate": total_singletons / max(total_subs, 1),
        "mean_enc_len_per_atom": total_subs / max(total_atoms, 1),
        "coverage": len(seen_subs) / max(rtok.tkmodel.get_vocab_size(), 1),
        "wallclock_s": dt,
    }
    logger.info(
        "[%s] %u songs, atoms=%u subs=%u singleton_rate=%.4f "
        "mean_enc_len=%.4f coverage=%.4f (%.1fs)",
        label,
        summary["n_songs"],
        summary["atoms"],
        summary["subs"],
        summary["singleton_rate"],
        summary["mean_enc_len_per_atom"],
        summary["coverage"],
        dt,
    )
    return summary, per_song


def load_rtok(tkmodel_path, tokens_csv):
    args = _Args()
    args.tkvocab = 0
    args.tkmodel = tkmodel_path
    args.tokenizer = "unigram"
    rtok = RegTokenizer(args, tokens=None)
    tokens_df = pd.read_csv(tokens_csv)
    tokens_df["n"] = tokens_df.index
    rtok.tokens = tokens_df
    n_frame = int((tokens_df["reg"] == FRAME_REG).sum())
    rtok.splitters = min(rtok.splitters, n_frame)
    with open(tkmodel_path) as f:
        rtok.load(f.read(), tokens_df)
    return rtok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tkmodel", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument(
        "--in-dist-uni-glob",
        required=True,
        help="Glob of pre-encoded .uni.zst files for the in-distribution baseline.",
    )
    ap.add_argument(
        "--cross-dump-glob",
        required=True,
        help="Glob of cross-composer .dump.parquet files (live-parsed).",
    )
    ap.add_argument(
        "--cross-sample-n",
        type=int,
        default=16,
        help="How many cross-composer dumps to sample (sorted by name, evenly spaced).",
    )
    ap.add_argument("--out-csv", default=None)
    ap.add_argument(
        "--instrument-pass", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument(
        "--gate-macro-pass", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument("--loop-pass", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("audit")

    rtok = load_rtok(args.tkmodel, args.tokens)
    decomp_lens = _decomposition_lengths(rtok.tkmodel, rtok.splitters)
    logger.info(
        "loaded tkmodel: vocab=%u, atomic alphabet=%u, splitters=%u",
        rtok.tkmodel.get_vocab_size(),
        len(rtok.tokens),
        rtok.splitters,
    )

    import zstandard as zstd

    in_files = sorted(glob.glob(args.in_dist_uni_glob))
    in_strs = []
    for p in in_files:
        with zstd.open(p, "r") as f:
            data = f.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        in_strs.append(data)
    in_summary, _ = audit_corpus("in-dist (eval)", in_strs, rtok, decomp_lens, logger)

    parse_args = _Args()
    parse_args.instrument_pass = args.instrument_pass
    parse_args.gate_macro_pass = args.gate_macro_pass
    parse_args.loop_pass = args.loop_pass
    parser = RegLogParser(parse_args)
    cross_files = sorted(glob.glob(args.cross_dump_glob))
    if args.cross_sample_n and len(cross_files) > args.cross_sample_n:
        idx = np.linspace(0, len(cross_files) - 1, args.cross_sample_n, dtype=int)
        cross_files = [cross_files[i] for i in idx]
    cross_strs = []
    skip_reasons = []
    skip_re = re.compile(r"op=(-?\d+)\s+reg=(-?\d+)\s+subreg=(-?\d+)\s+val=(-?\d+)")
    for p in cross_files:
        try:
            rotations = parse_dump_to_uni(parser, p, rtok, parse_args, logger)
        except (KeyError, AssertionError, ValueError) as e:
            logger.warning("skip %s: %s", Path(p).name, e)
            m = skip_re.search(str(e))
            if m:
                skip_reasons.append(
                    (Path(p).name, int(m.group(1)), int(m.group(3)), int(m.group(4)))
                )
            else:
                skip_reasons.append((Path(p).name, None, None, None))
            continue
        cross_strs.extend(rotations)
    cross_summary, _ = audit_corpus(
        "cross-composer", cross_strs, rtok, decomp_lens, logger
    )

    if skip_reasons:
        from collections import Counter

        cls = Counter((op, sr) for _, op, sr, _ in skip_reasons if op is not None)
        print()
        print(f"Skip distribution ({len(skip_reasons)} dumps):")
        for (op, sr), n in sorted(cls.items(), key=lambda kv: -kv[1]):
            print(f"  op={op} subreg={sr}: {n}")

    print()
    print("Comparison:")
    keys = [
        "n_songs",
        "atoms",
        "subs",
        "singleton_rate",
        "mean_enc_len_per_atom",
        "coverage",
    ]
    print(f"{'metric':<26} {'in-dist':>14} {'cross':>14} {'delta':>14}")
    for k in keys:
        a = in_summary[k]
        b = cross_summary[k]
        if isinstance(a, float):
            delta = f"{b-a:+.4f}"
            print(f"{k:<26} {a:>14.4f} {b:>14.4f} {delta:>14}")
        else:
            print(f"{k:<26} {a:>14} {b:>14} {b-a:>+14}")

    if args.out_csv:
        rows = [
            {"corpus": "in-dist", **in_summary},
            {"corpus": "cross", **cross_summary},
        ]
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        logger.info("wrote %s", args.out_csv)


if __name__ == "__main__":
    main()
