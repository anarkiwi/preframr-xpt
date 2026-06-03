#!/usr/bin/env python3
"""Training-free learnability triage for a tokenization/ordering -- ranks ENCODINGS by
information-theoretic proxies for how cheaply a bounded (~TC0) transformer can represent
the next-token map, WITHOUT training. Backing theory + reading guide:
design/learnability_token_ordering_theory.md.

The proxies (all computed on the tokenized atom stream, contexts never crossing a tune
boundary):

  - entropy-rate vs memory  h_k = H_{k+1} - H_k (bits/token), k=0..kmax. The FLOOR is the
                            achievable next-token CE loss; the k where h_k PLATEAUS is the
                            effective memory (dependency-horizon / causal-state proxy).
                            Reported per-token AND per-frame (h_k * tokens/frame) so the
                            cross-encoding comparison is not confounded by sequence length
                            (a compressing encoding packs ~constant tune information into
                            fewer tokens). Miller-Madow corrected.
  - MI decay                I(x_t ; x_{t-d}) vs d -- concentrated at small d = good; a fat
                            tail = a long-range counter SGD will shortcut (plugin; read the
                            SHAPE, the absolute is upward-biased).
  - induction-copy rate     share of positions completing a bigram seen earlier in the tune
                            -- the induction-head-able fraction (transformers' easiest
                            circuit). High = learnable.
  - first-occurrence rate   irreducible novelty proxy (lower = more reuse).

Read: low per-frame h_k + early h_k plateau + fast MI decay + high induction-copy ->
predicted learnable; fat MI tail + low copy-fraction -> predicted to collapse.

The metric functions are pure stdlib (host-importable, unit-tested in
tests/test_learnability_triage.py). The corpus loader imports preframr_tokens lazily.
--mode song (default) tokenizes the full-song parse() stream: robust + full coverage, but it
OVER-CREDITS codebook compression that accumulates over a whole song. --mode blocks targets the
SELF-CONTAINED-BLOCK stream the model actually trains/predicts on (references block-local) but is
EXPERIMENTAL -- it reproduces the block-builder standalone and drops tunes whose ops need parser
context (partial coverage); the faithful version must route through the Corpus block-builder.
Known signal from the partial block run: codebook compression does NOT survive to block scale
(its in-window induction-copy collapses), so full_macros pulls ahead of the codebook arm -- the
reverse of the song-mode ordering. Certify via the Corpus-API version before acting on it.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter

LN2 = math.log(2.0)


def _entropy_bits(counter, n):
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c > 0:
            p = c / n
            h -= p * math.log2(p)
    return h


def _miller_madow(h_plugin_bits, n_types, n):
    """Plugin entropy is downward-biased; Miller-Madow adds (m-1)/(2N) nats."""
    if n <= 0:
        return h_plugin_bits
    return h_plugin_bits + (n_types - 1) / (2.0 * n * LN2)


def block_entropy(seqs, k):
    """Miller-Madow-corrected entropy (bits) of k-grams, counted within each sequence so a
    context never spans two tunes."""
    c = Counter()
    for s in seqs:
        if len(s) >= k:
            for i in range(len(s) - k + 1):
                c[tuple(s[i : i + k])] += 1
    n = sum(c.values())
    return _miller_madow(_entropy_bits(c, n), len(c), n)


def entropy_rate(seqs, k):
    """h_k = H_{k+1} - H_k (bits/token): next-token entropy given the previous k tokens.
    h_0 is the unigram entropy."""
    if k <= 0:
        return block_entropy(seqs, 1)
    return block_entropy(seqs, k + 1) - block_entropy(seqs, k)


def mutual_information_lag(seqs, d):
    """I(x_t ; x_{t-d}) in bits (plugin; upward-biased -- read the decay shape)."""
    joint = Counter()
    left = Counter()
    right = Counter()
    for s in seqs:
        for i in range(d, len(s)):
            a, b = s[i - d], s[i]
            joint[(a, b)] += 1
            left[a] += 1
            right[b] += 1
    n = sum(joint.values())
    if n == 0:
        return 0.0
    mi = 0.0
    for (a, b), c in joint.items():
        pab = c / n
        mi += pab * math.log2(pab / ((left[a] / n) * (right[b] / n)))
    return max(0.0, mi)


def induction_copy_rate(seqs):
    """Fraction of positions t>=1 whose bigram (x_{t-1}, x_t) occurred earlier in the same
    tune -- the 1-context induction-head copy ceiling."""
    copy = total = 0
    for s in seqs:
        seen = set()
        for i in range(1, len(s)):
            total += 1
            bg = (s[i - 1], s[i])
            if bg in seen:
                copy += 1
            seen.add(bg)
    return copy / total if total else 0.0


def first_occurrence_rate(seqs):
    """Fraction of tokens that are the first appearance of their symbol in the tune."""
    first = total = 0
    for s in seqs:
        seen = set()
        for x in s:
            total += 1
            if x not in seen:
                first += 1
                seen.add(x)
    return first / total if total else 0.0


def summarize(seqs, frames, kmax=4, maxlag=16):
    n_tok = sum(len(s) for s in seqs)
    alphabet = len({x for s in seqs for x in s})
    tpf = n_tok / frames if frames else float("nan")
    hk = [entropy_rate(seqs, k) for k in range(kmax + 1)]
    return {
        "tunes": len(seqs),
        "tokens": n_tok,
        "frames": frames,
        "alphabet": alphabet,
        "tokens_per_frame": tpf,
        "h_k_per_token": hk,
        "h_k_per_frame": [h * tpf for h in hk],
        "branching": [2.0**h for h in hk],
        "mi_lag": [mutual_information_lag(seqs, d) for d in range(1, maxlag + 1)],
        "induction_copy_rate": induction_copy_rate(seqs),
        "first_occurrence_rate": first_occurrence_rate(seqs),
    }


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
_CODEBOOK = (
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
    "note_off",
)


def _config_flags(name):
    """Resolve a preset to the set of flag names available in THIS installed tokens build
    (drops flags not yet shipped, e.g. an in-flight ctrl_osc when run against a release).
    """
    from preframr_tokens.tokenizer_config import REGISTERED_MACROS

    try:
        from preframr_tokens.macros.flag_registry import macro_flag_names

        known = set(macro_flag_names())
    except Exception:
        known = None
    if name == "baseline":
        want = set()
    elif name == "full_macros":
        want = set(REGISTERED_MACROS)
    elif name == "codebook":
        want = set(_BASE + _CODEBOOK)
    else:
        raise SystemExit(f"unknown config: {name}")
    if known is not None:
        dropped = want - known
        if dropped:
            print(
                f"  [{name}] dropped flags unavailable in this build: {sorted(dropped)}"
            )
        want &= known
    return want


def _symbols(df, vocab):
    """Map a row df's (op, reg, subreg, val) tuples to interned int symbols."""
    ops = df["op"].tolist()
    regs = df["reg"].tolist()
    subs = df["subreg"].tolist() if "subreg" in df.columns else [-1] * len(df)
    vals = df["val"].tolist()
    return [
        vocab.setdefault((int(o), int(r), int(sb), int(v)), len(vocab))
        for o, r, sb, v in zip(ops, regs, subs, vals)
    ]


def _decoded_frames(df, register_state):
    """DECODED frame count (register_state EXPANDS loop/codebook refs) -- the true,
    cross-config-comparable timeline (a FRAME-marker count undercounts any loop_pass config).
    """
    try:
        return int(register_state(df).shape[0])
    except Exception:  # noqa: BLE001
        return 0


def tokenize_corpus(config_name, dump_paths, seq_len=4096, mode="blocks"):
    """Tokenize each dump under `config_name`; return (per-sequence symbol lists, total
    decoded frames). mode="blocks" (default) measures the SELF-CONTAINED BLOCK stream the model
    actually trains/predicts on -- `iter_voiced_blocks` (expand-to-literal -> slice -> re-encode
    -> voice-reg), one sequence per block, references block-local. mode="song" measures the
    full-song parse() stream (NOT what the model sees; kept for comparison)."""
    from preframr_tokens.reglogparser import RegLogParser
    from preframr_tokens.tokenizer_config import default_tokenizer_args
    from preframr_tokens.audit_primitives import register_state

    flags = _config_flags(config_name)
    args = default_tokenizer_args(seq_len=seq_len, **{f: True for f in flags})
    parser = RegLogParser(args)
    if mode == "blocks":
        from preframr_tokens.macros.blocks import iter_self_contained_row_blocks
        from preframr_tokens.blocks import remove_voice_reg

        frames_per_block = max(1, seq_len // 2)
    seqs = []
    total_frames = 0
    vocab = {}
    n_ok = 0
    for path in dump_paths:
        try:
            df = next(parser.parse(path, max_perm=1, require_pq=False))
        except StopIteration:
            print(f"  skip {path}: parse yielded nothing (excluded/empty/digi-gated)")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  skip {path}: {type(e).__name__}: {e}")
            continue
        # Frame denominator = the SONG's decoded music timeline (config- and mode-invariant),
        # counted once from the full parse -- NOT the sum of per-block register_state (which
        # double-counts block-boundary lead frames). tokens come from the blocks.
        song_frames = _decoded_frames(df, register_state)
        if mode == "song":
            units = [df]
        else:
            # Self-contained blocks (expand-to-literal -> slice -> re-encode, codebooks/loops
            # re-mined block-local) in ABSOLUTE pre-voice-reg form -- the voice-reg header is a
            # deterministic re-encoding orthogonal to DEF->REF locality, and skipping it avoids
            # the _add_voice_reg voicing step (irrelevant to this measurement).
            try:
                abs_df, _ = remove_voice_reg(df.copy(), {})
                units = [
                    b
                    for b in iter_self_contained_row_blocks(
                        abs_df, frames_per_block, args=args
                    )
                    if not b.empty
                ]
            except Exception as e:  # noqa: BLE001
                print(f"  skip {path}: blocks failed ({type(e).__name__}: {e})")
                continue
        got = False
        for unit in units:
            seq = _symbols(unit, vocab)
            if seq:
                seqs.append(seq)
                got = True
        if got:
            total_frames += song_frames
            n_ok += 1
    n_drop = len(dump_paths) - n_ok
    cov = f"  COVERAGE: {n_ok}/{len(dump_paths)} dumps contributed"
    if mode == "blocks" and n_drop:
        cov += (
            f" ({n_drop} dropped -- EXPERIMENTAL block re-encode trips on ops needing parser "
            f"context; the faithful version routes through the Corpus block-builder)"
        )
    print(cov, flush=True)
    return seqs, total_frames


def _fmt(xs, p=3):
    return "[" + ", ".join(f"{x:.{p}f}" for x in xs) + "]"


def _print_summary(name, s):
    print(f"\n=== {name} ===")
    print(
        f"  tunes={s['tunes']} tokens={s['tokens']} frames={s['frames']} "
        f"alphabet={s['alphabet']} tokens/frame={s['tokens_per_frame']:.2f}"
    )
    print(
        f"  h_k bits/token  (k=0..{len(s['h_k_per_token'])-1}): {_fmt(s['h_k_per_token'])}"
    )
    print(
        f"  h_k bits/frame  (length-normalized)            : {_fmt(s['h_k_per_frame'])}"
    )
    print(
        f"  effective branching 2^h_k                      : {_fmt(s['branching'], 2)}"
    )
    print(
        f"  MI(x_t;x_t-d) d=1..{len(s['mi_lag'])} bits          : {_fmt(s['mi_lag'], 2)}"
    )
    print(
        f"  induction-copy rate = {s['induction_copy_rate']:.3f}   "
        f"first-occurrence rate = {s['first_occurrence_rate']:.3f}"
    )


def _print_comparison(results):
    if len(results) < 2:
        return
    print(
        "\n\n===== HEADLINE (lower h_inf/frame + higher copy + faster MI decay = more learnable) ====="
    )
    print(
        f"  {'config':<12} {'alpha':>7} {'tok/fr':>7} {'h0/fr':>7} {'hinf/fr':>8} "
        f"{'hinf/tok':>9} {'copy':>6} {'MI@1':>6} {'MI@8':>6} {'MI@16':>6}"
    )
    for name, s in results.items():
        hinf_tok = s["h_k_per_token"][-1]
        hinf_fr = s["h_k_per_frame"][-1]
        h0_fr = s["h_k_per_frame"][0]
        mi = s["mi_lag"]
        print(
            f"  {name:<12} {s['alphabet']:>7} {s['tokens_per_frame']:>7.2f} "
            f"{h0_fr:>7.2f} {hinf_fr:>8.2f} {hinf_tok:>9.3f} "
            f"{s['induction_copy_rate']:>6.3f} {mi[0]:>6.2f} "
            f"{mi[min(7,len(mi)-1)]:>6.2f} {mi[-1]:>6.2f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--configs",
        default="baseline,full_macros,codebook",
        help="comma list of presets: baseline, full_macros, codebook",
    )
    ap.add_argument(
        "--dumps", nargs="+", required=True, help="*.dump.parquet paths (digi-excluded)"
    )
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument(
        "--mode",
        default="song",
        choices=["song", "blocks"],
        help="song = full-song parse() stream (robust, full coverage; over-credits codebook "
        "compression that accumulates over a whole song) [default]; blocks = the self-contained-"
        "block stream the model actually sees (EXPERIMENTAL: reproduces the block-builder "
        "standalone and drops tunes whose ops need parser context -- partial coverage; the "
        "faithful version must route through the Corpus block-builder)",
    )
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--maxlag", type=int, default=16)
    a = ap.parse_args()
    results = {}
    for cfg in [c.strip() for c in a.configs.split(",") if c.strip()]:
        print(
            f"\n##### tokenizing config '{cfg}' ({a.mode} mode) over {len(a.dumps)} dumps #####",
            flush=True,
        )
        seqs, frames = tokenize_corpus(cfg, a.dumps, a.seq_len, mode=a.mode)
        if not seqs:
            print(f"  no usable tunes for {cfg}")
            continue
        s = summarize(seqs, frames, a.kmax, a.maxlag)
        results[cfg] = s
        _print_summary(cfg, s)
    _print_comparison(results)


if __name__ == "__main__":
    main()
