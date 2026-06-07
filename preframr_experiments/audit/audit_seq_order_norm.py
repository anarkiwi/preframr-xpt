#!/usr/bin/env python3
"""Sequence-order normalization audit: decompose cross-engine intra-frame
divergence into composition / multiplicity / order, and prove that an
audio-safe reorder is inaudible.

FINDING (2026-05-27, eval_b x8, post-full_macros): the cross-engine gap from the
order-free reg SET (cosine 0.466) down to the full write TUPLE (0.297) is
+0.169, but it splits as -0.123 MULTIPLICITY (how many times each reg is written
per frame) + only -0.046 ORDER. Legal, voice-respecting, audio-safe reordering
recovers only ~+0.009 (~5%). 84% of intra-frame repeated writes carry DISTINCT
values (genuine sub-frame modulation = content). So per-frame ORDER normalization
is NOT a meaningful generalization lever -- the divergence is mostly real
modulation content, not notation. This audit is the instrument that decided it;
keep it to re-check on new tokenizer versions / corpora.

Two modes:

  --mode divergence  Parse each eval_b engine to post-full_macros atom streams;
                     report cross-engine cosine of per-frame reg SET vs MULTISET
                     vs TUPLE vs voice-canonicalized TUPLE, plus the
                     redundant-vs-modulation split of repeated writes.

  --mode fidelity    The inaudibility proof, on RAW register writes (macros off,
                     directly renderable). For each dump: (1) reg-state per frame
                     is byte-identical under any stable reorder (invariant); (2)
                     render original vs a within-frame shuffle vs the canonical
                     reorder and compare PCM. Canonical reorder ~ original
                     (corr~1) => an audio-safe reorder IS inaudible; shuffle /
                     shuf+ctrl drift => CTRL-anchoring + stable reg-sort are
                     load-bearing (the reorder is just a tiny lever).

VOICE SEMANTICS: VOICE_REG sets the active voice; its following writes are
voice-relative and MUST travel with it -- a write may never cross a VOICE_REG.
The reorder anchors VOICE_REG/CTRL/markers and stable-sorts only within a voice
run; voice-BLOCK reordering moves whole (VOICE_REG + its writes) units.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from types import SimpleNamespace

import numpy as np

CTRL_REGS = {4, 11, 18}


# ---- shared reorder rule -------------------------------------------------


def _is_anchor(reg):
    return reg < 0 or reg in CTRL_REGS


def canonical_order(items, reg_of, sub_of):
    """Gate-anchored stable canonical reorder of a single frame's items. Anchors
    (CTRL regs + reg<0 markers) stay in place; non-anchor runs between them are
    stable-sorted by (reg, subreg)."""
    out, run = [], []
    for it in items:
        if _is_anchor(reg_of(it)):
            out.extend(sorted(run, key=lambda x: (reg_of(x), sub_of(x))))
            run = []
            out.append(it)
        else:
            run.append(it)
    out.extend(sorted(run, key=lambda x: (reg_of(x), sub_of(x))))
    return out


def shuffle_order(items, reg_of, sub_of, rng, move_ctrl):
    """A different valid order: permute non-anchor reg-groups (same-reg order
    preserved). Mimics another engine's idiom; isolates the value-latch reorder.
    With move_ctrl, also permute CTRL writes (to bound the audible edge)."""
    anchored = (reg_of(it) for it in items)
    is_anchor = (lambda r: r < 0) if move_ctrl else _is_anchor
    out, run = [], []

    def flush():
        groups = defaultdict(list)
        order = []
        for x in run:
            k = (reg_of(x), sub_of(x))
            if k not in groups:
                order.append(k)
            groups[k].append(x)
        rng.shuffle(order)
        for k in order:
            out.extend(groups[k])

    for it, _r in zip(items, anchored):
        if is_anchor(reg_of(it)):
            flush()
            run = []
            out.append(it)
        else:
            run.append(it)
    flush()
    return out


def voice_units(body, voice_reg):
    """Split a frame body into voice-block units. A unit starts at a VOICE_REG
    atom and runs to the next VOICE_REG (writes travel with their voice). Atoms
    before the first VOICE_REG are a leading anchor segment kept in place."""
    lead, us, cur = [], [], None
    for a in body:
        if a[1] == voice_reg:
            if cur is not None:
                us.append(cur)
            cur = [a]
        elif cur is None:
            lead.append(a)
        else:
            cur.append(a)
    if cur is not None:
        us.append(cur)
    return lead, us


def voice_canonical(body, voice_reg, reg_of, sub_of):
    """Audio-safe canonical order: stable-sort writes within each voice block,
    then reorder whole voice-block units by a canonical key (the unit's atom
    signature). Writes never cross a VOICE_REG."""
    lead, us = voice_units(body, voice_reg)
    us = [[u[0]] + sorted(u[1:], key=lambda a: (reg_of(a), sub_of(a))) for u in us]
    us = sorted(us, key=lambda u: tuple((a[0], a[1], a[2], a[3]) for a in u))
    out = list(lead)
    for u in us:
        out.extend(u)
    return out


# ---- fidelity mode (raw writes, renderable) ------------------------------

_FID_BASE = dict(
    cents=50,
    exclude_list=None,
    min_irq=int(1.5e4),
    max_irq=int(2.5e4),
    min_song_tokens=256,
    diffq=4,
    loop_lookahead=3,
    coarsen_min_len=16,
    voice_trajectory_window=8,
    macro_flags="",
    meta_exclude_digi=False,
    meta_irq_lo=0,
    meta_irq_hi=0,
    meta_require=False,
)
_FID_MACROS = (
    "freq_trajectory_pass",
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c3",
    "legato_pass_c4",
    "legato_pass_c7",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
    "fuzzy_loop_pass",
    "fuzzy_fp_adsr",
    "coarsen_pass",
    "mode_vol_flip_pass",
    "voice_trajectory_pass",
    "voice_trajectory_distributed_pass",
    "set_to_diff_pass",
    "freq_nudge_pass",
    "release_update_pass",
    "ctrl_triple_pass",
    "lonely_catch_all",
)


def _raw_args():
    cfg = dict(_FID_BASE)
    for m in _FID_MACROS:
        cfg[m] = False
    return SimpleNamespace(**cfg)


def _reg_state(adf):
    """Per-frame state of SID registers 0-24 as an (n_frames, 25) array."""
    n_frames = int(adf["f"].max()) + 1
    state = np.zeros((n_frames, 25), dtype=np.int64)
    cur = np.zeros(25, dtype=np.int64)
    cf = 0
    for reg, val, frame in adf[["reg", "val", "f"]].to_numpy():
        reg, frame = int(reg), int(frame)
        while cf < frame and cf < n_frames:
            state[cf] = cur
            cf += 1
        if 0 <= reg <= 24:
            cur[reg] = int(val)
    while cf < n_frames:
        state[cf] = cur
        cf += 1
    return state


def _reorder_adf(adf, how, rng=None, move_ctrl=False):
    """Reorder rows within each frame, keeping the diff column POSITIONAL (frame
    timing preserved exactly)."""
    cols = list(adf.columns)
    reg_of = lambda r: int(r[cols.index("reg")])
    sub_of = lambda r: int(r[cols.index("subreg")]) if "subreg" in cols else -1
    rows = adf.to_numpy()
    diffs = adf["diff"].to_numpy().copy()
    fcol = adf["f"].to_numpy()
    out = []
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and fcol[j] == fcol[i]:
            j += 1
        block = [rows[k] for k in range(i, j)]
        if how == "canonical":
            block = canonical_order(block, reg_of, sub_of)
        else:
            block = shuffle_order(block, reg_of, sub_of, rng, move_ctrl)
        out.extend(block)
        i = j
    import pandas as pd

    nd = pd.DataFrame(out, columns=cols).astype(adf.dtypes)
    nd["diff"] = diffs  # positional diff => identical frame timing
    return nd


def _render(adf, rw, irq):
    from preframr_audio.audio_driver import render_to_wav
    import scipy.io.wavfile as wav

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        render_to_wav(adf, path, reg_widths=rw, irq=irq, cents=50)
        _sr, data = wav.read(path)
    finally:
        os.unlink(path)
    return data.astype(np.float64)


def _cmp(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.ndim > 1:
        a, b = a.mean(axis=1), b.mean(axis=1)
    peak = max(float(np.abs(a).max()), float(np.abs(b).max()), 1.0)
    maxabs = float(np.abs(a - b).max()) / peak
    rms = float(np.sqrt(np.mean((a - b) ** 2))) / peak
    sa, sb = a - a.mean(), b - b.mean()
    den = math.sqrt(float((sa * sa).sum()) * float((sb * sb).sum()))
    corr = float((sa * sb).sum() / den) if den else 1.0
    return n, maxabs, rms, corr


def _fidelity(dumps, seed):
    from preframr_audio.sidwav import sidq
    from preframr_tokens import RegLogParser, prepare_df_for_audio, read_initial_irq

    args = _raw_args()
    rng = np.random.default_rng(seed)
    print(
        "fidelity (raw writes): reg-state must be IDENTICAL under reorder; "
        "render diff shows audibility of intra-frame order.\n"
    )
    print(
        f"  {'dump':<26} {'frames':>6} {'reordered%':>10}  "
        f"{'state==':>7}  {'variant':<10} {'maxabs':>8} {'rms':>8} {'corr':>9}"
    )
    for dump in dumps:
        parser = RegLogParser(args=args)
        df = next(parser.parse(dump, max_perm=1, require_pq=False, reparse=True), None)
        if df is None or len(df) == 0:
            print(f"  {os.path.basename(dump):<26} (empty)")
            continue
        irq = read_initial_irq(df)
        adf, rw = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
        canon = _reorder_adf(adf, "canonical")
        shuf = _reorder_adf(adf, "shuffle", rng=rng, move_ctrl=False)
        shufc = _reorder_adf(adf, "shuffle", rng=rng, move_ctrl=True)
        nf = int(adf["f"].max()) + 1
        changed = int((adf["reg"].to_numpy() != canon["reg"].to_numpy()).sum())
        pct = 100.0 * changed / max(len(adf), 1)
        s0 = _reg_state(adf)
        ok = all(np.array_equal(s0, _reg_state(v)) for v in (canon, shuf, shufc))
        base = _render(adf, rw, irq)
        name = os.path.basename(dump)[:26]
        for tag, v in (("canonical", canon), ("shuffle", shuf), ("shuf+ctrl", shufc)):
            _n, ma, rms, corr = _cmp(base, _render(v, rw, irq))
            head = (
                f"  {name:<26} {nf:>6} {pct:>9.1f}%  {str(ok):>7}  "
                if tag == "canonical"
                else " " * 56
            )
            print(f"{head}{tag:<10} {ma:>8.5f} {rms:>8.5f} {corr:>9.6f}")
    print(
        "\n  read: canonical/shuffle ~ original (corr~1, maxabs~0) => "
        "value-latch intra-frame order is inaudible. shuf+ctrl worse => "
        "CTRL must stay anchored (it is, in canonical)."
    )


# ---- divergence mode (full_macros atoms) ---------------------------------

_CAP = {}


def _capture(streams, composers, **kw):
    _CAP.setdefault("s", []).extend(streams)

    class _D:
        def __len__(self):
            return 0

    return _D()


def _cos(a, b):
    ks = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in ks)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else float("nan")


def _mean_off(h, fams):
    s = [
        _cos(h[fams[i]], h[fams[j]])
        for i in range(len(fams))
        for j in range(i + 1, len(fams))
    ]
    s = [x for x in s if x == x]
    return (sum(s) / len(s), min(s), max(s)) if s else (float("nan"),) * 3


def _atom_blocks(stream, frame_reg):
    cur = []
    for a in stream:
        if a[1] == frame_reg:
            if cur:
                yield cur
            cur = [a]
        else:
            cur.append(a)
    if cur:
        yield cur


def _divergence(work, fams):
    from preframr.args import add_args, apply_macro_flags_to_args
    from preframr.utils import get_logger
    import preframr_tokens.motif_mine as MM
    from preframr_tokens.stfconstants import FRAME_REG, VOICE_REG

    MM.mine_motifs = _capture
    base = add_args(argparse.ArgumentParser()).parse_args(
        ["--no-require-pq", "--macro-config", "full_macros", "--max-files", "999999"]
    )
    apply_macro_flags_to_args(base)
    reg_of = lambda a: a[1]
    sub_of = lambda a: a[2]

    bodies = {}
    for fam in fams:
        _CAP["s"] = []
        try:
            MM.mine_dict_from_dumps(
                base,
                f"{work}/eval_b_{fam}/*/*.dump.parquet",
                max_files=999999,
                k=1,
                min_count=1,
                min_composers=1,
                logger=get_logger("ERROR"),
            )
        except ValueError:
            continue
        bodies[fam] = [
            blk[1:]
            for stream in _CAP["s"]
            for blk in _atom_blocks(list(stream), FRAME_REG)
        ]
    present = [f for f in fams if f in bodies]

    def cosine(keyfn, label):
        h = {f: defaultdict(int) for f in present}
        for f in present:
            for body in bodies[f]:
                h[f][keyfn(body)] += 1
        m = _mean_off(h, present)
        print(f"  {label:<44} {m[0]:.3f} ({m[1]:.3f}-{m[2]:.3f})")
        return m[0]

    print("cross-engine cosine -- composition vs multiplicity vs order:")
    s = cosine(lambda b: frozenset(a[1] for a in b), "SET (composition only)")
    ms = cosine(
        lambda b: tuple(sorted(Counter(a[1] for a in b).items())),
        "MULTISET (+ multiplicity)",
    )
    t = cosine(lambda b: tuple(a[1] for a in b), "TUPLE (+ order)")
    c = cosine(
        lambda b: tuple(a[1] for a in voice_canonical(b, VOICE_REG, reg_of, sub_of)),
        "TUPLE voice-canonicalized (legal)",
    )
    print(
        f"\n  multiplicity gap (SET-MULTISET): {s-ms:+.3f}   "
        f"order gap (MULTISET-TUPLE): {ms-t:+.3f}   "
        f"reorder recovers: {c-t:+.3f}"
    )

    same = diff = 0
    for f in present:
        for body in bodies[f]:
            by_reg = defaultdict(list)
            for a in body:
                if a[1] >= 0 or a[1] in CTRL_REGS:
                    by_reg[a[1]].append(a[3])
            for vals in by_reg.values():
                for k in range(1, len(vals)):
                    same += vals[k] == vals[k - 1]
                    diff += vals[k] != vals[k - 1]
    tot = same + diff
    print(
        f"  repeated intra-frame writes: {tot}  same-value (redundant) "
        f"{100*same/max(tot,1):.0f}%  distinct-value (modulation=content) "
        f"{100*diff/max(tot,1):.0f}%"
    )


# ---- cli -----------------------------------------------------------------


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["divergence", "fidelity"], default="fidelity")
    ap.add_argument(
        "--work",
        default="/work",
        help="divergence: dir with eval_b_<fam>/ staged dumps",
    )
    ap.add_argument("--dumps", nargs="*", help="fidelity: dump.parquet paths")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args(argv)
    if cli.mode == "divergence":
        fams = [
            "crisps",
            "daglish",
            "dobek",
            "follin",
            "marquis",
            "mibri",
            "wilson",
            "winterberg",
        ]
        _divergence(cli.work, fams)
    else:
        _fidelity(cli.dumps or [], cli.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
