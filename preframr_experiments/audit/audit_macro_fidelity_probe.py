#!/usr/bin/env python3
"""Macro fidelity probe: decode a dump under raw (all macros off) + leave-one-out
configs from the full strict spec, diff per-voice per-register-class per-frame
state vs raw (drift-free, exact), and rank each macro by the divergence its
removal clears. Renders the top offenders removed, for audition. NOT audio-RMS
(which drifts on long songs); register-state diff is the reliable signal.
"""

from __future__ import annotations

import argparse
import collections
import sys
from types import SimpleNamespace

import numpy as np

from preframr_audio.audio_driver import render_to_wav
from preframr_audio.sidwav import sidq
from preframr_tokens import (
    RegLogParser,
    prepare_df_for_audio,
    read_initial_irq,
)

DEFAULT_DUMP = (
    "/scratch/preframr/training-dumps/MUSICIANS/J/Jammer/Grid_Runner.1.dump.parquet"
)

_BASE = dict(
    cents=50,
    exclude_list=None,
    min_irq=int(1.5e4),
    max_irq=int(2.5e4),
    min_song_tokens=256,
    diffq=4,
    loop_lookahead=3,
    coarsen_min_len=16,
    voice_trajectory_window=8,
    pipeline_spec="",
    meta_exclude_digi=False,
    meta_irq_lo=0,
    meta_irq_hi=0,
    meta_require=False,
)

_MACROS = (
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

_STRICT = dict(
    freq_trajectory_pass=True,
    preset_pass=True,
    hard_restart_pass=True,
    legato_pass_c2=True,
    legato_pass_c4=True,
    voice_canonical_block_order=True,
    ctrl_bigram_pass=True,
    loop_pass=True,
    loop_transposed=True,
    freq_nudge_pass=True,
    release_update_pass=True,
    ctrl_triple_pass=True,
    lonely_catch_all=True,
)

_LOO_FLAGS = tuple(f for f in _STRICT if f != "loop_transposed")

_CLASS_REGS = {}
for _v in range(3):
    _CLASS_REGS[(_v, "FREQ")] = [7 * _v, 7 * _v + 1]
    _CLASS_REGS[(_v, "PW")] = [7 * _v + 2, 7 * _v + 3]
    _CLASS_REGS[(_v, "CTRL")] = [7 * _v + 4]
    _CLASS_REGS[(_v, "AD")] = [7 * _v + 5]
    _CLASS_REGS[(_v, "SR")] = [7 * _v + 6]
_CLASS_REGS[(-1, "FILTER")] = [21, 22, 23]
_CLASS_REGS[(-1, "MODE_VOL")] = [24]


def _make_args(flags):
    cfg = dict(_BASE)
    for macro in _MACROS:
        cfg[macro] = False
    cfg.update(flags)
    return SimpleNamespace(**cfg)


def _reg_state(dump, args):
    """Per-frame state of SID registers 0-24 as an ``(n_frames, 25)`` array."""
    parser = RegLogParser(args=args)
    df = next(parser.parse(dump, max_perm=1, require_pq=False, reparse=True), None)
    if df is None or len(df) == 0:
        return np.zeros((0, 25), dtype=np.int64)
    irq = read_initial_irq(df)
    adf, _ = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
    n_frames = int(adf["f"].max()) + 1
    state = np.zeros((n_frames, 25), dtype=np.int64)
    cur = np.zeros(25, dtype=np.int64)
    cf = 0
    for reg, val, frame in adf[["reg", "val", "f"]].to_numpy():
        reg = int(reg)
        frame = int(frame)
        while cf < frame and cf < n_frames:
            state[cf] = cur
            cf += 1
        if 0 <= reg <= 24:
            cur[reg] = int(val)
    while cf < n_frames:
        state[cf] = cur
        cf += 1
    return state


def _divergence(ref, st):
    """Per (voice, register-class) count of frames whose class state differs."""
    n = min(len(ref), len(st))
    if n == 0:
        return collections.Counter()
    diff = ref[:n] != st[:n]
    out = collections.Counter()
    for key, regs in _CLASS_REGS.items():
        c = int(diff[:, regs].any(axis=1).sum())
        if c:
            out[key] = c
    return out


def _fmt(counter):
    items = sorted(counter.items(), key=lambda kv: -kv[1])
    return ", ".join(
        f"v{k[0]}.{k[1]}={n}" if k[0] >= 0 else f"{k[1]}={n}" for k, n in items
    )


def _render(dump, flags, out_path):
    parser = RegLogParser(args=_make_args(flags))
    df = next(parser.parse(dump, max_perm=1, reparse=True), None)
    irq = read_initial_irq(df)
    adf, rw = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
    n = render_to_wav(adf, out_path, reg_widths=rw, irq=irq, cents=50)
    return n


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=DEFAULT_DUMP)
    ap.add_argument("--render-top", type=int, default=2)
    ap.add_argument("--out-prefix", default="/scratch/tmp/macroprobe")
    cli = ap.parse_args(argv)

    raw = _reg_state(cli.dump, _make_args({}))
    full = _reg_state(cli.dump, _make_args(_STRICT))
    full_div = _divergence(raw, full)
    full_total = sum(full_div.values())
    print(f"full strict vs raw: total={full_total}  [{_fmt(full_div)}]\n")

    ranked = []
    for flag in _LOO_FLAGS:
        loo = dict(_STRICT)
        del loo[flag]
        if flag == "loop_pass":
            loo.pop("loop_transposed", None)
        div = _divergence(raw, _reg_state(cli.dump, _make_args(loo)))
        cleared = full_div - div
        added = div - full_div
        ranked.append((flag, sum(cleared.values()), sum(added.values()), cleared))
    ranked.sort(key=lambda r: -r[1])

    print("leave-one-out (removing the macro CLEARS this much divergence vs raw):")
    for flag, cleared, added, cleared_by in ranked:
        tag = f"  -{flag:34s} clears={cleared:6d}"
        if added:
            tag += f" (adds {added})"
        if cleared_by:
            tag += f"  [{_fmt(cleared_by)}]"
        print(tag)

    top = [r for r in ranked if r[1] > 0][: cli.render_top]
    if top:
        print("\nrendering audition pairs (full vs full-minus-macro):")
        _render(cli.dump, _STRICT, f"{cli.out_prefix}.full.wav")
        print(f"  {cli.out_prefix}.full.wav")
        for flag, _c, _a, _cb in top:
            loo = dict(_STRICT)
            del loo[flag]
            if flag == "loop_pass":
                loo.pop("loop_transposed", None)
            path = f"{cli.out_prefix}.no_{flag}.wav"
            _render(cli.dump, loo, path)
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
