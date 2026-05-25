#!/usr/bin/env python3
"""Cohort audition: render each SID twice -- raw (all macros off, the faithful
reference) and the FREQ_TRAJ "new spec" (REGISTERED_MACROS, == the prodlike
target arm) -- so the post-refactor audio can be heard before re-cutting prodlike.
Reports the drift-free per-frame register-state divergence (reliable signal) and
compare_renders audio RMS (supplementary; drifts on long songs)."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from preframr_audio.fidelity import compare_renders, render_df_to_wav
from preframr_tokens import RegLogParser, read_initial_irq
from preframr_tokens.tokenizer_config import named_config

COHORT = [
    ("eval_a", "galway", "MUSICIANS/G/Galway_Martin/Comic_Bakery.2.dump.parquet"),
    ("eval_a", "hubbard", "MUSICIANS/H/Hubbard_Rob/Flash_Gordon.7.dump.parquet"),
    ("eval_a", "tel", "MUSICIANS/T/Tel_Jeroen/Ladys_Own.1.dump.parquet"),
    ("canary", "jammer", "MUSICIANS/J/Jammer/Grid_Runner.1.dump.parquet"),
    ("eval_b", "crisps", "MUSICIANS/C/Crisps/Mineshaft_Gap.2.dump.parquet"),
    ("eval_b", "daglish", "MUSICIANS/D/Daglish_Ben/Jack_the_Nipper.3.dump.parquet"),
    ("eval_b", "dobek", "MUSICIANS/D/Dobek_Eric/Reverb_Insanity.1.dump.parquet"),
    ("eval_b", "follin", "MUSICIANS/F/Follin_Tim/Test_Bits.4.dump.parquet"),
    ("eval_b", "marquis", "MUSICIANS/M/Marquis_Dave/Mystery_19.1.dump.parquet"),
    ("eval_b", "mibri", "MUSICIANS/M/Mibri/Tenebra_Macabre.3.dump.parquet"),
    (
        "eval_b",
        "wilson",
        "MUSICIANS/W/Wilson_Mark/Santas_Christmas_Capers.7.dump.parquet",
    ),
    ("eval_b", "winterberg", "MUSICIANS/W/Winterberg_Michael/Topcross.2.dump.parquet"),
]

_CLASS_REGS = {}
for _v in range(3):
    _CLASS_REGS[(_v, "FREQ")] = [7 * _v, 7 * _v + 1]
    _CLASS_REGS[(_v, "PW")] = [7 * _v + 2, 7 * _v + 3]
    _CLASS_REGS[(_v, "CTRL")] = [7 * _v + 4]
    _CLASS_REGS[(_v, "AD")] = [7 * _v + 5]
    _CLASS_REGS[(_v, "SR")] = [7 * _v + 6]
_CLASS_REGS[(-1, "FILTER")] = [21, 22, 23]
_CLASS_REGS[(-1, "MODE_VOL")] = [24]


def _parse(dump, args):
    """Parse one dump under ``args``; return (df, irq) or (None, None)."""
    parser = RegLogParser(args=args)
    df = next(parser.parse(dump, max_perm=1, require_pq=False, reparse=True), None)
    if df is None or len(df) == 0:
        return None, None
    return df, read_initial_irq(df)


def _reg_state(df, irq):
    """Per-frame state of SID registers 0-24 as an (n_frames, 25) array."""
    from preframr_audio.sidwav import sidq
    from preframr_tokens import prepare_df_for_audio

    adf, _ = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
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


def _divergence(ref, st):
    """Per (voice, register-class) count of frames whose class state differs."""
    n = min(len(ref), len(st))
    out = collections.Counter()
    if n == 0:
        return out
    diff = ref[:n] != st[:n]
    for key, regs in _CLASS_REGS.items():
        c = int(diff[:, regs].any(axis=1).sum())
        if c:
            out[key] = c
    return out


def _fmt(counter, top=6):
    items = sorted(counter.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(
        f"v{k[0]}.{k[1]}={n}" if k[0] >= 0 else f"{k[1]}={n}" for k, n in items
    )


def _audition_one(dump, raw_wav, new_wav):
    """Render raw + new-spec WAVs; return a result dict (or {'error': ...})."""
    raw_args = named_config("baseline")
    new_args = named_config("full_macros")
    raw_df, raw_irq = _parse(dump, raw_args)
    new_df, new_irq = _parse(dump, new_args)
    if raw_df is None or new_df is None:
        return {"error": "no rows from parse"}

    render_df_to_wav(raw_df, raw_irq, raw_args, raw_wav)
    render_df_to_wav(new_df, new_irq, new_args, new_wav)

    raw_state = _reg_state(raw_df, raw_irq)
    new_state = _reg_state(new_df, new_irq)
    div = _divergence(raw_state, new_state)
    total = sum(div.values())
    n_frames = min(len(raw_state), len(new_state))

    sr_a, a = wavfile.read(raw_wav)
    _, b = wavfile.read(new_wav)
    n = min(len(a), len(b))
    cmp = compare_renders(a[:n], b[:n], sr_a)
    return {
        "n_frames": n_frames,
        "reg_divergence_total": total,
        "reg_divergence_frac": (total / n_frames) if n_frames else 0.0,
        "reg_divergence_top": _fmt(div),
        "audio_passed": bool(cmp.passed),
        "audio_shape": cmp.shape,
        "audio_worst_frame_rel_rms": cmp.worst_frame_rel_rms,
        "sample_rate": int(sr_a),
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps-root", default="/scratch/preframr/training-dumps")
    ap.add_argument("--out-dir", default="/scratch/tmp/preframr_audition")
    cli = ap.parse_args(argv)
    root = Path(cli.dumps_root)
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []
    header = f"{'slug':<14}{'family':<8}{'frames':>7}{'reg_div%':>9}  {'audio':<22}top reg classes"
    print(header)
    print("-" * len(header))
    for family, slug, rel in COHORT:
        dump = root / rel
        raw_wav = out / f"{slug}.raw.wav"
        new_wav = out / f"{slug}.newspec.wav"
        if not dump.exists():
            print(f"{slug:<14}{family:<8}  MISSING {rel}")
            manifest.append({"slug": slug, "family": family, "error": "missing dump"})
            continue
        try:
            res = _audition_one(str(dump), str(raw_wav), str(new_wav))
        except Exception as exc:  # noqa: BLE001
            print(f"{slug:<14}{family:<8}  ERROR {exc}")
            manifest.append({"slug": slug, "family": family, "error": str(exc)})
            continue
        if "error" in res:
            print(f"{slug:<14}{family:<8}  {res['error']}")
        else:
            audio = "OK" if res["audio_passed"] else res["audio_shape"]
            print(
                f"{slug:<14}{family:<8}{res['n_frames']:>7}"
                f"{res['reg_divergence_frac'] * 100:>8.1f}%  {audio:<22}"
                f"{res['reg_divergence_top']}"
            )
        res.update({"slug": slug, "family": family, "source": rel})
        res["raw_wav"] = str(raw_wav)
        res["newspec_wav"] = str(new_wav)
        manifest.append(res)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWAVs + manifest.json under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
