"""Viability probe for inaudible-macro-perturbation augmentation: perturb a raw
voice-register class, re-parse + render, gate with compare_renders, and report
per knob+magnitude applicable/inaudible/inaudible-and-token-diverse (the last =
usable yield: inaudible AND a changed token stream). CPU-only."""

from __future__ import annotations

import argparse
import glob
import hashlib
import random
from pathlib import Path

import pandas as pd

from preframr.args import add_args
from preframr_audio.audio_driver import render_to_samples
from preframr_audio.fidelity import compare_renders
from preframr_audio.sidwav import sidq
from preframr_tokens.reglogparser import (
    RegLogParser,
    prepare_df_for_audio,
    read_initial_irq,
)
from preframr_tokens.stfconstants import DUMP_SUFFIX, FC_LO_REG, VOICE_REG_SIZE, VOICES


def _base_args():
    p = argparse.ArgumentParser()
    add_args(p)
    return p.parse_args([])


def _parse(args, path: str):
    rot = list(
        RegLogParser(args).parse(path, max_perm=1, require_pq=False, reparse=True)
    )
    return rot[0] if rot else None


def _render(df):
    irq = read_initial_irq(df)
    da, rw = prepare_df_for_audio(df, {}, irq, sidq(), strict=False)
    return render_to_samples(da, reg_widths=rw, irq=irq, cents=50)


def _token_sig(df):
    cols = [c for c in ("op", "reg", "subreg", "val") if c in df.columns]
    return (len(df), hashlib.md5(df[cols].to_numpy().tobytes()).hexdigest()[:8])


def _voice_regs(offset: int) -> list[int]:
    return [v * VOICE_REG_SIZE + offset for v in range(VOICES)]


KNOBS = {
    "FREQ_LO(freq_traj)": _voice_regs(0),
    "PW_LO(pwm_preset)": _voice_regs(2),
    "AD+SR(adsr/rel_upd)": _voice_regs(5) + _voice_regs(6),
    "FC_HI(fc_preset)": [FC_LO_REG + 1],
}
DELTAS = (2, 8, 32)


def _perturb(raw: pd.DataFrame, regs: list[int], delta: int):
    d = raw.copy()
    m = d["reg"].isin(regs)
    if not m.any():
        return None
    d.loc[m, "val"] = (
        (d.loc[m, "val"].astype(int) + delta).clip(0, 255).astype(d["val"].dtype)
    )
    return d if not d["val"].equals(raw["val"]) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/preframr/training-dumps")
    ap.add_argument("--tmp", default="/scratch/tmp/aug_probe")
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    args = _base_args()
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    dumps = glob.glob(f"{a.root}/**/*.dump.parquet", recursive=True)
    random.Random(a.seed).shuffle(dumps)
    dumps = dumps[: a.sample]

    tallies = {k: {d: [0, 0, 0] for d in DELTAS} for k in KNOBS}
    done = 0
    for path in dumps:
        try:
            raw = pd.read_parquet(path).iloc[: a.max_rows].copy()
            op = tmp / ("orig" + DUMP_SUFFIX)
            raw.to_parquet(op)
            orig = _parse(args, str(op))
            orig_s, sr = _render(orig)
            orig_sig = _token_sig(orig)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {Path(path).name}: {exc}")
            continue
        done += 1
        for knob, regs in KNOBS.items():
            for delta in DELTAS:
                pr = _perturb(raw, regs, delta)
                if pr is None:
                    continue
                t = tallies[knob][delta]
                t[0] += 1
                try:
                    pp = tmp / ("pert" + DUMP_SUFFIX)
                    pr.to_parquet(pp)
                    parsed = _parse(args, str(pp))
                    ps, _ = _render(parsed)
                    inaud = compare_renders(orig_s, ps, sr).passed
                    if inaud:
                        t[1] += 1
                        if _token_sig(parsed) != orig_sig:
                            t[2] += 1
                except Exception:  # noqa: BLE001
                    pass

    print(f"\nrendered {done}/{len(dumps)} dumps (max_rows={a.max_rows})")
    print("cells = applicable | inaudible | inaudible&token-diverse (= usable yield)\n")
    print(f"{'knob':22s} " + " ".join(f"d={d:<10d}" for d in DELTAS))
    for knob in KNOBS:
        cells = []
        for d in DELTAS:
            ap_, ia, dv = tallies[knob][d]
            cells.append(f"{ap_}|{ia}|{dv}" if ap_ else "-")
        print(f"{knob:22s} " + " ".join(f"{c:<12s}" for c in cells))


if __name__ == "__main__":
    main()
