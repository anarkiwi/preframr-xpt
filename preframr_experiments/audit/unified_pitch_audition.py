"""Faithful A/B audition of the unified pitch encoding on a REAL tune (design/unified_pitch_encoding.md).

Both WAVs go through the PRODUCTION audio path — `RegLogParser.parse(dump) -> render_df_to_wav` (the
same path `ablate_pwfilter`/audition tooling use) — so the original is rendered faithfully (proper
IRQ/frame detection, all registers). The only difference:
  *_raw.wav      — the actual dump, parsed + rendered unchanged.
  *_unified.wav  — a COPY of the dump with only the per-voice FREQUENCY register values replaced by the
                   unified encode->decode (skeleton note over the semitone LUT + ornament descriptor),
                   on the real timeline; gate/waveform/ADSR/PW/filter writes untouched. Then parsed +
                   rendered the same way.

So the A/B isolates exactly what the pitch encoding does to a real tune (incl. where RESID-heavy voices
flatten their arps). Runs in the preframr image.

Usage (image): python3 -m preframr_experiments.audit.unified_pitch_audition \
    --dumps '/data/mini_dumps/*.dump.parquet' --pick Commando.2 --out-dir /data/audition
"""

from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path

import pandas as pd

from preframr_experiments.audit import unified_pitch as U

FREQ_REGS = {0, 1, 7, 8, 14, 15}


def _fc(d):
    return int(d["irq"][d["irq"] > 0].mode().iloc[0]) if (d["irq"] > 0).any() else 19592


def _parse_df(path, args):
    from preframr_tokens import RegLogParser

    return next(
        RegLogParser(args=args).parse(
            str(path), max_perm=1, require_pq=False, reparse=True
        ),
        None,
    )


LO_REGS = {0: 1, 7: 8, 14: 15}  # freq-lo reg -> its hi reg, per voice


def build_unified_dump(raw, fc):
    """Copy of the raw dump where every freq write is reconstructed through the unified pitch encoding
    at SETTLED 16-bit resolution: nearest semitone (the skeleton/ornament integer-semitone domain) +
    the sub-semitone CENTS (the vibrato channel, quantised to CENTS_RES). Carries both bytes (the
    actual SID register state, never a mid-update read). With the cents channel the audio residual is
    driven to ~0 (<=CENTS_RES/2 cents, inaudible) — vibrato is preserved, not flattened. Uniform across
    pitched and percussion voices (noise waveform untouched). Returns (modified_df, descriptor stats).
    """
    out = raw.copy()
    regs = out["reg"].to_numpy()
    clks = out["clock"].to_numpy()
    vals = out["val"].to_numpy().copy()
    freqregs = set(LO_REGS) | set(LO_REGS.values())
    # Pass 1: per (voice, 512-bucket) settled freq -> ONE reconstructed 16-bit value (so the lo and
    # hi bytes written back form a coherent freq -- never two bytes from different reconstructions).
    cl = {v: 0 for v in range(3)}
    ch = {v: 0 for v in range(3)}
    recon = {}
    for i in range(len(out)):
        r = int(regs[i])
        if r in freqregs:
            v = r // 7
            if r % 7 == 0:
                cl[v] = int(vals[i])
            else:
                ch[v] = int(vals[i])
            nr = U.fn_to_note_resid((ch[v] << 8) | cl[v])
            recon[(v, int(clks[i]) // U.COMBINE_BUCKET)] = (
                U.fn_from_note_cents(nr[0], nr[1]) if nr else None
            )
    # Pass 2: write each freq byte from its bucket's single reconstructed value.
    done = nfreq = 0
    for i in range(len(out)):
        r = int(regs[i])
        if r in freqregs:
            nfreq += 1
            fn = recon.get((r // 7, int(clks[i]) // U.COMBINE_BUCKET))
            if fn is not None:
                vals[i] = (fn & 0xFF) if (r % 7 == 0) else ((fn >> 8) & 0xFF)
                done += 1
    out["val"] = vals
    # structured-coverage report (separate from audio): what fraction of notes hit a primitive
    stats = []
    for v, (l1, l2) in enumerate(U.voice_freq_events(raw)):
        recs = U.encode_voice(l1, l2)
        stats.append(
            (v, len(recs), dict(Counter(r["desc"].split("|")[0] for r in recs)))
        )
    print(
        f"  freq writes reconstructed (semitone + cents vibrato channel, {U.CENTS_RES}c quantum): "
        f"{done}/{nfreq} (audio residual ~0)"
    )
    return out, stats


def main():
    from preframr_audio.fidelity import render_df_to_wav
    from preframr_tokens import read_initial_irq
    from preframr_tokens.tokenizer_config import named_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--pick", action="append", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cents", type=int, default=50)
    cli = ap.parse_args()
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(cli.dumps))
    picks = (
        [f for f in files if any(p in f for p in cli.pick)] if cli.pick else files[:1]
    )
    args = named_config("baseline")
    args.cents = cli.cents

    for f in picks:
        name = Path(f).name.replace(".dump.parquet", "")
        raw = pd.read_parquet(f)
        fc = _fc(raw)
        uni_dump, stats = build_unified_dump(raw, fc)
        uni_path = cli.out_dir / f"{name}.unified.dump.parquet"
        uni_dump.to_parquet(uni_path)

        for tag, src in (("raw", f), ("unified", uni_path)):
            df = _parse_df(src, args)
            if df is None or len(df) == 0:
                print(f"{name} {tag}: parse empty")
                continue
            irq = read_initial_irq(df)
            n, _ = render_df_to_wav(df, irq, args, cli.out_dir / f"{name}_{tag}.wav")
            print(
                f"{name} {tag}: {n} samples, irq={irq}, {int((df['reg'] == -128).sum())} frames"
            )
        tot = sum(n for _, n, _ in stats)
        plain = sum(t.get("PLAIN", 0) for _, _, t in stats)
        print(
            f"  unified pitch: {tot} notes, PLAIN={plain} ({100 * plain / max(tot, 1):.0f}%)"
        )
        for v, n, t in stats:
            print(f"    voice {v}: {n} notes {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
