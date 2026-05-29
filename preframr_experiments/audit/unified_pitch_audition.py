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


FAITHFUL = {
    "PLAIN",
}  # only held notes are faithfully decoded today; ARP/SLIDE decode (order/rate/phase) is not yet
# faithful (it overshoots, octave-high), so those + OCTAVE/VIB/RESID pass through unchanged. NOISE
# (percussion) voices are NOT special-cased: the same pitch encoding applies, the noise waveform is
# preserved (ctrl untouched) so a snapped freq still renders as percussion.


def build_unified_dump(raw, fc):
    """Copy of the raw dump with per-voice FREQ replaced by the unified encode->decode — but ONLY
    for non-noise voices on FAITHFUL descriptors (PLAIN/OCTAVE/ARP/SLIDE). Noise (percussion) voices
    and RESID/VIB notes pass through unchanged (the encoding does not yet faithfully represent them).
    """
    end = int(raw["clock"].max() // fc) + 2
    voice_fn, stats = {}, []
    note_total = note_faithful = 0
    for v, (l1, l2) in enumerate(U.voice_freq_events(raw)):
        recs = U.encode_voice(l1, l2)
        fn_by_frame = {}
        for i, r in enumerate(recs):
            note_total += 1
            if r["desc"].split("|")[0] not in FAITHFUL:
                continue  # ARP/SLIDE/VIB/OCTAVE/RESID -> passthrough (residual; decode not built)
            note_faithful += 1
            onset = r["frame"]
            nxt = recs[i + 1]["frame"] if i + 1 < len(recs) else onset + 8
            for k, m in enumerate(
                U._desc_frames(r["desc"], r["note"], max(1, nxt - onset))
            ):  # NO waveform special-casing: noise (percussion) voices go through the same encoding;
                # the noise waveform is preserved (ctrl untouched), so a snapped freq still renders
                # as percussion.
                if onset + k <= end:
                    fn_by_frame[onset + k] = U.LUT[max(U.MIDI_LO, min(U.MIDI_HI, m))]
        voice_fn[v] = fn_by_frame
        stats.append(
            (v, len(recs), dict(Counter(r["desc"].split("|")[0] for r in recs)))
        )
    out = raw.copy()
    regs = out["reg"].to_numpy()
    clks = out["clock"].to_numpy()
    vals = out["val"].to_numpy().copy()
    replaced = 0
    for i in range(len(out)):
        r = int(regs[i])
        if r in FREQ_REGS:
            fn = voice_fn.get(r // 7, {}).get(int(clks[i]) // fc)
            if fn is not None:
                vals[i] = (fn & 0xFF) if (r % 7 == 0) else ((fn >> 8) & 0xFF)
                replaced += 1
    out["val"] = vals
    nfreq = int(pd.Series(regs).isin(FREQ_REGS).sum())
    print(
        f"  notes: {note_faithful}/{note_total} PLAIN-encoded, "
        f"{note_total - note_faithful} passthrough RESIDUAL (ARP/SLIDE/VIB/OCTAVE/RESID); "
        f"freq writes replaced {replaced}/{nfreq} ({100 * replaced / max(nfreq, 1):.0f}%)"
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
