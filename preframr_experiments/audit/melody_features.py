"""SID-dump musicality features (substrate-aware: ignores PW and filter regs).

Pipeline:
  dump.parquet (clock/irq/chipno/reg/val)
    -> per-voice notes  (start_s, end_s, midi_pitch, voice)
    -> pretty_midi.PrettyMIDI (one Instrument per voice, drums voice marked)
    -> muspy.Music
    -> feature vector  (pitch_class_entropy, scale_consistency,
                        pitch_in_scale_rate, pitch_range, n_pitches_used,
                        n_pitch_classes_used, polyphony, polyphony_rate,
                        groove_consistency, empty_beat_rate,
                        + SID-specific: gates_per_sec, median_note_frames,
                        unpitched_gate_rate, top_interval_share)

Run on a corpus to build a baseline, then score generations against it.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass

import pandas as pd

# muspy + pretty_midi are imported lazily so the SID-only paths (dump_to_notes,
# _ensure_clock, the score/verdict helpers, the tests for those) run on hosts
# without the MIR stack installed.

PAL_CLOCK = 17734475
FRAME_CYCLES = 19656
FRAME_HZ = PAL_CLOCK / FRAME_CYCLES  # ~50.125
PCM_BIT = 0x80  # control bit: noise/PCM waveform


def _voice_regs(v: int) -> dict[str, int]:
    base = v * 7
    return {
        "freq_lo": base + 0,
        "freq_hi": base + 1,
        "pw_lo": base + 2,
        "pw_hi": base + 3,
        "ctrl": base + 4,
        "ad": base + 5,
        "sr": base + 6,
    }


def _midi_of(freq_word: int) -> int | None:
    if freq_word <= 0:
        return None
    hz = freq_word * PAL_CLOCK / (1 << 24)
    if hz < 20 or hz > 8000:
        return None
    midi = 69.0 + 12.0 * math.log2(hz / 440.0)
    return int(round(midi))


@dataclass
class Note:
    voice: int
    start_s: float
    end_s: float
    pitch: int | None
    waveform: int  # control reg bits 4..7
    velocity: int = 80


PHI2_TO_MASTER = 18  # raw dumps store clock in PAL master cycles; audio dfs
                     # store diff in PHI2 cycles. Convert when needed so the
                     # rest of the analyzer uses one unit (PAL master).


def _ensure_clock(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dumps have a ``clock`` column in PAL master cycles; audio-render
    dfs (from predict.py ``--predict-dump``) have ``diff`` in PHI2 cycles
    only. Derive clock from cumsum(diff)*PHI2_TO_MASTER in the latter case
    so PAL_CLOCK-based time math applies uniformly."""
    if "clock" in df.columns:
        return df
    if "diff" in df.columns:
        df = df.copy()
        df["clock"] = df["diff"].cumsum() * PHI2_TO_MASTER
        return df
    raise ValueError("dump has neither 'clock' nor 'diff' column")


def dump_to_notes(df: pd.DataFrame) -> list[Note]:
    """Extract per-voice notes from a SID dump.

    PW (regs 2/3/9/10/16/17) and filter (regs 21/22/23) writes are ignored
    by design -- this is melody/rhythm/timbre-base only."""
    df = _ensure_clock(df)
    df = df.sort_values("clock", kind="stable").reset_index(drop=True)
    reg_to_voice = {}
    for v in range(3):
        for name, r in _voice_regs(v).items():
            reg_to_voice[r] = (v, name)
    state = {v: {"freq_lo": 0, "freq_hi": 0, "ctrl": 0, "gate": 0} for v in range(3)}
    open_note: dict[int, Note | None] = {v: None for v in range(3)}
    notes: list[Note] = []
    for row in df.itertuples(index=False):
        clock = int(row.clock)
        reg = int(row.reg)
        val = int(row.val)
        if reg not in reg_to_voice:
            continue
        v, name = reg_to_voice[reg]
        state[v][name] = val
        if name != "ctrl":
            continue
        new_gate = val & 0x01
        cur_gate = state[v]["gate"]
        if new_gate and not cur_gate:
            freq_word = state[v]["freq_lo"] + 256 * state[v]["freq_hi"]
            pitch = _midi_of(freq_word)
            wf = val & 0xF0
            open_note[v] = Note(
                voice=v,
                start_s=clock / PAL_CLOCK,
                end_s=clock / PAL_CLOCK,
                pitch=pitch,
                waveform=wf,
            )
        elif cur_gate and not new_gate:
            n = open_note[v]
            if n is not None:
                n.end_s = clock / PAL_CLOCK
                notes.append(n)
                open_note[v] = None
        state[v]["gate"] = new_gate
    # Close still-open notes at the final clock.
    tail = float(df["clock"].max()) / PAL_CLOCK
    for v in range(3):
        n = open_note[v]
        if n is not None:
            n.end_s = max(tail, n.start_s + 1.0 / FRAME_HZ)
            notes.append(n)
    return notes


def notes_to_pretty_midi(notes: list[Note]):
    """One Instrument per voice; voice 2 marked as drums when its
    waveforms are >50% noise (PCM_BIT set on the gate). Returns a
    pretty_midi.PrettyMIDI -- raises ImportError if pretty_midi missing."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    by_voice: dict[int, list[Note]] = {0: [], 1: [], 2: []}
    for n in notes:
        by_voice[n.voice].append(n)
    for v, vn in by_voice.items():
        if not vn:
            continue
        noise_share = sum(1 for n in vn if n.waveform & PCM_BIT) / len(vn)
        is_drum = v == 2 and noise_share > 0.5
        inst = pretty_midi.Instrument(program=0, is_drum=is_drum, name=f"V{v}")
        for n in vn:
            if n.pitch is None:
                continue  # muspy needs pitched notes; unpitched tracked separately
            if 0 <= n.pitch < 128 and n.end_s > n.start_s:
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=n.velocity,
                        pitch=int(n.pitch),
                        start=float(n.start_s),
                        end=float(n.end_s),
                    )
                )
        if inst.notes:
            pm.instruments.append(inst)
    return pm


# muspy keys: returned per-Music, computed by pooling all non-drum notes.
_MUSPY_FEATURES = (
    "pitch_class_entropy",
    "scale_consistency",
    "pitch_in_scale_rate",
    "pitch_range",
    "n_pitches_used",
    "n_pitch_classes_used",
    "polyphony",
    "polyphony_rate",
    "groove_consistency",
    "empty_beat_rate",
)


def _safe_call(fn, *args, **kwargs):
    try:
        v = fn(*args, **kwargs)
    except Exception:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def features_from_notes(notes: list[Note], total_seconds: float) -> dict[str, float]:
    """Per-tune feature vector. Drums are included as a flagged Instrument;
    muspy.scale_consistency / pitch_in_scale_rate skip is_drum tracks
    automatically. Adds SID-specific features that aren't in muspy. If
    muspy/pretty_midi aren't installed the muspy features are returned
    as None and the SID-specific features still compute."""
    out: dict[str, float | None] = {}
    try:
        import muspy
        pm = notes_to_pretty_midi(notes)
        has_pitched = pm.instruments and any(
            inst.notes for inst in pm.instruments if not inst.is_drum
        )
    except ImportError:
        muspy = None
        has_pitched = False
    if muspy is not None and has_pitched:
        music = muspy.from_pretty_midi(pm)
        out["pitch_class_entropy"] = _safe_call(muspy.pitch_class_entropy, music)
        out["scale_consistency"] = _safe_call(muspy.scale_consistency, music)
        out["pitch_in_scale_rate"] = _safe_call(
            muspy.pitch_in_scale_rate, music, root=0, mode="major"
        )
        out["pitch_range"] = _safe_call(muspy.pitch_range, music)
        out["n_pitches_used"] = _safe_call(muspy.n_pitches_used, music)
        out["n_pitch_classes_used"] = _safe_call(muspy.n_pitch_classes_used, music)
        out["polyphony"] = _safe_call(muspy.polyphony, music)
        out["polyphony_rate"] = _safe_call(muspy.polyphony_rate, music)
        out["groove_consistency"] = _safe_call(muspy.groove_consistency, music)
        out["empty_beat_rate"] = _safe_call(muspy.empty_beat_rate, music)
    else:
        for k in _MUSPY_FEATURES:
            out[k] = None
    # SID-specific features
    n_pitched = sum(1 for n in notes if n.pitch is not None)
    n_unpitched = sum(1 for n in notes if n.pitch is None)
    out["gates_per_sec"] = (len(notes) / total_seconds) if total_seconds > 0 else 0.0
    out["pitched_gates_per_sec"] = (
        (n_pitched / total_seconds) if total_seconds > 0 else 0.0
    )
    out["unpitched_gate_rate"] = (n_unpitched / len(notes)) if notes else 0.0
    durs = sorted(
        (n.end_s - n.start_s) * FRAME_HZ for n in notes if n.end_s > n.start_s
    )
    out["median_note_frames"] = durs[len(durs) // 2] if durs else 0.0
    intervals = []
    for v in range(3):
        vn = [n for n in notes if n.voice == v and n.pitch is not None]
        for i in range(1, len(vn)):
            d = vn[i].pitch - vn[i - 1].pitch
            if -24 <= d <= 24:
                intervals.append(d)
    if intervals:
        top = Counter(intervals).most_common(1)[0][1]
        out["top_interval_share"] = top / len(intervals)
        out["interval_repeat_share"] = (
            Counter(intervals).get(0, 0) / len(intervals)
        )
    else:
        out["top_interval_share"] = None
        out["interval_repeat_share"] = None
    return {k: (None if v is None else float(v)) for k, v in out.items()}


FEATURES = list(_MUSPY_FEATURES) + [
    "gates_per_sec",
    "pitched_gates_per_sec",
    "unpitched_gate_rate",
    "median_note_frames",
    "top_interval_share",
    "interval_repeat_share",
]


def analyze(path: str) -> dict[str, float | None]:
    df = pd.read_parquet(path)
    df = _ensure_clock(df)
    total_seconds = float(df["clock"].max()) / PAL_CLOCK
    notes = dump_to_notes(df)
    feat = features_from_notes(notes, total_seconds)
    feat["_path"] = path
    feat["_duration_s"] = total_seconds
    feat["_n_notes"] = float(len(notes))
    return feat


def main() -> int:
    rows = [analyze(p) for p in sys.argv[1:]]
    cols = ["_path", "_duration_s", "_n_notes", *FEATURES]
    for r in rows:
        print(r["_path"])
        for c in cols[1:]:
            v = r.get(c)
            s = "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
            print(f"  {c:30s} {s}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
