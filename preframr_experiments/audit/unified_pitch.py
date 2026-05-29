"""Unified pitch encoder/decoder (design/unified_pitch_encoding.md): factor a voice's per-frame
SID frequency stream into a SKELETON melody line (semitone intervals over one note->freq LUT) + a
per-note PITCH-ORNAMENT descriptor (PLAIN / OCTAVE-ARP / ARP-offset-cycle / VIBRATO / SLIDE / RESID),
and decode back to per-frame frequency. Host-side (no torch). The driver model behind this is in
design/sid_driver_ornament_reference.md.

Key facts used: SID pitch is a semitone index into a note->freq table; gate-on NOTES sit ~1.5c from a
semitone (so note = LUT index, ~lossless) while ornament writes spread (~17c, the cents live here);
notes must be segmented by intrinsic level-change UNION gate (held-gate/legato drivers), not gate alone.
"""

from __future__ import annotations

import argparse
import glob
import json
import math

from preframr_experiments.audit.extract_sid_melody import CLOCK_RATE, fn_to_midi

MIDI_LO, MIDI_HI = 16, 112
MIN_HOLD = (
    3  # frames a new pitch level must persist to count as a note (vs an arp step)
)
PLAIN_CENTS = 20.0  # |residual| below this with no intra-note motion = PLAIN
VIB_MIN_CENTS = 8.0
ARP_MAX_DISTINCT = 4


def midi_to_fn_f(mf: float) -> int:
    """Continuous MIDI (note + cents/100) -> 16-bit SID freq. The sub-semitone (cents) part is the
    vibrato channel; reconstructing through this keeps the audio residual below the cents-quantum.
    """
    hz = 440.0 * 2 ** ((mf - 69) / 12.0)
    return max(0, min(0xFFFF, int(round(hz * 16777216.0 / CLOCK_RATE))))


def midi_to_fn(m: int) -> int:
    return midi_to_fn_f(float(m))


# unified semitone LUT: MIDI note -> 16-bit SID frequency word
LUT = [midi_to_fn(m) for m in range(128)]
CENTS_RES = 4  # vibrato-channel cents quantum (<=2c reconstruction error -> inaudible)


def fn_from_note_cents(note: int, cents: float) -> int:
    """Reconstruct a 16-bit freq from a semitone note + a cents offset, quantised to CENTS_RES."""
    qc = round(cents / CENTS_RES) * CENTS_RES
    return midi_to_fn_f(note + qc / 100.0)


def fn_to_note_resid(fn: int):
    """16-bit freq -> (nearest MIDI semitone, residual in cents). None if silent/out of range."""
    if fn < 8:
        return None
    hz = fn * CLOCK_RATE / 16777216.0
    if hz < 16:
        return None
    mf = 69 + 12 * math.log2(hz / 440.0)
    note = int(round(mf))
    if not (MIDI_LO <= note <= MIDI_HI):
        return None
    return note, (mf - note) * 100.0


COMBINE_BUCKET = (
    512  # = RegLogParser diffmax: a coordinated lo+hi update lands in one bucket
)


def voice_freq_events(d):
    """Per-voice (l1, l2) of the SETTLED 16-bit freq. l1 = freq changes (frame, fn); l2 = gate-on
    (frame, settled fn).

    Freq is 16-bit across lo+hi; reading on every byte write yields half-updated garbage (a lo write
    with a stale hi byte = a spurious pitch). This reproduces RegLogParser._combine_reg: carry both
    bytes and take the final value per CLOCK//512 bucket, so a coordinated lo+hi update's settled
    value is read (never a mid-update or cross-frame straddle), reported at that fine granularity
    (sub-frame arp steps preserved; segmentation handles within-frame runs)."""
    fc = int(d["irq"][d["irq"] > 0].mode().iloc[0]) if (d["irq"] > 0).any() else 19592
    reg = d["reg"].to_numpy()
    val = d["val"].to_numpy()
    clk = d["clock"].to_numpy()
    out = []
    for v in range(3):
        lo, hi, ctrl = v * 7, v * 7 + 1, v * 7 + 4
        cl = ch = gate = 0
        buck_fn, buck_frame, gate_on = {}, {}, []
        for i in range(len(reg)):
            b = int(clk[i]) // COMBINE_BUCKET
            r, x = int(reg[i]), int(val[i])
            if r == lo:
                cl = x
            elif r == hi:
                ch = x
            elif r == ctrl:
                if (x & 1) and not gate:
                    gate_on.append(b)
                gate = x & 1
            buck_fn[b] = (ch << 8) | cl
            buck_frame[b] = int(clk[i]) // fc
        l1, prev = [], None
        for b in sorted(buck_fn):
            fn = buck_fn[b]
            if fn != prev and fn_to_note_resid(fn):
                l1.append((buck_frame[b], fn))
                prev = fn
        l2 = [
            (buck_frame[b], buck_fn[b])
            for b in sorted(set(gate_on))
            if b in buck_fn and fn_to_note_resid(buck_fn[b])
        ]
        out.append((l1, l2))
    return out


def segment_notes(l1, l2, min_hold=MIN_HOLD):
    """Note onsets = intrinsic level-change (a new semitone that HOLDS >= min_hold frames, so fast
    arp steps don't count) UNION gate-on. Returns sorted [(frame, note)]."""
    notes = [(fr, fn_to_note_resid(fn)[0]) for fr, fn in l1]
    onsets = {}
    cur = None
    for i, (fr, note) in enumerate(notes):
        nxt = notes[i + 1][0] if i + 1 < len(notes) else fr + min_hold
        if note != cur and (nxt - fr) >= min_hold:
            onsets[fr] = note
            cur = note
    for fr, fn in l2:  # union gate-ons (re-struck notes)
        onsets.setdefault(fr, fn_to_note_resid(fn)[0])
    return sorted(onsets.items())


def _fit_descriptor(base, seg_writes):
    """seg_writes = [(frame, fn)] within a note. Return (descriptor_string, note_relative_offsets)."""
    if not seg_writes:
        return "PLAIN", []
    notes = [fn_to_note_resid(fn)[0] for _, fn in seg_writes]
    resids = [abs(fn_to_note_resid(fn)[1]) for _, fn in seg_writes]
    offs = [n - base for n in notes]
    distinct = sorted(set(offs))
    nonzero = [o for o in offs if o != 0]
    max_resid = max(resids) if resids else 0.0
    if not nonzero:  # stays on the base semitone
        if max_resid >= VIB_MIN_CENTS:
            depth = min(3, int(max_resid // 16) + 1)
            return f"VIB|{depth}", offs
        return "PLAIN", offs
    if set(distinct) <= {0, 12} or set(distinct) <= {0, -12}:
        return f"OCTAVE|{'+' if 12 in distinct else '-'}", offs
    diffs = [b - a for a, b in zip(offs, offs[1:])]
    if (all(x >= 0 for x in diffs) or all(x <= 0 for x in diffs)) and abs(
        offs[-1] - offs[0]
    ) >= 2:
        return f"SLIDE|{'+' if offs[-1] >= offs[0] else '-'}", offs
    if len(distinct) <= ARP_MAX_DISTINCT:
        return "ARP|" + ",".join(str(x) for x in distinct), offs
    return "RESID", offs


def encode_voice(l1, l2, min_hold=MIN_HOLD):
    """Per-note records: {frame, note, skel(interval from prev note), desc, offs}. The skeleton is a
    signed semitone interval over the LUT (key-invariant); desc is the unified ornament primitive.
    """
    onsets = segment_notes(l1, l2, min_hold)
    if len(onsets) < 2:
        return []
    out = []
    prev = None
    for i, (fr, note) in enumerate(onsets):
        nxt = onsets[i + 1][0] if i + 1 < len(onsets) else fr + 8
        seg = [(f, fn) for f, fn in l1 if fr < f < nxt]
        desc, offs = _fit_descriptor(note, seg)
        skel = 0 if prev is None else max(-24, min(24, note - prev))
        prev = note
        out.append(
            {
                "frame": fr,
                "note": note,
                "skel": skel,
                "desc": desc,
                "offs": offs,
                "vib": _vib_bucket(seg),
            }
        )
    return out


def _vib_bucket(seg_writes):
    """Per-note sub-semitone VIBRATO depth bucket from the cents amplitude of the note's settled
    freqs that sit on ONE semitone (the held-note wobble): 0 none / 1 light (<=30c) / 2 heavy.
    Integer-semitone motion (arps) has ~0 cents on each pitch -> bucket 0 (vibrato is the
    sub-semitone channel, orthogonal to the arp/slide integer-offset descriptor)."""
    cents = [
        fn_to_note_resid(fn)[1]
        for _, fn in seg_writes
        if fn_to_note_resid(fn) and abs(fn_to_note_resid(fn)[1]) < 50
    ]
    if len(cents) < 3:
        return 0
    amp = (max(cents) - min(cents)) / 2.0
    return 0 if amp < VIB_MIN_CENTS else (1 if amp <= 30 else 2)


# ---- decode (notes -> per-frame freq) for round-trip + audition --------------------------------


def _desc_frames(desc, base, dur):
    """Per-frame MIDI notes (length dur) realising a descriptor on base note (ornament synthesis)."""
    t = desc.split("|")[0]
    if t == "OCTAVE":
        hi = base + (12 if desc.endswith("+") else -12)
        return [base if f % 2 == 0 else hi for f in range(dur)]
    if t == "ARP":
        offs = [int(x) for x in desc.split("|")[1].split(",")] or [0]
        return [base + offs[f % len(offs)] for f in range(dur)]
    if t == "SLIDE":
        end = base + (3 if desc.endswith("+") else -3)
        return [base + round((end - base) * f / max(1, dur - 1)) for f in range(dur)]
    return [
        base
    ] * dur  # PLAIN / VIB / RESID -> hold base (VIB is sub-semitone, omitted from LUT path)


def decode_notes(records, frames_per_note=10):
    """Reconstruct per-frame 16-bit freq from skeleton+ornament records (LUT[note] + ornament).
    Returns [(note, fn)] per frame. PLAIN notes are LUT-exact."""
    out = []
    base = (
        records[0].get("note", 60) if records else 60
    )  # key-invariant seed; contour is audible
    for r in records:
        base = max(MIDI_LO, min(MIDI_HI, base + r["skel"]))
        for m in _desc_frames(r["desc"], base, frames_per_note):
            mm = max(MIDI_LO, min(MIDI_HI, m))
            out.append((mm, LUT[mm]))
    return out


def decode_voice_timeline(records, end_frame):
    """Real-timeline decode: frame -> MIDI note for one voice, using each record's actual onset
    frame and the next onset as the note duration (so rhythm is preserved), synthesising the
    descriptor's ornament over that duration. Uses the encoded absolute note (LUT[note] is the
    decoded pitch). For a faithful round-trip of a real tune (not a key-invariant model render).
    """
    frames = {}
    for i, r in enumerate(records):
        onset = r["frame"]
        nxt = records[i + 1]["frame"] if i + 1 < len(records) else onset + 8
        dur = max(1, nxt - onset)
        for k, m in enumerate(_desc_frames(r["desc"], r["note"], dur)):
            frames[onset + k] = max(MIDI_LO, min(MIDI_HI, m))
    return frames


def main():
    """Extract per-(dump, lead voice) unified per-note records for the generalization probe."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--min-notes", type=int, default=12)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    import pandas as pd

    files = sorted(glob.glob(cli.dumps))
    seqs = []
    for di, f in enumerate(files):
        d = pd.read_parquet(f).sort_values("clock")
        cand = []
        for l1, l2 in voice_freq_events(d):
            recs = encode_voice(l1, l2)
            if len(recs) >= cli.min_notes:
                cand.append(recs)
        if not cand:
            continue
        recs = max(cand, key=len)  # lead = most notes
        seqs.append(
            {
                "dump": di,
                "notes": [
                    {"skel": r["skel"], "desc": r["desc"], "vib": r["vib"]}
                    for r in recs
                ],
            }
        )
    json.dump({"seqs": seqs}, open(cli.out, "w"))
    notes = sum(len(s["notes"]) for s in seqs)
    orn = sum(1 for s in seqs for n in s["notes"] if n["desc"] != "PLAIN")
    print(
        f"{cli.out}: {len(seqs)} seqs, {notes} notes, {orn} ornamented "
        f"({100 * orn / max(notes, 1):.0f}%) from {len(files)} dumps"
    )


if __name__ == "__main__":
    main()
