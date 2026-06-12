# Recovered-table pitch model — historical record

**Status:** **ABSORBED into the v3 event model 2026-06-12** (tokens 0.47.0). The design's three
components shipped as event-stream primitives: the universal semitone NOTE index → the `NI_*`
note-index lane (intervals are Δnote by construction); the **per-voice recovered note→freq table** →
the `NOTE_TABLE` per-voice stream header; per-voice tuning → the `TUNING` header; modulation residual
→ the `FD_*` freq-deviation lane. The gated `universal_pitch` macro flag this doc planned is moot —
the event encoding is unconditional. See the
[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens) for the shipped form.

## The findings that survive (cite these, not the mechanism)

- **There is ONE universal grid.** Both drivable trackers' ground-truth note→freq tables ARE the
  `2^(n/12)` curve anchored at C5≈4455 (PAL) within ±1 LSB (pysidwizard 96/96, pydefmon 119/127);
  "per-tracker LUT" is a myth — trackers differ only by ±1 rounding and an octave offset.
- **Static notes are PURE under a recovered table:** 83% of voiced frames have residual exactly 0
  (~20 table entries/voice); the remaining ~17% is genuine modulation (vibrato/slide). A fixed
  universal grid instead dumps a deterministic per-note offset into the residual (62% static-offset,
  only 2% within ±1) — why recovery, not quantization, is the right mechanism.
- **The chorus guardrail (load-bearing):** inter-voice detune IS musical content (Cauldron II voices
  1–2: median +12 cents, 90% in 1–60c). Tables/tuning must be **per-voice**; never normalize voices
  to a single tuning (the pitch analogue of the Facemorph waveform guardrail).
- **Transfer = two relative encodings:** effects in CENTS relative to the note (tuning-invariant
  gestures); base pitch as INTERVALS Δnote (transposition-invariant). Absolute onset pitch stays
  high-entropy ≈0 next-token — intervals fix within-melody transfer, not the absolute anchor; score
  onsets distributionally/by audition, not argmax.
- **Tracker pitch recovery validated exact:** SWM/defMON/Hubbard recover bit-exact, and the recovered
  note index == the tracker's OWN table index on sustained frames (40/40 SWM voice-traces 100%);
  committed as tokens `tests/test_tracker_pitch_recovery.py`.
