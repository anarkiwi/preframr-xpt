# Voice encoding reference — how the 3 SID voices are carried in the token stream

**Status:** Pointer (re-anchored 2026-06-12 to the v3 event model). In the **event stream** voices
are explicit `VOICE_*` tags inside each frame group (`<DT> ( <VOICE_v> <events>* )*`, voices
ascending; per-voice `TUNING`/`NOTE_TABLE`/`TICK` stream headers) — see the
[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens) stream grammar. The
**parse-domain** mechanics (FRAME-val base-4 voice-order packing, the `VOICE` reg −126 delimiter
trap, `VALID_VOICEORDERS`) still exist for audits/constrained-decode and are documented in the same
README's parse-domain section; they are not the trained encoding.

**Implications for modeling (xpt-internal, unchanged in substance):**

- Melodic onsets remain **multiplexed across voices** — within a frame group, consecutive events
  belong to different voices, so the next same-voice note is a long-range, position-unstable
  dependency. This is the structural cost the
  [`lane_demux_hypothesis.md`](../encoding/lane_demux_hypothesis.md) targets (open; evidence-gated).
- Voice identity is structural (P4) and explicit in v3; the de-mux question is about *ordering*,
  not attribution.
