# Voice encoding reference — how the 3 SID voices are carried in the token stream

**Status:** Pointer (re-anchored to the BACC step/tracker codec). In the BACC codec voices are
**de-muxed by construction**: recovery produces per-voice tracker-row streams, so each voice is its
own coherent sequence rather than interleaved into a shared next-token position — see the
[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens). (Historical: the v3 event
stream carried voices as explicit `VOICE_*` tags inside frame groups, with per-voice
`TUNING`/`NOTE_TABLE`/`TICK` headers and base-4 voice-order packing in the parse domain — retired.)

**Implications for modeling (xpt-internal, unchanged in substance):**

- Melodic onsets are **no longer multiplexed across voices** — the BACC codec emits per-voice row
  streams, so the next same-voice note is local, not a long-range position-unstable dependency. This
  is exactly what the [`lane_demux_hypothesis.md`](../landed/lane_demux_hypothesis.md) targeted; the
  landed step/tracker codec de-muxes voices by construction. (Historical: v3 frame groups interleaved
  consecutive events across voices.)
- Voice identity is structural (P4); with de-mux by construction the attribution is implicit in the
  per-voice stream.
