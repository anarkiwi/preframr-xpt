# Generator-MDL representation — historical record

**Status:** LANDED (tokens 0.45.0–0.46.x, deployed default; the per-pass macro zoo deleted with it)
→ **SUPERSEDED 2026-06-12 by the v3 event model** (tokens 0.47.0, `preframr_tokens/events/` — see the
[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens)). Kept as the record of the
first-principles decomposition whose ideas the event model absorbed: the per-tune/per-voice pitch
table → `NOTE_TABLE`/`TUNING` headers; semitone-grid intervals → the `NI_*` note-index lane;
HOLD/ACCUM/SWEEP/TABLE generator shapes → `*_STEP`/`*_RAMP` events with `SHAPE_POLY`/`SHAPE_PERIOD`;
self-verifying encode → `encode(verify=True)`. What it did NOT carry into v3: the cross-tune DEF→REF
bank (v3 has **no** DEF/REF ids — BPE merges are the only dictionary) and the macro-pass machinery.

## First principle

A SID driver is a 50/60 Hz loop that, per frame, either **looks up an indexed table** or
**accumulates a delta** into each register. So the shortest lossless description of a tune is: factor
the per-frame register stream into the *generators* a 6502 driver actually runs, reuse them by id.
Objective: minimum description length subject to **exact** reconstruction. Hard rules: no singletons,
no RESID/escape hatch, one uniform model over all channels, hypothesize from the driver doc then
prove on data.

## The representation (as landed)

Channelize (`register_state` → 3 voices × {freq, pw, ctrl, ad, sr} + globals) → per-tune-calibrated
semitone LUT on freq (`f → (note, residual)`, lossless bijection; note-relative keying measured
**−64.4% distinct shapes** on 530 tunes) → generators per channel: **HOLD** (len) / **ACCUM** (Δ,
len) / **SWEEP** (bounded reversing zigzag) / **TABLE** (periodic cycle: arp, wavetable, sine
vibrato) + END (dump truncation) → cross-tune DEF→REF bank. The decisive rule: **every primitive is
accepted only for the longest prefix its OWN decoder reproduces exactly** — lossless by construction.

Guardrails proven along the way (durable, inherited by v3):
- **Never classify pitch/percussion by the waveform bit** (Facemorph: noise bursts accent *pitched*
  notes; pulse carries percussion). The pitch split never consults ctrl.
- **Per-tune (then per-voice) tuning calibration is mandatory** — a fixed A440 LUT lands 2.6% of
  frames at residual 0; calibrated, `|resid|≤1` hits 60.7%.

## Evidence (all digi-excluded; the numbers that earned the landing)

| test | result |
|---|---|
| Scale (1580 tunes) | **100% byte-exact**; 0 interior / 0 bare events (all EVENTs = final-row truncation) |
| Hard engines (Baggis/JCH, Camerock, Commando, SoundMonitor, System6581) | 5/5 lossless, 0 bare |
| SID-Wizard 1.94 full suite (91 modules, verified player) | 91/91 byte-exact, 131 distinct FX codes covered |
| defMON (byte-verified player, 9 fixtures + 46 corpus tunes) | 9/9 + 46/46 byte-exact |

## Why it was superseded

The static learnability triage (window mode, seq_len 8192, 30 tunes) returned a **conditional
NO-GO**: vs the atomic baseline the generator alphabet was 3.7× larger, per-frame h∞ tied, and
in-block induction-copy slightly LOWER (0.916 vs 0.930) — root cause: **exact residuals embedded in
the `GEN_TABLE` key** fragmented near-identical gestures into distinct atoms. Rather than patch the
key, the encoding was redesigned from the triage's prescriptions (small fixed alphabet, digits as
atoms, intervals, no codebook keys to fragment) → the v3 event model. See
[`learnability_token_ordering_theory.md`](../references/learnability_token_ordering_theory.md).

The melody layer-2/3 stack this doc gated (interval skeleton, lane de-mux) resolved as: layer 2
**absorbed** by v3's `NI_*` interval lane; layer 3 remains the open
[`lane_demux_hypothesis.md`](../encoding/lane_demux_hypothesis.md).
