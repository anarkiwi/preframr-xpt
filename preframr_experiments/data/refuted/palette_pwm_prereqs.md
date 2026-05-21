# `palette_pwm` prereqs (a) + (b) — REFUTED (2026-05-16)

**Status:** prereqs refuted at prototype probe. Full `palette_pwm`
spec doesn't lose this entry's outcome — its own Layer-0 finding
(99.99% theoretical replay) still stands — but the prereq path
through `InstrumentPass PW window` + `PerRegBurstPass PW skip`
no longer makes sense to pursue.

## Hypothesis

Two adjacent encoder tweaks were proposed as `palette_pwm`
prereqs:

(a) **InstrumentPass PW capture window** — extend
`InstrumentProgramPass`'s capture window to include the PW
register so PWM tables become first-class inside instrument slots
rather than spilling to literal `PWM_OP` rows.

(b) **PerRegBurstPass PW skip** — suppress `PWM_OP` burst
emissions during the InstrumentPass capture window so the PW
writes aren't double-encoded once into the instrument slot and
again as a burst.

Together (a) + (b) were the foundation `palette_pwm`'s full
encoding stacked on top of — without them, a `palette_pwm` slot
would need its own PW-handling logic.

## Refutation

Prototype probe `597822f`
(`design/palette_pwm_prereq_ab_prototype_verdict.md`) implemented
both prereqs behind throwaway flags and measured the resulting
alphabet + atoms/song deltas on mini train.

- (a) alone: alphabet ~null, atoms/song +0.7% (cost without
  payback).
- (b) alone: alphabet -0.2%, atoms/song +0.4% (marginal).
- (a) + (b) together: alphabet -0.3%, atoms/song +0.9% (still
  net-positive cost).

The prereqs add complexity but don't even break even on the
alphabet-shrink-only axis they were supposed to enable.

A separate root-cause investigation
(`design/parser_pw_drop_investigation.md`, `6643aba`) showed the
99.6% PW-drop signal that motivated the `palette_pwm` prereq
framing was a **measurement artifact** of `_simplify_pcm` zeroing
pulse-off PW writes (audio-correct: SID ignores PW when pulse
waveform isn't selected) plus `_squeeze_changes` deduping the
zeros. The 0.5177 L1 PW envelope mismatch is on **inaudible**
register writes; the audio output is correct.

The PWM table doesn't need first-class capture for fidelity. The
prereq stack was solving a register-level metric, not an audio-
level problem.

## Evidence

- `integration_tests/design/palette_pwm_prereq_ab_prototype_verdict.md`
  — full prototype results.
- `integration_tests/design/parser_pw_drop_investigation.md` — root
  cause of the original PW-drop motivation.
- `6580026` — gate re-cal recommendation: switch PW fidelity
  metric to `audio_fidelity.compare_renders` or mask pulse-off
  frames in the L1 calc.

## Status of the parent `palette_pwm` spec

`palette_pwm`'s OWN Layer-0 (99.99% theoretical replay) stands
independent of the prereq stack. If the spec is ever revisited it
needs:

- A standalone PWM-table encoding path that doesn't depend on
  InstrumentPass capture-window extension.
- OR a redefinition of the win condition in audio terms, not
  register terms.

The fast-fail threshold (alphabet shrink null OR atoms/song
blowup) is empirically met by the prereqs themselves; the parent
spec would need to clear it again from scratch.

## Do not revisit without

- A PWM-table encoding scheme that doesn't rely on
  InstrumentPass capture-window extension.
- AND a fidelity gate framed in `audio_fidelity.compare_renders`
  terms, not register-byte L1.
