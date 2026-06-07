**Status:** Reference

# SID render fidelity contract

The audio-behaviour facts below are proven reproducibly against pyresidfp in
**preframr-audio** unit tests — that is the single source of truth. This doc states
the contract and cites the test for each fact; it does not restate a fact without a
citation, and other docs should cite the tests too rather than paraphrase them.

## Render timing model

The renderer clocks ~`_MIN_DIFF` (32 PAL cycles) **after each register write**; a
frame's writes take effect in sequence, not simultaneously. The FRAME marker holds
the remainder so the frame totals one IRQ period. (`preframr_audio/audio_driver.py`
`df_to_packets`/`ResidWorker`; both audibility test files clock between writes in
`_frame`.) An idealised clock-once-per-frame model hides intra-frame order/timing and
must not be used to judge audibility.

## Within-frame write order — PRESERVE, do not canonicalise

Within a voice, the SID-visible writes must appear in the **same order they were
input** (the raw dump's clock order). Voice blocks are grouped v0, v1, v2; filter/
volume (21–24) last. Do **NOT** sort a voice's writes into ascending register order:
the gate (CTRL) must keep its position relative to the freq it gates and the AD/SR
writes around it. `reglogparser._norm_pr_order` must sort within a voice by **emit
order**, never by register number.

**Why order is load-bearing — the ADSR bug.** The SID envelope rate is driven by a
15-bit prescaler the chip compares for **equality (not ≥)** against the rate value.
Switching a voice's A/D/R from a higher to a **lower** rate value — or a gate
transition that does so — can leave the prescaler already past the new value, freezing
the envelope until the prescaler overflows: up to **32768 cycles ≈ 1.7 PAL frames**.
So each AD/SR/gate write's effect depends on the prior value AND the write order;
reordering AD/SR relative to the gate mistriggers (or fails to trigger) the bug.
Proven against pyresidfp in `test_register_canonicalization::test_adsr_bug_attack_depends_on_prior_envelope_state`
(a re-gated note's attack changes by Δ5707 depending on the prior note's hold length);
external refs: codebase64 "classic hard-restart and about ADSR", c64-wiki "ADSR-Bug",
CSDb release 270701 (rate-prescale periods 0:9 … F:31251; max delay 32768 cyc ≈ 32.8ms).

**Players exploit this on purpose ("sexy-start").** The audible interleaved AD/SR-before-
gate order is usually *intentional*: writing AD+SR **before** the GATE with a big Release
(>3–4) + small Attack (0–1) makes the rate-counter miss the Attack compare → a deliberate
~32ms attack delay (1st frame silent, often with CTRL `$09` = gate-off+TEST set). The
conservative anti-bug order is AD → GATE → SR. Both are deliberate and audible, so the
encoder must preserve whichever the dump used. (A cycle-exact "Bottle" HR resets the
rate-counter in <1ms via envelope wraparound — a sub-frame variant.)

**Two distinct "hard restart" mechanisms — do not conflate (though they often co-occur,
e.g. the `$09` sexy-start frame):**
- **ADSR / envelope hard restart (the *classic* one):** gate-based, no TEST bit. Gate
  off + re-load AD/SR ~2 frames before the note so the ADSR-bug window (1.7 frames)
  elapses and the envelope attacks from a known state. Drivers often write **AD, SR,
  then Control(gate)** — envelope before gate. Its onset frames are gate-off/**release**,
  where freq IS audible.
- **TEST-bit oscillator reset:** sets CTRL bit 3 to hold the oscillator accumulator at
  0 (consistent waveform phase). Only on a TEST-bit frame is freq don't-care. SID Wizard
  uses both; they are not the same thing.

## Proven facts → tests (preframr-audio)

| Fact | Test |
|---|---|
| **The ADSR bug** (prescaler equality-compare): a re-gated note's attack depends on the prior envelope state → AD/SR/gate write order AND values are load-bearing (the root reason) | `test_register_canonicalization::test_adsr_bug_attack_depends_on_prior_envelope_state` |
| Intra-frame write ORDER is audible → preserve input order (a gate written before the freq it gates attacks at the wrong pitch) | `test_register_canonicalization::test_intra_frame_write_order_is_audible` |
| Interleaved ADSR/CTRL order is audible (gate between two distinct SR writes) — reordering to register-ascending changes the audio (real HVSC frame) → never reg-sort within a voice | `test_register_canonicalization::test_interleaved_adsr_ctrl_order_is_audible` |
| Multiple CTRL writes in a frame each take effect (TEST/un-TEST, gate toggles) → keep in time order | `test_register_canonicalization::test_intra_frame_gate_toggles_take_effect` |
| Test-bit-frame **PW is audible** (pulse threshold, pre-TEST window) → NOT discardable; test-bit-frame **freq is inaudible** → discardable | `test_register_canonicalization::test_test_bit_frame_pw_is_audible_but_freq_is_not` |
| Waveform bits on a test frame are audible (held DC level) | `test_register_canonicalization::test_waveform_bits_during_test_are_NOT_dont_care` |
| Ring-mod with an idle source silences the voice (not a no-op) | `test_register_canonicalization::test_ring_with_a_silent_source_silences_not_noop` |
| Hard-sync with an idle source IS a no-op (canonicalisable off) | `test_register_canonicalization::test_sync_with_a_non_oscillating_source_is_a_noop` |
| Release is not instant; a freq change during release is audible (so a classic-HR/release-window freq is NOT discardable) | `test_freq_write_audibility::test_release_zero_is_not_instant_and_keeps_sounding`, `::test_freq_change_during_release_is_audible` |
| Noise-waveform freq is audible (noise pitch/colour) | `test_freq_write_audibility::test_noise_freq_is_audible` |
| Combined-waveform freq is audible | `test_freq_write_audibility::test_combined_waveform_freq_is_audible` |
| TEST-bit-frame freq is absorbable only to a NEARBY value (a wild multi-octave triangle jump leaks via the pre-TEST window) | `test_freq_write_audibility::test_freq_during_test_bit_is_inaudible`, `::test_real_tune_test_bit_hr_freq_absorbable_to_a_nearby_value` |
| Noise+waveform LFSR lock decays even with gate+sustain held | `test_freq_write_audibility::test_noise_combined_with_pulse_lfsr_lock_decays` |

## Discardable vs must-preserve

- **Discardable / canonicalisable:** same-value writes (`squeeze_changes`); freq/PW/
  filter-cutoff VALUE settling+merge (`combine_reg`, the sanctioned quantization);
  **TEST-bit**-frame freq absorbed to the adjacent note's pitch; hard-sync off when the
  source is idle.
- **Must preserve exactly:** the per-voice write ORDER (freq/PW/CTRL/AD/SR in input
  order — never reg-sorted), CTRL/AD/SR values and frame membership; test-bit-frame PW
  and waveform bits; noise/release/combined freq; nominal intra-frame timing (every
  intra-frame write carries `_MIN_DIFF` — a frame-scale `diff` drives the FRAME budget
  negative and drops samples).

## Fidelity oracle

`register_state` (settled end-of-frame per-register snapshot) is order- and
timing-blind: a transform can be `register_state`-byte-exact yet render broken audio
(e.g. a codebook pass that emits a frame-scale `diff` → negative FRAME budget →
dropped samples; or a reg-sort that scrambles interleaved ADSR/CTRL). It is necessary
but not sufficient. The lossless gate operates at the **register level, not audio**:
per frame, per voice, assert (1) the CTRL/AD/SR ordered write sequence is byte-exact vs
the raw dump **in the dump's order** (CTRL/AD/SR may repeat, kept in input order); (2)
same frame number; (3) freq/PW/filter quantization-relaxed; (4) every intra-frame
decoded write carries nominal `_MIN_DIFF`. One canonical render core
(`prepare_df_for_audio` → `df_to_packets` → `ResidWorker`); no divergent SID renderer.

## Tokenizer constraint

Single-speed assumption: at most one freq and one PW write per voice per frame
(`combine_reg` settles them). Multiple AD/SR/CTRL writes per voice per frame DO occur
(~17% of single-speed tunes have interleaved ADSR/CTRL) and are **preserved in input
order**, not rejected or merged — the ADSR bug makes their order audible.
