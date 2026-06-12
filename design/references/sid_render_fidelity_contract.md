**Status:** Reference (updated 2026-06-11: the complete envelope mechanism is now pinned as a
canonical reference suite in preframr-audio, and the encoding-side contract moved from
"preserve byte order" to the v3 canonical form — see below).

# SID render fidelity contract

The audio-behaviour facts below are proven reproducibly against pyresidfp in
**preframr-audio** unit tests — that is the single source of truth. This doc states
the contract and cites the test for each fact; it does not restate a fact without a
citation, and other docs should cite the tests too rather than paraphrase them.

## Render timing model

The renderer clocks ~`_MIN_DIFF` (32 PAL cycles) **after each register write**; a
frame's writes take effect in sequence, not simultaneously. The FRAME marker holds
the remainder so the frame totals one IRQ period. (`preframr_audio/audio_driver.py`
`df_to_packets`/`ResidWorker`; the audibility/reference test files clock between writes in
`_frame`.) An idealised clock-once-per-frame model hides intra-frame order/timing and
must not be used to judge audibility — **measured consequence (2026-06-11): collapsed-timing
A/Bs showed write placements equivalent that real per-write timing proves audibly different**
(the gate=0 inter-write dwell is where ADSR-bug stalls arm). Never judge a write-placement or
ordering question without per-write clocking.

## The envelope mechanism, exactly (the 2026-06-11 canonical reference)

Three test files pin the complete gate/ADSR mechanism; envelope questions are answered from
these, not by new ad-hoc probes:

- **`test_gate_adsr_reference.py` — the ADSR (envelope delay) bug itself.** The rate prescaler
  is a free-running 15-bit LFSR compared for **EQUALITY** against the active nibble's rate
  period (A during attack, D during decay/sustain — even clamped, R whenever gate=0 — even with
  the envelope dead at 0), and **every envelope step resets it**. Therefore: the bug is
  **compare-change associated, NOT gate associated** — a write to the active nibble freezes the
  running phase ~33ms with no gate edge anywhere (attack, decay and release alike), and it is
  one-directional (raising the active compare never stalls). Gate edges stall by *swapping which
  nibble is active* (attack stall armed by the prior gate=0 epoch's compare — hard restart =
  forcing it small; release stall armed by D's compare). Internal handoffs (FF→decay, clamp
  arrival) can never stall (fresh LFSR reset). **Gate-edge position is content at single-write
  granularity**: even a same-value write before a CTRL edge write shifts the edge one
  inter-write slot and can flip a later note's attack stall outright. Gate 1→1 is not a
  retrigger; a re-attack resumes from the current counter; a short gate pulse releases a partial
  attack.
- **`test_adsr_write_liveness_matrix.py` — what is relocatable when.** The (phase × nibble)
  matrix: a nibble write is relocatable within an interval iff it is neither **value-live**
  (shaping the counter: A in attack, D in decay, R in release, S as clamp) nor **phase-live**
  (the active LFSR compare: D even in clamped sustain, R even in the dead zone). The sustain
  hold is an equality too (`counter == S*0x11`): **raising S mid-note un-matches it and the
  counter falls to zero — raising sustain kills the note.**
- **`test_release_write_position.py` — the R placement rules.** A mid-note R change relocates
  freely across its own note's gate=1 span (foldable to note-on emitted gate-then-SR) but never
  into gate=0 time: a live prior tail re-rates immediately; a provably-dead gap (ENV3=00) still
  stalls the *next* attack via the re-comparated LFSR phase.

Methodology (in the files; required for any new measurement): write-count-matched variants
(same-value pads keep chunk boundaries and write offsets identical), ENV3 on voice 3 for
state-level verdicts, equivalence floor calibrated to the measured ±8 resampler
nondeterminism, per-write clocking always.

## Within-frame write order — what the v3 encoding preserves vs canonicalizes

The old rule here was "PRESERVE, do not canonicalise." Measurement refined it: order is
load-bearing exactly where the mechanism above says it is, and the v3 canonical form
(`gen2 stream.canonical_writes`) re-orders only where measured chip-inert/masked:

- **Preserved**: all CTRL/AD/SR *change* activity as ordered events at sub-frame resolution
  (interleaved ADSR/CTRL order is audible — `test_interleaved_adsr_ctrl_order_is_audible`,
  ~17% of single-speed tunes); **the side of the gate edge each folded onset AD/SR write was
  on** (recorded per-nibble in NOTE_ON: driver conventions SPLIT — grid_runner/commando write
  gate-then-AD/SR, camerock/baggis the reverse — and a write crossing the edge flips ADSR-bug
  stall states value-dependently, so no fixed canonical order is faithful; measured 2026-06-11,
  the fixed-order variant was audibly wrong on grid_runner); HR prep writes stay on the gate=0
  side of the off edge.
- **Canonicalized (measured-faithful)**: settled freq/PW first in the voice group (sub-frame
  freq/PW/global transients are 0.13% of in-scope writes, −27 dB masked under coincident
  content); globals settled last, reg-ascending; same-value rewrites dropped (chip latch
  no-ops — `test_sid_same_value_writes`); gate-offs derived at onset+duration (no NOTE OFF);
  AD-before-SR within a gate-edge side.

**Players exploit the bug on purpose ("sexy-start").** Writing AD+SR **before** the GATE with a
big Release + small Attack deliberately arms the ~32ms attack delay (1st frame silent, often
CTRL `$09` = gate+TEST). This is *why* the gate-edge side is recorded content, not a
normalizable convention — the v3 encoding preserves whichever the dump used, by flag.

**Two distinct "hard restart" mechanisms — do not conflate (they often co-occur, e.g. the `$09`
frame):** the **ADSR/envelope hard restart** (gate-based: gate off + reload AD/SR ~2 frames
before the note so the prescaler epoch runs at a small compare — the arming rule above) and the
**TEST-bit oscillator reset** (CTRL bit 3 holds the accumulator at 0 for waveform phase). SID
Wizard uses both.

## Proven facts → tests (preframr-audio)

| Fact | Test |
|---|---|
| **The ADSR bug, complete mechanism**: equality compare + step-resets-LFSR ⇒ compare-change associated, write-only freezes in all phases, one-directional, handoffs never stall, edge position is content | `test_gate_adsr_reference.py` (13 tests; see section above) |
| **Write-relocation liveness matrix** (phase × nibble); raising sustain kills the note | `test_adsr_write_liveness_matrix.py` |
| **R-nibble placement rules** (foldable across gate=1; never across gate=0 — live tail re-rate / dead-gap stall) | `test_release_write_position.py` |
| Prior-state dependence of a re-gated attack (the classic two-note demo) | `test_register_canonicalization::test_adsr_bug_attack_depends_on_prior_envelope_state` |
| Intra-frame write ORDER is audible (gate before the freq it gates attacks at the wrong pitch) | `test_register_canonicalization::test_intra_frame_write_order_is_audible` |
| Interleaved ADSR/CTRL order is audible (real HVSC frame) → never reg-sort within a voice | `test_register_canonicalization::test_interleaved_adsr_ctrl_order_is_audible` |
| Multiple CTRL writes in a frame each take effect (TEST/un-TEST, gate toggles) | `test_register_canonicalization::test_intra_frame_gate_toggles_take_effect` |
| Test-bit-frame **PW is audible**, **freq is not** (absorbable to a NEARBY value only) | `test_register_canonicalization::test_test_bit_frame_pw_is_audible_but_freq_is_not`, `test_freq_write_audibility::test_freq_during_test_bit_is_inaudible`, `::test_real_tune_test_bit_hr_freq_absorbable_to_a_nearby_value` |
| Waveform bits on a test frame are audible (held DC level) | `test_register_canonicalization::test_waveform_bits_during_test_are_NOT_dont_care` |
| Ring-mod with an idle source silences (not a no-op); hard-sync with an idle source IS a no-op | `test_register_canonicalization::test_ring_with_a_silent_source_silences_not_noop`, `::test_sync_with_a_non_oscillating_source_is_a_noop` |
| Release is not instant; freq during release is audible; noise freq is audible (pitch/colour); combined-waveform freq is audible | `test_freq_write_audibility` (resp. tests) |
| Same-value rewrites are audibly bounded (chip latch no-ops) — the license for dropping them | `test_sid_same_value_writes` |
| Sustain nibble = held amplitude (per-voice volume); the perceptual harness's BAD floor | `test_sid_feature_behaviors::test_sustain_level_sets_held_amplitude_and_gate_off_releases`, `fidelity.py` calibration pairs |

## Fidelity oracle (v3)

The encoding-side oracle is **`stream.canonical_writes(dump)`** (gen2
`preframr_tokens/events/stream.py`): the dump's audibly-faithful canonical form — an exact
intra-frame permutation + derivation of the dump's writes (zero drops), per the
preserved/canonicalized split above. `stream.encode` self-verifies `decode == canonical_writes`
on every encode. Settled `register_state` is order- and timing-blind — necessary, never
sufficient; it survives only as an internal settling primitive. The old per-pass
`parse_audit`/`cb_div_audit` gates belonged to the retired substrate
(see [`verification_and_audits.md`](verification_and_audits.md) for the v3 tool map).

## Tokenizer constraint

Scope: single-speed, non-digi tunes (at most one *settled* freq/PW value per voice per frame;
multi-speed ~5% and digi ~3% are excluded — corpus globs must filter). Multiple AD/SR/CTRL
writes per voice per frame DO occur (~17% of single-speed tunes) and are preserved as ordered
events — the mechanism above makes their order audible.
