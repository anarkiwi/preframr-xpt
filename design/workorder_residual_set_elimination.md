# Work order: drive residual raw SETs to zero (mechanism-complete)

**Status:** Work order, scoped 2026-06-03. Principle (non-negotiable): a SID write stream is
deterministic driver output — **every raw `SET` is an unmodeled driver mechanism and an unlearnable
fatal parse error.** Target = **0 residual SETs**, every corner case mapped to a named mechanism and a
consistent macro. "Unique tail" / "irreducible" are banned — uniqueness only means the generator is
unidentified.

## Method
`register_state(df)` (decoded per-frame ground truth) used to classify every residual SET by the shape
of its generator + the existing-pass precondition it failed (`scratch/tmp/residual_mechanism.py`).
Sample: 4 player-diverse non-digi tunes (Camerock=JCH NewPlayer, Commando + Auf_Wiedersehen_Monty=
Hubbard, Advanced.1=DRAX), `max_perm=1`. 1775 residual SETs; **100% named** (12 RARE = 0.7%, each with
a mechanism). NB sample is small — run corpus-wide before sizing, but the taxonomy is the point.

## Census (aggregate, 1775 residual SETs)
The residual is **~85% CTRL, ~10% envelope, ~1.5% FREQ.** FREQ is **entirely startup** (not vibrato —
my earlier assumption was wrong; vibrato did not register as a bucket here).

| % | mechanism | what it is |
|---|---|---|
| 41.0 | CTRL/gate_off_release | gate bit cleared = note-OFF/release |
| 14.6 | CTRL/periodic_table(P=2) | ctrl 2-state per-frame toggle (waveform/gate LFO) |
| 11.7 | CTRL/step_hold | waveform/gate state set & held |
| 9.6 | STATE_CODEBOOK/CTRL/recurring | recurring ctrl state (rec up to 12k) — waveform-table |
| 5.1 | STARTUP/CTRL/step_hold | ctrl state during cold-start |
| 4.3 | ENVELOPE/periodic_table(P=2) | AD/SR 2-state toggle (tremolo / hard-restart) |
| 4.0 | CTRL/oscillation | ctrl >2-state oscillation |
| 2.1 | CTRL/gate_on_trigger | gate set = note-ON not note-ified |
| ~12 (sum) | STARTUP/* | first frames before detectors lock; values recur later |
| ~1.5 | ENVELOPE hard_restart/trigger/release/step | per the envelope work order |
| ~1.5 | FREQ (startup + 12 RARE) | startup arp (rec=4..33), not vibrato |
| 0.3 | INIT/one_time_setup (MODE_VOL/RES_FILT) | player init routine |

## Root finding
**The CTRL (waveform/gate) register is essentially unmodeled.** Today only `ctrl_bigram`/`ctrl_triple`
collapse short *identical* runs. But CTRL carries: note on/off (gate), waveform selection, waveform-
table arpeggios, and per-frame waveform toggles — none of which has a proper macro, so ~85% of all
residual SETs are CTRL. (My earlier "irreducible CTRL" call was exactly wrong; recurrence up to 12k
proves it's deterministic table-driven state.)

## Mechanisms → proposed macros (prioritized by volume)

### 1. Note-OFF / gate-release event (41%) — biggest single fix
The model has note ONSETS (skeleton/SKEL) but no note-OFF; every gate-clear leaks. **Fix:** model note
**duration** (note = onset frame + duration → gate-off implied at onset+duration), or a single
`GATE_RELEASE`/`NOTE_OFF` token. One consistent event per note-off. Pairs with the skeleton note model.

### 2. CTRL waveform-state codebook (~27%: step_hold + STATE_CODEBOOK + startup step_hold)
Waveform/gate selection set & held, drawn from a tiny recurring alphabet (waveform+gate combos), rec to
12k — a per-instrument **waveform table**. **Fix:** a CTRL/waveform codebook (WavetablePass-style but on
the ctrl register), `DEF`+`REF`. Captures the player's wave-table program as a reusable atom.

### 3. CTRL per-frame oscillation (~19%: periodic_table P=2 + oscillation + startup)
2-state (A,B,A,B) and >2-state per-frame ctrl toggles (buzz / PWM-via-waveform / retrigger LFO).
`ctrl_bigram`/`triple` only catch *constant* runs, not alternation. **Fix:** a `CTRL_OSC` macro
(period + states), the ctrl analogue of the proposed freq/PW sweep-osc — one parametric atom per run.

### 4. Envelope trigger/release/oscillation (~10%)
Covered by `workorder_envelope_trigger_bundling.md` (PatchPass hard-restart multiload + first-occurrence
+ note-trigger bundling), PLUS **ENVELOPE/periodic_table(P=2)** = tremolo/envelope LFO → same osc family
as #3 (an envelope-osc macro). `release_update_pass`/`lonely_catch_all` are also OFF in this arm.

### 5. Cold-start / startup priming (~12%)
First frames before the note/wavetable/ctrl/codebook detectors lock onto the steady-state pattern; the
SAME values recur later (rec 4..33), so it's the same tables leaking only at the head. **Fix:** let the
codebook/pattern passes emit `DEF`-on-first-occurrence (don't require ≥MINREP *prior* repetitions before
the first emission), or prime detectors from a one-pass scan. Drains FREQ (startup arp) too.

### 6. Tune INIT preamble (~0.3%)
Player init routine writes master volume / filter resonance / initial ADSR once (rec=1). **Fix:** an
`INIT`/preamble bundle = the tune's initial register state as a header atom (not raw SETs).

## Acceptance criteria
- `residual SET count == 0` corpus-wide (digi-excluded) with the proposed macros enabled — verified by
  re-running `residual_mechanism.py` (must report 0) and the parse audit.
- Byte-exact: every new macro register-exact via `arbitrate(validate=True)`, gated on `cb_div_audit`.
- No new "RARE"/unclassified bucket may survive; any residual that does is a new work item, not a result.

## Before building
- **Re-run the census corpus-wide** (hundreds of tunes, digi-excluded, multiple players) to confirm the
  CTRL-dominance and size each mechanism — the 4-tune sample sets the taxonomy, not the weights.
- **Reconcile each mechanism with the player** (JCH NewPlayer wave/pulse/filter tables; Hubbard) and
  `sid_driver_ornament_reference.md` so the macros match real driver primitives, not curve-fits.
