# Implementation plan: eliminate residual SETs (CTRL-first)

> **RELOCATED:** the canonical, self-contained executable spec now lives in the **preframr-tokens** repo
> at `IMPLEMENT_residual_set_elimination.md` (untracked), alongside the verification tool
> `residual_mechanism.py` (untracked). Hand that to an implementing agent. This file is the
> preframr-xpt research record only; keep the two in sync if edited.

**Status:** Implementation draft 2026-06-03, to iterate. Companion to
`workorder_residual_set_elimination.md` (the mechanism census) — this is the *how*. Census says the
residual is ~85% CTRL, ~10% envelope, ~1.5% FREQ(startup). So this plan is **CTRL-register-first**.

## The "add a macro" recipe (every macro follows this in the codebase)
A macro is one declared unit (see `macro_abstraction_consolidation.md`):
1. **op id(s)** in `stfconstants.py` (+ subreg constants); register in `macros/op_contracts.py`
   (`OP_PRODUCER`, reference-op tables) and `block_refire_pass_names()` if reference-emitting.
2. **encoder** = a `MacroPass` (`macros/*_pass.py`): mine the pattern, build `Claim`s, call
   `arbitrate(df, claims, validate=True)` (byte-exact; a non-exact claim stays literal). `GATE_FLAGS`
   declares the arg flag (auto-collected by `flag_registry.macro_flag_names`).
3. **decoder** = a `MacroDecoder.expand(row, state)` in `macros/decoders.py`, registered in `DECODERS`;
   drains via `state.pending_set_writes`/etc. as needed.
4. **flag** added to `REGISTERED_MACROS` (tokenizer_config) once shipping.
5. **validation**: full suite (byte-exact + per-frame fidelity + `test_sid_frame_diff`) + corpus
   `cb_div_audit`; re-run `residual_mechanism.py` — the macro's mechanism bucket must go to 0.

Templates to copy: **codebook** macros → `WavetablePass`/`StampPass` (DEF/STEP/END/REF); **parametric
run** macros → `SweepPass` (START/DELTA/LEN/PERIOD + per-frame drain); **note model** → `SkeletonPass`.

## Order (by volume × independence; each ships + validates standalone)

### PR1 — CTRL_OSC: per-frame ctrl oscillation (~19%, self-contained, lowest risk)
Mirror `SweepPass` on the CTRL register. **Mine:** per voice, a run of ≥`MIN_LEN` consecutive frames
whose ctrl value cycles with period P≤`CTRL_OSC_MAXP` (covers the dominant P=2 toggle and >2-state).
**Atom** `CTRL_OSC_OP`: subregs `STATES` (the P distinct ctrl bytes, as a short list like SWEEP's
delta-run), `PERIOD`, `LEN`. **Decode:** queue `state.pending_set_writes[ctrl_reg]` with
`states[k % P]` for `LEN` frames, draining one/frame (reuse the SWEEP drain path). Flag `ctrl_osc`.
Byte-exact via one `Claim`/run. Drains `CTRL/periodic_table(P=2)` + `CTRL/oscillation` + their startup
variants.

### PR2 — CTRL waveform codebook (~27%, mirror WavetablePass on ctrl)
The per-instrument waveform program: a recurring ctrl-value sequence (or held state) per note. **Mine:**
per voice, the ctrl-value run within a note span; intern recurring sequences into a codebook
(`CTRL_WT_DEF`/`STEP`/`END` + `CTRL_WT_REF`), exactly like `WavetablePass` but reading `ctrl_reg`
(`v*7+4`) instead of freq. A held single state is the length-1 case. **Decode:** REF replays the interned
ctrl sequence into `pending_set_writes[ctrl_reg]`. Flag `ctrl_wavetable`. Drains `CTRL/step_hold` +
`STATE_CODEBOOK/CTRL/recurring` (+ startup step_hold). **Dependency:** PR5 (DEF-on-first) for the
startup share.

### PR3 — Note-OFF / duration (41%, biggest; extends the note model)
Highest volume, in the skeleton note model. **Option A (preferred): note duration.** `SkeletonPass`
already finds note onsets (gate-on); pair each with its gate-off frame and carry `duration` in the SKEL
atom; the decoder re-emits the gate-clear ctrl write at `onset+duration`. **Option B: `NOTE_OFF` op** —
a standalone token consuming the gate-clear ctrl write (simpler, less structural, but one extra token
per note). Start with B behind a flag to drain the residual fast and measure learnability, then move to
A. Flag `note_off`. Drains `CTRL/gate_off_release` (+ `gate_on_trigger` if folded into onset). **Reconcile
with the player first** (JCH NewPlayer note/gate timing) per `sid_driver_ornament_reference.md`.

### PR4 — Envelope trigger/release/oscillation (~10%)
Per `workorder_envelope_trigger_bundling.md`: (a) teach `PatchPass` the hard-restart multi-load
(`patch_pass.py:92` `count!=1` bail) and first-occurrence (`PATCH_MINREP`); (b) `ENVELOPE/periodic_table(P=2)`
(tremolo) → reuse the PR1 oscillation machinery on AD/SR regs (an envelope-osc mode). Also: confirm
whether the codebook arm should re-enable `release_update_pass`+`lonely_catch_all` (currently omitted).

### PR5 — Cold-start DEF-on-first (~12%, cross-cutting modification, not a new op)
The startup share is the same tables leaking at the head before detectors lock (values recur later,
rec 4..33). Change the codebook passes (`WavetablePass`/`PatchPass`/PR2 CTRL-WT) to emit a `DEF` on the
**first** occurrence (DEF-on-first, REF thereafter) instead of requiring ≥`MINREP` prior repetitions
before the first emission. Gate behind a flag; verify alphabet/atoms unchanged in steady state. Drains
all `STARTUP/*` buckets (incl the FREQ startup arp).

### PR6 — INIT preamble (~0.3%)
One-time setup writes (master vol `$D418`, filter res, initial ADSR; rec=1, head frames). Emit a single
`INIT` atom = the tune's initial register snapshot, consumed at decode as the seed state. Smallest;
do last. Drains `INIT/one_time_setup/*`.

## Acceptance (per PR and overall)
- Per PR: the targeted `residual_mechanism.py` bucket → 0; full tokens suite green; `cb_div_audit` clean.
- Overall: `residual_mechanism.py` reports **0 residual SETs** corpus-wide (digi-excluded, multi-player);
  the parse audit shows no raw register SET survives. No bucket (incl RARE) may remain non-zero.

## Open items before/while coding
- **Corpus-wide census** to lock per-mechanism weights (4-tune sample sets taxonomy only).
- **Op-id allocation**: pick concrete ids for `CTRL_OSC_OP`, `CTRL_WT_*`, `NOTE_OFF_OP`, `INIT_OP` from
  the free op space; wire `op_contracts`.
- **Player reconciliation** (JCH NewPlayer wave/pulse tables, Hubbard) so PR2/PR3 match real driver
  primitives, not curve-fits.
- **Decode-state**: CTRL drains share `pending_set_writes[ctrl_reg]`; confirm tick-drain ordering vs the
  gate/note-off so PR1/PR2/PR3 compose byte-exact (the arbiter drain-interaction the partition work hit).
