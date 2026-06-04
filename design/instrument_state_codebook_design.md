**Status:** SUPERSEDED (2026-06-04) by [`instrument_program_codebook_design.md`](instrument_program_codebook_design.md)
— which models the instrument as a per-frame *program* (the driver's wave/pulse/filter-table walk), not
the *static* `(waveform,AD,SR)` state below, and collapses the full ten-pass note-associated cluster. The
localized-fix conclusion (§8/§9) was correct for the byte-exact bug; the program collapse is the
follow-through on "too much complexity vs the drivers."

**Status (original):** AMENDED / mostly WITHDRAWN (2026-06-04). An empirical reject-claim audit (below, §8) shows the
consistency failures are a **localized id-collision in the `ctrl_wt` family** (the added nibble pass
reusing `ctrl_wavetable`'s ids), NOT a modeling failure, and NOT shared by the other codebooks
(`patch`/`stamp`/`wavetable` have zero rejections; no cross-voice divergences exist). So the **correctness
fix is a localized disjoint-id-allocation change, not this redesign.** The per-voice instrument codebook
(below) remains a valid *learnability/consolidation* option but is NOT required for residual==0 — keep it
on the learnability backlog, gate behind the triage, do not build it for correctness.

## 8. Empirical assessment (2026-06-04): the issue does NOT generalize
Reject-claim audit over 203 non-digi tunes, arbiter instrumented to log every dropped codebook claim by
pass-label + whether the register_state divergence is the cross-voice (±VOICE_REG_SIZE) signature, with
the disjoint-id fix applied to the nibble pass:
- `patch` / `stamp` / `wavetable`: **0 rejected claims.** Cross-voice conflation does not affect them
  (`PatchPass` keys per-voice on freq_reg; stamp/wavetable are note/voice-relative).
- **0 cross-voice divergences corpus-wide.** The earlier "adjacent-voice (reg13/reg11) divergence" was
  the id collision corrupting the table lookup (the clobbered entry belonged to a different reg/voice),
  NOT genuine voice-conflation. `remove_voice_reg` voice reassignment is correct.
- Remaining rejections: `ctrl_osc` 6, `gradient` 2, `onset_inst` 1 — a SEPARATE, small, pre-existing
  byte-exactness tail in the *temporal* passes (oscillation/ramp/held-instrument replays), unrelated to
  the collision class. These are the next targeted fixes (per-pass, not a codebook redesign).

**Conclusion:** the control-register "consistency crisis" is one localized bug — `CtrlWavetableNibblePass`
sharing the `ctrl_wt` id-space with `CtrlWavetablePass` and allocating colliding ids across pipeline
stages. **Fix = principled disjoint id allocation** (§9), not a new/unified codebook. Plus the small
ctrl_osc/gradient/onset_inst tail.

## 9. The actual minimal fix
Replace the proof-of-concept `+ 100000` offset with a principled disjoint id space for the nibble pass.
Two clean options:
1. **Reserve a high id band** for nibble entries in the shared `ctrl_wt` table (e.g. nibble ids start at
   a documented constant `CTRL_WT_NIBBLE_ID_BASE` chosen above any realistic per-tune full-byte DEF
   count) — one-line, no new ops, but high id values (check vocab/alphabet impact for training).
2. **A distinct id sub-namespace** keyed by lane so full-byte and nibble entries never share an id
   (e.g. nibble ids stored/looked-up under a `('nib', id)` table key) — needs the codec to know nib-vs-
   full at REF time, which it can't from the shared op alone, so this implies either a nibble-specific
   REF op or carrying the lane on the REF row.
Lean: option 1 with a vocab check; if vocab bloat is real at training tkvocab, fall back to a nibble-
specific op-trio (the smallest "separate id space" that's vocab-clean). Validate: reject-audit clean +
corpus census `reparse=True` == 0 + 12-SID WAV audition before any default flip.

---
(Original proposal retained below as a learnability backlog item — NOT a correctness requirement.)

# Per-voice instrument-state codebook (unify the control-register macros)

# Per-voice instrument-state codebook (unify the control-register macros)

## 1. Diagnosis: why the control-register codebooks have consistency failures

The residual-SET drain on ctrl/AD/SR kept hitting arbiter-rejected codebook claims (non-byte-exact
replays). Looked at together (not as isolated exceptions), every failure is **one** disease: *multiple
codebook-emitting passes share one id-namespace and one register representation, but run at different
pipeline stages that transform that representation, and they do not coordinate.*

- **Cross-voice conflation.** `CtrlWavetablePass` phase-1 keys on `(canonical_reg, value)`. But
  `voice_canonical_block_order` collapses all three voices' SR onto canonical reg6 (+ a VOICE_REG
  marker). So `(reg6, 204)` groups voice-0/1/2's SR=204 into ONE entry; its replay round-trips only if
  `remove_voice_reg` perfectly re-splits it. Phase-1's docstring asserts "per-voice" — that invariant is
  **false** under voice-canonicalization. (Proven: claiming reg6=204 diverges register_state at reg13 =
  voice-1 SR.)
- **Nibble↔full id collision.** `CtrlWavetableNibblePass` (post-SubregPass) and full-byte
  `CtrlWavetablePass` (inline, pre-SubregPass) write the SAME CTRL_WT table with the SAME ops, allocate
  ids from the same base, and cannot see each other across the stage boundary → same id → one clobbers
  the other (`table[10]` = a nibble entry when a full-byte REF expects it). (Proven on Shadi_Music.8 /
  Incredible_Shrinking_Sphere.16.)

Both are the same flaw: **interning by `(register, value)` in a representation that has already
collapsed the dimensions that actually distinguish entries** — voice (collapsed by block-order) and lane
(full vs nibble, sharing an id space). When a later transform re-expands those dimensions (voice
reassignment, nibble split, lead-frame replace-not-append, per-block parsing) the flat replay does not
invert, and the arbiter (`validate=True`) correctly rejects it. **The non-zero residual is the arbiter
catching composition failures** — a symptom of the modeling mismatch, not the root.

The smell that confirms it: ctrl/AD/SR are currently modelled by ~8 overlapping passes
(`ctrl_wavetable` ×3 phases, the nibble pass, `onset_def`, `ctrl_osc`, `gradient`, `hard_restart`,
`legato`, `ctrl_bigram/triple`, `PatchPass`). No single abstraction owns the register, so each pattern
got a bolt-on; they overlap on the same bytes, share id-spaces, and conflate voices.

## 2. The reality the macros should capture

The SID control/envelope registers are **per-voice, composite, and instrument-shaped**:
- **ctrl** (reg 4/11/18): lo-nibble = gate(0)/sync(1)/ring(2)/test(3) = **event + articulation**;
  hi-nibble = waveform (tri/saw/pulse/noise) = **instrument**.
- **AD** (5/12/19): attack(hi)/decay(lo); **SR** (6/13/20): sustain(hi)/release(lo) = **instrument
  envelope**.

So per voice there is an **instrument** = `(waveform = ctrl-hi, AD, SR)`, loaded per note, plus a **gate
event** (ctrl-lo bit0) and occasional **modulation** (sync/ring/test; sweeps; gradients). The recurrence
that matters is *"voice V replays instrument X"* — at the `(voice, instrument)` level — not "byte 204
recurs somewhere." Note the ctrl lo/hi split is **semantically real** (gate vs waveform); the AD/SR
lo/hi split is not (it's just nibble halves of one envelope value).

## 3. Design

**One pass, one stage, voice-aware, one id space.** Replace `CtrlWavetablePass` + `CtrlWavetableNibblePass`
+ `onset_def` with a single `InstrumentPass` that mines per-voice instrument-field recurrence.

### 3.1 Layer 1 — per-(voice, field) value codebook (guarantees residual == 0)
The CORE fix (consistency + residual==0) is just: **voice-aware keys + single id space + single stage +
define-on-first**, mining the **full** ctrl/AD/SR bytes. Gate toggles (`0x41`/`0x40`) are captured as
recurring full-byte ctrl values, so no ctrl decomposition is *required* for correctness. Decomposing
ctrl into hi=waveform (instrument) / lo=gate (event) is a **learnability refinement** (§3.3), not a
correctness requirement — adopt it only once the core lands and the triage shows it helps.

- **Fields (minimal):** the full ctrl, AD, SR bytes per voice. (Refinement §3.3: ctrl-hi only, leaving
  ctrl-lo gate to the event passes.)
- **Key:** `(voice, field, value)` where **voice is derived from the voice-block context** (the
  `remove_voice_reg` `v`: VOICE_REG cumsum + FRAME sval), NOT from the canonical reg. So reg6-in-voice0
  and reg6-in-voice1 are distinct keys — **no cross-voice conflation**.
- **Allocation:** ONE id counter for the whole pass, in ONE stage → **no cross-stage collision**. Runs
  in the inline loop (has full bytes + voice-block markers), so nothing is nibble-split before it sees
  it; SubregPass then has only ctrl-lo (gate) left to split.
- **Emission:** recurring `(voice, field, value)` → DEF (first use) + REF (reuse); a once-only value →
  a lone **define-on-first** DEF. Every ctrl-hi/AD/SR write is therefore at least a DEF ⇒ **zero raw-SET
  residual** on these regs, by construction.
- **Decode:** the codec emits the canonical reg's field within the voice-block; `remove_voice_reg`
  resolves it to the right voice reg (reg6/13/20). ctrl-hi (waveform) is a hi-nibble (subreg-1) write;
  AD/SR are byte writes. One id table, keyed by id → no clobber.

This is "`ctrl_wavetable` done right": voice-aware keys + single stage + single id space + define-on-first,
which is exactly what removes the two consistency failures and the residual.

### 3.3 Refinement — ctrl waveform/gate decomposition (learnability, optional)
Split ctrl into hi=waveform (instrument, → Layer 1) and lo=gate/sync/ring/test (event, → event passes).
This is the *semantically real* nibble split (unlike AD/SR halves) and is what "capture reality" points
at — but it's a learnability bet, gated behind the triage, not needed for residual==0.

### 3.2 Layer 2 — optional patch-tuple bundling (compression / learnability)
When a voice loads `(waveform, AD, SR)` together (a full instrument), bundle the fields into one PATCH
atom (the existing `PatchPass` is the seed — it already keys per-voice on freq_reg and is voice-correct;
generalise it from `(AD, SR)` to `(waveform, AD, SR)`). Layer 2 is pure compression on top of Layer 1's
correctness; it is **not** required for residual == 0. Cross-voice instrument sharing (the old phase-2)
becomes an *explicit, controlled* option here (a shared DEF REF'd per voice), never an accident of
canonical-reg conflation.

## 4. What it subsumes / what stays
- **Subsumes (delete):** `CtrlWavetablePass` (all 3 phases), `CtrlWavetableNibblePass`, the `onset_def`
  phase. The nibble pass disappears because Layer 1 absorbs the recurring waveform/AD/SR pre-SubregPass,
  so nothing reaches SubregPass to become a nibble lane.
- **Generalises:** `PatchPass` → Layer 2 (add the waveform field).
- **Keeps (orthogonal temporal mechanisms):** `ctrl_osc` (oscillation), `gradient` (ramps),
  `hard_restart` (gate-off+reload + env multiload), `legato` (gate-retained waveform), `ctrl_bigram/
  triple` (gate-event sequences). These model *time patterns*; Layer 1 models *state recurrence*. They
  compose because Layer 1 owns instrument fields and the event passes own ctrl-lo.

## 5. Correctness & composition
- **Byte/register-state-exact** via the arbiter (`validate=True`) as today, but now claims actually
  round-trip because keys respect voice and there is no id collision.
- **Composes with voice-canonicalization** (voice-aware keys), the **lead frame** (a voice-block within
  frame 0 is just another voice context — the same keying applies), and **multi-block parsing** (one id
  counter per pass invocation; no cross-stage sharing to desync).

## 6. Migration, risk, gates
- New flag `instrument_pass` (or generalise `patch_pass`); **default OFF**, OUT of `REGISTERED_MACROS`
  until the audition gate (register-state-exact, but it changes the encoding ⇒ new vocab ⇒ re-tokenize).
- Ship as **tokens 0.45.0** (retires released ops/passes behaviour — the nibble pass shipped in 0.44.0
  with the collision bug; this supersedes it). Cross-repo per `release_build_cache.md`.
- **Acceptance:** `residual_set_census --step 10 reparse=True` == 0 corpus-wide (digi-excluded); the
  12-SID WAV audition before flipping any default; per-op accuracy unaffected.
- Validate on the corpus, not the 57-tune sample (that overfit hid the collision — see the census
  reparse-cache bug, now fixed).

## 7. Open questions
- **Field granularity for AD/SR:** byte (simpler; nibble-split was shown compression-neutral) vs hi/lo
  (captures sustain-held/release-varying recurrence). Start byte; revisit if the census shows a
  byte-level non-recurrence tail.
- **Generalise `PatchPass` vs new `InstrumentPass`:** PatchPass is already voice-correct and uses the
  `patch` family id-space (disjoint from the retired ctrl_wt). Generalising it is the least-new-surface
  path; a fresh pass is cleaner-slate. Lean: generalise PatchPass for Layer 2, new lean Layer-1 miner.
- **Cross-voice sharing:** drop it (per-voice ids; correctness, slight compression loss) or re-add as an
  explicit Layer-2 option. Lean: drop initially, measure, re-add only if the token budget needs it.
