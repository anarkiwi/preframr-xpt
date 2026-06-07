# Superframe + voice-lane encoding

**Status:** **REINSTATED 2026-06-05 as LAYER 3 of the melody plan — the cross-voice de-multiplexing the
generator-MDL pipeline does NOT provide** (it is per-voice but frame-interleaved). This is **complementary to,
not superseded by**, the generator-MDL ([`generator_mdl_representation.md`](generator_mdl_representation.md))
and the interval-skeleton ([`melody_skeleton_impl.md`](melody_skeleton_impl.md)): those make the per-voice line
clean + key-invariant, this makes it **contiguous** so the model's next-melody-note horizon is local (P3). It
is the project's measured *dominant* melody lever — deployed melody-onset ≈ 0 vs the ~0.34 per-voice ceiling,
the gap being cross-voice multiplexing; the interval-skeleton's 0.52 was measured on already-de-multiplexed
single-voice data, so it does not survive deployment without this layer. Untested at deployment → pre-screen
with `audit/learnability_triage.py`, gate by one canonical run. (Was wrongly deleted in the 2026-06-05
consolidation; restored.) Original draft notes follow.
This captures the layout + rationale the existing `super_frame` scaffold left blank;
do not implement until the current melody-stream diagnosis lands and the two open
prereqs below are answered.

**Learnability framing.** Voice-major lanes shorten the dependency horizon and make the same-voice predecessor positionally local ([`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md) Principles 1+4) — the structural realization of the de-mux lever; gate it on per-frame h_k, not only V0-onset accuracy.

## What already exists

A `super_frame` scaffold is shipped but inert:
- `SuperFrameTransform` (`preframr_tokens/macros/transforms_superframe.py`): *"pack N
  consecutive PAL frames into one super-frame block."* N=1 is a no-op; **N≥2 raises**
  (`_pack_super_frames`/`_unpack_super_frames` unimplemented). `LOSS_TIER=structural`,
  `TIER=bit_exact`.
- `SUPER_FRAME_REG = -124` reserved (`stfconstants.py`); decoded label "SUPERFRAME"
  in `audit/probes/inspect_frames.py`.
- Pass position: `MUST_FOLLOW voice_block_order`, `MUST_PRECEDE add_voice_reg` — i.e.
  it runs after voices are ordered but **before** the canonical voice-reg remap, which
  is exactly where a voice-aware reorganization belongs.

So the temporal-window axis (group N frames) and the pipeline slot are already
reserved. What's missing is the **intra-superframe byte layout** and the **why**.

## Two open prereqs (answer before implementing — not now)

1. **Original intent.** The scaffold's language is token-budget-first (fewer FRAME
   markers per N frames). Whether superframe was ever meant to *reorganize per voice*
   is unknown. The voice-lane layout below may be a new justification grafted onto the
   reserved slot, not the original plan. Confirm before claiming continuity.
2. **Why N≥2 stalled.** The doc "never landed" and N≥2 was deliberately left raising.
   There may be a known blocker (round-trip exactness across the pack boundary, or the
   self-contained-block interaction). Resolve what the wall was before building, so we
   address it rather than rediscover it.

## The problem (grounded in the current stream)

Today the stream is **frame-major**: each frame writes all active voices, in an order
the FRAME token's val encodes (see [`voice_encoding_reference.md`](voice_encoding_reference.md)).
This interleave taxes melody learning:

1. **Melodic fragmentation with position-shifting gaps.** A voice's line is one chunk
   per frame, landing at a *different offset* each frame because the FRAME header
   permutes voice order. "Predict V0's next note" is a long-range dependency over a
   variable, non-stationary gap.
2. **Implicit, non-local voice identity.** A write's voice = `(FRAME.val >> 2·k) & 3`
   integrated over a running marker count — bookkeeping with no architectural support.
3. **Multiplexed next-token target.** At any step the next token could be any active
   voice's write, so the prediction distribution mixes three lines — diluting the
   V0-onset signal. A structural contributor to V0_HI≈0, on top of the Unigram weld.
4. **Cross-voice Unigram merges.** Interleaving manufactures the cross-melody-boundary
   merges that `melody_merge_split` exists to undo.
5. **Fragility to PW/filter re-admission.** Re-adding per-voice timbre widens each
   voice's slot, pushing melodic tokens further apart. The substrate ablation's gain
   was partly just shrinking slots and tightening melodic spacing — indirect and
   fragile.

Prior voice work bounds the solution space: `voice_trajectory` (insertion +
distributed) added per-voice *state features* and was **refuted**; `sequence_order_normalization`
canonicalized write *order* and recovered only ~5%. So the lever is **neither
voice-feature-injection nor order-normalization — it is stream structure.**

## The proposal: voice-major lanes inside the superframe

Use the superframe as the temporal window (N frames), and lay it out **voice-major**:
each voice's events form a contiguous lane, with a compact per-event frame index
(delta/run-length encoded, near-free since notes sustain). Decode re-interleaves lanes
by `(frame_index, canonical_voice)` back to the exact frame-synchronous stream.

A **lane → register-class sub-lane** hierarchy is the extensibility mechanism:

```
superframe (N frames)
 ├─ V0 lane ─┬─ freq/melody sub-lane   (FREQ_TRAJ / FREQ_ONSET — stays contiguous)
 │           ├─ control sub-lane
 │           ├─ ADSR sub-lane
 │           └─ [PW sub-lane]          ← re-admit later; does NOT touch melody sub-lane
 ├─ V1 lane ─ …
 └─ V2 lane ─ …
```

Re-adding PW or per-voice filter becomes "add a sub-lane," not "thread another write
type through the global interleave." The melodic freq sub-lane stays contiguous
regardless of how much per-voice timbre is re-admitted — the flexibility the current
encoding lacks.

## Why this attacks the challenges

- V0's next note becomes the **previous token in the same sub-lane** — short-range,
  position-stable. The trigram-0.79 predictability ceiling becomes reachable.
- The within-sub-lane target is **one voice** — de-multiplexed, undiluted.
- Unigram merges become **within-lane** (same-voice, musically coherent: a note's
  onset+shape, or consecutive notes of one melody); the cross-melody-boundary merge
  problem largely dissolves.
- The **block stays the prompt/decode unit** (superframe ⊆ self-contained block) — no
  conflict with mid-song prompting.
- Canonical voice order falls out for free (kills permutation surface-forms), a minor
  bonus per the seq-order finding, not the bet.

## The honest risk

Serializing a 2D (voice × time) structure into 1D buys melodic contiguity at the cost
of **cross-voice adjacency**: which notes sound together becomes an implicit relation
over frame-indices, not positional adjacency. Harmonic/rhythmic alignment may regress.
For a program whose goal is melodic generalization, trading toward melody is the right
bet — but it must be measured: mini A/B voice-lanes vs frame-major on the clean
substrate, read by V0_HI onset acc **and** an audition/harmony metric (so a melody win
that wrecks harmony is caught), with the round-trip oracle green and seq-length held
~neutral via frame-gap delta-encoding.

## Sequencing (subordinate to current work)

1. **Now:** the `no_unigram` A/B isolates the merge weld on the *current* frame-major
   stream — get V0-onset off zero there first. (Superseded by the generator-MDL substrate, which already
   de-merges; the live use of this doc is the cross-voice de-mux LAYER 3 in the head note.)
2. **Then, if interleave is still taxing** (V0_HI moves but plateaus): answer the two
   open prereqs, then prototype voice-major lanes as the superframe N≥2 layout.
3. The lane hierarchy is the long-term home for PW/filter re-admission regardless of
   (2)'s magnitude.

## Code anchors

- `preframr_tokens/macros/transforms_superframe.py` — the scaffold to build on.
- `preframr_tokens/stfconstants.py` `SUPER_FRAME_REG`.
- [`voice_encoding_reference.md`](voice_encoding_reference.md) — the current frame-major
  voice encoding this would replace within the superframe.
