# Macro learnability triage + mini-experiment ladder

**Status:** PROPOSED 2026-06-02. Triages all 35 macro passes by **learnability ×
transferability × context-efficiency** (parse efficiency explicitly deprioritised) and
lays out a cheap-first mini-experiment ladder to *progress* learnability on the
byte-exact (preframr-tokens 0.40.0) substrate. Companion to the learnability audit
(`learnability_full_macros_mini`, `learnability_codebook_mini`). Cross-ref:
[`encoding_principles.md`](encoding_principles.md) (P1–P8),
[`ornament_transfer.md`](ornament_transfer.md) (parametric re-encoding REFUTED as a
transfer lever), [`melody_channel_factorization.md`](melody_channel_factorization.md)
(voice de-mux is the recurring lever), [`unified_pitch_encoding.md`](unified_pitch_encoding.md)
(skeleton generalises).

**Learnability framing.** This per-pass triage is the macro-scoped instance of [`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md), which supplies its information-theoretic backing and a whole-stream tool (`audit/learnability_triage.py`); relative + inline define→ref passes are Principle 2 (induction-head copy).

## The transfer test (one principle)

A macro transfers across songs/composers iff its payload is resolvable from either
**universal structure** or **local in-sequence context** — never from absolute
per-tune / per-driver identity. Two encodings satisfy this:

1. **Relativity / provenance-invariance (P7).** The same gesture at a different
   pitch/base must encode to the *same* token. Delta / interval / note-relative
   encodings collapse instances; absolute values fork them. STAMP keys on a
   transpose-invariant freq-delta signature; WAVETABLE on note-relative offset
   programs — both relative by construction.
2. **Inline define → reference → mutate.** The codebooks are *inline*: the DEF is
   emitted in-stream (WAVETABLE_DEF/STEP/END ops 65–67 → REF 68; STAMP_DEF →
   STAMP_REF 59 / REL_REF) just before use. The id is a **pointer into a
   locally-defined set**, so the model resolves it by copying from recent context — a
   universal induction/copy skill. *The contents being per-tune is fine; the
   mechanism is what transfers.* This is why a **global** id dictionary
   (`global_instr_ids_phase_a`) was **refuted** while the inline codebooks are the
   right shape.

Learnability then needs two more (both earned by results): **low-cardinality,
separable tokens (P1)** — de-merge lifted pitch onset 0.009→0.658 — sitting **near
their decision (P2)**.

## Macro triage

Production-registered set (`tokenizer_config.REGISTERED_MACROS`) is FRAME-svt-safe;
the codebook/skeleton passes (0.40.0) are newer and byte-exact on the skeleton path.

**Tier 1 — progress (high learnable × high transferable).**
| macro | why it transfers + learns |
|---|---|
| `skeleton_pass` (+ `trajectory_anchor_pass`) | per-note SKEL over one semitone LUT; generalization-tested; the melody substrate everything layers on |
| `loop_pass` + `loop_transposed` | universal repeat; back-ref distance is locally resolvable, *transposed* loop is pitch-invariant repeat — the cleanest copy mechanism |
| `stamp_pass` | transpose-invariant freq-delta + exact-ctrl signature, inline-redefinable, relative `STAMP_REL_REF`; drums recur heavily and the gesture is relative |
| `ctrl_bigram_pass` / `ctrl_triple_pass` | control-register role patterns (P8); driver-universal |
| `hard_restart_pass`, `legato_pass_c2`/`c4`, `voice_canonical_block_order` | universal structural/driver gestures; cheap, de-multiplex, provenance-invariant |

**Tier 2 — audit-then-progress (transferable *iff* the inline mechanism learns —
exactly what `learnability_codebook_mini` measures).**
| macro | open question |
|---|---|
| `wavetable_pass` | note-relative inline ornament codebook, per-composer reuse (Hubbard 6 tables/13×). Does DEF→REF *selection* learn + transfer to eval-B? |
| `sweep_pass` (+ `sweep_loop`) | parametric portamento (start/end/rate) — relative by construction; audit param/id learnability |
| `held_arp` | relative arpeggio as a cross-note loop |

**Tier 3 — hold / keep but not the lever.**
| macro | note |
|---|---|
| `patch_pass` | per-tune (AD,SR) codebook; inline-defined but ADSR is less obviously *relative* than pitch/drum gestures — weaker transfer; revisit if the audit shows ADSR id reuse |
| `preset_pass`, `release_update_pass`, `lonely_catch_all` | context-efficiency + RESID→0 completeness; learnability-neutral (don't expect a content win). `lonely_catch_all` is load-bearing for completeness — keep |
| `freq_trajectory_pass`, `freq_v0_interval`, `freq_onset_pass`, `freq_nudge_pass` | the *pre-skeleton* freq channel; superseded by skeleton for melody (and conflicts with it). Keep only as the non-skeleton A/B arm |

**Tier 4 — drop** (refuted or low-value): `coarsen_pass` (`macro_coarsening` refuted),
`fuzzy_loop_pass`, `fuzzy_fp_adsr`, `gate_slope_shift_pass`, `mode_vol_flip_pass`,
`voice_track_pass`, `legato_pass_c7`, and the wavetable micro-variants
(`wt_short`/`wt_oneshot`/`zero_plain`/`slide_wide`/`slide_landing`) unless a specific
fidelity gap demands them. Already-refuted: `motif_pass`, `palette_merge`,
`adsr_equivalence`, `global_instr_ids`.

## The learnability frontier: three gates + the yardstick

The architecture already generalises (`framework_arch_test`: train 1.000, val 0.903 on
unseen motifs) — the bottleneck is the encoding. Learnability is gated by:

1. **Separability (P1)** — *done* via `--tkvocab 0` (Unigram OFF). Both audit specs run
   here; reintroduce Unigram only later, mask-aware.
2. **De-multiplexing (P3)** — *the open lever.* Three voices multiplexed into the
   next-token slot dilute any one voice's signal. Within-voice ornament multiplex is a
   modest cost (+0.032 ×3 seeds); the **cross-voice frame multiplex is ~2× larger
   (−0.062 → 0.274)** and never reached ≈0 deployment. Every melody result converges
   here. **A transferable codebook over a multiplexed stream still won't learn — so
   de-mux precedes codebook progression.**
3. **Reuse-selection** — does inline define→reference→mutate learn, and *transfer*?
   Because ids are defined in-context, the test is whether REF resolves against the
   local DEF on **held-out** tunes (mechanism) vs memorising train-tune id stats.

**Yardstick (P6).** Multi-modal targets (pitch magnitude) cap at ~0.51 even for an
in-sample memorising n-gram. Score such targets **distributionally + by audition**
(the muspy `melody_features` automation already exists), or exact-token accuracy
under-counts a generalising model.

## Mini-experiment ladder (cheap-first; each step gates the next)

All mini-tier, body=large, ×3 seeds, `--tkvocab 0`, `--generalization-gate`, byte-exact
rebake, image `anarkiwi/preframr:0.2.15`. ~10 min/arm observed.

- **L0 — `learnability_full_macros_mini` (RUNNING).** full_macros (REGISTERED_MACROS)
  vs atomic baseline. Read: `gate/content_over_structural`, `op_acc(DIFF/BACK_REF)`,
  content_tier_report. **Go/no-go:** does the production compressing vocab beat atomic
  on content_acc on *corrected* tokens (re-confirm the 05-27 win).
- **L1 — `learnability_codebook_mini` (GATED on the codebook byte-exact audit).**
  skeleton+stamp+patch+sweep+held_arp+wavetable + compatible registered macros vs
  atomic. Read: `op_acc(SKEL/STAMP_REF/WAVETABLE_REF)` **+ the coupling metric below.**
  **Go/no-go:** do codebook payloads *select* correctly AND transfer to eval-B.
- **L2 — voice de-multiplexing (the biggest known lever; not yet run on the byte-exact
  skeleton substrate).** Voice-lane / superframe re-encoding vs frame-interleaved, both
  on skeleton. Decisive read: per-voice SKEL acc. Predicted ≈+0.06 from the cross-voice
  multiplex cost. **If it lands, this is the substrate** and codebooks layer on it.
  (Builds on `superframe_voice_lane_design.md`.)
- **L3 — reuse-selection encoding ablations (only if L1 shows weak REF transfer).**
  (a) relative-REF (distance-to-DEF / most-recent-of-type) vs absolute-id REF — lower
  card + locality; (b) BACK_REF as structural "repeat phrase k" vs raw frame distance.
  Read: eval-B `op_acc(REF/BACK_REF)`.
- **L4 — yardstick upgrade (cheap, do alongside).** Wire `melody_features`
  (pitch-class entropy, scale-consistency, in-scale rate) into the content read so
  multi-modal pitch is scored distributionally, not just exact-token.
- **L5 — capstone.** Scale the winning substrate (de-mux ± the green codebooks) to
  prodlike ×3 on content_acc + tier/op accuracy.

## The codebook coupling metric (instrument for L1/L3)

`op_acc(STAMP_REF)` alone conflates two things: a model can score it by **memorising
per-tune id frequencies** or by genuinely **tracking the inline DEF**. Distinguish them:

1. **Eval-family stratification.** REF-selection accuracy on **train vs eval-A vs
   eval-B**. Because ids are defined in-context, a model that learned the *mechanism*
   references correctly on held-out composers (the DEF is in the prompt); a memoriser's
   eval-B REF acc collapses. **eval-B REF acc is the transfer test.**
2. **DEF→REF distance stratification.** Accuracy bucketed by how far back the matching
   DEF is. Flat-with-distance = real tracking; near-only = recency bias.
3. **Learned validity (mask-off).** Under teacher-forced val (no constrained mask),
   rate at which the predicted REF id ∈ ids-already-DEFINED-in-this-tune. Separates
   *learned* validity from mask-guaranteed validity (the handoff caveat).
4. **Reuse descriptors.** refs/def and DEF→REF distance distributions per op, as corpus
   stats (how much reuse is even available to learn).

Implemented as a tested audit reader (`audit.codebook_coupling`) over the per-class
prediction data, stratified by eval family — not a bespoke `/scratch/tmp` script.

## Decision summary

Keep-and-progress core = **skeleton + transposed-loop + stamp + control/structure
roles** (relative + inline = learnable + transferable). **wavetable / sweep / held-arp**
progress on a green define→reference audit. **patch** holds. The freq-trajectory stack
and the experimental tail retire. **De-mux (L2) is the highest-leverage next step** and
should not wait behind codebook polish.
