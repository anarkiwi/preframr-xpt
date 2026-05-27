# Sequence-order normalization (intra-frame write-order collapse)

**Status:** Draft 2026-05-27. Empirically-motivated by a CPU cross-engine
divergence probe (below). Tokenizer-side normalization that collapses the
**write-order** degree of freedom within a frame — the sequence-level sibling
that [`audio_equivalence_normalization_design.md`](audio_equivalence_normalization_design.md)
explicitly defers ("that direction handles sequence-level equivalences (write
order, redundant writes, context-dependent collapses) and is a larger
redesign"). This doc scopes the write-order subset with measurements, not the
per-write `val` collapse (that doc) nor redundant-write dedup.

Generalization axis / representation. Builds on the engine-divergence finding
(engines **share content vocabulary**, full-atom content cosine 0.96; divergence
is structure + sequencing, not content) — see [[representation-thread]] and
`tokenization_vs_music_llms.md`.

## Why this matters

Project thesis: the lever is representation, not architecture. If the model must
learn the same audible frame written in 8 engine-specific orders, it spends
sequence-modeling capacity memorizing engine idiom instead of generalizing
content. Any write-order that the SID renders identically is a normalization the
tokenizer can do once, for free, before training — collapsing 8 idioms to 1.

## Evidence (probe: `/scratch/tmp/structure_seq_probe.py`)

Parsed all 8 eval_b engine families to post-`full_macros` atom streams (motif
off), tiered each atom, split into inter-frame blocks, and compared cross-engine
cosine of token-type and per-frame reg-order distributions (mean over the 28
family pairs). Ran in `anarkiwi/preframr:0.2.3`, CPU, on the STAGE 2 full_macros
seed0 staged eval_b dumps.

**A. What token TYPES differ (op,reg,subreg distribution):**

| tier | cross-engine cosine (mean, min–max) |
|---|---|
| structural | **1.000** (1.000–1.000) |
| content | 0.878 (0.526–0.989) |

Structural token *types* are universal; content types largely shared. The
divergence is **not** in which token types exist.

**B. What regs per frame vs in what order (per inter-frame block):**

| signature | cross-engine cosine (mean, min–max) |
|---|---|
| reg SET (which regs touched/frame, order-free) | 0.466 (0.023–0.931) |
| reg ORDER (write sequence/frame) | 0.297 (0.042–0.720) |
| **gap (SET − ORDER)** | **+0.169** |

Two distinct divergences: (1) the per-frame reg **composition** differs a lot
(SET 0.466) — this is genuine arrangement/timbre (how many voices active, pw
modulated, filter used) and is **content, not normalizable**; (2) on top of
that, the **order** of writes diverges a further +0.169 — same regs, different
sequence. That +0.169 is the inaudible-normalizable component.

**C. Which pairs disagree on order (precede-fraction spread across engines):**

| spread | reg pair | what | safe to canonicalize? |
|---|---|---|---|
| 0.838 | (−126, 5) | VOICE_REG marker vs AD env | **yes** — marker is a tokenizer artifact |
| 0.670 | (−126, 0) | VOICE_REG vs freq_lo | **yes** |
| 0.636 | (−126, 6) | VOICE_REG vs SR env | **yes** |
| 0.635 | (2, 5) | pw_lo vs AD env | **yes** — both value-latch |
| 0.615 | (0, 2) | freq_lo vs pw_lo | **yes** — both value-latch |
| 0.586 | (0, 5) | freq_lo vs AD env | **yes** |
| 0.568 | (0, 6) | freq_lo vs SR env | **yes** |
| 0.567 | (2, 4) | pw_lo vs **CTRL/gate** | verify — gate-edge-adjacent |
| 0.454 | (−126, 4) | VOICE_REG vs CTRL | yes (marker side) |
| 0.338 | (0, 4) | freq_lo vs **CTRL/gate** | verify — gate-edge-adjacent |
| ≤0.10 | (4,5)(4,6)(5,6)(·,21) | gate↔env, env↔env, ·↔filter | already canonical (leave) |

Regs (post-pipeline): `−126`=VOICE_REG (a tokenizer voice-select **marker**, not
a SID write), `0`=v1 freq_lo, `2`=v1 pw_lo, `4`=v1 CONTROL (gate/waveform/test —
**edge-sensitive**), `5`/`6`=AD/SR envelope, `21`=`FC_LO_REG` filter cutoff.

The high-spread pairs are dominated by (a) VOICE_REG marker placement and (b)
value-latch reg pairs — both inaudible to reorder. The gate↔envelope and
envelope↔envelope orders are **already consistent** across engines (≤0.10 spread)
— existing macro passes + the parser sort cover them; do not touch.

## Why the divergence survives the existing sort

`reglogparser.RegLogParser` already applies a canonical intra-frame sort
(`_norm_pr_order`: `sort_values(["f","v","reg","op","n"])`) — but at lines
901 and **909**, i.e. *before* `_add_voice_reg` (910), the optional `full_macros`
transforms (912), and `FreqNudgePass`/`CtrlUpdatePass`/`LonelyWriteValidatorPass`
(913–915). Those post-909 passes **insert VOICE_REG and reshape sequences with no
final re-canonicalization**, re-introducing exactly the order divergence the
probe measures. VOICE_REG (the #1 disagreement source) is placed by
`_add_voice_reg` after the last sort.

## Candidate: a final gate-anchored intra-frame re-order pass

Add a post-norm pass (after 915, before tokenization) that, **within each
inter-frame block**, sorts writes into a canonical order while:

1. **Anchoring CTRL (reg 4/11/18) writes in place** — never move a value-latch
   write across a gate/waveform/test write (preserves every edge-triggered
   event). Sort only the runs of non-CTRL writes between fixed CTRL anchors.
2. **Treating abstract macro tokens as atomic units** — FREQ_TRAJ_OP,
   CTRL_BIGRAM_OP, motif ops etc. have meaningful internal order; sort them as
   single units by a stable key, never reorder their interior.
3. **Canonicalizing VOICE_REG placement** — emit the marker in one fixed
   position relative to its voice's writes (it carries no audio).
4. **Within a non-CTRL run, sort by (reg, subreg)** — the value-latch regs
   (freq/pw/envelope/filter) in numeric order.

Expected effect: ORDER cosine rises from 0.297 toward the SET ceiling 0.466
(order-tuples become a function of the reg-set), a ~+0.17 / ~57% relative gain in
cross-engine sequence agreement — bounded, because the SET divergence (genuine
content) remains. Net: the model sees one canonical idiom per audible frame.

## Gates / risks

- **Round-trip audio fidelity is the load-bearing gate.** Reuse the
  `preframr_audio` render+compare path (same as the per-write doc's Phase 0
  audit): render original vs reordered for ≥100 dumps, require ≥95% within
  `mel_distance` tolerance, `max_frame_drift` small. Any audible drift ⇒ the
  gate-anchoring is too loose.
- **Decode order-dependence.** The tokenizer/decoder may assume the post-transform
  order; a re-sort must commute with `constrained_decode` + back-ref/pattern
  replay (those carry positional `subreg` offsets). Verify byte-exact round-trip
  on the existing fidelity suite before any A/B (this is the class of bug that
  bit the motif v2 frame-swallow — integration-test the full parse→tokenize→
  decode, not just the pass in isolation; see [[representation-thread]] lesson).
- **Bounded upside.** +0.169 is the entire pure-order budget; if sequence-idiom
  memorization isn't actually costing content capacity, the A/B reads flat. Cheap
  to test, so worth a mini A/B before committing.
- **Gate-adjacent pairs (0,4)/(2,4)** disagree but are *probably* inaudible
  (freq/pw is latched continuously, the gate edge reads the latched value either
  way). Anchoring CTRL is the conservative choice; a follow-up could test moving
  freq/pw across the gate if the audit clears it.

## Phase plan

| Phase | Scope | Gate |
|---|---|---|
| 0 | Implement `SeqOrderNormPass` (gate-anchored, macro-atomic, VOICE_REG-canonical); byte-exact round-trip on the fidelity suite | round-trip byte-exact; ORDER cosine on eval_b rises measurably toward SET |
| 1 | Mini A/B: full_macros vs full_macros+`--seq-order-norm`, 3 seeds, body=large | per_class content-tier val_acc ≥ baseline + diversity_ratio non-regressed + audio round-trip ≥95% |
| 2 | Prodlike single seed if Phase 1 wins | eval_a content ≥ baseline + 0.003 |

## References

- Per-write sibling (value collapse, write-order-independent):
  [`audio_equivalence_normalization_design.md`](audio_equivalence_normalization_design.md).
- Engine-divergence framing: `tokenization_vs_music_llms.md`,
  [[representation-thread]] (content vocab shared 0.96; lever is structure/seq).
- Probe: `/scratch/tmp/structure_seq_probe.py`. Reg constants:
  `preframr_tokens/stfconstants.py` (VOICE_REG=−126, FC_LO_REG=21, CTRL regs
  4/11/18). Pipeline order: `preframr_tokens/reglogparser.py:894–916`.
