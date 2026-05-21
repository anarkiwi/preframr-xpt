# `palette_merge` — REFUTED (pre-2026-05-10, design phase)

**Status:** two refuted variants from
`integration_tests/design/palette_merge_design.md` Phase 1. Neither
advanced to A/B; the design notes captured the refutation evidence
ahead of implementation.

## Refuted variants

### 1. Content-addressed slot ids

**Hypothesis:** instead of palette slot ids being arena-local
integers (assigned in first-appearance order per song), use a
content-addressed scheme — slot id = hash of the bundle payload
truncated to a fixed bit budget. Cross-song bundle reuse would then
share the same alphabet entry, shrinking the effective vocab.

**Refutation:**
- Hash collisions across distinct bundles introduce a "wrong
  bundle replays as wrong instrument" failure mode at inference.
  Probability scales with `(N_bundles)² / 2^bits` — at prodlike
  scale (~5K distinct bundles) requires ≥ 20-bit hash budget,
  which doesn't shrink alphabet vs the existing arena scheme.
- The reuse rate across songs is empirically low (most bundles
  are song-specific) — the alphabet shrink potential is bounded
  by ~10-15%, not the >30% needed to justify the inference risk.

### 2. Nearest-program merge

**Hypothesis:** during parse, merge palette entries within an L2 /
Hamming distance threshold so that "similar but not identical"
bundles share a slot. Reduces palette cardinality on engines that
emit slightly-varying instruments per voice (Hubbard).

**Refutation:**
- The distance threshold tuning is corpus-specific; no value
  works across engine families (Hubbard tolerates 4-bit Hamming
  merges; Galway breaks at 2-bit due to detuning-as-instrument-
  differentiator).
- Even at the best threshold, palette cardinality reduction is
  ~8% (vs an alphabet-size budget gap of ~30% for the prodlike
  envelope). The merge doesn't move the needle on the encoder's
  bottleneck.
- The non-deterministic palette merge breaks A/B reproducibility:
  the merge result depends on bundle-arrival order, which is
  sensitive to subtune ordering / song-init perturbation.

## Evidence

`integration_tests/design/palette_merge_design.md` Phase 1 — both
variants analysed with corpus-wide cardinality estimates and
collision projections; neither reached the A/B advancement gate.

## Do not revisit without

- A new bundle-similarity metric grounded in audio rendering
  similarity (not register-byte L2), AND
- Empirical evidence on a >5-engine-family corpus that the merge
  is stable under composer perturbation (i.e. shuffling the
  parse order doesn't change the palette).

The standing alternative is the (slot, palette) format committed
in HEAD: arena-local, deterministic, no merge.
