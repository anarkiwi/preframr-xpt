# Melody data-gap ladder — why Bach generalizes but real-SID melody doesn't

**Status:** Draft / proposed experiment program. Localises, by progressive
simplification of the EXISTING mini data (no new corpus), which data property makes
real-SID melody onsets unpredictable where Bach's are not. Follows from the settled
encoding result below.

## Settled conclusions (the arc that led here)

1. **Architecture generalizes** — `framework_arch_test`, held-out 0.90.
2. **The encoding is sufficient** — a deterministic rule generalizes through the real
   op45/clean-voice encoding single-voice (0.876) and 3-voice multiplexed multi-waveform
   (0.888); and **real Bach generalizes through the same encoding**
   (`bach_encoding_generalization`: held-out next-onset 0.513 > cross-chorale 2-gram
   ceiling 0.456, chance 0.026). See [`encoding_principles.md`](encoding_principles.md),
   [`melody_learnability.md`](melody_learnability.md).
3. Therefore the real-SID melody ceiling (V0 onset ≈ 0.35) is **the data**, not the model
   or the encoding.

## The sharpened observation (this doc)

Bach-on-SID and real-SID-mini are the **same scale, same model, same encoding**, yet:

| | onset count | distinct | marginal entropy | cross-seq 2-gram ceiling | model held-out |
|---|---|---|---|---|---|
| Bach chorales | 35k | 52 | 5.04 b | **0.456** | **0.513** |
| real SID mini (V0_LO) | 72k | 256 | 5.58 b | **~0.30** | 0.35 |

**The marginal entropy is similar (~5 b) — the gap is CONDITIONAL predictability**
(0.456 vs 0.30). Bach onsets are predictable *from context* (voice-leading, harmony);
SID V0 onsets are not. So it is not alphabet size (semitone-quantizing won't fix a
conditional-structure gap — consistent with the earlier flat-ceiling semitone probe) and
not corpus size.

**Leading hypothesis:** the SID **V0 onset is not the musical note.** It is a
value-run/trajectory onset that mixes the gate-anchored base note with **vibrato/slide/arp
re-onsets and cent-jitter** (gate-anchor probe: only 4–23% of freq writes are gate-on;
77–96% are sustain-phase modulation; up to 41 freq-writes per note). The *gate-anchored
base-note* line is low-entropy/Bach-like (Monty 2.68 b), but our onset token isn't it.
So real-SID melody onsets carry the *ornament's* entropy, destroying the conditional
predictability the underlying notes have. Bach (clean quantized notes) has no such pollution.

## The progressive simplification ladder

Each rung is a transform on the **existing mini data**, measured the SAME way as the Bach
test (held-out next-onset acc + cross-seq 2-gram ceiling), so Bach and every rung are
apples-to-apples. The rung where held-out acc jumps toward Bach (~0.45) localises the gap.

| rung | transform | tests | predicted |
|---|---|---|---|
| **L0** | real mini melody, current encoding | baseline | 0.35 |
| **L1** | single melodic voice (drop perc / effects / accompaniment) | voice confusion / "which voice is the melody" | small |
| **L2** | **gate-anchor**: onset = gate-on base note only; drop mid-note ornament re-onsets (reuse `TrajectoryAnchorPass` + a de-ornament step) | melody buried in ornament | **KEY — biggest jump expected** |
| **L3** | de-arpeggiate: collapse fast intra-note arps to a held root | arp-as-melody | medium (if arps present) |
| **L4** | semitone-quantize the onset pitch (collapse cent-jitter to pitch classes) | sub-semitone jitter spreading one note across tokens | small (marginal-entropy lever; secondary) |
| **L5** | rhythm-quantize onsets to a metric grid (like Bach's 16th grid) | timing irregularity | small |
| **target** | Bach-clean | — | ~0.45 |

**Order rationale:** prioritise *onset-purification* (L1→L2→L3) over *cardinality reduction*
(L4→L5), because the gap is conditional structure, not marginal entropy, and Bach itself
has ~5 b marginal entropy yet generalizes. The hypothesis is that L2 (gate-anchoring to the
true note) closes most of the gap; L4/L5 are controls that should NOT move it much (if they
do, the cardinality story revives).

## Measurement harness

Reuse `bach_encoding_generalization`'s pipeline: extract per-voice onset sequences from the
mini corpus at each rung, transcode to the current encoding, train/held-out split, report
held-out next-onset acc + cross-seq 2-gram ceiling. Identical metric to Bach → the gap is
read directly off the ladder. Gate-anchoring (L2) builds on the landed
[`trajectory_anchoring`](landed/trajectory_anchoring.md) / gate-anchor probes.

## Decision rule

The first rung that lifts held-out onset acc to Bach-like (≳0.45) is the data property that
makes SID melody hard. If **no** rung closes the gap, real-SID melody is genuinely more
aleatoric than chorales (composer/engine-specific), and the melody yardstick must go
distributional/audition. Either outcome is decisive and uses no new data.
