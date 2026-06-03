# Melody data-gap ladder — why Bach generalizes but real-SID melody doesn't

**Status:** EXECUTED 2026-05-29 (see RESULTS). Gate-anchor hypothesis REFUTED; the gap is
the data's intrinsic melodic predictability (heterogeneous, less-constrained game melody vs
homogeneous Bach chorales), not the encoding. Localised by progressive simplification of the
EXISTING mini data (no new corpus).

**Learnability framing.** The onset-scoped instance of [`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md)'s entropy-rate triage (`audit/melody_predictability.py` is the same math on V0 onsets); the key-invariant interval conclusion = anchoring a high-entropy determinant to a low-entropy reference (Principle 4.2).

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

## RESULTS (2026-05-29, `audit.melody_ladder`) — gate-anchor REFUTED; SID melody is intrinsically harder

Held-out next-onset-pitch acc (single-voice onset sequences, MIDI; same metric for all):

| level | 2-gram (oracle) | model held-out |
|---|---|---|
| **Bach soprano (target)** | 0.350 | **0.375** |
| L1 all-freq onsets | 0.541 | 0.559 |
| L2 gate-anchor (pooled 3 voices) | 0.196 | 0.247 |
| L3 de-arp (pooled) | 0.236 | 0.269 |
| L2 gate-anchor, **lead voice only** | 0.154 | 0.179 |
| L3 de-arp, **lead voice only** | 0.167 | 0.192 |

**The hypothesis was wrong, and the ornament was a trap.** L1's high 0.559 is an *artifact*:
consecutive freq writes during vibrato/sustain repeat, so "next freq" is trivially
predictable *ornament*, not melody. Extracting the actual note-ons (L2) **drops**
predictability to 0.247, and isolating the *lead* melodic voice drops it further to **0.179** —
the more melodic the line, the *less* predictable. No rung approaches Bach (0.375).

**Crucially the 2-gram ORACLE ceiling is also far below Bach at every note-level rung
(0.15–0.24 vs 0.35).** So this is not a model or encoding limit — the model beats its data
ceiling at every rung (0.179 > 0.154, 0.247 > 0.196, …, as it does for Bach 0.375 > 0.350).
**The SID note-melody simply has less conditional structure than Bach chorales.**

### Conclusion
The gap is the DATA's intrinsic melodic predictability, confirmed three ways (encoding proven
sufficient via Bach; oracle n-gram ceiling low; model at-ceiling everywhere). The likely
drivers, now evidenced rather than assumed:
1. **Heterogeneity** — mini is many composers/games; Bach chorales are one composer, one
   highly-constrained form. Comparing them assumes equal intrinsic predictability; they are not.
2. **Less harmonic constraint** — game melodies (riffs, arps, effects) are freer than chorale
   voice-leading.
3. The "melody buried in ornament" framing was half-right: the ornament *inflated* apparent
   predictability; the underlying note melody is genuinely harder.

So "strong melody prediction from SID" is NOT reachable by re-encoding or de-ornamenting this
corpus. The implicated (untested) data lever is a **homogeneous, harmonically-constrained
melodic SID corpus** (single composer/style) — i.e. the right *kind* of data, not more of it.
For the current heterogeneous corpus, exact-note melody is genuinely low-predictability and the
yardstick should be distributional/audition.

### Correction + DRAX homogeneity (2026-05-29, intervals)

The absolute-MIDI metric above is **key-confounded** — cross-tune absolute pitch doesn't
transfer when tunes are in different keys, and it does NOT match the encoding (which uses
key-invariant interval-V0). Re-measured on **intervals** (consecutive pitch diffs):

| (interval-based, lead voice) | 2-gram | model held-out | train tunes |
|---|---|---|---|
| Bach soprano | 0.349 | 0.394 | 480 |
| mini heterogeneous | 0.214 | 0.225 | 118 |
| **DRAX (single composer)** | 0.100 | **0.247** | **20** |

Findings: (1) intervals lift SID melody (DRAX 0.146→0.247 vs absolute) — absolute pitch was an
unfair, encoding-mismatched metric. (2) **DRAX edges the heterogeneous corpus (0.247 > 0.225)
on 6× less data**, and the model beats its oracle bigram ceiling 2.5× (0.247 vs 0.100) — so
single-composer melody has real, model-extractable structure and is **data-starved**, not
structureless. (3) Still below Bach (0.394), but the gap narrowed substantially.

Revised conclusion: the gap is **smaller than the absolute-MIDI ladder implied, and partly
data-volume**. The right melody representation is **interval** (key-invariant, as the encoding
already does); single-composer homogeneity helps modestly even data-starved. The next data lever
is a **larger single-composer / single-style melodic corpus**, measured on intervals — most
likely to approach Bach-like melodic generalization. Encoding remains proven-sufficient.
