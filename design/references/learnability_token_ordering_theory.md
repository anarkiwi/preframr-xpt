**Status:** Reference + tool (`audit/learnability_triage.py`, training-free). The v3 event model is
the lens's current product; the triage remains the pre-run ranking instrument for any proposed
representation change (run it at seq_len 8192, window mode, before spending a training run).

# A theoretical basis for token + ordering design (predict before you A/B)

The all-empirical loop (mini A/B per encoding) cannot see the thing it is testing: mini
**mode-collapses regardless** of vocab (`loop_collapse_rate` ~1.0), so the collapse→learning
transition only appears at canonical/prodlike. Theory + training-free measurement decide direction
and ordering; one canonical run decides the threshold.

## The load-bearing fact: the architecture is already exonerated
`framework_arch_test` — torchtune llama3_2, mini — reaches **val 0.903 on UNSEEN synthetic motifs**
(`design/landed/substrate_ablation_v1.md`). The model class generalizes when the structure is clean.
So a SID failure is **not capacity** — it is a mismatch between the *encoding* and what a bounded
transformer can cheaply represent. That mismatch is computable from the encoder + driver model
**without training**. (Confirmed empirically twice over: model-side interventions all refuted at the
~0.13 content ceiling; tokenizer-side representation lifted it — most recently the v3 event model's
atoms-only baseline at eval_a content 0.479.)

## Principle 1 — a transformer is a bounded automaton, not a recurrence
A constant-depth, log-precision softmax transformer sits at ~**TC⁰** (Merrill & Sabharwal 2023). It
**cannot maintain an unbounded sequential counter** and generalize over length. Liu et al. (ICLR'23):
a transformer *can represent* a finite automaton, but SGD finds a **shortcut** that fits the training
horizon and fails to extrapolate. A SID driver **is** a finite-state machine (wavetable pointer,
envelope phase, arp index, ramp counters), so learnability reduces to two quantities, derivable from
the encoder logic + the driver model:

1. **Causal-state size** — how big a latent must be reconstructed to predict the next token
   (Crutchfield causal states / statistical complexity C_μ).
2. **Dependency horizon** — how far back the determining tokens live.

Minimize both and you are back in the regime the synthetic motifs proved learnable.

## Principle 2 — prefer induction-head-expressible structure
Induction heads (copy "what followed this token last time") are the first, most reliable circuit a
transformer forms (Olsson et al. 2022). Explicit repetition-with-cheap-reference is the easy regime;
an implicit per-frame arp/ramp counter is not (Principle 1). The right notion of "compressible" is
**copy-fraction, NOT gzip** — gzip rewards redundancy the transformer cannot exploit.

## Principle 3 — eliminate implicit counters at the representation
Every encoding change that replaces "state implied by a counter run since some event" with an
explicit local parameter is a learnability win, independent of token budget: explicit durations kill
a latent counter; periodic/polynomial ramp *shapes* with explicit params move the per-frame counter
into the deterministic decoder, out of the prediction target. (v3: mixed-radix durations on
`FLD_NOTE_ON`, `SHAPE_POLY`/`SHAPE_PERIOD` ramps, settled end-of-frame values.)

## Principle 4 — ordering = topological order of the causal DAG
Ordering matters only under finite capacity + optimization + exposure bias; two training-free rules:
1. **If A causes B, emit A before B** — predicting an effect before its cause forces a high-entropy
   marginal. Derive ordering from the driver data-flow graph. (Open application: accompaniment
   before melody — [`lane_demux_hypothesis.md`](../landed/lane_demux_hypothesis.md).)
2. **Front-load determinants, but only low-entropy ones.** An early token must itself be highly
   determined. This is why absolute onset pitch ≈ 0 next-token while structure learns: high-entropy,
   no local determinant. Anchoring to a nearby reference (interval-from-previous — v3's `NI_*` lane)
   is the theory-prescribed fix; the absolute anchor stays ≈0 and must be scored distributionally.

## The training-free triage — `audit/learnability_triage.py`
Computed on the tokenized corpus, no transformer: **entropy-rate** h_k = H_{k+1} − H_k per token AND
per frame (cross-encoding comparison must be per-frame — token counts differ); **MI decay**
I(x_t; x_{t−d}) vs d (fat tail = a long-range counter the model will shortcut); **induction-copy
rate** (share of tokens completing a previously-seen bigram); alphabet/coverage context. Read: low
per-frame h_k + early plateau + fast MI decay + high copy ⇒ predicted learnable. **Measure at the
real block scale** (seq_len 8192, window mode) — whole-song mode over-credits cross-window reuse,
and smaller windows over-penalize codebooks (both measured failure modes).

**Track record (why the tool is trusted):** (1) at block scale it flipped the song-mode ordering and
predicted absolute-keyed codebooks don't pay in-window (copy 0.718 < baseline 0.852) — consistent
with the later full_macros content win; (2) on the generator-MDL encoding it returned a conditional
NO-GO (alphabet 3.7×, copy 0.916 < atomic 0.930 — exact residuals fragmenting the key) that a
canonical run would have cost a day to learn; the v3 event redesign followed its prescriptions
(small fixed alphabet, digit atoms, intervals, no fragmentable codebook keys). Numbers:
[`../landed/generator_mdl_representation.md`](../landed/generator_mdl_representation.md) + this
file's git history.

**Corollary: mini is not a research dimension** — it mode-collapses in training AND distorts the
static read via its window size. The triage gives the prodlike-scale read at mini cost; reserve
training runs for the collapse→learning threshold.

## Honest limit
Theory + these metrics give **sign, relative difficulty, and ordering** — not val_acc, and not the
scale threshold where collapse flips to learning (an emergence phenomenon current theory cannot pin
numerically). Block-entropy is undersampling-biased downward at high k — read the per-frame floor +
cross-config ordering, not absolute high-k values. Workflow: **theory + triage to design and rank →
one canonical confirmatory run.**

## References
Liu et al. 2023 (shortcuts to automata); Merrill & Sabharwal 2023 (log-precision ⊆ TC⁰); Hahn 2020;
Olsson et al. 2022 (induction heads); Crutchfield (computational mechanics); Bialek/Tishby
(predictive information).
