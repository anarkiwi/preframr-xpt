**Status:** Reference + prototype tool shipped (2026-06-03) — training-free triage, `audit/learnability_triage.py`. Predicts sign/ordering; the scale-threshold still needs one canonical run.

# A theoretical basis for token + ordering design (predict before you A/B)

The all-empirical loop (mini A/B per encoding) cannot see the thing it is testing: mini
**mode-collapses regardless** of vocab (`loop_collapse_rate` ~1.0), so the collapse→learning
transition only appears at canonical/prodlike. Yet we keep spending mini runs to pick *direction*
(does this macro help?), which theory can answer for free. This doc separates the two: **theory +
training-free measurement decide direction and ordering; one canonical run decides the threshold.**

## The load-bearing fact: the architecture is already exonerated
`framework_arch_test` — torchtune llama3_2, mini — reaches **val 0.903 on UNSEEN synthetic motifs**
(`design/landed/substrate_ablation_v1.md`). The model class generalizes when the structure is clean.
So the SID failure is **not capacity** — it is a mismatch between the *encoding* and what a bounded
transformer can cheaply represent. That mismatch is computable from the encoder + driver model
**without training**.

## Principle 1 — a transformer is a bounded automaton, not a recurrence
A constant-depth, log-precision softmax transformer sits at ~**TC⁰** (Merrill & Sabharwal 2023). It
**cannot maintain an unbounded sequential counter** and generalize over length. Liu et al. ("Transformers
Learn Shortcuts to Automata", ICLR'23) sharpen it: a transformer *can represent* a finite automaton, but
SGD finds a **shortcut** that fits the training horizon and **fails to extrapolate**. Mini mode-collapse
is consistent with shortcut-fitting a driver automaton it cannot actually simulate.

A SID driver **is** a finite-state machine: per-voice wavetable pointer, envelope phase, arp index,
pulse/filter ramp counters. So the learnability of an encoding reduces to two quantities, both derivable
from the encoder logic + the driver model we already have:

1. **Causal-state size** — how big is the latent the model must reconstruct to predict the next token
   (Crutchfield computational-mechanics *causal state* / statistical complexity C_μ).
2. **Dependency horizon** — how far back live the tokens that *determine* this token's value.

Minimize both and you are back in the regime the synthetic motifs already proved learnable.

## Principle 2 — prefer induction-head-expressible structure
Induction heads (copy "what followed this token last time") are the **first, most reliable** circuit a
transformer forms (Olsson et al. 2022). So **DEF/REF codebooks** (wavetable / stamp / patch, and the
PR2 ctrl codebook) are *provably* in the easy regime: REF→DEF is a single induction-head copy. An
**implicit per-frame arp/ramp increment is not** — it needs a maintained counter (Principle 1). The
right notion of "compressible" here is **copy-fraction**, NOT gzip — gzip rewards redundancy the
transformer cannot exploit, copy-fraction rewards the redundancy it *can*.

## Principle 3 — the residual-SET program is a learnability program
Each residual-elimination PR removes an implicit counter or exposes hidden state locally — a learnability
win, not just a token-budget win. The volume ranking happens to match the learnability ranking:
- **Note duration (PR3 Option A)** replaces "gate-off implied by a counter run since onset" with an
  explicit scalar → kills a latent counter. **Predicted strictly more learnable than Option B** (B still
  makes the model infer *when* the off lands). Act on this without an A/B.
- **Codebook DEF/REF (PR2)** → induction-head copy (Principle 2).
- **INIT snapshot / DEF-on-first (PR5/PR6)** → front-loads the cold-start latent instead of inferring it.

This is a stronger motivation for census-to-zero than compression, and it ranks the PRs.

## Principle 4 — ordering = topological order of the causal DAG
In the infinite-capacity limit the AR chain rule is order-invariant; ordering only matters under finite
capacity + optimization + exposure bias. Two training-free rules:
1. **If A causes B, emit A before B.** Predicting an effect before its cause forces a higher-entropy
   marginal that will not generalize. Derive the ordering from the driver data-flow graph.
   `voice_canonical_block_order` and "**FRAME header carries voice order + write counts**" *are* this — a
   header that front-loads structural determinants collapses every downstream conditional entropy.
2. **Front-load determinants, but only low-entropy ones.** Teacher-forcing compounds errors, so an early
   token must itself be highly determined (a write-count header is; an absolute onset pitch is not). This
   is exactly why **V0 onset ≈ 0** while trajectory *structure* learns: onset pitch is high-entropy with
   no local determinant, so it both fails to learn and derails what follows. Anchoring it to a nearby
   reference (interval-from-previous) is the theory-prescribed fix
   (see `melody_data_gap_ladder.md`, `melody_predictability.py`).

## The training-free triage (substitutes for direction-finding A/Bs)
Compute on the tokenized corpus, no transformer — `audit/learnability_triage.py`:
- **Entropy-rate vs memory** h_k = H_{k+1} − H_k (bits/token) for k=0..K. The *floor* is the achievable
  next-token CE loss; the *k where h_k plateaus* is the effective memory (the dependency-horizon proxy,
  i.e. an empirical stand-in for causal-state size). Miller–Madow corrected; report per-token AND
  **per-frame** (h_k·N/F) so the cross-encoding comparison is not confounded by sequence length (a
  compressing encoding has fewer tokens that each carry more — total tune information is ~constant).
- **MI decay** I(x_t ; x_{t−d}) vs d. Concentrated at small d = good; a fat tail = a long-range counter
  the model will shortcut.
- **Induction-copy rate** — share of tokens completing a seen bigram (∃ j<t: x_{j−1}=x_{t−1} ∧ x_j=x_t):
  the induction-head-able fraction. Plus **novel rate** (first occurrence in a window).
- **Alphabet size + token/frame counts** — context.

Read: a candidate with low per-frame h_k, an early h_k plateau, fast MI decay, and high induction-copy
is **predicted learnable**; a fat MI tail + low copy-fraction is **predicted to collapse**. This ranks
designs and prunes the experiment set. Exact determinant-distance (from encoder instrumentation) is a
follow-up; the h_k-plateau + MI-decay are its empirical proxies for now.

## First read (prototype, 9 player-diverse non-digi tunes, tokens 0.42.1)
`learnability_triage.py --configs baseline,full_macros,codebook` (codebook = base + skeleton/stamp/
sweep/pw/filter/wavetable/patch/held_arp; `ctrl_osc`/`note_off` not yet shipped at 0.42.1 → auto-dropped):

**`--mode song` (full-song parse, 9/9 tunes, robust):**

| config | alphabet | tok/frame | h∞/frame | h∞/token | induction-copy | MI@1 | MI@16 |
|---|---|---|---|---|---|---|---|
| baseline | 727 | 5.88 | 4.93 | 0.839 | 0.975 | 3.18 | 1.46 |
| full_macros | 1678 | 5.33 | 3.83 | 0.718 | 0.960 | 4.32 | 2.40 |
| codebook | 1781 | 4.78 | **3.65** | 0.762 | 0.955 | 3.84 | 1.91 |

Both macro arms cut per-frame information ~22–26% vs baseline, and here **codebook edges full_macros**.
**But song mode is the wrong stream:** the model trains/predicts on **self-contained blocks**, not whole
songs, and song mode lets a codebook's REFs accumulate over the entire tune — over-crediting its
compression (see `macro_learnability_risk_review.md`).

**`--mode blocks` at `seq_len=8192` (the real prodlike/predict block scale; EXPERIMENTAL, 5/9 tunes — the
standalone block re-encode trips on ops needing parser context, faithful version needs the Corpus
block-builder):**

| config | alphabet | tok/frame | h∞/frame | h∞/token | induction-copy | MI@1 | MI@16 |
|---|---|---|---|---|---|---|---|
| baseline | 567 | 3.67 | 3.03 | 0.825 | 0.942 | 3.52 | 1.58 |
| full_macros | 1073 | 0.96 | **0.38** | **0.400** | 0.852 | 5.88 | 4.05 |
| codebook | 1365 | 0.95 | 0.51 | 0.535 | **0.718** | 5.34 | 3.03 |

**At block scale the ordering FLIPS vs song mode: full_macros beats codebook on every metric** (h∞/frame
0.38 < 0.51, h∞/token 0.400 < 0.535), and the codebook arm's **in-window induction-copy is lower (0.718 vs
0.852)** with the largest alphabet (1365). Reading: the heavy codebook passes (stamp/wavetable/patch) must
recur *within one block* to pay off, and many don't — so they add vocabulary without buying back reuse.
**The codebook compression does not survive to the scale the model learns at.** Decision-relevant — it
cautions against the codebook pipeline as a *learnability* bet, consistent with the confirmed `full_macros`
content win (eval_a 0.219→0.324) and the codebooks remaining experimental.

**Scale matters — use the real seq_len.** This was first run at `seq_len=4096` (mini): the smaller window
*over-penalised* codebooks (copy 0.683, h∞/frame 0.55) because programs had less room to recur in-window.
At the correct `seq_len=8192` the gap narrows (copy 0.718, h∞/frame 0.51) **but full_macros still wins** —
the conclusion is robust, only the mini magnitude was an artifact. The triage default is now 8192.
**Corollary (a finding in itself): mini is not a useful research dimension** — it mode-collapses in
*training* (loop_collapse_rate ~1.0) AND distorts the *static* read via its window size. The resolution is
NOT "always train prodlike" (6–11 hr): the triage gives the prodlike-*scale* learnability read at
mini-*cost* (static analysis, minutes) — reserve training runs for the collapse→learning *threshold* only.
Caveats on these numbers: 5/9 tunes (biased simpler), absolute pre-voice-reg blocks, high-k undersampling
— trust the DIRECTION; **certify on full coverage via the Corpus block-builder before acting**. (The earlier "codebook edges full_macros, re-run the residual arm to push below 3.65"
claim was a song-mode artifact — withdrawn.)

## Honest limit
Theory + these metrics give **sign, relative difficulty, and ordering** — not val_acc, and **not the
scale threshold** where collapse flips to learning (an emergence phenomenon current DL theory cannot pin
numerically; "mini collapses, canonical settles it" is precisely this). Also: block-entropy h_k is
**undersampling-biased downward at higher k** (a 9-tune sample badly undersamples 4-grams over a ~1700
alphabet), so read the *per-frame floor + the cross-config ordering*, not absolute high-k values;
corpus-scale re-run tightens them. The ordering is trustworthy because the bias hits all configs similarly
— and it already agrees with the measured result. Workflow: **theory + triage to design and rank → one
canonical confirmatory run.** Keep the experiment for the *threshold*; stop spending it on *direction*.

## Cross-links
`macro_learnability_triage.md` (per-pass keep/retire triage — this doc supplies its information-theoretic
backing + a whole-stream tool), `melody_data_gap_ladder.md` (conditional-predictability gap on the melody
line), `audit/melody_predictability.py` (the same math scoped to V0 onsets),
`workorder_residual_set_elimination.md` / `impl_residual_set_elimination.md` (the PRs Principle 3 ranks).

## References
Liu et al. 2023 (shortcuts to automata); Merrill & Sabharwal 2023 (log-precision ⊆ TC⁰); Hahn 2020
(attention limitations); Olsson et al. 2022 (induction heads); Crutchfield (computational mechanics /
ε-machines, causal states, C_μ); Bialek/Tishby (predictive information / excess entropy).
