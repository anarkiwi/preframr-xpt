# Music-LLM landscape + cheap fail-fast experiment plan

**SUPERSEDED (2026-05-25):** this plan's pivot — the diffusion prodlike verdict
— came back FAIL, and the content win instead came **tokenizer-side**
(`full_macros` / FREQ_TRAJ). The model-side fail-fast arc is closed; kept as a
dated snapshot. Current state: `../AGENTS.md`.

**Status (2026-05-23):** strategic survey to leverage findings from
adjacent music-LM work, ranked alongside the preframr-specific cheap
probes we can run on CPU (no GPU contention with the in-flight
diffusion prodlike). Each entry has a fail-fast gate that lets us
spend ≤ 1 hr deciding whether the idea deserves a multi-hour or
multi-day commit.

## Where preframr is, after the content-head arc

Refuted at prodlike: `per_tier_heads_mos`, `per_tier_heads_entropy`
(v11+v12), `mask_structural_loss`, `cluster_conditional_content_head`.
Diffusion P2 mini hit val_acc gate + collapse PASS but
diversity_ratio 1.41 (FAIL strict 1.65 gate; +0.025 over baseline);
Phase 3 prodlike in flight on Branch C extrapolation.

The pattern across five model-side interventions: **the content tier
caps at ~13% eval_a acc** regardless of architecture. The bottleneck
keeps re-locating to "model can't disambiguate near-equivalent
content tokens from sparse data".

**Cheap probe (this snapshot, pre-FREQ_TRAJ):** 38.2% of the 7376 base
atoms appear <10 times in training. The worst content family (`SET
freq_lo voice 0`) has **65% of its 1926 atoms as long-tail**.

**Update (2026-05-26, post-FREQ_TRAJ, prodlike `full_macros`):** re-measured
from `tokens.csv`, the long tail is now **~11% of ~5491 atoms** and the
Unigram vocab is **100% used at tkvocab 8192** — FREQ_TRAJ + the vocab trim
largely closed the sparsity this snapshot flagged (and `full_macros` was the
content win). The composer-token fail-fast (#1) also resolves: prodlike has
**24** distinct composers (< the 50 floor → KILL); `prodlike_4x` has **100**
(viable). See `tokenization_vs_music_llms.md` for the current numbers.

## Cross-LLM landscape

| approach | preframr status | adjacent finding | applicability |
|---|---|---|---|
| Compound / grouped tokens (REMI+, CP-Words, Music X-Form) | drafted (`compound_token_tokenizer_design.md`, commit `731e0fc`) | Hsiao+ Yang ('21) and Music X-Form ('24) report 3-5× sequence reduction and improved long-range coherence | **HIGH** — directly attacks our sequence length + vocab-bottleneck problem |
| Pitch transposition augmentation | `transpose_xframe_2stage` REFUTED at mini ('26-05) | Universal in MIDI transformer literature (Music Transformer onward) | **MEDIUM** — refute was inconclusive at mini scale; revisit with tighter design |
| Composer / style condition tokens | not tried | MuseNet ('19) showed clean cross-composer separation from a single prepended id token | **HIGH** — cheap experiment; directly targets the cross-composer KPI |
| Audio-equivalence / acoustic normalization | drafted (`audio_equivalence_normalization_design.md`) | Not standard in music LLMs (most operate on MIDI which is already canonical) | **HIGH** — preframr-specific direction; sparsity probe says payoff is large |
| Multi-codebook hierarchical (MusicGen, EnCodec, AudioCraft) | per_tier_heads REFUTED (was wrong axis: tier ≠ codebook) | Meta's MusicGen uses 4 codebooks with delay pattern; reduces softmax bottleneck | **MEDIUM** — per_tier_heads tried this on the wrong axis; right axis might be (voice × time-quantization) not (structural/mid/content) |
| Relative attention (Music Transformer) | RoPE in use | Improved repetition modeling vs absolute | LOW — already done |
| Anticipation tokens (AMT, Stanford '24) | not tried | Lifts long-range coherence by pre-committing to event timing | LOW — SID is frame-locked; less applicable |
| Curriculum (data-then-model) | mini→canonical→prodlike progression | Universal | LOW — already done |
| Massive data scale (YuE, Suno-likes) | HVSC-bounded | Many problems vanish at 10×+ data | N/A — HVSC is the corpus |
| Multi-track structure-aware tokens (MMM, PopMAG) | `per_voice_aux_supervision` drafted | Multi-track modeling improves voice coherence | MEDIUM — relates to voice permutation augmentation |

## Fail-fast experiments (ranked by cost-vs-signal)

Each experiment has an **explicit fail-fast gate** that triggers
early termination if the idea is non-viable. The goal: spend ≤ 1 hr
deciding "abandon vs commit" for each.

### 1. **Composer-id style token** (LOW cost, HIGH signal) — 1 hr total

**Why:** MuseNet showed prepending a `[COMPOSER: X]` token cleanly
conditions the model on style. preframr's cross-composer KPI is the
project goal; this is the cheapest possible mechanism that targets
it directly.

**Mechanism:** add a special composer-id token at sequence start.
At parse time, look up the composer from the dump path (already done
by `composer_from_dump_path` in preframr-tokens). At training time
the model sees `[COMPOSER_N] [song tokens...]`. At inference, sample
with `[COMPOSER_N]` prefix for a specific style.

**Fail-fast gate (no training needed):**
- Count distinct composers in the prodlike corpus.
- If > 500 distinct composers, the per-composer mass is too low —
  KILL.
- If < 50, no separation pressure — KILL.
- If 50-500 (likely): proceed to a single-arm mini A/B (~1 hr train +
  ~15 min audit). Pass if eval_b cross-composer val_acc ≥
  baseline + 0.003 on ≥ 3 of 8 families.

**Time investment to "definitely useful or definitely not":** ~2 hr.

### 2. **Token-unigram pruning probe** (LOW cost, MEDIUM signal) — 30 min total

**Why:** if the long-tail content atoms (38% with <10 occurrences)
are dead weight, removing them won't hurt val_acc. If removing them
HURTS val_acc, they carry real information and audio normalization
must be careful not to lose it.

**Mechanism:** train mini with `--tkvocab 16384` (or smaller). The
existing tokenizer already truncates by count when vocab cap is
hit. Compare val_acc vs the baseline `--tkvocab 32768`.

**Fail-fast gate:**
- If val_acc drops ≥ 0.005 absolute: long-tail is real signal. Audio
  norm must be conservative (high K_or per register family).
- If val_acc unchanged or higher: long tail is noise. Audio norm
  is high-ceiling — be aggressive (low K_or).

**Time investment:** 1 train run (~15 min) + comparison vs existing
baselines (instant).

### 3. **Audio-equivalence Phase 0 calibration** (LOW cost, BLOCKS BIG IDEA) — 1 hr total

**Why:** the audio-norm design (`audio_equivalence_normalization_design.md`)
predicts ~30K → ~10K content vocab compression. If Phase 0
clustering doesn't separate per-register-family values into clean
acoustic equivalence classes, the whole direction is dead.

**Mechanism:** for each `(op, reg)` family, render all observed
values in a canonical SID context (re-use `preframr_audio.fingerprint`,
already validated on the cluster-head Phase 0b). Compute within /
between class fingerprint distance ratios per family.

**Fail-fast gate (per the design doc):**
- For families with `K_or > 8`: within/between distance ratio < 0.5.
  KILL if violated for > 30% of families with K_or > 8.
- Audition: 5 random class members from the 5 largest classes must
  sound indistinguishable to a human.

**Time investment:** ~30 min on fogbank + 30 min audition.

### 4. **Compound-token prototype** (MEDIUM cost, HIGH signal) — half day

**Why:** REMI+/CP-Words show 3-5× sequence reduction. preframr's
sequences are ~5K-10K tokens per song at mini — compression would
unblock both longer-context training and predict-host envelope.
Draft already exists at `compound_token_tokenizer_design.md`.

**Mechanism:** prototype the compound encoder on 100 dumps without
touching the main pipeline. Measure: (a) average compression ratio,
(b) resulting vocab size, (c) round-trip parse-decode integrity.

**Fail-fast gate:**
- (a) Compression < 3× → KILL (not worth the rewrite complexity).
- (b) Resulting vocab > 50K → KILL (defeats the densification goal).
- (c) < 95% round-trip → KILL (parser would lose information).

**Time investment:** ~4 hr prototype, no training needed for the
fail-fast decision.

### 5. **Voice permutation Phase 0 smoke** (LOW cost, sets up Phase 1) — 1 hr total

**Why:** voice permutation augmentation (`augment_voice_permutation.py`
just landed) needs a parser-admit + audio-non-silent check before
its Phase 1 mini A/B.

**Mechanism:** run on the 7 smoke-tier SIDs with all 5 non-identity
permutations. Verify (a) parser admits 100% of outputs, (b) renders
are non-silent, (c) spot-check 3 random renders sound musically
coherent.

**Fail-fast gate:**
- < 100% parser admission → KILL (data pipeline bug in the
  augmentation).
- Silent or DC-pinned audio → KILL (voice routing wrong).
- Audition fails → re-examine RES_FILT bit-permutation logic.

**Time investment:** ~30 min run + 30 min audition.

### 6. **Cross-engine generalization stratification** (LOW cost, MEDIUM signal) — 1 hr analysis

**Why:** eval_b is 8 cross-composer families. We've been treating
them as a single aggregate. Per-family content acc variance might
reveal whether failure is "model fails uniformly" (architecture
problem) or "model fails on specific engine families" (data
problem). Different remedies.

**Mechanism:** re-analyze the existing v10/v11/v12 audit JSONs.
Compute per-family content acc improvement from baseline. Identify
which families consistently lag (need targeted augmentation) vs
families that gain (the architecture works there).

**Fail-fast gate (analysis only, no kill threshold):**
- If 3+ families consistently lag (≤ baseline + 0.02): targeted
  data augmentation for those families is the cheap fix (synthesize
  more like-them training data).
- If improvements are uniform: the bottleneck is architectural;
  audio norm or compound tokens are correct directions.

**Time investment:** ~30 min analysis.

### 7. **Inference-only sampling sweep** (LOW cost, MEDIUM signal) — 2 hr

**Why:** all prompt-conditioning verdicts so far used T=0.5 fixed
sampling. Other strategies (top-k, top-p, beam search, nucleus) may
materially change diversity_ratio at fixed ckpt. If v11 ckpt + better
sampling clears the gate, the whole prodlike retest arc was
addressing the wrong layer.

**Mechanism:** take v11 (best content acc + borderline diversity)
ckpt. Re-run prompt_conditioning + loop_detection audits at:
- T=0.5 top-p=0.9 (current is greedy at T=0.5)
- T=0.7 top-p=0.95
- T=0.3 top-k=64

**Fail-fast gate:**
- Any sampling combo gives diversity_ratio > 1.2 on v11 ckpt: STOP
  ALL ARCHITECTURE WORK. Ship with that sampling. The whole arc
  has been chasing a sampling artifact.
- All combos remain ≤ 1.2: confirms the issue is the model not the
  sampler.

**Time investment:** ~2 hr (4 sampling configs × ~25 min for streams +
audit each).

## Recommended ordering for the diffusion-prodlike wait window

GPU is busy with diffusion prodlike for ~6-11 hr. CPU is idle.
fogbank (72 cores) is idle. Recommended:

1. **Composer-count probe (#1's fail-fast)** — 10 min, decides whether
   to commit to a 1-2 hr mini A/B for composer tokens. Pure CPU.
2. **Audio-equivalence Phase 0 calibration (#3)** — 1 hr fogbank +
   audition. Blocks the bigger audio-norm direction.
3. **Cross-engine stratification analysis (#6)** — 30 min, free of
   any compute. Reframes which direction matters most.
4. **Voice permutation smoke (#5)** — 1 hr. Unblocks the voice
   permutation Phase 1 A/B (drafted, ready to run when GPU frees).
5. **Compound-token prototype (#4)** — half day. Decides whether
   `compound_token_tokenizer_design.md` (your `731e0fc` draft) is
   worth promoting.

This sequence lands four kill-or-commit decisions before the
diffusion prodlike verdict comes in. By the time GPU is free, we
either have:
- diffusion PASS → ship and stack audio norm on top → biggest combined win
- diffusion FAIL → audio norm and/or compound tokens already
  validated as the next bet; no idle decision window

The inference-only sampling sweep (#7) is the cheapest experiment
that might invalidate the entire 2-week arc. **It's the strongest
fail-fast available** but needs the GPU. Queue it for the first
gap after diffusion prodlike finishes; ~2 hr of GPU on the
existing v11 ckpt could rewrite the project's framing entirely.

## References

- Compound tokens: Hsiao+Yang '21 (REMI+); Liu+ '24 (Music X-Form).
- Anticipation: Thickstun+ '24 (Anticipatory Music Transformer).
- Multi-codebook: Copet+ '23 (MusicGen / EnCodec).
- Composer tokens: OpenAI MuseNet '19 blog.
- Multi-track: Ens+Pasquier '20 (MMM); Ren+ '20 (PopMAG).
- Relative attention: Huang+ '18 (Music Transformer).

Internal references:
- Sparsity probe data (this doc, "Where preframr is" section).
- Refuted entries: `preframr-xpt:refuted/per_tier_heads_*.md`,
  `mask_structural_loss.md`.
- Sibling drafts:
  `compound_token_tokenizer_design.md` (commit `731e0fc`),
  `audio_equivalence_normalization_design.md`,
  `preframr-aug:design/melody_transfer_augmentation_design.md` (+ voice permutation
  variant).
