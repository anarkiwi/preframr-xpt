# Speculative / parallel encoding pipeline — claims + arbitration (proposal)

**Status:** Proposal (architecture). Motivated by the RESID=0 program. Not built.

**Learnability framing.** Make the Objective's "learnability" term concrete via [`learnability_token_ordering_theory.md`](../references/learnability_token_ordering_theory.md): score a claim by its per-frame entropy-rate contribution + induction-copy reuse and penalize implicit-counter structure (`audit/learnability_triage.py` deltas) — so a higher-learnability cover is measured, not asserted.

## The problem: strict-order destructive passes

Today the tokenizer is a **strict-order chain of destructive passes**: each `MacroPass.apply(df)`
*mutates* the df (consumes register writes, splices replacement tokens) and hands it to the next. Two
failures follow:

1. **One pass destroys information another needs.** `SkeletonPass` consumes the freq writes into
   `SKEL`/`ORN`/`RESID`; by the time drum-stamp or patch detection runs, the raw freq is gone (only
   semitone offsets remain) and the note has been segmented one particular way. **Evidence in this
   very investigation:** the drum/patch probes only work because I added an inert `_df_sink` to read
   the *pre-pass* df — a workaround for exactly this destruction.
2. **Order forces premature, global choices.** Drums currently fall to `RESID` because `SkeletonPass`
   tries to fit them as melodic notes *first*. A drum pass would fit them better — but strict order
   makes you choose who runs first, and either order starves the other (drum-first steals melodic
   notes; skeleton-first mangles drums). And the *same* chain is applied to every tune, though a
   drum-heavy tune, a legato lead tune, and a filter-sweep tune each want a different encoding.

The goal: **let passes propose competing encodings non-destructively, then pick the most appropriate
one per tune/region** — without any pass being able to delete what another could have used.

## The concept: passes PROPOSE claims; an arbiter SELECTS a lossless cover

- **Immutable source.** The parsed register log is read-only for the whole encode. Nothing is mutated
  in place.
- **A pass is a PROPOSER, not a mutator.** It reads the source and emits **Claims**. A `Claim` =
  - `writes`: the set of source rows (register writes) it would consume,
  - `tokens`: the replacement token rows it would emit,
  - `score`: `(fidelity, learnability, budget)` — see Objective.
  Passes read only the immutable source, so they have **no ordering dependency and can run in
  parallel** (the `_df_sink` need disappears: every pass sees the raw writes).
- **Speculation = overlapping/alternative claims.** Several claims may cover the *same* writes — from
  different passes (drum-stamp vs skeleton-note vs RESID for one span) or the same pass offering
  alternatives (a wide ramp as `SLIDE` vs `SWEEP` vs `RESID`). These are *speculative*: only the
  winner is kept.
- **Arbiter selects a lossless PARTITION.** The arbiter chooses an accepted subset of claims that
  (a) **covers every source write exactly once** (lossless — guaranteed because a trivial raw/`RESID`
  claim always exists as the fallback), and (b) **maximizes the global objective**. It resolves
  overlaps by score; losers are discarded. Output = the assembled token stream.
- **Per-tune / per-region selection.** Some claims are mode-level ("encode this voice with the stamp
  codebook", "this tune uses the patch bank"). The arbiter picks the **encoding that scores best for
  THIS tune** — different tunes get different encodings, by selection rather than a fixed chain.

This generalizes what `SkeletonPass` already half-does: `drop_idx` (claimed writes) + `new_rows`
(replacement tokens) **is** a Claim — just applied destructively and first-come. Lift it to a returned,
scored, arbitrated Claim and the architecture falls out.

## The objective (how "most appropriate" is decided)

Lexicographic per region, then summed:
1. **Fidelity (hard gate).** The accepted cover must reproduce the source **byte-exact** (the existing
   per-frame oracle + emulator round-trip). `RESID`/raw is the always-valid, lowest-fidelity-score
   fallback that guarantees coverage. *Lossy* claims (audition-gated, P8) are allowed only with an
   explicit fidelity penalty and an audition flag.
2. **Learnability.** Reward low-entropy, transferable structure: codebook **reuse** (a stamp/patch
   referenced N times), **rhythmic-grid** regularity, **Unigram-compressibility** (claims laid out as
   clusterable atoms), separability/no-multiplexing (`encoding_principles`). This is what makes a drum
   stamp beat a per-frame RESID even when both are lossless.
3. **Token budget.** Tie-break on fewer tokens.

So a drum span: the stamp claim (lossless, high reuse, gridded) outscores the skeleton-note claim
(would RESID — low fidelity-after-fit) and raw RESID (lossless but zero learnability) → the arbiter
picks the stamp. A legato lead: the skeleton+ornament claim outscores a spurious stamp (no reuse).
No global order decided it — the **score per region** did.

## Arbiter mechanics

- **Coverage constraint:** the accepted claims must partition the source writes (each write in exactly
  one). Model as weighted exact-cover / interval scheduling over write-sets.
- **Tractable resolution:** writes are naturally grouped per voice × time-span, and claims rarely
  overlap arbitrarily — so **greedy-by-score with a lossless backstop** (accept highest-scoring
  non-conflicting claims; uncovered writes → raw/RESID) is a sound first cut; refine to per-voice DP
  (weighted interval scheduling) where claims nest.
- **Determinism:** stable score + tie-break (pass priority, then source order) → reproducible encoding
  (required for the deterministic test suite).
- **Speculation budget:** passes may emit alternatives; cap per region to bound arbiter cost.

## Why this serves RESID=0 and generalization

- **Nothing is destroyed**, so every write gets considered by *every* primitive — the skeleton can't
  pre-empt a drum, a drum can't steal a lead. RESID becomes the *true* floor (what no proposer could
  fit), not an artifact of pass order.
- **The best encoding wins per region**, and **per tune** — directly "pick the most appropriate
  encoding for a tune".
- It's the natural home for the parallel primitives this program is producing: skeleton+ornament
  (pitch), stamp codebook (percussion), patch bank + mutations (timbre), sweep/slide, held-ARP — all
  become competing/cooperating proposers instead of a brittle ordered chain.

## Migration path (incremental, low-risk)

1. **Introduce `Claim` + an arbiter that reproduces today's behaviour** — wrap each existing pass so it
   returns `(claimed_writes, new_rows, score)`; arbiter applies them in the current priority order
   (greedy, no speculation). Byte-identical output → safe refactor, gated by the deterministic suite.
2. **Make passes read the immutable source** (not the running df) — removes ordering deps; delete the
   `_df_sink` workaround.
3. **Add competing claims** for the known conflicts first: drum-stamp vs skeleton-note vs RESID; let
   the arbiter pick by score. Re-trace RESID.
4. **Add per-tune mode selection** (stamp-codebook on/off, patch-bank on/off) scored globally.
5. **Add speculative alternatives** within passes (SLIDE/SWEEP/RESID for wide ramps).

## Open questions

- **Scoring calibration** — the learnability term must be validated against real tunes (same
  discipline as the plausibility judge): a higher-"learnability" cover must not hurt the per-frame
  oracle or the audition.
- **Arbiter complexity vs optimality** — greedy vs DP vs ILP; where do real claim overlaps actually
  nest (probably per-voice intervals → DP suffices)?
- **Cross-region coupling** — patch `PATCH_SET`/mutations and the global filter span regions; the
  arbiter must handle claims with non-local state (active patch, held filter) — likely a two-level
  arbiter (per-note claims under per-voice/global mode claims).
- **Lossy claims** — how the audition-gated lossy tier (P8) enters scoring without letting lossy beat
  lossless silently (explicit penalty + audition flag).
- **Determinism vs search** — keep the arbiter deterministic; speculation is bounded enumeration, not
  stochastic search.
