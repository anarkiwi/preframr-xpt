# Tier-4 — DAgger / scheduled-sampling on re-canonicalised rollouts (attacks exposure bias, M1)

**Status:** Design (2026-06-16). Opened because **Tier-3 augmentation RAN and did not fix free-running**
(`aug_ab_evalb_results.md`): corpus changes lifted teacher-forced metrics (instrument transplant
novel-content +26%) but free-running content acc stayed flat ~0.05–0.07 at every dose. That is the
exposure-bias (M1) signature — the binding constraint is the **train↔inference mismatch**, not the data
distribution. Tier 4 is the model-side axis the [remediation
ladder](free_running_pathology_remediation_design.md) gated last; it is now the live arc.
**Next concrete action = the P0 prerequisite below** (a verified re-canonicalisation function): an
attempt to run the recoverability triage proved the naive recanon composition reframes the stream
(delta ≈ 0.99) and does not re-prompt — there is no working oracle yet, so that is the first build.

## The precise problem (and the hard part, stated up front)

The model is trained **teacher-forced** (next-token CE on ground-truth windows) and deployed
**self-conditioned** (its own tokens feed back). It never learns to recover from its own errors, so once
it leaves the prompt it drifts to the empty-frame drone. The classic fix is on-policy training (train on
states the *policy* visits). **The hard part: there is no ground-truth continuation for a self-generated
state — it is novel music — so the loss target is the open question.** Every naive exposure-bias fix
here founders on that, and on a SID-specific failure mode: a raw model rollout drifts into
*grammatically invalid* / empty-frame garbage (the audition showed this), so blind scheduled sampling
would train recovery from states the model will never actually be scored on.

## The asset that makes this tractable: the exact re-canonicalisation oracle

Two existing, exact operations turn a rollout into a **valid, in-distribution, plausible** state instead
of garbage:

1. **Constrained decode** (`preframr_tokens/events/constrained.py:EventStreamState`, used by the
   audition gate) — a grammar mask so every sampled token keeps the stream **decodable**. Rollouts are
   never grammatically broken.
2. **Re-canonicalisation** — `decode(g)` → ordered writes → `stream.canonical_writes` → `stream.encode`
   maps a decodable rollout `g` to the canonical atom stream `ĝ` for the *same* SID register state
   (settled freq/PW first per voice, globals last, same-value rewrites dropped, no NOTE OFF). It is a
   deterministic projection onto the canonical manifold — **the exact surface form the training data is
   in**. `encode(verify=True)` self-checks it.

Together: on-policy states that are valid SID states in the training surface form — the *plausible
mistakes* free-running must recover from, not token garbage. This projector is the thing blind scheduled
sampling lacks, and the reason this bet is worth funding where the refuted RL/energy bets were not.

## Anti-queue alignment (why this is a NEW axis, not a re-run)

The umbrella refutation killed **content-distribution-objective** bets (per-tier/MoS heads, InfoNCE,
discrete-diffusion/cluster content heads, class-weighted CE) and **sequence-reward ranking** (DPO/energy
on render quality). This is **neither**: no reward model, no preference ranking, no content-head
surgery. It is on-policy exposure-bias correction with an **exact state projector** — the "fresh
decoding-time story" the anti-queue explicitly left open against Approach-D. Keep it that way: **no
render-fingerprint reward, no DPO** (that re-opens the refuted axis).

## Objectives, by soundness (build the first; the rest only if it stalls)

1. **Scheduled sampling + re-canonicalisation (LEAD).** During training, with annealed probability
   `p_t` (0 → p_max), replace some teacher-forced input tokens with the model's own
   constrained-decoded + re-canonicalised tokens; keep CE loss vs the tune's ground-truth continuation.
   Honest approximation: once the rollout diverges, the real continuation only approximately fits —
   mitigated by **short rollout horizons** (diverge little) and re-canon (stay on-manifold). The
   re-canon is the SID-specific safety ingredient that turns the textbook method from "feed garbage" to
   "feed plausible mistakes."
2. **Consistency / fixed-point self-distillation (NO ground-truth target).** Minimise the divergence
   between the model's distribution on a **raw** rollout prefix and on its **re-canonicalised** form
   (`KL(p(·|g) ‖ p(·|ĝ))`), and/or push rollouts toward being canonical fixed points (`ĝ ≈ g`). Drives
   on-manifold self-consistency without needing a continuation target — sidesteps the hard part
   entirely. Weaker theoretical tie to "musical quality," but immune to the target-mismatch objection.
3. **Excluded:** reward/preference ranking on render fingerprint (anti-queue-refuted); beam search
   (worsens open-ended degeneration, busts the Orin envelope).

## P0 PREREQUISITE (found 2026-06-16 by attempting the triage): a VERIFIED re-canonicalisation function

The triage build surfaced that **the re-canonicalisation oracle this whole tier leans on does not yet
exist as a working function.** The naive composition
`block_to_ids(ordered_writes(writes_to_dump_df(ids_to_writes(g))))` is wrong:

- It **reframes the stream** — recanon vs rollout atom-delta ≈ **0.99** (almost nothing preserved), and
  the result **does not re-prompt** (continuations from `ĝ` fail to decode). The round-trip through
  `writes_to_dump_df` → `ordered_writes` does not reconstruct the original `OrderedWrites` (frame/clock
  assignment + the KEYFRAME conditioning segment differ).
- It is *idempotent* on the few stored blocks that decode (13/14) — so the projector **concept** is
  sound — but idempotent onto the WRONG fixed point (a reframed stream), which is useless for training.
- Stored `.blocks.npy` rows decode-fail ~89% as-is (they need frame-trimming, like the audition's
  `decode_tolerant`); fixed-length rollouts end mid-event and also need trimming before decode.

**So Tier-4's real first task is tokens-side, not trainer-side:** build a content-preserving
re-canonicalisation `recanon(atoms) -> atoms` in `preframr_tokens/events/` with a hard test suite:
(1) **round-trip identity** — `recanon(block) == block` for every real corpus block (after frame-trim);
(2) **idempotency** onto the canonical form; (3) **re-promptable** — `EventConstraint` primes cleanly
from `ĝ` and a continuation decodes; (4) preserves the decoded register writes
(`decode(recanon(g)) == decode(g)` up to canonicalisation). This is small and well-scoped, and **must be
green before the triage below or any DAgger build** — without it there is no oracle to roll out against.

## Triage FIRST (training-free; gates the build, per project discipline)

Before touching the trainer, validate the premise on the existing `v2_atoms_baseline.ckpt` (and the
`instrument_full` ckpt): free-run K tokens (constrained-decoded) from real eval-B prompts, then for each
rolled-out state measure **teacher-forced next-token accuracy continuing from the RE-CANONICALISED state
`ĝ` vs from the RAW drifted state `g`**, scored against the tune's continuation.

- **If recanon states are meaningfully more recoverable** (higher next-token acc, lower entropy) → the
  projector buys real signal → build objective 1.
- **If flat** → re-canon doesn't restore recoverability (drift is semantic, not surface) → objective 1's
  premise is weak; go to objective 2 (consistency needs no recoverability claim) or reconsider Tier 4.

This is the cheap analog of `learnability_triage` / the Tier-2 gate — do not build objective 1 ahead of
this read.

## Build (if triage passes)

- A `--scheduled-sampling` path around `preframr/train/model/lightning.py:training_step`. **KV-cache
  caveat** (the doc's own warning): training is a parallel teacher-forced forward with **no** KV-cache;
  in-step sequential rollout fights that. So prefer **offline DAgger-classic aggregation**: between
  epochs, generate a batch of constrained-decoded + re-canonicalised rollouts from train prompts, append
  the `(recanon-prefix → continue)` windows to the dataset, retrain. Cheaper, KV-cache-friendly, and the
  truest DAgger form. In-step two-pass scheduled sampling is the fallback if aggregation underperforms.
- Reuse the audition machinery (`event_gate`/`predict` constrained decode) for the rollout; reuse
  `events.generate` + `stream.canonical_writes` for the projection; `encode(verify=True)` as the guard.

## Gates (same yardstick as the A/B that sent us here)

- **Decision metric: `free_running_gap` free-run content acc must RISE above the baseline 0.062 floor**
  (the exact metric Tier-3 was flat on; `data/audit/aug_ab_evalb_results.md`). This is THE read — not
  teacher-forced val_acc / copy_novel, which Tier-3 proved misleading (they rose while free-running did
  not).
- Plus the [generation quality gate](generation_quality_gate.md) render audition (does it sound less
  drone-y?) and **no-regression on teacher-forced eval-B content**. Same canonical tier, atoms-only,
  ep100, vs the existing baseline; one A/B (baseline vs scheduled-sampling/aggregation).

## Honest priors and the off-ramp

Model-side bets here have a losing record, and scheduled sampling is known-unstable — but this is the
**first** attempt on the exposure-bias axis (genuinely unexplored, per the anti-queue) and the exact
re-canonicalisation oracle is a real SID-specific lever the generic method lacks. **If both the cheap
triage and a scheduled-sampling A/B come back flat on `free_running_gap`, the conclusion is structural:**
absolute-onset multimodality (P6 caps a perfect n-gram at ~0.51) bounds self-conditioned generation, and
the realistic deliverable is **usable-but-imperfect generation via constrained decode + Tier-1 caps**
(forbid the empty-frame drone, re-anchor on drift) rather than a "fix." That off-ramp is itself a
shippable outcome — the audition path already works (preframr #168) — just not the open-ended one.
