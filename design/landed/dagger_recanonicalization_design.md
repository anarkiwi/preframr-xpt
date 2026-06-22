# Tier-4 — DAgger / scheduled-sampling on re-canonicalised rollouts (attacks exposure bias, M1)

**SUPERSEDED (2026-06-20):** an event-codec exposure-bias remedy mooted by the representation-level fix
(step/tracker codec). In the model-side anti-queue (`../refuted/multi_modal_objective_design.md`).

**Status:** Design (2026-06-16). Opened because **Tier-3 augmentation RAN and did not fix free-running**
(`aug_ab_evalb_results.md`): corpus changes lifted teacher-forced metrics (instrument transplant
novel-content +26%) but free-running content acc stayed flat ~0.05–0.07 at every dose. That is the
exposure-bias (M1) signature — the binding constraint is the **train↔inference mismatch**, not the data
distribution. Tier 4 is the model-side axis the [remediation
ladder](free_running_pathology_remediation_design.md) gated last; it is now the live arc.
**TRIAGE RAN (2026-06-16) — objective 1 NOT supported; lean off-ramp or objective 2.** The
prior-state-aware recanon shipped (preframr-tokens #84/#85: `seed_keyframe`/`decode_windowed`/
`writes_to_ordered`; verified — windowed eval-B states recover ~729 frames vs 0). With it, the
recoverability triage finally ran on real eval-B rollouts (`/scratch/tmp/recover_triage.py`,
instrument_full ckpt). **Verdict: re-canonicalising the drifted state does NOT restore recoverability.**
On *matched* prompts (both raw `g` and recanon `ĝ` continuations decode) the continuation empty-frame
fraction is a **wash** (mixed better/worse, ~split); the apparent aggregate "improvement" is a confound
(ĝ-continuations are often undecodable, so the means compare non-matched subsets). Crucially
**`recanon_delta ≈ 1.0`** — the model emits ~99% *non-canonical* atom surface (grammar-valid but not the
canonical order it trained on), so re-prompting from the canonical `ĝ` is itself **off-distribution** and
frequently breaks the continuation. So scheduled-sampling + re-prompt-from-recanon (objective 1) has a
weak premise here. **Redirect:** either **objective 2 (consistency / fixed-point** — train the model so
its *own* output is canonical, i.e. minimise the delta; needs no off-distribution re-prompt and is
motivated by the delta≈1.0 finding) or the **off-ramp** (usable generation via constrained decode +
Tier-1 caps — already works). Given Tier-3 (M1) + this triage, the off-ramp is the highest-confidence
ship; objective 2 is the one remaining model-side bet worth a cheap probe before committing.

**OBJECTIVE 2 DE-RISKED DEAD (2026-06-16).** The cheap training-free check ran (n=28, instrument_full,
`/scratch/tmp/delta_drone_corr.py`): per free-running rollout, recanon-delta vs empty-frame fraction.
**Both premises fail.** (a) delta is saturated at mean 0.972 / std 0.016 — the model emits ~97%
non-canonical surface CONSTANTLY, whether droning (empty 0.99) or making music (empty 0.03); corr(delta,
empty)=+0.21 is noise on a no-variance predictor → non-canonicality is NOT a degeneration signal.
(b) recanon preserves the empty-frame fraction (mean|Δ|=0.021) — `recanon(drone)` IS a drone, because
recanon canonicalises surface ORDER, not CONTENT, so the fixed-point target cannot teach un-droning.
**Conclusion: the model-side recanon lever is exhausted.** The drone is a CONTENT failure under
self-conditioning that no surface-level objective touches; Tier-3 (data) was flat, DAgger objective-1
unsupported, objective-2 dead. Remaining handles on the drone: the decode-time frame-budget cap
(cosmetic) or not generating open-ended (phrase-prompting). The recanon oracle (#84/#85) stands as a
useful codec capability regardless.

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

**UPDATE (2026-06-16): recanon built for the continuous case; the windowed-prior-state case is the
real blocker.** `events.generate.recanon` now exists (preframr-tokens #84): on a **continuous
keyframe-free** stream it is identity / idempotent / write-preserving (5 green unit tests; real DRAX
×12 content-exact, identity 9/12 — the 3 are a leading-rest frame-base off-by-one). **But running the
recoverability triage exposed the critical gap:** real rollouts are prompted from **windowed eval-B
blocks whose leading `[KEYFRAME …]` carries prior state**, and `strip_keyframes` + `decode` cannot
restore that state — so recanon of a windowed rollout is ~100% different (delta ≈ 1.0) and won't
re-prompt. **The DAgger oracle therefore needs a *prior-state-aware* recanon: a decode that CONSUMES the
leading keyframe to seed register state, then canonicalises the body.** That (not the trainer change,
not the continuous-case recanon already shipped) is the binding P0. Until it lands, the triage can only
run on continuous-from-t=0 (whole-tune-prefix) prompts.

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
