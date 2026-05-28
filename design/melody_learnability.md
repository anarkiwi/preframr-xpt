# Melody learnability — the open content frontier

**Status:** ACTIVE research arc. The program's goal is content/melodic generalization. After
the first confirmed content win (`full_macros`) turned out to be **SET scaffolding, not melody**
(FREQ_TRAJ ~0.026 acc; the model rarely emits FREQ_TRAJ at all), four landed encoding/loss
features attack the melody question and converged probes localise the blocker.

## Converged diagnosis (2026-05-27 → 2026-05-28)
Melody onset prediction is **NOT aleatoric, NOT rare, NOT (only) a representation defect** —
it is the genuinely **hard cross-song prediction** that mini lacks scale to learn, while the
model freely learns easier (regular) content.

Evidence chain:
- **Predictability ceiling:** a trigram on the anchored onset line predicts the next note at
  **0.79–0.82** (cond. entropy k=2 ≈ 2.2 bits, matches the design's 2.68b for gate-anchored
  Hubbard bass). Not aleatoric — `audit.melody_predictability`.
- **Capacity is there:** removing PW+filter timbral noise (`freq_core_ablation_mini`) lifted
  **op0 SET acc 0.078→0.419 (~5×)** while V0-onset stayed 0 — the model has ample mini
  capacity, it just doesn't spend it on the hard onset.
- **Melody is ~13.4% of the stream (not rare)**, fragmented across **three** freq ops:
  op0 SET on freq regs 12.1% (acc 0.0013) + op45 V0-onset 0.9% (0.000) + op47 FREQ_NUDGE
  pitch 0.4% (0.000) — comparable to ops the model learns fine.
- **Five mini A/Bs flat:** full_macros / anchored / anchored+interval / freq_core (PW+filter
  off) / onset_loss_weight (W=10) — **V0-onset = 0.000 every time**. Only scale moves it
  (prodlike absolute op45 = 0.067).
- **Loss-weight nudge:** `--onset-loss-weight 10` moved V0-onset 0.000→0.002 at zero all-tier
  cost — the lever direction is right, mini is the wrong instrument.
- **Tolerance / metric not the cause:** `audit.ordinal_tolerance_audit` showed wider tolerance
  bands don't rescue op45 (the model emits a *different* op ~99% of the time at onset positions
  at mini, 73% at prodlike) — not a metric artifact.

## Landed encoding/loss stack (the best target we have)

| feature | tokens / preframr | what it does | landed design |
|---|---|---|---|
| `TrajectoryAnchorPass` | 0.25.0 / 0.2.6 | anchor FREQ_TRAJ at gate/sweep origins (not value-run boundaries) | [`landed/trajectory_anchoring.md`](landed/trajectory_anchoring.md) |
| `--freq-v0-interval` | 0.26.0 / 0.2.7 | encode V0 as a signed interval from the previous voice onset → transposition-invariant | preframr-tokens `landed/freq_v0_interval.md` |
| `--freq-onset-pass` (FREQ_ONSET op48) | 0.27.0 / 0.2.9 | re-tag residual op0 SET on TRAJ_REGS → 1-token onset; SET only carries control/ADSR/routing | preframr-tokens `landed/freq_onset_channel.md` |
| `--onset-loss-weight` | preframr 0.2.8 | up-weight FREQ V0-onset CE class — force capacity onto the rare-and-ignored onset | [`landed/onset_loss_prioritization.md`](landed/onset_loss_prioritization.md) |

Decisive read: `audit.content_tier_report --onset` with the op-aware `melodic_onset_bucket`
(op45 V0 + op48 FREQ_ONSET + op47 NUDGE pitch on freq regs 0/7/14) gives the full freq-pitch
accuracy — not just op45's slice.

## In flight
- `freq_onset_channel_mini` (2026-05-28): anchored + interval + `--freq-onset-pass` vs split
  baseline. Tests whether the unified onset channel + SET cleanup move anything at mini on top
  of the best stack. Mini is scale-bound for melody; primary value is metric un-confounding +
  SET homogeneity + the de-fragmentation signal.

## Open frontier
1. **Prodlike full-stack arbiter — the deferred decisive test.** Anchor + interval +
   freq-onset-channel + onset-loss-weight (sweep W) vs the absolute baseline, at the scale
   where op45 has non-zero signal (0.067). Five mini A/Bs have exhausted cheap diligence; mini
   provably cannot learn melody. The prodlike read = unified V0-onset acc.
2. **Generative / distributional metric pivot — in reserve.** If even the prodlike full stack
   doesn't lift V0-onset meaningfully, exact next-onset prediction may be intrinsically
   scale-hungry / partly aleatoric for generalization. Pivot the melody success metric to
   interval/n-gram statistics + the 12-SID WAV audition gate (`music_llm_landscape_and_fail_fast_plan.md`
   territory).

## Reusable readers (audit/)
- **`content_tier_report.py --onset`** — per-tier + by-op + the unified `melodic_onset_bucket`
  (op45 V0 + op48 FREQ_ONSET + op47 NUDGE pitch on freq regs); replaces the bespoke
  `parse_per_class.py` / `freqtraj_distribution.py`.
- **`melody_predictability.py`** — non-neural predictability ceiling (n-gram + copy-from-history)
  on the onset line; refuted the aleatoric hypothesis.
- **`ordinal_tolerance_audit.py`** — tolerance-band recovery; refuted the "near-miss noise"
  explanation.
- **`audit_checkpoint_per_class.py`** — per-class/per-tier accuracy (consumes
  `preframr_tokens.tier_accuracy`).

## What this doc is for
The forward-looking single home for the melody-learnability arc. Per-feature design docs
(anchoring, interval-V0, FREQ_ONSET, onset-loss-weight) live in `landed/`; new probes, levers
and results land here. Refuted directions get a `data/refuted/<exp>.md` stub.
