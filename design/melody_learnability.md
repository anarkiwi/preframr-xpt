# Melody learnability — the open content frontier

**Status:** ACTIVE research arc. The program's goal is content/melodic generalization. After
the first confirmed content win (`full_macros`) turned out to be **SET scaffolding, not melody**
(FREQ_TRAJ ~0.026 acc; the model rarely emits FREQ_TRAJ at all), four landed encoding/loss
features attack the melody question and converged probes localise the blocker.

## OVERTURNED 2026-05-28/29 — it was tokenization, not scale (the section below is superseded)

The "scale-bound" diagnosis was wrong. A focused tokenization sweep on the clean
PW+filter-ablated substrate, read by an op×subreg content-tier split:

- **`--tkvocab 0` (disable Unigram) lifts op45 V0_HI 0.009 → 0.658 at mini.** V0-onset≈0
  was a **Unigram-merge artifact**: the merger welds the interval-coded onset forward into
  ~9489 compound tokens (merge-len 7.37) bundling pitch + V0_LO + DELTA shape; the model
  nails the compound *shape* but can't isolate the *pitch*. Control: op48 (a single, never-
  merged token) stays 0.000→0.000, ruling out a vocab-size artifact. → melody IS learnable at
  mini once the onset is a separable low-cardinality atom; the lever is tokenization STRUCTURE,
  not model scale. (`melody_no_unigram_mini`, `/scratch/tmp/preframr_no_unigram`.)
- **Re-read of "V0_HI 0.66":** that is the model predicting the near-constant HIGH byte of
  the signed interval (≈the SIGN, ~2 values), NOT pitch. The real pitch **MAGNITUDE = V0_LO
  0.35**. So de-merged melody = sign easy (0.66), magnitude hard (0.35), absolute unlearned.
- **Voice attribution is NOT the lever.** `--voice-id-on-marker` (minimal) null-to-negative;
  `--voice-order-on-marker` (clean: FRAME→0 tick, voice id solely on the VOICE reg per run,
  decode auto-detects FRAME.val==0) content-neutral (V0_HI 0.668 ≈ 0.658) — but makes the
  structural tier trivially predictable (0.79). See [`voice_encoding_reference.md`](voice_encoding_reference.md).
- **op48 interval-coding REFUTED.** `--freq-onset-interval` (per-reg mod-256 byte interval)
  moved op48 only 0.000→0.013 despite entropy dropping 6.5→5.3b. A single byte fuses
  sign+magnitude (40 eff) in an op the model barely emits — the wrong factoring.

**New frontier = the pitch MAGNITUDE (V0_LO ≈ 0.35), not channel coverage.** Candidate levers:
HI/LO-split op48 like op45 (sign byte → easy target) or fold isolated onsets into op45;
**semitone-quantize the magnitude** (collapse sub-semitone cent jitter to a pitch-class
alphabet); or pivot the melody yardstick to distributional/audition. Open question under test:
is FREQ_TRAJ's multi-attribute bundling itself what makes V0_LO unlearnable?

## Converged diagnosis (2026-05-27 → 2026-05-28) — SUPERSEDED, see above
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
| `--freq-v0-interval` | 0.26.0 / 0.2.7 | encode V0 as a signed interval from the previous voice onset → transposition-invariant | [`landed/freq_v0_interval.md`](landed/freq_v0_interval.md) |
| `--freq-onset-pass` (FREQ_ONSET op48) | 0.27.0 / 0.2.9 | re-tag residual op0 SET on TRAJ_REGS → 1-token onset; SET only carries control/ADSR/routing | [`landed/freq_onset_channel.md`](landed/freq_onset_channel.md) |
| `--melody-merge-split` | 0.28.0 / 0.2.10 | post-Unigram-encode pass: split cross-melody-boundary merges so pitch is a separable prediction target | [`landed/melody_merge_split.md`](landed/melody_merge_split.md) |
| `--onset-loss-weight` | preframr 0.2.8 | up-weight FREQ V0-onset CE class — force capacity onto the rare-and-ignored onset | [`landed/onset_loss_prioritization.md`](landed/onset_loss_prioritization.md) |

Decisive read: `audit.content_tier_report --onset` with the op-aware `melodic_onset_bucket`
(op45 V0 + op48 FREQ_ONSET + op47 NUDGE pitch on freq regs 0/7/14) gives the full freq-pitch
accuracy — not just op45's slice.

## freq_onset_channel_mini result (2026-05-28) — biggest SET-cleanup win to date, melody still 0
3-seed mini, onset_chan (anchored + interval + `--freq-onset-pass`) vs split baseline. Massive
all-tier + content-tier lift: **val_acc 0.125→0.215 (+0.09), val_loss 11.23→8.22 (−3.0)**,
**content-tier 0.076→0.249 (+0.173)**, **content/structural 0.237→1.835** — biggest content
lift the program has seen at mini. But the mechanism is **SET cleanup, not melody**: op0 SET
**0.154→0.831 (~5.4×)** because removing the 12% freq-pitch noise (now in op48) leaves SET as
homogeneous control/ADSR/routing — same lever as freq_core_ablation, larger pull. **V0-onset
acc = 0.000 in both arms** (unified bucket op45 V0 + op48 + op47 pitch). Sixth mini confirmation
that melody is scale-bound. The encoding-stack win for *content* (SET) is real and likely to
hold at scale — folds into the prodlike A/B as a co-confirmed content lift.


## Open frontier
1. **`melody_stack_prodlike` — RUNNING** (the deferred decisive test). full_macros +
   anchor + interval V0 + FREQ_ONSET channel + `--onset-loss-weight 10` vs plain full_macros,
   3 seeds, deployment config (tkvocab 8192, B=4/accum=8) on `:0.2.9`. Dual-purpose:
   (a) does unified V0-onset acc rise above the absolute baseline's op45 = 0.067 → real
   melody at scale; (b) does the SET-cleanup content lift (mini 0.076→0.249) hold at
   prodlike scale → content/deployment win regardless. Seed-major runner gives a 1-seed
   cross-arm signal ~6–11 h in, not 30 h. Read: `content_tier_report --onset`.
2. **Generative / distributional metric pivot — in reserve.** If `melody_stack_prodlike`
   shows the SET-cleanup win at scale but V0-onset stays flat (~0.067 absolute baseline), the
   encoding/loss axis is exhausted for exact-next-onset and we pivot the melody success
   metric to interval/n-gram statistics + the 12-SID WAV audition gate
   (`music_llm_landscape_and_fail_fast_plan.md` territory).

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
