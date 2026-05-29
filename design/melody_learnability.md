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

**New frontier = the pitch MAGNITUDE (V0_LO ≈ 0.35), not channel coverage.**

**"FREQ_TRAJ too complicated?" — mostly refuted (2026-05-29 V0_LO predictability probe).**
Per-voice V0_LO is **255 distinct values, 5.64 bits**; in-sample n-gram ceilings: mode 0.254,
1-gram 0.301, 2-gram 0.511 (optimistic/memorized). Model eval acc 0.35 sits *between* 1- and
2-gram, i.e. it already captures the available low-order structure — and it predicts the
*structural* op45 sub-tokens fine (FLAGS 0.72, sign 0.66, COUNT 0.63), failing only on the two
high-entropy fields (V0_LO 0.35, DELTA 0.15). So the bundling isn't strangling prediction; V0_LO
is hard because it is **intrinsically high-entropy cent-resolution pitch**. (A residual 0.35→0.51
gap may include a long-range-context/bundling cost — V0_LO history is buried among FLAGS/COUNT/
DELTA + other voices — but the 0.51 ceiling itself is the binding limit.)

**Semitone-quantization REFUTED as a lever (2026-05-29 pre-flight probe).** Snapping the
onset value (cents=50, so a semitone = 2 bins) shrinks the alphabet but leaves the 2-gram
predictability ceiling FLAT: raw 5.70b/362→2gram 0.514; semitone-snap 4.96b/194→0.492;
whole-tone 4.04b/104→0.506. The magnitude's difficulty is **not** removable sub-semitone
cent-jitter — it is genuine pitch-range/leap *sequence* entropy that survives quantization
(reducing the alphabet can't help when which-note-follows-which is the hard part). So
`melody_onset_semitone_mini` was NOT built (lossy for zero predictability gain).

**Locality / freq_traj-pairing REFUTED for the magnitude (2026-05-29 cross-song probe).**
Hypothesis: V0_LO is unpredictable because consecutive same-voice onsets are separated by
too many tokens. Test: cross-song n-gram (fit 104 songs → 53 held-out): 1-gram 0.304,
2-gram 0.269 (bigram contexts don't transfer). The model's 0.35 **already exceeds** the
cross-song adjacent-2-gram ceiling — so the in-sample 0.51 was pure memorization, and
bringing onsets adjacent (pairing / voice-lanes) gives the model no information it isn't
already extracting. Locality is not the missing lever for the magnitude.

**INDEPENDENT CONFIRMATION — synthetic rule in the real encoding generalizes (2026-05-29,
`audit.encoding_generalization_test`).** The same deterministic successor-motif rule as
`framework_arch_test`, expressed in the REAL FREQ_TRAJ atom encoding (FRAME + VOICE + op45
FLAGS/V0_HI/V0_LO/COUNT/DELTA; pitch in V0_LO; ~7 structural tokens between onsets, 512-token
seqs) vs a FLAT single-token encoding of the same rule: **HELDOUT inside-motif onset acc
real 0.876 ≈ flat 0.862** (both train 1.000). The real encoding's byte-split + trajectory
bundling + onset separation do NOT block rule generalization → **encoding SUFFICIENT**; the
real-data V0_LO=0.35 is the DATA (multi-modal melody), not the encoding. Doubly refutes the
locality hypothesis (synthetic separation didn't hurt). **Airtight multi-voice follow-up
(`audit.multivoice_audition_test`):** 3 multiplexed voices (pulse lead / triangle bass /
noise percussion, palette mined from mini), op45 onsets, clean-voice encoding, 2208-token
seqs → HELDOUT onset acc **0.888** (train 1.000); a held-out prompt greedily continued is
88.7% token-identical to ground truth and renders to an audible WAV matching ground truth.
Multiplexing + multiple waveforms do not break rule generalization → encoding SUFFICIENT,
confirmed.

**STRONGEST PROOF — real Bach generalizes through this encoding (2026-05-29,
`audit.bach_encoding_generalization`).** 200 public-domain Bach chorales (music21, SATB→3
SID voices) transcoded into the current encoding (op45 onsets, MIDI→freq→freq-mapper bin,
pulse/tri, clean voice), 160 train / 40 held-out: **HELDOUT next-onset pitch acc 0.513,
ABOVE the cross-chorale 2-gram ceiling 0.456 and 20× chance (0.026)** (train 0.936). The
model generalizes real musical structure to unseen chorales through the exact encoding that
gives 0.35 on real SID melody. Clean contrast, same encoding+model: **Bach model 0.513 >
its 0.456 data ceiling (carries real music); real-SID 0.35 ≈ its 0.30 data ceiling
(multi-modal data).** Encoding is musically sufficient; the SID-melody difficulty is the
DATA. Audible: `bach_prediction.wav` vs `bach_ground_truth.wav`.

**Strategic inflection: exact magnitude is the wrong yardstick; pivot melody success to
distributional/audition.** Even an in-sample, memorizing 2-gram caps at ~0.51 on the
magnitude — so a *generalizing* model can only emit a plausible-not-exact next pitch, and
exact-token acc structurally undersells it. The de-merge win (0→0.66 on the learnable SIGN)
is the real, bankable representation result; the residual magnitude is multi-modal. Next:
score melody by interval/n-gram distribution + the 12-SID WAV audition cohort
(`music_llm_landscape_and_fail_fast_plan.md` territory), not exact V0_LO acc.

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
