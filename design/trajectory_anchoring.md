# Trajectory anchoring — correcting the noise-corrupted melodic encoding

**Status:** **LANDED + MINI A/B DONE (content win, melody hypothesis unconfirmed at mini);
PRODLIKE A/B QUEUED.** `TrajectoryAnchorPass` shipped in preframr-tokens 0.25.0; framework
toggle `--trajectory-anchor-pass` wired in preframr 0.2.6 (PR #138). Mini
`trajectory_anchor_mini` (2026-05-27): **content-tier acc 0.036→0.080 (+0.044, seed-stable),
all-tier val_acc 0.113→0.137 — but SET-carried (op0 0.063→0.175); FREQ_TRAJ op45 flat at the
floor (0.001→0.002).** Mini is too data-starved to test melody (op45 ~0 in both arms; prodlike
baseline was 0.067), so the melody claim is **deferred to a prodlike A/B** (see Validation
plan). Impl spec: [`preframr-tokens/design/freq_trajectory_anchoring.md`](../../preframr-tokens/design/freq_trajectory_anchoring.md).
This doc is the research-level framing + why it gates the program.

**Re-frame at impl (gating):** the impl doc proposed an opt-*out* gate "mirroring
FreqTrajectoryPass" (default ON). The tokens pass shipped that way, but default-ON
silently anchored every 0.2.5 parse with no toggle — confounding baselines and making
the A/B impossible. The framework flag is therefore **opt-in (default OFF)** and is NOT
registered in `_PIPELINE_NAME_TO_FLAG` (it modifies `freq_trajectory` like the absorber
macros / `--motif-pass`, toggled per-arm via `extra_cargs`). Promote to default-ON only
if the A/B wins the content-tier gate.

## The corruption

`FreqTrajectoryPass` segments each trajectory register (`TRAJ_REGS` = per-voice freq +
PW, global filter) into ramp/oscillation/run primitives **purely from value-runs and
frame-contiguity — it never consults the gate or any note/sweep structure.** So a
trajectory's start anchor (`V0`) lands on an arbitrary value-run boundary, not the real
note/sweep origin. The melodic note-onset pitch — the thing a model must predict — is
buried as a delta inside a value-segmented trajectory whose boundaries are musically
meaningless. The encoding is **lossy in the dimension that matters**: it preserves the
samples but destroys the anchor that gives them melodic meaning. What reaches the model
is, for melody, noise.

This is the root cause of the program's content ceiling:
- `full_macros` (the only confirmed content win) is carried by **SET register
  scaffolding (acc 0.44)**, NOT melody. **FREQ_TRAJ ≈ 0.026 acc, 0.6% of content hits
  despite 9% of positions** (`/scratch/tmp/freqtraj_distribution.py`).
- The model literally won't emit FREQ_TRAJ: at FREQ_TRAJ positions it predicts a
  different op 73% of the time, and tolerance-banding the metric doesn't rescue it
  (`/scratch/tmp/audit_freqtraj_tolerance.py`) — it has no anchor to predict *from*.
- Model-side and data-side melodic interventions are all refuted; the leverage is
  representation, and *this specific encoding defect is the representation bug*.

## Why melody is anchored, and where the anchor is lost

Source drivers anchor a note at the **gate (control-register) retrigger + a base
pitch**, then hold the gate while writing per-frame `freq = base + arp + vibrato`
(confirmed by modelling defMON `pydefmon` and SID-Wizard `pysidwizard`, and SIDdecompiler
for Hubbard-style drivers). So **77–96% of frequency writes are sustain-phase modulation;
only 4–23% coincide with a note-on**, and the gate-anchored base-note sequence is far
lower entropy than the full stream (e.g. Hubbard bass 4.09→2.68 bits). The raw dump
preserves this; `_simplify_ctrl` keeps the gate bit; **`FreqTrajectoryPass` is where the
anchor is discarded** (its `TRAJ_REGS` excludes control; it segments on value, not gate).

Critically, the anchor is **not** "note-on": PW and filter sweeps are frequently armed
**off-gate** (defMON cutoff re-arms 72/127 times off-gate, 30 mid-note; SID-Wizard bass
filter-table loops a sweep off-gate; *A Mind Is Born* — non-tracker hand-written 6502 —
sweeps the filter across the whole tune, gate irrelevant). The anchor is an **observable
trajectory (re)initialization**, recovered intrinsically per register from value
dynamics, with gate as one corroborating observable — generalizing to arbitrary
generators we haven't seen.

## The fix (validated prototype; ready to implement)

An **annotation-only** two-pass detector marks each register's true trajectory origin
frames; `FreqTrajectoryPass` then starts trajectories at the anchor (`V0` = anchored
base value) instead of a value-run boundary. Pass 1: sustained-departure origins ∪ gate
retriggers (recall ~1.0, incl. off-gate sweeps). Pass 2: collapse runs that are a *ramp*
(monotonic) or *oscillator* (periodic value waveform, by autocorrelation) to one onset —
so an arp/vibrato/PWM/continuous-sweep is one trajectory, a melody is kept.

Validated across **4 tunes × 3 registers** with source-derived ground truth: Hubbard
(`Auf_Wiedersehen_Monty`), defMON (`glow_worm`), SID-Wizard (`flashitback`), and the
non-tracker `A_Mind_Is_Born`. Recall ~1.0; off-gate sweeps recovered intrinsically;
oscillator collapse restores precision (arp 0.15→0.98) without collapsing fast melodies.
Prototype + artifacts: `/scratch/tmp/anchor_val/` (`anchor_probe_final.py`,
`{hubbard,defmon,swm,amib}.{writes.parquet,truth.json}`, `{defmon,swm}.traj_anchors.json`).
The remaining low per-register numbers are a **truth-granularity** artifact (tracker
truth counts every driver re-init; the detector targets note/sweep level, which is what
a write-only model needs), not a detector defect.

## Why this is critically blocking

The program's primary goal is content/melodic generalization. The single confirmed
content win is scaffolding, not melody, *because melody is unlearnable under the current
encoding*. Every melodic lever — targeted augmentation for laggard families, any
model-side content objective, cross-engine transfer — is capped until the melodic signal
is anchored. This is the prerequisite representation fix. It is tokenizer-side,
annotation-only (no token-stream change by itself, byte-exact round-trip preserved), and
opt-in-gated (`--trajectory-anchor-pass`, default OFF), so it is low-risk to land.

## Learnability hypothesis (what the A/B tests)
The defect is not that melody is high-entropy — gate-/sweep-anchored base-note sequences
are *lower* entropy than the raw stream (Hubbard bass 4.09→2.68 bits). It is that the
model has **no stable token to predict the note-onset pitch from**: under value-run
segmentation the onset pitch is a delta off a musically-meaningless boundary, so it
manifests as the FREQ_TRAJ failure mode we measured (op 45 ~0.026 acc; a *different* op
predicted 73% of the time at FREQ_TRAJ positions; tolerance-banding does not rescue it —
so it is an **encoding** defect, not a metric artifact and not demonstrably aleatoric).
Anchoring gives every trajectory a stable, musically-meaningful `V0` at the note/sweep
origin; the prediction target becomes "next anchored base note", which is what the source
drivers actually sequence. **Prediction:** FREQ_TRAJ content acc rises off ~0.026 and
overall content-tier acc lifts beyond the SET-scaffolding plateau (~0.32). Risk: if the
core is genuinely aleatoric *after* anchoring, the A/B stays flat — that is the
`freq_core_ablation_mini` question (core learnable vs drowned by PW/filter noise), the
complementary diagnostic.

## Validation plan (status)
1. **DONE** — tokens pass + tests landed (`TrajectoryAnchorPass`, preframr-tokens 0.25.0);
   FREQ_TRAJ byte-exact round-trip green. Framework toggle wired
   (`--trajectory-anchor-pass`, opt-in, preframr 0.2.6, PR #138).
2. **DONE (mini, 2026-05-27) — content win, melody hypothesis NOT confirmed at mini.**
   `trajectory_anchor_mini` (3 seeds, `:0.2.6`; read via the reusable
   `audit.content_tier_report`). Tokenization differs as intended (alphabet 4211 vs 4255;
   tok/song 8039 vs 7561; op45 atoms 16.0k vs 29.5k — arps/vibrato collapsed, 14.1%→6.9% of
   content). **Content-tier acc 0.036→0.080 (+0.044, seed-stable ~2.2×); content/structural
   0.096→0.222; all-tier val_acc 0.113→0.137.** BUT the lift is **SET-carried (op0
   0.063→0.175)** — **FREQ_TRAJ (op45) stayed at the floor (0.001→0.002).** The prediction
   "op45 rises" did NOT hold at mini. **Caveat:** op45 is ~0 in *both* arms at mini (prodlike
   baseline op45 was 0.067, ~30–60× higher), so mini cannot test the melody hypothesis —
   it confirms only that anchoring is a real, seed-stable content gain (SET) and does not
   regress, clearing the bar for prodlike.
3. **Prodlike A/B (RUNNING, launched 2026-05-27 ~20:16; `specs/trajectory_anchor_prodlike.py`,
   `--root /scratch/tmp/preframr_anchor_prodlike`, ETA ~36-66h)** — the only regime that tests the melody claim: does the content
   win hold AND does op45 rise where it has baseline signal (0.067)? If op45 moves → melody
   is learnable, supersedes the SET-only story, re-opens preframr-aug on a learnable
   substrate. If op45 stays flat while op0/content rise → anchoring is another SET-scaffolding
   gain, not the melodic fix; pair with `freq_core_ablation_mini` (core aleatoric vs drowned
   by PW/filter) before concluding melody is intrinsically unlearnable under this encoding.

## Supersedes / relationship to refuted work
- **Naive "anchor on gate"** is wrong and explicitly superseded: gate is one observable,
  not the definition; PW/filter and legato sweeps are off-gate.
- **Motif pass** (`data/refuted/motif_pass.md`) is refuted — it chunked the un-anchored
  stream; it did not address the anchor defect.
- **`full_macros`** stands as the scaffolding win; this fix targets the orthogonal,
  larger melodic axis it could not reach.
