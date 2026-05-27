# Trajectory anchoring — correcting the noise-corrupted melodic encoding

**Status:** **UNIMPLEMENTED — CRITICALLY BLOCKING.** The current FREQ_TRAJ / PW /
filter encoding corrupts the melodic signal before the model sees it; until this is
fixed, melodic content is effectively un-anchored noise and the content/generalization
ceiling cannot lift. Implementation-level spec (preframr-tokens, self-contained):
[`preframr-tokens/design/freq_trajectory_anchoring.md`](../../preframr-tokens/design/freq_trajectory_anchoring.md)
(landed uncommitted on tokens `main`). This doc is the research-level framing + why it
gates the program.

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
opt-out-gated, so it is low-risk to land.

## After it lands (validation plan)
1. Land the tokens pass + tests (see the impl doc); FREQ_TRAJ byte-exact round-trip
   must stay green.
2. Re-cut a mini `full_macros` dataset with `trajectory_anchor_pass` on; **re-run the
   content-tier per_class audit** — does FREQ_TRAJ acc rise from ~0.026, and does
   overall content-tier acc lift beyond the SET-scaffolding plateau? That is the decisive
   test that the anchor fix makes melody learnable.
3. If yes, this supersedes the SET-only content win and re-opens the melodic-augmentation
   thread (preframr-aug) on a learnable substrate.

## Supersedes / relationship to refuted work
- **Naive "anchor on gate"** is wrong and explicitly superseded: gate is one observable,
  not the definition; PW/filter and legato sweeps are off-gate.
- **Motif pass** (`data/refuted/motif_pass.md`) is refuted — it chunked the un-anchored
  stream; it did not address the anchor defect.
- **`full_macros`** stands as the scaffolding win; this fix targets the orthogonal,
  larger melodic axis it could not reach.
