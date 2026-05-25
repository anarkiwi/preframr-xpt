# Unified FREQ-trajectory primitive (SLOPE + OSCILLATE_ENV + FREQ_VIBRATO + FREQ_RUN)

**Status (2026-05-25):** Phase 0 implementation **LANDED** (preframr-tokens
`feat/freq-traj-and-profiling`, uncommitted WIP; not yet released). **Reviewed
against `OSCILLATE_REWORK.md` — matches the spec:** `FreqTrajectoryPass` +
`FreqTrajectoryDecoder` + 2-atom `FREQ_NUDGE`, SUBTYPE MONOTONE_RAMP/OSCILLATE/RUN,
locked gap2/alt0.5/hc3, SLOPE's lossy ramp fit preserved, delta payload with
periodic-collapse + 2-byte escape, the 0.14.1 queue-all-drain rule honored, a
`last_frame` drain guard, old ops/passes (SLOPE_OPS/FREQ_VIBRATO/FREQ_RUN/
envelope) retired. preframr-tokens suite **676 pass**; FREQ_TRAJ synthetic
round-trips **9/9**. **VALIDATION COMPLETE (2026-05-25) — all 3 gates pass, see
below; release-ready** (committed `498e000`+`3fc8022`; profiling fix `926fb8b`).
Scope: **full unification** — one `FREQ_TRAJ` op (45) with a `SUBTYPE` field
superseding `SLOPE`, `OSCILLATE_ENV`, `FREQ_VIBRATO`, `FREQ_RUN`; `FREQ_NUDGE` →
2-atom delta. Work order: `preframr-tokens:OSCILLATE_REWORK.md`.

## Validation results (all PASS, 2026-05-25)

1. **Per-frame byte-exact fidelity oracle** — **PASS**. `test_full_pipeline_fidelity.py`
   (`test_ctrl_collapse_lossless` + `test_freq_trajectory_lossless`) run on a
   freshly `anarkiwi/headlessvice`-rendered Grid_Runner, on a docker-capable host
   (a venv, not nested docker). 2 passed, 0 skipped. This is the gate that catches
   the 0.14.1 multi-frame-drain bug class — FREQ_TRAJ's queue-drain reconstruction
   is byte-exact at the per-frame register-state level.
2. **Firing coverage** — **PASS**. `trajectory_coverage` captured_frac **0.522**
   (≥0.40) over 100 dumps (full_macros, freq tier).
3. **Token efficiency** — **PASS**. `tokenizer_profile --compare baseline
   full_macros` runs; the macro set compresses raw heavily (SET 1.71M→123K,
   DIFF 609K→0); FREQ_TRAJ (op 45) = 695K atoms (~53%, the dominant op).

Note: the fidelity oracle must run on a **docker-capable host** (it renders via a
`headlessvice` sibling); confirm CI runs it there rather than skipping (it
SKIPs when docker is unreachable, e.g. nested).

## Problem (data-grounded)

The structural FREQ-trajectory primitives barely fire, so the pitch motion they
are meant to name lands in the lossless-but-opaque mop-up absorbers
(`FREQ_RUN`, `FREQ_NUDGE`) at 2–3 atoms per frame. Measured on 38 real HVSC
songs, current preframr-tokens HEAD, full A/B flag set
(`/scratch/tmp/firing_probe.py`, `freq_traj_probe.py`):

| op | % of atoms | gate |
|---|---|---|
| `OSCILLATE_ENV` (45) | **0.088%** | ≥3 chained SLOPE atoms, each a ≥5-frame near-linear ramp |
| `FREQ_VIBRATO` (52) | **0.151%** | strictly consecutive frames + (near-)exact period 2..8 |
| `FREQ_RUN` (48) | 5.59% | strictly consecutive run ≥2 (2 atoms/frame) |
| `FREQ_NUDGE` (47) | **30.5%** | residual single write (3 atoms/write) |

Decoded per-frame FREQ trajectory (tokenization-independent ground truth):
of **220,307 FREQ motion-frames**, **54.1% are oscillatory** (sign-alternating,
in gap≤2 runs ≥3 frames). The two structural ops capture ~0.24% of atoms
between them.

Three gate mismatches, each pinned:

1. **`OSCILLATE_ENV` is built on SLOPE atoms it can never get.**
   `OSC_MIN_SLOPES=3` × `SLOPE_MIN_RUN_LEN=5` requires ≥3 alternating ≥5-frame
   ramps. Real oscillation half-cycles are 2–4 frames, far below the SLOPE
   floor, so the building blocks never form. Only rare slow wide sweeps fire.
2. **`FREQ_VIBRATO` requires (near-)exact global periodicity.**
   `raw_vibrato_pass.py:_value_period` demands `all(vals[i]==vals[i-p])`. Only
   ~22% of clean ≥4-frame alternating runs are exactly periodic (26% within 20%
   error); real vibrato has depth attack, drift, and the note moving under it.
   v0.14 traded firing for losslessness and the exact-repeat gate rejects ~¾ of
   true vibrato.
3. **Both run-collapsers require strict frame adjacency (gap==1).**
   `FREQ_RUN_MIN_LEN=2`, so the only reason **~157K of 171K sustained-oscillation
   frames don't collapse is intermittency** — SID players write FREQ every
   2nd/3rd frame. Gap≤2 grouping captures 4× more writes. Those intermittent
   frames each become a 3-atom `FREQ_NUDGE`, which is why NUDGE is 30.5%
   (≈ 49K isolated frames × 3, reconciles to the 161K count).

**Root cause:** the gates model an idealized phenomenon (clean ramps, exact
periodicity, strict adjacency); real SID oscillation is *short, drifting,
intermittent* alternation. The unconstrained mop-ups absorb it at 2–3 atoms/frame
with no structural signal — so the model never sees a "this is vibrato/sweep"
token, and the corpus pays the context for it. This is the untapped headroom the
prodlike A/B identified: strictness ~doubled content accuracy, but the trajectory
primitives that were supposed to supply the musical abstraction never fired.

## Hypothesis

Recognizing oscillation by the statistic real SID actually exhibits
(gap-tolerant sign-alternation) lights up an explicit `OSCILLATE` token on ~44%
of FREQ motion (sweep-measured), and a lossless delta + single-byte payload
shared across FREQ_RUN/NUDGE cuts FREQ atoms ~30%. The two are separable: the
token is the **learnability** lever (AGENTS.md priority #1), the delta payload is
the **efficiency** lever (#2, frees predict-host context). Both hold fidelity
(lossless by construction).

## Approach

### Principle: separate recognition from reconstruction

Recognition is robust and statistical (sign-alternation over gap-tolerant
spans). Reconstruction is exact (a lossless delta payload). The two were
conflated before: v0.12 made recognition broad but
reconstruction lossy (envelope fit); v0.14 made reconstruction lossless but
recognition brittle (exact period). Splitting them gets both.

### Sweep correction (load-bearing): the payload is delta, not terminal-list

The first-drafted reconstruction (parametric envelope → terminal-list →
residuals) was **refuted by the host-side sweep** (`/scratch/tmp/osc_sweep.py`,
78 songs). Two measured facts redirected it:

- FREQ values are effectively **8-bit** (max 258; ~0% of frames need a nonzero
  hi byte), yet `FREQ_RUN` stores hi+lo and `FREQ_NUDGE` stores mode+hi+lo — the
  hi byte is wasted everywhere.
- **96.4% of per-write deltas fit a signed byte.**

A terminal-list does not collapse fast vibrato (every frame is a turning point,
so nothing collapses) and adds per-turning-point timing overhead — modeled net
atoms were **negative** (worse than baseline) across the whole grid. The lever
is a **delta + single-byte payload**, which is *orthogonal* to oscillation
structure: it saves **30.1%** of FREQ atoms applied to all writes
(1,158,461 → 809,766). Recognition then rides on top atom-neutrally.

### One op (45) with a SUBTYPE field

Reuse `OSCILLATE_ENV_OP` (45) as the unified `FREQ_TRAJ`; retire
`FREQ_VIBRATO_OP` (52), the envelope path (`envelope.py`, `OSC_STEP_MODE`), the
parametric/terminal-list idea, the standalone `SlopePass`, and `SLOPE_OPS`
(32–34). A `SUBTYPE` field selects the payload:

- **`MONOTONE_RAMP`** (subsumes SLOPE): SLOPE's existing terminal + runtime fit,
  **unchanged** — start = running value, decoder interpolates over runtime
  frames. Its ramp-fit lossiness is deliberate (`project_slope_filter_lossy`) and
  its lossless rework stays deferred; we re-house it under op 45, we do not make
  it exact.
- **`OSCILLATE`** (subsumes OSCILLATE_ENV + FREQ_VIBRATO) and **`RUN`** (subsumes
  FREQ_RUN): a shared **lossless delta-run** — `v0` + per-active-frame signed-byte
  deltas (16-bit escape for the 3.6% large jumps), hold rows for gap frames, and
  an optional periodic-cycle collapse when the delta stream is exactly cyclic
  (~22% of oscillation — bonus, never required). Cumulative deltas off `v0`
  reconstruct the exact cent-bin values; gaps hold. `OSCILLATE` vs `RUN` is the
  structural class the model sees.

`FREQ_NUDGE` (isolated writes) reworks to a 2-atom `mode + signed-delta` form.
The 30% efficiency is in the delta payload + byte-width across RUN/NUDGE
(MONOTONE_RAMP is already compact), so the rework spans the whole FREQ-trajectory
family, not just the OSCILLATE_ENV+FREQ_VIBRATO merge.

### Recognition gate (LOCKED from the sweep)

Operate on the per-voice per-frame FREQ trajectory (post cent-quantization).
A span qualifies when:

- writes are grouped gap-tolerantly, `OSC_MAX_GAP = 2` frames;
- sign-change fraction among nonzero frame-to-frame deltas `>= OSC_MIN_ALTERNATION
  = 0.5`;
- `>= OSC_MIN_HALFCYCLES = 3` turning points.

These are **locked** (no Phase-0 sweep needed). Measured at this setting: **43.9%
coverage** of all FREQ motion, **37.9%** atom-saving on recognized spans. The
knee: gap=1 misses intermittent oscillation (40% coverage); gap=3 over-merges
(63% but crosses gestures); alt=0.6 collapses coverage to ~32%; alt=0.4 admits
near-monotone runs.

### Pipeline placement

One `FreqTrajectoryPass` replaces `SlopePass` + `OscillationEnvelopePass` +
`RawVibratoEnvelopePass` + `FreqRunPass`. It runs on the **raw per-frame FREQ
trajectory** (not on SLOPE atoms — removing root cause #1) and, per voice,
segments + classifies in precedence: `MONOTONE_RAMP` (SLOPE's ≥5 arithmetic
detect) → `OSCILLATE` (gap≤2, alt≥0.5, ≥3 turning points) → `RUN` (delta);
isolated writes fall through to the delta-reworked `FreqNudgePass`. The subtypes
are disjoint by construction (a monotone ramp has low alternation; an oscillation
isn't a monotone ≥5 progression), so precedence is unambiguous.

New order (FREQ regs): `FreqTrajectoryPass → … → FreqNudgePass`.

### Efficiency is the delta payload (FREQ_RUN + FREQ_NUDGE rework)

The 30% efficiency does **not** come from oscillation recognition — it comes from
the shared delta + single-byte payload, applied to all FREQ runs and to isolated
writes. So the rework reworks `FreqRunPass` (delta + gap-tolerant) and
`FreqNudgePass` (2-atom `mode + signed-delta`), not only the OSCILLATE merge.
Measured (78 songs): FREQ atoms 1,158,461 → 809,766 (**−30.1%**). FREQ_NUDGE is
~30% of all corpus atoms today, so this is a large net reduction. Caveats: the
measurement modeled monotone-ramp frames as FREQ_RUN in the baseline — with
MONOTONE_RAMP keeping SLOPE's already-compact fit, ramps are unchanged, so the
realized saving concentrates on RUN + NUDGE (slightly under the modeled 30%). And
it is atom-level; the Unigram tokenizer recovers some multi-subreg regularity, so
the post-tokenization token-level saving will be smaller (the delta stream is
*more* regular for Unigram to merge).

## Flags (additive; one default flip proposed after audition)

```
--freq-trajectory-pass      BooleanOptionalAction, default True
                            (replaces --slope-pass AND --oscillate-env-pass)
--osc-max-gap               int, default 2     (locked)
--osc-min-alternation       float, default 0.5 (locked)
--osc-min-halfcycles        int, default 3     (locked)
```

`--slope-pass`, `--oscillate-env-pass`, `--vibrato-env-pass`, `--freq-run-pass`
and `FREQ_VIBRATO_OP` removed. Migration note: op-code reuse + retiring SLOPE_OPS
invalidates any pinned alphabet/vocab — re-cut and re-run from scratch
(consistent with the strict-no-diff rework's no-transfer rule).

## Phase plan

| phase | scope | wallclock | gate |
|---|---|---|---|
| 0 | Tokenizer impl per `OSCILLATE_REWORK.md`: `FreqTrajectoryPass` + decoder + 2-atom `FreqNudgePass` replacing the 4 old passes. **IMPL DONE + reviewed (2026-05-25), 676 suite + 9/9 round-trips pass.** | done | **Gates pending — see "Validation owed":** per-frame oracle byte-exact, ≥40% oscillatory-motion coverage, FREQ atoms/song < current. Not yet run (oracle needs a fixture host; coverage/atoms need the profiling tools). |
| 1 | preframr-tokens release + main floor bump + image rebake (`build.sh`, bust the proxpi mirror). Register `freq_trajectory` as a `pipeline_spec` transform so the dataset cache key is correct (avoid the `extra_cargs` collision noted in AGENTS.md). | ~½ day | `--help` smoke on predict + audit scripts; pipeline-spec round-trip test. |
| 2 | `unified_freq_traj_mini` A/B at mini, body=large, 60 ep, 3 seeds. Arms: `freq_traj` (target), `baseline` (current full_macros set with old SLOPE/OSC/VIBRATO/RUN). Audits: `audit_checkpoint_per_class.py`, `loop_detection_audit.py`, `prompt_conditioning_audit.py` at T=0.5. | ~12–20 min/arm | (1) content val_acc ≥ baseline − 1σ; (2) `diversity_ratio` ≥ baseline; (3) `loop_collapse_rate` ≤ baseline; (4) atoms/song < baseline. |
| 3 | `unified_freq_traj_prodlike` at prodlike, 60 ep, 1 seed (escalate to 3 on PASS), `PREFRAMR_DATASET_CACHE_DISABLE=1` if still on extra_cargs. eval_a + 8 eval_b_*. | ~6–11 hr | (1) eval_a content val_acc ≥ baseline − 1σ (target: ≥ the full_macros 0.274); (2) ≥5/8 eval_b_* non-negative content lift; (3) `diversity_ratio` > 1.2; (4) no structural-tier regression > 1σ. |
| 4 | 12-SID WAV audition (non-negotiable before flipping default + re-cutting training data). On PASS: flip `--freq-trajectory-pass` default, write `landed/unified_freq_trajectory_primitive.md`. On FAIL: refuted stub in `preframr-xpt:refuted/`. | audition | WAV cohort acoustically indistinguishable from current tokenizer render. |

## Decisions taken (rationale)

1. **Recognize by sign-alternation, not periodicity or arithmetic purity.**
   The measured statistic is alternation (94.5% of long runs), not exact period
   (~22%). Recognition keys on what the data does.
2. **Delta + single-byte payload, not envelope/terminal-list.** The sweep
   refuted the parametric/terminal-list reconstruction (net atoms negative); the
   delta payload is where the 30% lives and is trivially lossless (cumulative
   deltas off `v0`, with a 16-bit escape).
3. **One op (45) + a SUBTYPE field.** MONOTONE_RAMP / OSCILLATE / RUN under one
   op gives the model an explicit trajectory class and removes the seams between
   the four old passes. SLOPE folds in as MONOTONE_RAMP.
4. **SLOPE folds in but keeps its lossy ramp-fit.** MONOTONE_RAMP re-houses
   SLOPE's terminal+runtime payload unchanged — the lossless ramp rework is
   deferred (`project_slope_filter_lossy`), so we unify the op, not the lossiness.
   Running on raw frames (not SLOPE atoms) removes root cause #1.
5. **Efficiency rework spans RUN + NUDGE, not just the merge.** The 30% saving is
   in the delta + byte-width payload applied broadly; recognition is atom-neutral
   on top (MONOTONE_RAMP is already compact).
6. **Cent-bin lossless = audio-neutral.** The op reconstructs the exact
   cent-quantized values the corpus already uses, so it introduces no new audio
   loss vs the current tokenizer; the WAV audition is a guard, not a new risk
   surface. (Content-tier cent-quantization remains deliberately lossy, per
   `project_slope_filter_lossy`; unchanged here.)

## Risk + non-goals

**Risks:**
- **Variable-length delta stream** must place writes on the right frames — the
  same multi-frame-drain hazard that caused the 0.14.1 bug. Decoders must queue
  all bytes and emit none immediately. (No `constrained_decode.py` change: FREQ
  ops are not masked there — verified.)
- **Recognition interaction with downstream passes** (voice rotation,
  frame consolidation) caused the 0.14.0 multi-frame-drain bug; the per-frame
  oracle (`test_full_pipeline_fidelity.py`) is the gate that catches it — keep
  it byte-exact, do not relax to value-sequence round-trip.
- **No-transfer re-run cost.** Op-code reuse invalidates pinned vocab; a full
  re-cut + baseline re-run is required (same rule as the strict-no-diff rework).
- **Firing ≠ learnability.** The A/B showed strictness, not trajectory capture,
  drove the prior content lift. Making the trajectory explicit is a *bet* that
  the structural abstraction helps; Phase 2/3 gates can refute it.

**Non-goals:**
- Making MONOTONE_RAMP (SLOPE) lossless, or any content-tier lossless rework
  (deferred) — SLOPE's ramp-fit lossiness is preserved as-is.
- Cross-voice oscillation tracking (`VOICE_TRACK` stays refuted — FREQ is an
  additive cent-bin index, not multiplicative).
- Changing cent-quantization.

## Implementation handoff

The canonical, locked library-side work order is
**`preframr-tokens:OSCILLATE_REWORK.md`** (delta payload, locked params, exact
files/tests, no sweeps). The sweep that locked the design is complete; the
implementer does not run any sweep. Main-repo side (`args.py` flag rename +
`pipeline_spec` registration + `requirements.txt` floor + `build.sh` rebake) and
Phases 2–4 (the mini/prodlike A/Bs + WAV audition) stay on this repo's side.

## References

- AGENTS.md "Underperforming macros" + "Proposed next step (2)".
- preframr-tokens CHANGELOG 0.10.0–0.14.1 (OSCILLATE_ENV / FREQ_VIBRATO history).
- Diagnostic + sweep probes (removed from `/scratch/tmp` after the numbers
  above were captured here) are reimplemented permanently per
  `tokenizer_profiling_tooling_design.md` / `preframr-tokens:PROFILING_TOOLS.md`.
- `tokenization_vs_music_llms.md` (content ceiling is tokenization-induced).
- Fidelity gate: `preframr-tokens:tests/test_full_pipeline_fidelity.py`
  (per-frame register oracle); 12-SID WAV audition (orinnx_audition_design).
