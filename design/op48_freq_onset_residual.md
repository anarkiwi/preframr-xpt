# op48 FREQ_ONSET is the lead-voice sub-frame ornament residual

**Status:** DIAGNOSED 2026-05-29 (`melody_no_unigram_mini` seed0, audits `op48_probe` /
`op48_context`). op48 is stuck at 0.000 not for lack of data but because it is the intra-frame
ornament stream of voice 0, encoded as absolute high-cardinality residuals and starved at the
op level. The lever is the ornament codebook ([`unified_pitch_encoding.md`](unified_pitch_encoding.md),
[[ornament-encoding-transfer-gap]]), not scale and not re-routing into the skeleton.

## Context

On the de-merged substrate (`--tkvocab 0`, PW+filter ablated, freq stack on), the content-tier read
moved the melodic pitch atom **op45 V0-onset 0.012 → 0.338** (merged → no_unigram, same-session A/B).
But **op48 FREQ_ONSET stayed 0.000 in both arms**. de-merge fixed op45 and did nothing for op48 — so
their blockers are different. This note characterises op48 so the prodlike eval reports it correctly
and we don't chase it with scale.

## What op48 is

`FreqOnsetPass` re-tags every **residual `op0` SET on TRAJ_REGS** (freq/PW/filter-cutoff, subreg −1,
reg/val unchanged) to op48. So op48 = the freq writes the `FreqTrajectoryPass` did **not** fold into an
op45 per-frame trajectory. Measured on the eval set:

- **register mix: 98.8% freq-lo, and 100% voice 0** (voices 1 & 2 emit *zero* op48).
- **value distribution: diffuse, 173 distinct, top value only 3.6%** — i.e. raw absolute bytes, the
  opposite of op45 V0's peaked 0/±1/±2 interval signature (value 0 = 29%). Same register (freq-lo),
  opposite learnability, purely from the encoding (op45 got `--freq-v0-interval`; op48 stayed absolute).
- **op-level starvation:** at op48 ground-truth positions the model predicts op45/op25/op0/op42 and
  emits op48 only **142/9059 = 1.6%** of the time; exact-atom match 1/9059 ≈ 0. It fails at the op
  level (predicts the wrong op entirely), before value matters. op48 is 1.4% of all predictions vs
  op45's 59% — a rare tag that loses to the dominant trajectory op.

## Why the trajectory pass leaves them residual

The pass models **one settled freq value per frame**. Voice 0 writes its freq register **multiple
times within a single frame** (fast arps / vibrato / slides):

- **41% of op48 writes land in frames with 4+ freq writes on the same reg**; 34% share the exact
  frame of the previous freq write; 70% are within 0–1 frames of another freq event; only 11% are
  isolated (±2 frames). 56% sit adjacent to an op45 trajectory.

So the pass keeps one per-frame value as the op45 V0 trajectory and dumps the **other intra-frame
writes** as residual → op48. Voices 1 & 2 write freq ~once per frame, form clean trajectories, and
produce no residual. **op48 is therefore the lead voice's sub-frame ornament** — the "~40 raw freq
writes per note" the per-frame skeleton cannot represent.

## Implications

- **Not mis-routed skeleton melody.** Forcing op48 into interval-V0 would interval-encode arp steps and
  corrupt the skeleton. The op45 V0 skeleton (0.338) is the melody target; op48 is ornament.
- **Scale will not move it.** Absolute sub-frame writes stay high-cardinality and op-starved at any
  scale; the model never even emits the tag. "op48-interval refuted" is consistent — residual ornament
  has no coherent previous-note to interval against.
- **The right lever is the ornament codebook**: collapse each note's sub-frame writes into one per-note
  descriptor (ARP-cycle id / vibrato depth / slide) per [`unified_pitch_encoding.md`](unified_pitch_encoding.md),
  not per-write absolute freq. This is the open `[[ornament-encoding-transfer-gap]]`.
- **For now op48=0 is acceptable** if the eval reports skeleton (op45 V0) and ornament (op48)
  separately — which `content_tier_report --onset` now does. Prodlike should not be blocked on op48.

## Reproduce

- `audit/op48_probe`-style: forward eval blocks, map uid→(op,reg,subreg,val), report op48 value/reg
  distribution + predicted-op histogram at op48 positions (op-level recall) vs op45 V0.
- `audit/op48_context`-style: walk the op-tagged post-pass block dfs, frame-indexed, classify op48
  freq-lo writes by same-reg gap / sub-frame multiplicity / nearest-neighbor op.
