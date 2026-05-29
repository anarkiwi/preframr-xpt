# Melody/ornament channel factorization — surface melody while keeping ornamentation

**Status:** EXECUTED 2026-05-29 (see RESULTS + cross-voice follow-up). Within-voice note/ornament
multiplexing is a **real but MODEST** cost (skeleton acc +0.032 ×3 seeds, always positive) — not
the deployed-≈0 explanation. The **cross-voice** (3-voice frame) multiplex is ~2× larger (−0.062,
to 0.274) and stacks, but still not ≈0; the residual to deployment is the structural-token
interleave + Unigram weld (the banked de-merge axis). Strong within-voice form refuted; the lever
for the deployed gap is **voice de-multiplexing**, not ornament factorization. Probes + render
committed (`audit.melody_channel_probe`, `audit.melody_multiplex_probe`, `audit.melody_channel_render`,
`extract_sid_melody --channels/--multiplex`); no new corpus, mini scale.
Cross-ref: [`melody_data_gap_ladder.md`](melody_data_gap_ladder.md),
[`encoding_principles.md`](encoding_principles.md),
[`landed/freq_v0_interval.md`](landed/freq_v0_interval.md),
[`landed/freq_onset_channel.md`](landed/freq_onset_channel.md),
[`superframe_voice_lane_design.md`](superframe_voice_lane_design.md).

## The gap in the prior verdict

The melody arc concluded "encoding is sufficient; the limit is the data." That was proven for
melody **measured in isolation** — by *destroying* the ornament (gate-anchor / de-arp probes,
lead-voice-only, PW+filter substrate) and measuring the surviving note line. Those subtractive
moves are audit probes (`audit.melody_ladder`, `extract_sid_melody`, `gate_anchor_probe`), **not
the deployed encoding** — but they are also the *only* condition under which "sufficient" was
shown. The deployed encoding never drops ornament; it interleaves it.

What the arc measured (`melody_data_gap_ladder.md` RESULTS):

| condition | held-out predictability |
|---|---|
| L1 all-freq onsets (ornament-dominated) | **0.559** |
| L2 gate-anchor note-ons (melody alone) | 0.247 |
| deployed V0-onset, **full interleaved stream** | **~0** |

The ornament is *trivially* predictable (0.559); the note melody alone reaches 0.247; but in the
deployed stream the note onset sits near **0**. The ~0.25 between deployed-interleaved and
melody-alone is **not** the data ceiling (0.247 is). It is the cost of **multiplexing the
high-entropy note target into the same next-token position as the trivially-predictable ornament
and shape tokens** — the model spends its loss budget on the easy tokens. The arc proved a data
ceiling on melody-in-isolation; it never tested whether note and ornament can **coexist as two
separately-predicted channels** at full fidelity. That is the deployment condition, and it is
this doc's question.

This is encoding principle **P3** (don't multiplex the target) extended from *across voices* to
*within a voice*: the note skeleton and its ornamentation are independent streams fused into one
prediction position.

## The asset already in the tokenizer

`TrajectoryAnchorPass` (preframr-tokens) already **classifies** every freq write: gate-anchored
note-on / aperiodic-melodic (the **skeleton**) vs ramp / periodic-oscillator — vibrato, slide,
arp — (the **ornament**), via `pass2_collapse` + the gate union + the autocorrelation periodicity
test. Today that classification is used only as a *segment boundary* that folds everything back
into one interleaved op45/op48 stream. We discard the most valuable thing it computes: the
note-vs-ornament label.

## The factored encoding (the change, if the probe confirms)

Promote the anchor classification from a boundary into a true channel factorization. Emit two
coupled streams instead of one interleaved one:

- **Skeleton channel** — the gate-anchored note-onset as an interval-V0 token (key-invariant;
  already built, `freq_v0_interval`). This is the melody prediction target, scored
  distributionally / by audition per **P6** (exact-token acc undersells a multi-modal target).
- **Ornament channel** — the modulation expressed **relative to the current skeleton note** as a
  compact **parametric** token (vibrato depth+rate / slide target+rate / arp pattern-id) instead
  of N raw delta samples. Decodes back byte-near-exact, so **ornamentation is fully maintained**
  (fidelity floor, **P1**). This is the trivially-predictable (~0.55) stream, now kept *off* the
  melody prediction position rather than diluting it.

This is **additive separation, not subtraction** — the explicit completion of the line already
written into `freq_onset_channel.md` ("decoupling onset-from-shape inside trajectories are
follow-ups") and `freq_v0_interval.md` ("the shape is fine; the onset is the defect"). It is a
distinct axis from `superframe_voice_lane_design` (which de-multiplexes *voices*, not
note-from-ornament).

## The decisive cheap probe (do this FIRST — fail-fast, no prodlike, no new data)

Isolate the one variable — **prediction-target multiplexing** — at fixed melodic content and
fixed mini scale, before building any tokenizer change. Reuse the `melody_ladder` harness (same
llama3_2 mini body, same held-out-by-dump split, same interval metric).

Two arms from a **single** extraction of the existing mini lead-voice melody:

1. **interleaved** — one temporally-ordered token stream containing both skeleton note-ons
   (interval from previous skeleton) and ornament writes (interval from the active skeleton
   note), disjoint id ranges, with an `is_skel` mask. Train next-token; **score held-out accuracy
   on skeleton positions only.** This is the deployment-like condition: the model must also
   predict the ornament tokens.
2. **skeleton_only** — the same sequences with ornament tokens removed. Train; score held-out
   skeleton accuracy. This reproduces the ladder L2-interval anchor (~0.247).

Tooling:
- `extract_sid_melody.py --channels` — new output mode emitting `{"dump", "tokens", "is_skel"}`
  per (dump, lead voice); existing L1/L2/L3 modes untouched.
- `audit.melody_channel_probe` — loads the labeled json, trains both arms, prints per-arm
  held-out **skeleton-position** accuracy + the multiplexing delta.

## Decision rule

- **Skeleton accuracy (interleaved) ≪ skeleton accuracy (skeleton_only ≈ 0.247)** → multiplexing
  is stealing melody predictability. The encoding is **not** sufficient under the deployment
  condition; channel factorization is justified. Reopens the verdict; proceed to the parametric
  ornament encoding above (tokens-side, audition-gated).
- **Skeleton accuracy (interleaved) ≈ skeleton_only** → multiplexing is *not* the lever; the
  0.247 really is a data ceiling that survives interleaving. Refute this thread
  (`data/refuted/melody_channel_factorization.md`) and pivot to the single-composer-corpus data
  lever from `melody_data_gap_ladder.md`.

Either outcome is decisive, costs one mini-scale probe (no docker A/B, no prodlike build), and
tests the exact gap — note/ornament multiplexing within a voice — the prior arc left open.

## RESULTS (2026-05-29, `audit.melody_channel_probe`, mini 147 seqs, ×3 seeds)

Extraction (`extract_sid_melody --channels`, mini train.list, 150 dumps → 147 lead-voice seqs):
**94,262 skeleton + 230,487 ornament tokens — ornament is 71% of the stream** (consistent with
the gate-anchor probe's 77–96% sustain-modulation). Held-out (by dump) skeleton-position accuracy:

| seed | interleaved (score skel only) | skeleton_only | delta |
|---|---|---|---|
| 0 | 0.322 | 0.334 | +0.012 |
| 1 | 0.350 | 0.388 | +0.038 |
| 2 | 0.335 | 0.381 | +0.045 |
| **mean** | **0.336** | **0.368** | **+0.032** (range +0.012..+0.045, always positive) |

**Verdict: the strong hypothesis is REFUTED; a minor factorization bonus is CONFIRMED.**

1. **De-multiplexing helps, but only a little.** Removing ornament from the prediction position
   lifts skeleton accuracy by a *consistent* +0.032 (≈10% relative, positive in all 3 seeds) —
   real, not noise. So the encoding is *not perfectly* sufficient: separating the channels does
   recover some melody predictability. The user's instinct is partially right.
2. **But within-voice multiplexing is NOT the dominant lever, and does NOT explain the deployed
   V0-onset≈0.** The *interleaved* lead-voice melody already predicts at **0.336** — squarely in
   the data-ceiling region (ladder L2-interval 0.247; Bach 0.394), nowhere near 0. The deployed
   ~0 must come from the OTHER multiplexing — the 3-voice frame-interleave + structural
   (FRAME/ADSR/PW) tokens + the Unigram weld — i.e. the **de-merge / voice-lane axis already
   known** ([`encoding_principles.md`](encoding_principles.md) P1/P3,
   [`superframe_voice_lane_design.md`](superframe_voice_lane_design.md)), not note-vs-ornament.
3. **Reconciles the prior verdict and the user's critique.** The "data-limited" conclusion
   *largely stands* for the within-voice melody question (interleaved ≈ skeleton ≈ data ceiling),
   with a modest factorization bonus on top. The de-arp/ornament-removal moves were addressing a
   real but small effect; the bigger melody loss is the cross-voice/structural multiplexing and
   the corpus's intrinsic heterogeneity, not ornament pollution of the single-voice note line.

**Audition** (`audit.melody_channel_render`, seed 1, held-out tune = Hubbard *BMX Kidz*;
WAVs in `/scratch/tmp/enc_audition/channel_*.wav`). Single-voice (probe scope):
- `channel_ground_truth.wav` — the true lead-voice note line.
- `channel_skeleton_pred.wav` / `channel_interleaved_pred.wav` — the two models' free-run
  continuations from a 1/3 prompt; comparably (im)perfect, matching the small +0.032.
- `channel_ground_truth_orn.wav` — the **same notes with ornament** layered back in: the encoding
  **preserves ornamentation** while the skeleton channel is a clean melody line — factorization is
  *possible* at fidelity, it just buys ~+0.03 here.
- `channel_interleaved_pred_orn.wav` — the interleaved model's *sampled* continuation, ornamented.
  **Finding: the model under-generates ornament** — greedy emits ≈0 ornament (the diffuse ornament
  distribution loses every argmax to a peaked skeleton interval), and temp-1 sampling emits only a
  handful. So a model trained on the interleaved stream does not reproduce ornament density; the
  ornament channel would need its own emission handling (loss weight / separate head), reinforcing
  that interleaving is the *wrong* home for it even though it is not the dominant accuracy lever.

3-voice GROUND-TRUTH polyphony (NOT predictions — reconstructed from the skeleton+ornament
decomposition on the real frame timeline, absolute pitches, inter-voice harmony preserved):
- `channel_polyphony_skeleton.wav` / `channel_polyphony_orn.wav` — all 3 SID voices, note-ons only
  vs notes+ornament. Demonstrates the **decomposition carries full polyphony and ornament**
  losslessly to audio. (Polyphonic *prediction* across the frame-multiplexed voices is the deployed
  problem — the cross-voice/structural multiplexing axis — measured next.)

## Cross-voice multiplex follow-up (EXECUTED 2026-05-29, `audit.melody_multiplex_probe`)

The RESULTS above attribute the deployed ≈0 to cross-voice/structural multiplexing rather than
within-voice ornament. Tested directly: extract **all 3 voices frame-multiplexed into one stream**
(`extract_sid_melody --multiplex`, 148 seqs, 142 fully 3-voice; voice+channel+interval-coded ids),
train the same mini body, score held-out skeleton accuracy. The effect **stacks** and is ~2× the
ornament effect:

| condition | held-out skeleton acc |
|---|---|
| single voice, skeleton only | 0.368 |
| single voice, +ornament interleaved | 0.336 (−0.032) |
| **3 voices, frame-multiplexed** | **0.274** (−0.062 more; mean ×3 seeds 0.254–0.302) |

Per-voice: v0≈0.31, v1≈0.20 (worst), v2≈0.29. So **cross-voice multiplexing is the larger lever**
(−0.062 vs −0.032), confirming the attribution. **But 0.274 is still not ≈0** — this multiplex
stream is still "clean" (note+ornament intervals only, voice-tagged ids, *no* structural FRAME/
ADSR/PW tokens, *no* Unigram). The residual 0.274→≈0 in deployment is therefore the **structural-
token interleave + the Unigram weld** — exactly the de-merge result already banked (`--tkvocab 0`
lifts op45 V0_HI 0.009→0.658; see [[representation-thread]] in memory / AGENTS). The full ladder of
melody-prediction losses, mini, interval metric:

```
0.368  clean single-voice melody (data ceiling region; Bach 0.394)
0.336  + ornament interleaved        (−0.032  within-voice)
0.274  + 3-voice frame multiplex     (−0.062  cross-voice)   ← this follow-up
 ≈0    + structural tokens + Unigram weld (deployed)         ← the de-merge / voice-lane axis
```

Audition (`--render-dir`, held-out DRAX *Advanced*): `channel_multiplex_gt.wav` (3-voice ground
truth) vs `channel_multiplex_pred.wav` (the multiplex model's polyphonic continuation from a 1/3
prompt — per-voice base pitches offset for audibility; a real model PREDICTION across the
multiplexed voices). Implicated lever for the deployed gap: **voice de-multiplexing**
([`superframe_voice_lane_design.md`](superframe_voice_lane_design.md)) on top of the landed
de-merge — not within-voice ornament factorization.

### Decision

By the rule above this is closer to "**multiplexing is not the (dominant) lever; the data ceiling
survives**" — with the caveat that a small, real +0.032 is on the table. **Do NOT build the
parametric-ornament tokenizer as a melody-accuracy bet** (the +0.03 doesn't justify the
audition-gated tokenizer work on its own). It may still be worth doing as a **token-budget** play
(one param token vs N delta samples) — re-evaluate it there, on axis 2, not axis 3. The melody
lever remains where the ladder left it: **a larger single-composer/style corpus, interval-measured**
([`melody_data_gap_ladder.md`](melody_data_gap_ladder.md)), plus the cross-voice de-merge/voice-lane
work for the deployed-stream ≈0.

## If it had confirmed: landing checklist (NOT pursued — kept for the budget-play reopen)

1. Parametric ornament tokenization in preframr-tokens (vibrato/slide/arp params relative to the
   anchored note), byte-near-exact decode behind the **12-SID WAV audition gate** (P1, not
   merely round-trip-exact since param-fit is lossy).
2. Skeleton channel scored distributionally + audition, **not** exact-token (P6).
3. Net token-budget delta vs the Jetson envelope (P-context-budget): parametric ornament should
   *shrink* tokens (one param token vs N delta samples), a budget win to confirm not assume.
4. Content-tier read via `content_tier_report` (the decisive gate), spotlighting the skeleton op.
