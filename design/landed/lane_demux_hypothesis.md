# Lane de-mux hypothesis — voice- or role-contiguous ordering of the event stream

**Status:** Open hypothesis (merged 2026-06-12 from `superframe_voice_lane_design.md` +
`role_lane_factorization.md`, restated against the v3 event model). **Trigger:** open this only if
the canonical event-model runs localize a content failure to cross-voice interference (e.g. `NI_*`
per-op accuracy flat while within-voice structure learns — the AGENTS.md "IF CONTENT NOT LEARNED"
branch). Do not build ahead of that evidence: every prior de-mux-adjacent bet (`voice_trajectory`,
`sequence_order_normalization`) refuted or recovered ~5%, so "de-mux helps" is not a safe prior.

## The problem (v3 form)

The event stream is **frame-major**: `<DT> ( <VOICE_v> <events>* )*` with voices ascending inside
each frame group. A voice's melodic line is therefore chopped into per-frame slices separated by the
other voices' events — the next same-voice note is a long-range, position-unstable dependency, and
the next-token target mixes three lines (P3). Measured history (pre-v3 numbers, but the multiplexing
is structural and survives): deployed melody-onset ≈ 0 vs a ~0.34 per-voice ceiling; held-out
next-interval 0.52 was measured on *already de-multiplexed* single-voice data. v3's `NI_*` interval
lane fixes key-invariance (layer 2), **not** the interleave (layer 3 — this doc). A v3-specific
sub-question: do BPE merges across voice boundaries re-introduce the cross-melody weld that
`melody_merge_split` once fixed? (Check the learned merge table before blaming the interleave.)

## Two forms, one mechanism

- **Voice-form (byte-exact substrate, build first if triggered):** within a window (the KEYFRAME
  segment is the natural unit), reorder events **voice-major** — each voice's events contiguous,
  carrying their `DT`s — with an exact inverse permutation back to render order. Same-voice
  prediction becomes short-range and position-stable; merges become within-lane.
- **Role-form (the truer target, strictly after voice-form):** roles HOP voices — a fixed voice-lane
  splits one melodic line and welds two roles. The real lever is **causal-DAG ordering:
  accompaniment roles BEFORE the melody role**, so melody is predicted with its harmony in-context
  (measured: harmony conditions the next melody interval, +0.294 bits; 63% of lead lines hop
  voices). Role identification is therefore the *mechanism*, not a refinement — plain physical lanes
  can backfire by pushing the harmonic determinant out of locality.

## Hard constraints (durable, from the role-form analysis)

- **Sync/ring wiring:** SID hard-sync/ring (ctrl bits 1/2) tie voice N's audio to voice N−1's
  oscillator freq in a fixed physical ring. Never separate a modulator from its carrier into
  independent absolute-freq lanes (forces non-local cross-lane arithmetic); encode the edge as a
  carrier↔source *relationship*, keyed by physical voice index. Never drop or merge a
  silent-but-modulating voice. Modulation is an **edge**, not a role — a voice can play melody and
  feed sync simultaneously.
- **Filter is global:** cutoff/res/routing/modevol are one bus *bound to voices via the reg-23
  routing bits*; a routing change is a timbre event with an owner, not a per-voice property.
- **Mis-segmentation is worse than no de-mux:** an unclassifiable span stays in physical-voice
  order; never force a role.

## Tractability ladder (climb only as far as evidence demands)

1. **Percussion lane** — hardware-detectable (noise waveform / drum-ADSR); drums dilute the pitched
   target and pulling them out needs no latent inference.
2. **Wiring graph** — filter routing + sync/ring edges are explicit in the registers; no inference.
3. **Melody/bass role inference** — latent and lossy (pitch range, ornament signature, gate rhythm);
   build the segmenter only if the triage shows role-lanes ≫ voice-lanes.

## Triage RESULT (voice-form RAN 2026-06-14 — does NOT clear the gate)

Voice-form triage ran on the v2 corpus (frame-major vs voice-major event ordering,
`learnability_triage` proxies, window-mode 8192 + song-mode; `data/audit/lane_demux_triage_v2.md`).
**Verdict: voice-form does not clear the gate.** Window-mode: **induction-copy is flat**
(0.9457→0.9469 — the gate requires it to *rise*); per-frame h_k drops only at high memory-depth
(k=4 −14%, k=3 −6%) and *rises* at k=1 — weak and mixed. Consistent with the refuted
`sequence_order_normalization` (~5% recovery). The interleave (M2) is **not** the binding learnability
constraint — induction-copy is already ~0.946 regardless of ordering (copy-dominance M4 is
corpus-inherent, not interleave-caused). **Deprioritized.** Role-form (the truer target, +0.294 bits
prior) is untested but needs the role segmenter, and voice-form's flat copy tempers it — build only on
a stronger signal. The binding constraints stay M4/M1 (see
`../generation/free_running_pathology_remediation_design.md`).

## Gate

Triage before any build: reorder a labeled sample into frame-major / voice-major / role-major and run
`audit/learnability_triage.py` (seq_len 8192, window mode) — the de-mux wins only if per-frame h_k
drops and induction-copy rises *more than* voice-form alone. Then one canonical A/B, read on the
content tier (`NI_*` per-op accuracy) **plus** a no-regression check on the other lanes and the
[generation quality gate](../generation/generation_quality_gate.md) (a melody win that wrecks
ensemble coherence is a loss). Byte-exactness via the standard encode self-verify on the
inverse-permuted stream.
