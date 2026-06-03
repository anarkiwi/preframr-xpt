**Status:** Draft / design resolution 2026-06-03. Reframes (and supersedes the premise of)
[`superframe_voice_lane_design.md`](superframe_voice_lane_design.md): factor by musical **role**, not
physical voice. Learnability-framed by [`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md);
grounded in [[control-aware-encoding]] (the ctrl reg already tags per-frame role) and the refuted
[`melody_channel_factorization.md`](melody_channel_factorization.md).

# Role-lane factorization: voices are a resource, roles are the model target

## The problem with voice-lanes
A SID voice is a **hardware resource** (oscillator+envelope) the composer **allocates dynamically**; a
musical **role** (melody / bass / harmony-arp / percussion) is assigned to a voice and the assignment
**changes over time** — a voice time-shares roles, a role hops voices, and the filter (a single global
resource) is routed to a shifting subset. Grouping tokens by **physical voice** (the voice-lane design)
therefore *splits one melodic line across lanes* the moment the composer juggles voices, and *welds two
roles* into one lane. The predictive structure that transfers across tunes is **role-continuity** (a
melody predicts the next melody note; a bass-line predicts the next bass note), not channel-continuity.
So the learnable factorization is by role — voice-lanes are a crude proxy that fails on exactly the
resource-management the composer is doing.

## The abstraction (three layers)
Invert the rendering — music is generated as role-lines, *orchestrated* onto voices, then rendered to
register pokes; recover the inverse:

1. **Role lanes** (the learnable channels): `melody`, `bass`, `harmony/arp`, `percussion` — each collects
   its role's events with role-internal continuity, so the model predicts *within role* (the structure
   that generalizes). This is the cross-voice de-mux that `melody_channel_factorization` pointed to
   (within-voice was refuted; the cost is cross-voice/structural).
2. **Orchestration map** (the resource channel, makes it invertible): a low-rate channel recording, over
   time, **which physical voice carries which role** — voice-allocation changes ARE the composer's
   resource juggling, and a single explicit "role→voice (re)assignment" event is musically meaningful and
   byte-exact-restoring. Without this the factorization can't round-trip.
3. **Filter / timbre bus** (shared resource, NOT a voice property): the global filter — cutoff (regs
   21/22), resonance + **routing bits** (reg 23 `$D417` bits 0–2 = which voices are filtered), mode/vol
   (reg 24) — as **one** trajectory channel, *bound to voices via the routing bits*. A filtered voice's
   role/timbre is "the bus applied to it"; a routing change (a voice enters/leaves the filter) is a
   timbre-transition event, not a per-frame poke. This matches the hardware (filter is global) and the
   music (timbre is a cross-voice gesture); `filter_sweep` already mines the cutoff ramp — this gives it
   an owner.

## Tractability ladder (by ambiguity — build low-risk first; role is LATENT)
Role assignment is a hidden variable, and **mis-segmentation fragments a line worse than physical
voice-lanes**. So do NOT infer aggressively; climb from hardware-detectable to latent:

1. **Percussion lane — FIRST (mostly mechanical).** Percussion has a hardware signature: noise waveform
   (ctrl reg `v*7+4` bit 7) and/or short drum-ADSR hits; [[control-aware-encoding]] already tags
   noise-timbre/percussion per frame. Low-error, high-value — drums are rhythmically regular and currently
   **dilute the pitched voices' prediction target**. Pulling them to their own lane is the cleanest win.
2. **Filter/timbre bus — SECOND (mechanical-ish).** Routing is explicit in reg 23; regroup the global
   filter writes + routing into one channel with binding tokens. No latent inference.
3. **Melody / bass role inference — LAST, GATED (latent, lossy).** Heuristic seeds: pitch range
   (bass = low, melody = mid/high sustained), ornament signature (vibrato/slide ⇒ lead), gate/rhythm;
   `audit/extract_sid_melody.py` already extracts per-(dump,voice) melodic lines. Build the segmenter
   ONLY if it pays off (below).

## Measure before building (the gate)
Same discipline as the codebook/voice-lane work — prove direction with the triage before the multi-week
encoder:
- **Role-lane vs voice-lane vs frame-major:** on tunes with heuristic role labels, reorder into each and
  run `audit/learnability_triage.py` — does role-lane concentrate MI at shorter lag / drop per-frame h_k
  **more than** voice-lane? If role≈voice, the latent role inference isn't worth its lossiness; if
  role≫voice, it justifies the segmenter.
- **Percussion-separation win (directly measurable):** does removing noise/drum frames from the pitched
  lanes lower *their* h_k / sharpen *their* MI? This needs no role inference — just the noise-waveform tag.

## Cautions / open
- **Latency of role** — a wrong assignment is worse than no de-mux; keep a `RESID`/escape so an
  unclassifiable span stays physical-voice, never force a role.
- **Byte-exactness** — the orchestration map must record role→voice→original write order to invert; this
  is strictly harder than voice-lanes (which the deferred design already flagged a round-trip blocker for).
- **Priors** — `voice_trajectory` (all variants) and within-voice `melody_channel_factorization` are
  already **refuted**; this is the cross-voice/role lever they pointed at, but "de-mux helps" is not a safe
  prior in this codebase — the triage gate is mandatory, not optional.
- **Sequencing** — defer behind the in-flight residual-SET stream changes (new ctrl/note-off atoms) and the
  codebook-vs-substrate decomposition; reorganizing roles on a moving stream chases a moving target.
