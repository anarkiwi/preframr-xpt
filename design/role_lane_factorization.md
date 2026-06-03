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

## The abstraction (four layers)
Invert the rendering — music is generated as role-lines, *orchestrated* onto voices and *wired* together,
then rendered to register pokes; recover the inverse:

1. **Role lanes** (the learnable channels): `melody`, `bass`, `harmony/arp`, `percussion`, plus a
   `modulator` sub-role (below) — each collects its role's events with role-internal continuity, so the
   model predicts *within role* (the structure that generalizes). This is the cross-voice de-mux that
   `melody_channel_factorization` pointed to (within-voice was refuted; the cost is cross-voice/structural).
2. **Orchestration map** (the resource channel, makes it invertible): a low-rate channel recording, over
   time, **which physical voice carries which role** — voice-allocation changes ARE the composer's
   resource juggling, and a single explicit "role→voice (re)assignment" event is musically meaningful and
   byte-exact-restoring. Without this the factorization can't round-trip.
3. **Voice-wiring graph** (the per-span "patch" — shared/cross-voice state, NOT voice properties). Two
   kinds of binding the orchestration manages, encoded as relationships, not scattered pokes:
   - **Filter / timbre bus:** the global filter — cutoff (regs 21/22), resonance + **routing bits**
     (reg 23 `$D417` bits 0–2 = which voices are filtered), mode/vol (reg 24) — as **one** trajectory
     channel, *bound to voices via the routing bits*. A routing change (a voice enters/leaves the filter)
     is a timbre-transition event; `filter_sweep` already mines the cutoff ramp — this gives it an owner.
   - **Sync / ring modulation edges** (the hard constraint — see next section): a directed binding
     "voice N is sync/ring-modulated by voice N−1," carried as the **modulator↔carrier frequency
     relationship**, not two independent absolute freqs.

## Cross-voice dependency: sync & ring (must be accounted for)
SID hard-**sync** (ctrl bit 1) and **ring**-mod (ctrl bit 2) wire the voices in a **fixed ring tied to
physical index**: osc1←osc3, osc2←osc1, osc3←osc2 (voice N modulated by voice N−1, wrap-around; the
SYNC/RING bit sits in the *modulated* voice's ctrl reg). So **voice N's audio depends on voice N−1's
frequency** — even when N−1 is gated *off* and serves purely as a silent modulation source. This breaks
the "independent lanes" assumption in two ways and constrains the design:

- **Lanes are coupled.** A sync/ring edge ties two physical voices; the modulator's freq is meaningless in
  isolation — only the **ratio/interval to the carrier** carries musical intent (a sync-sweep is a relative
  gesture; a fixed ring ratio is a timbre). Splitting the two voices into independent role-lanes with
  *absolute* freqs forces the model to recover the relationship by cross-token arithmetic across the lane
  gap — exactly the non-local dependency Principle 1 says it can't. **Encode the edge as the relationship**
  (interval/ratio modulator→carrier) so the binding token co-locates the dependency and keeps it local;
  keep the exact freq recoverable for byte-exactness.
- **A "silent" modulator is load-bearing.** A gated-off voice driving another's sync/ring has no audible
  line but cannot be dropped or merged — tag it `modulator-for-voice-N` (a functional sub-role, freq is
  relative not melodic). The percussion/role detector MUST recognize this case or it will corrupt the sound
  by discarding a "do-nothing" voice.
- **Topology is physical, not role.** Because the modulator is always the ring-adjacent physical voice, the
  sync/ring edges live in the **wiring graph (layer 3) keyed by physical voice index** — one place where
  physical-voice structure genuinely cannot be abstracted into roles; the role lanes reference it, they
  don't absorb it.

## Tractability ladder (by ambiguity — build low-risk first; role is LATENT)
Role assignment is a hidden variable, and **mis-segmentation fragments a line worse than physical
voice-lanes**. So do NOT infer aggressively; climb from hardware-detectable to latent:

1. **Percussion lane — FIRST (mostly mechanical).** Percussion has a hardware signature: noise waveform
   (ctrl reg `v*7+4` bit 7) and/or short drum-ADSR hits; [[control-aware-encoding]] already tags
   noise-timbre/percussion per frame. Low-error, high-value — drums are rhythmically regular and currently
   **dilute the pitched voices' prediction target**. Pulling them to their own lane is the cleanest win.
2. **Voice-wiring graph — SECOND (hardware-explicit, no latent inference).** Filter routing is in reg 23;
   sync/ring edges are in each voice's ctrl bits 1/2. Regroup both into the wiring channel: filter as a
   bound bus, sync/ring as modulator↔carrier **relationship** tokens. Surfaces the `modulator` sub-role for
   free (a voice whose only consumer is another voice's sync/ring).
3. **Melody / bass role inference — LAST, GATED (latent, lossy).** Heuristic seeds: pitch range
   (bass = low, melody = mid/high sustained), ornament signature (vibrato/slide ⇒ lead), gate/rhythm;
   `audit/extract_sid_melody.py` already extracts per-(dump,voice) melodic lines. **Must run AFTER the
   wiring graph** so a silent modulator voice is already claimed and not mis-labelled melody/bass. Build
   the segmenter ONLY if it pays off (below).

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
- **Byte-exactness** — the orchestration map + wiring graph must record role→voice→original write order AND
  the sync/ring edges to invert; the modulator's exact freq stays recoverable even when its learnable form
  is the carrier-relative ratio. Strictly harder than voice-lanes (which already flagged a round-trip blocker).
- **Sync/ring coupling (the hard constraint)** — never separate a sync/ring modulator from its carrier into
  independent absolute-freq lanes (forces non-local arithmetic, Principle 1); encode the edge as a
  relationship, and never drop/merge a silent-but-modulating voice.
- **Priors** — `voice_trajectory` (all variants) and within-voice `melody_channel_factorization` are
  already **refuted**; this is the cross-voice/role lever they pointed at, but "de-mux helps" is not a safe
  prior in this codebase — the triage gate is mandatory, not optional.
- **Sequencing** — defer behind the in-flight residual-SET stream changes (new ctrl/note-off atoms) and the
  codebook-vs-substrate decomposition; reorganizing roles on a moving stream chases a moving target.
