**Status:** **REINSTATED 2026-06-05 — the harder/truer form of LAYER 3 of the melody plan** (cross-ROLE
de-multiplexing). The generator-MDL ([`generator_mdl_representation.md`](generator_mdl_representation.md)) +
interval-skeleton ([`melody_skeleton_impl.md`](melody_skeleton_impl.md)) make each voice's line clean +
key-invariant but emit it frame-interleaved; de-multiplexing into contiguous lanes is the dominant melody
lever (P3). [`superframe_voice_lane_design.md`](superframe_voice_lane_design.md) does it by physical VOICE;
this doc argues the truer target is musical **role** (melody/bass/harmony/percussion), because roles HOP
voices — a fixed voice-lane splits one melodic line and welds two roles. **Role identification is the MECHANISM
that makes de-mux actually help, not a follow-up:** layer 3's real lever is **causal-DAG ordering —
accompaniment roles BEFORE the melody role** (so the melody is predicted with its harmonic context in-context,
P4), which is impossible without knowing which lane is which. Voice-lanes are the byte-exact substrate; the
role/causal-order is what turns contiguity into a melody win (plain physical lanes can backfire by pushing the
harmonic determinant out of locality). Learnability-framed by
[`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md); grounded in
[[control-aware-encoding]] (the ctrl reg already tags per-frame role). (Was wrongly deleted in the 2026-06-05
consolidation; restored — it is complementary to the generator-MDL, not superseded by it.)

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

1. **Role lanes** (the learnable channels): `melody`, `bass`, `harmony/arp`, `percussion` — each collects
   its role's events with role-internal continuity, so the model predicts *within role* (the structure
   that generalizes). This is the cross-voice de-mux the melody-channel-factorization experiment pointed to
   (within-voice factoring was a minor bonus; the dominant cost is cross-voice/structural multiplexing —
   deployed melody-onset ≈ 0 vs the ~0.34 per-voice ceiling). **Modulation is NOT a role here** — it's
   a wiring *function* (layer 3) orthogonal to a voice's audible role: a voice can play melody/bass *and*
   simultaneously have its oscillator feed the next voice's sync/ring.
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
SYNC/RING bit sits in the *modulated* voice's ctrl reg). Sync/ring consume the source voice's
**oscillator frequency**, which is independent of that voice's gate/waveform/envelope — so **voice N's
audio depends on voice N−1's frequency** whether or not N−1 is also audible. This breaks the
"independent lanes" assumption and constrains the design:

- **Modulation is a dual function, not a role.** The common case is **dual**: voice N−1 plays its own
  line (melody/bass/…) *and* its oscillator simultaneously feeds voice N's sync/ring. The degenerate case
  is a **silent modulator** (N−1 gated off, its only purpose is to drive N). So "modulator" is an **edge**
  in the wiring graph, layered on top of whatever audible role the voice has — never a role that displaces
  it. The role detector must allow a voice to be melody *and* a modulation source at once.
- **The edge references the source's existing freq — no duplication.** Voice N−1's frequency already lives
  in its role lane (its own note). The sync/ring edge just *points at it* and expresses voice N's carrier
  as a **relationship** to it (interval/ratio). So there's nothing extra to model for the source — and for
  the silent case, the source contributes *only* a freq (its lane has no audible content, just the value
  the edge consumes).
- **Lanes are coupled; encode the relationship.** Only the ratio/interval carries intent (a sync-sweep is
  a relative gesture; a fixed ring ratio is a timbre). Splitting carrier and source into independent
  *absolute*-freq lanes forces the model to recover the relationship by cross-token arithmetic across the
  lane gap — the non-local dependency Principle 1 says it can't. Encode the edge as `carrier = source ⊕
  interval`, co-locating the dependency; keep exact freq recoverable for byte-exactness.
- **Never drop a source voice.** A silent-but-modulating voice has no audible line but is load-bearing —
  the percussion/role detector must not discard or merge a "do-nothing" voice whose freq drives an edge.
- **Topology is physical, not role.** The source is always the ring-adjacent physical voice, so sync/ring
  edges live in the **wiring graph (layer 3) keyed by physical voice index** — the one place physical-voice
  structure genuinely cannot be abstracted into roles; the role lanes reference it, they don't absorb it.

## Tractability ladder (by ambiguity — build low-risk first; role is LATENT)
Role assignment is a hidden variable, and **mis-segmentation fragments a line worse than physical
voice-lanes**. So do NOT infer aggressively; climb from hardware-detectable to latent:

1. **Percussion lane — FIRST (mostly mechanical).** Percussion has a hardware signature: noise waveform
   (ctrl reg `v*7+4` bit 7) and/or short drum-ADSR hits; [[control-aware-encoding]] already tags
   noise-timbre/percussion per frame. Low-error, high-value — drums are rhythmically regular and currently
   **dilute the pitched voices' prediction target**. Pulling them to their own lane is the cleanest win.
2. **Voice-wiring graph — SECOND (hardware-explicit, no latent inference).** Filter routing is in reg 23;
   sync/ring edges are in each voice's ctrl bits 1/2. Regroup both into the wiring channel: filter as a
   bound bus, sync/ring as source↔carrier **relationship** tokens. Surfaces the modulation **edges** + the
   silent-only-source case (wiring-only voice) for free — without forcing a role on dual-function voices.
3. **Melody / bass role inference — LAST, GATED (latent, lossy).** Heuristic seeds: pitch range
   (bass = low, melody = mid/high sustained), ornament signature (vibrato/slide ⇒ lead), gate/rhythm;
   `audit/extract_sid_melody.py` already extracts per-(dump,voice) melodic lines. **Runs AFTER the wiring
   graph**, and assigns an audible role to *every* voice with audible output — **including dual voices that
   also feed a sync/ring edge** (modulating ≠ no role); only a silent-only source gets no role lane. Build
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
- **Priors** — `voice_trajectory` (all variants) and within-voice melody-channel factoring are already
  **refuted/minor**; this is the cross-voice/role lever they pointed at, but "de-mux helps" is not a safe
  prior in this codebase — the triage gate is mandatory, not optional.
- **Sequencing** — this is **layer 3**, built on the generator-MDL + interval-skeleton (layers 1–2). Start it
  only after the interval-skeleton lands; defer behind the in-flight
  codebook-vs-substrate decomposition; reorganizing roles on a moving stream chases a moving target.
