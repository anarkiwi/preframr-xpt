# RESID-archetype program — model them all (RESID→0), then unify

**Goal (user, 2026-05-30):** the skeleton encoding's job is *learnable per-frame pitch/freq
prediction for every patch* — melody, drum, effect, anything. `RESID` is the un-learnable escape
hatch: a per-frame trajectory the current primitives can't parameterise, dumped as raw offsets the
model can't predict. **Target: `RESID = 0`.** Every non-zero RESID is a driver mechanism we have
not modelled — trace it to the driver (sidid → `sid_driver_ornament_reference.md`), understand why
it slipped through, and add the missing *parametric, learnable* primitive. **Sequence: model every
archetype first (completeness), THEN unify the primitives into one common parametric trajectory
model** (the universal driver, [[collapse-driver-abstractions]] taken to its conclusion).

Not a content classifier (drum vs melody) — drums aren't a category to route out. A Hubbard kick
(PW pitch-drop + noise) is just one more freq trajectory that must become predictable. Waveform/gate
are the wrong axis. The right axis is **trajectory shape × domain**: movement regular in log-freq →
semitone tokens; movement regular in linear-freq → a freq-domain sweep; pick the domain where the
trajectory is low-entropy.

## Survey (probe `audit/probes/resid_archetype_survey.py`; corpus `/scratch/preframr/hvsc`)
One random dump per composer dir for driver diversity; classify every RESID note's offset sequence
into a trajectory archetype. Pilot (34 dumps, 48k ORN notes, RESID share 0.23):

| archetype | pilot share | what it is | the missing primitive |
|---|---|---|---|
| transient/attack | 36% | stable note + ≤2 brief excursions (`[62,0,62,16,62]`, `[26]`) | **transient-tolerant fit**: note = stable core; spikes = attack/grace, don't poison classification |
| genuinely-irregular | 22% | mixed (`[-2,-5,-5,32,58,58,59]`) | the **driver-tracing frontier** — decompose further, do NOT tolerate |
| rebased-melodic-run | 17% | melodic move within ~octave of a *wrong* base (`[9,0,9,-34,9]`) | **base re-assignment** + segment as notes |
| uniform-slide (overflow) | 9% | clean log-freq ramp past SLIDE's ±24/rate (`[-2,-4,-7,-11,-11]`) | **extend SLIDE range** |
| accelerating-sweep / skydive | 6% | linear-freq drop = kick/skydive (`[46,-12,-14,-17,-21]`) | **freq-domain sweep** primitive (slope in linear-freq / MSB-rate) |
| wide octave/ratio-oscillation | 6% | wide octave/arp oscillation (`[-3,34,57,34,57]`) | **OCTAVE/ARP without the ±24 clamp** |

So ~62% is structured/recoverable via five parametric extensions; the 22% "irregular" needs
driver-tracing (and includes percussion-timbre freqs, which must get their *own* learnable form, not
an escape). Worst composers (RESID=0 trace targets, ≥20 notes): Bakewell_Dwayne 0.71, AGS 0.67,
Ivory 0.56, Jensen_Henrik 0.56, TBB 0.55, Fantastic_Zool 0.45, Tichelmann_Kay 0.45,
Tron_Olsson_Mikael 0.42, Mixer 0.37, Sharp 0.36 — identify each driver via `sidid`.

## Plan
1. **Model archetypes biggest-first**, each a learnable domain-chosen primitive, re-surveying after
   each to confirm the share drops and nothing regresses: transient-tolerance (36%) → base-reassign
   + run (17%) → SLIDE-range (9%) → freq-sweep/skydive (6%) → wide-octave (6%).
2. **Drive the irregular tail to 0** by tracing the worst composers' drivers; each surfaced mechanism
   becomes a primitive (incl. a learnable percussion-timbre primitive).
3. **Unify** once RESID≈0: fold the primitives into one common parametric trajectory model
   (domain-tagged sweep/oscillate/hold/transient with low-cardinality params) — the end-state
   universal driver. The provenance-invariance test (#11.4) is the guarantee it stays uniform.

Output of survey runs → `/scratch/tmp` (never `/scratch/preframr`, the corpus/data dir).

## Strategic update (2026-05-30, from reading driver source — confirm vs the full 52k audit)

The freq-only archetype survey is **working blind to the control register**, and that mis-attributes
structured frames as RESID. The driver source (SID Wizard, defMON, GoatTracker, Maniacs of Noise,
SID Factory II, Hubbard — see `sid_driver_ornament_reference.md`) says the fix is **control-aware
role assignment**, not freq heuristics:
- **Hard-restart onset window** (test-bit + pre-HR gate-clear, ~3 frames) = the **transient/attack**
  archetype (~34%). Detect via the control register; the HR frames' freq is don't-care → absorb.
- **Noise-waveform frame = timbre, not pitch** (Wiklund *Facemorph*: a noise-tik accent on a *pitched*
  lead; also drums). A note's melodic pitch comes from its **pitched (non-noise) frames**; a note with
  no pitched frames is **percussion** (freq = drum-timbre/sweep). This addresses the wide/noise RESID
  (65% Commando / 76% Baggis of wide jumps are noise).
- The freq-MSB **SWEEP** is a named first-class effect in 3 independent drivers (Hubbard skydive / MoN
  "Tonesweep up" / SF2-d13 "Dive") — keep it core.

So the build pivots from freq-only `_rebased_note`/`_is_transient_blip` (#16 V1) to a **control-aware
pass**: co-read gate/test/waveform with freq and assign each frame's role (HR-transient / noise-timbre
/ melody / percussion) — likely collapsing transient + much of the wide/irregular archetypes at once.

**Novel-mechanism frontier** (cross-driver audit; new primitives ONLY if they actually leak to RESID
at 52k scale): target+duration SLIDE (lands exact), sine/curved VIB with delay+length, auto-triggered
"Dive" SWEEP, wavetable-index scrub (timbral). Pulse-arp + noise-tik are control/PW-channel, not pitch.

**Gate:** finalize the build order against the full 52k audit (the corpus-wide proportions + the
RESID=0 validation set) when it lands — only model what actually leaks at scale.

## Build log + the exact-primitive wall (2026-05-30)

**DONE — control-aware foundation** (tokens branch `feat/transient-tolerance`, commit `0278693`,
green, NOT released): `_context` co-reads per-voice ctrl writes; `_ctrl_at`/`_is_pitched_frame`
expose each freq frame's role; `_rebased_note` bases the SKEL pitch on **pitched frames only** (noise
bit7 = timbre, test bit3 = HR/transient). Verified on Facemorph (no SKEL note mis-based on the
noise-tik freq≈107); baggis 0.240→0.223. **Deferred:** the ornament-side noise-snap (neutralise noise
frames to offset 0) — it regressed the #13 fast-run guard (snapped noise frames create new fast-run
patterns); needs the segmentation itself to be control-aware, sized vs the audit.

**THE EXACT-PRIMITIVE WALL (key negative result).** Two cheap exact extensions were tried and are
**no-ops on the fixtures**: (1) widening SLIDE past the ±24 offset clamp (RESID unchanged), and (2) a
uniform freq-delta **SWEEP** — best-fit over a range of deltas reproduces the real wide ramps
**EXACTLY 0/9**, ≤1-frame-off 1/9, worse 8/9 (even on Commando, where Hubbard "skydive" should be a
clean MSB decrement). So the "slide-overflow"/"accelerating-sweep" survey buckets are **mislabelled** —
the real wide ramps are NOT clean parametric sweeps; they're **contaminated / concatenated / noisy**
descents (base-outlier first frames, several mechanisms merged, settling+semitone-quantisation noise).
**Implication:** exact-reproducing parametric primitives are exhausted; you cannot drive these to
RESID=0 with another lossless ORN type. The two remaining levers are:
1. **Control-aware *segmentation*** (make `_segment_notes`/`_resegment_*` ignore noise/test frames and
   cut on pitched level-changes) — cleans the base/transient/concatenation contamination so the
   *residue* is genuinely-irregular content only.
2. **Audition-gated content-tier fidelity relaxation** — for the genuinely-noisy residue, a *lossy*
   parametric fit (sweep/ramp within a small semitone tolerance) is MORE learnable than raw RESID, but
   it is a deliberate fidelity decision: gate it on the 12-SID WAV audition (does the approximation
   sound right?), not on exact round-trip. This is the point where "learnable" trades against
   "byte-exact", and it must be decided with the audit (how much leaks) + an audition (does it sound).

So RESID=0 is reachable only by (1) + a principled (2); it is NOT a stack of exact primitives. Update
the build order accordingly once the full 52k audit lands.
