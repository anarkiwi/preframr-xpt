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
