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

## The iterate-to-RESID-0 loop (user directive, 2026-05-30)

**Strategy shift:** stop sampling broad; instead drive RESID to **literally 0 on a SMALL sample** by
tracing every unmodelled write to its driver mechanism (consult `sid_driver_ornament_reference.md`;
where a driver is undocumented, reverse-engineer it), refining primitives/segmentation until RESID=0;
THEN expand the sample and repeat, laddering up to the whole corpus. Each rung surfaces new
mechanisms → new/extended primitives → RESID=0 → expand.

**Instrument:** `audit/probes/resid_trace.py <fixtures|N|paths> [procs] [worst_n]` — per-RESID-note
MECHANISM tracer. Labels each note's driver (via `sidid`) and a precise mechanism (percussion /
noise-accent / HELD-ARP / freq-slide / fast-run / wide-overflow / glissando / irregular), co-reading
the control register AND the raw freq word (the enriched `_resid_diag` sink now records
`(offset, ctrl, is_pitched, fn)` per frame). Prints the worst (longest) notes with full per-frame
detail (`offset:waveform`, raw `fn`) for manual tracing of anything unclassified.

**Rung 0 = the documented fixtures** (`Baggis`/`Camerock` = JCH_NewPlayer, `Gridtrap` = Crowther).
First trace (545 RESID notes) revealed the dominant JCH mechanisms — and that the freq-only archetype
names were wrong:
- **HELD-ARP** — the biggest pitched bucket "genuinely-irregular" is actually a **chord-table arp with
  wave-delay holds** (e.g. 5-note `[0,-10,-7,-12,-4]` each step held 2 frames → period 10). Missed
  only because `ARP_MAX_PERIOD=8` and the detector didn't see the expanded cycle. The decoder already
  replays an arbitrary-length period via the plain `cycle_frame_offsets`, so the full expanded cycle
  just needs to fit. **→ ITER 1.**
- **freq-slide** — the "wide irregular" giants contain perfect **linear-freq ramps** (`fn` −119/frame =
  JCH portamento); semitone-domain SLIDE can't fit (accelerating), a freq-domain slide is exact.
- **giant held-gate notes** (1300–3900 frames) concatenate arp+slide+noise — a **segmentation** gap.
- **percussion: noise** (~12%) and **noise-interleaved arps** (the `+noise/test` buckets) need the
  control-aware ornament / percussion channel.

**ITER 1 LANDED (verified, unreleased) — extend ARP to held chord-arps.** `ARP_MAX_PERIOD` 8→16
(tokens `feat/transient-tolerance`; `ARP_MAX_DISTINCT` was dead, untouched). Exact &
emulator-safe by construction — `_orn_rows`/`_reconstruct` already verify the cycle reproduces the
semitone floor or falls back to RESID, so audio is unchanged. **Fixtures RESID 545→503** (clean
period-10 held-arps absorbed); **full tokens suite 738 passed**. The 33 clean HELD-ARP that remain are
**irregular-hold / period>16** → need a true `(cycle, per-step-duration)` ARP form (ITER 3), not a
higher cap (token-bloat / false-period risk).

**ITER 2 (next) — percussion as a RECURRING-STAMP CODEBOOK (validated, user reframing).** Percussion
is NOT "a noise frame" (waveform is the wrong axis). It is a **low-entropy temporal pattern: an exact
series of register writes stamped down repeatedly in time** (a drum pattern). Detect it
ALGORITHMICALLY by viewing each voice's RESID stream as a musical scope and finding write-series that
RECUR. Probe `audit/probes/resid_percussion.py` tested 4 stamp signatures (abs exact `(fn,ctrl)` /
rel note-relative / shape contour / ctrl-rhythm) on rung 0:
- **85–90% of the remaining 503 RESID notes are recurring stamps** (≥3 identical occurrences), **~85%
  on a rhythmic GRID** (IOIs clean multiples of a base pulse: 80/160/320 frames).
- Stamps are drums — incl. **PITCHED ones the noise rule misses**: `(2973,65),(1986,65),(1251,65)…`
  (ctrl 65 = pulse+gate) is a tom/kick pitch-drop sweep ×4 gridded; `(8913,129),(37745,128)…` a noise
  hat every 160f. Waveform-agnostic, as required.
- The stamp **IS the exact write-series** → encoding a note as `STAMP(id)` is **LOSSLESS** (byte-exact
  replay). No percussion-timbre channel / approximation needed — the two-channel worry dissolves;
  this is "ornament by table-id" (driver-ref reuse/banks) made concrete. Coverage is 85% of *notes*
  but 42% of *frames* (drums are short/many; the frame-mass remainder = the long held-gate giants,
  iter 5 — a different mechanism).

**RSC design:** a per-tune mining pass finds exact `(fn,ctrl)`-series recurring ≥K (corroborated by
grid-regularity); assigns each a stamp id (a per-tune codebook); emits `STAMP(id)` for matching notes;
decoder replays the exact writes. Generalizes beyond drums to ANY exact-recurring stamp (lossless
completeness). ⚠️ This is per-tune mined-codebook infra, the SAME SHAPE as the refuted `motif_pass`
(content-acc null + a 0.23.0 perf regression) — but the GOAL differs (lossless RESID-drain, not
content-acc) and the evidence is far stronger (85% coverage, gridded, byte-exact). Confirm the infra
decision before building; reuse/repair motif-pass machinery rather than duplicate.

**ITER backlog (rung 0):** 2 = recurring-stamp percussion codebook (RSC) · 3 = held-ARP irregular
duration · 4 = freq-domain SLIDE · 5 = held-gate giant-note re-segmentation (the frame-mass).
Re-trace after each; RESID=0 on rung 0 before expanding to a random-N rung.

**SCALED TO 1500 + ENGINE-TRACED to RESID≈0 (2026-05-30).** The 10x rung confirmed the stack drains
RESID to a small tail, and that tail is **NOT irreducible — it is unmodelled ENGINES** (`sidid` on the
worst composers). Documented the residue engines in `sid_driver_ornament_reference.md`: **SoundMonitor**
(freq-domain looping sweep, RE'd from register output), **System6581** (chord-arp + per-cycle noise-tik
accent), plus an auto-profiled table (SoedeSoft / Music_Assembler / AMP / DMC / GMC / Adam_Gilmore) —
all collapse to existing primitives. Built the **auto-RE profiler** `audit/probes/resid_engine_profile.py`
(sidid-label + parametric model-fit per engine = mechanised trace-to-driver; the acceptance instrument:
drive per-engine UNRESOLVED→0). `resid_final_accounting.py` now carries a fitter per documented
mechanism (STAMP abs/rel/wild · ARP noise-incl · ARP_accent · SWEEP loop/glide · PERC · SEGMENT/DECOMP)
→ unaccounted **~0.7%**, all held-gate concatenations of KNOWN mechanisms. Literal 0 = the encoder's
segment-then-fit; the probe is a classification proxy (do not loosen fitters for a fake 0). Full build
spec for the tokens agent: **`design/IMPLEMENTATION_resid_zero_tokens.md`** (incl. MANDATORY docker /
full-CI-run-before-PR testing protocol §8.0).

## Control-aware corpus split — the audit gate LANDED (2026-05-30)

The survey is now **control-aware**: `SkeletonPass._resid_diag` (inert sink, default `None`, no
prod behaviour change) records each RESID note's frames as `(offset, is_pitched)` via the real pass's
`_is_pitched_frame`, so the probe splits contamination from melodic content faithfully (post-hoc
recovery was impossible — the post-pass block drops the `irq` frame column). Representative run
(150 dumps one-per-composer, 131 parsed, **45,937 RESID notes**, ORN share 0.140):

| | share of RESID | meaning |
|---|---|---|
| **percussion/timbre (NO pitched frame)** | **27%** | every frame noise/test → not melody at all; needs its OWN audible percussion/effect primitive (freq = drum-timbre/sweep, ENCODED not absorbed) |
| **reclaimable contamination** (pitched-core ≤2 / transient) | **~34%** | melody is ≤2 frames + noise/test; candidate for control-aware *segmentation* (lever 1) |
| **irreducible PITCHED** | **38%** | the true melodic RESID=0 target = slide 17% + sweep 5% + rebased-run 5% + octave-osc 4% + irregular 4% |

26% of RESID *frames* are control contamination (noise/test/HR). **Decisive reframing: the freq-only
0.218 share massively over-stated MELODIC RESID.** Two-thirds of RESID is non-melodic (percussion) or
control-contamination; the genuinely-irreducible melodic residue is only **~5.3% of all ORN notes**
(0.140 × 0.38). The dominant single bucket is the freq-only "short-note" archetype (31% of RESID):
**11,246 of its 14,461 notes are percussion (no pitched frame)** — it was never melody. Probe:
`audit/probes/resid_archetype_survey.py N PROCS` (default 150-dump representative; `full` = corpus).
Output `/scratch/tmp/resid_survey_ca150.out`.

**Build-order consequence — split RESID into two channels FIRST.** The unambiguous, heuristic-free
lever is the *has-any-pitched-frame* test the survey validated: route the 27% no-pitched-frame notes
to a **percussion/effect channel** (a learnable timbre+envelope primitive), removing them from the
melodic skeleton's burden with zero segmentation risk. Then lever 1 (control-aware segmentation,
`_resegment_levelchange`, already built but conservatively gated) works the ~34% contamination on a
clean melody-only stream, and the audition-gated lossy fit (lever 2) targets only the 38% irreducible
residue. Percussion (27%) is now co-equal with segmentation as a top lever and is the cleaner problem.
NOTE: "reclaimable" is a *candidate* label (contamination EXPLAINS the irreducibility) — segmentation
actually reclaiming it is the hypothesis to test next, not a settled result.

## ⚠️ Absorption safety — emulator-proven (2026-05-30)

Before discarding ANY freq write, it must be proven inaudible on the SID emulator — do not assume
from a mental model of the envelope. Reference: `preframr-audio/tests/test_freq_write_audibility.py`
(pyresidfp, 9 tests). Proven: **only a freq write on a TEST-bit frame** (oscillator held in reset)
does not reach the output. NOISE-frame freq = noise pitch/colour (audible); freq during RELEASE is
audible (release-0 is NOT instant; freq takes effect in every envelope phase); COMBINED-waveform
freqs audible; noise+pulse LFSR-locks. So the control-aware `_rebased_note` correctly picks the
melody PITCH from pitched frames, but the noise/release/transient freqs are **audible content** that
must be ENCODED (a percussion/effect channel), not absorbed to 0 — measure melodic RESID=0 on the
PITCHED content, with percussion as a separate audible channel. The "settle/prefix-strip" lever
(~28% measured) is only *safe* for its test-bit frames; the rest of a prefix is audible and must be
represented. See principle P8.
