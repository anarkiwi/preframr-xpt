# Generator-MDL representation — the perfect, lossless SID encoding (first principles)

**Status:** PROTOTYPED + SCALE-VALIDATED 2026-06-05. A first-principles, register-agnostic generative
model of every SID write, derived as an MDL problem and proven lossless with **zero unexplained writes**
on 1580 corpus tunes + every historically-hard engine + SID-Wizard's own player output. Prototype:
`/scratch/tmp/decompose5.py` (fitter), `measure_lut.py` (pitch-LUT payoff), `swm_test.py` (SWM
compatibility). Supersedes the per-pass macro zoo and the semitone-ornament/RESID framing for the
*representation* question. Cross-ref [`sid_driver_ornament_reference.md`](../references/sid_driver_ornament_reference.md),
[`digi_detection_reference.md`](../references/digi_detection_reference.md), [`encoding_principles.md`](../references/encoding_principles.md),
memory `generator-model-prototype`. **This doc supersedes the prior per-pass pitch/ornament/melody stack**
(unified-pitch-encoding, ornament-transfer, sweep-oscillation, the melody-channel/skeleton designs, the
residual-SET workorders, and the macro-zoo triage) — all removed 2026-06-05; the generator model subsumes them.

## First principle

A SID driver is a 50/60 Hz loop that, per frame, either **looks up an indexed table** or **accumulates a
delta** into each register. So the shortest lossless description of a tune is: factor the per-frame
register stream into the *generators* a 6502 driver actually runs, reuse them by id, and let provenance
(which driver) fall out at the register log. The objective is **minimum description length** subject to
**exact** reconstruction — not "mostly structured." Hard rules (user-enforced): no singletons, no RESID /
escape hatch (a non-zero lossless escape is a renamed singleton — the mechanism set must be COMPLETE), one
uniform model over all channels (not register-by-register), hypothesize from the driver doc then prove on data.

## The representation

**Layer 0 — Channelize.** `register_state(df)` → per-frame settled values, split into independent
value-channels by driver semantics: 3 voices × {freq, pw, ctrl, ad, sr} + global {cutoff, res, modevol}.
(2SID doubles this — regs 25–49 — a channel-count extension, not a new mechanism.)

**Layer 1 — Pitch basis change (the unified LUT, lossless, NO cents quantization).** For freq channels
only, map each 16-bit word `f → (note, residual)`: `note = nearest semitone index`, `residual = f − LUT[note]`
stored **exactly**. This is a lossless bijection (residual carries everything), and it is the lever: in
semitone (log) space an octave is **additive +12**, so arp/slide/octave SHAPES are **transposition-invariant**
and collapse to one shared bank entry across pitches — measured **64.4% fewer distinct freq TABLE+ACCUM
shapes** vs absolute-freq keying (`measure_lut.py`, 530 tunes). The LUT must be **per-tune calibrated** (a
single tuning offset from `circular-mean(12·log2(f) mod 1)`): a fixed A440 LUT lands only 2.6% of frames at
residual 0; per-tune calibration lifts `|residual|≤1` to 60.7%. The residual is NOT ≈0 — the remaining ~40%
is real ornament (vibrato/slide) + off-grid transients. **Pitch encoding is an MDL choice per voice/segment,
NOT a waveform choice:** LUT-split (note+residual; best for on-grid melodies) vs raw 16-bit freq decomposed
directly (best for low/swept/percussive voices, where the note channel degenerates).

**Never classify pitch/percussion/accent by the waveform bit (the Facemorph guardrail).** Noise is routinely
used for onset **accents on *pitched* notes** (Facemorph voice 0: a 1-frame freq≈213 NOISE burst *inside* a
pulse sweep), and **pulse is used for percussion** (drum/bass sweeps). So the waveform is not a pitch signal.
The note/residual split is **uniform and lossless for every frame regardless of waveform**
(`f = recon(note)+fresid` always); the `ctrl`/waveform is an independent timbre channel, never consulted by
the pitch split. Verified: the prototype reads the waveform bit *nowhere* and is byte-exact + 0-bare on
Facemorph including every noise-tik frame. (Worked example — Facemorph v0 freq words are 54–213, far below
the LUT floor, so `note` degenerates to ~2 values and the real gesture, the 118→98→72→54 sweep, is a clean
**ACCUM on the raw freq / residual** — exactly the case the per-voice raw-vs-split MDL fallback exists for.)

**Layer 2 — Generators (the primitive set).** Each channel's series = a sequence of generators, each
producing a run of frames from O(1) params. The complete set (every driver mechanism maps onto it):

| generator | params | reproduces (driver mechanism) |
|---|---|---|
| **HOLD** | (len) | held/PLAIN value, sustained register |
| **ACCUM** | (Δ, len) | portamento, slide, skydive, freq/PW/cutoff ramp, target+duration glide |
| **SWEEP** | (step, lo, hi, dir, len) | vibrato, PW auto-reverse, filter triangle (bounded reversing zigzag) |
| **TABLE** | (cycle[p], len) | arp, octave-arp, chord, wavetable, **sine vibrato** (periodic), SoundMonitor looping freq-sweep, noise-tik cycle |
| **(END)** | — | dump-boundary marker: the tune cut mid-gesture on its final note-row |
| **(SAMPLE)** | raw block | digi PCM — a SEPARATE sub-frame modality (see below), typed not pretended-structured |

**Layer 3 — Bank (cross-tune DEF→REF).** Distinct generator shapes are interned into a shared bank,
referenced by id; transposition-invariant (LUT) + recurring instrument programs make shapes repeat.
`L = 2·instances + bank_DEF_costs` (a REF + a length per instance). Minimize L s.t. exact reconstruction.

### The decisive design rule: self-verifying fitter ⇒ lossless by construction
Every primitive candidate is accepted **only for the longest prefix its OWN decoder reproduces exactly**
(`fit_run`). A SWEEP whose fixed (lo,hi) triangle diverges from the data is truncated/rejected, never
shipped. This single rule made the prototype **100% byte-exact** and is the production contract.

## Why there are no singletons / outliers / unexplained writes
The v4 prototype's 0.04% "bare events" were **two artifacts, both eliminated**, not missing mechanisms:
1. **SWEEP over-match** (a span the decoder couldn't reproduce) → killed by the self-verifying fitter.
2. **End-of-tune truncation** — the recording stops mid-gesture, so the final note-row's per-channel values
   can't form a run. This presents as a **synchronized cross-channel jump on the FINAL frame** (Captain_Stark
   F=6166: 9 channels change at frame 6165). It is an **END marker**, not a driver mechanism.

After both fixes, with a strict `F−st ≤ 4` tail window, **every** EVENT across 1580 tunes is final-row
truncation: **0 interior, 0 anchor-needed, 0 bare.** The note-row ANCHOR (a synchronized cross-channel load)
remains the natural BLOCK header for constrained decode (below), but is not needed to reach zero unexplained.

## Evidence (all `reparse=True`, digi-excluded)

| test | result |
|---|---|
| **Scale** (step-50, 1580 tunes) | **100% byte-exact (1580/1580)**; EVENTs 2115 (0.04%) = 100% final-row trunc; **0 interior / 0 bare**; 0 register-state SAMPLE |
| **Hard engines** (Baggis/JCH, Camerock, Commando, SoundMonitor=Howard_Jones, System6581=Wow_Man) | 5/5 lossless, 0 bare — the old "RESID floor" (wide/aperiodic) is just TABLE/ACCUM |
| **SID-Wizard authored** (6 LukHash corpus dumps) | 6/6 lossless, 0 bare |
| **SWM full suite** (ALL 91 SID-Wizard 1.94 example modules, asid-vice-verified player, `swm_suite.py`) | **91/91 byte-exact, 0 bare** — feature coverage: wf_table-arp 91, pw_table 80, filter_table 79, vibrato 76, vib_delay 71, hard_restart 90, chord_table 35, octave_shift 46, gateoff_fx, multispeed, funktempo, transpose, **131 distinct per-row FX codes** |
| **defMON player** (pydefmon, byte-verified vs the real defMON binary; 9 .prg fixtures, `defmon_test.py`) | **9/9 byte-exact, 0 bare** — coverage: pitch-mod(vib/slide/arp) 9, pw_sweep(PS) 7, filter-cutoff(ACID) 9, routing(RE) 9, resonance 9, filter-mode 9, test-bit/HR 9, gate-retrig 9, waveform-walk(sidTAB) 8 |
| **defMON corpus** (46 DefMon-fingerprinted HVSC tunes, `defmon_dumps.txt`) | **46/46 byte-exact, 0 bare** |
| **Pitch LUT** (530 tunes) | note-relative keying −64.4% distinct shapes; per-tune calib `|resid|≤1` 60.7% |

## Corner cases — resolved/characterized
- **Lossless:** self-verifying fitter (above). 100%, the gate.
- **Unexplained writes:** 0 interior; the only EVENTs are END-of-tune truncation (an END token).
- **Wide/aperiodic engines:** captured as TABLE (periodic, incl. sine vibrato + SoundMonitor sawtooth) / ACCUM.
- **Digis:** `register_state` is per-frame; a digi's PCM is **sub-frame**, below this resolution. A per-frame
  SAMPLE detector correctly fires 0 (wrong granularity). Detect digi channels at **raw-write density** (the
  `is_digi` domain: vol/pw/ctrl writes-per-frame), type as a separate **SAMPLE/PCM modality**, keep OUT of the
  generator model. (Admitted digis are lossless vs register_state but that drops their sub-frame PCM — lossy
  vs audio; this is why `is_digi` gates them.) PWM digis that `is_digi` misses are a raw-write-density fix.
- **2SID / heavy multispeed:** parser declines (`StopIteration`, 29/1609) — channel-count extension.
- **LUT calibration:** per-tune tuning offset required; per-engine LUT variant for non-12-TET drivers.

## Melody learnability — a SEPARATE three-layer stack (this pipeline is only layer 1)
This encoding makes **structure** learnable and **de-ornaments** (layer 1) — but melody needs two more layers,
both in [`melody_skeleton_impl.md`](melody_skeleton_impl.md) (Pending, BLOCKED on this landing):
- **Layer 2 — interval-skeleton:** re-key freq note-onsets to **key-invariant intervals** (measured held-out
  next-interval 0.52 > cross-tune ceiling 0.41). Fixes the absolute-pitch ≈ 0 problem (P4.2).
- **Layer 3 — de-multiplex AND causally order the lanes (the DOMINANT lever):** the 0.52 was on
  *de-multiplexed* single-voice data; deployed, the voices are frame-interleaved (P3) so melody-onset ≈ 0 vs
  the ~0.34 per-voice ceiling. The real lever is **causal-DAG ordering — accompaniment roles before the melody
  role** (predict melody with its harmony in-context, P4) — so ROLE identification is the mechanism, not a
  follow-up; plain physical lanes can backfire ([`superframe_voice_lane_design.md`](superframe_voice_lane_design.md)
  / [`role_lane_factorization.md`](role_lane_factorization.md)). Untested at deployment → triage + canonical gate.
- **Layer 4 (deferred hypothesis):** surface rhythmic/harmonic determinants + scale-degree anchoring (lossy).
This pipeline is a good *substrate* for all of them (each voice's line is already a coherent unit), but
provides none — do not mistake "structure learnable" for "melody learnable."

## Open design choices (decide before production code)
1. **Pitch:** note-index + residual as two channels, vs one freq channel with note-relative bank keying.
   (Split gives the clean transposition-invariant note line; residual is a real ~40% channel.)
2. **Pitch encoding mode:** uniform LUT-split everywhere vs per-voice MDL choice of LUT-split-vs-raw-freq
   (the low/swept-voice fallback). NOT a waveform decision — waveform never gates pitch (Facemorph guardrail).
3. **End handling:** an explicit END/EOF token vs the note-row ANCHOR block header (the latter also enables
   constrained-decode BLOCKs — gen-tokens/tune median ~1.8k, p90 ~5.6k, so long tunes need blocking for 2048/8192).
4. **ctrl/AD/SR coupling:** keep as 3 scalar channels (current) vs one instrument-program tuple channel
   (the shipped `InstrumentProgramPass` codebook) — affects bank reuse, not losslessness.
5. **Digi modality:** SAMPLE container format + the raw-write detector (close the `is_digi` PWM gap).

## Mapping to the codebase (when we write it)
The generators are the macro primitives, unified: HOLD/ACCUM/SWEEP already exist as `SweepPass`-family ops;
TABLE is the wavetable/arp codebook; the bank is the DEF→REF codebook registry. This representation is the
target the whole-chip-zero arc and the ornament stack converge on — one fitter, one bank, no per-driver code.
