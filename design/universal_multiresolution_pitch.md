# Universal multi-resolution pitch LUT — the learnable, lossless, transferable pitch model

**Status:** Active design (2026-06-06). Supersedes the per-tune-LUT and content-tier-residual fixes for
generator freq (`design/generator_measurement_readiness.md` §3). Validated against driver ground-truth tables
+ corpus measurement + the Cauldron II chorus case before implementation. Default-OFF flag, byte-exact +
residual-zero gated, as with all generator work. Cross-ref `generator_mdl_representation.md` (the encoding it
evolves), `learnability_token_ordering_theory.md` (why the coarse pass is the learnable target).

## 1. The problem (a trilemma the current model loses)
A single pitch representation cannot be all three at once:
- **lossless** — reproduce the exact 16-bit SID freq;
- **learnable** — small alphabet the model can predict;
- **universal/transferable** — the same pitch means the same token across tunes/trackers.
A coarse semitone LUT is learnable+transferable but lossy (needs a residual). A 16-bit-exact LUT is lossless but
a 65 536-symbol alphabet (unlearnable). The deployed generator uses a coarse ET LUT with **one per-tune scalar
tuning offset** `ref` (`generator_fit._lut`/`tune_ref`, `LUT[n]=round(2^((n+ref)/12)·_FBASE·16)`), applied to
**all voices**, plus an exact per-frame residual carried in the `GEN_TABLE` key.

**Measured consequence** (44-tune corpus sample, `/scratch/tmp/measure_lut_hypothesis.py`): the residual is
**62% static per-note** (single value, no spread) yet **only 2% within ±1** — i.e. a large, deterministic,
per-note offset. That is a *mis-calibrated anchor + per-voice detune dumped into the residual*, not modulation.
The remaining ~37% is genuine vibrato/slide spread.

## 2. There is ONE universal grid (evidence, not assumption)
Both drivable trackers' ground-truth note→freq tables ARE the universal `2^(n/12)` curve anchored at
C5≈4455 (PAL), within ±1 LSB (`/scratch/tmp/compare_luts.py`): pysidwizard `NOTE_FREQ` 96/96 within ±1;
pydefmon `NOTE_PITCH` 119/127 within ±1 (8 top-octave saturations); sw vs defmon 72% byte-identical, every
diff ≤1 LSB. So the "per-tracker LUT" is a myth — there is one canonical equal-tempered grid; trackers differ
only by ±1 rounding and an octave index offset. The grid can therefore be **shared/universal**, fixed, not
per-tune.

## 3. The model: one universal grid, multiple resolution passes
`idx = round(K · log2(freq / fbase))` against a fixed canonical `fbase` (C5 anchor), shared across all tunes.
No single `K` works, so decompose by resolution (coarse→fine), lossless by construction:

| pass | resolution | what it is | tier / role |
|---|---|---|---|
| **NOTE** | semitone (K=12) | the musical line; intervals = Δidx (transposition-invariant) | **structural — the model predicts this** (small alphabet, universal) |
| **TUNING** | fine, **per-voice**, per-tune-optimized step | sub-semitone offset = global tuning ± per-voice detune (chorus) | low-entropy (near-constant per voice); inter-voice difference = content |
| **LSB** | exact | residue to hit the exact 16-bit freq | content tier — *carried losslessly, not predicted hard* |

**Modulation** (vibrato/slide/portamento) = a **trajectory in idx-space**, emitted with the existing generator
primitives in log-idx units: slide→`ACCUM(Δidx)`, vibrato→`TRI`, arp→`TABLE(Δidx)`. "Multiple passes/atoms per
pitch event, by whatever means" — exactly the right framing; the ~37% measured spread is this.

The learnability win: the model's *prediction burden* is the small NOTE alphabet; the large alphabet only lives
in the LSB tier, which is **carried, not predicted** (the content-vs-structural split, applied to pitch
*resolution*). Universality gives **cross-tune transfer** — same pitch → same NOTE token everywhere.

## 4. Per-voice tuning + the chorus guardrail (load-bearing)
The TUNING pass is **per-voice**, not per-tune. **Cauldron II Remix (Linus)** is the proof case
(`/scratch/tmp/chorus_detune.py`): voices 1&2 are a chorus pair — 890 co-gated frames, **median +12 cents,
90% in the 1–60-cent range** (std 27c = a varying/phasing detune). That inter-voice detune **is the chorus
effect = musical content**; a single per-tune `ref` flattens both voices to one note and dumps the 12c into the
residual. The design carries it as voice 2's per-voice TUNING offset (explicit, lossless, low-entropy), and its
slow variation as a modulation trajectory. **Never normalize voices to a single tuning** (the pitch analogue of
the Facemorph waveform guardrail).

## 5. Per-tune-optimized fine resolution (with a floor)
The TUNING-pass step `K_fine` is **chosen per tune** to minimize the LSB tail (token cost), **bounded below by
the finest musically-meaningful detune in the tune** so chorus/vibrato survive — Cauldron II needs ≤~6c steps
to resolve a 12c chorus with headroom (≈1/32-semitone). On-grid tunes can use a coarse `K_fine` (tiny LSB tail);
detuned/chorus/NTSC tunes pick a finer one. The chosen `K_fine` is emitted once per tune (a few tokens).

## 6. Losslessness, tiers, gates
- **Lossless by construction:** NOTE + per-voice TUNING + LSB reconstructs the exact 16-bit freq; the fitter
  self-verifies (longest-prefix accept) as today. Residual-zero gate unchanged (`test_whole_chip_no_singleton_set`).
- **Byte-exact:** validated via `parse_audit='raise'` corpus sweep (`cb_div_audit.py`) — the standing oracle.
- **Tiering:** NOTE = structural; TUNING = low-entropy structural (or its own tier); LSB = content tier
  (de-weighted in training, scored separately) — wire op→tier in `tier_map`/`op_name_tiers`.
- **Default-OFF flag**, landed incrementally, never flips the deployed default until the triage (`--mode
  window`) shows the NOTE-pass induction-copy beats the current encoding AND the round-trip tests stay green.

## 7. Implementation outline (evolves `generator_fit` + `generator_pass` + `codebook`)
1. **Shared grid:** replace `_lut(ref)`'s per-tune `ref` with a fixed canonical `fbase`/anchor (the C5 table);
   keep `_FBASE`. `note_of`/`recon` become grid-absolute.
2. **Per-voice TUNING fit:** per voice, fit the fine offset (and `K_fine` per tune) that lands its notes on the
   shared grid; the chorus falls out as a per-voice offset, not residual.
3. **Passes as atoms:** NOTE (existing freq note), new TUNING atom (per-voice fine offset, op + content tier),
   LSB content atom; modulation reuses `SWEEP_OP`/`GEN_TRI`/`GEN_TABLE` in idx units.
4. **Codebook key:** `GEN_TABLE` freq keys on **NOTE offsets only** (de-fragments — the original §3 goal,
   achieved as a side effect because tuning/LSB leave the key).
5. **Gates:** single-tune byte-exact (`parse_audit`) → corpus `cb_div_audit` → residual-zero census →
   `--mode window` triage delta → SWM/defMON round-trip (now in tokens `tests/`).

## 8. Evidence index
- Driver tables = canonical `2^(n/12)@C5` ±1: `/scratch/tmp/compare_luts.py`.
- Residual = 62% static / 37% modulation, 2% within ±1: `/scratch/tmp/measure_lut_hypothesis.py` (broad 44-tune).
- §3 refragmentation (1.55× DEF collapse, 97.7% residuals nonzero): `/scratch/tmp/measure_refrag.py`.
- Chorus = per-voice detune content (Cauldron II, voices 1-2 ~12c/90%): `/scratch/tmp/chorus_detune.py`.
- Block-scale triage (current encoding: copy 0.916 ≤ baseline 0.930, alphabet 3.7×): `design/generator_measurement_readiness.md` §1.
