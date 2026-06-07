# Recovered-table pitch model — the learnable, lossless, transferable pitch model

**Status:** Active design (2026-06-06, revised). Supersedes the per-tune-LUT and content-tier-residual fixes
(`design/measurement/generator_measurement_readiness.md` §3). **Mechanism corrected (user-led):** a tracker plays note N as
an EXACT entry from its note→freq table, so the residual is not per-frame noise to quantize against a fixed grid
(the earlier sub/LSB framing) — it is the table to **recover**. Validated on real corpus + the Cauldron II
chorus. Default-OFF flag, byte-exact + residual-zero gated. Cross-ref `generator_mdl_representation.md` (the
encoding it evolves), `learnability_token_ordering_theory.md` (why the NOTE index is the learnable target).

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

## 3. The model: shared note index + recovered per-voice table
The note **index** `n = round(12·log2(freq/anchor))` is the universal abstraction — note 49 ≈ C5 in every tune,
intervals = Δn (transposition-invariant). The exact pitch is a per-voice **recovered table** `table[n] = the
exact 16-bit freq the voice uses for note n` (the tracker's table + the tune's tuning + per-voice detune),
recovered as the modal freq per note over the voice's frames. Lossless for any table: `resid = freq − table[n]`,
`freq = table[n] + resid`.

| component | what it is | role |
|---|---|---|
| **NOTE index** | shared-grid semitone; the musical line | **structural — the model predicts this** (small universal alphabet, transferable) |
| **per-voice TABLE** | the exact note→freq map (~20 entries/voice), emitted once; a small codebook (usually 1, +1 per chorus detune) | recovered exactly; makes static notes PURE |
| **residual** | `freq − table[note]`, **0 for static notes**, nonzero ONLY for genuine modulation | content — exists only where there is real vibrato/slide |

**Static notes are PURE** (residual 0) — measured **83% of voiced frames** on real corpus (vs 20% for the
discarded fixed-grid framing). The model predicts the NOTE-index stream (universal, mostly the whole signal);
the table is a cheap per-tune dictionary. **Modulation** (the ~17%) = a residual **trajectory** emitted with the
existing generator primitives: slide→`ACCUM`, vibrato→`TRI`, arp→`TABLE` (around/between table notes).

Why the earlier fixed-grid + sub/LSB framing was wrong: it approximated each freq against a universal log grid
and carried the gap as an LSB, so ~70% of frames spuriously looked "modulated." Recovering the tune's own table
collapses that to the true ~17%, because the freqs were exact table lookups all along.

## 4. Per-voice table + the chorus guardrail (load-bearing)
Tables are **per-voice**, not per-tune. **Cauldron II Remix (Linus)** is the proof case
(`/scratch/tmp/chorus_detune.py`): voices 1&2 are a chorus pair — 890 co-gated frames, **median +12 cents,
90% in the 1–60-cent range** (std 27c = a varying/phasing detune). That inter-voice detune **is the chorus
effect = musical content**; a single per-tune `ref` flattens both voices to one note and dumps the 12c into the
residual. The design carries it in voice 2's **recovered table** (its entries are the detuned exact freqs), so
the two voices share the NOTE-index stream (the unison line) but reference distinct tables — the +12c lives in
the table values, exact, and Cauldron II decodes 75% pure. **Never normalize voices to a single tuning** (the
pitch analogue of the Facemorph waveform guardrail).

## 5. Table recovery (replaces the discarded per-tune resolution knob)
There is no resolution/`K_fine` knob to tune — the table holds the EXACT 16-bit entries, so there is no
quantization to trade off. Recovery: assign each frame a note index (`note_index`), and `table[n]` = the modal
exact freq for note n over the voice's frames (the entry the tracker actually wrote). ~20 entries/voice. A note
seen only modulated still gets a table anchor (its modal freq) and the modulation rides as residual. Per-voice
tables form a small codebook (DEF→REF): non-chorus tunes share one table across voices; a chorus adds a second.

## 5b. Transfer: base pitch and effects use two relative encodings
The model transfers pitch across tunes via two complementary normalizations (see also
`learnability_token_ordering_theory.md`):
- **Effects → CENTS relative to the note** (tuning-invariant): modulation is emitted as `mod` in cents from the
  note's table entry + an exact `closure`. A +Xc vibrato/slide is the SAME tokens at any tuning, so the model
  learns the gesture, not absolute freq. (`decompose_voice` does this; proven by the tuning-invariance test —
  same gesture at note 36 vs 72 → identical cents, raw deltas differ >20.)
- **Base pitch → INTERVALS (Δnote)** (transposition-invariant): the absolute NOTE index anchors register
  (C5=49 everywhere), but cross-*key* transfer is the **interval** stream (`MELODY_INTERVAL`, the melody layer).
  A melody transposed +2 semitones in a tune tuned +44c yields the IDENTICAL interval stream (the +2 cancels in
  the difference, the +44c lives in the per-voice tuning token). **The interval op must key off this universal
  `note_index`, not the old single-`ref` `note_of`** (it currently doesn't — the integration gap, §7).
- **Per-voice tuning** (the tune's detune, chorus) is genuinely tune-specific and is NOT transferable — it is a
  small per-voice attribute token, kept OUT of the note/interval stream so it doesn't pollute the transferable
  content. Ceiling: absolute onset pitch (first note / key) stays high-entropy ≈0 next-token — intervals fix
  within-melody transfer, not the absolute anchor (distributional/audition scoring, not exact argmax).

## 6. Losslessness, tiers, gates
- **Lossless by construction:** NOTE index + per-voice TUNING + per-frame `mod` (cents) + `closure` reconstructs
  the exact 16-bit freq (`reconstruct == freq`, proven on real corpus + 4 trackers). Residual-zero gate
  unchanged (`test_whole_chip_no_singleton_set`).
- **Byte-exact:** `parse_audit='raise'` corpus sweep (`cb_div_audit.py`) — the standing oracle.
- **Tiering:** NOTE index / interval = structural (the model's prediction target); per-voice TUNING = low-entropy
  attribute; `mod` cents = structural-ish (transferable gestures); `closure` = content tier (carried, not
  predicted) — wire op→tier in `tier_map`/`op_name_tiers`.
- **Default-OFF flag** (`universal_pitch`), landed incrementally, never flips the deployed default until the
  triage (`--mode window`) shows the NOTE/interval induction-copy beats the current encoding AND the tracker
  pitch-recovery + round-trip tests stay green.

## 7. Implementation — the gated `universal_pitch` wiring (active; this session owns it)
Foundation landed (`feat/universal-pitch-grid`, `preframr_tokens/macros/pitch_grid.py` + tests):
`note_index` / `voice_tuning` / `note_freq` (shared grid-recon) / `recover_table` / `decompose_voice` /
`reconstruct` — lossless, validated bit-exact on real corpus + 4 trackers. The pass change is one coupled,
byte-exact, **default-OFF `universal_pitch`** flag (NOT in `REGISTERED_MACROS`) replacing the single per-tune
`ref` (`tune_ref`/`note_of`/`recon`) with per-voice tuning throughout:
1. **Per-voice tuning:** compute `pitch_grid.voice_tuning` per freq voice; emit a `GEN_TUNING` atom **per voice**
   (add a `VOICE` subreg) instead of one per tune.
2. **Decoder:** `state.gen_ref` becomes per-voice; the `GEN_TUNING` decoder stores by voice.
3. **Note assignment (interval unification = the load-bearing step):** `_melody_rows`/`_stability` and the
   `MELODY_INTERVAL` decoder use `pitch_grid.note_index(f, voice_tuning)` + `note_freq` (the residual stays
   exact). This fixes the chorus/detune bug and makes intervals transferable; the interval op now keys off the
   universal index, composing pitch + melody.
4. **Arps:** `GEN_TABLE` freq keys on NOTE offsets only via the same per-voice `note_index` (de-fragments — §3).
5. **Causal ordering:** connect `role_lane` → `lane_rank` → `voice_lane` in the pass (they landed #70 but are
   unwired) so melody is emitted after its accompaniment.
6. **Gates:** `parse_audit='raise'` one `universal_pitch`-on tune → corpus `cb_div_audit` → residual-zero census
   → `learnability_triage --mode window` NOTE/interval-copy delta → the tracker pitch-recovery + round-trip
   tests. Default unchanged until the triage shows a gain.

## 8. Evidence index
- **EXACT tracker pitch + note recovery (strict, not within-cents):** SWM/defMON/Hubbard recover the trackers'
  literal pitches bit-exactly AND the recovered note index == the tracker's OWN table index (`FREQTBL`/
  `NOTE_PITCH`) on sustained frames — **40/40 SWM voice-traces 100%** (`/scratch/tmp/swm_note_correctness.py`,
  validated against the player's internal note state); Hubbard ET-residual 0.5c, Galway recovers (heavy gradient
  vibrato → modulation). Committed: tokens `tests/test_tracker_pitch_recovery.py`. **The earlier 44–96% "miss"
  was a validation bug** — comparing to the player's base pattern note `v.note`, which the wavetable/arp
  transposes; vs the actual table index it is 100%.
- **Per-voice tuning** fixes detuned tunes (Galway ~+44c) + chorus; fit over DISTINCT held freqs so arp/vibrato
  excursions don't skew it (`/scratch/tmp/hubbard_galway_pitch.py`, `swm_note_correctness.py`).
- **Recovered table → 83.1% PURE notes** (30 tunes, 11/30 100% pure, ~20 entries/voice): `/scratch/tmp/recover_table.py` — the decisive validation that trackers use pure-note tables under everything.
- Driver tables = canonical `2^(n/12)@C5` ±1: `/scratch/tmp/compare_luts.py`.
- Residual is per-NOTE deterministic (the table), not per-frame noise — 62% static / 2% within ±1 of the *universal* grid (why a fixed grid fails; the table fixes it): `/scratch/tmp/measure_lut_hypothesis.py` (44-tune).
- §3 refragmentation (1.55× DEF collapse, 97.7% residuals nonzero): `/scratch/tmp/measure_refrag.py`.
- Chorus = per-voice detune content (Cauldron II, voices 1-2 ~12c/90%): `/scratch/tmp/chorus_detune.py`.
- Block-scale triage (current encoding: copy 0.916 ≤ baseline 0.930, alphabet 3.7×): `design/measurement/generator_measurement_readiness.md` §1.
