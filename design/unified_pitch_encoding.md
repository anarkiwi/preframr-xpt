# Unified pitch encoding — skeleton + pitch ornament over one semitone LUT

**Status:** PROBED + IMPLEMENTED + GENERALIZATION-TESTED 2026-05-29 (see RESULTS). The encoder/decoder
is built (`audit.unified_pitch`), the cents revisit and segmentation are validated, and an
encoding-generalization test passes: the skeleton generalizes well above the prior gate-anchored
melody and ornament generates at corpus rate. Synthesises the melody/ornament arc into a single pitch
encoding:
a **skeleton** melody line + a **pitch-ornament** channel, both expressed over **one unified
note→frequency LUT in the semitone domain**, with universal, learnable, token-efficient ornament
primitives drawn from the driver model. Cross-ref:
[`sid_driver_ornament_reference.md`](sid_driver_ornament_reference.md) (the driver mechanics this is
built on), [`encoding_principles.md`](encoding_principles.md),
[`melody_channel_factorization.md`](melody_channel_factorization.md) (per-note alignment is the lever),
[`ornament_transfer.md`](ornament_transfer.md) (codebook/parametric ornament),
[`landed/freq_v0_interval.md`](landed/freq_v0_interval.md) (interval-V0),
[`landed/trajectory_anchoring.md`](landed/trajectory_anchoring.md) (level-change segmentation),
[`superframe_voice_lane_design.md`](superframe_voice_lane_design.md) (cross-voice).

## Goal

One pitch encoding that is:
- **Universal** — every element maps to a documented driver mechanism, so it covers Hubbard, Galway,
  SID Wizard, defMON (and arbitrary write-only streams).
- **Learnable** — de-multiplexed/aligned and key-invariant, so the model predicts it (or, where the
  target is genuinely multi-modal, is scored distributionally — P6).
- **Token-efficient** — bounded per-note cost, Unigram-friendly without welding pitch into shape.

## The unified semitone LUT, and the cents revisit

Every driver produces pitch by indexing a **note→16-bit-frequency table at `note<<1`** (equal
temperament, PAL). So there is a single canonical **semitone LUT** that all tunes share. The encoding
should adopt it as the pitch substrate.

**Revisiting "quantize to cents."** The content tier currently cent-bins pitch, and a prior probe
found semitone-quantizing the onset did **not** raise the prediction ceiling (P5). That result is
about *predictability*, not *representation* — and the driver model plus a direct measurement settle
the representation question:

> Mini gate-on **notes** sit a **median 1.5 cents** from the nearest semitone (69% within ±12.5 c);
> **sustain/ornament** freq writes spread to a **median 17 cents**.

So **the note is a semitone index** (cent error ≈ the LUT's own rounding), and **cent-level variation
is ornament** (vibrato/slide/arp micro-pitch in the frequency domain), exactly as the drivers generate
it (`note` via LUT, then a freq-domain delta). The revisit, therefore:

- **Skeleton note = a semitone LUT index** (lossless for the note at ~1.5 c), **not** cents. This
  removes the cent-jitter that spread one note across many tokens (separability, P1) and is
  token-efficient.
- **The cents are not discarded — they move to the ornament channel** as a vibrato *depth* parameter
  (and as arp offsets / slide targets), where they are actually generated. We stop cent-binning the
  *melody* and start cent-parameterising the *modulation*.
- Honest non-claim: this does **not** raise exact next-note accuracy (the ceiling is a data property,
  P6) — its wins are correct factoring, separability, and efficiency.

## Decompose every freq write into (note-index, residual)

Using the unified LUT, factor each per-frame voice frequency `f` as:

```
note  = argmin_n |f − LUT[n]|        # nearest semitone (the note-index domain)
resid = f − LUT[note]                # frequency-domain micro-deviation (vibrato/slide)
```

The **note-index** stream feeds the skeleton + arp channels (semitone domain, transposition-invariant);
the **residual** feeds the vibrato/slide parameters. This is the driver's own split.

## Channel A — Skeleton (the melody line)

- **Segment notes by intrinsic level-change ∪ gate-on**, not gate alone — the landed
  `TrajectoryAnchorPass` pass-1 detector. This is mandatory: held-gate/legato drivers (Hubbard:
  note-flag bit 6 = "appended, no attack") move pitch under one sustained gate, so gate-only
  segmentation dumps melody into the ornament channel (the gate-anchor-refuted finding).
- **Encode each skeleton note as a signed semitone interval from the previous skeleton note**
  (interval-V0, already built): key-invariant, low-cardinality (clusters at 0/±few), the bankable
  melody atom. First note absolute.
- This is the melody prediction target. It is genuinely multi-modal/data-limited, so **score it
  distributionally + by audition** (P6), not by exact-token accuracy.

## Channel B — Pitch ornament (one descriptor per note, aligned to the skeleton)

The load-bearing structural choice (proven): **one ornament descriptor token per note, aligned to the
skeleton note-on.** Per-note alignment is what restores ornament emission (the per-frame interleave
collapses it; see [`ornament_transfer.md`](ornament_transfer.md)). The descriptor is drawn from a
**unified, universal primitive set**, each note-relative and parametric/codebook (so:
transposition-invariant + learnable + low-cardinality):

| primitive | params | driver basis | domain |
|---|---|---|---|
| **PLAIN** | — (72% of notes) | held note, no modulation | — |
| **OCTAVE-ARP** | direction (+12/−12), rate | Hubbard fx bit 2 (`note/note+12 @50Hz`) | note-index |
| **ARP(table-id)** | codebook id, rate, phase | SID Wizard `wf_table`, Galway `FOL*`, defMON `TR` | note-index (offset cycle) |
| **VIBRATO** | depth (cents bucket), rate, delay | Hubbard depth byte, Galway/SW envelope | frequency (residual) |
| **SLIDE** | target (= next skeleton note, or interval), rate, dir | Hubbard per-note portamento, defMON LUT | frequency (residual) |
| **RESID** | short raw-delta escape | unstructured tail (~4% of notes) | raw |

Notes:
- **Arps are a mined codebook of note-relative semitone offset cycles**, decoded as
  `LUT[note + offset[frame mod len]]` — driver-faithful and transposition-invariant. Reused per
  composer (measured: Hubbard 6 tables/13×, Jammer 31/17×; top-128 cover 84% globally), so add
  **per-composer/instrument bank conditioning**. OCTAVE-ARP is the special case `{0,+12}` given its
  own short token because it dominates the Hubbard corpus.
- **VIBRATO depth is where the cents go** — sub-semitone modulation as a cents-bucketed parameter, not
  melody cents.
- **SLIDE target defaults to the next skeleton note** (portamento bends toward the next pattern note),
  so it is usually implicit (one "slide-to-next @rate" token), tying ornament to the skeleton.
- **RESID** keeps fidelity graceful and the codebook small (cover the reused 80–95%, escape the tail).

## Token layout & efficiency

Per note: **`[SKEL_interval, ORN_descriptor]`** — two tokens, both low-cardinality atoms (P1
separable). Efficiency:
- **PLAIN dominates (72%)**, so the ornament stream is mostly one repeated cheap token — Unigram
  compresses runs of `(interval, PLAIN)` *without* welding distinct pitches into compound shape tokens
  (the merge that destroyed melody learnability; see `melody_merge_split`).
- Ornament that was **~40 raw freq writes per note collapses to one descriptor** (Commando: ~7.6k
  ornament writes → ~190 descriptors). VIBRATO/SLIDE are parameters, not sample streams.
- Net: roughly **2 tokens/note** vs today's V0 + N delta samples per trajectory — a large budget win,
  to confirm against the Jetson envelope (P-context-budget).

## Decode & fidelity

- skeleton interval → running note-index → `LUT[note]` (16-bit freq).
- descriptor → per-frame freq: OCTAVE/ARP via `LUT[note + offset]`; VIBRATO via a depth/rate
  oscillator on the residual; SLIDE via an accumulator toward `LUT[next note]`.
- **PLAIN notes are byte-exact** (`LUT[note]`, ≈1.5 c from source). Ornamented notes are a **lossy
  parametric fit** → behind the **12-SID WAV audition gate** (P1), with the RESID raw escape for
  deviations. (Decode reuses the synthesis already prototyped in `melody_channel_render`.)

## Why it meets the three criteria (with evidence + confidence)

- **Universal** — every primitive ↔ a documented driver mechanism
  ([`sid_driver_ornament_reference.md`](sid_driver_ornament_reference.md)); the semitone LUT is the
  shared substrate across all engines. *(high confidence — first-principles + 4 drivers incl.
  C=Hacking #5)*
- **Learnable** — per-note alignment restores ornament emission (measured 0.18–0.42 vs the per-frame
  collapse); interval-V0 is the bankable, key-invariant melody atom (sign acc 0.009→0.66 de-merged);
  note-relative primitives transfer across keys/tunes; one peaked descriptor/note beats N diffuse
  deltas. *(structural choices high-confidence; melody exactness remains data-limited → P6
  distributional, not a claim of higher exact-acc.)*
- **Token-efficient** — semitone (not cents) shrinks the pitch alphabet; PLAIN-dominated descriptor
  stream + codebook'd arps + parametric vib/slide give ~2 tokens/note and big raw-write collapse.
  *(high confidence on direction; exact budget to measure.)*

## Composition with the rest of the encoding

This is **per voice** and **pitch only**. It composes with the orthogonal, already-scoped pieces:
- **Cross-voice** multiplexing — the bigger melody lever — is handled by voice-major lanes
  ([`superframe_voice_lane_design.md`](superframe_voice_lane_design.md)); this skeleton+ornament pair
  lives inside each voice lane.
- **Pulse-width** is a per-voice swept value, and **filter is one global channel** (controller voice +
  routing) — both per [`sid_driver_ornament_reference.md`](sid_driver_ornament_reference.md); they are
  separate trajectories (not note-aligned) and out of scope here.

## Validation plan (cheap probes first — fail-fast, no tokenizer build)

1. **LUT round-trip / residual census** (host): snap all freq writes to the LUT; measure the
   PLAIN-note exact-match rate and the residual-size distribution → confirms semitone losslessness for
   notes and sizes the RESID escape. *(Note-cent measurement above already strongly indicates this.)*
2. **Segmentation A/B** (host + audition): level-change ∪ gate vs gate-only — does Commando's melody
   reappear (RESID share drops, skeleton note count rises)?
3. **Codebook coverage / per-composer reuse** — already measured (top-128 → 84%, Hubbard 6 tables).
4. **Honest open question — arp-content transfer**: single-composer (Hubbard/DRAX) held-out, does the
   descriptor stream regenerate plausible ornament (emission rate + distributional match + audition)?
   This is the one genuinely-untested claim; score it distributionally (P6), and **do not gate the
   design on exact-token accuracy.**

Land only behind the 12-SID WAV audition gate and a content-tier read (`content_tier_report`).

## RESULTS (2026-05-29, `audit.unified_pitch` + `audit.unified_pitch_probe`, mini)

**Probe — cents revisit (LUT census).** Gate-on notes sit a **median 1.5 c** from the nearest semitone
(58% within ±6 c, inaudible); sustain/ornament writes spread to **17 c**. ⇒ the note is a semitone LUT
index (near-lossless), cents are ornament. Confirmed.

**Probe — segmentation A/B.** Level-change ∪ gate vs gate-only: **+42% more notes** recovered, and the
ornament mix shifts toward clean notes (**PLAIN 0.33→0.58, RESID 0.19→0.11, ARP 0.31→0.17**). ⇒
held-gate melody is recovered and the factoring is cleaner. (Genuinely-continuous tunes — Commando —
stay ~99% RESID either way; honest limit.)

**Implementation.** `audit.unified_pitch`: LUT, `voice_freq_events`, level-change `segment_notes`,
`_fit_descriptor` (the unified primitives), `encode_voice`, `decode_notes` (round-trips; PLAIN notes
LUT-decoded). Mini extraction: 74 lead-voice seqs, 43,970 notes, 43% ornamented.

**Generalization test** (`unified_pitch_probe`, held-out-by-dump, ×3 seeds), the same style as the
prior encoding-generalization tests:

| | seed0 | seed1 | seed2 | mean |
|---|---|---|---|---|
| **SKELETON held-out next-interval acc** | 0.472 | 0.489 | 0.592 | **0.518** |
| skeleton 2-gram ceiling | 0.412 | 0.352 | 0.456 | 0.407 |
| ORNAMENT emission (corpus) | 0.41 (0.30) | 0.54 (0.49) | 0.59 (0.49) | — |
| ORNAMENT JS(type) | 0.045 | 0.043 | 0.063 | 0.050 |

- **The skeleton generalizes**, and **beats its cross-tune 2-gram ceiling every seed** (0.518 vs 0.407)
  → genuine transfer, not memorization. It is **well above the prior gate-anchored interval melody**
  (mini-hetero 0.225, Bach 0.394; `melody_data_gap_ladder.md`) — because de-ornamenting + level∪gate
  segmentation isolate a cleaner, more learnable note line (more PLAIN/repetition). *(Honest reading:
  the win is representational separability/learnability, not that exact melody got musically easier;
  the hard interval transitions remain data-limited, P6.)*
- **Ornament generates at ~corpus rate with a matching type distribution** (low JS ~0.05) — the per-note
  channel emits ornament (no collapse) and transfers its distribution. Per-note alignment + the unified
  primitives deliver universal + learnable + token-efficient as designed.
- **Audition (model free-run, single voice — thin, do not over-read):**
  `/scratch/tmp/enc_audition/unified_{gt,pred}.wav`. These are a *model* continuation of one lead voice
  with fixed per-note duration — they undersell the encoding (fake rhythm, mono). The earlier
  "audibly melodic with ornament" framing overstated them; retracted.
- **Faithful encode→decode A/B on REAL tunes (`audit.unified_pitch_audition`)** — the representative
  test. **Render-path bug fixed (2026-05-29):** the first version hand-rolled the audio_df with a wrong
  frame period (`mode(irq)`=19592 — the dump's `irq` column is not the frame period); that render was
  broken for *both* arms. Now both go through the **production path** `RegLogParser.parse(dump) →
  render_df_to_wav` (irq=19656, the same path `ablate_pwfilter`/audition tooling use): `_raw.wav` parses
  the dump unchanged; `_unified.wav` parses a copy of the dump with only the per-voice FREQUENCY values
  replaced by the unified encode→decode (timbre/rhythm/gate untouched). WAVs
  `/scratch/tmp/enc_audition/<tune>_{raw,unified}.wav` for **Commando**, **DRAX Camerock**, **Goto80
  Baggis**. Waveform sample-correlation raw↔unified ~0.08 (a harsh phase-sensitive metric; the encoding
  clearly alters pitch where it loses ornament). **Honest per-tune:** PLAIN/ARP/OCTAVE/SLIDE notes
  reconstruct pitch+motion; **RESID voices flatten to a held note** (Commando voice 1 = 192/193 RESID →
  its fast arp becomes a sustained note), and **VIB notes lose the wobble** in the current LUT-only
  decode (Commando voice 0 has 26 VIB). So the encoding carries melodic + structured-ornament content
  and loses the unstructured-arp tail + sub-semitone vibrato — the RESID/VIB gaps the codebook and a
  residual/vibrato channel (`ornament_transfer.md`) would close, not yet built.

**Verdict:** the unified encoding is built, validated, and carries learnable, transferable pitch
structure end-to-end. Remaining before a tokenizer build: the ARP-content codebook + per-composer bank
(single-composer transfer, P6/audition), byte-exact round-trip + RESID-escape sizing, and the
12-SID WAV gate.

## Audition follow-up (2026-05-29) — waveform-awareness gap found and the real blockers

Listening to the faithful A/B exposed two encoder gaps (the render-path bug above was separate):
1. **The encoder pitched percussion.** It treated *every* voice's freq as a melodic note, ignoring the
   **waveform (control register)**. Noise-waveform voices (drums) got snapped to LUT semitones → a
   high pitched note instead of percussion (DRAX *Camerock* voices 0/1 are 50%/42% noise; Commando
   voice 2 is 33% noise). **Fix:** the encoder must read the waveform and treat **noise as a distinct
   PERC class**, not a pitch.
2. **RESID/VIB flattened to a wrong sustained pitch.** Commando voice 1 (a sawtooth arp) is 192/193
   RESID; holding one base note put it **~1192 cents (an octave+) off** → audibly out of tune. VIB
   notes lose their wobble (held base).

The audition now substitutes freq **only for non-noise voices on faithful descriptors** and passes
the rest through. **Per-voice pitch diagnosis (2026-05-29) corrected the root cause:** noise
passthrough works (Δ=0 on noise frames), but the **ARP/SLIDE *decoder* is wrong** — it sorts the
offsets (losing cycle order) and uses a fixed rate/phase, overshooting; Camerock voice 0 (43% ARP)
came out **+3 st median, 30% an octave-plus too high** = the "high notes". Commando came out
pitch-identical to raw (Δ≈0). **Restricting substitution to PLAIN only collapses divergence to 0% on
every voice** — confirming the note representation/LUT/noise-passthrough are correct and the
**ornament audio-decoder is the sole failure**. So the current faithful audition substitutes only
clean held notes (≈ original) and passes ALL ornament + percussion through — a **passthrough crutch**,
not a real encoding.

**Root cause of the out-of-tune / octave-high notes (2026-05-29): the encoder read RAW freq bytes.**
`voice_freq_events` originally emitted a freq event on every lo *or* hi register write, so a lo write
with a stale hi byte produced a half-updated garbage pitch (offsets like +57..+71 semitones) that
contaminated the ornament descriptors → octave overshoot on decode. This bypassed the parser's
existing 16-bit freq collapse (`RegLogParser._combine_regs` / `freq_unq`, which ffills each byte and
keeps the settled value per time bucket). **Fix:** sample the **settled per-frame 16-bit freq** (carry
both bytes, never read mid-update). Result: Commando garbage offsets → 0 and its octave arps now
classify as OCTAVE (233) instead of being flattened to RESID. Multi-speed/dense tunes (Camerock) still
show some cross-frame *straddle* garbage (a freq update whose lo/hi land in different frames); the fully
correct fix is the parser's whole-series bucketed combine (`_combine_reg`), to be wired in (it ffills
each byte independently, immune to straddle). **Lesson: never re-derive freq from raw lo/hi bytes — use
the parser's collapsed 16-bit value.**

**Residual driven to the semitone floor + generalization re-run (2026-05-29).** After the settled-
freq fix, the audition substitutes EVERY freq write with the nearest semitone of the settled 16-bit
value (`build_unified_dump`), i.e. the integer-semitone resolution the encoding represents — so
whatever the SID played (arps, wide jumps) is reproduced at semitone resolution, uniformly across
pitched and percussion voices (noise waveform preserved). **Audio residual is now at the integer-
semitone floor; the remaining residual is sub-semitone vibrato** — the cents the design routes to the
(not-yet-built) vibrato channel. Reaching *literal* 0 requires that vibrato/cents channel (list item
5); per-primitive faithful DECODE (ARP/SLIDE) was made moot for audio by snap-all (it reproduces the
real motion directly). Generalization re-run on the cleaned encoding (settled freq, octave arps now
classified): **skeleton held-out next-interval 0.544 ×3 seeds (beats 2-gram ceiling 0.40 every seed,
up from 0.518); ornament emits at ~corpus rate, JS(type) 0.03–0.07.** WAVs Commando `dd9bc9…`,
Camerock `cc8373…`.

**Sub-semitone VIBRATO/CENTS channel built (2026-05-29) → residual ~0.** The decode now reconstructs
each freq as `semitone + cents` (`fn_from_note_cents`, cents quantised to CENTS_RES=4c) instead of
snapping to the integer semitone — so vibrato is preserved, not flattened. Reconstruction is per
512-bucket (one coherent 16-bit value per bucket) to avoid writing lo/hi bytes from different
reconstructions. **Result: Commando audio residual ~0 (median 0.2c, p90 1.6c, 0% >1.5 st);** Camerock
median 0.5c / p90 2.8c (94% transparent) with a **6% tail** of single-byte (lo-only) glissando updates
whose reconstructed value crosses a hi-byte boundary — in-place byte patching can't make those
coherent; the clean fix is to **re-emit** the freq stream from the combined value (add the needed
lo+hi writes), not patch in place (separable follow-up). The cents channel is an audio-decode addition;
the token stream / generalization is unchanged (skeleton 0.544; a per-note VIB-depth token already
exists for the structured view).

**Literal-0 residual (re-emit) + vibrato token (2026-05-29).** (1) The in-place byte patch left a ~6%
tail on dense tunes (a lo-only glissando update whose recon crosses a hi-byte boundary). Fixed by
**re-emitting** each reconstructed 512-bucket as a coherent lo+hi pair (`build_unified_dump`): Camerock
*and* Commando now median 0.2–0.5c, p90 ≤1.6c, **0% >50c, 0% >1.5 st — audio residual at literal 0**
(within the 4c cents quantum), uniform across pitched and percussion voices. (2) A per-note **VIB token**
(sub-semitone cents-amplitude bucket 0/1/2) is folded into the stream (`SKEL, DESC, VIB`). Generalization
×3 seeds: skeleton 0.506 (beats its 2-gram ceiling every seed; slight dip from 0.544 as the 3-token
stream dilutes skeleton context); ornament emits at corpus rate (JS 0.03–0.07); **VIBRATO emits at
~corpus rate (0.16–0.32 vs 0.15–0.36) with depth-distribution JS ≈ 0.000–0.002 — vibrato transfers
near-perfectly.** So the full pitch encoding (skeleton + ornament + vibrato) is faithful (literal-0
audio) and its structured tokens transfer.

**The real gap is faithful ornament DECODE** (distinct from the token encoding, which the
generalization probe measured): ARP needs the *ordered* cycle + rate + phase (the codebook), SLIDE
needs real target/rate, VIB needs an oscillator. Until those decoders exist, only the held-note
skeleton is faithfully decodable.

**Percussion needs NO special path (confirmed 2026-05-29).** Noise voices were initially passed through
specially; removing that and running them through the *same* pitch encoding is fine — the **waveform is
a separate timbre channel** (ctrl untouched), so a semitone-snapped freq on a noise voice still renders
as percussion, and drums are mostly *swept* (RESID/SLIDE) so they pass through as residual anyway. So
the encoding is uniform across pitched and percussive voices; "percussion vs pitched" lives in the
waveform channel, not the pitch encoding.

**Residuals (what the encoding does NOT yet capture), measured PLAIN-only:**
- **Ornament residual (the big one):** ~56% (Camerock) / ~66% (Commando) of notes are
  ARP/SLIDE/VIB/OCTAVE/RESID — currently passed through (original freq), not encoded, because their
  *decode* isn't built. Closing this = the ornament codebook + slide/vibrato decoders.
- **Sub-semitone snapping residual:** each held (PLAIN) note is snapped to the nearest A440 semitone,
  discarding up to ~50 c (median ~1.5 c). Per-note inaudible on average, but a **systematically
  detuned tune** (≠ A440 / non-equal-tempered) would shift uniformly — a per-tune tuning offset (or
  finer skeleton pitch) may be needed; measure before relying on semitone-exactness for all tunes.

**Blockers to close before re-running generalization** (so the roundtrip is legitimate, not passthrough):
- **PERC class** — waveform-aware encode: noise notes → a `PERC` token (decoded as a drum/noise hit),
  the smallest and clearest fix; it caused both audible problems.
- **ARP codebook + per-composer bank** — so RESID arps encode/decode instead of flattening
  ([`ornament_transfer.md`](ornament_transfer.md)).
- **VIBRATO decode** — a residual/depth oscillator channel so VIB notes wobble instead of holding flat.

## Risks / explicit non-claims

- **Not a melody-accuracy bet.** Exact next-note acc is data-limited and will not rise; the wins are
  factoring, emission, transfer-of-*plausible*-ornament, and tokens. (Don't relitigate P5/P6.)
- **Parametric vs raw ornament is a budget/fidelity choice, not a transfer win** — the ornament probe
  showed per-note *alignment* is the lever, not parametrisation. The codebook is justified by driver
  truth + reuse + budget, not by a transfer claim.
- **Fit/segmentation robustness**: combined slide+vibrato, short segments, and controller-less held
  gates need defined precedence (reuse the anchor pass's mode-aware logic). RESID escape must stay
  small (measure).
- **LUT edge cases**: non-equal-tempered or detuned engines widen the residual; the escape + a
  per-engine LUT variant cover it.
