# Transplant augmentation — donor/host melody & instrument recombination + the instrument bank

**Current status (FLAT v2):** the model-facing codec is the FLAT v2 typed-atom vocab (VOCAB=576,
`flat_serialize.py`; BACC remains the instrument primitive). Melody and instrument objects below map to the
flat equivalents (absolute-grid NOTE atoms + content-addressed `REF`/signed Δ for transposed reuse;
instrument = `GEN_*` / BACC params).

**Status:** Design. The data-side member of the generation program. Implementation home:
**preframr-aug** (owns augmentation; consumes preframr-tokens as a library — `events.oracle`,
`events.gestures`, `events.stream`; torch-free). Supersedes the *cross-song-transfer* axis of
`preframr-aug:design/melody_transfer_augmentation_design.md` (pre-event-model); the
inaudible-perturbation and voice-permutation axes remain as designed there. Sequenced with the rest
of the program: build after the canonical run verdict; the P0 instrument bank is independently
useful earlier (the [phrase compiler](prompt_interface_design.md) wants it).

## Why this, and why it should work

HVSC is a fixed corpus; the generalization target (eval_b held-out composers) is precisely
*recombination* competence — melodies the model hasn't seen, played through instruments it has, and
vice versa. In every natural training example, melody and instrument are **spuriously bound**: a
composer's lines always arrive in that composer's timbres. Transplanting breaks the binding — the
same melody under different instruments, the same instrument under different melodies — forcing the
model to factor content from timbre. This is the **data-side counterpart of P1 separability**: the
encoding makes melody and instrument *representationally* separable; transplants make them
*distributionally* separable.

The BACC codec is what makes it tractable and safe:
- **Melody is already an object:** the absolute 12-TET A440 grid-index note stream + the backward
  Transpose op — re-keying a donor melody to a host is a Transpose/anchor choice, not a rewrite.
- **Instrument is already an object:** the note-onset generator is BACC params (one BACC primitive
  subsumes ADSR/PWM/sweeps + VIB/SLIDE/ARP), and recurs near-exactly per tune (the measured ~98%
  exact program recurrence).
- **Cross-driver notes:** the absolute-grid index is driver-independent, so a transplanted voice
  carries its pitch natively — no global re-quantization.
- **Validity is free:** augmentation operates in the register-write domain and re-encodes through the
  BACC encode path (`verify=True`) — the artifact is the BACC sid/sid-trace (recoverable via
  `recover_from_sid`; no `.dump.parquet` in the shipped path), so the training pipeline needs **zero
  changes** (the runner sees a bigger corpus + list file).

**Honest scope:** unlike inaudible perturbation, a transplant has no correctness oracle — it is new
music. The gates below make it *valid* and *plausible*; only the dosage A/B makes it *useful*.

## P0 — the instrument bank (standalone value; build first)

A corpus-wide miner extracting every note-onset instrument program:

1. Per (tune, voice): segment notes (gate edges ∪ note onsets); per note collect
   the onset-anchored program as BACC params: ctrl/waveform walk (W frames from onset), AD/SR,
   HR-prep presence, PW class/trajectory, ornament class (vibrato depth/rate, arp cycle) — all
   carried by the BACC primitive.
2. Collapse exact-recurring programs within a tune (expect 3–10 instruments/tune at ~98%
   recurrence), then dedupe across the corpus: exact-key first on the program template
   (tokens `engine_fingerprint` is serviceable **only** here, as a write-domain near-exact hash —
   it measures write idiom, never sonic similarity), then sonic clustering (below).
3. Bank record: id, program template (decoded write form), provenance (tune, composer, engine
   family), usage count, role hint (bass/lead/percussion heuristic — pitch range, noise %, gate
   rhythm), sonic fingerprint + cluster id.

### Sonic clustering (so sonically-similar instruments cluster)

The engine already exists: **`preframr_audio.fingerprint_writes`/`fingerprint_batch`** (built, per
its own docstring, for "content-token clustering … `sequence_of_writes -> acoustic_fingerprint`"),
with custom `scaffold_writes` + custom feature callables + `chip_model`. The renderer **is** the
first-principles model — every nonlinearity that breaks naive parameter metrics (envelope
LFSR-compare mechanism, pulse-duty symmetry, combined waveforms, chip-specific filter) is in the
render, and the pinned chip facts dictate the protocol:

- **Instrument scaffold** (replaces the default 3-voice-triad scaffold): one target voice, others
  silent, volume 15; the program compiled at **two fixed pitches** (C2 + C4 — combined-waveform and
  filter timbre are register-dependent; fixed pitches make octave-invariance a protocol property,
  not a feature-engineering problem); per-write clocking as recorded (gate-edge sides + HR prep are
  identity); `chip_model` from provenance.
- **Cold AND warm renders.** The pinned ADSR-bug facts (`test_attack_stall_armed_by_prior_gap_compare`,
  `test_adsr_bug_attack_depends_on_prior_envelope_state`) mean an attack depends on the prior
  note's compare state — that is *why* hard restart and sexy-start exist. Render each program from
  a clean chip AND after one standardized prior note; the cold−warm delta is an identity feature
  ("prior-state sensitivity"; ≈0 for HR-prepped programs by design).
- **Feature fn** (one new pluggable callable beside `FEATURE_FNS`; existing extractors all
  time-pool, blurring the envelope — half of instrument identity): per lifecycle window
  (attack/sustain/release — window edges are *known exactly* because we author the render script):
  `band_power_features` (the repo's deliberately pitch-tolerant SID timbre descriptor — its
  docstring records why mel is wrong for bass-heavy SID spectra) + `spectral_features`; plus
  envelope time-constants from the RMS trajectory (log-time — the rate table is geometric),
  modulation reads (f0-deviation cents → vibrato; spectral-flux periodicity → PWM/walk rate), and
  the cold−warm delta.
- **Metric calibration — the `fidelity.calibrate()` precedent** (≥2× INERT/BAD separation margin,
  `test_separation_margin` style): INERT pairs = re-renders (the ±8-count nondeterminism floor),
  cross-tune exact-key duplicates (free from the 98% recurrence), **25% vs 75% pulse duty**
  (phase-inverted waveform — physics says distance 0), sub-band detunes; CONTRAST pairs = waveform
  flips, AD ±4, filter-routing toggles, drum-vs-tonal. Fit a diagonal group weighting, require the
  2× margin, freeze. Physics symmetries become metric unit tests ("the test is the citation").
- **Cluster:** agglomerative/HDBSCAN in the calibrated space; medoids carry provenance; the
  dendrogram is the patch taxonomy (role hints become a validation read, not an input). When
  ring/sync instruments are later admitted, `render_per_voice` solves their isolation (muted
  modulators' oscillators keep running).

Artifact: parquet + JSON index, **cached/regenerated, never committed** (derived from copyrighted
HVSC — the fixture policy applies; provenance pointers only). Consumers: this doc's transplants;
the phrase compiler's **patch realism** (mitigation 3 — sample a real program instead of a fixed
default); future style exemplars. Side benefit: a browsable, renderable SID patch library mined
from HVSC — auditable by ear.

**P0 gates:** coverage (% of corpus note-onsets explained by a bank entry — expect high), bank
size + dedupe ratio reported, N spot-rendered entries sound like instruments (not artifacts).

## P1 — instrument transplant (easier; build second)

Keep the host's music, replace a voice's timbre: at each of host voice v's note onsets, fire a
donor/bank instrument program instead of the host's (donor AD/SR + ctrl walk + PW behavior + HR;
host pitch, durations, phrasing; ornament defaults to donor's — in trackers vibrato/arp belong to
the patch — with a host-ornament variant flag). Mechanically: rewrite voice v's writes in the
settled/ordered-write domain from the host note list × donor program template, re-encode verified.

**Exclusions (v1, from the lane-demux wiring analysis):** never transplant onto/from a voice
entangled in sync/ring (ctrl bits 1/2 on the voice or its ring neighbor — the relationship, not the
voice, carries the music); percussion-role targets excluded (drum programs onto melodic lines is a
deliberate *variant*, not a default); filter routing stays host (global timbre context is the
host's by definition).

## P2 — melody transplant (harder; build third)

Keep the host's arrangement, replace a voice's line: donor voice's note list (onsets, durations on
its own `TICK` grid, interval sequence) drives the host voice's instrument program.

- **Re-keying:** donor intervals are key-invariant; choose the anchor note by snapping the donor's
  median pitch to the host's pitch-class histogram mode, clamped to the host voice's playable range.
  This is the riskiest musical step — keep the scorer trivial (range fit + out-of-key note count)
  and let the dosage A/B decide whether naive anchoring suffices before building consonance logic.
- **Length:** transplant whole phrases (cut at gate-off boundaries), loop/truncate to the host
  slot; role-match donor→host (lead→lead, bass→bass) via the bank's role hints.
- Same exclusions as P1; donor rhythm keeps the donor's onset timing (native in the BACC stream).

## Quality + leakage gates (all phases)

- **Structural:** BACC `encode(verify=True)` passes (automatic by construction); multispeed-aware
  framing has landed, so multispeed tunes are in scope.
- **Plausibility filter:** render + fingerprint must land within the corpus band — reuse the
  [generation quality gate](generation_quality_gate.md)'s machinery (fingerprint distance,
  write-domain structure metrics) as the *augmentation* filter; reject outliers, report the
  rejection rate.
- **LEAKAGE (hard rule):** donors AND hosts come from the **train split only** — eval_a/eval_b
  composers never contribute melody or instrument material; the augmented set carries its own
  dataset-cache hash and the metrics ledger flags any cross-hash comparison (existing discipline).
- **Provenance sidecar** per augmented tune: (host, donor(s), voice, transform, anchor) — needed to
  debug a bad batch and to keep the memorization audit honest (a generated continuation matching a
  *transplant* is still corpus material).

## The experiment (what decides if any of this stays)

Dosage A/B at canonical tier, target arm first: baseline vs +25% vs +100% transplant-augmented
train set (instrument-only first, then +melody). **Decision metric: eval_b held-out-composer
content tier** (the recombination claim is a cross-composer claim) + quality-gate no-regression +
the standard confound rules. Cheap pre-read before any training: `learnability_triage` on the
augmented vs natural corpus (the augmented set should look statistically like more corpus, not like
a new dialect — per-frame h_k and copy-fraction within the natural band).

## Non-goals

Synthetic *de novo* instruments or melodies (this is recombination of real material, not
generation); cross-engine program translation (a 6581-idiomatic program on an 8580 tune is in
scope only as the chips' natural overlap). Multispeed material is no longer out of scope —
multispeed-aware framing has landed; digi remains lower priority.
