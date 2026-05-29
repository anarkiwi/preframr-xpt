# Ornament transfer — make ornament a learnable, transferable channel

**Status:** EXECUTED 2026-05-29 (see RESULTS). **Surprise: the parametric reframe is REFUTED as a
transfer lever — but the underlying problem is solved.** The ornament "under-generation" was a
per-frame *interleaving* artifact, not an encoding-generality failure: once ornament is given a
per-note slot (aligned to the skeleton), the **raw** per-frame encoding already emits ornament at
≈corpus rate and transfers its type distribution; a parametric descriptor does **not** beat it
(comparable JS) and over-emits a catch-all. So the lever is **per-note alignment / de-multiplexing**
(the channel-factorization structure) — the same recurring culprit as the melody arc — not a
parametric re-encoding. Parametric ornament reverts to a pure token-budget question. **Follow-up proposal (below): a mined WAVETABLE codebook to
encode the expressive RESID tail** — the wide arps are 83% cleanly cyclic, reused heavily per composer
(Hubbard 6 tables/13×), so they are compressible/round-trippable, not inexpressible; a fidelity+budget
(and maybe within-composer transfer) play, probe-ready. Cross-ref:
[`melody_channel_factorization.md`](melody_channel_factorization.md),
[`encoding_principles.md`](encoding_principles.md),
[`landed/freq_onset_channel.md`](landed/freq_onset_channel.md),
[`landed/trajectory_anchoring.md`](landed/trajectory_anchoring.md), and the
`ornament-encoding-transfer-gap` memory.

## The problem (measured)

A model trained on the interleaved skeleton+ornament stream reproduces the melody skeleton but
**does not reproduce ornament** — it emits almost none, even when sampled. The diagnosis: the
current ornament encoding (per-frame relative freq writes / FREQ_TRAJ delta samples) is the
**rendered output of a low-entropy driver program** (a SID arp wavetable, a vibrato depth+rate, a
portamento). The per-frame samples are high-entropy and tune-specific; the *program* that produced
them is low-entropy and reusable. Forcing the model to predict the rendered samples means
memorising, not learning a transferable structure — so it neither transfers across tunes nor
survives sampling (the diffuse per-step distribution loses every argmax to a peaked skeleton
interval). This is **encoding_principles P5** (alphabet/entropy at the wrong representational
source) + P6 (wrong yardstick) applied to the ornament channel.

## Evidence — what ornament actually IS (mini, 60 dumps, 58,497 note-segments)

Per held note, classify the intra-note freq writes (the ornament between gate-on note-ons):

| class | share of all segments | share of *ornamented* |
|---|---|---|
| **plain** (no ornament, <2 intra-note writes) | 72.0% | — |
| slide (monotonic ramp, range ≥2 st) | 5.3% | 19% |
| arp/vibrato (clean-periodic) | 2.4% | 9% |
| **"aperiodic"** | 20.2% | **72%** |

The dominant ornament class looked aperiodic to a strict autocorrelation test, but it is **not
unstructured**: median **4 distinct pitches**, **44% with ≤3 distinct pitches**, median range
**33 semitones** (p90 70). That is **wavetable/arp behaviour** — a *small set of note-relative
semitone offsets cycled* (often a chord: 0/+3/+7, or an octave/fifth jump), just short (median 8
writes) or irregular enough to fail a periodicity threshold. Only ~24% are too short (≤4 writes)
to fit, and the genuinely-unstructured (>~5 distinct pitches, no pattern) tail is small.

**Conclusion:** ornament is overwhelmingly **a small offset-set relative to the current note**,
plus slides. Both are **note-/transposition-invariant** and **low-cardinality** — exactly the
properties the per-frame encoding throws away, and exactly what should transfer.

## The proposal — a per-note parametric ornament descriptor channel

Encode ornament as **one parametric descriptor token per note**, aligned to the skeleton note-on
(the channel-factorization layout), expressed **relative to that note**:

- **plain** — no ornament (72% of notes; a single cheap token, makes the channel mostly trivial).
- **wavetable/arp** — the (quantised) **set of note-relative semitone offsets** + a rate bucket.
  A 0/+3/+7 arp on any note in any key/tune is the **same token** → transfers. Captures the arp +
  most of the "aperiodic" mass.
- **vibrato** — depth (≈±1–2 st) + rate bucket. (Rare but cheap.)
- **slide/portamento** — target interval (note-relative) + rate bucket.
- **residual** — a fallback "complex" token (or a short raw-delta escape) for the unstructured tail,
  so fidelity degrades gracefully rather than mis-fitting.

The descriptor decodes back to per-frame freq writes (re-render the program). This is **lossy**
(param-fit) → it lands behind the **12-SID WAV audition gate** (P1), not byte-exact round-trip.
`TrajectoryAnchorPass` already delimits the segments and partially classifies them
(`_periodic`/ramp/`pass2_collapse`); the parametric fit is a new read on the segments it bounds,
and the FREQ_TRAJ *shape* it currently stores becomes the descriptor instead of raw deltas
(the "decouple onset-from-shape" follow-up named in `freq_onset_channel.md`).

**Why this transfers and the per-frame form doesn't:** same program → same token regardless of
pitch/key/tune (invariance), and **one peaked descriptor per note** instead of N diffuse per-frame
deltas — fixing *both* the transfer failure and the emission collapse at once.

## How to measure it — transfer, not exact-token (P6)

Exact-token accuracy is the wrong yardstick for a multi-modal generative target. Measure:

1. **Emission rate** — fraction of generated note-segments that carry ornament, and the per-class
   mix, vs the corpus (~28% ornamented; the per-frame baseline collapses to ≈0). The first bar.
2. **Distributional transfer** — JS/KL between generated and **held-out** distributions of
   (primitive type, offset-set, rate). Does a model trained on train-tunes generate held-out-like
   ornament?
3. **Audition** — render generated ornament back to SID audio; A/B vs ground truth.

## Cheap probe FIRST (fail-fast, no tokenizer build, no new data)

Reuse the channel harness (`extract_sid_melody`/`melody_channel_probe`/`melody_channel_render`):

1. Add a parametric fit over the existing mini ornament segments (host-side; the classifier above
   is the seed) → one descriptor per note.
2. **A/B two ornament encodings** of the *same* held-out tunes, both as a per-note channel aligned
   to the skeleton: **(A) raw per-frame** (current) vs **(B) parametric descriptor**.
3. Train each; free-run from a 1/3 prompt; report **emission rate + held-out distributional match +
   audition** (metric above).

**Decision rule.** If **B is emitted at ~corpus rate and its (type, offset-set, rate) distribution
matches held-out while A collapses** → parametric ornament is transferable; build it in
preframr-tokens (audition-gated, descriptor replaces FREQ_TRAJ shape). If **B also collapses or its
distribution does not transfer** → SID ornament is genuinely tune-/driver-specific; stop trying to
*predict* it exactly, score it distributionally, and treat ornament as a sampled texture conditioned
on the (transferable) skeleton. Either outcome is decisive and mini-cheap.

## RESULTS (2026-05-29, `audit.ornament_transfer_probe`, mini 147 seqs, corpus ornament rate 0.233)

Per-note held-out generation, RAW (per-frame offsets per note) vs PARAM (one descriptor per note),
3 seeds. Metric: emission = fraction of generated notes carrying ornament; JS(type) = JS divergence
(bits) of the generated vs held-out descriptor-**type** histogram (PLAIN/ARP/SLIDE/VIB/RESID; for
RAW the generated offsets are re-fit to a type):

| seed | RAW emission | RAW JS | PARAM emission | PARAM JS |
|---|---|---|---|---|
| 0 | 0.181 | 0.056 | 0.417 | 0.029 |
| 1 | 0.197 | 0.033 | 0.399 | 0.054 |
| 2 | 0.224 | 0.050 | 0.343 | 0.039 |

**The decision rule's premise was wrong — neither branch fires.** (1) **RAW does NOT collapse**:
per-note emission 0.18–0.22 ≈ corpus 0.233, with low type-JS (~0.04). The earlier "model emits ≈0
ornament" (`melody_channel_render`, 6/64) was a **per-frame interleaving** artifact — ornament
diluted across 71% of token positions into a diffuse next-token distribution. **Giving ornament a
per-note slot (aligned to the skeleton) restores emission for the raw encoding too.** (2) **PARAM
does not win**: comparable type-JS, and it *over*-emits (0.34–0.42 vs 0.23) by leaning on the RESID
catch-all (~25% of its output vs 4% of corpus) — a degeneracy, not better ornament.

**Conclusion.** The ornament-generation/transfer problem is fixed by **per-note alignment /
de-multiplexing** (the channel-factorization layout) — the *same* recurring lever as the melody arc
(within-voice ornament interleave −0.032, cross-voice multiplex −0.062, Unigram weld; see
[`melody_channel_factorization.md`](melody_channel_factorization.md)). A **parametric descriptor is
NOT justified as a transfer/generalization lever** — raw per-note offsets already transfer their type
distribution, and the descriptor adds a catch-all degeneracy. Parametric ornament therefore reverts
to a **pure token-budget** consideration (one token/note vs variable), to be weighed on axis 2, if at
all — *not* a melody/ornament-quality bet.

**Audition** (`--render-dir`, held-out tune; `/scratch/tmp/enc_audition/ornament_*.wav`):
`ornament_gt.wav` (true notes+ornament) vs `ornament_raw_pred.wav` (RAW per-note model continuation —
audibly carries ornament, the point) vs `ornament_param_pred.wav` (PARAM continuation — ornament too,
but RESID-heavy). The raw per-note prediction having audible ornament *is* the progress: per-note
alignment, not parametrization.

## First principles — how the drivers generate ornament

The driver mechanics (pitch, pulse-width, filter) are background, now in their own reference:
[`sid_driver_ornament_reference.md`](sid_driver_ornament_reference.md) (read of defMON, SID Wizard,
Hubbard *Commando* + C=Hacking #5, Galway Ocean drivers). The points that shape *this* (pitch-ornament)
design:

- **Arpeggio = a note-relative semitone offset cycle**, stepped per frame, applied `note+offset` →
  freq-table lookup, reused per instrument (Galway "OFFSET LIST", SID Wizard `wf_table` `arp_byte`,
  defMON `TR` rows). ⇒ the wavetable-codebook is the right model, with **semitone-domain,
  transposition-invariant** entries — **plus an octave-arp special primitive**: Hubbard's early routine
  (Commando family) does *only* a `note / note+12` octave toggle (C=Hacking #5), the dominant Hubbard
  form.
- **Vibrato & slide = frequency-domain parametric modulation** (Hubbard depth/per-note-portamento byte;
  Galway gradient stages; SID Wizard triangle + accumulator). ⇒ encode **parametric**, *not* codebook
  entries / raw deltas.
- **Gate-on is not the note boundary** for held-gate / legato drivers — confirmed by C=Hacking #5
  (Hubbard note flag bit 6 = "appended, no attack" = legato, no re-gate). ⇒ use an **intrinsic
  level-change** note segmenter (`TrajectoryAnchorPass` pass-1), not the gate, before the codebook
  probe. (So Commando's RESID is octave-arp + legato melody + portamento under a held gate, not an
  arbitrary wavetable.)
- **Reuse by instrument/voice bank id** is real ⇒ codebook + per-composer/instrument bank conditioning.

### Net effect on the proposal
- Arp primitive → **note-relative semitone offset-cycle codebook** (as proposed). ✓ confirmed, and it
  must be semitone-domain (transposition-invariant), matching `note+offset` application.
- **Split vibrato/slide OUT of the codebook into parametric primitives** (gradient-stage vibrato,
  rate/target slide) — the drivers generate them in the frequency domain with a handful of parameters,
  not as offset tables. (Refines the earlier ≤4-set `ARP`/`VIB`/`SLIDE` taxonomy with the right
  domains.)
- **Add a level-change-based note segmenter** (reuse `TrajectoryAnchorPass` pass-1) so held-gate
  drivers (Hubbard) don't dump melody into the ornament channel — do this before mining the codebook.

## Progressing the expressive tail — a mined WAVETABLE codebook (proposal, 2026-05-29)

The RESID tail is the elaborate arpeggios of the virtuoso composers (Hubbard *Commando* 99% RESID,
DRAX *Camerock* 93%, Goto80). Measured, they are **not unstructured**:

- **83% of wide arps (>4 distinct offsets) are cleanly cyclic** — an *ordered* note-relative offset
  cycle (a driver wavetable), period ≤ ~12. Only 17% are non-cyclic (the true residual).
- They **reuse as a small per-composer bank**: Hubbard **6 distinct tables** (13× reuse, top-16 =
  100%), Jammer 31 (17×), Goto80 142 (10×), DRAX 232 (5.5×). Globally 423 tables; top-128 cover
  84%, top-256 cover 95%.

So the current `ARP` primitive fails for three *fixable* reasons, not because the content is
inexpressible: it (1) **caps distinct offsets at 4** (these have median 8), (2) encodes an
**unordered set**, discarding the cycle order that *is* the arp, and (3) **fits per-segment** instead
of recognising a reused table. (This is FIDELITY + BUDGET + within-composer arp transfer — *not* the
melody-quality lever, which is de-multiplexing. Pursue as a budget/fidelity play, or alongside the
voice-lane work, not as a generalisation bet.)

### Revisions / new primitives

1. **Replace `ARP(≤4 unordered set)` with `WAVETABLE(table-id)` from a mined codebook.** Extract each
   ornament segment's **canonical cyclic offset-tuple** (period-detect via the autocorrelation already
   in `TrajectoryAnchorPass`; rotation-normalise for dedup; note-relative ⇒ transposition-invariant).
   Mine the top-N tuples into a codebook (global + per-composer banks). Encode the segment as
   `(table-id, rate, phase, repeats)` — **one token (+small params) for what is now ~40 raw writes**
   (Hubbard *Commando*: ~7.6k ornament writes → 191 table refs). Keep `SLIDE`/`VIB`; they are fine.
2. **Table-bank conditioning (the transfer lever).** Because banks are small and reused within a
   tune/composer (driver-style), emit the bank once (header / absorber-macro) and *reference* table-ids
   — the model learns "composer/context → table-id", the reusable structure the generic ≤4 primitive
   could not expose. This is where within-composer arp transfer could actually appear.
3. **Graceful raw-delta escape** for the 17% non-cyclic remainder, so fidelity degrades cleanly and the
   codebook stays small (cover the reused 80–95%, escape the long tail; measure the escape rate).

**Distinct from the refuted `motif_pass`:** that mined the *skeleton/structural token* stream (no
compression, content-neutral → refuted). This mines the *ornament wavetable* subspace, where reuse is
*measured* and high (Hubbard 13×) — a domain-appropriate codebook, not token-BPE on the melody.

### Measure (the right axes) + cheap probe sequence

- **Coverage / budget** (host, done in part): coverage vs N + escape rate; raw-writes→tokens on the
  RESID-heavy tunes. Strong already (per-composer banks tiny).
- **Within-composer transfer** (the honest test): re-run `ornament_transfer_probe` with the
  `WAVETABLE` codebook replacing the ≤4-arp primitive, on a **single-composer split** (Hubbard-only,
  DRAX-only) — does `table-id` now cover the RESID tail *and* match the held-out distribution where the
  generic primitive over-emitted RESID? Test within-composer because banks are composer-specific
  (DRAX 5.5× reuse ≪ Hubbard 13×).
- **Fidelity audition** (12-SID WAV gate): does `decode(table-id, rate, phase)` reproduce *Commando*'s
  arps perceptually? P1 — lossy fit, audition-gated, not byte-exact.

**Decision rule.** If per-composer banks are small (they are) and the codebook+escape covers the tail
with good audition fidelity and shrinks tokens → adopt `WAVETABLE` as the ornament primitive (budget +
fidelity win), and report whether within-composer table-id transfer materialises. If within-composer
transfer still does not beat raw → keep raw per-frame (byte-exact) for fidelity and treat the codebook
purely as a budget compressor. Either way the wide-arp tail stops being "unencodable".

### Risks / open
- The **17% non-cyclic** segments need the escape; if a composer is escape-heavy, budget/fidelity suffer
  there.
- **Rate/phase quantisation** and **period/rotation detection** robustness (short or nested tables);
  reuse the anchor pass's mode-aware logic.
- **Cross-composer tables barely transfer** (DRAX's 232 tables vs Hubbard's 6) → per-composer banks,
  conditioned on a composer/engine token.

## Landing path (NOT pursued as a transfer bet — kept for a token-budget reopen)

1. Parametric ornament tokenization in preframr-tokens: per-segment descriptor relative to the
   anchored note, consuming the `TrajectoryAnchorPass` segmentation; byte-near-exact decode behind
   the 12-SID WAV audition gate (P1).
2. Quantisation grid for offset-sets / rate / slide-target, sized from the measured distributions
   (above) — keep the vocabulary low-cardinality (P1/P5).
3. Skeleton + ornament as two aligned per-note channels (the channel-factorization layout);
   ornament scored distributionally + audition (P6).
4. Token-budget delta (one descriptor vs N deltas should *shrink* tokens — a P-context-budget win to
   confirm), and content-tier read via `content_tier_report` spotlighting the ornament op.

## Risks / open

- **The unstructured residual tail** (irregular, many distinct pitches): the residual/raw escape
  must be measured — if it is large, fidelity/budget both suffer.
- **Rate/timing quantisation**: ornament rate interacts with the frame grid; coarse buckets may be
  audible (audition-gate it).
- **Fit robustness**: short segments (≤4 writes) and combined slide+vibrato need a defined
  precedence in the fitter (reuse the anchor pass's mode-aware logic).
