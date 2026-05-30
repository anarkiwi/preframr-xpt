# Melodic patch preamble — the non-drum twin of the drum stamp codebook (proposal)

**Status:** Proposal (RESID=0 program; twin of `percussion_stamp_encoding.md`). Not built.

## The idea (user, 2026-05-30)

A melodic note has two orthogonal parts: its **PITCH** (already encoded as skeleton + ornament) and
its **TIMBRE** — the instrument it is played on. The timbre is a reusable **PATCH** identified by its **waveform CYCLE** (the wavetable's minimal
repeating unit) + its **ADSR** envelope. Both are reusable and few-per-tune.

**The waveform-cycle (corrected 2026-05-30).** A wavetable instrument writes a looping waveform
sequence; a note just walks the loop for its duration. So the instrument identity is the **minimal
repeating cycle**, NOT the full-length walk — exactly the ARP period fix, applied to the waveform
channel. Concrete (Baggis voice 14): the family `[0x50,0x40]×k` at lengths 8/11/14/16/40 is **one**
instrument (cycle `[0x50,0x40]` = pulse+tri ⇄ pulse), not 8; Camerock voice 0's 839 notes use just
**5** distinct progressions (`[0x80]` noise ×280, `[0x40,0x0]` ×144, …) — distinct cycle = distinct
*reusable* instrument. (An earlier draft wrongly called the waveform-progression "too volatile to be
identity" — that volatility was duration-length variation of a looping cycle; cycle-collapse removes
it.) **PW remains a separate continuous channel** (sweeps aren't note-aligned, driver-ref) — the one
thing that is genuinely not patch identity.

Encode it exactly like a drum stamp, but for melodic instruments:
- **Define the patch inline as a preamble** the first time an instrument appears; **redefinable** when
  the composer retunes/rewrites it (streaming dictionary, same as drums).
- A patch is **ambient/active**: a `PATCH_SET id` changes the current instrument; **new melodic notes
  inherit it and just carry their skeleton + ornament** (pitch). The model learns *when the instrument
  changes* (rare, low-entropy) and *the pitch line* under it — separated cleanly.
- The definition is laid out as **regular small atoms so Unigram sub-tokenizes/clusters** similar
  patches: same control-progression / same ADSR → shared sub-tokens → patches with similar
  articulation+envelope cluster in token space (the model sees two "hard-restart pulse-lead" patches
  as related).

This is the driver-native **instrument bank** (driver-ref "reuse/banks": Hubbard 8-byte instrument
records, SID Wizard `wf_table|pw_table|filter_table` per id) made explicit in the token stream — and
the twin that, with the drum codebook, factors the whole stream into **pitch (skeleton+ornament) ×
timbre (patch) × percussion (stamp)**.

## Measured basis (prototype `audit/probes/resid_patch_codebook.py`)

Per-note aux footprint extracted from the raw df (ctrl=reg+4, AD=reg+5, SR=reg+6, PW=reg+2), clustered
into patches. Rung-0 fixtures (3 tunes, 5102 claimed notes):
- **Top-8 patches cover median ~84% of a tune's notes** — a tune uses a small instrument bank; a few
  patches dominate (reuse median ~28 notes/patch). Highly compressible + learnable.
- **Dropping PW collapses the codebook 69% (541 → 166 patches)** — PW is the high-variance part (often
  note-tracking or swept); **ctrl-progression + ADSR is the stable patch CORE**. This is the
  Unigram-cluster signal: keep PW as a separate low-order param/sweep, cluster on the core.
- Common patches are recognizable articulations: `['0x41','0x40','0x20','0x9'] AD=15 SR=0` (pulse-gate
  → release → saw → hard-restart), `['0x40','0x9']`, `['0x21','0x51','0x50','0x9']` (saw→pulse blend).
- Many notes show no ADSR write in-span because **ADSR is set once per instrument and held** — itself
  evidence for the patch abstraction (the instrument is established, then reused).

**150-tune rung (133 tunes, 343,843 claimed notes) — confirms + strengthens:**
- **Top-8 patches cover median 90.9% of a tune's notes** (reuse median 62.8, p90 200 notes/patch) —
  the instrument-bank structure is even tighter at scale.
- Codebook: NEAR (ctrl-prog+ADSR) median **22 patches/tune** (p90 53), 3742 total; EXACT (incl PW)
  median 43, 8016 total → **dropping PW collapses 53%** (PW is the high-variance axis, confirmed).
- **~32% of notes (109,720) have NO aux write in-span at all** — they inherit a fully-held instrument.
  Strong evidence for the ambient-patch design (notes carry only pitch), AND the reason the held-state
  ffill refinement matters: the prototype's `None`/empty keys conflate "no write" with the actual held
  patch; ffill will RESOLVE those into their real instrument → a cleaner, smaller codebook than even
  the 22/tune measured here.

## Token design

```
PATCH_DEF id=P  [adsr=(A,D,S,R)]  [pw=…]  ctrl_prog=[c0,c1,…]      # inline preamble, redefinable
…
PATCH_SET id=P                                                       # active instrument := P
SKEL … ORN …                                                        # melodic notes inherit P
SKEL … ORN …
PATCH_SET id=Q   …                                                  # instrument change (rare)
```

- **`PATCH_DEF`** packs the patch core (ADSR + ctrl-progression) as regular atoms; PW as a param or a
  small sweep descriptor (it varies, so it is NOT in the cluster key — kept low-order so Unigram
  clusters the core).
- **`PATCH_SET`** is the ambient program-change; emitted only when the active instrument changes
  (low-entropy). Notes carry only pitch.
- The patch is **pitch-invariant** (timbre doesn't depend on the note), so it composes cleanly with the
  skeleton/ornament pitch channel — apply patch (timbre) + skeleton (pitch) + ornament (pitch-mod).

## Patch MUTATION primitives — update inline, don't redefine (user, 2026-05-30)

A full `PATCH_DEF` for a one-nibble tweak is wasteful and obscures that it's *the same instrument,
slightly changed*. So add a compact **mutation** token that updates one field of an existing patch
in-place, keeping its id (and the model's learned sense of the instrument):

```
PATCH_MUT id=P  field=SUSTAIN  val=v        # P := P with one field changed; id stays
```

Measured granularity (`audit/probes/resid_patch_mutation.py`, what actually changes note→note on a
voice; **150-tune rung, 343,469 transitions**):
- **ADSR is the stable patch core** — unchanged on **85%** of note transitions (set once, held; even
  more stable at scale than the 77% on fixtures). So ADSR is the right patch identity, and a change to
  it is the event a mutation expresses.
- Of the ADSR changes (15% of transitions), **66% are 1–2 nibble** (cleanly mutation-expressible) and
  only **33% are 3–4 nibble** (treat as redefine) — the mutation primitive is *more* valuable at scale
  than fixtures suggested (44%).
- **The mutating nibbles are RELEASE (20.3k) ≈ SUSTAIN (19.9k) > DECAY (13.4k) ≫ ATTACK (1.8k).**
  Music sense: **attack** is the instrument's identity (barely moves → an attack change ≈ a new
  instrument → redefine); **sustain/release/decay** are articulation/envelope tweaks → the MUT targets.

**So the mutation primitive set is small and ADSR-centric:** `PATCH_MUT field∈{RELEASE, SUSTAIN,
DECAY, ATTACK} val=v` (one nibble), id preserved. Rule: ≤2 nibbles differ → emit `PATCH_MUT`(s);
≥3 (or attack moves) → `PATCH_DEF` (redefine). Release/sustain/decay MUTs dominate; they
Unigram-cluster (a recurring "release→v" tweak is a reusable sub-token), and the model learns the
articulation gesture rather than re-reading a whole patch.

**Patch identity = (waveform-cycle, ADSR); mutations touch ADSR nibbles.** The apparent ~83%
note-to-note "waveform change" was two real things, neither of which makes the waveform un-identity:
(1) **duration-length variation of one looping cycle** — removed by cycle-collapse (above); (2)
genuine **switching between the few reusable instruments** on a voice — handled by `PATCH_SET`. So a
waveform-cycle change = a different (reusable) patch → `PATCH_SET`/`PATCH_DEF`, NOT a mutation; an ADSR
nibble change with the SAME cycle = a `PATCH_MUT`. **PW is the one genuinely non-identity field**
(changes ~55%, a continuous not-note-aligned sweep) → its own continuous channel, never mutated as a
patch field.

## Why this is the right shape

- **Separability / locality / no-multiplexing** (encoding_principles): the note's pitch target is no
  longer multiplexed with raw per-frame ADSR/ctrl/PW writes — timbre moves to a rare ambient token, so
  the model's per-note prediction is pitch-only under a known instrument.
- **Transfer:** the patch vocabulary (ctrl-progression + ADSR shapes) is largely cross-composer
  (hard-restart pulse-lead, saw-blend, …) → Unigram clusters them → the model transfers "what an
  instrument sounds like" across tunes.
- **Token budget:** a tune's thousands of notes' ADSR/ctrl/PW writes collapse to ~8–30 inline patch
  defs + sparse `PATCH_SET`s + per-note pitch only.
- **Lossless:** the patch carries the exact ADSR/ctrl/PW writes (byte-exact replay), gated on the
  emulator round-trip; non-reusable one-off notes keep their raw writes (escape).

## Open questions / refinements

- **Held ADSR/PW** — the patch should be the ffill STATE in effect at the note (ADSR persists across
  notes), not just writes-in-span; the prototype undercounts this (many `None`). Refine extraction to
  carry the held state → fewer spurious patch variants.
- **PW as param vs sweep** — PW is the high-variance axis; encode as a small (set/sweep) descriptor on
  the patch, or a separate PW channel (driver-ref: PW sweeps are not note-aligned, persist across
  notes — argues for a continuous per-voice PW channel, like filter).
- **Ctrl-progression duration-invariance** — the articulation is the ORDERED distinct ctrl values
  (RLE), duration carried by the note's frame span; confirm this replays exactly (gate-off timing).
- **Patch vs drum unification** — a drum stamp and a patch are the same mechanism (define + ambient/
  backref a reusable register footprint); consider one unified DEF/REF family with a `kind`
  (percussion vs instrument) attribute, so Unigram clusters across both.
- **Where PATCH_SET sits vs the FRAME/VOICE headers** — must attribute to the right voice; align with
  `voice_encoding_reference`.

## Build plan

1. Extend the patch prototype with held-ADSR/PW ffill; measure the codebook at the 150-rung +
   per-composer reuse (transfer evidence).
2. Tokens: `PATCH_DEF` / `PATCH_SET` ops + Unigram-clusterable atom layout; decoder keeps a live
   id→patch table and an active-patch register; notes apply the active patch.
3. Gate: deterministic suite green + emulator byte-exact round-trip + confirm Unigram clusters patches
   (inspect merges) + re-measure note-stream token budget.
