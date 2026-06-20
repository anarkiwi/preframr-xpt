# Cross-driver note unification + tuning factoring — design note

**Status: PROPOSAL (2026-06-20).** Makes every driver emit notes in ONE representation on ONE pitch axis,
so a model sees *the same note as the same token* regardless of driver, with tuning resolved as a separate
parameter. Motivated by the asymmetry the GoatTracker backend exposed (PR #93): Hubbard notes are
structured relative-interval tracker rows; GoatTracker notes are raw `.SNG` bytes. Builds on the landed
step/tracker codec ([`sid_player_decompiler.md`](sid_player_decompiler.md)) and the absolute-anchored
12-TET pitch encoder (memory `libsidplayfp-groundtruth`). Subordinate to the LEARNABILITY north star
(`design/README.md`): the win is a driver-invariant note alphabet, not a token-count change.

## The two problems

1. **Representation split.** `serialize.py` (Hubbard) factors the score into per-voice rows
   `(dt, note_interval, instr_ref, lnth, porta)` + inline backward `REPEAT`, pitch as a **relative
   12-TET semitone interval**, pitch-invariant instruments defined once. `gt_serialize.py` (GoatTracker)
   stores the reconstructed `.SNG` module as **raw bytes** (2 LEB digits/byte); a note is an *absolute*
   GoatTracker pattern note byte buried in pattern data. Same LEB alphabet, totally different structure —
   a model trained across both sees two distributions for the same musical content.

2. **Tuning split.** Even once both are "notes," the *pitch axis* differs. Hubbard's note index is an
   offset into Hubbard's ET-but-±4c note table; GoatTracker's note byte indexes GoatTracker's freq table;
   two drivers (or two tunes) at slightly different base tuning (A=440 vs A=438), or with different ET
   rounding, would map "concert C-4" to *different* token values if pitch carried any table-specific or
   frequency-specific information. We want C-4 → the same token everywhere.

## The goal

ONE note encoding, ONE canonical pitch axis, tuning as a **separate decode-side parameter that never
enters the note alphabet**. The model predicts notes on a universal 12-TET grid; the exact frequency is
rendered deterministically from the instrument + a tuning parameter it can otherwise ignore.

## Part A — unify GoatTracker into the per-voice tracker-row representation

Stop serializing raw `.SNG` bytes. The reconstructed `pygoattracker.Song` already holds exactly the fields
the Hubbard rows use — decompose it into the same stream:

- **Orderlist → patterns** give the per-voice row sequence; GoatTracker's orderlist *is* the backward
  reference, so map it to the codec's inline `REPEAT(offset, length)` over prior rows (or keep an explicit
  orderlist op — same backward-only, no forward declaration).
- **Pattern row** `(note, instrument, command, data)` → `(dt, note_interval, instr_ref, lnth, porta)`:
  note byte → canonical interval (Part B); instrument number → pitch-invariant `instr_ref` (the GoatTracker
  instrument = ad/sr/wave/pulse/filter/vib params, defined inline once, exactly like Hubbard's instrument
  bytes); command/data → the duration/effect fields (tempo/funktempo → `dt`/step grid; tone-porta → `porta`;
  gate/legato → `lnth` flags).
- The four GoatTracker tables (wave/pulse/filter/speed) become instrument-definition payload (the
  pitch-invariant generator params), referenced by `instr_ref`, not per-note.

Result: GoatTracker emits the *same* `(dt, interval, instr_ref, lnth, porta)` + `REPEAT` tokens as Hubbard.
The `.SNG`-bytes path is retired (it was the residual-zero shortcut; this is the learnable form). Gate
unchanged: render the rows back through pygoattracker → byte-exact vs the dump.

## Part B — canonical pitch axis: the note token is driver-invariant

Define the note's IDENTITY as its **12-TET semitone index on a fixed reference (A440), integer-valued** —
NOT a table offset, NOT a frequency. Every driver maps onto it the same way the landed pitch encoder
already does for Hubbard:

```
onset frequency Fn  ──snap──▶  nearest 12-TET grid index n   (the NOTE; A440 reference)
                                exact: 12·log2(Fn / Fn_A440) rounded to the semitone
```
- **Hubbard:** Fn comes from the register write at onset (already done — "absolute-anchored 12-TET encoder,
  byte-exact, base-snap sub-cent").
- **GoatTracker:** the pattern note byte is already a semitone index, but in *GoatTracker's* tuning; resolve
  it through GoatTracker's freq table → Fn → snap to the SAME A440 grid. So GoatTracker C-4 and Hubbard C-4
  land on the identical grid index n.

The **NOTE token stays the relative interval** `n - n_prev` (per voice), exactly as Hubbard emits today —
now driver-invariant. The same musical pitch is the same token across drivers, by construction.

## Part C — tuning as a separate, layered parameter (the resolution)

The actual frequency differs from the canonical grid frequency by a tuning residual. Factor it into a
**static per-tune component** and a **dynamic per-onset component**, neither in the note alphabet:

```
Fn(note onset)  =  canonical_grid_Fn[n]            (fixed, A440 12-TET — shared by ALL tunes/drivers)
                 +  tuning_table_delta[n]           (static: the tune's ET-rounding + base-tuning offset)
                 +  micro_onset                     (dynamic: modulation phase + inter-voice detune)
```

1. **Static tuning = a per-tune deviation table `Δ(n)`** (cents, signed, small). For ET drivers this is the
   note table's rounding residual (≤~4c, per note *class*, deterministic) plus a per-tune base-tuning
   scalar. It is defined **once per tune** (referenced by every note), and factored toward a
   **corpus-shared canonical table**: the tune carries only `Δ` from the shared ET reference, which is
   near-zero for standard ET drivers → almost free (this is the deferred "corpus-shared table vocab /
   defcost ceiling" in the memory log). A tune at A=438 differs from one at A=440 by a single scalar in
   `Δ`; both emit identical note tokens.

2. **Dynamic tuning = the per-onset `micro`** (signed sub-semitone, bounded — the arc's existing field):
   the vibrato/slide phase sampled at the onset frame + chorus detune between paired voices. Where the
   instrument generator is recovered, `micro` is **derived** from the generator's phase at onset (not
   stored), as the landed codec already does — for GoatTracker it is **0** (pygoattracker the VM
   regenerates the vibrato). Where a generator is not yet recovered, the onset modulation SAMPLE is carried
   as a small signed field (it is a deterministic generator-onset value, not an irreducible residual — once
   the generator is recovered it too is regenerated, per HARD RULE #0: nothing is irreducible).

The MODEL sees only the canonical interval token. `Δ(n)` is per-tune conditioning (one small table, or a
shared-table reference + a scalar); `micro` is a low-entropy side channel the model may ignore for note
prediction. Decode is exact: `note + Δ + micro + instrument generator → Fn`, byte-exact (residual-zero
gate unchanged; the arc proved this reconstructs Fn to sub-cent, max 0.86c, then exactly via micro).

## Why this satisfies "the same notes as the same notes"

- **Token identity = canonical 12-TET semitone index** (A440), computed from the rendered frequency, so it
  is independent of which driver, which note table, or which base tuning produced it. Concert C-4 is one
  token everywhere.
- **All tuning variation is off the note axis:** systematic table/base differences → the per-tune `Δ`
  table (static, shared-factored); per-onset modulation/detune → `micro` (dynamic, mostly derived). A
  model can learn pitch structure on a clean universal grid and treat tuning as optional conditioning.
- **No information lost:** `Δ` + `micro` + the instrument generator reconstruct the exact per-frame Fn, so
  residual = 0 still holds. Tuning is *factored out of the alphabet*, not discarded.

## Validation / gates

- **Residual-zero preserved** on Monty, 5TT, Grid_Runner after the GoatTracker rows + canonical-pitch
  change (the existing byte-exact gates).
- **Driver-invariance probe (new gate):** the same musical pitch emits the same note token across a Hubbard
  and a GoatTracker tune — e.g. assert a known concert-pitch note lands on the same grid index `n` from
  both backends. Add to the cross-driver test once Part B lands.
- **Tuning is per-tune, not per-note:** assert `Δ(n)` is constant across a note's occurrences (a single
  table), and that two tunes differing only in base tuning differ only in the `Δ` scalar, not in tokens.

## Open risks / sequencing

- **Note-on parity across drivers:** the canonical grid needs the onset frequency; legato/tie notes (no
  gate cycle) and GoatTracker tone-porta need the same note-on detection the generic-recovery analysis
  flags (`generic_bacc_recovery.md` §3a) — resolve onsets consistently before snapping.
- **GoatTracker note byte vs register Fn agreement:** verify GoatTracker's note-byte→freq-table→snap lands
  on the identical grid index as Hubbard's register-Fn→snap for the same concert pitch (they should, both
  ET A440; any constant offset is the per-tune base scalar, not a token difference).
- **Sequencing:** Part A (GoatTracker rows) and Part B (canonical pitch) are independent and can land
  separately; Part C's static `Δ` table reuses the existing micro machinery and the corpus-shared-table
  work. Land Part B's driver-invariance probe first — it is the executable definition of "same note."
