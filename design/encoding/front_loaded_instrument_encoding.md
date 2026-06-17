# Front-loaded instrument encoding — tracker-style DEF→REF for the onset program

**Status:** Design + build work-order (2026-06-17). The first experiment that targets the *confirmed
root cause* of the free-running failure on a *new axis*. Implementation home: **preframr-tokens**
(`events/`); the README is the authoritative grammar reference — read it first.

## Why (the diagnosis this serves)

The representation probes (`../generation/representation_abstraction_probe.md`) showed the model is a
**strong local abstractor but a weak long-range structural one** — a bag of local content bounded by
~1024 effective context. It cannot hold form, which is exactly what generation needs. The current event
codec has **no DEF/REF** (README): the recurring onset *instrument program* (envelope/waveform/PW) is
re-emitted inline at *every* note onset — **~38% of the atom stream** (census), ~750–1240 onsets/tune
but a handful of distinct programs (exact-program recurrence: top-10 cover **81–95%** of onsets,
0.5–8% true singletons). So the model is forced to *re-derive* a ~15-atom program it should be *referencing*.

Front-loading instruments — define them once in a header, reference them by id in the body, exactly how
trackers author this music — does three things the diagnosis says should help: (1) it **matches the
data's true generative process** (strong learnability prior); (2) it is a clean **DEF→REF the model's
strong local abstraction handles natively** (vs treating an instrument as poorly-represented long-range
repetition); (3) it **compresses the body ~1.5×**, so the ~1024-atom window spans ~1.5× more musical
*form*, attacking the structural bottleneck directly.

## The scheme: reference-or-inline (byte-exact, v1)

Keep it the simplest thing that is **byte-exact** (the project's non-negotiable invariant —
`encode(verify=True)` must hold; `decode(encode(ow)) == canonical_writes(ow)`):

- **Bank (DEF), in the preamble.** Extend the existing per-voice header preamble with an **instrument
  bank**: the distinct exact onset-programs of the tune, ranked by frequency, the top-K defined as
  numbered entries `0..K-1`. An instrument program = the onset *timbre* writes (CTRL/waveform walk,
  AD, SR, PW behaviour, hard-restart prep, recorded gate-edge side) — **NOT** the note's pitch or
  duration. New structural atom(s): an `INSTR_DEF` opener; id is positional.
- **Body (REF or inline).** At each note onset (currently the `FLD_NOTE_ON` fold), emit
  **`INSTR_REF <id>`** when the onset's program is byte-identical to bank entry `id`; otherwise keep the
  **existing inline `FLD_NOTE_ON` fold** (the 5–19% tail). The note's pitch (`NI_*`) and duration stay in
  the body in both cases.
- **Decode.** `INSTR_REF <id>` expands to bank[id]'s exact program writes; inline is unchanged. Trivially
  byte-exact (reference is an exact substitution; tail is the current encoding).
- **Constrained decode.** The grammar mask (`events/constrained.py:EventStreamState`) enforces
  define-then-reference: the bank is in the preamble (defined before the body), and an `INSTR_REF` id
  must be `< K`. The model literally cannot reference an undefined instrument — clean and exactly the
  mechanism the operator flagged.

Reference-or-inline is byte-exact by construction and reclaims the recurring 81–95%. A residual/override
(encode the *diff* from a near-match bank entry, shrinking the inline tail) is a **v2 refinement** —
do not build it first.

## The hard part (study the encoder before designing the grammar)

`FLD_NOTE_ON` (atom 70) "owns the envelope lifecycle: onset AD/SR fold, recorded gate-edge side,
hard-restart prep, **mixed-radix duration**." So the onset fold **mixes the instrument (AD/SR/edge/HR/
waveform/PW) with the note's duration** — and duration is note-specific, not instrument. The bank entry
must capture the *instrument* part and leave duration (and pitch) in the body. **First task: read the
`FLD_NOTE_ON` encode/decode path precisely and define the exact instrument/note split**, then the
"exact onset-program" key for bank dedup is well-defined (two onsets share an instrument iff their
instrument-part writes are byte-identical, regardless of duration/pitch).

## Alphabet / version impact

New atoms (`INSTR_DEF`, `INSTR_REF`, id values) change the **127-atom alphabet** → bump
`EVENT_FORMAT_VERSION` (v2→v3) and `ATOM_CACHE_VERSION`; the `.atoms.zst` caches recompute. This is a
3-repo change at release time (tokens→PyPI, then preframr floors it, then xpt BASE bump) — but the
*experiment* runs from source mounts, so no release needed to test. Keep K and the id encoding within a
sane atom budget (positional ids via a small varint; do not blow the alphabet up).

## Build plan (preframr-tokens, branch off main; byte-exact gate is the guardrail at every step)

1. **Measure exact-program recurrence precisely** (the precise version of `instr_census`/`recur_triage`):
   on real tunes, cluster onsets by the *exact instrument-part* key, report distinct-K and onset coverage
   of top-K and the singleton tail. Sets K (or the rule "define every program with ≥2 uses") and confirms
   the ~38%/81–95% payoff at the byte-exact level (my earlier triage used a *static* signature that
   over-splits — the exact key may differ; verify).
2. **Design the grammar** (write it down): `INSTR_DEF` preamble block, `INSTR_REF <id>` body form, the
   instrument/note split, the decode expansion. Keep it minimal.
3. **Encoder**: extract the per-tune bank, emit DEF blocks in the preamble, emit `INSTR_REF` for exact
   matches, inline the tail. **`encode(verify=True)` must stay green** at every commit.
4. **Decoder**: expand `INSTR_REF`; byte-exact.
5. **Constrained decode**: extend `EventStreamState` so the mask tracks the defined bank size and forbids
   out-of-range / pre-definition references.
6. **Version bump** + cache version + tests.

## Gates

- **HARD: byte-exact round-trip.** `encode(verify=True)` green on the 5 reference drivers + a corpus
  sample (the existing roundtrip tests + corpus-sample tests must pass; add ones for the bank path).
  Non-negotiable — a single non-exact tune fails the gate.
- Alphabet utilisation / no dead atoms blow-up; the repo lint gate (no narrative `#`, docstrings ≤5
  lines, xdist) green.
- **Cheap pre-train read:** `learnability_triage` on a sample encoded v3 (front-loaded) vs v2 (inline) —
  does induction-copy rise / h_k drop? (Expected: yes — the DEF→REF is induction-friendly and the stream
  is shorter.)
- **Decisive A/B (operator-run on GPU, after the codec is green):** encode the corpus both ways, train
  atoms-only same config (tkvocab 0, seq_len 8192, ep100), read on the metrics that *diagnosed* the
  problem — `free_running_gap` (does free-run content acc rise above 0.062?), the structural probe
  (does block-reverse / `[A][A]`-repeat-lift improve?), and eval-B content. That closes the loop.

## Scope / non-goals

This is the **instrument** half of tracker structure. The **pattern** half — front-loading recurring
*melodic phrases* as references (the `[A][A]` repeat the probe is really about) — is the complement and
the harder build; defer it (instruments first: biggest immediate compression, cleanest recurrence,
validates the whole DEF→REF approach). No residual/override in v1 (reference-or-inline only). No model or
training-objective change. Aligned with the existing instrument-bank design
(`../generation/transplant_augmentation_design.md` P0) — reuse its program-extraction thinking.
