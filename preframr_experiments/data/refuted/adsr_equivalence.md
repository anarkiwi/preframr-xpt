# `adsr_equivalence` (static ADSR / CTRL canonicalisation) — REFUTED

**Status:** static-equivalence canonicalisation refuted via the audit
in `integration_tests/design/adsr_equivalence_report.md`. No A/B run;
the report demonstrated the canonicalisation is unsafe before
implementation.

## Hypothesis

Two ADSR / CTRL byte triples that differ only in fields the engine
doesn't sonically use (e.g. release nibble during a sustained gate)
could canonicalise to a single palette entry, shrinking alphabet
without changing audio output.

## Refutation

The audit enumerated the ~6500 distinct (ATK, DECAY, SUSTAIN,
RELEASE, CTRL) triples observed in the smoke + mini corpora and
checked which fields are sonically observable per ADSR phase:

- **Release nibble during sustain** — observable on legato
  transitions (release sets next attack's slope), NOT
  canonicalisable.
- **Attack nibble during release** — observable on retrigger,
  NOT canonicalisable.
- **CTRL gate-only field changes** (waveform flip mid-gate) —
  observable as timbral shift, NOT canonicalisable.

The remaining "truly static" canonicalisation pool covered ~3% of
the corpus's ADSR/CTRL byte distribution. Alphabet shrink: <1%.
Cost: per-merge equivalence proof gets re-run on every parse, ~30%
parse wallclock overhead.

Net: cost dominates negligible benefit. The standing canonical
encoding (per-byte SET preserved verbatim) is correct.

## Evidence

`integration_tests/design/adsr_equivalence_report.md` — full audit
with per-field observability matrix and per-corpus byte
distribution.

## Do not revisit without

- A canonicalisation scheme that targets a different equivalence
  class (e.g. cross-voice CTRL transposition modulo voice ID),
  with audit-time evidence of >5% alphabet shrink at <5% parse
  wallclock overhead.

Note: the existing `coarsen_pass` (kept as a tracker-export tool,
not an encoder pass) implements a SIMILAR-but-different
coarsening; see `data/refuted/macro_coarsening.md` for that
separate refutation.
