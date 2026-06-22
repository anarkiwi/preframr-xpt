"""Generic per-axis SID decompiler.

Spec: /scratch/tmp/re/generic_sid_decompiler.md (design / research proposal).

- Stage 0 (``stage0_amind``): re-expresses the white-box A Mind Is Born recovery as
  a fixed-size ``{G_A}`` recurrence model + an interpreter that re-executes the
  *recurrences* (NOT the py65 white-box player). Its ``AmindRecurrence._play`` still
  re-executes the original seed image on ``cpu6502`` -- an acknowledged scaffolding
  crutch.
- Stage 2 (``sdst`` + ``generalize``): the HOST GENERALIZER. Consumes the bounded
  tracer artifact (SIDDF + STATESEQ + SNAP) and derives ``{G_A}`` AUTOMATICALLY,
  re-executing the synthesized ``(S,U,O,K)`` recurrences themselves -- the seed-image
  crutch is RETIRED. ``sdst`` parses the artifact; ``generalize`` fits the recurrences
  (Daikon-style templates + Berlekamp-Massey), recovers the shared master counter and
  the output maps, applies the admission gate, and honestly surfaces any axis that
  does not close as a fixed-size recurrence (design 5).
"""
