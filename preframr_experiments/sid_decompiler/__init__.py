"""Generic per-axis SID decompiler — Stage 0 (generative proof).

Spec: /scratch/tmp/re/generic_sid_decompiler.md (design / research proposal).
Stage 0 re-expresses the white-box A Mind Is Born recovery as a fixed-size
``{G_A}`` recurrence model + an interpreter that re-executes the *recurrences*
(NOT the py65 white-box player) to produce a byte-exact ``[N, 25]`` SID register
stream. See ``stage0_amind`` for the model, interpreter, and the residual gate.
"""
