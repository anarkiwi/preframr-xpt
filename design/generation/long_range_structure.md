# Long-range structure — generating whole tunes beyond one window

**Status:** Design — re-scoped to the BACC codec. Under BACC (VOCAB=34, absolute 12-TET grid notes +
backward Transpose, residual-zero), **whole tunes fit one 4096-token window** (Monty 1,313;
5_Title_Tunes 1,394; Grid_Runner 2,817; A_Mind_Is_Born 496), and the training default context is
4096 (whole-song-in-context). The old problem statement — tunes ~30k tokens, 82% exceeding a window,
seq_len 8192 + KEYFRAME windows, "no mechanism to produce a whole tune" — is **OVERTURNED**: the norm
is now a whole tune in one window. The only live long-range problem is the **>4096 STRETCH TAIL**:
the corpus stretch goal is ≥90% of tunes under 4096, so chaining is needed only for the ≤10% that
exceed one window. This doc covers that stretch-tail mechanism and the long-horizon coherence
measurement. It deliberately does NOT propose whole-tune-scale context blowup or a hierarchical model
(out of the single-4090 / Orin envelope — refuse in design, per AGENTS.md).

## Stretch-tail mechanism: decode-and-recompile chaining

For the ≤10% of tunes that exceed 4096, exploit the property the BACC codec already guarantees —
**a BACC block is self-contained** (settled state recoverable from the sid/sid-trace; the
`recover_from_sid` path reconstructs from the SID itself, no `.dump.parquet`):

```
prompt block ─► generate until the window budget (or an emitted terminal silence/loop)
            ─► decode to register writes via the BACC decode/recover path (raises on invalid)
            ─► append writes to the growing tune
            ─► re-encode the TAIL of the accumulated writes into a fresh self-contained
               BACC block (same code that builds training windows)
            ─► that block is the next prompt; repeat until the length budget
            ─► render the full accumulated write stream once
```

The key property: chaining happens **in the register domain, not the token domain** — each re-prompt
is re-canonicalized from decoded writes, so the model's per-window state (settled registers, BACC
accumulator state) is *exact by construction* rather than drifting through a token-tail copy. Seams
land on BACC block boundaries. Cost: one extra encode per window (~seconds); no model change, no new
tokens. This is a stretch-tail tool only — whole tunes under 4096 take the plain single-window path.

Window-overlap variant (only if seams are audible): carry the last F frames of decoded writes
into the next block's prefix so the model conditions on a real musical tail, not just settled state.

## What chaining cannot give (and how we'd know)

Chaining yields *locally coherent* indefinite continuation; it does not plan **song form** (intro /
theme / variation / outro, long-period repetition). Whether that matters at this corpus's scale is
an empirical question — SID tunes are heavily loop-structured, and the model may reproduce
section-scale repetition from in-window statistics. So: **measure before designing form control.**

Long-horizon reads (added to the [generation quality gate](generation_quality_gate.md) cohort, on
3–5-window chained generations):
- **Self-similarity structure:** autocorrelation / self-similarity matrix of the note-onset stream
  at multi-bar lags, compared distributionally to corpus tunes (corpus norm: strong periodicity at
  pattern lengths). Flat = wandering; delta-function = stuck loop.
- **Drift across seams:** instrument-program mix, voice utilization, tempo (DT histogram), and key
  (absolute-grid pitch-class histogram) per window — corpus tunes hold these roughly stationary with
  discrete section changes; monotone drift indicts the chaining.
- **Seam audibility:** render with and without a seam-spanning crossfade-free cut at each BACC block
  boundary; any rendered discontinuity (envelope cut, click) is a re-encode bug, not a model
  failure — the fidelity contract applies at seams too.
- **Termination:** corpus length distribution sets the budget; emitted sustained silence or a
  detected terminal loop (`detect_tail_cycle`) ends generation early — report which.

## Escalation ladder (evidence-gated, in order)

1. **Chaining (build now)** — the mechanism above; measure.
2. **Section-conditioned chaining** — if (and only if) chained stretch-tail tunes measurably lack
   form: mine section boundaries from the corpus (the landed structural index +
   self-similarity segmentation), and prompt each window with an *exemplar* of the intended section
   (same induction-head-friendly, zero-vocab trick as style steering in
   [`prompt_interface_design.md`](prompt_interface_design.md)) before considering section tokens
   (alphabet change, major bump).
3. **Hierarchical / sketch model** — rejected for the current envelope; revisit only on hardware
   change.

## Gates

1. Chained 3-window generation decodes clean (strict grammar, zero invalid) and renders seam-free
   on a 10-prompt cohort.
2. Long-horizon metrics above reported in the quality-gate scorecard; thresholds calibrated from
   corpus statistics (like every other floor in this repo: measure 2–3 baselines first).
3. Wall-clock: a full ~3-minute tune generates offline within the audition budget
   (the ~6.5 min/song Orin reference predates BACC — re-measure on a BACC checkpoint; faster on the
   4090).
