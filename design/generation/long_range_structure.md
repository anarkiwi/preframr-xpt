# Long-range structure — generating whole tunes beyond one window

**Status:** Design (2026-06-12). The model trains on KEYFRAME-led self-contained windows of
seq_len 8192; tunes average ~30k tokens and **82% exceed one window** (mean ~4.2 windows/tune). So
today the system can continue a prompt for part of a tune but has **no mechanism to produce a whole
one**, and no measurement of coherence beyond a window. This doc picks the mechanism and the
measurement; it deliberately does NOT propose a bigger context or a hierarchical model (out of the
single-4090 / Orin envelope — refuse in design, per AGENTS.md).

## v1 mechanism: decode-and-recompile chaining

Exploit the property the encoding already guarantees — **every KEYFRAME block is self-contained**
(per-voice TUNING/NOTE_TABLE/TICK headers + settled state; the `memorize` gate decodes complete
blocks for exactly this reason):

```
prompt block ─► generate until the window budget (or an emitted terminal silence/loop)
            ─► strict-grammar decode to register writes (events.generate; raises on invalid)
            ─► append writes to the growing tune
            ─► re-encode the TAIL of the accumulated writes into a fresh self-contained
               KEYFRAME-led block (events.pipeline — same code that builds training windows)
            ─► that block is the next prompt; repeat until the length budget
            ─► render the full accumulated write stream once
```

The key property: chaining happens **in the register domain, not the token domain** — each re-prompt
is re-canonicalized from decoded writes, so the model's per-window state (tuning, note tables,
settled registers) is *exact by construction* rather than drifting through a token-tail copy. Seams
land on KEYFRAME boundaries, which is precisely the discontinuity the model saw in training. Cost:
one extra encode per window (~seconds); no model change, no new tokens.

Window-overlap variant (v1.1, only if seams are audible): carry the last F frames of decoded writes
into the next block's prefix so the model conditions on a real musical tail, not just headers +
settled state.

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
  (NI pitch-class histogram) per window — corpus tunes hold these roughly stationary with discrete
  section changes; monotone drift indicts the chaining.
- **Seam audibility:** render with and without a seam-spanning crossfade-free cut at each KEYFRAME;
  any rendered discontinuity (envelope cut, click) is a re-encode bug, not a model failure — the
  fidelity contract applies at seams too.
- **Termination:** corpus length distribution sets the budget; emitted sustained silence or a
  detected terminal loop (`detect_tail_cycle`) ends generation early — report which.

## Escalation ladder (evidence-gated, in order)

1. **Chaining (build now)** — the mechanism above; measure.
2. **Section-conditioned chaining** — if (and only if) chained tunes measurably lack form: mine
   section boundaries from the corpus (the landed structural index +
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
   (~6.5 min/song on Orin per the measured envelope; faster on the 4090).
