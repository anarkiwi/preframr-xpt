# Prompt interface — from SID continuation to musical-phrase prompting (MIDI / keyboard)

**Status:** Design (2026-06-12). The input side of the generation program. Continuation from a SID
register prompt exists (`inference/predict.py`, `event_gate.py`); the **ultimate goal is generation
from diverse prompts — e.g. a short musical phrase from a MIDI file or keyboard — arranged and
continued as a SID tune.** Nothing here changes the model or the alphabet; the bet is that prompting
is a *compilation* problem plus a *distribution-shift* problem, both attackable with existing
machinery. Sequenced after the in-flight canonical learnability run settles (the model must first be
worth prompting).

## Goal ladder

- **G1 — SID continuation (exists):** prompt = a window of a real tune; the model continues it.
- **G2 — phrase prompting (THE goal):** prompt = a short melodic phrase (MIDI file, keyboard
  capture, or hand-entered notes); the model treats it as the lead it must adopt, then harmonizes,
  arranges, and continues in corpus style.
- **G3 — style steering (refinement):** G2 plus "in the style of X" control.

## G2 mechanism: the phrase compiler (phrase → native event-token prompt)

A phrase is already expressible in the event alphabet — that is the deep advantage of the v3
encoding (notes are `NI_*` intervals over a per-voice `NOTE_TABLE`, durations are explicit on
`FLD_NOTE_ON`). So the interface is a small deterministic compiler, **not** a model change:

```
MIDI/keyboard phrase (note, onset, duration[, velocity])
  → quantize onsets/durations to the PAL frame grid (50 Hz)
  → synthesize a minimal one-voice register dump: freq from the note (standard 2^(n/12) grid),
    gate on/off per note, one default instrument program (waveform/AD/SR), voices 1–2 silent
  → events.oracle.ordered_writes → stream.encode(verify=True)
  → KEYFRAME-led prompt block (events.pipeline machinery)
```

Properties: the prompt lives in **exactly the model's native token space** (no new vocab, no
adapters); it is **self-verifying** (the encode contract applies); and it is **renderable** — the
user can hear precisely what the model was given. SID has no velocity: v1 drops it (map to sustain
level later if it earns its keep). Polyphonic input: take the top line v1 (lead extraction); chords
are a G2.5 extension (spread across voices).

## The crux: distribution shift (the prompt is off-manifold)

A bare single-voice, default-patch prompt looks like no training window (real windows are 3-voice
textures with engine-specific programs). Mitigations, in order of expected leverage:

1. **Reduction augmentation (training-side, the load-bearing bet).** Derive (melody-only prefix →
   full-texture continuation) training pairs from the corpus itself: take a real window, strip
   voices 1–2 (and the lead's ornament, optionally) from the first K frames, keep the target intact.
   This *teaches arrangement from a lead sheet* with zero new data. Natural home: preframr-aug (it
   already owns inaudible-perturbation/voice-permutation augmentation); the event encoder makes the
   reduction trivial (drop the other voices' events; re-encode self-verifies). Run as an A/B arm:
   reduction-augmented vs not, judged on phrase-prompted cohort quality (below) with no regression
   on plain continuation.
2. **Prompt scaffolding (inference-side, cheap).** Embed the phrase in a more corpus-like scaffold:
   phrase on voice 0 + a minimal drum/bass vamp vs bare. A/B by the off-manifold probe + quality
   gate; pick per-use-case.
3. **Patch realism.** Sample the default instrument program from corpus statistics (common
   (waveform, AD, SR) programs) instead of one fixed patch.

**Off-manifold probe (cheap, run first):** mean per-token CE of the trained model on compiled
prompts vs natural prompts. If compiled prompts are wildly out-of-distribution the model will ignore
them (the prompt-conditioning audit will show it) — measure before building mitigation 1.

## Keyboard / live input

Same compiler fed by captured MIDI events; quantization to the frame grid is the only extra step.
Real-time generation is out of envelope (Orin single-stream is ~9× short —
`../performance/orin_inference_optimization_design.md`), so v1 UX is **capture phrase → generate
offline (~minutes) → play**. A live jam mode is explicitly out of scope until the predict envelope
changes.

## G3: style steering

Try **exemplar prompting first**: prepend a short window of a target-style tune before the phrase
block. Zero vocab change, and it leans on exactly the circuit the encoding is designed for
(induction-head copy). Only if exemplars demonstrably fail, consider reserved conditioning atoms
(composer/engine ids) — that is a tokens-side alphabet change (major version bump; checkpoint
invalidation) and needs its own design. Engine choice (6581 vs 8580 render) is free — it is a render
parameter, not a model input.

## Gates

1. **Compiler round-trip:** compile → encode → decode → render reproduces the phrase exactly
   (the encode self-verify makes this structural); golden-file tests on a few phrases.
2. **Off-manifold probe** above, before and after each mitigation.
3. **Phrase adherence (the G2-specific metric):** does the continuation *use* the phrase? Measure
   interval-n-gram overlap between the prompt melody and the continuation's lead lane, plus key
   consistency (NI histogram alignment). Report alongside the
   [generation quality gate](generation_quality_gate.md) scorecard on a phrase-prompted cohort
   (which also guards against the failure mode of parroting the phrase verbatim — the memorization
   audit reads both ways).
4. No regression on G1 continuation quality when reduction augmentation is in the training mix.
