# Register-log → SWM recompiler — re-render-equivalent SID-Wizard module from a register log

**SUPERSEDED (2026-06-20):** the landed step/tracker codec IS the register-log → editable-program
decompiler. A tracker-module export endpoint can be rebuilt on it if the editable-output goal returns.

**Status:** Design (2026-06-06; IR source retargeted 2026-06-12 — the generator-MDL tokens pass is
retired, so the decomposition is derived standalone, below). Does not exist yet: `pysidwizard` has
`build_swm`/`write_swm` (typed model → SWM bytes) + a reader + a bit-exact player, but nothing
compiles a *register log* → SWM. Output-side relevance: this is how a **generated** tune becomes a
human-editable tracker module rather than a register dump — see
[`../generation/prompt_interface_design.md`](../generation/prompt_interface_design.md) for the
input-side sibling (musical phrase → prompt).

## 1. What "lossless" means here (load-bearing)

**NOT byte-identical SWM** — the inversion is many-to-one. **Lossless = re-render
register-equivalence:** the emitted SWM, played by `pysidwizard`'s player, yields the same register
stream as the input log (CTRL/AD/SR/globals byte-exact in order, FREQ/PW within quantization
tolerance). Same registers, same order, same delay ⇒ same output by construction; no WAV needed.

## 2. The expressibility boundary (report, never approximate silently)

SID-Wizard's player has fixed expressivity. Guarantee losslessness **on the expressible subset** and
REPORT (per voice/frame) anything outside it. Large expressible subset (the wavetable is a per-frame
waveform+pitch program; pw/filter tables set/sweep per frame). The lossy boundary to detect + report:
mid-note ADSR re-trigger, exact hard-restart timing, intra-frame write order the player's fixed
DOTRACK order can't match, sub-frame digis, out-of-range multispeed.

## 3. Architecture

```
register log ─► per-frame settled grid (events.oracle.settled_grid)
             ─► generator-style IR: per-channel HOLD/ACCUM/SWEEP/TABLE runs + note segmentation
                + onset instrument programs  (standalone fitter — the retired generator pass's
                decomposition, reusable from the prototype; self-verifying fit rule applies)
             ─► SWMFile model ─► build_swm() ─► VERIFY: player(SWM) == input log  (the GATE)
```

Two compile paths:
- **Path A — brute-force wavetable (ship first):** one pattern note per voice firing an instrument
  whose `wf_table` walks the exact per-frame (waveform, absolute-note+detune) sequence, `pw_table`/
  `filter_table` walk the exact per-frame PW/filter. Ugly-but-correct; proves losslessness.
- **Path B — structured mapping (the real goal):** notes → pattern rows + transpose; instrument
  programs → `Instrument`; arps → note-relative `wf_table` bytes; slides → portamento; sweeps →
  vibrato / table auto-reverse. Per span choose B where exact, else fall back to A — verified, not
  guessed.

## 4. The gate + phasing

Render every emitted module through the bit-exact player and assert equality; the divergence IS the
inexpressible residue — report it in a per-tune fidelity manifest, never ship a failing SWM.
Phasing: P1 Path A + gate (round-trip the 91 SID-Wizard 1.94 examples; hard-engine sample with
manifest) → P2 Path B with per-span fallback → P3 corpus-scale expressibility manifest (+ the
`pydefmon` sibling backend). Natural home: `pysidwizard` (owns the model + verifying player), with a
thin `swm-recompile <dump|log> -o tune.swm` CLI.
