# Register-log → SWM recompiler — lossless (re-render-equivalent) SID-Wizard module from a parsed log

**Status:** Design (2026-06-06). A tool that turns a **preframr parsed register log** (the per-frame SID
register-write stream) back into a **SID-Wizard SWM module** that, played by SID-Wizard's own player, produces
the **same register output**. **Does not exist today:** `pysidwizard` has `build_swm(SWMFile)`/`write_swm`
(typed model → SWM bytes) + a reader + a bit-exact player, but nothing compiles a *register log* → SWM (the
hard front half). This is the constructive **reverse** of the generator pipeline, and the implementation of the
"reverse round-trip" the generator work order's §7B asserts. Cross-ref
[`generator_mdl_representation.md`](generator_mdl_representation.md) (the IR this reuses),
[`sid_render_fidelity_contract.md`](../references/sid_render_fidelity_contract.md) (what "same output" means),
[`sid_driver_ornament_reference.md`](../references/sid_driver_ornament_reference.md) (SID-Wizard's player model).

## 1. What "lossless" means here (the load-bearing definition)
**NOT byte-identical SWM** — the inversion is many-to-one (many different SWMs render to the same registers),
so the original SWM bytes are unrecoverable and byte-identity is neither achievable nor meaningful. **Lossless =
re-render register-equivalence:** the emitted SWM, run through `pysidwizard`'s player, yields a register stream
**identical to the input log** under the project fidelity oracle `sid_frame_diff.diff_dump_vs_pipeline`
(CTRL/AD/SR/RES_FILT(23)/MODE_VOL(24) byte-exact in input order + nominal `_MIN_DIFF` timing; FREQ/PW/filter
within the quantization tolerance). That register-log match **is** the arbiter — same registers in the same
order with the same delay produce the same output by construction, so no WAV render is needed. This is exactly the
equivalence criterion the user set for the round-trip ("equivalent if they produce the same output").

## 2. The expressibility boundary (be honest, don't silently approximate)
SID-Wizard's player has a FIXED expressivity; not every register log is SWM-producible. The tool guarantees
losslessness **on the SID-Wizard-expressible subset**, and **REPORTS** (per voice / per frame) anything outside
it — it never ships an SWM that fails the re-render check silently.
- **Fully lossless, trivially:** logs that *came from* a SID-Wizard tune (the round-trip case — the player can
  reproduce its own output) and logs within the player's model.
- **Large expressible subset** because SID-Wizard's wavetable is a **per-frame program** (waveform + pitch
  column) and the `pw_table`/`filter_table` set/sweep per frame — so arbitrary per-frame waveform, PW, filter,
  and (via absolute-note + detune) pitch are reachable.
- **The lossy boundary** (the constructs SID-Wizard's player can't reproduce exactly, to be detected + reported,
  not faked): mid-note **ADSR re-trigger** (SID-Wizard loads AD/SR at note onset), exact **hard-restart timing**
  / gate-off-then-on sequences, **intra-frame write ORDER** that the player's fixed DOTRACK order can't match
  (audible via the ADSR bug — see the fidelity contract), **sub-frame** timing (digis), and **multispeed**
  beyond the SWM `frame_speed` range. Foreign-driver logs (Hubbard/JCH/Galway) are lossless where these don't
  bite, with a reported residual otherwise.

## 3. Architecture (pipeline)
```
parsed log ─► register_state (per-frame 25 regs)
           ─► generator-MDL IR  (the musical structure; reuse the generator pass)
           ─► SWMFile model      (compile IR → SID-Wizard constructs)   ── two paths, §4 ──
           ─► build_swm()        (pysidwizard backend → SWM bytes)
           ─► VERIFY: player(SWM) == input log   (the lossless GATE)
```
- **Input** — the canonical per-frame target is `register_state(df)` (order/timing-blind settled snapshot) PLUS
  the ordered raw writes for the fidelity oracle (intra-frame order is audible). Both come from the parse.
- **IR — reuse the generator-MDL decomposition** (`generator_fit.channels`/`decompose` + the instrument-program
  codebook + note segmentation): each channel is HOLD/ACCUM/SWEEP/TABLE; notes are segmented (level-change ∪
  gate); ctrl/AD/SR onset = an instrument program; the DEF→REF bank = reusable instruments/tables. This IR is
  *already* the tracker's vocabulary expressed register-agnostically — the natural thing to compile to SWM.
- **Backend** — `pysidwizard.build_swm(SWMFile)` / `write_swm`. The tool's job is constructing the `SWMFile`
  (instruments, patterns, sequences, tables) so re-render matches.

## 4. Two compile paths (P1 universal+ugly, P2 structured+editable)
**Path A — brute-force wavetable (universal, lossless on the expressible subset; ship FIRST).** Encode each
voice's per-frame behavior directly into SID-Wizard's per-frame tables: one pattern note per voice that fires an
instrument whose **`wf_table` walks the exact per-frame (waveform, absolute-note + detune) sequence** and whose
**`pw_table`/`filter_table` walk the exact per-frame PW/filter**; AD/SR from the onset; gate from the note
on/off. This reproduces almost all per-frame register content (SID-Wizard's wavetable is exactly a per-frame
register program), with ADSR/gate-timing as the only residual. Proves losslessness with minimal cleverness; the
output is an ugly-but-correct SWM.

**Path B — structured mapping (clean, editable, compact; the real goal).** Map the IR to *musical* SID-Wizard
constructs so the SWM is human-editable in SID-Wizard and small:
| generator-MDL IR | SID-Wizard construct |
|---|---|
| note segmentation + interval/skeleton | pattern rows (note-on) + sequence `Transpose`/`PlayPattern` |
| instrument program (ctrl-walk, AD, SR) | `Instrument` (ADSR + `wf_table` waveform col) |
| `TABLE` arp (note-relative offsets) | `wf_table` arp bytes (note-relative semitone cycle) |
| `ACCUM` on freq (slide) | portamento command / `wf_table` rel-pitch ramp |
| `SWEEP` (triangle) on freq | vibrato (amp/freq); on PW → `pw_table` auto-reverse |
| `TABLE`/`SWEEP` on PW, filter | `pw_table` / `filter_table` set+sweep rows (filter global + routing) |
| DEF→REF bank reuse | one shared `Instrument`/table per bank id (the reuse maps 1:1) |
| absolute / off-grid freq (drum/noise) | `wf_table` absolute-note + 16-bit detune |
Per voice/segment, choose Path B where the construct expresses it exactly, else **fall back to Path A** for
that span. The choice is verified, not guessed (§5).

## 5. The lossless GATE (verification is the contract, not a hope)
After emitting `M`, **render it through `pysidwizard`'s bit-exact player → register stream `R`**, and assert
`R == input log` under the fidelity oracle. If it matches → ship `M`. If not → the divergence (voice, frame,
register) IS the SID-Wizard-inexpressible residue; **report it** (a per-tune fidelity manifest) and either keep
that span on Path A or flag it as out-of-scope. **Never emit an SWM that fails its own re-render check.** This
is the same `arbitrate(validate=True)` discipline used everywhere — re-render is the validator.

## 6. Where it lives + dependencies
Natural home: **`pysidwizard`** (it owns the SWM model + `build_swm` + the verifying player). It imports the
generator-MDL IR from `preframr-tokens` (or re-derives the channel decomposition locally to stay
dependency-light). A thin CLI: `swm-recompile <dump.parquet|log> -o tune.swm` → emits the SWM + the fidelity
manifest. (Sibling, out of scope here but identical architecture: a `pydefmon` `DefmonSong.to_file` backend for
register-log → defMON `.prg`.)

## 7. Why it's worth building
- **Closes the loop to a real editor:** preframr's generated/learned tunes become **human-editable SID-Wizard
  modules**, not just register dumps — the difference between "the model emits registers" and "the model writes
  a tune a chiptune musician can open and edit."
- **It IS the §7B reverse round-trip test** (module → log → macros → module, same output) — building it makes
  that gate real, and proves the generator-MDL IR maps onto genuine tracker constructs (constructive
  provenance-invariance).
- **A SID-Wizard-expressibility oracle:** the per-tune fidelity manifest quantifies how much of arbitrary HVSC
  (foreign-driver) content is SID-Wizard-reachable — a research artifact in itself.

## 8. Phasing
- **P1:** Path A (brute-force wavetable) + the re-render gate → lossless SWM on the expressible subset, proven
  on the SID-Wizard 1.94 example tunes (round-trip to same output) and on a hard-engine sample (with manifest).
- **P2:** Path B structured mapping (musical, editable, compact SWMs); per-span A/B fallback.
- **P3:** the fidelity-manifest report at corpus scale (SID-Wizard expressibility of HVSC) + the `pydefmon`
  sibling backend.

## 9. Honest non-claims
- **Not universal losslessness.** Guaranteed only on the SID-Wizard-expressible subset, *verified per tune by
  re-render*; the inexpressible residue (mid-note ADSR, exact HR timing, intra-frame order, sub-frame digis,
  out-of-range multispeed) is reported, not hidden.
- **Not byte-identical SWM.** The right notion is re-render equivalence; the emitted module may look nothing
  like any human-authored original yet be behaviorally identical.
