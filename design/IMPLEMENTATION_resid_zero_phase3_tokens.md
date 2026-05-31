# Implementation work order — RESID→0 Phase 3: wavetable codebook + frontier tail + constrained decode

**Status:** Build spec. **Agent scope: `preframr-tokens` ONLY.** This document is self-contained — every
mechanism spec, data point, and cross-repo API contract you need is embedded below. **Do NOT read other
repositories** (`preframr-xpt`, `preframr`, external driver sources): they are the author's reference and
are summarised here. You implement and unit-test entirely inside `preframr-tokens`, behind default-OFF
flags; the **author** runs corpus-scale validation (sidid per-engine residue, 12-SID audio audition,
cross-repo release) after you hand back. Branch from a clean `main` after Release 2 (`v0.37.0`).

> **Operating rule (author directive, non-negotiable):** "If `resid > 0` it's because the engine that
> produced it isn't modelled yet." Every tune's engine is known. So `resid > 0` ⇒ a documented mechanism
> is missing, not an "irreducible floor." **Never write "irreducible floor" / "genuinely aperiodic" in
> code, comments, or memories** — the residue below is proven structured and codebook-able.

> **This supersedes** the "lossless toolbox exhausted / ~90% aperiodic / literal-0 needs P8 lossy"
> conclusion in `RESID_ZERO_IMPLEMENTATION.md`'s resume guide. That conclusion mistook the residue
> profiler's `UNRESOLVED` bucket (defined as "not a constant-Δ ramp and not a period-≤8 cycle") for
> "irreducible." It is wavetable-engine ornament, and it recurs (data in §0).

## 0. The data this plan is built on (embedded — you do not need the source artifacts)

Measured by the author on the **post-stack** residue (STAMP + PATCH + REL + SWEEP + held-ARP all ON — the
shipped `v0.37.0` stack), so this is what genuinely survives today.

**(a) The residue is concentrated in a few documented wavetable engines** (500-tune sample):

| engine | RESID notes | UNRESOLVED | ornament mechanism |
|---|---|---|---|
| GoatTracker_V2.x | 2279 | 93% | wavetable rel/abs-note sequences |
| DMC (Demo Music Creator) | 2212 | 73% | slow freq-sweep + arp + arp-accent |
| Music_Assembler | 1051 | 73% | arp + up-sweep + drums |
| Hermit/SidWizard_V1.x | 920 | 63% | `wf_table` arp sequences |
| MoN/FutureComposer | 376 | 51% | target+dur glide, sine-vib, noise-tik |
| JCH_NewPlayer | 250 | 77% | wave-table rel/abs notes |
| (tail: Soundmonitor, GoatTracker_V1, Vibrants/Laxity, DefleMask, System6581, GMC, …) | <220 each | varies | see §3 |

**(b) The residue RECURS across notes — it is a codebook, not one-offs** (the linchpin; recurrence of the
note-relative offset sequence across notes within a tune):

| engine | recur ≥2 (codebook-able) | median codebook / tune |
|---|---|---|
| Hermit/SidWizard_V1.x | **89%** | 15 |
| Music_Assembler | **84%** | 3 |
| GoatTracker_V2.x | **80%** | 9 |
| DMC | 61% | 6 |
| MoN/FutureComposer | 53% | 5 |
| JCH_NewPlayer | 46% | 2 |

Codebooks are small and bounded (median 2–15/tune, like STAMP's ~11). This is the **pitched twin of the
STAMP drum codebook** and the single highest-value primitive here.

## 1. In-repo correctness tools (everything you need is in `preframr-tokens`)

- **Byte-exact round-trip** is the correctness oracle: `register_state()` (runs `expand_ops`) must
  reproduce the raw register log. A wrong detector falls back to RESID (the skeleton's verify-or-RESID
  net) — it never corrupts; lean on that.
- **Test through real `RegLogParser.parse()`**, never synthetic dfs — follow the existing
  `tests/test_stamp_pass.py` and `tests/test_held_arp.py` patterns (real fixtures, post combine/quantize).
  Synthetic-df tests ship false-green (a pass can be a no-op on real data). Reuse those fixtures; add
  small synthetic register dumps only as *supplementary* unit cases.
- **Isolation oracle for a pass:** capture the pass's input vs output df, `register_state` both — NOT
  all-off-parse vs on-parse (skeleton-off still runs `quantize_freq_to_cents`, so configs differ beyond
  the pass; this invalidated an earlier measurement).
- **RESID introspection:** `SkeletonPass._resid_diag` (inert sink) records every claimed note
  `(reg, is_resid, note, onset_fr, rec)` with `rec=[(offset, ctrl, is_pitched, fn)]` — the per-frame
  **note-relative semitone offsets** are exactly the wavetable sequence you mine in §2. Keep this sink.
- **Everything runs in docker** (the host toolchain is not the gate): per-change
  `docker run --rm --network host -v /scratch:/scratch -w $PWD anarkiwi/preframr-tokens-test python3 -m
  pytest tests/ -k <area> -q`; full gate `docker build -f Dockerfile .`.
- **Drain measurement is author-side.** Corpus-scale per-engine `UNRESOLVED→0` (needs sidid + the HVSC
  corpus, which live outside this repo) is the author's acceptance check. Your in-repo proxy: byte-exact
  round-trip + a RESID-note-count drop on the test fixtures with the new flag ON vs OFF (assert it in a
  test). Do not import or shell anything outside `preframr-tokens`.

## 2. Workstream A — the WAVETABLE codebook primitive (the big drain)

**Mechanism background (embedded; you implement on register writes, not driver tables).** The dominant
engines drive pitch ornament from a **wavetable**: a per-frame program of note-relative semitone offsets
(transpose by the note), interspersed with absolute notes, a *hold* (repeat previous), and a *jump* (loop
to an index). E.g. JCH/GoatTracker wave-table note column: `00–5F` = note-relative offset (the arp),
`80–DF` = absolute note (melodic note straight from the table), `7E` = hold, `7F` = jump (next byte = loop
index). The same program is replayed at every note (transposition-invariant), which is **why the offset
sequences recur** (§0b). You never parse the driver's table — you mine the **rendered** note-relative
offset sequence that already lands in the RESID dump (`_resid_diag` `rec` offsets).

**Concept.** A wavetable run = a recurring note-relative offset sequence. The RESID escape already stores
it losslessly; this primitive **byte-identically reclassifies** that dump into a codebook DEF+REF so the
model predicts "play wavetable #k" instead of an arbitrary offset list. It is held-ARP generalised from
"period-≤8 cycle" to "arbitrary sequence with a loop point, shared via a codebook."

**Follow the existing patterns:** STAMP (`macros/stamp_pass.py` + `StampDecoder` in `macros/decoders.py` +
ops in `stfconstants.py` + `DecodeState.stamp_table`) for the inline-redefinable codebook; held-ARP
(`skeleton_pass._orn_rows` RESID→reclassification, `held_cycle_offsets`) for the byte-identical replay.

- **Detect (per tune), gate `wavetable_pass` default OFF.** For each skeleton note that floors to RESID,
  take its note-relative offset sequence from `_resid_diag`. (1) **Onset-strip** the HR/test/noise onset
  frames using the control-aware `_is_pitched_frame` already in the skeleton. (2) **RLE-collapse** holds.
  (3) Detect a **loop point** (a repeating tail) → store `(prefix, loop_body)`. (4) Make it
  **noise-inclusive**: detect over ALL frames (the table drives waveform AND pitch), carrying a per-step
  waveform/accent marker so a noise-tik step is part of the program, not a sequence break (this folds in
  the System6581 / §3.1 fix). (5) Canonicalise; group sequences recurring `≥ WT_MINREP` (start at 2; STAMP
  uses 3) into a per-tune codebook.
- **Encode — inline, redefinable** (mirror STAMP): ops `WAVETABLE_DEF id [steps] WAVETABLE_END` +
  `WAVETABLE_REF id voice [base-note]`. Steps are small signed atoms (offset / absolute-note-flag+note /
  hold / loop-index) laid out **Unigram-clusterable** (shared sub-sequences → shared sub-tokens). A later
  `WAVETABLE_DEF id` rebinds (streaming dictionary). Note-relative storage → one DEF serves all
  transpositions (the reason recurrence is high).
- **Decode** — new `WavetableDecoder`, registered in `DECODERS` and in the §4 `OpContract` registry: live
  `id→sequence` table on `DecodeState` (a later DEF rebinds); on REF, replay offsets onto
  `last_skel_note[reg]` (+ base) via `pending_set_writes`, one per frame, loop unrolled to the note length
  — exactly `StampDecoder._ref` / `OrnamentDecoder._queue`. **Byte-identical** to the RESID it replaces, or
  fall back to RESID (prove with the isolation oracle).
- **Relationship to existing passes.** held-ARP stays the within-note-periodic fast path; WAVETABLE is the
  cross-note-recurring general case. STAMP stays the no-pitched-frame / exact-freq-series drum case;
  WAVETABLE is note-relative pitched — keep them separate ops. Where held-ARP and WAVETABLE both fire,
  WAVETABLE wins only if byte-exact and shorter.
- **In-repo acceptance:** byte-exact round-trip on fixtures with `wavetable_pass` ON; a test asserting
  RESID-note count drops materially on a wavetable-heavy fixture; codebook size bounded. (Author then
  confirms the corpus per-engine drain: SidWizard 63→≤15%, GoatTracker 93→≤30%, Music_Assembler 73→≤20%.)

## 3. Workstream A2 — frontier tail primitives (mechanisms embedded; build only what still leaks)

Re-add a primitive only if the residue still shows it after WAVETABLE (the author re-profiles and tells
you, or you see it persist on fixtures). Each: own branch, gated default-OFF, byte-exact, own tests.

1. **Noise-inclusive control-aware ARP.** Carry a per-cycle gate-off / noise-tik accent **inside** the arp
   cycle instead of breaking period detection on it (System6581: a period-3 chord `[+5,+2,0]` with a
   gate-off + noise-tik frame interleaved each cycle). Likely subsumed by WAVETABLE's noise-inclusive
   detection — verify first; only add if a pure-ARP residue remains.
2. **SLIDE target+duration form.** Some engines (MoN/FutureComposer) compute `(target−cur)/duration` and
   land *exactly* on the target after N frames; the current rate-only SLIDE can't reproduce it. Add a
   target+duration variant to the ORN SLIDE (byte-exact landing).
3. **VIB delay+length(+shape).** Sine-LUT vibrato with an onset delay and finite length (MoN); current VIB
   is depth+rate only. Add delay+length(+shape) params.
4. **SWEEP loop-period + auto-Dive flag.** Extend the shipped SWEEP with a `loop_period` (a looping
   freq-domain arp, e.g. constant −624/frame, period 15 — SoundMonitor) and an auto-trigger-at-onset flag
   (an instrument-property "Dive" applied per note with no command).
5. **PERC primitive.** A no-pitched-frame drum = noise waveform + a freq-hi table or sweep, control-gated.
   These are real drums STAMP didn't catch because they don't recur exactly; a parametric percussion
   primitive drains them.
6. **Un-traced engines (AUTHOR-SIDE — do not attempt from this repo).** A few small engines are 100%
   UNRESOLVED and need register-level reverse-engineering against driver sources (outside this repo): GMC/
   Superiors, SynC, Jeff, SkyLine_Editor, LordsOfSonics. The **author** traces these and hands you a
   primitive spec in the same shape as 1–5 if one is needed. Do not guess them.

## 4. Workstream B — constrained decode: registry refactor + materialization

**Why this exists.** The STAMP/PATCH/WAVETABLE codebooks introduce DEF→REF backrefs. The decoders build a
**live id→def table during decode** and **silently drop a REF to an undefined id** (`StampDecoder._ref`:
`frames = state.stamp_table.get(id); if not frames: return None` — `decoders.py:819`; `PatchDecoder._ref`
same). At training time DEFs precede REFs so the table is populated; **at inference a prompt window — or a
generation that slid past context — can carry a REF whose DEF is out-of-window → the drum/patch/wavetable
silently vanishes from the render.** This must be made impossible.

**The abstraction is broken — fix it before wiring codebooks in (author directive).**
`preframr_tokens/constrained_decode.py` is **1111 lines and shares ZERO code with the decoders**
(`grep -c 'DECODERS|DecodeState|expand_ops' constrained_decode.py` = 0). `StreamState` **reimplements the
decode state machine a second time** — the per-macro subreg walk (`PendingSlot`/`MacroShape`/`_ShapeRule`/
`_classify_macro_shape`, ~340 lines) and per-op vocab-array enumeration (`precompute_vocab_arrays`/
`precompute_subtoken_arrays`, ~325 lines) — and `macros/validators.py` reimplements the backref walk a
**third** time. Every op lives in three hand-kept copies. **STAMP and PATCH are the proof of drift:** they
were added to `DECODERS` only, so constrained decode silently never learned them. The author's requirement:
**a missing constrained-decode implementation must fail at unit-test time, not ship.** Do **B0/B1 first**;
they are prerequisites, and they make B2–B4 trivial.

### B0 — one op-contract registry (single source of truth)

Define one `OpContract` per op, **co-located with its `MacroDecoder`** (so adding a decoder and a contract
is one edit), collected into a registry `OP_CONTRACTS: {op_code: OpContract}`. `OpContract` declares:

- `op_code`.
- `decode` — the existing `MacroDecoder.expand(row, state)`, unchanged.
- `shape` — the subreg / sub-token walk as data, not code: an ordered list (or small DFA) of expected slots
  with their value-classes. Variable-length DEFs use a "repeat-until-END" slot. Examples:
  `STAMP_DEF → [id] then STEP* until END`; `PATCH_DEF → [id, STEP(AD), STEP(SR)]`;
  `WAVETABLE_DEF → [id] then STEP* until END`; `SWEEP → [START_HI, START_LO, DELTA_HI, DELTA_LO, LEN]`;
  `*_REF → [id (+voice/base)]`. This replaces `MacroShape`/`_ShapeRule`/`_classify_macro_shape`.
- `legal_next(absstate) -> predicate over the vocab` — which (op, subreg, value-class) tokens may come next
  given the abstract state. **For a `*_REF`/`*_SET`: legal iff its id ∈ the relevant live table.** For a
  `*_DEF`: always legal (and `update` adds the id). For the old distance backrefs: keep the existing
  "distance can't reach before frame 0" rule, expressed as a `legal_next`.
- `update(absstate, token)` — advance the abstract state (frame_count, budget, sval/fn, pending slot, and
  the live `stamp_table`/`patch_table`/`wavetable_table` id-sets, back-ref `output_frame_count`).

Then **generate** `mask_logits`/`compute_invalid_mask`, the `validate_*` net, and `precompute_*_arrays` by
iterating `OP_CONTRACTS` — no per-op hand-coding in three files. `validate_*` becomes exactly *"replay the
stream; assert each token ∈ `legal_next(state_before_it)`"* (one function, delete `validators.py`'s bespoke
walks). The 1111 lines collapse to: the registry + one generic walker + the genuinely-inference-only
Unigram sub-token assembler (which still feeds assembled atomic ops into the **shared** `update`).

`AbsState` is the **mask-relevant projection of `DecodeState`** (frame_count, budget, sval, fn,
pending-slot, the live id-sets). Document the projection explicitly so B1's equivalence test can check it.

### B1 — one state machine for training AND inference (author directive)

Decode is already shared (`expand_ops`/`DECODERS`); only the mask runs on a separate state. Make the mask's
`AbsState` advanced by the **same `OpContract.update`** the decoder's state transitions imply (decode and
mask are two views of one machine). Make **self-containment ONE function** used by both
`iter_self_contained_row_blocks` (training blocks, `macros/blocks.py`) and the inference prompt builder —
so materialization (B3) is identical code on both paths. Any perf-shaped duplicate (the GPU-hot
`precompute_*` arrays) must be **generated from** the registry and covered by the B0/B1 equivalence test,
never hand-maintained.

### B0/B1 regression test spec (golden-master — REQUIRED before deleting any old code)

The old constrained decode **works today** for the existing ops; the refactor must be provably
behaviour-preserving before you remove the hand-written code. Build the tests in this order:

1. **Capture golden outputs of the OLD code** (a new `tests/test_constrained_decode_golden.py`). On a
   corpus of real token streams (the existing tokenizer test fixtures + a handful of representative tunes,
   both vocab modes: atomic and sub-token), serialize as committed golden fixtures: (a) the OLD
   `precompute_vocab_arrays` + `precompute_subtoken_arrays` arrays; (b) the OLD `mask_logits` invalid-mask
   **at every position** of every stream (replay, snapshot each step); (c) the OLD `validate_back_refs` /
   `validate_pattern_overlays` accept/reject verdict on every stream **and** on a set of deliberately
   corrupted streams (truncated macro, orphan overlay, out-of-range distance).
2. **Build the registry-generated code** (B0/B1) alongside the old, not replacing it yet.
3. **Equivalence test — assert byte-identical (old vs new).** New `precompute_*` arrays `==` golden
   (element-wise); new `mask_logits` invalid-mask `==` golden at every position of every stream; new
   `validate_stream` verdict `==` golden on every valid and corrupted stream. **Only when green do you
   delete the old `StreamState` internals / `validators.py` walks.**
4. **mask⟺decode equivalence (forward invariant).** Replay each stream through `expand_ops`/`DecodeState`
   AND the new `AbsState`; assert the mask never forbids a token the decoder accepts, and the generated
   validator rejects exactly the streams the decoder would mis-handle (a dangling ref the decoder would
   silently drop → the validator must reject). Guards a contract that is self-consistent but wrong.
5. **Completeness test (the "fail at unit-test time" requirement).** Assert
   `set(OP_CONTRACTS) ⊇ set(DECODERS)` **and** ⊇ every op any pass can emit. Derive the emit-set
   programmatically (a registry of pass→ops, or scan the op constants the passes reference) — not a hand
   list, so a new op cannot be quietly omitted. Demonstrate it bites: a unit test that registers a dummy
   op in `DECODERS` without a contract and asserts the completeness test goes red.
6. **Property tests.** Random `AbsState`s: the mask never permits a `*_REF`/`*_SET` to an id ∉ the live
   table; `*_DEF` always permitted; after `DEF(id)`, `REF(id)` becomes permitted; a `*_REF` to a just-
   rebound id resolves to the new def.

### B2 — codebook contracts (register, don't hand-code)

Add the `OpContract`s for `STAMP_REF`/`STAMP_REL_REF`/`PATCH_SET`/`WAVETABLE_REF` (legal iff id ∈ live
table) and `*_DEF` (always legal, adds id) + the live `stamp_table`/`patch_table`/`wavetable_table` to
`AbsState`. With B0 done, `mask_logits`, `validate_*`, and `precompute_*` pick them up automatically, and
the completeness test now **requires** them.

### B3 — prompt-window materialization (shared train/inference path)

When a window references a codebook id whose DEF is outside it, **materialize** so the live table is
seeded. `iter_self_contained_row_blocks` already self-contains *training* blocks (expand to literals →
slice → re-tokenize, re-mining DEFs) — extend the **same function** (B1) to cover the codebooks, and
provide the **tokens-side API** the framework calls (see API contract below). Two interchangeable forms,
provide at least the snapshot-seed:
- **Decoder snapshot-seed:** the decoder accepts an initial `{stamp_table, patch_table, wavetable_table,
  last_skel_note, last_val, last_freq_v0}` snapshot, so render resolves out-of-window refs without
  re-emitting tokens. Cheapest.
- **DEF preamble:** `materialize_codebook_preamble(slice_df, prior_state) -> def_rows` emits the minimal
  `*_DEF` rows at the window head (needed when the *model*, not just the renderer, must condition on the
  def). Also re-anchor the first `SKEL` per reg to `SKEL_SUBREG_ABS` and seed `last_val`/`last_freq_v0` —
  the self-contained re-tokenize already does this for relative pitch; confirm it also re-emits codebook
  DEFs when the new passes are on.

### B4 — validation = legality replayed (generated)

With B0, `validate_*` is the registry's `legal_next` replayed — "every STAMP/PATCH/WAVETABLE ref resolves
to a prior DEF" is enforced for free. Keep a single public `validate_stream(df)` entry point. Tests:
(i) the mask never permits an undefined ref (property, generated per registry); (ii) a mid-song window
whose DEF is out-of-window **round-trips byte-exact after materialization**; (iii) a stream with a
forced-undefined ref is rejected by the generated validator, not silently dropped.

### Cross-repo API contract (so you never read the framework)

The framework (author-wired, outside this repo) calls only this **stable tokens-side public API** — keep
these names/signatures so the author's one-line hooks compile:
- `StreamState(vocab_arrays, init_*, irq, remaining_steps, logger=…)` with `.update(token_id)` and
  `.mask_logits(logits) -> logits` — already the inference contract; preserve it through the refactor.
- `precompute_vocab_arrays(tokens_df)` and `precompute_subtoken_arrays(tokens_df, regtokenizer, pad_id)` —
  preserve signatures; change only the internals (now registry-generated).
- `validate_stream(df)` (supersedes `validate_back_refs`/`validate_pattern_overlays`; keep thin shims with
  the old names re-exported so nothing breaks, or coordinate the rename with the author).
- The materialization entry point (`materialize_codebook_preamble(...)` and/or the decoder snapshot-seed
  kwarg) — give it a stable signature and document it in the module docstring.
Anything the framework must change beyond calling these is **out of your scope**: list it in your handback
notes for the author. Do not edit, or depend on reading, the `preframr` repo.

## 5. Gates (every step — non-negotiable)

- **Docker only; full CI before any PR.** `docker build -f Dockerfile .` runs the whole `run_tests.sh`
  gate (black, pytest, pylint-curated, pyright, **coverage ≥85**). Never gate on the host.
- **Byte-exact round-trip** via `register_state` is the correctness floor; new primitive ⇒ a **shared
  encode/decode expansion fn** (encoder verify and decoder cannot disagree) + a `MacroDecoder` in
  `DECODERS` + an `OpContract` (§4, enforced by the completeness test).
- **Default-OFF flags**, one mechanism per branch → PR. A release never changes default behaviour.
- **Unigram-clusterable** atom layout (small DEF/REF atoms the downstream Unigram tokenizer sub-tokenises —
  the opposite of the refuted `motif_pass`).

## 6. Cross-repo release (author-handled — you keep flags default-OFF and the API stable)

A tokens op/flag change is a 3-repo change, but **the author handles the PyPI tag + the `preframr` bridge +
the audio audition.** Your responsibility: keep every new mechanism behind a default-OFF flag, keep the §4
API contract stable, and note any required framework change in your handback. Do not touch other repos.

## 7. Build order

1. **WAVETABLE codebook (§2)** — biggest drain; recurrence-codebook first, then loop/absolute-note/
   noise-inclusive. Byte-exact + RESID-drop tests on fixtures.
2. **Constrained-decode abstraction fix (§4 B0/B1) — BEFORE wiring any codebook into it**, with the
   golden-master regression spec; else you add a fourth hand-kept copy and more drift. Then B2–B4 are
   small registry entries the completeness test forces. May run in parallel with (1); codebooks are not
   deployable until this lands.
3. **Frontier tail (§3.1–3.5)** — only what still leaks. Re-test between each.

## 8. Acceptance

**Agent-side (your gate — all green in-repo before handback):**
- Each primitive: byte-exact round-trip through real `parse()` on fixtures with the flag ON; a test showing
  RESID-note count drops with the flag ON vs OFF; codebook bounded.
- B0: completeness test red when an op lacks a contract (demonstrated with a dummy op); golden-master
  equivalence (old==new) green; mask⟺decode equivalence green; property tests green; `validators.py`
  bespoke walks deleted; `constrained_decode.py` materially smaller.
- B3: a mid-song-window-with-out-of-window-DEF round-trips byte-exact after materialization.
- Full `docker build` gate green (coverage ≥85).

**Author-side (author runs after handback — not your responsibility, do not build for it):**
- Corpus per-engine `UNRESOLVED→0` re-profile (sidid + HVSC); spot-check drained notes match the documented
  mechanism (not loosened fitters); 12-SID WAV audition before any default flip; PyPI release + `preframr`
  bridge.

## 9. What is NOT a floor

The non-recurring in-tune wavetable sequences (11–54% per engine) are still **structured wavetable
programs**, just not repeated *in that tune* — encode them as an inline structured wavetable token (same
op, no codebook ref) and/or a cross-tune codebook (the author measures cross-tune recurrence). Do not label
them aperiodic. Only after every engine's documented mechanism is modelled byte-exact does any genuinely-
improvised one-off go to the audition-gated lossy tier — author's call, never a default, never a substitute
for modelling a mechanism.

## References (in `preframr-tokens` — your working set)

- Code to mirror/extend: `macros/stamp_pass.py`, `macros/skeleton_pass.py` (held-ARP, `_resid_diag`,
  `_is_pitched_frame`), `macros/sweep_pass.py`, `macros/decoders.py`
  (`StampDecoder`/`PatchDecoder`/`OrnamentDecoder`/`DECODERS`), `macros/blocks.py`
  (`iter_self_contained_row_blocks`), `constrained_decode.py` (`StreamState`/`mask_logits`/`precompute_*`),
  `macros/validators.py` (to be folded into the registry), `stfconstants.py` (op/subreg constants).
- Test patterns to follow: `tests/test_stamp_pass.py`, `tests/test_held_arp.py`, `tests/test_sweep_pass.py`,
  `tests/test_arbiter.py`; resume guide `RESID_ZERO_IMPLEMENTATION.md`.

*(Author-side background — summarised above; you do NOT need to read these: the per-engine residue +
recurrence data, the SID driver ornament reference, the xpt probes, and the `preframr` inference/training
code. They informed this spec; everything actionable is embedded here.)*
