# Macro abstraction: consolidate the fragmented registration surface

**Status:** Analysis (2026-06-03). Architectural read prompted by "is the macro abstraction too
complicated?" Conclusion: the codec *mechanics* are essential; the *declaration/wiring* is accidental
complexity worth collapsing. Matches the standing preference to collapse hand-maintained config
surfaces into one registry-driven mechanism (accepts breaking changes).

## Essential complexity — leave alone
- **Encode/decode duality.** Every macro round-trips byte-exact; two halves is intrinsic.
- **Single `FrameWalker` traversal.** Wide hook surface (`before_row`/`after_row`/`on_marker`/
  `on_frame_tick`/…) but it is ONE decode oracle shared by passes, validators, `expand_ops`, snapshots.
  Splitting into per-use traversals = multiple oracles to keep byte-identical = worse.
- **`DecodeState` + `tick_frame` drain.** Decode is stateful and the per-frame drain interleaves every
  in-flight macro in an AUDIBLE order (ADSR). The cross-cutting coordination can't be factored per-macro
  without re-introducing global ordering anyway.

## Accidental complexity — the real problem
A single macro's identity is spread across ~5 surfaces:
1. a `Transform` in `_REGISTRY` / `pipeline_spec` (`transform_registry.py`) — `OP_CODES`,
   `forward`/`inverse`, `DECOMPOSES_TO_ATOMS`;
2. a `MacroPass` in hand-maintained lists (`FREQ_BLOCK_PASSES`/`PASSES`/`POST_NORM_PRE_VOICE_PASSES`);
3. inline pass sequences hardcoded in `reglogparser.parse`;
4. a `MacroDecoder` in the `DECODERS` registry;
5. a flag in `flag_registry.py`.

The tell: `pipeline_check.py` exists solely to RECONCILE these, carrying hardcoded bridge-sets
(`_KNOWN_PHANTOM_NAMES`, `_HARDCODED_PRE_NORM_TRANSFORM_NAMES`), plus the `block_refire_contract` test.
When a checker with hardcoded bridges is needed to stop N declarations drifting, the declaration is
duplicated. `PassBackedTransform` (bundles `PASS_CLASS`+`DECODER_CLASS`) shows the consolidation was
STARTED but stalled half-done — so both the unifier and the originals coexist, which is why it reads
as "too many abstractions."

## The simpler decomposition (finish the migration, delete the rest)
One declaration per macro: `op_codes · phase · flag · encode(window,state) · decode(atom,state) ·
drain_span(atom)`. DERIVE everything else from the registry: `DECODERS`, the per-phase pass pipeline
(order via explicit phase+priority), the `reglogparser` inline sequences, the `OP_CODES` map, flag
names, block-refire membership. Deletes: the three hand-lists, the inline parser sequences, most of
`pipeline_check.py`, both bridge-sets, and the contract test (membership becomes derived → drift is
structurally impossible). Co-locates a macro's two halves (today split across `passes/*_pass.py` and
`decoders.py`), which is the biggest readability win.

## Caveats
- **Byte-exact-core refactor.** Audible pass order must be reproduced exactly by the derived
  phase+priority sort; gate against current output corpus-wide (`PREFRAMR_VERIFY`/`cb_div_audit`). Risk
  is in ORDERING, not per-macro logic.
- **Genuine non-macros stay out** (`combine_regs`/`add_frame_reg`/`consolidate_frames` — today's
  `_KNOWN_PHANTOM_NAMES`) but should be modeled as explicit ordered pipeline stages, not "names the
  checker ignores."
- **Performance:** NEUTRAL directly (wiring is O(once), not per-row). Value is maintainability + it
  de-risks the decode-core compile and the arbiter `drain_span` contract (one place to add each).

## Localized wart (separable, low-risk)
`FreqTrajectoryDecoder` threads a mutable dict with string keys (`pending_ft = {"subtype","steps",
"esc","in_esc","count","fields",…}`) across calls — a state machine as an ad-hoc dict. A small typed
per-in-flight-macro object would make the hardest decoder readable, independent of the big consolidation.
