**Status:** Review 2026-06-03 — grounded audit of the macro implementations (stable + in-flight residual-SET PRs) against the learnability basis. One HIGH cross-cutting risk found; two MEDIUM; the rest confirm the thesis. Companion to [`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md) (the principles) and [`macro_learnability_triage.md`](macro_learnability_triage.md) (per-pass keep/retire).

# Macro implementation learnability-risk review

Read of `preframr_tokens/macros/*` (stable: skeleton/wavetable/stamp/patch/sweep/loop; in-flight:
ctrl_osc/note_off/ctrl_wavetable). Judged against the basis: a token is learnable when predictable from
LOCAL context with no maintained counter, and recurrence is expressed as DEF→REF copy with the DEF
reachable in-window.

## HIGH — unbounded DEF→REF distance in every codebook (× the mid-song-prompt goal)
**Mechanism.** `wavetable`, `stamp`, `patch`, and the in-flight `ctrl_wavetable` all emit the `*_DEF`
**once, at first occurrence**, then `*_REF`/`*_SET` by id thereafter with **no re-emit, no refresh, no
distance bound** (`if wid not in defined: emit_def` / `emit_first if j==0 else emit_ref`). A DEF at the
song head and a REF 3000 tokens later asks the model to hold a live id→program binding across the whole
tune with no reminder — beyond an induction head's reach.
**Why it's acute here, not generic.** The project's headline task is **arbitrary mid-song prompts**
(PROMPT=2048). A prompt window that starts mid-tune **almost never contains the song-head DEFs**, so
every REF in the prompt points to a program the model never sees — it must hallucinate or fail. The
codebook design and the generalisation goal collide exactly at inference. This is the risk
`macro_learnability_triage.md` flagged as the "codebook-coupling / DEF→REF-distance" gate; the audit
confirms it is structurally present (once-only DEF, unbounded span), and sharpens *why* (the prompt
window excludes the DEF).
**Fixes (Principle 2 — keep the copy local).** Sliding-window DEF refresh: re-emit a DEF when a REF
would otherwise reach back > W tokens (or > PROMPT). Equivalently, guarantee each PROMPT-sized window is
self-contained (a DEF or a self-describing escape precedes the first REF in it). Do **not** solve it with
a global codebook preamble — `global_instr_ids` was refuted for transfer; the locality, not a shared id
space, is what's load-bearing.
**Measure first** (cheap, no training): histogram DEF→REF token distance across the corpus and report
the fraction > 2048 — that fraction is the inference-unresolvable REF rate. Natural extension to
`audit/learnability_triage.py`.

## MEDIUM — `note_off` ships Option B (standalone token), not duration
In-flight `note_off` is **Option B**: a standalone `NOTE_OFF_OP` re-labelling the gate-clear at the
off-frame. The off-event is then a *separate* prediction whose only determinant — the note's intended
length — was set at the onset, which for a held note is many tokens back (a long dependency horizon), and
under teacher-forcing a mistimed off derails the rest. **Option A (carry `duration` in the SKEL atom,
gate-off implied at onset+duration)** co-locates the determinant with the onset → strictly more learnable
(hub Principle 4). The spec's plan (ship B to drain the residual + measure, then migrate to A) is sound;
the review flags B as a **learnability stopgap, not the destination** — schedule the A migration and
compare h_k/onset-consistency, don't leave B as default.

## MEDIUM — cross-voice frame multiplexing raises the per-voice horizon
Not a single macro but the FRAME ordering: 3 voices interleaved per frame put the same-voice melodic
predecessor ~3× further back (the `melody_channel_factorization.md` multiplex finding). Every per-voice
line pays a longer dependency horizon. The structural fix is voice-major lanes
(`superframe_voice_lane_design.md`) — gate it on per-frame h_k, not only onset acc.

## LOW / confirmed-good (the thesis working)
- **`ctrl_osc`, `sweep`** — fully parametric (`PERIOD`+cycle bytes+explicit `LEN` / `START`+signed
  `DELTA`+`LEN`); one atom per run, **no per-frame counter in the encoded stream**. Exactly the
  counter-elimination win (Principle 3). `ctrl_osc`'s redundant held-frame decode writes are audio-inert
  decode artefacts, **not** a learnability issue (the atom is clean).
- **Codebook ids are pure tune-local ordinals**, never value-snapped (`stfconstants` forbids snapping;
  `STAMP_DEF` de-entangled). The old [[codebook-id-snap-corruption]] failure — same gesture → different
  ids → copy-fraction collapse — is **RESOLVED**; induction-copy works *within* a window. (The remaining
  problem is purely the DEF→REF *distance* above, not id instability.)
- **`skeleton`** — SKEL anchors onset as an interval to the prior note on the same reg (local reference,
  Principle 4.2); ornament is a constant-size per-note descriptor, no counter.
- **`loop`** — body inlined (first occurrence self-contained); `PATTERN_REPLAY` back-distance is 16-bit.
  Same window-resolvability caveat as the codebooks but milder (inlined body), lower priority.
- **`stamp` REL** — transpose-relative as a signed delta from a per-hit base (local anchor); good for
  transfer.

## Priority
1. **Measure DEF→REF distance** corpus-wide (the HIGH risk's go/no-go number) → if the >2048 fraction is
   material, windowed DEF-refresh is the fix and is **prerequisite to any codebook learnability A/B and to
   mid-song-prompt inference**.
2. Schedule `note_off` B→A (duration) migration; keep B only as the residual-drain stopgap.
3. Voice-lane de-mux remains the standing ordering lever.
