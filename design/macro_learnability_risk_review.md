**Status:** Review 2026-06-03 (corrected) — grounded audit of the macro implementations (stable +
in-flight residual-SET PRs) against the learnability basis. **The initially-flagged HIGH DEF→REF risk was
WITHDRAWN** — the self-contained-block architecture already bounds it (see below); two MEDIUM ordering/
note-off items remain, the rest confirm the thesis. Companion to [`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md) (the principles) and [`macro_learnability_triage.md`](macro_learnability_triage.md) (per-pass keep/retire).

# Macro implementation learnability-risk review

Read of `preframr_tokens/macros/*` (stable: skeleton/wavetable/stamp/patch/sweep/loop; in-flight:
ctrl_osc/note_off/ctrl_wavetable). Judged against the basis: a token is learnable when predictable from
LOCAL context with no maintained counter, and recurrence is expressed as DEF→REF copy with the DEF
reachable in-window.

## WITHDRAWN — DEF→REF is already block-local (self-contained blocks)
The first cut flagged unbounded DEF→REF distance as HIGH (codebooks emit `*_DEF` once at first occurrence
in the *full-song* tokenization, then REF by id with no re-emit — so a song-head DEF and a far-later REF,
and mid-song prompt windows that exclude the DEF). **This is real of the full-song stream but NOT of what
the model sees.** The model trains and predicts on **self-contained blocks**, built by
**expand-to-literal → slice → re-encode**: `expand_to_literal_form` expands *every* macro (codebooks
included) to literal SET rows, the window is sliced, then `block_refire_pass_names` **re-fires every
reference-op producer per block** — so each codebook DEF is re-mined locally and **always sits in the same
block (≤ seq_len) as its REFs**, never out-of-window. Confirmed: `corpus.make_tokens` ("materialise its
self-contained blocks") for training; `self_contained_prompt_df` for the mid-song prompt (same
expand→slice→re-encode); and `constrained_decode` + the validators' "B3 snapshot materialized from outside
the window" for the generation side. **No sliding-window-DEF-refresh fix is needed — the architecture
already guarantees it.** Residual (LOW): within a block a DEF→REF can still span up to ~block length, but
that is in-window long-range copy (the normal induction-head case), not an unresolvable reference.

> **Tool refinement (done, partial):** `audit/learnability_triage.py` gained `--mode blocks` (the
> self-contained-block stream the model sees) vs the default `--mode song` (full-song `parse()`). Block
> mode is **EXPERIMENTAL** — reproducing the block-builder standalone trips re-encode assertions on tunes
> whose ops need parser context (5/9 in the sample), so it under-covers; the faithful version must route
> through the **Corpus block-builder**. Even partial, it overturned the song-mode read: at block scale
> **full_macros beats the codebook arm on every metric** (h∞/frame 0.40 vs 0.55, in-window induction-copy
> 0.83 vs 0.68) — the codebook's compression doesn't survive to the window the model learns at. See
> `learnability_token_ordering_theory.md` "First read"; certify on full coverage before acting.

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
- **`skeleton`** — SKEL anchors onset as an interval to the prior note on the same reg (local reference,
  Principle 4.2); ornament is a constant-size per-note descriptor, no counter.
- **`loop`** — body inlined per block (self-contained); `PATTERN_REPLAY` whose target precedes the prompt
  is materialised into literals by `self_contained_prompt_df`. Block-local like the codebooks.
- **`stamp` REL** — transpose-relative as a signed delta from a per-hit base (local anchor); good for
  transfer.
- **Codebook ids are pure tune-local ordinals**, never value-snapped — the old
  [[codebook-id-snap-corruption]] copy-collapse is RESOLVED; and per the self-contained-block re-mining,
  ids are effectively *block*-local, so cross-block transfer rides re-mined structure, not a persistent id.

## Is the codebook design itself the problem? (decomposition — wired, run pending)
The triage's `codebook` arm underperforms `full_macros` at block scale, but the comparison is
**confounded**: the codebook arm *swaps the freq substrate* (`freq_trajectory_pass` → `skeleton_pass`,
the proven substrate-ablation win) AND adds the DEF→REF codebooks. To isolate, `audit/learnability_triage.py`
now resolves three configs (`--configs full_macros,skeleton,codebook`): `skeleton` = `codebook` MINUS the
DEF→REF codebooks (stamp/wavetable/patch/wt_*), so **full_macros-vs-skeleton isolates the substrate** and
**skeleton-vs-codebook isolates the codebook effect on a fixed substrate**. If skeleton≈codebook the loss is
all substrate; if codebook is worse, the codebook design hurts — and the likely reasons are: DEF never
amortized at block scale (first use = full DEF+REF, only ~3+ in-window repeats net-win); per-tune/block
ordinal ids carry no transferable meaning (alphabet bloat, copy intra-block only); DEF interior is raw STEP
content not parametric (low copy — cf. `ornament_transfer.md`'s `(table-id,rate,phase)` proposal);
recurrence gate calibrated for song scale not the block window. **Run after the residual PRs land.**

## Priority
1. **Run the codebook-vs-substrate decomposition** (above) once the residual PRs merge — settles whether to
   keep investing in the codebook pipeline or revert to the FREQ_TRAJ substrate.
2. **Faithful block measurement via the Corpus block-builder** — the current `--mode blocks` is 5/9-tune
   partial (standalone re-encode trips on ops needing parser context); route through the Corpus API for
   full coverage before acting on any block-scale finding.
3. Schedule `note_off` B→A (duration) migration; keep B only as the residual-drain stopgap. The off
   determinant is in-window (block-bounded), so this is a consistency/exposure-bias refinement, not a
   horizon fix.
4. Voice-lane de-mux remains the standing ordering lever.
