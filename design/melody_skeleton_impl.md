# WORK ORDER (SELF-DIRECTING): the melody-skeleton interval layer on the generator-MDL freq channel

**Status:** Pending impl — **auto-gated on the generator-MDL pipeline landing** (the other agent's
`preframr-tokens/AGENT_TASK_generator_pipeline.md`). **This file is self-directing: an agent told only "execute
this .md" must run §A's start-gate first — wait (autonomously, by re-checking on an interval) until the
generator pipeline is the deployed default on tokens `origin/main`, then start §1 with NO further help or
decisions.** Do not change the other agent's instructions; do not start partial work before the gate passes.
Cross-ref [`generator_mdl_representation.md`](generator_mdl_representation.md) (the substrate), the
[`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md) "Compatibility" section (why
this is the melody fix), [`encoding_principles.md`](encoding_principles.md) (P4.2/P5/P6).

## §A. START-GATE — run this FIRST; it is the entire "wait for the other agent" mechanism

Executing this file means: **(1) run the landing check; (2) if WAITING, schedule a re-check and STOP this turn;
(3) if LANDED, start §1 immediately.** No human decision is needed at any step — the check is the decision.

**The landing check (copy-paste; LANDED only when the generator pipeline is merged AND is the default AND the
subsumed zoo is gone on tokens `origin/main`):**
```bash
T=/scratch/anarkiwi/preframr-tokens
git -C "$T" fetch origin -q && \
  git -C "$T" show origin/main:preframr_tokens/tokenizer_config.py 2>/dev/null | grep -q '"generator_pass"' && \
  ! git -C "$T" cat-file -e origin/main:preframr_tokens/macros/freq_trajectory_pass.py 2>/dev/null && \
  ! git -C "$T" cat-file -e origin/main:preframr_tokens/macros/skeleton_pass.py 2>/dev/null && \
  echo LANDED || echo WAITING
```
- **`WAITING`** → the generator pipeline is not yet the default. **Schedule a wake-up ~30–60 min out
  (`ScheduleWakeup`, or a `/loop` re-running this file) and STOP — do not block, do not start.** On each
  wake-up, re-run the check. (Optional faster signal: also poll `gh -R anarkiwi/preframr-tokens pr list
  --state merged --search "generator"` — but the file check above is authoritative.)
- **`LANDED`** → proceed immediately to §1. The tokens 0.45.0 PyPI release is orthogonal and **not** required
  to start this in-repo work (you build on tokens source `origin/main`).

**On LANDED, set up and go (no decisions):**
```bash
cd /scratch/anarkiwi/preframr-tokens && git fetch origin -q && git switch -c melody-skeleton origin/main
```
Confirm the base is green first (`git -C "$T" log origin/main -1` is a merged PR; the tokens suite passes on
it). Then execute §1–§6 end-to-end and open a PR (merge it on green if the repo allows; else leave it open).
You operate **entirely inside preframr-tokens** (this doc is your spec; you need not move it). Everything below
is fully specified — implement it as written, raise nothing back.

## 0. Why this exists (the gap the generator pipeline leaves)

The generator-MDL pipeline makes **structure** learnable (gesture type/shape, DEF→REF copy, counter
elimination) and **de-ornaments by construction** (arp/vibrato/slide are separate note-relative atoms, off the
melody line). But it encodes each freq note-onset as a **raw 16-bit absolute pitch** (`SWEEP_OP` start /
`GEN_TABLE` base_note). **Absolute onset pitch is high-entropy with no local determinant ⇒ ~0 next-token**
(the V0-onset≈0 result; Principle 4.2). So the generator pipeline alone does NOT make melody learnable — it
removes the *pollution* (ornament) but leaves the melody line in its *unlearnable absolute form*.

**This work order adds the missing layer: encode the melody line as key-invariant intervals.** It is the one
mechanism that demonstrably transfers melody — measured **held-out next-interval ≈ 0.52, beating the cross-tune
2-gram ceiling (0.41) every seed** (genuine transfer, not memorization; ≫ the gate-anchored 0.225). Without
it the generalization goal fails on melody; with it the melody line is learnable in the only sense the data
supports (a *plausible*, transferable next note — exact pitch caps ~0.51 even for a memorizer, P5/P6).

## 1. The design (three pieces; the generator pipeline already gives #3)

1. **Note segmentation — intrinsic level-change ∪ gate** (NOT raw gate). Held-gate/legato drivers move pitch
   under one sustained gate, so gate-on under-segments. Use the landed `TrajectoryAnchorPass` pass-1 detector
   (sustained pitch-level change ∪ gate-on for re-struck same-pitch notes). **NOTE:** the generator pipeline
   DELETES `trajectory_anchor.py` (§4 of its work order) — recover the pass-1 detector from git
   (`git show <pre-deletion>:preframr_tokens/macros/trajectory_anchor.py`) and re-introduce it as the
   note-onset segmenter (segmentation only; it does not emit ops).
2. **Interval onset encoding (THE melody token).** For each note onset on a voice's freq channel, encode the
   note's pitch as a **signed semitone interval from the previous note's pitch**, in the LUT note domain:
   `interval = note_of(f_onset, ref) − note_of(f_prev_onset, ref)`. The first note of a voice is absolute.
   Lossless (a running sum reconstructs absolute note; the exact freq residual rides as in the generator's
   freq encoding). Intervals are **key-invariant** (transfer across transposition) and **low-cardinality**
   (cluster at 0/±few) — exactly Principle 4.2's "anchor to a nearby reference."
3. **Within-note ornament = the generator's note-relative atoms** (already built): `GEN_TABLE` arps are
   note-relative; `ACCUM`/`TRI` *within a note span* should be keyed relative to **this note's** pitch (extend
   the generator's note-relative keying from TABLE to the within-note ACCUM/TRI start). De-ornamentation is
   thus complete: the melody line is the interval sequence; everything per-frame is a note-relative gesture.

## 2. Precise spec (builds on the generator pipeline's freq channel)

The generator pipeline encodes freq as one 16-bit channel → `SWEEP_OP`(HOLD/ACCUM) / `GEN_TRI` / `GEN_TABLE`
atoms, with `GEN_TUNING` carrying `ref_q` and `note_of`/`recon` the LUT maps (§1A there). This layer changes
ONLY how the **note-onset base pitch** of each voice's freq atoms is keyed:

- **Run the segmenter** (piece 1) per voice on the freq channel → note-onset frames.
- **A freq atom that STARTS on a note-onset frame is a melody onset.** Replace its absolute start/base note
  with a **`MELODY_INTERVAL` atom** (new op, free id in the generator's range; e.g. `MELODY_INTERVAL_OP=89`):
  subreg `INTERVAL` = signed semitone delta from the previous onset's note (zig-zag/bias-encoded to stay a
  small non-negative token), subreg `FIRST`=1 + `NOTE_ABS` for a voice's first note. The atom's residual +
  its generator kind (HOLD/ACCUM/TABLE/TRI) and length are unchanged — only the *base pitch* is re-keyed.
- **Decoder:** maintain per-voice `cur_note`; on a `MELODY_INTERVAL` atom, `cur_note += interval` (or
  `=NOTE_ABS` if FIRST); the freq value = `recon(cur_note, ref) + residual`. Within-note ornament atoms decode
  relative to `cur_note` (offsets added to `cur_note`). Byte-exact: the running sum reproduces the absolute
  note; residual is exact (the generator's losslessness is preserved — this is a re-keying, not a value change).
- **Non-melodic voices (percussion/swept/low — e.g. Facemorph v0):** a voice whose freq never settles to a
  stable note grid (the generator's note channel is degenerate) is NOT interval-segmented — leave it as the
  generator's raw freq atoms. Decide per voice by a cheap stability test (fraction of frames whose `|residual|`
  is small); this is waveform-AGNOSTIC (never read the waveform bit — Facemorph guardrail).

## 3. Losslessness + the byte-exact gate (unchanged contract)
The interval re-keying is a lossless bijection on the note index (running sum), and the residual is carried
exactly, so `decode == register_state` byte-exact is preserved. Gate exactly as the generator pipeline:
`arbitrate(validate=True)`; corpus `reparse=True` byte-exact vs the generator-pipeline default; raw `SET`==0
(unchanged). The segmentation/interval layer must not introduce any residual.

## 4. The learnability gate (THIS is the payoff — not byte-exactness)
Melody is multi-modal; exact next-note caps ~0.51 even for a memorizing n-gram (P5/P6), so **do NOT gate on
exact-token accuracy.** Gate on:
- **Held-out next-interval accuracy** (held-out-by-dump, ×3 seeds) **must beat the cross-tune 2-gram ceiling**
  (the transfer test; the prior interval-skeleton hit 0.52 vs ceiling 0.41 — match or beat). Reuse the
  generalization-probe harness style.
- **Distributional + audition** — ornament/interval emission at ~corpus rate (JS small), and the 12-SID WAV
  audition. The model emits a *plausible*, transferable melody; that is the success criterion.
- **Learnability triage** (`audit/learnability_triage.py --mode blocks --seq_len 8192`): the interval melody
  token must show fast MI-decay + high induction-copy vs the absolute-pitch baseline (the interval is the
  key-invariant, low-cardinality form the theory predicts learnable).

## 5. Composition with the generator pipeline (what changes, what doesn't)
- **Changes:** add the note-onset segmenter + the `MELODY_INTERVAL` re-keying on freq note-onsets; extend
  within-note ACCUM/TRI keying to note-relative.
- **Unchanged:** the generator's `{SWEEP_OP, GEN_TRI, GEN_TABLE}` atoms, the LUT/`GEN_TUNING`, all non-freq
  channels, `InstrumentProgramPass` (ctrl/AD/SR), the residual-zero + byte-exact gates, the digi exclusion.
- **Default-OFF flag** (`melody_skeleton`) until it gates clean + the learnability read is positive; then fold
  into `REGISTERED_MACROS` and ship in the SAME breaking release as the generator pipeline if timing allows,
  else the next minor.

## 6. Tests + PR (same discipline as the generator work order)
- Through the real `RegLogParser.parse()`; xdist-chunked; lint forbids non-directive `#` comments.
- Byte-exact corpus gate (`reparse=True`) + the held-out interval transfer test + the triage read.
- **Module↔macros round-trip:** the SWM/defMON round-trip (§7B of the generator work order) must still pass —
  a melody re-keyed to intervals must still render to the SAME OUTPUT.
- New tests: segmentation correctness (held-gate legato → one note line, not over-segmented), interval
  round-trip (running sum == absolute), non-melodic-voice passthrough (Facemorph v0 unchanged).
- Stay in preframr-tokens; no release/tag without the cross-repo procedure; PR through to merge.

## 7. Honest non-claims (state in the PR; do not relitigate)
- **Not a melody-accuracy bet.** Exact next-note pitch is data-limited (~0.51 ceiling, P5/P6) and will not
  rise; the win is **key-invariance + de-ornamentation → transferable, plausible melody**, scored
  distributionally + by audition. That is "learning melody" in the only sense the data supports.
- **Interval, not absolute, is the lever** — measured (0.52 held-out beats the 0.41 cross-tune ceiling); the
  generator pipeline supplies the de-ornamentation that makes the interval line clean.
