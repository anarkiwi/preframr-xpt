# WORK ORDER (SELF-DIRECTING): melody learnability — interval-skeleton (layer 2) + cross-voice de-mux (layer 3)

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

**Melody learnability is a THREE-LAYER stack; this work order builds layers 2 AND 3 (both REQUIRED).**
- **Layer 1 (done by the generator pipeline):** de-ornamentation — ornament off the melody line.
- **Layer 2 (§1–§4 here): the interval-skeleton** — encode each voice's note-onsets as **key-invariant
  intervals** (measured held-out next-interval ≈ 0.52 > cross-tune 2-gram ceiling 0.41 — genuine transfer).
- **Layer 3 (§4B here, REQUIRED — the DOMINANT lever): cross-voice de-multiplexing** into contiguous,
  causal-DAG-ordered lanes. **Layer 2 alone does NOT deliver deployed melody learnability:** the 0.52 was
  measured on *extracted, single-voice (already de-multiplexed)* data, while deployed the three voices are
  frame-interleaved, so a melody voice's consecutive notes are separated by the other voices' tokens (P3
  violation — long horizon). The project measured this directly: **deployed melody-onset ≈ 0 vs the ~0.34
  per-voice ceiling, the gap being cross-voice multiplexing** (within-voice factoring was only a ~+0.03
  bonus). So **shipping layer 2 without layer 3 encodes a learnable line the model still cannot reach.**

Without all three the generalization goal fails on melody; with them the melody line is learnable in the only
sense the data supports (a *plausible*, transferable next note — exact pitch caps ~0.51 even for a memorizer,
P5/P6). See [`superframe_voice_lane_design.md`](superframe_voice_lane_design.md) (voice-lane form) and
[`role_lane_factorization.md`](role_lane_factorization.md) (the truer role-lane form) for layer 3.

## 1. The interval-skeleton (LAYER 2 — three pieces; the generator already gives within-note ornament)

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

## 4B. LAYER 3 — cross-voice de-multiplexing (REQUIRED; the dominant melody lever)
Layer 2 makes each voice's line clean + key-invariant but the stream is still **frame-major** (voices
interleaved), so the model's next-melody-note horizon is polluted by the other voices' tokens. Layer 3
reorders the stream into **voice-major lanes** so each voice's interval line is **contiguous** (short horizon,
P3). Build this in the SAME work order — shipping layer 2 without it leaves melody at ~0 deployed.

**Spec (start with VOICE lanes — physical, byte-exact; role lanes are the harder follow-up):**
- **A `voice_lane` reordering pass**, per self-contained block (reuse the `block_refire` machinery and the
  existing `voice_canonical_block_order` / `super_frame` scaffold as the template — see
  [`superframe_voice_lane_design.md`](superframe_voice_lane_design.md)). Within a block, group each voice's
  events contiguously (voice from the FRAME-header packing / `remove_voice_reg`'s `v`), with voice→reg-class
  sub-lanes so PW/filter re-admit without re-fragmenting the melody line; the global filter/mode lane stays
  shared. The **block (superframe) is the harmonic window** — keep blocks short enough that cross-voice
  harmonic context is still in-window (the lane gives melody-self-locality; the block bounds harmonic-context
  distance — do not make blocks so large that harmony becomes unreachable).
- **Lossless = a permutation with a recorded byte-exact inverse.** The decode MUST restore the canonical
  voice-respecting, reg-ascending, frame-major render order (intra-frame write order is audible — the ADSR
  bug; only the canonical order the dumps already use is inaudible; an arbitrary reorder is NOT — see
  `sid_render_fidelity_contract.md` / the sequence-order-normalization finding). So lane reordering carries an
  inverse map (or is derivable from the FRAME header) that reproduces the exact render order. Gate byte-exact
  exactly as everything else (`arbitrate(validate=True)` / `register_state`).
- **Role lanes (follow-up, do NOT block layer 3 on it):** musical roles HOP voices, so a fixed voice-lane can
  split one melodic line; the truer target is role lanes (within-role continuity), per
  [`role_lane_factorization.md`](role_lane_factorization.md). Ship voice-lanes first (implementable, byte-exact);
  open role-lanes as the next refinement once voice-lanes pass the gate.

**Layer-3 gate — UNTESTED at deployment, so gate hard (do not assume "de-mux helps"):**
1. **Triage pre-screen (mandatory, cheap):** `learnability_triage.py --mode blocks --seq_len 8192` on the
   voice-major stream vs the frame-major baseline — the melody-onset token's **MI-decay must shorten and
   induction-copy must rise**. If it does not, STOP and report (do not ship a de-mux that doesn't help).
2. **One canonical run (the go/no-go):** deployed melody-onset accuracy must recover from ~0 **toward the
   ~0.34 per-voice ceiling** vs the frame-major default. Default-OFF flag (`voice_lane`) until this passes.
- Lossless is necessary but NOT sufficient here; the learnability recovery is the whole point.

## 5. Composition with the generator pipeline (what changes, what doesn't)
- **Changes:** (layer 2) add the note-onset segmenter + the `MELODY_INTERVAL` re-keying on freq note-onsets +
  within-note ACCUM/TRI note-relative keying; (layer 3) add the `voice_lane` block reorder pass + its byte-exact
  inverse.
- **Unchanged:** the generator's `{SWEEP_OP, GEN_TRI, GEN_TABLE}` atoms, the LUT/`GEN_TUNING`, all non-freq
  channels, `InstrumentProgramPass` (ctrl/AD/SR), the residual-zero + byte-exact gates, the digi exclusion.
- **Two default-OFF flags:** `melody_skeleton` (layer 2) and `voice_lane` (layer 3). Each folds into
  `REGISTERED_MACROS` only after BOTH its byte-exact gate AND its learnability gate (§4 / §4B) pass. Ship
  layer 2 first if layer 3's triage/canonical gate isn't yet green — but the deployed melody win needs both.

## 6. Tests + PR (same discipline as the generator work order)
- Through the real `RegLogParser.parse()`; xdist-chunked; lint forbids non-directive `#` comments.
- Byte-exact corpus gate (`reparse=True`) + the held-out interval transfer test + the triage read.
- **Module↔macros round-trip:** the SWM/defMON round-trip (§7B of the generator work order) must still pass —
  a melody re-keyed to intervals must still render to the SAME OUTPUT.
- New tests: segmentation correctness (held-gate legato → one note line, not over-segmented), interval
  round-trip (running sum == absolute), non-melodic-voice passthrough (Facemorph v0 unchanged), **and layer 3:
  the `voice_lane` reorder + inverse is byte-exact render-order (canonical, not arbitrary) on the corpus, and
  the SWM/defMON round-trip still produces the SAME OUTPUT under reordering.**
- **Layer-3 learnability gate (§4B) is part of this work order:** run the triage pre-screen; if green, the one
  canonical run is the go/no-go for the deployed melody recovery. Report it in the PR.
- Stay in preframr-tokens; no release/tag without the cross-repo procedure; PR through to merge.

## 7. Honest non-claims (state in the PR; do not relitigate)
- **Not a melody-accuracy bet.** Exact next-note pitch is data-limited (~0.51 ceiling, P5/P6) and will not
  rise; the win is **key-invariance + de-ornamentation → transferable, plausible melody**, scored
  distributionally + by audition. That is "learning melody" in the only sense the data supports.
- **Interval, not absolute, is the lever** — measured (0.52 held-out beats the 0.41 cross-tune ceiling); the
  generator pipeline supplies the de-ornamentation that makes the interval line clean.
- **De-mux (layer 3) is the DOMINANT but UNTESTED-at-deployment lever** — the 0.52 was on de-multiplexed
  single-voice data; deployed melody needs contiguous lanes (P3). Gate it hard (triage + one canonical run);
  do not assume "de-mux helps." Role-vs-voice (melody hops voices) is a real open subtlety — voice-lanes first,
  role-lanes the follow-up.
