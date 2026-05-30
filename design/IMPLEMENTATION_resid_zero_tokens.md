# Implementation spec — RESID→0 encoding stack (for the preframr-tokens agent)

**Status:** Build spec. Audience: an agent working **solely in `preframr-tokens`**. Goal: implement the
full mechanism stack that drives skeleton `RESID` toward 0, under a speculative encoding pipeline.
Everything here is validated by probes in `preframr-experiments/audit/probes/` (paths given per
mechanism) — you do not need to run them, but they are the oracles and the source of the numbers.

## 0. The goal, honestly

`RESID` is the skeleton encoder's lossless escape — a per-frame offset dump the model can't predict.
**Every `RESID` note is a documented engine mechanism** — `sidid`-label the tune and the technique is
always findable (the program's thesis, now verified: see §7 + `sid_driver_ornament_reference.md`).
There is NO "genuinely-irregular floor"; the only place losslessness can't reach is a note a player
genuinely improvised → the audition-gated lossy tier (P8), the LAST resort, never the default.

Measured accounting (`audit/probes/resid_final_accounting.py`). Each mechanism below maps to a
documented engine abstraction; adding a fitter per mechanism drove the unaccounted share down
monotonically (100-song: 1.69→1.44 rel-stamp →1.20 sweep/arp-accent/perc →0.91 decompose →**0.75**
wildcard-stamp+glide). Mechanisms: `STAMP_abs/STAMP_rel/STAMP_wild` (recurring write-series, exact /
note-relative / wide-or-noise-element-wildcarded), `ARP` (noise-inclusive wavetable cycle),
`ARP_accent` (cycle + periodic noise-tik/gate-off accent — System6581), `SWEEP` (freq-domain
const-Δfreq, one-shot/looping/target+duration glide — SoundMonitor/DMC/MoN), `PERC` (no-pitched-frame
drum), `SEGMENT`/`SEGMENT_fit`/`DECOMP` (held-gate concatenations split into the above). **1500-song result
(539,611 RESID notes):** STAMP_abs 96.68 · STAMP_rel 0.78 · STAMP_wild 0.29 · ARP 0.32 · ARP_accent
0.07 · SWEEP 0.23 · PERC 0.10 · SEGMENT 0.44 · SEGMENT_fit 0.05 · DECOMP 0.22 · **UNACCOUNTED 0.83%**
(4485 notes; from 1.69% pre-fitters).
The residue is held-gate concatenations of KNOWN mechanisms; **literal 0 = the encoder's segment-then-fit (§6,
§7b) decomposing them losslessly** — the probe is a classification proxy, do NOT push it to 0 by
loosening fitters (that would be false coverage).

## 1. Start from a clean `main` — merge the foundation branches FIRST

**Begin from a clean `main` in every repo. Do NOT build on session working branches.** The
prerequisite foundation has been pushed as PRs; merge them into `main`, then branch from `main` for all
your work:
- **`preframr-tokens` branch `feat/transient-tolerance`** (pushed; open a PR → merge to `main`):
  control-aware note basing (`_rebased_note`/`_is_pitched_frame`/`_ctrl_at`/`_context`), held-gate/
  level-change/fast-run re-segmentation, **iter-1 `ARP_MAX_PERIOD` 8→16** (held chord-arps), and the
  **inert probe sinks** (below). Gate it through the §8.0 full-docker CI run before merging.
- **`preframr-xpt` branch `feat/resid-zero-design`** (pushed; reference only — the design docs + RE
  probes): merge or leave as a branch; you READ these, you don't ship them.

After merging, `main` of `preframr-tokens` contains:
- `preframr_tokens/macros/skeleton_pass.py` — `SkeletonPass` (gate `--skeleton-pass`): segments freq
  regs into notes, emits `SKEL` (pitch) + `ORN` (PLAIN/OCTAVE/ARP/SLIDE/VIB/RESID). Operates on RAW
  per-voice regs (freq 0/7/14, ctrl 4/11/18, pw 2/9/16, adsr 5,6/…, filter 21–24). Voice-canonicalisation
  to regs 0–6 happens AFTER this pass. `_orn_rows` emits the constant-size ORN descriptor;
  `drop_idx`/`new_rows` is the existing **destructive, first-come claim** — the seed of §2.
- the control-aware foundation + iter-1 ARP cap above.
- **Inert probe sinks (default `None`, no prod effect):** `SkeletonPass._resid_diag` records every
  claimed note `(reg, is_resid, note, onset_fr, rec)` where `rec=[(offset, ctrl, is_pitched, fn)]`;
  `SkeletonPass._df_sink` captures the raw apply-df (`+_fr` frame index). These are the interface for
  the `preframr-xpt` probes; keep them. The §2 pipeline removes the need for `_df_sink`.

If for any reason the foundation PR is NOT merged, re-derive iter-1 (`ARP_MAX_PERIOD=16`) and the inert
sinks from `feat/transient-tolerance` as your first commits — but the intended path is merge-then-branch
from a clean `main`.

## 2. Build the speculative pipeline FIRST (the framework)

(`design/speculative_encoding_pipeline.md`.) Today's passes mutate the df in strict order, so one pass
destroys what another needs (the `_df_sink` workaround proves it) and order forces premature choices
(drums become RESID because the skeleton fits them as notes first). Replace with **claims +
arbitration**:

- **Immutable source.** Passes read the parsed register log read-only.
- **`Claim` = (`writes`: source row-set consumed, `tokens`: replacement rows, `score`).** Each pass is
  a PROPOSER returning claims; no ordering dependency.
- **Speculation:** claims may overlap (drum-stamp vs skeleton-note vs RESID for one span; SLIDE vs
  SWEEP vs RESID for a ramp), from different passes or the same pass offering alternatives.
- **Arbiter** selects an accepted subset that **partitions the source writes (lossless)** and maximises
  the objective. Raw/`RESID` is the always-valid fallback claim → coverage guaranteed.
- **Objective (lexicographic):** (1) **fidelity** — byte-exact per-frame oracle + emulator round-trip;
  lossy only with explicit penalty + audition flag (P8); (2) **learnability** — codebook reuse,
  rhythmic-grid regularity, Unigram-clusterable atom layout, separability (`encoding_principles.md`);
  (3) **token budget**. Per-region AND per-tune (mode-level claims) → "most appropriate encoding for a
  tune".
- **Arbiter mechanics:** writes group per voice × time-span; greedy-by-score with a lossless backstop
  is the first cut, refine to per-voice DP (weighted interval scheduling) where claims nest.
  Deterministic tie-break (pass priority, then source order) — required by the deterministic suite.

**Migration (each step byte-identical → suite-gated):** (a) wrap existing passes to RETURN
`(claimed_writes, new_rows, score)`; arbiter applies in current priority → identical output; (b) make
passes read the immutable source (delete `_df_sink`); (c) add competing claims for drum-vs-skeleton;
(d) per-tune mode selection; (e) speculative alternatives.

## 3. Mechanism: held-ARP (extend the landed iter-1)

- **Detect:** pitched-offset cycle. iter-1 covers period ≤16 via `_minimal_period`. **Extend** to
  **irregular per-step duration**: RLE-collapse the offset run, detect the cycle on the collapsed
  values, store `(cycle, per-step-hold)`. Covers wave-delay chord-arps whose holds vary.
- **Encode:** existing `ORN_TYPE_ARP` cycle + a hold descriptor (or keep the expanded cycle when holds
  are uniform — already works). **Decode** via `cycle_frame_offsets` (+ holds). `_reconstruct` must
  still equal the floor or fall to RESID (byte-exact).
- **Validate:** `audit/probes/resid_trace.py` (`HELD-ARP` bucket); `resid_final_accounting.py` ARP%.

## 4. Mechanism: recurring-STAMP codebook (drum + effect) — the big one (~98%)

(`design/percussion_stamp_encoding.md`.) A drum/effect is an **exact write-series stamped repeatedly
in time**, not a waveform. Validated: 96% of RESID notes / 79% frames are recurring stamps, ~85%
on a rhythmic grid (`audit/probes/resid_drum_codebook.py`, `resid_percussion.py`).

- **Detect (per tune):** group RESID notes by signature recurring ≥`MINREP` (=3): **ABS** `(fn,ctrl)`
  series (fixed-freq drum/effect) AND **REL** `(offset,ctrl)` series (transposable pitched gesture;
  backref carries a base note). REL adds ~0.4% over ABS.
- **Full footprint (consistency-attribution).** A stamp is ALL the voice's writes in its span
  (freq/ctrl/PW/ADSR) PLUS drum-scoped filter. Rule: **a register's in-span writes that are IDENTICAL
  across every occurrence → fold into the stamp; varying → leave on the global/continuous channel.**
  Measured: 51% of stamps fold PW/ADSR; 24% have drum-scoped filter (global SID filter ⇒ must attribute,
  not assume). (`audit/probes/resid_drum_footprint.py`.)
- **Encode — INLINE, REDEFINABLE (not a preamble):** define a stamp the first time inline; reuses are
  backrefs; a changed write-series emits a NEW inline def (streaming dictionary; ~1/3 of drums redefine
  — measured). Tokens: `STAMP_DEF id char [folded-footprint as regular per-frame atoms] STAMP_END`
  (Unigram-clusterable layout — common sub-series shared); `STAMP_REF char id voice [xpose]` per hit
  (voice-agnostic: stamp in voice-relative subregs, voice on the ref; onset = stream position; gridded
  ⇒ low entropy). `char` ∈ `{KICK,TOM,SNARE,HAT,CYMBAL,NOISE_FX,PITCH_FX,…}` from waveform/freq/sweep
  — a GLOBAL transfer vocabulary (the model predicts `char`; bytes come from the in-context def).
- **Decode:** live id→stamp table (a later `STAMP_DEF id` rebinds); replay the stamp's writes on the
  ref's voice, +xpose on pitched freq, restore the pre-stamp **global filter** state after a drum-scoped
  filter span (save/restore — open item).
- **Codebook economics:** median ~11 defs/tune (max 57); bounded. Reuse/repair the refuted
  `motif_pass` mining machinery (same SHAPE, different GOAL = lossless RESID-drain).

## 5. Mechanism: melodic PATCH preamble (timbre)

(`design/patch_preamble_encoding.md`.) A melodic note = PITCH (skeleton+ornament) × TIMBRE (a reusable
instrument **patch**). This does NOT reduce RESID (melodic notes are already SKEL) — it factors the
non-pitch footprint out of the per-note stream (separability/no-multiplexing) and is the twin of §4.

- **Patch identity = (waveform-CYCLE, ADSR).** The waveform-cycle is the wavetable's minimal repeating
  unit (collapse duration-length — same op as ARP — after stripping the onset transient): a different
  cycle = a different reusable instrument. **ADSR is held on 85% of note-transitions** (the stable
  core). **PW is NOT patch identity** — a continuous, not-note-aligned sweep → its own channel.
  (`audit/probes/resid_patch_codebook.py`, `resid_patch_mutation.py`.) Validated: top-8 patches cover
  ~91% of a tune's notes; ~22 patches/tune.
- **Encode:** `PATCH_DEF id (waveform-cycle, ADSR)` inline+redefinable; `PATCH_SET id` is an AMBIENT
  program-change (emit only on instrument change — rare/low-entropy); melodic notes then carry only
  skeleton+ornament. Lay out as Unigram-clusterable atoms (similar cycle/ADSR share sub-tokens).
- **MUTATION primitive (avoid redefining for a tweak):** `PATCH_MUT id field∈{RELEASE,SUSTAIN,DECAY,
  ATTACK} val=v` keeps the id. Measured: of ADSR changes, **66% are 1–2 nibble**; fields
  RELEASE≈SUSTAIN>DECAY≫ATTACK (attack defines the instrument → redefine, not mutate). Rule: ≤2 nibbles
  & attack-stable → `PATCH_MUT`; ≥3 or attack moves → `PATCH_DEF`.

## 6. Mechanisms: SLIDE/SWEEP + held-gate re-segmentation (the tail)

- **Freq-domain SLIDE/SWEEP:** the wide ramps are linear in **raw freq** (skydive = freqhi decrement;
  −Δ/frame constant) → accelerating in semitones, so the semitone-uniform SLIDE misses them. Add a
  **freq-domain slide** (constant raw-freq delta) AND **strip the onset transient** (noise/HR first
  frame, e.g. Danko `[46,-12,-14,-17,-21,…]`) before fitting. Onset-strip uses the same control-aware
  `_is_pitched_frame` as the skeleton.
- **Held-gate re-segmentation:** giant held-gate notes concatenate a held note + a separate gesture
  (e.g. Neptune `[0,0,0,0,0,49,…noise…]`). Extend the landed `_resegment_levelchange` to split at the
  first post-plateau pitched-level change (noise/test transparent). These dominate the RESID *frame*
  mass (vs note count).

## 7. The residual (~1.5%) — it is UNMODELLED ENGINES, not irreducible noise

**Correction (do not call this an "irreducible floor").** `sidid` on the worst-residue composers shows
the tail is concentrated in **specific engines**, each of which deterministically *encoded* these notes
— so a compact abstraction EXISTS; trace it to the driver (the program's thesis). Two groups:

**(A) DOCUMENTED engines → close now from `sid_driver_ornament_reference.md` (no new research):**
- **MoN / FutureComposer** (Dalton, Moppe, Tron_Olsson_Mikael): the residue IS the "novel-mechanism
  frontier" I documented but never implemented — **target+duration glide** (computes
  `(target−cur)/dur`, lands exact → the accelerating sweeps like `[46,-12,-14,-17,-21,…]`), **noise-tik
  onset**, **sine vibrato (delay+length)**, **Tonesweep**. Implement these primitives.
- **JCH NewPlayer** (Danko): `[58,10,14,23,14,23,…]` is a **wavetable arp `[14,23]` on the NOISE
  waveform** (after a `[58,10]` onset). The engine's wavetable offset-cycle is **waveform-agnostic** —
  the ARP detector wrongly gates on `is_pitched` and skips noise frames. **Fix: make ARP/wavetable-cycle
  noise-INCLUSIVE** (detect the offset cycle over ALL frames, RLE-collapse holds, strip the onset
  transient). This alone closes a large share of the tail.

**(B) Newly-documented engines (in `sid_driver_ornament_reference.md`):**
- **SoundMonitor** (Danko, Gilmore) — a **freq-domain sweep engine**: ornament = a constant-Δfreq
  decrement, **looped** for pitched "arps" (e.g. −624/frame, period 15), one-shot **on noise** for
  drums. Fix = the freq-domain SWEEP primitive `(start, Δfreq, length, loop_period)`, waveform-agnostic
  (§6 extended with a loop period). Every semitone-domain primitive misses it (1986, pre-tracker).
- **System6581** (Moppe) — note-relative **ARP** chord-cycles (period ~3, e.g. `[5,2,0]`) **with a
  per-cycle gate-off + noise-tik accent interleaved**; leaks only because pitched-only period detection
  breaks on the accent frames. Fix = control-aware noise-inclusive ARP (carry the accent as part of the
  cycle). No new primitive.

**(B′) STILL UNDOCUMENTED → next trace-to-driver targets:** SoedeSoft, Music_Assembler, AMP, DMC,
GMC, Adam_Gilmore's custom engine, Groovy_Bits, RoMuzak, Electrosound, SidTracker64. Source each
(CSDb / codebase.c64.org / disassemblies / register-output RE); expect the same universal primitives.

So the path to RESID=0 is **continue tracing engines**, not accept a floor. The only place
losslessness can't reach (a one-off a player genuinely improvised) goes to the audition-gated lossy
tier (P8) with an explicit penalty — but that is the LAST resort, after (A) and (B), not the default.

## 7b. Auto-RE profiler — a reusable coverage instrument (clean up + adopt)

`audit/probes/resid_engine_profile.py` automates the trace-to-driver loop: it `sidid`-labels each tune
and fits every RESID note to a parametric model library (`SWEEP_loop/SWEEP_once/ARP/ARP_accent/
PERC_sweep/PERC_other/UNRESOLVED`), aggregating **per engine** → an auto-generated technique profile.
One run profiled 11 engines and showed **every one reduces to the same primitives** (SWEEP/ARP/
ARP_accent/PERC) — the collapse hypothesis, mechanised. **The fitters ARE the driver-abstraction
recognisers; a persistently-UNRESOLVED engine = an un-RE'd technique = add a fitter (and a primitive).**

Adopt it as the **acceptance instrument** for this whole effort: after each new primitive, run it and
drive per-engine UNRESOLVED→0; the corpus is "covered" when no engine has a significant UNRESOLVED
residue that isn't a stamp. Cleanup tasks (the work for the tokens agent — even if just polishing):
1. **Profile only the NON-stamp residue** — cross-reference the §4 stamp recurrence first, so recurring
   drums (lossless via the codebook) don't read as UNRESOLVED (the dominant cause of the high rates).
2. **Segment-then-fit** — fit the held-gate-segmented constituents (§6), not the whole concatenated
   note (the other UNRESOLVED cause: arp+drum+onset in one note).
3. **Emit the recognised parameters as the primitive's params** (e.g. `SWEEP_loop d=-624,p=15` → the
   SoundMonitor arp token) — i.e. wire the recogniser output into the encoder, closing the loop.
4. Tidy into a tokens-side dev tool (it currently lives in xpt and imports via `sys.path`); keep the
   inert `_resid_diag` sink as its interface.

## 8. Gates (non-negotiable, every step)

### 8.0 Testing protocol — RUN EVERYTHING IN DOCKER; full CI run before any PR (MANDATORY)

**Do not run tests on the host. Run the FULL suite inside the docker image, and do a complete
CI-equivalent run at the end before pushing a PR — exactly what CI does.** The host lacks the pinned
toolchain/deps; a "green" host run is not a real gate and host/CI drift will fail the PR.

- **Per-change loop (fast):** run the affected tests in the baked image
  `docker run --rm --network host -v /scratch:/scratch -w <preframr-tokens> anarkiwi/preframr-tokens-test
  python3 -m pytest tests/ -k <area> -q`. `--network host` + the gitignored `.env` `PIP_OPTS` (proxpi
  mirror) are required for the image to reach packages (the bridge can't); the baked image avoids a
  reinstall.
- **Before EVERY PR (full, CI-equivalent):** rebuild the image and run the WHOLE gate the way
  `.github/workflows/docker.yml` does — `docker build -f Dockerfile .` (which runs `run_tests.sh`:
  black, pytest **all**, pylint-curated, pyright, coverage ≥77) — and confirm it is fully green. A
  PR may be opened ONLY after this full in-docker run passes. Do not rely on `-k`-filtered subsets or
  host runs for the final gate.
- **Cross-repo:** the preframr-side bridge test runs in preframr CI, not tokens CI — see §9; a tokens
  op/flag change is not "done" until tokens is published AND preframr CI is green.

### 8.1 Correctness gates

- **Deterministic suite green** (`run_tests.sh`: black, pytest, pylint-curated, pyright, coverage ≥77;
  `tests/test_lint.py` rejects narrative `#` comments + >5-line docstrings — keep rationale in design
  docs, not inline).
- **Byte-exact round-trip** — the accepted cover must reproduce the raw register log (the per-frame
  oracle); prove any discarded write inaudible on the SID emulator (pyresidfp), never assume
  (`preframr-audio` reference suites; only TEST-bit-frame freq is freely discardable).
- **Unigram interaction** — emit defs/refs as regular small atoms so the downstream Unigram tokenizer
  sub-tokenizes/clusters them (the opposite of `motif_pass`, which fought Unigram). Re-check vocab
  merges cluster similar stamps/patches.
- **12-SID WAV audition** — required before flipping any tokenizer default or admitting a lossy claim.

## 9. Cross-repo & release (see memory `cross-repo-release-ordering`, `docker-build-cache`)

A tokens flag/op change is a 3-repo change: publish tokens to PyPI (push `vX` tag → OIDC) BEFORE
preframr CI floors the new version; preframr's args bridge mirrors new flags. Keep every mechanism
behind a gate flag, default OFF. Local gate loop: baked `anarkiwi/preframr-tokens-test` image,
`--network host` + `.env PIP_OPTS` (proxpi).

## 10. Suggested build order

0. **Start clean (§1):** merge the foundation PR (`feat/transient-tolerance`) into `main` via the
   §8.0 full-docker CI gate, then branch from `main`. Every step below is its own branch → §8.0 gate → PR.
1. Speculative pipeline framework (§2 a–b) — byte-identical refactor, delete `_df_sink`.
2. STAMP codebook (§4) — biggest RESID win; ABS then REL; footprint consistency-attribution; inline
   redefinable + char. Re-run `resid_final_accounting.py`, expect STAMP ≈98%.
3. Patch preamble + mutations (§5) — token-budget/separability (not RESID); validate Unigram clusters.
4. held-ARP irregular-duration (§3), freq-SLIDE/SWEEP + onset-strip, held-gate re-seg (§6).
5. Arbiter competing claims + per-tune mode selection (§2 c–e); re-trace residual (§7); decide the
   audition-gated lossy tier for the last ~1%.

## References

- Designs: `speculative_encoding_pipeline.md`, `percussion_stamp_encoding.md`,
  `patch_preamble_encoding.md`, `resid_archetype_program.md`, `encoding_principles.md`,
  `sid_driver_ornament_reference.md`.
- Probes (xpt `audit/probes/`): `resid_final_accounting`, `resid_trace`, `resid_drum_codebook`,
  `resid_percussion`, `resid_drum_footprint`, `resid_patch_codebook`, `resid_patch_mutation`,
  `resid_archetype_survey`.
- Memory: `control-aware-encoding`, `speculative-encoding-pipeline`, `universal-driver-already-exists`,
  `collapse-driver-abstractions`, `cross-repo-release-ordering`, `test-through-real-parse`.
