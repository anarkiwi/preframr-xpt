# Encoding-density frontier — BPE refuted, atoms-only is correct, density is exhausted

**Status: DECISION (2026-06-12).** Settles the "compress the encoding to fit tunes in the
window" question with data from the canonical learnability run + a corpus-wide encoding survey.
Verdict: **unigram/BPE is refuted as the content lever; the atoms-only event encoding is the
content-correct representation; value-encoding density is already at its frontier (parametric
ramps + per-voice pitch are shipped); the only open density lever is head-amortization (~10–15%),
which does not change the context-window picture.** Context length is a `seq_len`/windowing
problem, not an encoding problem. Melody is intrinsically high-entropy next-token.

## 1. BPE harms content generalization (~6–11×) — `data/refuted/unigram_bpe_content_generalization.md`

Canonical `generalize` 14M body, atoms-only (tkvocab=0) vs unigram BPE (tkvocab=2048), same
corpus/holdouts, content-tier accuracy at matched maturity (~epoch 100):

| eval subset | BPE-2048 | atoms-only |
|---|---|---|
| eval_a | 0.049 | 0.479 |
| eval_b_daglish | 0.088 | 0.559 |
| eval_b_follin | 0.039 | 0.416 |

**Mechanism (localized):** merged BPE tokens are **~1% predictable** (base atoms 4–8%). BPE is a
data-driven byte-merge blind to the event grammar; it welds content atoms into multi-atom merges
*across* event boundaries, which are unpredictable, and the welding degrades the surviving base
atoms too. All-tier `val_acc` is confounded across tokenizations (bigger vocab ⇒ higher per-token
entropy) — content-tier is the verdict. **The "BPE dial is THE context lever" framing is refuted.**

## 2. Per-KIND learnability map (atoms-only model, context-aware audit)

Content-digit accuracy on held-out composers, bucketed by the event KIND tracked from context:

| KIND | acc (a / daglish / follin) | % of stream | read |
|---|---|---|---|
| G_STEP (global filter regs 21–24) | 0.72 / 0.77 / 0.70 | 16–27% | learnable, huge |
| PW_RAMP (pulse-width sweep) | 0.56 / 0.51 / 0.49 | 11–23% | learnable, huge |
| G_RAMP, EVT74/72 | 0.29–0.78 | small–mid | mostly learnable |
| FD_STEP/FD_RAMP (freq residual) | 0.37–0.42 | ~15% | moderate |
| **NI_STEP (note-index = melody)** | **0.18 / 0.30 / 0.31** | 9–14% | **intrinsically hard** |
| NI_RAMP (portamento) | 0.38–0.40 | 6–10% | hard |

**The atom model is good at timbre/envelope, weak at melody.** The low NI_STEP accuracy is **not an
encoding gap** — `universal_multiresolution_pitch.md` already established that absolute onset pitch
is high-entropy ≈0 next-token; **score onsets by audition/distribution, not argmax.** Intervals fix
within-melody transfer; the absolute anchor is creative content.

## 3. Value-encoding density is already at the frontier (shipped)

The two density ideas one reaches for are already implemented and MDL-optimal:

- **Parametric ramps** — `stream.py:_series_events` (≈line 265) runs `cover(series, cost_model)` to
  segment every per-(voice,kind) value series into HOLD/POLY/PERIOD gestures; ramps emit
  `[kind][SHAPE][length][degree][v0][deltas]` (66–73% vs per-step). Shipped tokens 0.16/0.17
  (`unified_oscillation_primitive_design.md`), incl. gap-tolerant oscillation recognition (43.9%
  FREQ coverage) + lossless delta payload (30% FREQ-atom reduction).
- **Per-voice note-table pitch** — shipped tokens 0.47.0 (`universal_multiresolution_pitch.md`):
  universal semitone NI_* lane (Δnote intervals) + per-voice recovered NOTE_TABLE + per-voice
  TUNING + FD_* modulation residual. 83% of voiced frames have residual exactly 0.

Corpus survey (120 tunes, the in-place `.atoms.zst`): **avg 6.42 atoms/event, ~49k atoms/tune.**
Composition: ~48% content value-digits (irreducible musical content), **25.9% `[VOICE][KIND][reg]`
head markers** (the amortization ceiling), remainder gesture params (SHAPE/length/degree) + KEYFRAME
conditioning. **67% of events are 2–4 atoms** (modal event = 3: kind + ~2 payload).

## 4. The only open density lever: head-amortization (~10–15%, optional)

Each event pays a `[KIND]` atom (and `[reg]` for globals); the `[VOICE]` atom is already amortized
per frame-group. Since 67% of events are 2–4 atoms, heads are head-heavy. Ceiling = 25.9% of the
stream; realistic recovery ~10–15% (you cannot eliminate all event identification). Candidate
mechanisms (deterministic, byte-exact-preserving, single-token-per-unit — *not* data-driven merges):
combined `(voice,kind)` or `(kind,reg)` atoms for the common cases; context-predicted kind elision
where the grammar makes it unambiguous. **Worth doing as polish, but it does not move the needle on
context** — 49k → ~42k atoms/tune is still ≫ 8192.

## 5. Context length is a `seq_len`/windowing problem, not an encoding problem

There is no large remaining density win to fit tunes in the window — BPE was the only thing
achieving 2.6× compression, and it does so by destroying content learnability (§1). Therefore:

- **Use tkvocab=0 (atoms-only).** It is the content-correct encoding; it beats BPE ~10× on content.
  Keep `tkvocab` as a dial but not as the strategy.
- **For more tune-per-window, scale `seq_len`** (8192 → 16384; the 14M body fits 24 GB; costs a
  dataset re-cut + wallclock) and/or cut **musically-aligned KEYFRAME windows** at pattern/loop
  boundaries (dataset-side, no alphabet change). These are the real context levers.
- **Evaluate melody by audition/distribution, not next-token argmax** — it is high-entropy by nature.

## 6. For the next agent — what to do, what not to do

- **Do not** re-attempt BPE/unigram for content, a "denser alphabet" to fit the window, parametric
  ramps, or a per-voice pitch table — all refuted or already shipped.
- **Optional polish:** implement head-amortization in `events/stream.py` (≤~13% length; keep
  `encode(verify=True)` byte-exact; bump `EVENT_FORMAT_VERSION` + `ATOM_CACHE_VERSION`).
- **Context arc:** run `seq_len` 16384 + musically-aligned windows on the atoms-only encoding; this
  is the live lever for whole-tune structure in the training distribution.

**Artifacts:** `/scratch/tmp/v4_audit*.json` (BPE content-tier), per-KIND + token-budget scripts in
the session log, `data/refuted/unigram_bpe_content_generalization.md`.
