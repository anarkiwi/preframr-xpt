# Subword tokenization (BPE / Unigram) on the BACC token-id stream — grounded design analysis

**Status: ANALYSIS — vanilla BPE/Unigram REFUTED on the BACC stream (2026-06-20).** Vanilla subwording
welds 63–65% of token occurrences across field boundaries (note↔instr / instr↔dur / whole rows) and
collapses induction-copy 0.886 → 0.114 (a learnability NO-GO, same outcome as the refuted old-codec BPE
but via welding, not density-fitting). Melodic-sequence capture does NOT materialize (the inline row-LZ
already factored phrases; < 1% of subword mass is multi-note). Only HARD field-boundary-segmented BPE is
safe (0% welding, ~1.4× compression, copy-preserving) and is help-neutral at best. **Do not apply vanilla
BPE/Unigram; do not pre-collapse to a whole-field vocab (welding → 86%).** Refuted stub:
`data/refuted/bpe_unigram_subword.md`. (Implication for the absolute-grid note encoding: BPE will NOT
recover the intra-phrase compression that the absolute canonical grid gives up vs relative intervals —
the hoped-for melodic-capture win is absent.)

**Date:** 2026-06-20 · **Scope:** does a subword tokenizer (BPE, Unigram) *on top of* the
current **sparse** BACC token-id stream (a) capture melodic sequences as units (the win), and
(b) avoid welding semantically distinct fields (the risk)?  Every number below is measured on
real recovered streams. Scripts/logs: `/scratch/tmp/sidemu/bpe_*.py`.

This is **not** the refuted framing. The refuted item (`AGENTS.md`, "Refuted") is *frame/event-codec
density compression (BPE / boundary dictionary) as the context lever* — BPE signal-fitting a **dense
trace**. The premise here is the opposite: the BACC step/tracker/generator codec is already
**< 1 token/frame** and already factors repeats with an inline backward-LZ (`REPEAT`). The question
is whether residual subword structure helps the **next-token map**, judged by learnability (the
north-star in `learnability_token_ordering_theory.md`), not by compression alone.

---

## 0. Corpus and the field-id instrument

> **Stale baseline (atom counts only).** The absolute per-tune atom / tok-frame numbers in the table
> below predate the latest BACC sparsification (e.g. it lists Monty 5622 atoms / 0.320 tok/frame; current
> is **1,313 / 0.075**). Treat the absolute counts as a stale baseline — only the **relative** welding /
> learnability conclusions carry over (those are ratios, not affected by the later sparsification). Do
> not recompute the exact numbers from this table.

Recovered BACC id streams (white-box `.sid` + dump → `recover_program` → `program_to_ids`), each
paired with a **per-token FIELD-ID annotation** emitted by a re-serializer that mirrors
`serialize.py` / `gt_serialize.py` emit-for-emit and **asserts** id-equality with the upstream
serializer (`bpe_field_serialize.py`), so the role of every atom is exact.

| tune | driver | atoms | frames | tok/frame |
|---|---|---|---|---|
| Monty | hubbard_monty | 5622 | 17544 | 0.320 |
| 5_Title_Tunes | hubbard_5tt | 1654 | 2049 | 0.807 |
| Grid_Runner | goattracker | 4132 | 15681 | 0.264 |
| A_Mind_Is_Born | lft | 496 | 8191 | 0.061 |
| gt_consultant | goattracker | 1401 | 3000 | 0.467 |
| gt_sanction | goattracker | 2193 | 3005 | 0.730 |
| gt_hyperspace | goattracker | 5487 | 3000 | 1.829 |

Field roles tracked: `DT, NOTE, INSTR, INSTRDEF, DUR, PORTA, EFFECT, DATA, KIND, MARK (REPEAT),
COPY (LZ operands), OLIST, TABLE, NROWS, HEADER, IMAGE`. The **field-study corpus** excludes
`A_Mind_Is_Born` because lft is a *pure generator* (the whole tune is a ~254-byte program image,
one `IMAGE` role, no score, no musical fields) — it is the sparsity floor (0.061 tok/frame), and
there is nothing for a subword tokenizer to do there at all. Field-study corpus = **20,489 atoms**,
role mass: NOTE 15.1%, INSTR 11.2%, COPY 10.9%, DATA 7.8%, KIND 7.2%, EFFECT 7.2%, HEADER 6.4%,
TABLE 5.8%, DT 5.2%, DUR 5.1%, MARK 4.6%, OLIST 4.5%, PORTA 4.1%, INSTRDEF 3.5%, NROWS 1.5%.

A learned subword = a contiguous span of atoms. It is **within-field** iff all covered atoms share
one role; **cross-field (= WELDING)** iff it covers ≥2 distinct roles.

---

## 1. What BPE vs Unigram each do to this stream + the compression curve

Compression over the field corpus (raw atoms → post-subword tokens). My minimal-BPE numbers were
cross-checked against **HF `tokenizers` BPE** and **SentencePiece Unigram** on the identical stream
(`bpe_hf_unigram.py`) — they agree.

| scheme | vocab | ratio (atoms/token) |
|---|---|---|
| minimal BPE (vanilla) | 256 / 512 / 1024 / 4096 | 2.16× / 2.58× / **3.19×** / 4.27× |
| HF BPE (vanilla) | 512 / 1024 / 4096 | 2.58× / **3.20×** / 8.03× |
| SentencePiece **Unigram** | 512 / 1024 / 4096 | 2.54× / **3.43×** / 4.64× (real vocab capped at 2214) |
| boundary-constrained BPE | 256 / 512 / 1024 | 1.69× / 1.86× / **2.01×** |
| **hard field-segmented BPE** (merges cannot cross *any* field) | 1024 | **1.41×** |

Per-tune (vanilla minimal BPE, vocab 1024): Monty 3.03×, 5_Title_Tunes 3.26×, Grid_Runner 3.55×,
gt_consultant 2.89×, gt_sanction 2.95×, gt_hyperspace 3.29× — uniform ~3×.

- **BPE** (greedy frequency merges) and **Unigram** (likelihood-pruned vocabulary) reach **similar
  compression** (~3.2–3.4× at vocab 1024). Unigram favors slightly **longer** high-frequency pieces
  (it merges whole *rows*), so its per-piece welding is a touch higher; it also could not fill a
  4096 vocab (only 2214 useful pieces) — direct evidence the sparse stream has **limited subword
  headroom**.
- The headline 3× compression is **mostly NOT musical**. It comes from two non-musical sources
  (Exp 3 below): (i) the one-shot `HEADER` prologue (the 256-byte static image / boot / seed),
  and (ii) cross-field welds. The BACC codec already removed phrase-level redundancy with its
  inline backward-LZ, so little melodic redundancy is left for the subword layer to find.

---

## 2. WELDING — the central risk, measured

Cross-field rate per scheme (minimal BPE; `bpe_experiment.py`), confirmed by HF/Unigram:

| scheme | vocab | merges | cross-field merges | **token-mass cross%** | atom-mass cross% |
|---|---|---|---|---|---|
| VANILLA | 256 | 223 | 57.8% | 29.9% | 47.6% |
| VANILLA | 512 | 479 | 61.6% | 38.1% | 56.8% |
| VANILLA | 1024 | 991 | 59.9% | **45.9%** | **64.6%** |
| VANILLA | 4096 | 1803 | 67.1% | 52.6% | 72.0% |
| CONSTRAINED | 1024 | 756 | 9.1% | 2.8% | 4.6% |
| **HARDSEG** (field-pretokenized) | 1024 | 726 | **0.0%** | 0.0% | 0.0% |

HF BPE vocab 1024: **62.5%** of multi-atom token *occurrences* are cross-field, **74.8%** of token
mass. Unigram vocab 1024: 65.1% / 79.2%. All three tokenizers weld the **majority** of their mass.

**Concrete welds (the user's exact nightmare).** Top cross-field merges by mass:

- `NOTE+INSTR` — the last digit of a note interval fused to the first digit of the next
  instrument_ref. (minimal BPE x40; HF BPE x191 occurrences.)
- `NOTE+NOTE+INSTR` — a note token welded into the following instrument. (HF BPE x286 — the single
  most common weld.)
- `INSTR+DUR+PORTA+DT` — instrument→duration→porta→next-row-delta all in one token (HF x104).
- `INSTR+EFFECT+DATA+KIND` (GoatTracker row tail welded into the next row's kind, HF x105).
- `NOTE+INSTR+EFFECT+DATA+KIND` and `KIND+NOTE+NOTE+INSTR+EFFECT+DATA` — a whole GT *row* (or a
  row spanning into the next) as one token (HF x99; Unigram x54).
- `DUR+PORTA`, `DUR+PORTA+MARK`, `MARK+COPY` — fusing literal-field tails into the LZ control op.

**Why welding hurts learnability (not just aesthetics).** A weld like `NOTE+INSTR` makes the
instrument choice un-selectable independently of the note: to emit "this note with a *different*
instrument" the model must reach a *different leaf token*, and the (note, instr) pair becomes a
joint symbol whose marginal the model must memorize rather than compose. That is precisely the
compositional-generalization killer the task names — the model can no longer recombine the
note-axis and the instrument-axis freely. The triage confirms it costs the model (§5).

---

## 3. Melodic-sequence capture — the hoped-for win is largely absent

Searched every learned subword for **pure multi-NOTE runs** (≥2 consecutive `NOTE`-role atoms — a
motif/phrase as a unit) (`bpe_melody.py`):

- Vanilla BPE vocab 1024: only **21 distinct** pure-NOTE subwords, total **mass 276** atoms — i.e.
  **~1.4%** of the 20,489-atom corpus. Hard-segmented: 23 subwords, mass 295.
- And the top ones — `(8,18)`, `(14,17)`, `(12,17)` — are **2-atom tokens where the note field is
  itself 2 LEB digits**: they are *single notes with a 2-digit interval*, not multi-note melodies.

**Why the win is missing:** the BACC codec already runs an inline **backward LZ over whole rows**
(`REPEAT`). Repeated melodic phrases are *already* factored out as `MARK`+`COPY` ops **before** the
subword layer sees the stream — so the residual NOTE region has little phrase redundancy left. BPE's
gain therefore comes from elsewhere. Where the gain actually comes from (atoms saved, vocab 1024):

| | vanilla BPE | hard-segmented BPE |
|---|---|---|
| cross-field (welded) saving | **73.2%** | 6.4%* |
| within HEADER (one-shot prologue) | 18.4% | 55.7% |
| within TABLE / INSTRDEF / OLIST / COPY | ~7.8% | ~33.9% |
| within **NOTE** (the actual melodic win) | **0.7%** | 2.7% |

(*residual non-zero only because the per-occurrence boundary check has a seam edge case; the
**hard-segmented** variant that pretokenizes at every field boundary is provably 0% cross-field —
"cross-field merges = 0 by construction".)

So in vanilla BPE, **73% of the compression is welding** and **<1% is melodic**. The "musical
motif as a unit" win the user hoped for does **not** materialize on this representation — the LZ
already captured it, and the high-value remaining merges are field conflations.

The higher-level-vocab experiment (Exp 4iii) makes this starkest: collapse each field to one
whole-value token, then BPE — now **85.8%** of merges are cross-field, because within a field there
is only one token left and the *only* adjacency to exploit is across fields (note→instr→dur→…).
A higher-level vocabulary **amplifies** welding rather than fixing it.

---

## 4. Field-aware alternatives — measured

(i) **Vanilla BPE / Unigram** — 3.2–3.4× compression, **63–65% of token occurrences welded,
75–79% of mass welded.** Maximum compression, maximum field conflation. **Don't ship.**

(ii) **Boundary-constrained** (merges may not cross a field-role boundary) — implemented two ways:
   - *soft* constraint (block a merge whose seam straddles a role): 2.0× compression, 9% cross
     (residual seam artifacts).
   - *hard field-segmentation* (pretokenize at every field boundary; a subword can never span two
     fields): **1.41× compression, 0% welding by construction.** This keeps the within-field and
     within-phrase merges (HEADER prologue, TABLE/INSTRDEF runs, multi-digit field values, the few
     real NOTE pairs) and drops every weld.

(iii) **Higher-level whole-field-value vocab, then BPE** — 2.0× over the LEB stream, but **86% of
   merges cross-field.** Rejected: it removes the within-field digit structure that was the only
   clean BPE win and leaves only cross-field adjacency to exploit.

**Recommendation:**
- **DO** field-segmented (boundary-constrained) subwording *if* you want a subword layer at all —
  it is the only scheme that captures the genuine within-field digit/value regularity while keeping
  welding at zero. But note its compression is only **1.41×**, and most of that is the one-shot
  HEADER/TABLE prologue, not score.
- **DON'T** apply vanilla BPE or Unigram to this stream. The 3× number is a mirage: ~¾ of it is
  field welding that the learnability metrics show is actively harmful, and <1% is the melodic
  capture you wanted.
- **DON'T** pre-collapse to a higher-level whole-field vocabulary and then BPE — it maximizes welding.

---

## 5. Learnability verdict (training-free triage)

Ran the project triage (`audit/learnability_triage.summarize`) on RAW vs the two subword schemes,
per-frame `h_k` for cross-encoding comparability (`bpe_learnability.py`):

| stream | tok/frame | alphabet | induction-copy | first-occ | MI(lag1) | MI tail (lag6–12) |
|---|---|---|---|---|---|---|
| **RAW-LEB** | 0.463 | 34 | **0.886** | 0.010 | 0.482 | ~0.11–0.22 (decays) |
| VANILLA-1024 | 0.145 | 817 | **0.114** | 0.293 | 5.512 | **~5.35 (flat, fat tail)** |
| HARDSEG-1024 | 0.327 | 535 | **0.741** | 0.089 | 1.693 | ~1.3 (tame) |

Reading per the theory (low per-frame `h_k` + early plateau + fast MI decay + **high induction-copy**
⇒ learnable; fat MI tail + low copy ⇒ predicted collapse):

- **Vanilla BPE = HURTS, decisively.** Induction-copy **collapses 0.886 → 0.114** (the corpus's
  dominant induction-head-able structure is destroyed), first-occurrence novelty jumps 0.01 → 0.29
  (every merged token is nearly unique — the 817-alphabet on 6.4k tokens is severely undersampled,
  which is *also* why its per-token `h_k` looks deceptively low), and MI gains a **flat ~5.5-bit
  tail across all lags** — the signature of a long-range dependency SGD will shortcut. This is the
  same failure mode that refuted BPE on the old codec, reproduced here via a *different* mechanism
  (welding rather than density-fitting). Verdict: **NO-GO.**
- **Hard field-segmented BPE = roughly NEUTRAL, mildly negative on copy.** It preserves most of the
  induction structure (copy 0.886 → **0.741**) and keeps the MI tail tame (~1.3 vs 5.5). It is the
  only subword scheme that does not wreck the next-token map. But it *does* lower copy somewhat and
  buys only 1.41× — so it is at best a small, safe compression, not a learnability *win*.

**Overall verdict: subword tokenization on the BACC stream is HELP-NEUTRAL at best (only the
field-segmented variant), and HARMFUL in its natural (vanilla) form.** The hoped-for melodic-capture
win is pre-empted by the codec's own row-LZ; the natural BPE/Unigram gain is welding, which the
triage flags as a copy-collapse + fat-MI-tail regression. The representation is already near the
learnable regime the theory wants (RAW induction-copy 0.886) — and vanilla subwording moves it the
wrong way.

---

## 6. Concrete next experiment (only if pursued)

The analysis does not motivate a training run for *compression*. The one **defensible** follow-up,
and only if a subword layer is wanted for context-budget reasons on long multi-subtune programs:

> **Field-segmented BPE, vocab ≈ 512, applied per field-role**, then a single canonical
> confirmatory run comparing RAW vs field-segmented on the standard 8192-token / window-mode setup.
> Pre-register the triage prediction: field-segmented should hold induction-copy ≥ ~0.74 and keep
> the MI tail < ~2 bits (measured here), so it should be *non-regressive*; the open question a run
> would settle is whether the 1.4× context saving on long programs is worth the copy drop from
> 0.886 → 0.741.

Do **not** spend a run on vanilla BPE/Unigram or higher-level-vocab BPE — both are pre-refuted by
the welding + triage measurements above (copy collapse to 0.11, MI tail to 5.5 bits).

**Bottom line for the two stated concerns:**
(a) *melodic-sequence capture* — **does not happen** on this stream (the row-LZ already did it;
<1% of subword mass is multi-note). (b) *not welding distinct fields* — **vanilla BPE/Unigram weld
63–65% of token occurrences (≈75–79% of mass), exactly across note↔instr / instr↔dur / row
boundaries**, and the triage confirms this kills the induction structure. The only acceptable
subword scheme is **field-boundary-constrained** (0% welding, ~1.4× compression, copy-preserving).
