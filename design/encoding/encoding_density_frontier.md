# Encoding-density frontier — atoms-only is the default, BPE refuted as the context lever (true magnitude ~1.4× bits/atom), radix + head-amortization the open density levers

**Status: DECISION (2026-06-12; the §7 de-confounding audit RAN — verdict revised, magnitude
resolved).** *Codec note (2026-06-13): the live reference baseline was re-anchored to the **v2 codec**
(tokens 0.51.0, `EVENT_FORMAT_VERSION=2`) — current numbers content-tier 0.505/0.552/0.485,
bits/canonical-atom 1.998/2.058/2.272 (`data/audit/v2_atoms_baseline_audit.json`), ≈ parity with v1.
The §1/§7 numbers below are the **v1** record they were computed on; conclusions are unchanged.*
Settles the "compress the encoding to fit tunes in the window" question with the
canonical learnability run, a corpus encoding survey, and the §7 audit (results in
`data/audit/deconfound_summary.md`). Verdict: **the atoms-only event encoding (tkvocab=0) is the
shipped default and the content-correct representation; *unconstrained* unigram/BPE is refuted as the
content/context lever — but the 6–11× argmax harm was CONFOUND-DOMINATED (all three §1a confounds
confirmed real). The true model-quality gap is ~1.4× in bits/canonical-atom at ep174 (§7A), ~2–4× in
position-matched argmax (§7B), and SHRINKS to ~1.2–1.3× at the matched-steps endpoint (§7C, ep299 —
still descending) — modest, not catastrophic.** The boundary-crossing-welding mechanism motivated an
**event-boundary-respecting dictionary**, which shipped (tokens 0.51.0) and was **triage-resolved
NOT-adopted** (§9, 2026-06-13: compression caps ~1.7× < 1.8× ADOPT bar, merge table ~89%
deterministic-pack-shaped → the packs win). Value-encoding density:
**digit radix is a live ~12%-of-stream polish lever** (§3/§7E, not dead) scoped to multi-digit varint
lanes; head-amortization recoverable share is **17.4%** (§4/§7F). Neither is a context lever: tunes
are ~50k atoms/median (§7F, NO dataset truncation — `BlockMapper` tiles every atom), so context stays
a `seq_len`/windowing/chaining problem (§5). Melody is high-entropy next-token at this regime (§2/§7D).

## 1. BPE harms content modestly (~1.4× bits/atom) — the 6–11× argmax headline was confound-dominated

Canonical `generalize` 14M body (8L-d320-im896), atoms-only (tkvocab=0, ckpt ep99) vs unigram
BPE-as-dictionary (tkvocab=2048, ~2.73× stream compression, ~98% live vocab; best ckpt ep174),
same corpus/holdouts. Registry entry: `data/refuted/unigram_bpe_content_generalization.md`;
audit artifacts in `data/audit/` (`deconfound_summary.md`).

**As originally measured (per-token content-tier argmax) — now known to be confounded:**

| eval subset | BPE argmax | atoms-only argmax |
|---|---|---|
| eval_a | 0.049 | 0.479 |
| eval_b_daglish | 0.088 | 0.559 |
| eval_b_follin | 0.039 | 0.416 |

(The atoms-only column above was a noisy 24-block sample. At **full eval** the atoms-only content
baseline is **0.515 / 0.516 / 0.479** — `v4_audit_posmatched.json` `atoms_content_overall` — so the
three subsets are ~level; "daglish beats in-distribution" does **not** hold at full eval.)

**De-confounded (§7, the decision-grade measures):**

| measure | eval_a | daglish | follin | read |
|---|---|---|---|---|
| **bits/canonical-atom** (§7A, tokenization-invariant) | 1.93→2.71 | 2.00→2.97 | 2.22→3.04 | **BPE ~1.4× worse** |
| **position-matched argmax** (§7B) atoms-only@base | 0.264 | 0.323 | 0.245 | vs BPE base ~0.05–0.13 → **~2–4×** |

The true model-quality gap is **~1.4× (bits) / ~2–4× (matched argmax)** — modest, NOT 6–11×. The
mechanism (BPE welds content atoms into multi-atom merges *across* event boundaries; merges ~1%
argmax-predictable) is real in *direction*, but the 6–11× magnitude was an artifact of the three
confounds, all now CONFIRMED.

### 1a. The three confounds — all CONFIRMED by §7 (magnitude retracted)

- **(a) Population — CONFIRMED.** The BPE content column ≈ surviving base atoms (the rare tail unigram
  didn't merge). §7B: restricting the *atoms-only* model to those same base positions drops it
  0.51→**0.26** (eval_a) — base positions are genuinely the harder tail. But atoms-only still beats
  BPE-base (~0.05–0.13) ~2–4× there, so population explains much of the gap, not all.
- **(b) Granularity — CONFIRMED structurally.** A merged token's argmax is JOINT over its k atoms;
  cross-tokenization argmax is not comparable. §7A (bits/atom) is the tokenization-invariant fix → ~1.4×.
- **(c) Training — CONFIRMED; matched-steps endpoint pinned (ep299).** §7C: the BPE arm trained
  100→174 (`version_0`) then extended 174→300 (`version_1`/`version_2`); the **matched-steps endpoint
  checkpoint EXISTS** — `version_2/checkpoints/best-epoch=299-val_loss=4.7437.ckpt` (`save_top_k=1` is
  per-version, so the extension saved its own best). The monitored **val_loss** (not train loss)
  descended 5.5728 (ep174) → 4.7437 (ep299) and is **still descending at ep299** (last 12 evals
  monotonic 4.787→4.744, no plateau). Auditing ep299 full-eval (`v4_audit_ep299.json` vs
  `v4_audit_ep174_fulleval.json`): the bits/atom gap to atoms-only **shrinks 1.43/1.50/1.39 (ep174) →
  1.24/1.31/1.21 (ep299)** and content-tier rises 0.187→0.226 / 0.147→0.190 / 0.141→0.182. So the
  matched-steps gap is **~1.2–1.3× and still closing** — no `save_last` re-run is needed (the endpoint
  was saved). **Counter-signal (the writeback omitted this):** the content/structural accuracy ratio
  FELL ep174→ep299 (eval_a 1.70→1.20) — extended training improved *structure* faster than content, so
  the content gain is partly a structural-prediction gain, not pure content learning.

**Operational verdict (direction unchanged, magnitude retracted):** atoms-only stays the default — it
is genuinely better (1.4× bits/atom) and is the shipped, content-correct representation. But the harm
is modest, so the §6 ban narrows to *unconstrained* merges; an event-boundary-respecting dictionary was
the candidate but triage-resolved NOT-adopted (§9). Carry-over: all-tier `val_acc` AND raw cross-tokenization content-tier argmax are
confounded — only bits/canonical-atom or position-matched argmax are decision-grade.

## 2. Per-KIND learnability map (atoms-only model, context-aware audit)

Content-digit accuracy on held-out composers, bucketed by the event KIND tracked from context:

| KIND | acc (a / daglish / follin) | % of stream | read |
|---|---|---|---|
| G_STEP (global filter regs 21–24) | 0.72 / 0.77 / 0.70 | 16–27% | learnable, huge |
| PW_RAMP (pulse-width sweep) | 0.56 / 0.51 / 0.49 | 11–23% | learnable, huge |
| G_RAMP, EVT74/72 | 0.29–0.78 | small–mid | mostly learnable |
| FD_STEP/FD_RAMP (freq residual) | 0.37–0.42 | ~15% | moderate |
| **NI_STEP (note-index = melody)** | **0.18 / 0.30 / 0.31** | 9–14% | **high-entropy (§7D: arpeggio 0.21–0.34 / stepwise 0.11–0.15)** |
| NI_RAMP (portamento) | 0.38–0.40 | 6–10% | hard |

**The atom model is good at timbre/envelope, weak at melody — with one caveat.** `NI_*` is the
Δnote *interval* lane by construction (`design/landed/universal_multiresolution_pitch.md`), while
the ≈0-entropy result that doc established is for *absolute onset* pitch (the anchor). The §7D split
ran: it bucketed NI_STEP by interval size (|Δnote|≥5 vs stepwise), **not** by phrase position — and
the large-interval bucket is 63–79% of NI_STEP with **median |Δ| = 12 semitones**, i.e. the SID
**arpeggio / large-interval class, not phrase onsets**. So the read is *large-interval (arpeggio)
0.21/0.34/0.33 vs stepwise 0.15/0.15/0.11*: arpeggio jumps are somewhat induction-predictable,
stepwise motion is the hard part — **both high-entropy, the claim does NOT narrow to anchors.** A
true *anchor* split (first onset after rest / voice start) remains untested. Either way the scoring
guidance stands: **score onsets by audition/distribution, not argmax** — intervals fix within-melody
transfer; the absolute anchor is creative content.

## 3. Value-encoding density is at the frontier (shipped); radix resolved LIVE (~12% polish lever)

The two density ideas one reaches for are already implemented and MDL-optimal **under the codec
cost model**:

- **Parametric ramps** — `stream._series_events` (events/stream.py) runs `cover(series, cost_model)`
  to segment every per-(voice,kind) value series into HOLD/POLY/PERIOD gestures; ramps emit
  `[kind][SHAPE][length][degree][v0][deltas]`, cutting those series 66–73% vs per-step emission.
  Shipped tokens 0.16/0.17 (`design/landed/unified_oscillation_primitive_design.md`), incl.
  gap-tolerant oscillation recognition (43.9% FREQ coverage) + lossless delta payload (30%
  FREQ-atom reduction).
- **Per-voice note-table pitch** — shipped tokens 0.47.0
  (`design/landed/universal_multiresolution_pitch.md`): universal semitone NI_* lane (Δnote
  intervals) + per-voice recovered NOTE_TABLE + per-voice TUNING + FD_* modulation residual. 83% of
  voiced frames have residual exactly 0 (codec v1; the v2 modal-table pitch fix, §9, raises this — re-measure).

Full-corpus survey (§7F, 862-tune `.atoms.zst`): **atoms/tune mean 85k, median 50k** (the 120-sample
~49k ≈ median). The repo-standard "~30k atoms/tune" was wrong-unit — it is **BPE tokens/tune** (mean
31k = atoms/2.73×, median 18k); AGENTS.md's "4.16 KEYFRAME windows/tune" is the mean *BPE* windows
(31k/8192), not atoms. **No dataset truncation:** `BlockMapper` (`preframr/train/block_mapper.py`)
tiles every block of every tune — the model trains on every atom (atoms-only ~6 windows/tune median,
BPE ~2.2). Composition (authoritative, incl. typed nibbles as content): **content 69.0%** (varint
50.1% + nibbles 18.9%), **head 28.5%** (KIND 15.9% / VOICE 11.1% / reg 1.5%), gesture SHAPE 2.5%.
**67% of events are 2–4 atoms** (modal event = 3). The §4–5 conclusions are robust (50k ≫ 8192).

- **Digit radix is a LIVE polish-grade lever (§7E), not dead.** Mean 1.351 digits/value overall
  (36755978 digits / 27196187 values), but a deterministic byte-pack (one token per value,
  `saved = d − ⌈d/2⌉`) saves **8.79M atoms = ~12% of the stream** (24% of varint content) — head-amortization-class, and on a *disjoint* atom family, so
  the two **stack** (~1.3× combined density). The juice is per-lane: **FD_STEP 1.62, PW_STEP 1.53,
  PW_RAMP 1.52, FD_RAMP 1.49, NI_STEP 1.46**; FLD_CTRL/FLD_SR are dead (1.001, nibble-based). **P1
  scope:** P1 permits one-token-per-value for *multi-digit varints* (one decision → one token) but
  forbids welding distinct decisions, so the **18.9% single-nibble content (NIB_WAVE/ART/ENV) is OUT
  of scope** — a *selective per-lane* typed-byte family only. Still not a context lever (50k → ~44k ≫
  8192); gate on `learnability_triage` + P1 litigation before any run. (Supersedes the crude ≳1.7
  threshold: the decision metric is the computed byte-pack saving, not the raw mean.) **Codec caveat:
  these per-lane digits/value are codec v1; the 0.51.0 pitch fix (§9) moved the FD_\* residual split
  specifically, so re-measure digits-per-value on v2 before implementing the byte-pack — FD_STEP 1.62
  / FD_RAMP 1.49 are the most exposed lanes.**

## 4. The second open density lever: head-amortization (~10–15%, optional)

Each event pays a `[KIND]` atom (and `[reg]` for globals); the `[VOICE]` atom is already amortized
per frame-group. Since 67% of events are 2–4 atoms, heads are head-heavy. §7F breaks the 28.5% head
share into KIND 15.9% / VOICE 11.1% / reg 1.5%: VOICE is already amortized (only **~0.7 VOICE atoms
per event** — frame-groups are *tiny*, so per-frame-group amortization is already near its limit; do
not reopen voice-order work, see `data/refuted/sequence_order_normalization_design.md`), so the
**recoverable ceiling is KIND+reg = 17.4%**, realistic recovery ~10–15% (you cannot eliminate all
event identification). Candidate mechanisms (deterministic,
byte-exact-preserving, single-token-per-unit — *not* data-driven merges): combined `(voice,kind)`
or `(kind,reg)` atoms for the common cases; context-predicted kind elision where the grammar makes
it unambiguous. **Audit caution:** combined atoms lower measured structural/all-tier accuracy by
construction (joint granularity, §1a-b) — pre/post-amortization runs compare on content-tier only.
**Worth doing as polish, but it does not move the needle on context** — 50k median → ~44k (85k mean
→ ~75k) atoms/tune is still ≫ 8192. (Together with radix §3 the two stack to ~1.3× density; neither
is a context lever.) The head/composition shares are the v1 survey; the v2 pitch fix barely touches
head markers, but re-confirm against the v2 re-encode when implementing (§9 codec note).

## 5. Context length is a `seq_len`/windowing/chaining problem, not an encoding problem

There is no large remaining density win to fit tunes in the window — BPE was the only thing
achieving 2.73× compression, and on current evidence it does so at a modest but real content cost
(~1.4× bits/atom, §1), by welding content across event boundaries. Therefore:

- **Use tkvocab=0 (atoms-only).** Keep `tkvocab` as a dial but not as the strategy. **Accepted
  cost:** the Orin PROMPT=2048 carries ~2.73× less music without merges — prompt-side mitigation
  belongs to `design/generation/prompt_interface_design.md`, not the encoding.
- **For more tune-per-window, scale `seq_len`** (8192 → 16384; AGENTS.md expects the 14M body fits
  24 GB — verify before the re-cut; costs a dataset re-cut + wallclock) and/or cut
  **musically-aligned KEYFRAME windows** at pattern/loop boundaries (dataset-side, no alphabet
  change). These raise tune-per-window; they are **not** whole-tune mechanisms (50k median / 85k
  mean ≫ 16384).
- **Whole tunes never required whole-tune windows:** register-domain decode-and-recompile
  **chaining** (`design/generation/long_range_structure.md`) re-canonicalizes state at KEYFRAME
  seams, so the window only has to carry local structure — chaining is the norm path for full
  tunes, which is what makes context a windowing problem rather than an encoding one.
- **Evaluate melody by audition/distribution, not next-token argmax** — high-entropy by nature (§2).

## 6. For the next agent — what to do, what not to do

- **§7 RAN** (`data/audit/deconfound_summary.md`) — verdict revised: harm is ~1.4× bits/atom, not 6–11×.
- **Do not re-attempt:** *unconstrained* data-driven merges (BPE/unigram) that cross event boundaries
  (still refuted — they welded content and cost 1.4× bits/atom); the "denser alphabet to fit the
  window" framing (no density lever fits tunes in the window — §5); parametric ramps or per-voice
  pitch tables (shipped).
- **RESOLVED — boundary-respecting dictionary NOT adopted (§9 triage, 2026-06-13).** The
  event-boundary-respecting dictionary (proposal `event_boundary_dictionary_proposal.md`, shipped
  tokens 0.51.0) ran its static triage on the v2 codec: compression caps at **1.58/1.65/1.71×**
  (tkvocab 1024/2048/4096), **below the 1.8× ADOPT bar at every vocab** and asymptoting ~1.7×, and
  the merge table is **~89% deterministic-pack-shaped** (head+payload + within-value/DT digits). Per
  the PARTIAL gate the deterministic packs capture the same gain more cheaply → **the packs are the
  density path; the dictionary is not adopted** (`data/audit/boundary_dictionary_triage_summary.md`).
- **LIVE polish levers (the chosen density path — stackable, ~1.3× combined, neither a context lever):**
  (1) **selective per-lane digit byte-pack** (§3/§7E, ~12%, scoped to FD/PW/NI multi-digit varints,
  P1-litigated); (2) **head-amortization** in `events/stream.py` (≤~13–17%; combined `(kind,reg)` /
  context-predicted kind elision). Both: keep `encode(verify=True)` byte-exact; bump
  `EVENT_FORMAT_VERSION` + `ATOM_CACHE_VERSION`; compare pre/post on content-tier only.
- **Context arc:** `seq_len` 16384 + musically-aligned windows on the atoms-only encoding + the
  chaining gate (`design/generation/long_range_structure.md`).

## 7. De-confounding audit (RAN 2026-06-12 — results in `data/audit/`)

Outcome: the 6–11× magnitude is **retracted**; the direction (BPE costs more per canonical atom)
**survives**. Full writeup `data/audit/deconfound_summary.md`; per-task JSONs alongside. Summary:

- **A. Bits per canonical atom (decisive). RAN** (`v4_audit_bits_per_atom.json`, full eval
  `--max-blocks 0`, all three subsets). True gap is **~1.4× bits/atom**, not 6–11×: BPE pays
  ~1.93→2.71 bits/atom vs atoms-only across subsets. The headline argmax table was
  confound-dominated.
- **B. Position-matched argmax. RAN** (`v4_audit_posmatched.json`). On base (length-1) positions,
  atoms-only scores **0.264 / 0.323 / 0.245** content accuracy (eval_a / b_daglish / b_follin) —
  refutation survives (well above the ~5–9% confound-(a) bar) but ~2–4×, not the raw table's gap.
  Confound (a) confirmed: the atoms-only population accuracy drags 0.51→0.26 once restricted to
  base positions, so the raw cross-tokenization table compared unlike populations.
- **C. Matched-steps extension. RAN** (`v4_audit_ep299.json` + `v4_audit_ep174_fulleval.json`;
  endpoint ckpt `version_2/best-epoch=299-val_loss=4.7437.ckpt`). The 174→300 extension's endpoint
  WAS saved (`save_top_k=1` is per-version, in `version_2`). Monitored **val_loss** descended 5.5728
  (ep174) → 4.7437 (ep299) and is **still descending at ep299** (no plateau). At the matched-steps
  endpoint the bits/atom gap to atoms-only **shrinks 1.43/1.50/1.39 → 1.24/1.31/1.21** and content
  rises 0.187→0.226 / 0.147→0.190 / 0.141→0.182 (full eval). Confound (c) confirmed; matched-steps
  gap **~1.2–1.3× and still closing**. **Counter-signal:** content/structural ratio FELL ep174→ep299
  (extended training favored structure over content). No `save_last` re-run needed — endpoint pinned.
- **D. NI_STEP split. RAN** (`v4_audit_nistep.json`). The split is by interval size (|Δnote|≥5 vs
  stepwise), **not** phrase position — and the |Δ|≥5 bucket is 63–79% of NI_STEP with median |Δ| =
  12 semitones, i.e. the SID **arpeggio / large-interval class, not phrase onsets**. Large-interval
  0.21/0.34/0.33 vs stepwise 0.15/0.15/0.11: both high-entropy — keep the high-entropy read,
  regime-conditioned; the melody claim does **not** narrow to anchors. A true *anchor* split (first
  onset after rest / voice start) remains untested.
- **E. Digits-per-value distribution. RAN** (`data/audit/deconfound_windows_and_digits_per_value.txt`).
  Radix is a **polish-grade ~11–12% per-lane lever**, not dead and not a context lever: multi-digit
  varint lanes (FD_STEP 1.62, PW_STEP 1.53, NI_STEP, G_STEP digits/value) carry the juice; 18.9% of
  content is single-nibble (already optimal, out of scope). P1-scoped, stackable with
  head-amortization.
- **F. Full-corpus survey re-run. RAN** (same file). **No truncation** — `BlockMapper.__len__` sums
  all blocks and tiles every atom (49k-vs-30k was atoms-vs-BPE-tokens, not a windowing loss). Tunes
  ~50k atoms median / 85k mean; ~18k BPE-tokens median / 31k mean. Composition: content **69%**,
  recoverable head (KIND + reg) **17.4%** (replaces the earlier 25.9%); VOICE markers ~0.7/event
  already amortized (see `data/refuted/sequence_order_normalization_design.md` — do not reopen
  voice-order work). Denominators: the §7A/B/D audits eval **856 dumps** (provenance corpus); the F
  composition survey is over **862 tunes** (full corpus).
- **G. Provenance + writeback. RAN** — JSONs copied to `data/audit/`; §8 mapping applied to this
  doc, the refuted entry, and the AGENTS.md resolved log.

## 8. What would reopen this (falsifiers → actions)

- **A/B ≈ parity** (per-position or bits/atom) → retract the 6–11× magnitude in the registry;
  promote the boundary-respecting dictionary (§6) to a live lever. **FIRED (partial):** not parity
  (A ~1.4× bits/atom, B ~2–4× position-matched, both real), but the 6–11× magnitude is retracted
  and the boundary-respecting dictionary is **promoted to a live lever** (§6).
- **C shows a late content transition** approaching baseline → same as above; re-take the verdict
  at matched steps. **FIRED:** at the matched-steps endpoint (ep299, pinned ckpt) the bits/atom gap
  shrinks 1.4×→~1.2–1.3× and val_loss is still descending — verdict re-taken as ~1.2–1.3× and
  closing (counter-signal: content/structural ratio fell, so the gain skews structural).
- **E mean ≳1.7 nibbles/value** → radix is live; `learnability_triage` it against P1 before any run.
  **FIRED (revised):** threshold replaced by computed savings — radix is a **live ~11–12% per-lane
  polish lever** (FD/PW/NI multi-digit varint lanes), P1-scoped, stackable with head-amortization;
  not a context lever.
- **D: within-phrase fine, anchors drag** → melody claim narrows to anchors (encoding exonerated);
  **both ~0.18** → keep the high-entropy read, regime-conditioned. **FIRED (latter), with a label
  correction:** the split was by interval size (large-interval/arpeggio 0.21–0.34 vs stepwise
  0.11–0.15), not by phrase position — both high-entropy, claim does **not** narrow to anchors; a
  true anchor split remains untested.
- **F shifts composition shares materially** → redo §4's ceiling/recovery arithmetic. **FIRED:**
  content 69% / recoverable head 17.4% (was 25.9%); §3/§4 arithmetic redone. No truncation.
- **None of the above fire** → strike "provisional" throughout; §6's ban becomes unconditional.
  **DID NOT FIRE:** confounds (a)/(b)/(c) all confirmed → the ban stays scoped to *unconstrained*
  cross-boundary merges; the gap is real but modest (~1.2–1.4×) and still closing at matched steps.

## 9. Boundary-dictionary triage (RAN 2026-06-13 — NOT adopted, deterministic packs win)

The §6-promoted event-boundary-respecting dictionary (proposal
`event_boundary_dictionary_proposal.md`, shipped tokens 0.51.0) ran its static triage on the **v2
codec** (`EVENT_FORMAT_VERSION=2` — tokens 0.51.0 bundled an owner-directed pitch fix that bumped the
codec; the v1 baseline is stale, so the matched-steps A/B was deferred and ultimately not needed).
Full artifacts `data/audit/boundary_dictionary_triage_v2.json` + `..._summary.md`.

- **Compression (eval_a aggregate / train): 1.58×/1.58× (1024), 1.65×/1.65× (2048), 1.71×/1.71×
  (4096).** Survives the ≥1.5× kill gate but is **below the 1.8× ADOPT bar at every vocab** and
  asymptoting ~1.7× (+0.07 per vocab-doubling). Unconstrained BPE was 2.73×; the boundary constraint
  costs ~40% of the compression and undershot the proposal's 1.8–2.5× estimate. ⇒ **ADOPT is
  structurally unreachable** for this corpus, *independent of the bits/atom A/B* (which gates on
  compression ≥1.8× AND bits ≤1.05×), so the A/B was not run.
- **Merge table ~89% deterministic-pack-shaped** (tkvocab=2048, 1919 multi-atom pieces by count):
  **57.7% head+payload** (single-kind `[kind][payload digits]` → §4 head-amortization), **31.2%
  within-value/DT digits** (→ §3 radix byte-pack), 11.1% other. The dictionary captures almost
  exactly what the two deterministic packs target. Per the **PARTIAL gate** ("prefer whichever
  captures the gain more cheaply") the packs win — no dictionary infra, no codec/A-B-baseline.
- **Weld-free invariant holds** (0 real crossings at every vocab; the 3 flagged at 4096 are
  `[VOICE][TUNING][digit]` header-unit pieces — a `bpe_audit` heuristic false-positive, since a
  header-section VOICE marker legitimately starts a multi-atom unit). **Live-vocab healthy**
  (96.7/98.3/99.2%). Window math: median tune ~26k dict-tokens at 2048 ≈ 3.1 windows @8192 / ≈1.6
  @16384 — does not rescue ADOPT.

**Decision: the boundary-respecting dictionary is NOT adopted.** The density path is the deterministic
packs (§3 radix + §4 head-amortization, ~1.3× combined); the context arc proceeds on atoms-only +
`seq_len`/windowing/chaining (§5). Open (separate, operator's call): tokens 0.51.0's v2 codec is live
on PyPI but the framework/xpt are still on 0.50.0/0.2.29 — adopting v2 (e.g. for the packs work or any
new canonical run) requires the corpus re-encode + a fresh v2 atoms-only baseline.

## Provenance

Stack: tokens 0.50.0 / preframr 0.2.29; spec `generalize` (canonical 14M body, 8L-d320-im896);
single-speed 856-dump corpus. Baseline: atoms-only v3c, epoch 99/100, val_acc 0.561, ckpt
`/scratch/tmp/v3c_final.ckpt`. BPE: root `/scratch/tmp/preframr_experiments/unigram_canonical_v4`;
the canonical arm ran to ep174 (`version_0`, best val_loss 5.5728), then was **extended 174→300**
(`version_2`, best `epoch=299-val_loss=4.7437.ckpt`) for the §7C matched-steps test — monitored
val_loss descended 5.57→4.74 and was still descending at ep299. Audits: §7A/B/C/D ran **full-eval
(`--max-blocks 0`)** on the v4 tkmodel, GPU-ized on defroster. Artifacts in `data/audit/`:
`v4_audit_bits_per_atom.json` (A, atoms-only + BPE ep174), `v4_audit_posmatched.json` (B),
`v4_audit_ep299.json` + `v4_audit_ep174_fulleval.json` (C, full-eval content + bits/atom at both
endpoints), `v4_audit_ep174.json` + `v4_audit_ep174_postext.json` (24-block per-class trajectory;
the latter is a post-extension re-audit of ep174, NOT an ep300 audit), `v4_audit_nistep.json` (D),
`deconfound_windows_and_digits_per_value.txt` (E/F), `deconfound_summary.md` (full writeup).
