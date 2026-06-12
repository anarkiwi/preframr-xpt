# Encoding-density frontier — atoms-only is the default, BPE refuted as the context lever (magnitude provisional), head-amortization the one open lever

**Status: DECISION (2026-06-12; evidence re-scoped same day after a confound review).** Settles the
"compress the encoding to fit tunes in the window" question with data from the canonical
learnability run + a corpus encoding survey. Verdict: **the atoms-only event encoding (tkvocab=0)
is the shipped default and the content-correct representation on current evidence; unigram/BPE is
refuted as the content/context lever — operationally final, but the measured 6–11× harm magnitude
is PROVISIONAL (three named confounds, §1a) until the §7 de-confounding audit runs. Value-encoding
density is at its frontier per P1-separability + the shipped ramp/pitch primitives, with one number
outstanding (digit radix, §3). The only open density lever is head-amortization (~10–15%, §4),
which does not change the context-window picture.** Context length is a `seq_len`/windowing/chaining
problem, not an encoding problem (§5). Melody is high-entropy next-token at this regime (§2).

## 1. BPE harms content generalization as measured (~6–11×) — magnitude provisional

Canonical `generalize` 14M body (8L-d320-im896), atoms-only (tkvocab=0) vs unigram BPE-as-dictionary
(tkvocab=2048, ~2.6× stream compression, ~98% live vocab), same corpus/holdouts, content-tier
accuracy at matched ~epoch 100. Definitions: **content-tier** = value-digit atoms only (markers are
structural); **eval_a** = in-distribution holdout; **eval_b_daglish / eval_b_follin** = held-out
composers. Registry entry: `data/refuted/unigram_bpe_content_generalization.md`.

| eval subset | BPE-2048 | atoms-only |
|---|---|---|
| eval_a | 0.049 | 0.479 |
| eval_b_daglish | 0.088 | 0.559 |
| eval_b_follin | 0.039 | 0.416 |

**Mechanism (localized; direction plausibly real):** merged BPE tokens are ~1% argmax-predictable
(surviving base atoms 4–8%). BPE is a data-driven merge blind to the event grammar; it welds content
atoms into multi-atom merges *across* event boundaries. The ~1% sits below even the parity-expected
joint accuracy for 2–3-atom merges (~0.11–0.23 at the baseline's 0.42–0.56/atom), so merges look
genuinely degraded — but see confound (b).

### 1a. Why the magnitude is provisional (three confounds, none yet excluded)

- **(a) Population.** The BPE content-tier column ≈ the *surviving base atoms* (4.9/8.8/3.9% ≈ the
  "4–8%" base range) — exactly the rare tail unigram chose NOT to merge — while the atoms-only
  column averages over ALL content atoms. Frequent (≈predictable) content sits inside merges on the
  BPE side; the two columns score different populations by construction.
- **(b) Granularity.** A merged token's argmax accuracy is JOINT over its k atoms: at parity with a
  0.48/atom model a 3-atom merge scores ~0.11, so low merge accuracy is partially expected at zero
  harm. Per-token argmax does not compare across tokenizations.
- **(c) Training.** Matched epochs ≠ matched steps: at seq_len 8192, BPE tunes are ~1.4 windows vs
  ~4.2 atoms-only → ~3× fewer optimizer steps by epoch 100; v4 was stopped mid-descent (val_loss
  7.08→6.27) and the documented plateau→steep-drop content transition (AGENTS.md) is unexcluded —
  the table may sample v4 pre-transition.

**The operational verdict stands while §7 runs:** atoms-only stays the default — even at per-atom
parity the dial buys no content, and if welding is the mechanism, a LARGER vocab means MORE welding
(which closes the planned 4096–16384 sweep by mechanism, conditional on §7 confirming welding).
What the confounds gate is the *strength of the ban* (§6) and the registry magnitude. Carry-over,
now extended: all-tier `val_acc` is confounded across tokenizations (bigger vocab ⇒ higher per-token
entropy) — **and per §1a content-tier is too**; cross-tokenization comparisons must be
position-matched or in bits/canonical-atom (§7A/B).

## 2. Per-KIND learnability map (atoms-only model, context-aware audit)

Content-digit accuracy on held-out composers, bucketed by the event KIND tracked from context:

| KIND | acc (a / daglish / follin) | % of stream | read |
|---|---|---|---|
| G_STEP (global filter regs 21–24) | 0.72 / 0.77 / 0.70 | 16–27% | learnable, huge |
| PW_RAMP (pulse-width sweep) | 0.56 / 0.51 / 0.49 | 11–23% | learnable, huge |
| G_RAMP, EVT74/72 | 0.29–0.78 | small–mid | mostly learnable |
| FD_STEP/FD_RAMP (freq residual) | 0.37–0.42 | ~15% | moderate |
| **NI_STEP (note-index = melody)** | **0.18 / 0.30 / 0.31** | 9–14% | **high-entropy (split pending)** |
| NI_RAMP (portamento) | 0.38–0.40 | 6–10% | hard |

**The atom model is good at timbre/envelope, weak at melody — with one caveat.** `NI_*` is the
Δnote *interval* lane by construction (`design/landed/universal_multiresolution_pitch.md`), while
the ≈0-entropy result that doc established is for *absolute onset* pitch (the anchor). The
0.18–0.31 here therefore mixes anchor-like phrase-initial jumps with within-phrase steps — the §7D
split separates them. Until then the claim is **high-entropy at this regime** (14M body, 1 seed,
24-block audit sample, held-out composers), not "intrinsic". Either way the scoring guidance
stands: **score onsets by audition/distribution, not argmax** — intervals fix within-melody
transfer; the absolute anchor is creative content.

## 3. Value-encoding density is at the frontier (shipped), one number outstanding

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
  voiced frames have residual exactly 0.

Corpus survey (120-tune `.atoms.zst` sample): **avg 6.42 atoms/event, ~49k atoms/tune.**
Composition: ~48% content value-digits, **25.9% `[VOICE][KIND][reg]` head markers** (the
amortization ceiling), remainder gesture params (SHAPE/length/degree) + KEYFRAME conditioning.
**67% of events are 2–4 atoms** (modal event = 3: kind + ~2 payload). Two caveats:

- **49k conflicts with the repo-standard ~30k atoms/tune** (AGENTS.md,
  `design/generation/long_range_structure.md`, `design/references/tokenization_vs_music_llms.md`) —
  likely pre-encode sample bias (the scope-filtered, interruptible pre-encoder) or dumps-vs-tunes
  counting; §7F re-runs the survey over the full 856-dump corpus. The §4–5 conclusions are robust
  to either number (26k or 42k ≫ 8192).
- **Digit radix is the one unexamined token-count lever.** "~48% content digits are irreducible" is
  true in *bits* and is grounded in **P1-separability**
  (`design/references/encoding_principles.md`, earned by the 0.009→0.658 pitch-onset de-merge) —
  but values are typed nibbles + BE varints, and a deterministic nibble-pair→typed-byte merge is
  the same blessed family as §4 (single-token-per-unit, codec-pinned). One number closes it: the
  digits-per-value distribution (§7E). Mean ≈1.x nibbles/value ⇒ dead; ≈2 ⇒ a ~20% lever bigger
  than head-amortization, then litigated against P1 via `learnability_triage` before any run.

## 4. The only open density lever: head-amortization (~10–15%, optional)

Each event pays a `[KIND]` atom (and `[reg]` for globals); the `[VOICE]` atom is already amortized
per frame-group. Since 67% of events are 2–4 atoms, heads are head-heavy. Ceiling = 25.9% of the
stream — but that share lumps VOICE (already amortized): the recoverable fraction is the KIND+reg
share, which the survey did not break out (§7F reports it); realistic recovery ~10–15% pending that
breakdown (you cannot eliminate all event identification). Candidate mechanisms (deterministic,
byte-exact-preserving, single-token-per-unit — *not* data-driven merges): combined `(voice,kind)`
or `(kind,reg)` atoms for the common cases; context-predicted kind elision where the grammar makes
it unambiguous. **Audit caution:** combined atoms lower measured structural/all-tier accuracy by
construction (joint granularity, §1a-b) — pre/post-amortization runs compare on content-tier only.
**Worth doing as polish, but it does not move the needle on context** — ~49k → ~42k (or ~30k →
~26k) atoms/tune is still ≫ 8192.

## 5. Context length is a `seq_len`/windowing/chaining problem, not an encoding problem

There is no large remaining density win to fit tunes in the window — BPE was the only thing
achieving 2.6× compression, and on current evidence it does so by destroying content learnability
(§1). Therefore:

- **Use tkvocab=0 (atoms-only).** Keep `tkvocab` as a dial but not as the strategy. **Accepted
  cost:** the Orin PROMPT=2048 carries ~2.6× less music without merges — prompt-side mitigation
  belongs to `design/generation/prompt_interface_design.md`, not the encoding.
- **For more tune-per-window, scale `seq_len`** (8192 → 16384; AGENTS.md expects the 14M body fits
  24 GB — verify before the re-cut; costs a dataset re-cut + wallclock) and/or cut
  **musically-aligned KEYFRAME windows** at pattern/loop boundaries (dataset-side, no alphabet
  change). These raise tune-per-window; they are **not** whole-tune mechanisms (30–49k ≫ 16384).
- **Whole tunes never required whole-tune windows:** register-domain decode-and-recompile
  **chaining** (`design/generation/long_range_structure.md`) re-canonicalizes state at KEYFRAME
  seams, so the window only has to carry local structure — chaining is the norm path for full
  tunes, which is what makes context a windowing problem rather than an encoding one.
- **Evaluate melody by audition/distribution, not next-token argmax** — high-entropy by nature (§2).

## 6. For the next agent — what to do, what not to do

- **Run §7 first** — cheap, mostly CPU, and it gates everything marked "provisional".
- **Do not re-attempt:** *unconstrained* data-driven merges (BPE/unigram) as a content or context
  lever (refuted as measured; §7 sets the final magnitude); the "denser alphabet to fit the window"
  framing; parametric ramps or per-voice pitch tables (shipped).
- **Scoped untested (deprioritized, NOT banned):** an *event-boundary-respecting* dictionary —
  data-informed merge selection but merges never cross event boundaries, codec-pinned (the §4
  family generalized; bounded ~2× given modal event = 3). Revisit only if §7A/B exonerates merges
  per-position AND a window-fit need survives chaining.
- **Optional polish:** head-amortization in `events/stream.py` (≤~13% length; keep
  `encode(verify=True)` byte-exact; bump `EVENT_FORMAT_VERSION` + `ATOM_CACHE_VERSION`).
- **Context arc:** `seq_len` 16384 + musically-aligned windows on the atoms-only encoding + the
  chaining gate (`design/generation/long_range_structure.md`).

## 7. De-confounding audit (specified, PENDING — run before hardening any ban)

All tasks reuse the two existing checkpoints + audit machinery; A/B/D/E/F are audit/corpus scripts
(`--device cpu` works), C is the only training.

- **A. Bits per canonical atom (decisive).** Per eval dump (all three subsets, full eval,
  `--max-blocks 0`): total ground-truth NLL of the token sequence under each checkpoint ÷ the
  dump's canonical atom count. Tokenization-invariant up to the deterministic-encoding caveat;
  report per-subset bits/atom + per-tune NLL.
- **B. Position-matched argmax.** Decompose BPE ids into constituent atoms via the v4 tkmodel/merge
  table; mark atom positions covered by length-1 (base) tokens; score the *atoms-only* checkpoint's
  content accuracy restricted to those positions vs BPE's base-atom accuracy on the same positions.
  Atoms-only also ~5–9% there ⇒ confound (a) explains the table; holds ~40%+ ⇒ refutation survives.
- **C. Matched-steps extension.** Train BPE-2048 to ~300 epochs (≈3× steps ≈ matched; BPE epochs
  run ~3× faster) or until content-tier flattens; audit content-tier every ~25 epochs — checks the
  late-transition explanation (c). Set `max_epochs` deliberately (early-stop never fires under
  schedule-free, AGENTS.md gotcha).
- **D. NI_STEP split.** Content accuracy split phrase-initial (first onset after rest/voice start,
  or large |Δnote|) vs within-phrase steps, per eval subset.
- **E. Digits-per-value distribution** over the corpus `.atoms.zst` (mean + histogram of
  value-digit count per value field) — decides the §3 radix question.
- **F. Full-corpus survey re-run** after full pre-encode (`preencode_corpus.sh --only-missing` on
  fogbank): atoms/tune mean+median (49k-vs-30k reconciliation), composition shares, and the
  per-marker-type breakdown of the 25.9% head share (VOICE vs KIND vs reg) for §4's arithmetic.
- **G. Provenance + writeback.** Copy `/scratch/tmp/v4_audit*.json` + all new audit JSONs into
  `data/audit/`; then apply §8's mapping to this doc, the refuted entry, and the AGENTS.md
  resolved log.

## 8. What would reopen this (falsifiers → actions)

- **A/B ≈ parity** (per-position or bits/atom) → retract the 6–11× magnitude in the registry;
  promote the boundary-respecting dictionary (§6) to a live lever.
- **C shows a late content transition** approaching baseline → same as above; re-take the verdict
  at matched steps.
- **E mean ≳1.7 nibbles/value** → radix is live; `learnability_triage` it against P1 before any run.
- **D: within-phrase fine, anchors drag** → melody claim narrows to anchors (encoding exonerated);
  **both ~0.18** → keep the high-entropy read, regime-conditioned.
- **F shifts composition shares materially** → redo §4's ceiling/recovery arithmetic.
- **None of the above fire** → strike "provisional" throughout; §6's ban becomes unconditional.

## Provenance

Stack: tokens 0.50.0 / preframr 0.2.29; spec `generalize` (canonical 14M body, 8L-d320-im896);
single-speed 856-dump corpus. Baseline: atoms-only v3c, epoch 99/100, val_acc 0.561, ckpt
`/scratch/tmp/v3c_final.ckpt`. BPE: root `/scratch/tmp/preframr_experiments/unigram_canonical_v4`,
stopped ~epoch 100 mid-descent (val_loss 7.08→6.27). Audits: `audit_checkpoint_per_class` +
`content_tier_report`, **1 seed, 24-block samples** (full-eval pending §7A). Artifacts
`/scratch/tmp/v4_audit*.json` + session-log scripts are **ephemeral until §7G copies them into
`data/audit/`**.
