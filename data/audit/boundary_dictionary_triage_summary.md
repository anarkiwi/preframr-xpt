# Boundary-dictionary static triage — results (2026-06-13)

Stack: preframr-tokens **0.51.0** (v2 codec, `EVENT_FORMAT_VERSION=2`, `unit_starts` boundary
segmenter) over the framework HEAD + xpt, run on fogbank. Fresh root
`/scratch/tmp/preframr_experiments/unigram_boundary_v1` (never reused v4's unconstrained-merge
caches). Corpus: canonical train+eval, **856 in-scope** (789+80+16+16 listed = 901; dropped 42
multispeed + 3 digi — matches the work order's 856). Non-destructive: `.2.atoms.zst` sidecars written
alongside the v1 `.1` caches. Raw: `boundary_dictionary_triage_v2.json`.

## Verdict: SURVIVES kill gate, resolves PARTIAL → deterministic packs win → dictionary NOT adopted

The boundary-respecting dictionary clears the ≥1.5× kill gate but **cannot reach ADOPT** (compression
<1.8× at every vocab) and the merge table is **~89% deterministic-pack-shaped**, so per the proposal's
PARTIAL gate the deterministic packs (frontier §3 radix + §4 head-amortization) capture the same gain
more cheaply (no dictionary infra, no codec/A-B-baseline). The matched-steps bits/atom A/B was **not
run** — with compression <1.8×, ADOPT is unreachable regardless of the bits result.

## Compression (atoms / dictionary-tokens) — the gate

| tkvocab | eval_a | eval_b_daglish | eval_b_follin | train | live-vocab | weld-free | tok/tune med (eval_a) |
|---|---|---|---|---|---|---|---|
| 1024 | 1.578× | 1.560× | 1.548× | 1.582× | 96.7% | ✓ (0) | 27153 |
| 2048 | 1.650× | 1.661× | 1.642× | 1.652× | 98.3% | ✓ (0) | 25833 |
| 4096 | 1.712× | 1.741× | 1.728× | 1.714× | 99.2% | ✓ (3*) | 25748 |

(aggregate ratios; per-tune medians track within ~1%. * the 3 "crossings" at 4096 are
`[VOICE][TUNING][digit]` header-unit pieces — a `bpe_audit` heuristic false-positive, not real welds.)

- Survives the ≥1.5× kill gate, but **below the 1.8× ADOPT bar at every vocab**, asymptoting ~1.7×
  (+0.07 per vocab-doubling). Unconstrained BPE-2048 was **2.73×**; the boundary constraint costs
  ~40% of the compression and undershot the proposal's 1.8–2.5× estimate.
- Window math: median tune ~26k dict-tokens at 2048 ≈ **3.1 windows @8192 / ≈1.6 @16384** — does not
  rescue ADOPT (which gates on compression, not window count).

## Merge-table classification (tkvocab=2048, 1919 multi-atom pieces; by piece count)

| class | share | maps to |
|---|---|---|
| head+payload (single-kind `[kind][payload digits]`) | **57.7%** (1107) | §4 head-amortization |
| within-value / DT digit runs (all-digits) | **31.2%** (599) | §3 radix byte-pack |
| other (no-kind) | 11.1% (213) | — |

~89% of the learned multi-atom vocabulary is exactly what the two deterministic packs target. The
packs capture the same gain without dictionary training, on a disjoint atom family (stack ~1.3×).
Piece-length histogram (2048): modal 2–6 atoms, max 16. (Frequency-weighted savings shares were a
supplementary metric; the piece-count shares + the structural <1.8× compression cap already determine
PARTIAL, so the verdict does not depend on them.)

## Codec note (the P0.2 STOP that shaped this run)

tokens 0.51.0 bundled, beyond the dictionary, an **owner-directed pitch fix** (`recover_table` modal
vs median) that changes encode output on real dumps → `EVENT_FORMAT_VERSION`/`ATOM_CACHE_VERSION`
bumped 1→2. Measured v1→v2 atom-stream delta on 18 eval dumps: **−0.78% aggregate, fat-tailed**
(Hades_Nebula −10.9%, Northstar −5.0%, several ±0.1–0.3%, some unchanged). The v1 atoms-only baseline
(`v3c_final.ckpt`, bits/atom 1.931/2.001/2.221) is therefore **stale**; a valid matched-steps A/B
would require re-encoding the corpus on v2 + a fresh v2 atoms-only baseline. Since the triage already
caps the outcome at PARTIAL, that re-baseline was not run.

## Method

Boundary dictionaries trained via the real `unit_starts` segmenter (`make_tokenizer` +
`tk.unit_segmenter = dataset.unit_starts` + `train_tokenizer`, identical to `Corpus.preload`), on
train+eval atoms (cached `.2.atoms.zst`). Compression = `len(atoms) / len(tk.encode(atoms))` per tune.
Merge table via `preframr_tokens.bpe_audit.audit_vocab`. Scripts: `/scratch/tmp/boundary_triage.py`,
`/scratch/tmp/inspect_welds.py`.
