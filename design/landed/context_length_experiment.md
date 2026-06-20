# Context-length arc — does longer `seq_len` improve held-out continuation? (v2)

**Status: RAN, INCONCLUSIVE — step-confounded (2026-06-13); decisive re-do = matched STEPS (see
Results).** The first experiment after the v2 re-baseline; AGENTS NEXT #1. Overnight `seq_len` sweep
on the atoms-only v2 model, decided in bits/canonical-atom + content-tier (full-eval). The
matched-epochs design conflated context with optimizer-step budget; do not read the raw sweep as
"context hurts". Batch: `preframr_experiments/ctx_overnight_batch.sh`.

## Why this is the experiment

The encoding question is closed (atoms-only wins; BPE refuted, boundary dictionary triaged
not-adopted). The remaining **first-order** lever on generalisation is **context**: the median tune
is ~50k atoms ≈ 6 windows at `seq_len` 8192, so the model trains/predicts on **self-contained
windows, never whole tunes**. More coherent music per window → more in-context structure a bounded
(~TC⁰) transformer can exploit for the next-token map. Density (the deterministic packs) and
embedding/conditioning are second-order behind it.

Framing caveat (frontier §5): this is **not** about fitting a whole tune in one window — whole-tune
generation uses register-domain **chaining** at KEYFRAME seams
(`design/generation/long_range_structure.md`). The arc measures *how much music one window buys*,
which improves both training signal and the inference prompt budget.

## Design

`generalize --tkvocab 0` (atoms-only v2) at increasing `seq_len`, **holding the effective batch
constant at 32** (reduce `--batch-size`, raise `--accumulate-grad-batches` as `seq_len` grows to fit
24 GB), **matched 100 epochs** (`--max-epochs`; early-stop never fires under schedule-free), 1 seed,
on the baked `:latest` (0.2.30 / tokens 0.51.0 / v2 codec — no `--bind-src`).

| seq_len | role | comparator |
|---|---|---|
| 8192 | **baseline (DONE)** — `/scratch/tmp/v2_atoms_baseline.ckpt` | content 0.505/0.552/0.485, bits/atom 1.998/2.058/2.272 |
| 12288 | 1.5× trend point | vs 8192 |
| 16384 | 2× — headline (likely-adopted config) | vs 8192 |
| 24576 | 3× — stretch (may need batch 1; robust-last) | vs 8192 |

Run order = decisive first (16384, 12288), stretch last (24576) so a 12 h window still yields the
headline + trend even if the stretch overruns or OOMs.

## Decision

Per subset, **full-eval** (the only decision-grade metrics; raw val_acc is a within-arm progress
signal only):
- **content-tier accuracy** (`stream.is_content_atom`) and **bits/canonical-atom** (total
  teacher-forced NLL ÷ canonical atom count). Audit: `/scratch/tmp/audit_ckpt.py <ckpt> <out.json>`
  on `:latest`.

Reads:
- **Monotonic content-tier ↑ (and/or bits/atom ↓) with `seq_len`** → context helps; adopt the
  longest length that fits + budget allows, then layer musically-aligned KEYFRAME windows (the other
  §-NEXT-1 sub-lever; a separate dataset-side dev item, not in this batch).
- **Flat / declining by 16384** → context is *not* the lever at this body/regime; the encoding-side
  arc is done — pivot to embedding/conditioning (NEXT #2). A real, publishable negative.

## Results (RAN 2026-06-13 — STEP-CONFOUNDED, verdict INCONCLUSIVE)

The sweep executed cleanly (3 trains + full-eval audits, no failures; artifacts
`data/audit/context_length_sweep_v2.md` + `ctx_audit_sl*.json`). Raw result — longer context is
**monotonically worse** on every decisive metric vs the v2-8192 baseline (content 0.505/0.552/0.485,
bits/atom 1.998/2.058/2.272):

| seq_len | steps | bits/atom (a/dag/fol) | content (a/dag/fol) | val_loss |
|---|---|---|---|---|
| 8192 | **10600** | 1.998/2.058/2.272 | 0.505/0.552/0.485 | 1.391 |
| 12288 | 7100 | 2.129/2.318/2.430 | 0.470/0.504/0.440 | 1.550 |
| 16384 | 5400 | 2.315/2.511/2.607 | 0.428/0.451/0.390 | 1.691 |
| 24576 | 3800 | 2.669/2.869/2.969 | 0.366/0.397/0.346 | 1.979 |

**But this is the §7C trap: matched EPOCHS ≠ matched STEPS.** Longer windows tile into far fewer
windows/tune (`block_stride = seq_len//4` also scales), so at 100 epochs the longer-`seq_len` arms did
**half (16384) to a third (24576) the optimizer steps**. Under schedule-free (no LR decay, more steps
= more convergence) they are simply **undertrained** — the higher val_loss confirms it, and the
degradation tracks the step deficit. The "Effective batch held at 32" risk-note above guarded the
batch dimension but **missed the step dimension**: matched epochs holds tokens-seen + effective batch
constant but NOT the number of updates. So the sweep conflates context with optimization budget and
**cannot conclude context hurts.**

**Decisive re-do = match STEPS, not epochs.** Scale `--max-epochs` so each arm reaches ~10600 steps:
12288→~149, 16384→~196, 24576→~279 ep. The single clean headline is **16384 @ ~196 ep (~7 h) vs the
8192 baseline**. A `--max-steps` cap (cleaner than epoch-scaling) is worth adding to the trainer.

**But a stronger reason this lever is unpromising (added 2026-06-14):** `effective_context_audit` on
the v2 baseline shows teacher-forced accuracy **saturates at k≈1024 atoms** (acc 0.35→0.58, flat past
1024; `data/audit/effective_context_audit_v2.json`). The model already uses only ~1024 of its 8192
window — there is no exploitable long-range signal beyond ~1/8 of the *current* `seq_len`, so longer
windows are doubly unlikely to help (step-confound AND no long-range dependency to exploit).
**Reprioritized:** the matched-steps re-run drops below the free-running-pathology arc
(`../generation/free_running_pathology_remediation_design.md`) — the binding constraint is generation
collapse (copy-dominance + exposure bias), not window length. The real context lever is making
dependencies shorter/learnable (representation: lane-demux), not `seq_len`.

## Risks / watch

- **24 GB fit** at long `seq_len` — mitigated by the constant-effective-batch reduction + a 16384
  preflight (`ctx_preflight_16384`). 24576 may OOM at batch ≥2 → robust-last, batch continues on
  failure.
- **Per-epoch time grows ~with `seq_len`** (8192 ≈ 1.4 min/ep; 16384 ≈ 2–3.5×) — the budget reason
  the sweep is ordered decisive-first.
- **Effective batch held at 32** so the only variable is context — a longer window with a *smaller*
  effective batch would confound the read.
- Aligned-windows is **out of scope here** (needs the dataset-side structural-index windowing path);
  this batch is the raw-`seq_len` half of NEXT #1.
