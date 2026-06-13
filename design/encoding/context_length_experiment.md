# Context-length arc — does longer `seq_len` improve held-out continuation? (v2)

**Status: DESIGN + RUNNING (2026-06-13).** The next experiment after the v2 re-baseline; AGENTS NEXT
#1. Overnight `seq_len` sweep on the atoms-only v2 model, decided in bits/canonical-atom +
content-tier (full-eval). Batch: `/scratch/tmp/ctx_overnight_batch.sh`.

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
