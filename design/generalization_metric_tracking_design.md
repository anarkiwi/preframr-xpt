# Generalization metric tracking — decisive-audit gate, scorecard, ledger

**Status:** Drafted, pending impl (touches `run.py`/`base.py`/`report.py` — land
only when no run is in flight, per AGENTS.md mid-run-edit rule). Reuses existing
audit modules + the metric registry; this is **wiring, not new audits**.

## Problem

The program's North Star is generalization (content prediction across
composers/engines), but the metrics that measure it are not tracked
consistently:

1. **The decisive metric is run by hand.** All-tier `val_acc` is CONFOUNDED
   across tokenizations; only the content-tier `per_class` audit settles a
   representation A/B. It lives in `preframr_experiments/audit/` and is invoked
   manually — so AGENTS.md has to *remind* every reader to run it ("Always run it
   before calling a win"). The confound trap recurs.
2. **The generalization KPIs are scattered.** Per-eval_b-family acc, loop_collapse,
   prompt_conditioning live across `metrics.json` + separate audit JSONs; the
   report leads with 20 `val_loss` rows, not the cross-composer signal.
3. **Cross-run trends are narrative.** "Is this a win vs program history?" is
   answered from AGENTS prose, and confounded comparisons are caught by human
   discipline, not structurally.

## Design

### 1. Decisive-audit runner stage (highest value)

After each `(arm, seed)` finishes training, the runner optionally runs the
content-tier `audit_checkpoint_per_class` (and `prompt_conditioning_audit`,
`loop_detection_audit`) on the best checkpoint, inside the spec's image (the
audit container already needs preframr+torch). Results merge into `metrics.json`
as `content_acc_eval_a`, `content_acc_eval_b_<family>`, `loop_collapse`,
`prompt_conditioning`. Gated by a spec field (`decisive_audit: bool`, default ON
for representation-axis specs) so non-representation runs don't pay the GPU cost.
`report.py` promotes the content-tier rows to the headline and **labels all-tier
`val_acc` as "CONFOUNDED for representation A/Bs."** This makes the
un-confounded metric always-computed instead of always-remembered.

### 2. Generalization scorecard (report.py section)

A dedicated block at the top of the cross-arm report:

- content-tier `eval_a` acc (Δ vs baseline);
- **per-eval_b-family content acc + spread** (min/max/stdev across the 8 families
  — the cross-composer transfer signal; the `evalb_stratify` probe becomes this);
- `loop_collapse` / `prompt_conditioning` flags;
- tokenizer health (`longtail_frac`, `worst_family_longtail_frac`,
  `alphabet_size`, `encoded_tokens_per_song` — the registry metrics).

The family spread is the load-bearing read: a wide spread (e.g. STAGE 2's
0.245–0.556) says failure is engine-family-specific → targeted augmentation, not
an architectural gap.

### 3. Cross-run ledger keyed by tokenizer-hash

Append each `(spec, arm, seed)` summary to a tracked append-only JSONL
(`preframr_experiments/data/metrics_ledger.jsonl`) keyed by the **dataset-cache
hash** (already computed by the runner — e.g. `fdad35d6…`/`eda8b138…`, the
parse/tokenize fingerprint). A small `python -m preframr_experiments.ledger_query`
surfaces trends and **auto-flags any comparison across two different
tokenizer-hashes as CONFOUNDED**. This encodes the confound rule in data rather
than in a prose warning, and turns the AGENTS program-history ledger into
queryable records.

## Reuse / non-goals

- Reuses `audit_checkpoint_per_class`, `prompt_conditioning_audit`,
  `loop_detection_audit` (exist) + the `metrics.py` registry (extended in this
  branch with the tokenizer-health extractors) + `report.py`. No new audits.
- Tokenizer *atom* profiling stays in preframr-tokens (`tokenizer_profile`); the
  xpt extractors read run artifacts only — don't duplicate it.

## Lifecycle

Land items 1→3 in order; #1 is the highest value (kills the confound trap). All
three edit the runner/report, so schedule them for a window with **no run in
flight** (the mid-run-edit rule silently invalidates A/Bs). #1 (the
tokenizer-health metric extractors) already landed as the safe, additive first
step.
