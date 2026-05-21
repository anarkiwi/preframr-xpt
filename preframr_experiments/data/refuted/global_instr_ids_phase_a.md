# `global_instr_ids` Phase A — REFUTED (2026-05-17)

**Status:** refuted at mini A/B (rerun). Default
`--engine-fp-palette` stays OFF. Tier 1 #1 closed.

## Hypothesis

Engine-fingerprint-keyed canonical palette ids (`gi_on`,
`--engine-fp-palette`) replace per-dump slot atom noise with a
shared cross-composer slot vocabulary. The expectation was that
shared canonical ids would lift cross-composer `val_acc` while
holding alphabet flat (per-dump noise removed, canonical
sharing fills the same slots).

## Refutation

Mini m_large A/B rerun finished 2026-05-16 23:30 (after the
canonical-palette bind-mount fix `634487e` made the palette
actually engage inside the training container).

| arm | alphabet | val_acc_at_best_loss | val_loss_best | epochs |
|---|---|---|---|---|
| `gi_off` (baseline) | 17,804 | 0.0647 ± 0.0024 | 10.72 ± 0.08 | 80 (clipped) |
| `gi_on` | 21,249 | 0.0813 ± 0.0018 | 10.96 ± 0.27 | 80 (clipped) |
| Δ | **+3,445 (+19.4%)** | +0.0166 (~7σ) | +0.234 worse | — |

Encodability is **identical** across all 8 eval subsets
(overall 0.0408 on both arms; cross-engine Eval-B all zero
except mibri 0.0719 on both). The canonical palette adds slot
atoms without expanding what the encoder can cover.

**Fast-fail trigger:** alphabet bloat (+19%) breaches the
prodlike envelope constraint. Jetson predict-host feasibility
("Smaller `tkvocab` is the cleanest full-context Jetson
unblocker", AGENTS.md) is a hard ceiling, not a soft cost.
A 19% alphabet increase for a same-encodability +7σ val_acc
lift is buying capacity the deployment target can't carry.

The val_loss-vs-val_acc split (val_acc up, val_loss worse) is
also a red flag — the LM is getting more confident on a
narrower subset of slots while losing on the long tail.

## Evidence

- `/scratch/tmp/preframr_experiments/results/global_instr_ids_phase_a/report.md`
  — rerun A/B numbers.
- `/scratch/tmp/preframr_experiments/results/global_instr_ids_phase_a/rerun.log`
  — 2026-05-16 20:44 → 23:30 wallclock.
- `integration_tests/design/global_instr_ids_phase_a_verdict.md`
  — verdict template (first-run interpretation with
  canonical-palette silently no-op'd).
- `1a35031` bit-budget audit — Phase B+ would need ≥13-bit
  slot ids (14-bit recommended); canonical 12-bit budget
  already breached by mini clusters 4+7.
- `701a3ae` cluster coverage — canonical Eval-B daglish/follin
  is 100% OOD vs the palette anchor set; canonical tier can't
  test the cross-composer-sharing hypothesis at all.

## Phase B+ status

Phase B+ (the multi-engine canonical-palette extension) was
pre-blocked by the bit-budget audit (need 14-bit slot ids) and
by canonical Eval-B coverage (100% OOD). With Phase A refuted
at mini, the prodlike escalation that would have tested
canonical sharing in a corpus where the palette anchors are
in-distribution is also off the table.

## Do not revisit without

- A canonical-palette design that **holds alphabet flat** (the
  win must come from re-allocating existing slot ids, not
  adding new ones), AND
- A corpus tier where the palette anchors are in-distribution
  for at least one Eval-B family (canonical isn't; prodlike
  may be — needs cluster-coverage audit before any rerun), AND
- A bit-budget plan that fits inside 12-bit slot ids without
  losing rows (per `1a35031`).
