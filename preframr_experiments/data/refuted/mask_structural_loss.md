# `mask_structural_loss` (option 1 separator probe at mini) — REFUTED 2026-05-22

Single-arm probe at mini, body=large, 3 seeds: plain CE single-head
+ `--mask-structural-tier-loss` (zero CE loss contribution from
positions where the target token is in the structural tier). Spec
`preframr-xpt:preframr_experiments/specs/mask_structural_loss_mini.py`.

This was the option-1 framing for the per_tier_heads prodlike refute:
**hypothesis** — the prodlike failure mode (router saturation, prompt
conditioning collapse) is downstream of structural-tier targets being
the easiest to predict and dominating the CE gradient. If hard-masking
structural targets on the SIMPLER plain-CE architecture restored
diversity_ratio at mini, the per_tier_heads architecture is the wrong
abstraction and a 1-line training-side fix is sufficient. If masking
didn't help, the failure mode is downstream of the router architecture
itself, not gradient dominance, and the entropy regularisation framing
(per_tier_heads_mos_revisited.md "Open questions" → tested in
per_tier_heads_entropy_retest, PASSED at mini) is load-bearing.

## Gate result

| Criterion | Value | Verdict |
|---|---|---|
| loop_collapse_rate at T=0.5 ≤ baseline | 0/12 (0%) vs baseline 0/12 | **PASS** |
| diversity_ratio at T=0.5 > 1.2 | **0.863** vs baseline 1.220, entropy 1.535 | **FAIL** (below 1.0) |
| content acc not regressed vs baseline | 0.0091 vs baseline 0.0001 | PASS (data-floored at mini) |

3-seed val_acc 0.0155 ± 0.0013 (vs baseline 0.0832 ± 0.0074); this is
EXPECTED — the unmasked val_loss / val_acc is now computed on a model
whose training never saw a structural-tier gradient, so it has no
prediction mass on the 23% of positions that are structural. Val_acc
is not the gate; content acc + diversity_ratio are.

Audit artefacts: `integration_tests/data/audit/mask_structural_loss_mini/`
(audit_checkpoint_per_class_seed0.json, loop_detection_mask_T0.5.json,
prompt_conditioning_mask_T0.5.json, 12 stream CSVs).

## Failure-mode interpretation

The diversity_ratio at 0.863 is **below 1.0** — meaning real-prompt
outputs are LESS diverse than random-prompt outputs:

| metric | masked | lambda=0 baseline | entropy lambda=0.01 |
|---|---|---|---|
| mean_jaccard real | 0.469 | 0.119 | 0.124 |
| mean_jaccard random | 0.384 | 0.371 | 0.429 |
| diversity_ratio | **0.863** | 1.220 | 1.535 |

Both real and random outputs are 3-4× more self-similar than baseline.
And real prompts produce MORE similar outputs to each other than random
prompts do.

Read directly: masking structural targets collapsed the model to a
narrow content-attractor — a learned "default sequence" the model
emits regardless of prompt content. Real prompts trigger this
attractor MORE reliably than random prompts (real prompts match the
training distribution, the attractor is shaped by it), driving
diversity_ratio below 1.

Mechanism: **structural supervision was load-bearing for
prompt-conditioning.** Structural tokens encode where-in-song
positional information (FRAME boundary, VOICE selector, IRQ tick
phase). Supervising those provided the model with positional anchors
it used to interpret content prompts. Without that signal, the model
has no place-in-song reference and can't differentiate prompts.

## What this changes

1. **Refutes the gradient-dominance framing.** The prodlike failure
   mode is NOT "structural tokens are easy, gradient comes from them,
   content gets starved." If it were, masking structural would have
   helped. It made things worse.
2. **Confirms the router-architecture framing.** The entropy retest
   at mini already PASSED on the exact same gate criteria
   (`per_tier_heads_entropy_retest.py`, lambda=0.01: diversity_ratio
   1.535). The contrast is sharp: same data, same body, same training
   length — entropy regularisation lifts diversity_ratio from 1.22
   to 1.535; loss masking drops it from 1.22 to 0.863.
3. **Drops the masked-loss thread.** No prodlike retest; the mini
   refute is decisive. The `--mask-structural-tier-loss` flag stays
   in the codebase as a research artefact but defaults False and is
   not part of the recommended training path.

## Open questions (not blocking)

- **Would masking structural + mid (keep only content) be a different
  experiment?** That extreme variant might reveal whether ANY
  structural-like supervision matters or whether the issue is
  specifically the lowest tier. Not worth running standalone — bake
  into a future cluster-conditional content head probe if needed.
- **Does the model's collapsed "default sequence" tell us anything
  about which content tokens it has memorized?** Inspection of the
  generated streams might reveal which content tokens dominate the
  attractor. Cheap to investigate post-hoc.

## References

- Original refute: `per_tier_heads_mos_prodlike.md` (this directory).
- Companion confirming experiment:
  `preframr-xpt:preframr_experiments/specs/per_tier_heads_entropy_retest.py`.
- Open-questions origin:
  `per_tier_heads_mos_revisited.md` "Open questions".
- Design doc:
  `../../../../preframr/integration_tests/design/per_tier_heads_design.md`
  (main repo).
