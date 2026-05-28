# Substrate ablation v1 — `melody_substrate_iter_mini` result

**Status:** Landed 2026-05-28. Three findings: substrate ablation works, macros add nothing on the clean substrate, core architecture can generalize. Next-step recipe at the bottom.

## Background

After months of model-side and tokenizer-side interventions, `full_macros_prodlike` was the only confirmed content-tier win (eval_a content 0.219 → 0.324, ×3 seeds). But the win was **SET-carried**: op45 (FREQ_TRAJ — the actual melody primitive) was ~0.026, V0 onset (the pitch atom) was 0.000. Mini couldn't move op45 in any A/B. Two architecturally-induced bugs were hiding the real picture:

- **`project-content-tier-per-op-broken`** — `content_tier_report.id_to_op` used tokens.csv row index as the unigram uid; ~58% mis-assigned, so every per-op number in the design history was unreliable. Fixed via vocab_atom emission in `audit_checkpoint_per_class`; consumer reads sidecar.
- **`project-expand-literal-strips-freq-passes`** — `iter_self_contained_row_blocks` decompiled FREQ_TRAJ back to op=0 SETs and `run_passes` re-applied only PW/filter/ctrl. Freq stack dropped pre-tokenize. Fixed by `run_freq_block_passes` (commit `a71f676 freq_passes_re_fire_on_blocks`).

With both bugs fixed, the question reopened: is the SID melody-learning failure a model-architecture capacity issue, an encoding issue, or a substrate-noise issue?

## Experiments

Three diagnostics, all on mini body=large (layers=6 heads=8 kv_heads=4 embed=288, 5.5M params).

### 1. `framework_arch_test` — architecture can generalize

`preframr_experiments.audit.framework_arch_test` trains the torchtune `llama3_2` body on a synthetic deterministic-motif task. Vocab 64, 8 motifs total (6 shared between train + val, 2 held-out-only in val), within-motif rule `token[i+1] = token[i] + 1`. Train 1024 / val 256 / 20 epochs.

**Result:** train_inside_acc 1.000, val_inside_acc 0.903 on UNSEEN held-out motifs, gap 0.097. **Architecture can generalize.** SID failure is downstream of the model body.

### 2. `freq_core_ablation_mini` re-read

Re-ran the audit with the vocab_atom fix on the existing seed0/1/2 ckpts. Verdict flipped from "mini capacity, not melody" (under the broken row-index reader) to: substrate ablation lifts content acc **0.042 → 0.173** (Δ +0.131) and content/structural ratio 0.105 → 1.781 (17×) — but with op45=0 because the spec never enabled the freq stack flags.

### 3. `melody_substrate_iter_mini` — the program-defining A/B

3 arms × 3 seeds at mini body=large, 60 epochs. Each arm runs anchor + interval-V0 + freq-onset-pass; arms differ on (a) substrate ablation and (b) absorber macros:

| arm | ablation | macros |
|---|---|---|
| `substrate_no_macros` | PW=50% + filter dropped | OFF |
| `substrate_full_macros` | PW=50% + filter dropped | ON |
| `baseline_full_stack` | none | ON |

**Result (audit_checkpoint_per_class with vocab_atom; content_tier_report --onset):**

| arm | content acc | op45 (FREQ_TRAJ) | V0 onset |
|---|---|---|---|
| `substrate_no_macros` | **0.089** ±0.001 | **0.206** (43% of cont pos) | 0.008 |
| `substrate_full_macros` | **0.089** ±0.001 | **0.202** (44%) | 0.011 |
| `baseline_full_stack` | 0.056 ±0.001 | 0.085 (63%) | 0.009 |

**op45 (FREQ_TRAJ) jumps 0.085 → 0.206 — ~2.4× lift on the melody primitive's structural framing.** Seed-stable across 3 seeds.

The lift is in the trajectory header tokens. V0 onset (pitch content) stays near zero across all three arms.

### 4. Audition cohort — 12 predictions

Ran `preframr_experiments.audit.melody_compare_arms` on `--predict-dump` output of each arm × seed0 × 4 val prompts, scored against a 4359-tune prodlike-train baseline (`melody_baseline_corpus`). All 12 predictions PASS verdict (no collapse). Per-arm headline (0..1, higher = closer to training distribution):

```
substrate_no_macros    0.77 ± 0.04   ← best
baseline_full_stack    0.75 ± 0.03
substrate_full_macros  0.73 ± 0.08
```

`substrate_no_macros` is closest to corpus on pitched_gates_per_sec, median_note_frames, pitch_in_scale_rate. All arms are still ~10× too sparse vs the corpus (13–23 gates/sec vs 174).

## Verdicts

1. **Substrate ablation works.** Content acc +60% relative, op45 +140% relative, seed-stable. The substrate (freq + control + ADSR + frame, no PW/filter dynamics) is the right learning surface.
2. **Absorber macros add nothing on the clean substrate.** `substrate_no_macros` ties `substrate_full_macros` on every content metric, and beats it on the audition headline + variance. The macros were absorbing filter/PW SET noise that's now gone.
3. **The core architecture can generalize at mini scale.** SID failure is downstream of the body.
4. **What's still unlearned is V0 absolute pitch.** Trajectory framing learns. Pitch content doesn't. Interval-V0 encoding (`--freq-v0-interval`) was enabled in all substrate arms — it hasn't moved V0-onset acc at mini.

## Next steps (in priority order)

1. **Scale `substrate_no_macros` to prodlike.** Cleanest baseline we have; macros add nothing so the simpler spec wins. Use deployment config (`--tkvocab 8192 --batch-size 4 --accumulate-grad-batches 8`). Reuse the `preframr-aug`-free ablation hook.
2. **Probe V0 predictability ceiling on the ablated corpus** via `audit.melody_predictability`. If V0 trigram entropy was 0.79 on the noisy corpus, expect it higher on substrate. If the ceiling rises, scale is the only remaining lever for V0. If flat, the problem is in V0's *encoding*, not data.
3. **Question the interval-V0 representation.** Verify the flag fired (grep `pipeline_spec.json` per arm-seed). If interval is on and V0 is still 0, the issue may be that the interval is computed per-block (resets at boundaries) — investigate.
4. **Skip the absorber macros in any further substrate work.** They're load-bearing on noisy data, dead weight on clean data.

## Artifacts

- `preframr_experiments/specs/melody_substrate_iter_mini.py` — the 3-arm spec.
- `preframr_experiments/audit/ablate_pwfilter.py` — the substrate ablation hook.
- `preframr_experiments/audit/melody_features.py` — SID dump → muspy-backed feature vector.
- `preframr_experiments/audit/melody_baseline_corpus.py` — corpus-level feature baseline builder (4437 tunes ~minutes).
- `preframr_experiments/audit/melody_score_generation.py` — z-score vs baseline + PASS/WARN/FAIL/COLLAPSE verdict.
- `preframr_experiments/audit/melody_compare_arms.py` — cross-arm comparative scoring.
- `preframr_experiments/audit/framework_arch_test.py` + `tests/test_framework_arch.py` — architecture sanity diagnostic.
- `preframr_experiments/audit/audit_checkpoint_per_class.py` — emits `vocab_atom` sidecar (fixes the broken per-op assignment).
- `preframr_experiments/audit/content_tier_report.py` — prefers `vocab_atom` sidecar, falls back to row-index with RuntimeWarning.
- `tests/test_melody_features.py` + `tests/test_content_tier_report.py` — gate the above.

Framework-side patch (separate commit in `preframr`): `--predict-dump <path>` flag emits the prediction-window audio_df as parquet for automated scoring.
