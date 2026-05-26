# Landed designs (archive)

Reference-only. Implementation has shipped to HEAD or to the
sibling `preframr-{audio,tokens,experiments}` repos. Kept for
historical context; consult git log for the actual commits.

## Library extractions

| Doc | What landed |
|---|---|
| [`preframr_tokens_extraction_design.md`](preframr_tokens_extraction_design.md) | `preframr-tokens` PyPI library (reglog parsing + tokenisation + macros + stfconstants). |
| [`preframr_tokens_corpus_class_design.md`](preframr_tokens_corpus_class_design.md) | `preframr_tokens.corpus.Corpus` (~560 LoC extracted from RegDataset). |
| [`constrained_decode_torch_free_design.md`](constrained_decode_torch_free_design.md) | `preframr_tokens.constrained_decode` numpy-mask + boundary `masked_fill`. |
| [`experiments_extraction_design.md`](experiments_extraction_design.md) | `preframr-experiments` sibling repo (runner + specs + tier lists + refuted registry) at `/scratch/anarkiwi/preframr-xpt`. |
| [`model_regdataset_decomposition_design.md`](model_regdataset_decomposition_design.md) | `preframr/train/model/` subpackage + `regdataset.py` slimmed to ~210 LoC adapter. |
| [`train_inference_split_design.md`](train_inference_split_design.md) | `preframr/train/` vs `preframr/inference/` bind-mount split; `Dockerfile.predict` slim eval image. |

## Data tiers + infrastructure

| Doc | What landed |
|---|---|
| [`prodlike_tier_design.md`](prodlike_tier_design.md) | Prodlike data tier (~4.4K train + 385 Eval-A + 8 Eval-B families); pinned in `preframr_experiments/data/prodlike/`. |
| [`hvsc_version_pinning_design.md`](hvsc_version_pinning_design.md) | Runner startup HVSC-version check via `DOCUMENTS/HVSC.txt`; wired into preflight. |
| [`engine_fingerprint_evalb_design.md`](engine_fingerprint_evalb_design.md) | 8 cross-engine Eval-B families pinned in prodlike spec; engine-family map JSON. |
| [`corpus_structural_index_design.md`](corpus_structural_index_design.md) | One-shot CPU audit emitting structural index of full HVSC; output at `data/audit/`. |
| [`stage_dumps_basename_fix_design.md`](stage_dumps_basename_fix_design.md) | Composer-subdir staging in `stage_dumps`; recovers 50 dumps lost to basename collisions in prodlike. |

## Audit + validation tooling

| Doc | What landed |
|---|---|
| [`alphabet_cooccurrence_audit_design.md`](alphabet_cooccurrence_audit_design.md) | `profile/alphabet_cooccurrence_audit.py` for per-tier co-occurrence checks. |
| [`audio_fidelity_helper_design.md`](audio_fidelity_helper_design.md) | Shared render-and-compare helper. Originally landed at `integration_tests/audio_fidelity.py`; subsequently moved to `preframr-audio/preframr_audio/fidelity.py` (where it sits next to the renderer it depends on). |
| [`encodability_metric_design.md`](encodability_metric_design.md) | Per-cluster Eval-B encodability metric extractor. |
| [`orinnx_audition_design.md`](orinnx_audition_design.md) | Orin NX predict-host audition harness. |

## Bug fixes

| Doc | What landed |
|---|---|
| [`tokenizer_alphabet_coverage_bug.md`](tokenizer_alphabet_coverage_bug.md) | RegTokenizer alphabet-coverage bug (Int64 + relaxed substitution). |
