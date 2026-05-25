# repo focus cleanup — scope (2026-05-25)

**Goal:** keep `preframr` (main repo) to the **core framework** — the trainable/
inferable package + its unit tests + build. Move experiment orchestration,
dataset/corpus curation ("pickers"), research audits, and experiment data to the
sibling `preframr-xpt`. This is scope only; execution deferred (see Sequencing —
do not disrupt the in-flight re-arc).

## Current weight (tracked files)
`integration_tests/` = **293** files vs `preframr/` 25, `tests/` 31. So the
experiment surface dominates the "framework" repo — exactly what this fixes.
`integration_tests/data/` is 211 files (195 in `audit/` = generated artefacts).
Of the experiment `.py`: **20 import preframr/torch (coupled), 14 pure.**

## Classification

### STAYS (core framework)
- `preframr/` package (train/, inference/, args, parse, stftokenize, utils).
- `tests/` — framework unit tests (pass for any experiment-only ones; most stay).
- Build/CI: `Dockerfile*`, `build.sh`, `run_tests.sh`, `requirements*.txt`,
  `.github/`, `pyrightconfig.json`, `.coveragerc`, README, LICENSE, `.env.example`.
- Top-level framework CLI shims: `train.sh tokenize.sh parse.sh render*.sh
  predict-*.sh tb.sh retrain.sh`.

### MOVES → preframr-xpt (clean — pure orchestration/curation)
- **Pickers / curation:** `pick_*.py`, `build_prodlike_4x_list.py`,
  `build_content_clusters.py`, `aggregate_corpus_index.py`.
- **Int-test + eval runners:** `run_*_int_test.sh`, `int_test_common.sh`,
  `run_eval_per_composer_8k.py`, `eval_per_composer.py`, `check_generalize.py`,
  `run_orinnx_render_smoke.sh`, `cross_composer_encoding_audit.py`.
- The 14 **pure** experiment `.py` (no preframr/torch).

### MOVES → preframr-xpt, BUT framework-coupled (the hard 20 — THE decision)
`integration_tests/profile/` audits + probes that `import preframr`/`torch`:
`audit_checkpoint_per_class`, `per_class_acc_audit`, `loop_detection_audit`,
`prompt_conditioning_audit`, `generate_for_audit`, `ddp_scaling`, the `augment_*`
probes, etc. These are research tooling (belong in xpt) but violate xpt's
**"pure orchestration: no preframr/torch imports"** charter.
**RESOLVED (2026-05-25) — layered image.** Added `preframr-xpt/Dockerfile`
(`FROM anarkiwi/preframr`): the experiment image layers the runner + audits on
the framework image, so coupled audits `import preframr`+torch from the base and
**run + are tested inside `anarkiwi/preframr-xpt`**. Verified: editable
`--no-deps` install, 63 xpt tests pass, `preframr`+`preframr_experiments` import.
The runner core stays pure host orchestration; the no-preframr-import charter is
scoped to it, and audits move into `preframr_experiments` (e.g. an `audit`
subpackage) where they're covered by this image. Main repo keeps only the
framework. (Superseded options: relax-charter-in-place; keep audits in main.)

### Data + design
- `integration_tests/data/audit/` (195) = **generated** per-experiment artefacts
  → xpt, and consider gitignoring/regenerating rather than tracking 195 blobs.
- `data/{prodlike,mini,canonical}/` tier metadata + `content_clusters/` → xpt
  (the `.list` files + `HVSC_VERSION` already live there).
- `design/` split: **framework-architecture** docs stay (audio_driver_split,
  train_inference_split, model_regdataset_decomposition, preframr_tokens_extraction,
  constrained_decode_torch_free, audio_fidelity_helper); **research/experiment**
  docs → xpt (per_tier_heads, content_diffusion, cluster_*, melody_transfer,
  multi_modal, orin_inference, prodlike_tier, this scope doc, …). Decision needed.

## Cross-repo edges / risks
- **Build + run_tests:** `Dockerfile` COPYs `integration_tests/` and `--help`-smokes
  profile scripts; `run_tests.sh` lints/tests it. Both must drop those refs so the
  framework image/build no longer carries experiment tooling. Audits then run from
  the **predict image** (they already need it) invoked by the xpt runner.
- **PREFRAMR_SRC_DIR** (xpt → main `preframr/` source) is unchanged — the single
  intended cross-repo edge. Coupled audits `import preframr` at runtime *in-image*
  (baked), not from xpt source, so the move is import-safe.
- **`tests/`** referencing `integration_tests/` fixtures/data must be repointed or
  the data moved with them.
- **In-flight re-arc** uses the frozen baked image (has `integration_tests/`); the
  monitor's content audit runs `integration_tests/profile/audit_checkpoint_per_class`
  from that image. Moving + rebaking mid-arc would desync the runner's audit path.

## Sequencing (proposed)
1. **After the re-arc completes** (avoid disrupting in-flight monitoring/audits).
2. Land Option A in xpt (audit subpackage + charter note) — move the coupled audits
   + pure scripts + pickers + data + experiment design docs.
3. Update main `Dockerfile`/`run_tests.sh` to drop `integration_tests/`; repoint the
   xpt runner to invoke audits from the predict image; update `tests/` refs.
4. Rebake; smoke the runner end-to-end (one mini spec + one audit) before relying on it.

## Migration follow-ups (post-re-arc)
- Move coupled audits into `preframr_experiments/audit/` + the pure scripts +
  pickers + experiment data; runner invokes them via `docker run
  anarkiwi/preframr-xpt -m preframr_experiments.audit.<x>` (instead of the
  framework image). Add an xpt `build.sh`/CI for the new image.
- Drop `integration_tests/` from the main `Dockerfile` + `run_tests.sh`.

## Decisions for the user
1. **xpt purity:** RESOLVED via the layered image (above).
2. **design/ docs:** split (framework stays / research moves) or move all to xpt?
3. **data/audit/ (195 generated):** move-and-track, or gitignore + regenerate?

## Executed (2026-05-25, during the live re-arc)
Moved out (safe — no runtime/build/test dep, untouched by the arc):
- **design/** (42 docs) → preframr-xpt/design/
- **integration_tests/data/audit/** (195 generated artefacts) → preframr-xpt/data/audit/
- **top-level integration_tests scripts** (12: pickers, int-test runners, eval/
  curation) → preframr-xpt/integration/
Main `integration_tests/` 293 → **45 files** (profile/ 26 + data/ 16 + fixtures/ 3).
Verified: no remaining import references a moved file; black clean.

**Deferred to post-arc** (entangled — the live mini stage reads
`data/content_clusters/` + `profile/augment_voice_permutation.py` host-side; the
monitor still runs `profile/audit_checkpoint_per_class`; `Dockerfile.predict`
smokes 4 profile audits; 5 `tests/` import profile audits):
- `profile/` audits → `preframr_experiments/audit/` (run via the layered image).
- Move the 5 audit-testing `tests/` to xpt.
- Drop `integration_tests/` from main `Dockerfile`/`Dockerfile.predict`/`run_tests.sh`.
- `data/` tier metadata → xpt; `fixtures/` follow their tests.
- Then rebake + smoke the runner+audit path end-to-end before relying on it.
