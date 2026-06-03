# RUNBOOK — residual-SET landing → release → census → learnability follow-ups

**Audience:** a fresh agent context with no memory of the session that wrote this. Execute the phases
in order. Each phase has a **GATE** (must pass to continue) and a **STOP** (surface to the human, do not
proceed). The irreversible steps (PyPI publish, Docker push, merge-to-main) are explicitly marked; the
human has pre-authorized proceeding through them **only when the phase GATE is green** — never on a
failing or ambiguous gate.

## 0. Orientation (read first, ~5 min)
- **What landed:** the `preframr-tokens` "residual-SET elimination" work order — new byte-exact macros
  draining every raw `SET` (unmodeled driver mechanism) to zero. Spec:
  `/scratch/anarkiwi/preframr-tokens/IMPLEMENT_residual_set_elimination.md`. Passes (PR1–PR6):
  `ctrl_osc`, `ctrl_wavetable`, `note_off`, envelope-bundling/`envelope_osc`, cold-start DEF-on-first,
  `init_preamble`. (Exact flag names: check what shipped — see Phase A.)
- **Goal of THIS runbook:** verify it → release `preframr-tokens` to PyPI → cross-repo release
  (framework image + xpt) → **prove `residual_SETs == 0` corpus-wide** → run the queued learnability reads.
- **Authoritative references (read the relevant one per phase, don't reinvent):**
  - `design/release_build_cache.md` — per-repo release procedure, proxpi cache, build/test commands, hosts.
  - AGENTS.md "Packages" / "Images" / cross-repo gotchas; memory `cross-repo-release-ordering`,
    `docker-build-cache`, `tokens-test-gate-xdist`.
  - `design/macro_learnability_risk_review.md` + `design/learnability_token_ordering_theory.md` — the
    learnability frame and the Phase E reads.
- **Repos (all under `/scratch/anarkiwi/`):** `preframr-tokens` (PyPI lib), `preframr` (framework + Docker
  app), `preframr-xpt` (this repo: runner/specs/audits). Non-GPU work runs on `fogbank` (`ssh fogbank`,
  shared `/scratch`, has the images). Use the baked `anarkiwi/preframr-xpt:<latest>` image + `PYTHONPATH`
  to local source for fast loops.
- **Key tools this runbook drives:**
  - `preframr-tokens/residual_mechanism.py` (the agent's definition-of-done classifier; untracked).
  - `preframr_experiments/audit/residual_set_census.py` (THIS repo; the corpus gate — parallel,
    digi-excluded, progress-logged; verdict `residual_SETs=0 dirty_tunes=0`).
  - `/scratch/preframr/cb_div_audit.py` (byte-exact corpus sweep; `parse_audit='raise'`).
  - `preframr_experiments/audit/learnability_triage.py` (Phase E).

## 1. DETECT DONE (precondition gate)
The tokens agent may have left work uncommitted, committed on a branch, or already merged. Establish state:
```
cd /scratch/anarkiwi/preframr-tokens
git log --oneline -8 ; git status -s ; git branch -vv
ls preframr_tokens/macros/ | grep -E 'ctrl_osc|ctrl_wavetable|note_off|envelope|init'
```
Then run the agent's own done-check on a small sample (fast):
```
# pick ~12 player-diverse non-digi dumps (e.g. Hubbard/Galway/DRAX/JCH); or use the census below at high --step
PYTHONPATH=. python residual_mechanism.py <dump1.parquet> ... | tail -30
```
- **GATE:** all PR artifacts present AND `residual_mechanism.py` reports ~0 residual on the sample AND the
  tokens suite passes (Phase A). If artifacts are missing or residual is materially non-zero, the agent is
  **not done** → **STOP**, re-arm the watcher (Appendix A), do not release.

## 2. PHASE A — Verify tokens (no side effects)
Fast loop = mount the tree into the baked test image (memory `tokens-test-gate-xdist`).
```
cd /scratch/anarkiwi/preframr-tokens
docker run --rm --network host -v $PWD:/tok -w /tok -e PYTHONPATH=/tok \
  anarkiwi/preframr-xpt:<latest> sh run_tests.sh        # black + pytest -n auto + lint; ~10-15s
```
Byte-exact corpus sweep (must be clean — no `SET` clobbers, no order-fidelity regressions):
```
docker run --rm --network host -v /scratch/anarkiwi/preframr-tokens:/tok -v /scratch/preframr:/scratch/preframr \
  -w /tok -e PYTHONPATH=/tok anarkiwi/preframr-xpt:<latest> python /scratch/preframr/cb_div_audit.py 20
```
Sample residual census (sanity, full corpus run is Phase D):
```
docker run --rm --network host -v /scratch/anarkiwi/preframr-tokens:/tok -v /scratch/anarkiwi/preframr-xpt:/xpt \
  -v /scratch/preframr:/scratch/preframr -w /xpt -e PYTHONPATH=/tok:/xpt anarkiwi/preframr-xpt:<latest> \
  python -m preframr_experiments.audit.residual_set_census --step 200 --workers 48
```
- **GATE:** `run_tests.sh` green; `cb_div_audit` DIRTY=0; census sample `residual_SETs=0` (ignore
  SUSPECTED-MISSED-DIGI outliers — exclude and re-run if they appear). **STOP** on any failure with the
  output; a non-zero residual bucket here is a real gap, not releasable.

## 3. PHASE B — Merge + release preframr-tokens  *(IRREVERSIBLE: PyPI publish)*
**Ordering rule (memory `cross-repo-release-ordering`): publish tokens to PyPI BEFORE framework floors the
new version.** tokens has NO image — release = git tag `v*` → `release.yml` (OIDC → PyPI).
1. Get the work onto `main`:
   - If still on a feature branch: open a PR and merge (repo uses PRs; or fast-forward to main if that's
     the established pattern — check `git log origin/main`). Stamp `CHANGELOG.md` `[Unreleased]` → the new
     version with the residual macros + the perf commits.
2. Determine the version: `git tag --list 'v*' | sort -V | tail` → next **minor** bump (0.43.0 → **v0.44.0**
   expected; the perf branch + residual macros are new features). Bump `fallback_version` in
   `pyproject.toml` to match.
3. Tag + push:
   ```
   git -C /scratch/anarkiwi/preframr-tokens tag v0.44.0 && git -C /scratch/anarkiwi/preframr-tokens push origin v0.44.0
   ```
4. Watch `release.yml` (`gh run watch` / `gh run list`); confirm the version is live on PyPI
   (`pip index versions preframr-tokens` or the PyPI page) before Phase C.
- **GATE:** tag's CI green AND the version resolves on PyPI. **STOP** if the publish fails (OIDC/env);
  do NOT floor the framework against an unpublished version (that's the classic failure — memory).
- **Decision (verify, don't assume):** the new residual passes are register-state/audio-exact but **NOT**
  raw-write-order-exact, so they belong OUT of `REGISTERED_MACROS` (research arm, like sweep/skeleton/stamp;
  they'd fail `test_register_order_fidelity` on `full_macros`). **Confirm the agent did NOT add them to
  `REGISTERED_MACROS`** (`grep -n REGISTERED_MACROS preframr_tokens/tokenizer_config.py`). If it did →
  **STOP** and surface (architecture call): either they pass order-fidelity, or they must come back out.

## 4. PHASE C — Cross-repo release  *(IRREVERSIBLE: Docker push, main merge)*
### C1 framework (`preframr`)
1. Floor the new tokens version in **all three** req files (memory: jetson's is the one that gets missed):
   `requirements.txt`, `predict-requirements.txt`, `jetson/predict-requirements.txt` → `preframr-tokens>=0.44.0`.
2. `./run_tests.sh` (black, pytest, pylint curated, pyright, coverage ≥77) — in the image or on fogbank.
   `tier_map.build_op_map` reads tokens `op_name_by_id`; macro flags derive from `macro_flag_names()`
   (auto-picks any new flags — no manual map). **STOP** on failure (a removed/renamed tokens `*_OP` breaks
   train tests AND `run_tests.sh` — memory).
3. Bump `VERSION`, commit to `main`. `release.yml` builds + pushes cuda/predict/xpu/jetson on main-push (or
   a `v*` tag). **Build locally in parallel with the push** (memory `release_build_cache`) so you don't wait
   on CI + a slow pull. Confirm `docker-test` + `docker-release` green.
### C2 xpt (this repo)
1. Floor not needed (xpt gets tokens via the image), but if any req pins it, bump to `>=0.44.0`.
2. **Specs:** `full_macros` is unchanged (residual passes are NOT in `REGISTERED_MACROS`), so no migration
   for existing arms. Verify all specs still resolve: `pytest tests` (or `test_experiment_spec.py`). If you
   want a residual arm as a spec, add one using `Arm(macro_flags=(<residual flags>))`.
3. Rebake the xpt image on the new framework base (`docker build -f Dockerfile .` runs `pytest tests`;
   `.github/workflows/docker.yml` on push). Tag/bump per `release_build_cache.md`.
- **GATE:** framework `run_tests.sh` + `docker-release` green; xpt `pytest tests` green + image rebaked.
  **STOP** on any red.

## 5. PHASE D — Corpus census = 0 (the acceptance of the whole work order)
Full-corpus residual census, digi-excluded, parallel, on `fogbank` (72 cores) against the **released**
tokens (or the merged tree):
```
ssh fogbank
docker run --rm --network host -v /scratch/anarkiwi/preframr-tokens:/tok -v /scratch/anarkiwi/preframr-xpt:/xpt \
  -v /scratch/preframr:/scratch/preframr -w /xpt -e PYTHONPATH=/tok:/xpt anarkiwi/preframr-xpt:<latest> \
  python -m preframr_experiments.audit.residual_set_census --step 10 --workers 64
```
- **GATE (work-order acceptance):** VERDICT `residual_SETs=0 dirty_tunes=0` (after excluding
  SUSPECTED-MISSED-DIGI outliers — investigate those separately; they're PWM digis `is_digi` misses, per
  memory `digi-detection`). Any surviving WORK-QUEUE bucket = a new unmodeled mechanism → **STOP**, report
  the surviving `(reg,subreg,val)` histogram (that's the next work item, not a result).
- Record the result in AGENTS.md "Resolved log" + move the residual design docs to `design/landed/` (set
  their `**Status:**` to landed; update `design/README.md` axis-2 row).

## 6. PHASE E — Learnability follow-ups (now unblocked)
These were waiting on the residual passes existing. Run them against the released tokens. **Cheap, static,
no training.** All in the xpt image, `PYTHONPATH=/tok:/xpt`, **`--seq-len 8192`** (prodlike scale; NOT mini).
1. **Codebook-vs-substrate decomposition** (settles "is the codebook design the problem, or the FREQ_TRAJ→
   skeleton substrate swap?"):
   ```
   python -m preframr_experiments.audit.learnability_triage --mode song \
     --configs full_macros,skeleton,codebook --seq-len 8192 --dumps <~12 player-diverse non-digi dumps>
   ```
   Read: `full_macros` vs `skeleton` = substrate effect; `skeleton` vs `codebook` = the DEF→REF codebook
   effect. Now the residual passes (`ctrl_osc`/`note_off`/`ctrl_wavetable`) resolve into `codebook` (they
   were auto-dropped pre-release). Report per-frame h∞ + induction-copy per config. (`--mode blocks` is the
   true stream but 5/9-tune partial — note coverage, treat as the scale check; the faithful version still
   needs the Corpus block-builder, see `macro_learnability_risk_review.md` priority 2.)
2. Update `design/learnability_token_ordering_theory.md` "First read" + the risk-review with the numbers.
- This phase has **no irreversible steps**; run freely and report.

## 7. Failure / STOP summary
STOP and surface to the human (do not proceed) on: missing PR artifacts or non-zero sample residual
(Phase 1); any test/audit red (A, C); residual passes added to `REGISTERED_MACROS` (B decision); PyPI
publish or image build failure (B, C); corpus census `residual_SETs != 0` after exclusions (D). Everything
else is autonomous.

## Appendix A — re-arm the done-watcher (for an agent that wants to be woken)
A background watcher that fires on a tokens commit or a long quiescence (so you re-check Phase 1):
```
/scratch/tmp/tokens_done_watch.sh        # QUIET=2700 bash ... & via Bash run_in_background
```
(See the session's `tokens_done_watch.sh`; it baselines HEAD + quiescence. On fire, run Phase 1; if not
done, re-arm.)

## Appendix B — version/branch facts to confirm at run time (don't trust stale values)
- tokens latest tag (`git tag --list 'v*' | sort -V | tail`); next = v0.44.0 expected.
- the branch the residual work is on (was `perf/byte-exact-pandas-hygiene` atop `c41543d` mid-session).
- framework current `VERSION`; the three req files' current floor.
- `anarkiwi/preframr-xpt:<latest>` tag actually present (`docker images | grep preframr-xpt`).
