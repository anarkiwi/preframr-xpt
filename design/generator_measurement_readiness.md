# Generator-MDL measurement & experiment readiness — what to run when the pipeline lands

**Status:** Active plan (2026-06-06). The generator-MDL pipeline **has landed on `preframr-tokens`
`origin/main`** (PR #62/#64/#65/#66/#67/#68): `generator_pass` is the deployed default in
`REGISTERED_MACROS`, the per-pass macro zoo (`skeleton`/`freq_trajectory`/`sweep`/`gradient`/`global_osc`/
`preset`/`stamp`/`wavetable`/`per_reg_burst`/`note_off`/`init`) is **deleted**, ops `GEN_TRI=83`/
`GEN_TUNING=84`/`GEN_TABLE_{DEF=85,STEP=86,END=87,REF=88}` + reused `SWEEP_OP=64` are live. It is **NOT yet
released to PyPI** — latest is `preframr-tokens 0.44.0`; the generator landing is held on `main` to bundle
whole-chip-zero into one breaking **0.45.0** (memory `tokens-0.45.0-release-pending`,
`cross-repo-release-ordering`).

This doc is the readiness register: which measurements are runnable **now** (cheap, against local `main`
source) vs **gated on the 0.45.0 release + image rebuild + dataset re-cut** (the decisive training run). It
makes concrete the "NEXT" block in `AGENTS.md`. Cross-ref `learnability_token_ordering_theory.md` (why
copy-fraction is the lever), `generator_mdl_representation.md` (the encoding under test),
`verification_and_audits.md` (correctness gates — orthogonal to learnability).

## Two release states gate two classes of measurement

| measurement | needs | runnable now? |
|---|---|---|
| **Static learnability triage** (the cheap go/no-go) | local `main` source on `PYTHONPATH` | **YES** |
| **Op-distribution / content-tier wiring** | local `main` source + `op_name_by_id()` | **YES** (op→tier map) / training output (the report) |
| **Residual-in-key refragmentation** | local `main` source | **YES** |
| **Canonical-tier learnability A/B** (decisive) | 0.45.0 on PyPI → xpt floor + image rebuild → dataset re-cut | **NO — release-gated** |
| **Generalization metric automation** | the canonical run's checkpoints | **NO — follows the run** |

The cheap reads run against **local tokens `main`** with `PYTHONPATH=/scratch/anarkiwi/preframr-tokens`
inside the xpt image (no PyPI release needed — that is the whole point of the triage being static + minutes).
The training run must run on **released, baked** code (xpt convention: arms run the baked image, not the
working tree), so it waits on 0.45.0.

## 1. Static learnability triage — the cheap go/no-go (RUN FIRST, runnable now)

The single decisive cheap read: **does the generator encoding's in-block induction-copy rate beat the old
codebook arm's recorded 0.718, at the block scale the model actually sees?** Copy-fraction (not gzip/MDL) is
the learnability lever (`learnability_token_ordering_theory.md`); the generator design's bet is that
provenance-invariant DEF→REF generators + a transposition-invariant pitch LUT are induction-head-friendly.

Tool: `preframr_experiments/audit/learnability_triage.py` (training-free; h_k entropy-rate, MI-decay,
induction-copy). Run on **fogbank**, in the xpt image, `PYTHONPATH` to local tokens `main`:

```
python3 -m preframr_experiments.audit.learnability_triage \
  --configs baseline,full_macros --mode blocks --seq-len 8192 \
  --dumps <digi-excluded *.dump.parquet sample>
```

**Reading it:**
- `--mode blocks` is mandatory — it measures the **self-contained block stream the model trains/predicts on**
  (`iter_self_contained_row_blocks` → slice → re-encode), not the full-song parse (`--mode song` over-credits
  codebook compression that accumulates over a whole song and does not survive to block scale).
- `full_macros` now resolves dynamically to `REGISTERED_MACROS` = **the generator encoding**. `baseline` = the
  atomic control. The old **`codebook`/`skeleton` presets are dead** — their flags
  (`skeleton_pass`/`stamp`/`wavetable`/`patch`/…) are deleted on `main`, so `_config_flags` drops them all and
  the arm collapses toward baseline (the tool degrades gracefully, prints "dropped flags unavailable", does
  not crash). Do **not** trust a `codebook` arm number from this run — compare the generator's induction-copy
  against the **historical recorded 0.718** instead.
- **GO** if generator (`full_macros`) shows: induction-copy > ~0.718 **and** lower per-frame `h_∞` (entropy
  floor) **and** earlier `h_k` plateau than baseline. **NO-GO / investigate** if induction-copy regresses to
  baseline — that would mean the generator atoms fragment the in-block bigram statistics (the refragmentation
  risk, §3).
- Use the **same digi-excluded dump sample** the residual census uses (filter by parquet row count; digis
  hammer ctrl regs and manufacture false bottlenecks — memory `log-progress-in-sweep-tools`).

This costs minutes and is the queue-or-not signal for the expensive canonical run.

## 2. Op-distribution + content-tier wiring (runnable now / partial)

Confirm the deployed stream is generator atoms with raw `SET` ≈ 0, and that the op→tier map covers the new ops.

- **op→tier map** (runnable now): `tier_map.build_op_map` reads op→name from tokens `op_name_by_id()`. With
  the new ops live (`GEN_*`, reused `SWEEP_OP`), confirm every generator op resolves to a tier and none falls
  through to "unknown". The old subsumed-pass op-ids are now holes — confirm nothing in xpt still references
  them by id (grep specs/audits for retired `*_OP` names; renames silently disable, per `AGENTS.md`).
- **content-tier report** (needs a training run's `audit_per_class.json`): `audit_checkpoint_per_class`
  (emits `vocab_atom`) → `content_tier_report --results-root <dir>`. This is the **de-confounder** —
  all-tier `val_acc` is not comparable across tokenizations; `content_over_structural` + per-op `op_acc`
  is. Wire it as part of the canonical run (§4), not before.
- **Expectation by construction:** PW/cutoff/res/modevol are ordinary generator channels
  (HOLD/ACCUM/TRI/TABLE), so the old `PWM_PRESET`/`FC_PRESET` +16/+19/+6pp blowup is gone — verify it is
  absent in the op distribution, don't re-litigate it.

## 3. Residual-in-key refragmentation (cheap risk, runnable now)

The generator's freq `GEN_TABLE` key carries exact residuals (`("note", offsets, residuals)`); a worry is that
near-identical gestures with tiny residual differences mint distinct DEF ids, **fragmenting** the DEF→REF
reuse the learnability bet depends on. Measure directly: tokenize the digi-excluded sample under `full_macros`,
histogram DEF-id reuse counts (REF-per-DEF) and the fraction of DEFs that are singletons; compare to the
exact-recurrence rates the design claimed (instrument program 98% exact-recurrence). High singleton fraction →
the residual belongs in a separate low-order channel, not the codebook key. This is a static corpus pass
(no training); fold it into the same triage sweep.

## 4. Canonical-tier learnability A/B — the decisive run (RELEASE-GATED)

Mini collapses regardless of vocab (`loop_collapse_rate` ~1.0); only canonical/prodlike settles whether the
generator vocab's **payload** learns. This is the real verdict and it is **not queued** — it waits on the
0.45.0 release.

**Arms** (target first, baseline last — `AGENTS.md` ordering):
1. **generator** — `Arm(macro_config="full_macros")` on released 0.45.0 (= the generator encoding).
2. **atomic baseline** — `Arm(baseline=True)` (all passes OFF; survives the release, always available).
3. **pinned old-full_macros** (optional A/B reference) — the pre-generator encoding. **This must be preserved
   by a git pin BEFORE relying on it**: the zoo is deleted on `main`, so reproducing the old encoding needs a
   tokens commit pin (last zoo-present commit is `056cf98`'s parent line / pre-#66) + its own image. If the
   2-arm generator-vs-atomic contrast is decisive, skip this third arm — don't rebuild a deleted pipeline just
   for symmetry.

**Tier:** `canonical` (the `generalize` spec is already `tier="canonical", seq_len=8192`). **Gate:** per-tier
`content_over_structural` + per-op `op_acc` from `content_tier_report` (NOT all-tier val_acc).
**Prediction:** generator's provenance-invariant DEF→REF + LUT pitch lift content-tier accuracy over atomic.

**Pre-run sequence (the release chain, all on fogbank per `release_build_cache.md`):**
1. Release tokens **0.45.0** to PyPI (`vX` tag → OIDC `release.yml`) — **only after** the 12-SID WAV audition
   gate passes (non-negotiable before flipping a default + re-cutting data).
2. Re-floor `preframr-tokens>=0.45.0` in **all** xpt req files; rebuild the xpt image on it.
3. Re-cut datasets (`PREFRAMR_DATASET_CACHE_DISABLE=1` not needed — but the op set shifted, so the dataset
   cache key changes; let it re-tokenize). parse+tokenize ~25 min/prodlike uncached.
4. Launch the canonical A/B (`nohup`+`disown`, `ScheduleWakeup` to check — don't poll). canonical
   ~60–120 min/arm.

## 5. Generalization metric automation (follows the run)

Wire the cross-composer `val_acc` (primary) + the generalization gate (`GENERALIZE_MIN_VAL_ACC`) read into the
canonical run's harness so the verdict is one command, not a manual audit. The `generalize` spec + `integration/
check_generalize.py` exist; the work is connecting `content_tier_report` + per-op `op_acc` into a single
pass/fail the run emits. Stretch: cross-engine eval-B families.

## 6. The melody caveat (DOES NOT come for free with the generator)

The generator pipeline makes the **representation lossless + induction-friendly**; it does **not** make
**melody** learnable. Melody generalization needs the **melody work order** (`melody_skeleton_impl.md`, the
self-directing `preframr-tokens/AGENT_TASK_melody_skeleton.md`): layer 2 interval-from-previous onset encoding
+ layer 3 `voice_lane` de-mux with **causal-DAG lane ordering** (accompaniment roles before melody). Measured
levers: harmony conditions the next melody interval (+0.294 bits), 63% of lead voices hop (role-id needed but
coarse). Two cheap risks to measure when that lands: lane-order variants (triage) + **no-other-content
regression** under de-mux. Exact next-note accuracy caps ~0.51 (the data ceiling, `melody_predictability`), so
the melody verdict needs **distributional/audition** scoring, not just argmax val_acc. Melody is a **separate
gate** sequenced after this one — do not read a flat melody-onset number off the generator run as a generator
failure.

## Runnable-now checklist (do these the moment fogbank is free; no release needed)
1. **Triage go/no-go** (§1): `learnability_triage --configs baseline,full_macros --mode blocks --seq-len 8192`
   on a digi-excluded sample → generator induction-copy vs 0.718 + baseline. **The queue-or-not signal.**
2. **Refragmentation** (§3): DEF-id reuse histogram under `full_macros` → singleton-DEF fraction.
3. **op→tier coverage** (§2): confirm every `GEN_*`/`SWEEP_OP` resolves in `tier_map.build_op_map`; grep xpt
   for dead retired `*_OP` references.

Everything in §4–§5 is **release-gated** and starts only after tokens 0.45.0 + the 12-SID audition + the image
rebuild.
