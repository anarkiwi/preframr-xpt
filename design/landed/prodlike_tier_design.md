# `prodlike` tier — near-production scale-up for actionable macro A/Bs

## Why

Mini-tier results have repeatedly produced ambiguous reads (4 of 5
specs in the 2026-05-11 batch were null / inconclusive / no-fire). The
`loop_lookahead` win on mini is large (Δval_acc +0.0094 ≈ 10σ) but is
not actionable as a production-default flip because:

1. Mini @ m_large is capacity-bound (per `mini_capacity_diag`). la3
   shrinks the alphabet 18.6% — some of the val_acc lift is "easier
   prediction problem", not "richer encoding". We can't separate the
   two at the mini capacity ceiling.
2. Both arms reported `epochs_to_best_val_loss = 80.00 ± 0.0000`,
   suggesting the win was measured at the train budget edge.
3. Canonical (901 SIDs, 5 composers) is too narrow stylistically and
   too small to expose cross-composer transfer behaviour.

`prodlike` is the next-tier corpus + body + train regime: large enough
that a 10σ mini result either replicates or doesn't, broad enough that
per-composer Eval-B subsets are statistically meaningful, and slow
enough that we run it once a macro candidate looks promising on mini.

## What we get out

- **Capacity-independent** Δ on the loop_lookahead question.
- **Per-eval-subset** val_acc / val_loss (resolves a tracked pipeline
  hole — required for cross-composer transfer claims).
- **Reusable infra**: once the tier lands, queued specs
  (`hard_restart_ab` post-gate-fix, `palette_pwm`,
  `global_instr_ids` Phase A) all get prodlike A/Bs by spec change.

## Corpus

Pinned at `integration_tests/data/prodlike/`. Generated via the
existing `pick_multi_composer.py` (already supports `--train-top-n`
and per-composer cap).

| | Pinned | Notes |
|---|---|---|
| Train composers | 25 (top by HVSC `.dump.parquet` count) | Bayliss, Blues_Muz, DRAX, Merman, Bond_Alan, Tel_Jeroen, Whittaker, Electronic_Speech_Systems (gate-skipped), Hubbard, Cross_Saul, Ass_It, Jammer, JCH, Mermaid, Zyron, Galway, Laxity, SIDwave, Nagie, Detert, Nordischsound, Mitch_and_Dane, Leitch, Waz, Gilmore |
| Train SIDs | **4437** | `--train-cap 200`; 17 composers hit the cap, others bottlenecked by the 30 s duration gate or sub-200 post-filter pool |
| Eval-A | **385** | 16 per train composer, stratified by duration (a few composers contributed fewer due to small post-filter pools) |
| Eval-B | **32** | Daglish_Ben (16) + Follin_Tim (16). Originally-proposed 6-8 cross-engine expansion is deferred -- a separate engine-fingerprint review (see Risks) is needed first |

Engine-fingerprint dedup is already in the picker (first 2000 raw
register writes, SHA-256). A one-shot **leak audit**
(`integration_tests/profile/audit_eval_leak.py`) runs before pinning
to catch sub-song self-references and near-duplicate melodies that
the prefix-only fingerprint dedup misses. Output:
`integration_tests/data/prodlike/leak_audit.json`.

`HVSC_VERSION` pinned to 84 to match canonical.

## Body

```
--layers 16 --heads 12 --kv-heads 4 --embed 768 --intermediate 2048
```

~50M params. Sizing rationale:
- Geometric step over canonical (~14M): mini (~3M) → canonical
  (~14M) → prodlike (~50M) → future "frontier" (~150M).
- Grouped-query attention (kv-heads=4) keeps KV cache predict-tractable
  on Orin NX (15.6 GB).
- Above the mini capacity ceiling so loop_lookahead's
  "easier-problem" component should attenuate; the residual is the
  real encoder effect.

## Train regime

```
--model=llama3_2
--shuffle 0.4 --accumulate-grad-batches 8 --batch-size 4
--learning-rate 2e-4 --weight-decay 0.01
<body flags>
--attn-dropout 0.1 --max-epochs 60
--early-stop-patience 10 --early-stop-min-delta 0.005
--val-check-every 1
```

- 60 epochs × ~4000 SIDs ≈ ~5× the tokens-seen budget of canonical's
  200 epochs × 789 SIDs.
- Looser min-delta (0.005 vs canonical's 0.01) since val_loss curves
  get noisier at scale.
- Wider early-stop patience (10 vs canonical's 5) so plateaus aren't
  cut off prematurely.
- LR midway between mini's 3e-4 and canonical's 1e-4 — matches the
  intermediate body scale.

## Wallclock estimate

Per arm: ~50M body × ~4000 SIDs × 60 epochs.

Scale factors over canonical (~14M body, 789 SIDs, 200 epochs):
- Body: 50/14 ≈ 3.6×
- Data: 4000/789 ≈ 5.1×
- Epochs: 60/200 ≈ 0.3×
- Combined: ~5.5×

Canonical specs are 60-120 min per arm → **prodlike ~6-11 hr per arm**.

A/B: la1 + la3 × **3 seeds** = 6 runs sequential → **36-66 hr** total,
about 1.5-3 days. Seeds bumped from the original n=2 floor because
with n=2 the within-arm sigma estimate has ~70% relative uncertainty,
which makes the 3σ decision rule fragile near the threshold; n=3
materially tightens the σ for ~50% more wall.

If multi-GPU arm-level parallelism (framework follow-up) lands first,
6-way → ~½ day wall.

## Pipeline prerequisites (must land first)

### Blocker 1: per-eval-subset metrics

Currently `--eval-reglogs` points at a single `eval/` directory that
the runner builds by concatenating `eval-A.list + eval-B-daglish.list
+ eval-B-follin.list`. The LightningModule logs one combined
`val_loss` / `val_acc` per epoch.

**Change**:
1. `base.py.resolve_data_layout` emits separate `eval_A/`,
   `eval_B_daglish/`, `eval_B_follin/` (plus prodlike additions) dirs.
2. `train.py.get_val_loader` returns a dict of dataloaders keyed by
   subset name, or accepts a `--eval-reglogs` value that's a
   pipe-separated list `<name>=<glob>|<name>=<glob>`.
3. `validation_step(batch, batch_idx, dataloader_idx)` looks up the
   subset name and logs `val_loss/<subset>` + `val_acc/<subset>`. Also
   keep the legacy `val_loss` / `val_acc` as the macro-average so
   `monitor='val_loss'` / EarlyStopping continue to work without
   touching callback code.
4. `metrics.py` adds extractors `val_loss_eval_a`, `val_loss_eval_b`,
   `val_acc_eval_a`, `val_acc_eval_b`. Aggregate Eval-B as the
   composer-averaged mean over its subsets (so per-subset reporting
   stays available but the report has a single cross-composer column).

Verify on the mini tier first: re-run `loop_lookahead` spec with the
new metric set and confirm Eval-A vs Eval-B-Daglish vs
Eval-B-Follin break out cleanly in `report.md`.

### Blocker 2: docker root-owned artefacts

`_docker_run` in `base.py:284` doesn't pass `--user`. Add:

```python
cmd += [
    "--user",
    f"{os.getuid()}:{os.getgid()}",
]
```

Container processes (parse/tokenize/train + the metric extractor
subprocs) then write as the runner UID; `shutil.rmtree(work_dir)` on
re-run no longer EPERMs.

Verify with `run_memorize_int_test.sh` re-run loop: first pass should
populate artefacts; second pass should rmtree + repopulate without
the `[Errno 13]` cascade currently masked by the manual
`docker run … rm -rf` workaround in AGENTS.md.

### Blocker 3: eval-set leak audit

`integration_tests/profile/audit_eval_leak.py` — one-shot script
taking train.list + each eval-*.list and reporting:

1. **Cross-set fingerprint overlap** — same SHA-256 over first 2000
   register writes as the picker uses, applied across train/eval-A/
   eval-B-* simultaneously rather than just within composers.
2. **Parse-equivalence overlap** — encode each candidate train + eval
   SID with the production encoder flags, hash the resulting token
   sequence (or its first 4K tokens), report any train/eval-* SID
   pair with identical hashes.
3. **N-gram overlap** — for each eval SID, count what fraction of its
   `seq_len=8192` token sequence appears verbatim as any contiguous
   n=64 window in any train SID. Threshold: log eval SIDs with >1%
   overlap; abort the pin if any are >5%.

Audit runs once before the prodlike pin and once before any future
re-pin. Output: `integration_tests/data/prodlike/leak_audit.json`,
committed alongside the lists.

## Spec: `loop_lookahead_prodlike`

```python
from integration_tests.experiments.base import Arm, ExperimentSpec

_TRAIN_ARGS = (
    "--model=llama3_2 "
    "--shuffle 0.4 --accumulate-grad-batches 8 --batch-size 4 "
    "--learning-rate 2e-4 --weight-decay 0.01 "
    "--layers 16 --heads 12 --kv-heads 4 --embed 768 --intermediate 2048 "
    "--attn-dropout 0.1 --max-epochs 60 "
    "--early-stop-patience 10 --early-stop-min-delta 0.005 "
    "--val-check-every 1"
)

spec = ExperimentSpec(
    name="loop_lookahead_prodlike",
    doc="Loop lookahead A/B at ~50M body × ~4000 SIDs × 25 composers.",
    tier="prodlike",
    arms=[
        Arm(label="la1", extra_cargs="--no-instrument-pass --loop-lookahead 1", baseline=True),
        Arm(label="la3", extra_cargs="--no-instrument-pass --loop-lookahead 3"),
    ],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "val_loss_eval_a",
        "val_acc_eval_a",
        "val_loss_eval_b",
        "val_acc_eval_b",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=3,
    seq_len=8192,
    tkvocab=131072,
    max_perm=2,
    train_args=_TRAIN_ARGS,
)
```

Pre-flight: `validate_branches.sh` extended to invoke this spec's flag
combination on a representative prodlike SID before the long run kicks
off, same gate logic as the current overnight wrapper.

## Decision rules

After the run:

1. **Primary (val_acc averaged over Eval-A + Eval-B):** if
   `la3 − la1` is positive and ≥ 3σ over within-arm std, recommend
   flipping `--loop-lookahead` default to 3.
2. **Cross-composer test:** secondary `val_acc_eval_b > val_acc_eval_a`
   delta tells us whether the win transfers off-distribution. A win
   that lives only in Eval-A is weak evidence for a production
   default flip.
3. **Capacity-attenuation test:** compare prodlike Δ to mini Δ. If
   prodlike Δval_acc ≥ ~½ × mini Δval_acc, the win is structural.
   If it collapses to <¼, mini was capacity compensation and we
   shouldn't flip.
4. **Null result:** keep `--loop-lookahead 1` as the production
   default; document the refutation in AGENTS.md "Refuted
   alternatives".

## Sequencing

Suggested order, with hard dependencies:

1. Commit AGENTS.md fold (orthogonal, ~5 min).
2. Blocker 2: docker UID fix (~30 min including smoke verify).
3. Blocker 1: per-eval-subset metrics (~4-6 hr including testing on
   mini). Most code-heavy step.
4. Blocker 3: leak audit script (~1-2 hr).
5. Pin prodlike corpus (run picker + audit + commit, ~1-2 hr; dump
   cache for new composers may take longer if `vsiddump` needs to
   build dumps that aren't cached).
6. Tier infra in `base.py` (~30 min).
7. Spec + validate_branches entry (~30 min).
8. Run `validate_branches.sh` pre-flight on prodlike target.
9. Launch the spec under `nohup`; check via
   `check_overnight_batch.sh` (extended to recognise prodlike if
   needed).

Steps 2 and 4 can run in parallel with step 3. Step 5 needs steps 3
+ 4 done (per-subset metrics must work and audit must clear before
the pin commit lands).

## Risks / open questions

- **Dump cache completeness for new composers.** The 1852 composer
  dirs under `/scratch/preframr/training-dumps/MUSICIANS/` may not all
  have `.dump.parquet` populated; the picker silently skips empties.
  Need to spot-check the top-25 list for cache holes before pinning.
- **GPU memory headroom.** 50M body + seq_len 8192 + batch 4 with
  grad-accum 8 — needs verification on the actual training host. If
  it OOMs, drop batch to 2 (effective batch unchanged).
- **Eval-B composer diversity.** Daglish + Follin are both
  C64-engine-family adjacent. Picking ~6 cross-engine composers needs
  a separate engine-fingerprint review to confirm structural
  diversity (otherwise Eval-B is a less-strong cross-distribution
  test than it appears). Tracked in
  `integration_tests/design/engine_fingerprint_evalb_design.md`;
  gates a re-pin with expanded Eval-B subsets after the first
  loop_lookahead_prodlike run lands.
- **Wallclock variance at scale.** A 6-11 hr per-arm estimate is
  load-dependent; preemption (e.g. host reboots, other workloads)
  could turn the 1-2 day estimate into a week. Recommend running
  unattended only on a host where preframr training is the sole
  workload.

## Out of scope (separate work)

- Frontier-tier (~150M) and full-HVSC corpus (~10k+ SIDs).
- Multi-GPU arm-level parallelism. Listed as a framework follow-up;
  would cut the prodlike wall by ~4× but is not on this critical
  path.
- Predict-side throughput at 50M. Orin NX prediction at this body
  size needs separate validation; not required for the train-side
  A/B decision.
