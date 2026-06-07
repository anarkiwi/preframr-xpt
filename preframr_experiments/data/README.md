# Pinned data tiers

Stable HVSC-relative path lists consumed by ``preframr_experiments``.
The lists do NOT carry the
.dump.parquet bodies -- those are materialised by the dump cache at
``/scratch/preframr/training-dumps/`` (or rebuilt via vsiddump on
hosts without the cache).

Each list is one HVSC-relative path per line; trailing ``# ...``
comments are stripped at parse. Path format:
``MUSICIANS/<L>/<Composer>/<song>.<tune>.dump.parquet``.

## Tiers

### `smoke.list`

7 hand-picked SIDs covering every macro class the encoder produces.
End-to-end wallclock target < 10 min per experiment arm with a warm
dump cache. Used by the `smoke` tier of the experiment runner; the
intended use is fast iteration on a single arm or quick smoke A/Bs.

### `mini/`

Cluster-stratified subsample for fast first-pass arm-level decisions.
Each train composer covers a distinct engine-fingerprint cluster
(``engine_families.json`` k=7); each Eval-B subset is a held-out
composer in one of the covered clusters. Cluster 5 is skipped (only
2 composers, can't stratify cleanly).

- ``train.list``        150 dumps across 6 train composers, one per
                       cluster: DRAX (1), Goto80 (2),
                       Whittaker_David (3), Jammer (4),
                       Galway_Martin (6), Hubbard_Rob (7). 25 SIDs
                       per composer.
- ``eval-A.list``       30 in-distribution holdouts (5 per train
                       composer).
- ``eval-B-{daglish,follin}.list``  8 dumps each (cluster 4).
- ``eval-B-{crisps,mibri,marquis,winterberg,wilson}.list``  8 dumps
                       each, one per remaining cluster.
- ``picker_summary.json``  pick metadata (composers, clusters,
                          counts).
- ``leak_audit.json``    cross-set fingerprint + n-gram audit
                        (``audit_eval_leak.py`` output).
- ``engine_fp_palettes.json``  canonical engine-fp cluster palette
                              built by
                              ``profile/build_engine_fp_palettes.py``
                              from this tier's ``train.list``.
- ``HVSC_VERSION``      pinned corpus version (84).

Re-pinned 2026-05-16 from a 5-composer / 4-cluster predecessor (DRAX,
Goto80, Galway_Martin, Hubbard_Rob, Tel_Jeroen) that yielded
structurally 0% Eval-B encodability against the
``engine_fp_palettes`` artifact (Eval-B-{daglish,follin} were both
cluster 4 which the old train didn't cover). The amendment widens
cluster coverage for the Phase A A/B and unblocks every subsequent
mini A/B on the Tier 1 ladder. Pick script:
``integration_tests/pick_mini_stratified.py`` (uses pre-pinned
canonical + prodlike pools as sources -- no re-parsing). Historical
mini results retain their directional conclusions but specific
numeric thresholds (notably ``mini_baseline_seeds`` σ) need
re-calibration before the 3σ A/B decision-rule applies on the new
composition.

### `canonical/`

Pinned superset of the existing 5-composer multi-composer pick:
- ``train.list``        789 dumps across 5 train composers (Goto80,
                       Hubbard_Rob, Galway_Martin, Tel_Jeroen, DRAX),
                       30s minimum, capped at 200 per composer.
- ``eval-A.list``       80 in-distribution holdouts (16 per composer).
- ``eval-B-daglish.list``  16 Daglish_Ben dumps (cross-composer eval).
- ``eval-B-follin.list``   16 Follin_Tim dumps (cross-composer eval).
- ``picker_summary.json``  metadata (post-filter pool sizes per
                          composer, durations gates, picker config).
- ``HVSC_VERSION``      pinned corpus version (currently 84). A re-pin
                       commit changes both the lists and the version
                       together so a future operator knows which
                       upstream snapshot the lists came from.

Used by the ``canonical`` tier of the experiment runner.

### `prodlike/`

Near-production pin for capacity-independent macro A/Bs. Sized
between canonical and a future "frontier" tier; used to re-test
mini-tier wins that are credibly capacity-bound (e.g. the
2026-05-11 ``loop_lookahead`` la3 result at ~10sigma but at the
mini m_large ceiling).

- ``train.list``           4437 dumps across the top-25 composers
                          by HVSC ``.dump.parquet`` count (excluding
                          Eval-B holdouts). 30s minimum, capped at
                          200 per composer; cross-composer
                          fingerprint dedup applied.
- ``eval-A.list``          385 in-distribution holdouts
                          (~16 per train composer, stratified by
                          duration).
- ``eval-B-daglish.list``  16 Daglish_Ben dumps (cross-composer,
                          held out from train).
- ``eval-B-follin.list``   16 Follin_Tim dumps (cross-composer,
                          held out from train).
- ``picker_summary.json``  picker config + per-composer post-filter
                          pool sizes.
- ``leak_audit.json``      output of
                          ``integration_tests/profile/audit_eval_leak.py``:
                          fingerprint pass (any cross-set hash
                          collision flagged) + n-gram pass (128-write
                          rolling-window overlap; eval SIDs >5%
                          overlap with train abort the pin).
- ``HVSC_VERSION``         pinned corpus version (currently 84).

Used by the ``prodlike`` tier of the experiment runner. Body /
train regime: see ``base.prodlike_train_args()`` (~50M body, 60
epochs, looser early-stop than canonical). Per-arm wallclock
estimate: 6-11 hr (canonical scaling factor ~5.5x). Design
rationale: ``design/landed/prodlike_tier_design.md``.

## Re-pin procedure

Re-curation is a deliberate event (committed alongside a rationale).
To regenerate ``canonical/`` lists after an HVSC corpus update:

```bash
# Inside the docker image so pyarrow / pandas versions match.
docker run --rm \
    -v /scratch/anarkiwi/preframr/integration_tests:/integration_tests \
    -v /scratch/preframr:/scratch/preframr \
    anarkiwi/preframr \
    python3 /integration_tests/pick_multi_composer.py \
        --hvsc-root /scratch/preframr/training-dumps \
        --out-dir /scratch/preframr/canonical_lists_pinned

# Inspect picker_summary.json + diff against the pinned lists.
diff /scratch/preframr/canonical_lists_pinned/train.list \
     /scratch/anarkiwi/preframr/integration_tests/data/canonical/train.list

# When happy:
cp /scratch/preframr/canonical_lists_pinned/{train,eval-A,eval-B-daglish,eval-B-follin}.list \
   /scratch/anarkiwi/preframr/integration_tests/data/canonical/
cp /scratch/preframr/canonical_lists_pinned/summary.json \
   /scratch/anarkiwi/preframr/integration_tests/data/canonical/picker_summary.json
echo "<new-version>" > /scratch/anarkiwi/preframr/integration_tests/data/canonical/HVSC_VERSION
```

The smoke list is hand-edited; re-curation is similarly deliberate
(record the rationale in the commit message).

For ``prodlike/`` re-pins, swap the picker invocation to
``--train-top-n 25 --train-cap 200 --eval-per-composer 16`` and
the output dir to ``prodlike_lists_pinned``. After the picker,
run the audit:

```bash
docker run --rm \
    -v /scratch/anarkiwi/preframr/preframr:/preframr \
    -v /scratch/anarkiwi/preframr/integration_tests:/integration_tests \
    -v /scratch/preframr/training-dumps:/scratch/preframr/training-dumps:ro \
    -v /scratch/preframr/prodlike_lists_pinned:/scratch/preframr/prodlike_lists_pinned \
    anarkiwi/preframr python3 -m integration_tests.profile.audit_eval_leak \
    --hvsc-root /scratch/preframr/training-dumps \
    --train-list /scratch/preframr/prodlike_lists_pinned/train.list \
    --eval-list /scratch/preframr/prodlike_lists_pinned/eval-A.list \
    --eval-list /scratch/preframr/prodlike_lists_pinned/eval-B-daglish.list \
    --eval-list /scratch/preframr/prodlike_lists_pinned/eval-B-follin.list \
    --out /scratch/preframr/prodlike_lists_pinned/leak_audit.json
```

Audit exit 0 + zero fingerprint violations + zero n-gram aborts is
required before the pin commit lands.

## Why pin?

Pre-pin: ``pick_*`` scripts enumerated HVSC at run time, so the
selection drifted whenever the corpus mirror moved. Two arms run weeks
apart weren't necessarily on the same data, breaking arm-level
comparability.

Post-pin: every arm of every experiment runs on identical paths until
a re-pin commit explicitly changes them.
