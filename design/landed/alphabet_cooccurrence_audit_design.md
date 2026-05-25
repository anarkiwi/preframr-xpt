# `alphabet_cooccurrence_audit` design

Diagnostic-only profile pass over a converged prodlike checkpoint
and its 9 eval reglogs. Surfaces alphabet-utilization tails that
the encoder ladder is paying for in `tkvocab` width but not
cashing in across composers / engines. Output is a single JSON +
markdown roll-up; gates the next encoder-side spec class
(atom pruning vs content-addressed merging vs BACK_REF DIST
hi/lo split). Zero training cost.

**Status:** LANDED 2026-05-17. Script at
`integration_tests/profile/alphabet_cooccurrence_audit.py`; tests
at `tests/test_alphabet_cooccurrence_audit.py` (25 cases,
all green); first run on `accuracy_push_prodlike/apush/seed0`
finished in ~63s on 8 CPUs.

**Initial verdict:** ALL THREE triggers fire on the in-flight
`accuracy_push_prodlike` (epoch ~150, val_loss 4.86). Three
encoder-side spec classes unblocked simultaneously:

- `atom_prune`: 25.58% of alphabet (10,540 atoms) trains but never
  appears in any of 9 eval subsets.
- `cluster_disjoint`: 63.19% (26,044 atoms) is per-cluster or
  unseen-in-train. 20,492 atoms have train_count=0 in this spec's
  data (tokens.csv is a shared global vocab; that's dead capacity).
- `back_ref_dist_split`: 10,353 / 10,753 DIST atoms (96.28%) would
  collapse under hi/lo split — only 144 distinct hi-bytes × 256
  distinct lo-bytes. Directly validates the `1a35031` bit-budget
  audit recommendation.

## Motivation

Tier-1 macro ladder closed with 3/4 REFUTED on vocab bloat:
`global_instr_ids` Phase A (+19.4% alphabet, identical
encodability across all 8 eval-B-* subsets), `palette_pwm`
(prereqs refuted), `fuzzy_loop_ab` (+35% alphabet vs no-op).
The single Tier-1 PASS, `hard_restart_ab`, was a 3% shrink.
`macro_coarsening` (refuted earlier): 9% shrink at L=16 needs
+160% row growth. Pattern: every "tracker primitive widens
alphabet to compress rows" axis has either lost or shrunk
marginally. The remaining headroom on the **width** axis is in
the existing alphabet, not in new primitives.

Bit-budget audit (`1a35031`) named two concrete tails without
measuring their utilization across eval subsets:

- `BACK_REF` DIST: 1,533 / 7M rows overflow Int16. hi/lo split
  proposed but unowned; needs evidence it actually fragments per
  composer.
- Per-cluster legato ops (4 new atoms each for c2/c3/c4/c7, 256
  for c7) landed `7160f80` without per-eval-family utilization
  data; cluster 3 already refuted on val_acc, but its atoms are
  still in the alphabet of any spec that doesn't opt-out.

Phase A's "identical encodability across 8 eval-B-* subsets"
finding (`global_instr_ids_phase_a` verdict, 2026-05-17) is the
strongest single piece of evidence that alphabet width is
decoupled from eval coverage at current scale — meaning the
expensive width buys ~nothing on cross-composer val_acc. Audit
quantifies which atoms specifically.

## Inputs

- Converged checkpoint dir, default:
  `/scratch/tmp/preframr_experiments/results/accuracy_push_prodlike/apush/seed0/`
  (run is in flight at 2026-05-17 21:50 UTC, epoch 150, val_loss
  4.8635 and still climbing; audit runs against latest
  best-ckpt regardless of plateau status).
- `tokens.csv` (alphabet, `op,reg,subreg,val,count,n` —
  prodlike has 41,218 atoms).
- Parsed parquets under `train/<composer>/<basename>.parquet`
  and `eval_a/`, `eval_b_*/` subtrees — already exist for the
  in-flight run, no re-parse needed.
- HVSC composer → cluster map from `engine_families.json` (same
  pin the train artifact used).

No model inference required — audit is corpus-side only. A
follow-up audit *with* model inference (top-1 / top-k=8 emission
distributions on eval) is scoped as a separate `predict`-side
pass and is **not** part of this design.

## Output

`/scratch/tmp/preframr_experiments/results/alphabet_cooccurrence_audit/<ckpt-tag>.json`
with three sections:

### `per_atom`

For each of the 41k atoms in `tokens.csv`:

| field | meaning |
|---|---|
| `n` | atom index (matches `tokens.csv`) |
| `op` | op id |
| `train_count` | total occurrences across train parquets |
| `train_composers` | distinct composer count in train (1-25 for prodlike) |
| `train_clusters` | distinct cluster count in train (1-6) |
| `eval_a_count` | occurrences in eval-A |
| `eval_b_*_count` | per-family count for each of 8 eval-B subsets |
| `eval_subsets_hit` | distinct subsets (0-9) where atom appears |

### `per_op_summary`

For each op, the rolled-up Gini + tail metrics:

| field | meaning |
|---|---|
| `n_atoms` | atoms in this op |
| `n_atoms_train_single_composer` | atoms only in 1 train composer |
| `n_atoms_eval_unreached` | train atoms with 0 eval occurrences |
| `n_atoms_eval_single_subset` | atoms in exactly 1 eval subset |
| `gini_train_count` | concentration of train usage |
| `share_of_alphabet` | atoms in op / total atoms |

### `cluster_disjoint`

Atom partition by which (set of) train clusters they appear in.
Specifically: count atoms whose train-cluster-set is a subset of
{cluster N} for each N, then count atoms shared across ≥2
clusters. Measures whether the alphabet is structurally
per-composer (high disjoint count → strong overfitting risk) or
genuinely shared (high overlap → cross-composer headroom exists
but isn't being learned).

## Markdown roll-up

`report.md` alongside the JSON with:

- Top-10 ops by `n_atoms_eval_unreached` (alphabet weight that
  trains but never appears in any eval — pure noise).
- Top-10 ops by `n_atoms_eval_single_subset` (eval-touching but
  composer-overfitting candidates).
- `BACK_REF` DIST sub-table: train DIST atom histogram bucketed
  at hi/lo nibble boundaries. Tests whether the proposed split
  shows the expected long-tail-on-hi structure.
- Per-cluster legato op utilization across the 9 eval subsets.
  Sanity check that c2/c4 (defaults ON post-`7160f80`) actually
  fire in eval; c3 (default OFF) atoms should be near-zero in
  eval (presence ⇒ leak).

## Decision rule

Audit is diagnostic. The output gates follow-up specs via three
pre-stated triggers; the back_ref metric uses split-savings
share (atoms collapsed / total DIST atoms), not a std-ratio.

| Finding | Triggers |
|---|---|
| `n_atoms_eval_unreached ≥ 20%` of alphabet | Design `atom_prune_layer0_audit` spec (drop train atoms with `train_count < T` and 0 eval hits; T calibrated from the histogram). Pure vocab shrink, no row impact, deterministic refute-or-pass on alphabet delta alone. |
| `split_savings_atoms / n_back_ref_dist_atoms ≥ 0.5` | Design `back_ref_dist_split` spec (hi/lo split per `1a35031` recommendation). Two atoms per back-ref; rows up, alphabet down. Test on mini before prodlike. |
| `cluster_disjoint` partition shows >40% atoms per-cluster | Re-stage `global_instr_ids` Phase B (cross-cluster canonical IDs) on the **14-bit budget** the bit-budget audit recommended. Phase A's identical-encodability finding was Phase-A's specific encoder; Phase B's sharing mechanism is a different test. |

If none fire (alphabet is already tight and well-shared), the
audit refutes the "vocab efficiency has headroom" hypothesis and
the next move shifts to data / arch / scale per AGENTS.md's
escalation rule for Eval-B-* generic-timbre collapse.

## Methodology notes

- **Parquet walk reuses existing tooling.** Pattern matches
  `profile/audit_global_instr_reuse.py` (parallel pq read,
  composer extraction from path). Atom key is the tuple
  `(op, reg, subreg, val)` matching the `tokens.csv` schema;
  hash to `n` via a dict built from the CSV.
- **No re-parse, no live tokenization.** The audit is a pure
  aggregation over the spec's frozen `train/` + `eval_*/`
  parquet trees. Re-runnable in minutes; deterministic.
- **Eval reglogs are arm-invariant for a given spec.** Phase A
  established this. So the audit's output describes the spec's
  encoder, not the checkpoint — the ckpt path is in the audit
  only as a metadata stamp. Future audits against different
  spec encoders will produce comparable JSONs.
- **Per-cluster atom partition.** A composer's atoms are
  aggregated to the cluster level via `engine_families.json`;
  `cluster_disjoint` counts use cluster sets, not composer sets.
  Composer-level is in `per_atom.train_composers` for finer
  inspection.

## Cost

Estimate: ~15-25 min on the host (CPU only).
- 4437 train parquets × ~7M total rows: ~3 min parallel.
- 385 eval-A + 8×16 eval-B = 513 eval parquets: <1 min.
- Aggregation + JSON write: <30s.

No GPU needed. Audit can run while `accuracy_push_prodlike` is
still training (different cores, different parquet trees).

## Reproduce

```
docker run --rm \
    -v /scratch/anarkiwi/preframr/preframr:/preframr \
    -v /scratch/anarkiwi/preframr/integration_tests:/integration_tests \
    -v /scratch/tmp:/scratch/tmp \
    anarkiwi/preframr \
    python3 -m integration_tests.profile.alphabet_cooccurrence_audit \
        --spec-root /scratch/tmp/preframr_experiments/results/accuracy_push_prodlike/apush/seed0 \
        --families-json integration_tests/data/prodlike/engine_families.json \
        --out /scratch/tmp/preframr_experiments/results/alphabet_cooccurrence_audit/accuracy_push_prodlike.json
```

## Out of scope

- **BPE token (`tkmodel.json` vocab) utilization audit.** This
  audit operates on pre-tokenization (op,reg,subreg,val) tuples
  via `tokens.csv`. The actual model vocab is 131,072 BPE tokens
  in `tkmodel.json`; per-vocab-index utilization across composer
  / cluster / eval subsets is a separate audit reading
  `*.blocks.npy` files. Higher signal-to-noise for the
  "dead capacity in logit head" question; left as follow-up
  because the parser-side findings (especially BACK_REF DIST
  split) already give a concrete actionable spec.
- **Model-side emission audit.** Top-1 / top-k=8 distribution on
  eval would add "what the model thinks it needs" on top of
  "what the corpus contains". Higher value but needs a predict
  pass per eval subset. Separate design once the corpus-side
  audit lands and surfaces concrete tails worth predict-time
  follow-up.
- **`palette_merge` re-design.** Refuted in design phase
  (`palette_merge_design.md`); audit may produce new evidence
  (cluster_disjoint > 40%) that re-opens it, but the merge spec
  itself is a separate design.
- **Capacity-attenuation re-test.** Audit informs encoder
  changes; capacity-attenuation gates the *resulting* spec's
  prodlike Δ via the existing m_large→prodlike rule. No change
  to that pipeline.

## AGENTS.md fold (proposed)

Add to §Forward-looking under Tier 2 (vocab-side, distinct from
the existing tracker-primitive Tier 2):

```
- **alphabet_cooccurrence_audit** — LANDED 2026-05-17. Diagnostic
  over converged prodlike train+eval parsed parquets; per-atom
  train_count / eval_subsets_hit + per-op Gini + cluster_disjoint
  partition + BACK_REF DIST hi/lo split-savings.
  Output: `/scratch/tmp/preframr_experiments/results/alphabet_cooccurrence_audit/`.
  First run on accuracy_push_prodlike fires ALL THREE triggers:
  atom_prune (25.58% eval-unreached), cluster_disjoint (63.19%),
  back_ref_dist_split (96.28% atom-collapse). Three follow-up
  spec classes unblocked; ordering TBD by audition outcome.
```
