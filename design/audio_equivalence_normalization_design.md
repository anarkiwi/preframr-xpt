# Audio-equivalence normalization (parameter-space collapse)

**Status:** Draft 2026-05-23. Tokenizer-side data normalization that
collapses (op, reg, val) tuples producing perceptually-equivalent SID
output to a single canonical representative, shrinking the content
vocab and densifying per-token training statistics. Complements the
existing macro-level normalization in `preframr_tokens.macros`
(rule-based) by adding an emulation-grounded value-level pass.

Sibling to (but smaller than) the full audio-equivalence
tokenization research direction — that direction handles
sequence-level equivalences (write order, redundant writes,
context-dependent collapses) and is a larger redesign. This doc
covers the per-write subset: independent of write order, only
canonicalize the `val` within each `(op, reg)` family.

## Problem (re-anchored after the content-head arc)

Across five model-side interventions (MoS, entropy, mask-structural,
cluster-conditional, diffusion content), the content tier hits a
ceiling that the loss + sampling shape can't break. The bottleneck
framing has converged on **data sparsity in the content vocabulary**:

- Mini tokenizer: ~30K content vocab, ~1.3M content positions →
  ~43 occurrences per token average, heavy long tail.
- Prodlike tokenizer: ~30K content vocab, ~13M content positions →
  ~430 occurrences per token average, still long-tailed.

Many of those distinct content tokens encode register writes that are
**audibly indistinguishable**:

- FREQ_LO LSB differences below the cent-quantization threshold
- ADSR step values within just-noticeable-difference (JND) bins
- PWM micro-modulations below human-pulse-width perception
- Filter resonance fine-tuning below the resonance JND
- Master volume LSBs at the top of the volume curve

A model that treats audio-equivalent tokens as distinct vocab ids
must learn the equivalence implicitly from data. Sparse data + huge
distinct-token set ⇒ poor generalization. Tokenizer-side
canonicalization sidesteps the problem by collapsing equivalents
before training sees them.

## Hypothesis

For each `(op, reg)` family, partition the value space into
acoustic-equivalence classes (each class's members produce
indistinguishable audio when rendered in a canonical SID context).
Replace every raw write `(op, reg, val)` with `(op, reg, canon(reg,
val))` during the parse stage. The tokenizer sees a smaller value
alphabet → smaller content vocab → denser per-token training
statistics → better content-tier generalization, **independent of
model architecture**.

Predicted wins:
1. Content vocab shrinks from ~30K to maybe ~5-15K (depending on
   aggressiveness of binning). Smaller softmax, faster training,
   smaller predict-host envelope.
2. Per-content-token training mass increases proportionally (~2-6×).
3. Generalization improves because the model sees each acoustic
   class many times rather than each-class-member-individually-rarely.

Predicted risks:
1. Aggressive binning may collapse acoustically-distinct writes if
   the canonical-scaffold render misses context that matters
   (e.g., FREQ_LO differences are audible when voice is sweeping;
   stationary-context render won't see the sweep).
2. Round-trip fidelity may drop: rendered canonicalized output may
   sound subtly different from rendered original. Pinned by
   `tests/test_sid_same_value_writes.py` + a new audit (see Phase 0
   below).
3. The macro-pass pipeline (`preframr_tokens.macros`) expects
   specific values for some rule firing (e.g., HARD_RESTART
   pattern matches exact ADSR=0 writes). Canonicalization must
   commute with macro detection or the macro pass produces
   different output.

## Approach

### Offline equivalence-class table (Phase 0)

For each `(op, reg)` family present in the corpus:

1. Enumerate observed values `V_or = {v : (op, reg, v) appears in any
   dump}`. Mini: ~10K total; prodlike: ~40K total.
2. For each `v ∈ V_or`, fingerprint the write under a canonical
   scaffold (re-use `preframr_audio.fingerprint.fingerprint_writes`
   from v0.3.0). Mel features, 128-dim.
3. Per `(op, reg)`, cluster fingerprints. Cluster count is
   register-specific:
   - ADSR (regs 5, 6, 12, 13, 19, 20): K_or = 16 (4-bit-ish
     acoustic resolution per nibble).
   - FREQ_LO (regs 0, 7, 14): K_or = 16 (cent-quantization parallel).
   - FREQ_HI (regs 1, 8, 15): K_or = 8 (octave-scale).
   - PWM (regs 2, 3, 9, 10, 16, 17): K_or = 8.
   - Filter (regs 21, 22): K_or = 16.
   - Filter mode/vol (reg 24): K_or = 8.
   - CTRL (regs 4, 11, 18): K_or = |V_or| (no collapse; every bit
     matters for waveform / gate).
4. Pick the centroid-nearest member as the canonical representative
   per class. Output:
   `data/audio_norm/<tokenizer_hash>/canonical_map.json` with shape
   `{(op, reg, raw_val): canon_val}`.
5. Validation gates (per `(op, reg)` family):
   - Within-class mean fingerprint pairwise distance < 0.5 × the
     mean between-class distance (clean separation).
   - No class has > 30% of `|V_or|` (no degenerate "everything
     collapses here" class).
   - Audio audition: render 3 random class members from the 5
     largest classes and confirm they sound the same.

Pipeline cost: 40K fingerprints × ~50 ms / 72 cores ≈ 30 sec on
fogbank. K-means is trivial. Validation audition is the human
bottleneck (~30 min).

### Tokenizer integration

Adds a single transform `AudioCanonPass` to the macro pipeline,
inserted **before** all existing macro detection (so macros see
canonical values). Behind the
`--audio-canon` flag (default OFF). Flag accepts a path to
`canonical_map.json`; flag-on requires the JSON or hard-errors.

Per-write transform:
```python
def apply(self, df):
    for (op, reg) in self._family_index:
        mask = (df["op"] == op) & (df["reg"] == reg)
        df.loc[mask, "val"] = df.loc[mask, "val"].map(
            self._canonical_map[(op, reg)]
        ).fillna(df.loc[mask, "val"])
    return df
```

Falls through (no remap) for `(op, reg, val)` tuples not in the map
— defensive against vocab drift between map build time and parse time.

Touched files:
- `preframr_tokens/macros/audio_canon.py` (new) — `AudioCanonPass`
  implementing the transform interface.
- `preframr_tokens/macros/transform.py` — register the new pass via
  `register()`.
- `preframr/args.py` — new flag `--audio-canon <path>`.

No model-side changes. No prediction-time changes (the model emits
canonical values; the rendered output uses them as-is — the
canonical class members are equivalent to the originals by
construction).

### Round-trip audio audit (new test)

`tests/test_audio_canon_round_trip.py` (or
`profile/audit_audio_canon_fidelity.py`): for a sample of 100
random dump.parquets, render both the original and the
canonicalized dump (using `preframr_audio.audio_driver.render_to_samples`),
compare via `preframr_audio.fidelity.compare_renders` with
`max_frame_drift=2` (canonicalization may slightly shift
timing-sensitive macro decisions) AND `feature_diff_fn=mel_distance`
with `feature_diff_tolerance=<TBD from Phase 0 audition>`. Test
asserts ≥ 95% of samples pass.

Both `max_frame_drift` and `feature_diff_fn` are the preframr-audio
v0.3.x additions queued in the cluster-head design doc
(`cluster_conditional_content_head_design.md` "preframr-audio
enhancements" section). This audit is the second consumer that
promotes those API additions out of "deferred until needed" status.

## Comparison to other interventions

| | cluster head (refuted) | audio canon (this) |
|---|---|---|
| where it lives | model head | tokenizer transform |
| changes model size | -60% content head params | content head shrinks with vocab (~30K → ~10K = -67%) |
| changes inference | new sampling shape | unchanged (canonical val ≡ raw val acoustically) |
| training cost | per-cluster scatter (~4× per step) | none (smaller softmax = faster) |
| can combine with [mos, entropy, mask] | needs careful gate plumbing | yes, freely |
| failure mode if wrong | distribution narrowing, prompt collapse | quiet quality drop on round-trip audit |
| gate strictness needed | runtime audits (loop / diversity) | offline audit + round-trip test |

Crucially: **audio canon is orthogonal to every model-side
intervention.** Any future content-head experiment runs the same
spec with `--audio-canon` flipped on as an A/B; if canon helps,
it stacks on whatever other architectural choice wins.

## Phase plan

| Phase | Scope | Pass gate |
|---|---|---|
| 0. Calibrate | Build `canonical_map.json` for the prodlike tokenizer corpus. Validation per (op, reg) gates above; audition spot-checks on the 5 largest classes. | within/between class distance ratio < 0.5 in all families with `K_or > 8`; audition: human can't tell class members apart |
| 1. Mini A/B | `audio_canon_mini_body_large` spec: 2 arms (baseline plain CE; baseline + `--audio-canon`), mini body=large, 3 seeds. Run round-trip audit on the produced ckpts. | (1) val_acc ≥ baseline within 1σ; (2) content vocab size shrunk ≥ 30%; (3) round-trip audit ≥ 95% pass; (4) **no diversity_ratio regression at T=0.5** (this caught the cluster head + mask probes; load-bearing for any architecture-side win to stack) |
| 2. Canonical A/B | 901 SIDs, same gate at canonical scale | val_acc ≥ baseline + 0.003; rest unchanged |
| 3. Prodlike | Prodlike single seed, same gate; pair with the best surviving content-head (currently diffusion if Phase 3 PASSES, else plain CE) | val_acc on eval_a ≥ baseline + 0.005 AND diversity_ratio ≥ baseline AND content vocab reduction holds |

## Risks / open questions

- **Are register-family K_or counts right?** The proposed K_ors are
  educated guesses from SID-engineering experience. Phase 0
  silhouette + within/between distance ratios validate; if FREQ_LO
  K_or = 16 is too coarse, audition catches it.
- **Sequence-level equivalences not handled.** E.g., two consecutive
  writes to the same reg with values v1, v2 where the first is
  overwritten before any clock advances. Macro pass already handles
  some such cases (`DedupSetPass`); leave the rest to a future
  audio-equivalence-tokenization redesign.
- **Will the macro pipeline misfire on canonical values?** E.g.,
  `HARD_RESTART_OP` triggers on specific ADSR-zero patterns. If
  canonicalization rounds those to non-zero, HARD_RESTART stops
  firing. Mitigation: enumerate macro trigger conditions, pin them
  out of the canonical map (i.e., specific (op, reg, val) tuples
  are excluded from collapse).
- **Tokenizer hash dependency.** Canonical map is built for one
  tokenizer's `(op, reg)` set. Adding new transforms changes the
  vocab → map needs rebuild. Mitigation: cache the map by tokenizer
  hash (same pattern as `data/content_clusters/<hash>/`).
- **Audition is the soft gate.** Automated within/between distance
  ratio is necessary but not sufficient. A human listening pass on
  the 5 largest classes is the Phase 0 commitment.

## Why this might land where the model-side interventions didn't

Every refuted intervention (MoS, entropy, mask, cluster, diffusion)
tried to change how the model **uses** the content vocab. None of
them changed **what the vocab is**. If the vocab itself is the
problem — too sparse, too many lookalike tokens that the model can't
disambiguate from data — then no model-side fix will recover the
generalization gap. Audio-canonicalization changes the vocab
directly: shrinks it, densifies per-token statistics, removes a
class of generalization failures by construction (any "the model
picked the wrong almost-identical token" miss disappears).

## References

- Cluster-head design (refuted at Phase 2):
  `cluster_conditional_content_head_design.md`. Same acoustic-cluster
  primitive, applied at the head; this doc applies it at the
  tokenizer.
- Refuted entries chain: `refuted/per_tier_heads_mos.md`,
  `per_tier_heads_mos_prodlike.md`, `mask_structural_loss.md`,
  `per_tier_heads_entropy_prodlike.md` — the chain of
  model-side-tried-and-failed evidence motivating data-side
  intervention.
- `preframr_audio.fingerprint` v0.3.0 (the Phase 0 calibration
  primitive) — `preframr-audio:preframr_audio/fingerprint.py`.
- Macro pipeline interface: `preframr_tokens.macros.transform`
  (`register`, `Transform`, `PipelineEntry`).
