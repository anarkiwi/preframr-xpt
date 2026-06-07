# Audio-equivalence normalization (parameter-space collapse)

**Status:** Draft 2026-05-23. Tokenizer-side data normalization that
collapses (op, reg, val) tuples producing equivalent SID
output to a single canonical representative, shrinking the content
vocab and densifying per-token training statistics. Complements the
existing macro-level normalization in `preframr_tokens.macros`
(rule-based) by adding an emulation-grounded value-level pass.

**Gate (revised):** "equivalent" means **register-log equivalence within the fidelity
`freq_tol`** — verified by `cb_div_audit` (same regs/order/delay; FREQ/PW/filter within
tolerance), NOT a perceptual/listening judgment. Same registers/order/delay ⟹ same output
by construction, so there is no WAV render or audition step. This narrows the pass to
collapses already within tolerance (see Risks) — it cannot collapse writes that merely
"sound similar" but differ beyond `freq_tol`.

**Learnability framing.** The vocab-size / data-density argument here serves learnability — but compare schemes by copy-fraction / per-frame h_k, not gzip-style compressibility ([`learnability_token_ordering_theory.md`](../references/learnability_token_ordering_theory.md) Principle 2).

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
5. Validation gate (per `(op, reg)` family):
   - **Register-equivalence within `freq_tol`** — every member of a class must collapse to
     its canonical value *within the fidelity contract's tolerance* (control regs exact;
     FREQ/PW/filter within `freq_tol` cents), checked corpus-wide by `cb_div_audit` after
     applying `canonical_map.json`. A class whose members diverge beyond `freq_tol` is
     **invalid** and must be split — there is no human-listening/audition step.
   - (Diagnostics, not the gate: within-class fingerprint distance < 0.5× between-class; no
     class > 30% of `|V_or|`.)

Pipeline cost: 40K fingerprints × ~50 ms / 72 cores ≈ 30 sec on
fogbank. K-means is trivial. The gate is the automated `cb_div_audit` register check — no
human listening.

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

### Round-trip register audit (new test)

`tests/test_audio_canon_round_trip.py`: for a sample of dump.parquets, apply
`canonical_map.json` and assert the decoded register stream still matches the original
under the fidelity oracle — `PREFRAMR_PARSE_AUDIT=raise` per tune / `cb_div_audit.py`
corpus-wide (control regs exact in input order + nominal `_MIN_DIFF` delay; FREQ/PW/filter
within `freq_tol`). **No rendering** — same registers/order/delay ⟹ same output by
construction ([`../references/sid_render_fidelity_contract.md`](../references/sid_render_fidelity_contract.md)).
A canonicalization that pushes any member beyond `freq_tol` fails the gate and the class is
split; there is no mel-distance / WAV-comparison step.

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
| 0. Calibrate | Build `canonical_map.json` for the prodlike tokenizer corpus. Validation per (op, reg) gate above; corpus register-equivalence check (`cb_div_audit`). | every class member within `freq_tol` under `cb_div_audit` after applying the map (diagnostic: within/between class distance ratio < 0.5 in all families with `K_or > 8`) |
| 1. Mini A/B | `audio_canon_mini_body_large` spec: 2 arms (baseline plain CE; baseline + `--audio-canon`), mini body=large, 3 seeds. Run the round-trip register audit on the canonicalized corpus. | (1) val_acc ≥ baseline within 1σ; (2) content vocab size shrunk ≥ 30%; (3) round-trip register audit (`cb_div_audit`) clean within `freq_tol`; (4) **no diversity_ratio regression at T=0.5** (this caught the cluster head + mask probes; load-bearing for any architecture-side win to stack) |
| 2. Canonical A/B | 901 SIDs, same gate at canonical scale | val_acc ≥ baseline + 0.003; rest unchanged |
| 3. Prodlike | Prodlike single seed, same gate; pair with the best surviving content-head (currently diffusion if Phase 3 PASSES, else plain CE) | val_acc on eval_a ≥ baseline + 0.005 AND diversity_ratio ≥ baseline AND content vocab reduction holds |

## Risks / open questions

- **Are register-family K_or counts right?** The proposed K_ors are
  educated guesses from SID-engineering experience. Phase 0
  silhouette + within/between distance ratios validate; if FREQ_LO
  K_or = 16 is too coarse, the register-equivalence gate (`cb_div_audit`)
  catches it — a too-coarse class diverges beyond `freq_tol`.
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
- **The register-equivalence gate is the hard gate.** A collapse is admissible iff every
  class member's decoded registers stay within the contract's `freq_tol` (`cb_div_audit`).
  The within/between distance ratio is only a clustering diagnostic. **Consequence:** this
  constrains the design to collapses that are *already* register-equivalent within tolerance —
  any "sounds the same but writes different registers beyond `freq_tol`" collapse is invalid,
  which substantially narrows what this pass can do versus the original perceptual ambition.

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
