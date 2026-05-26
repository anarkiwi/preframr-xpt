# `corpus_structural_index` — design

**Status (2026-05-19):** READY (impl + smoke). One-shot CPU-only audit on fogbank that emits a queryable structural index of the **full HVSC corpus** so future macro hypotheses can be tested against the index instead of re-parsing 77k SIDs each time. Sequenced after the running `accuracy_push_prodlike` audition; sized for ~24-36h fogbank wallclock.

## Motivation

Every macro design question we've shipped (digi detection, ctrl-burst admission, VoiceBlockOrderPass, ArpeggioPass, PwmPass extension, 2× IRQ consolidation, engine-fingerprint clustering …) eventually needs the same shape of evidence: *"on the full corpus, how often does pattern X occur, broken down by composer / engine / voice / value-distribution, and what does the per-frame neighbourhood look like?"* Today each audit re-parses some portion of the corpus and answers one question; the next audit re-parses again. The single big-CPU job here pays the parse cost **once** and emits an index that resolves the next 10-20 macro questions by SQL queries (seconds to minutes), not parses (hours to days).

## What

One parser run over `/scratch/preframr/training-dumps/MUSICIANS/**/*.dump.parquet` (77,944 SIDs) with **every encoder pass disabled** so the parser emits raw consolidated per-frame writes — the lowest semantically meaningful layer. Each SID produces a structural index parquet plus a roll-up to a per-corpus aggregate. The parser closure stays single-threaded per-SID but fogbank's 72 cores parallelize across SIDs.

Output root: `/scratch/preframr/corpus_index/`

```
/scratch/preframr/corpus_index/
  manifest.parquet                 # per-SID summary (one row per SID)
  per_frame/                       # partitioned by engine_fp_cluster
    cluster=0/composer=*/sid=*.parquet
    ...
  per_voice_frame/                 # partitioned by engine_fp_cluster
    cluster=0/composer=*/sid=*.parquet
    ...
  aggregates/
    reg_op_val_freq.parquet        # global per-(reg, op, val) frequency
    bigram_atoms.parquet           # consecutive-atom bigrams
    trigram_atoms.parquet          # consecutive-atom trigrams (sampled)
    gate_transition_patterns.parquet
    arpeggio_signatures.parquet
    macro_potential.parquet        # per-macro estimated savings per SID
```

## Schemas

### `manifest.parquet`

One row per SID. Cheap to read, easy to filter for downstream audits.

| column | type | source |
|---|---|---|
| sid_path | str | source dump.parquet path (rel) |
| composer | str | parent dir name |
| engine_fp_cluster | i16 | `engine_fingerprint.assign_cluster` |
| engine_fp_canonical | u64 | fingerprint hash |
| irq | u32 | parser-detected IRQ |
| n_frames | u32 | post-`_consolidate_frames` |
| n_atoms_raw | u32 | atoms with NO encoder passes |
| n_voice_frames_active | u32 | (frame, voice) pairs with ≥1 write |
| max_writes_per_frame | u16 | max write count, any reg, any frame |
| max_ctrl_writes_per_voice_frame | u8 | for digi/ctrl-burst admission audits |
| max_vol_writes_per_frame | u16 | for digi audit |
| has_pcm_voice | bool | any voice meets digi entropy + autocorrelation thresholds |
| skipped_by_filter | bool | would the current `_filter` skip this SID |
| skip_reason | str | `_filter`'s reason string, if skipped |

### `per_frame/` parquet (one row per (sid, frame))

| column | type |
|---|---|
| sid_path, composer, engine_fp_cluster | str / str / i16 |
| frame_idx | u32 |
| frame_marker | i8 (FRAME_REG vs DELAY_REG) |
| delay_consolidation | u8 (DELAY_REG val) |
| voice_order_natural | u8 (svt under numeric order) |
| voice_order_canonical | u8 (svt under VBO content key) |
| n_writes_total | u16 |
| n_writes_voice_0, _1, _2 | u8 |
| n_writes_filter, _mode_vol | u8 |
| ctrl_v0, ctrl_v1, ctrl_v2 | u8 (end-of-frame CTRL byte per voice) |
| freq_v0_hi, freq_v0_lo, freq_v1_hi, freq_v1_lo, freq_v2_hi, freq_v2_lo | u8 (end-of-frame) |
| pwm_v0, pwm_v1, pwm_v2 | u16 |
| ad_v0, ad_v1, ad_v2, sr_v0, sr_v1, sr_v2 | u8 |
| filter_cutoff, filter_mode, mode_vol | u16 / u8 / u8 |
| gate_on_v0, gate_on_v1, gate_on_v2 | bool |
| gate_flipped_v0, _v1, _v2 | bool (transition from previous frame) |
| waveform_changed_v0, _v1, _v2 | bool |
| pcm_burst_candidate | bool (per-frame digi heuristic precursor) |
| voice_state_fp_v0, _v1, _v2 | u64 (hash of ctrl|ad|sr|pwm|freq16 — for instrument reuse) |
| intra_frame_op_hist | bytes (RLE: per (op, reg, voice) write counts inside the frame) |

### `per_voice_frame/` parquet (one row per (sid, frame, voice))

Per-voice expansion of the above; same partition layout. Used by audits that want a single voice's full state trajectory.

| column | type |
|---|---|
| sid_path, frame_idx, voice | str / u32 / u8 |
| ctrl, ad, sr, pwm, freq16 | u8 / u8 / u8 / u16 / u16 |
| ctrl_writes_this_frame | u8 |
| freq_writes_this_frame | u8 |
| pwm_writes_this_frame | u8 |
| step_from_prev_freq | i16 |
| step_from_prev_pwm | i16 |
| gate_transition | i8 (-1 off, +1 on, 0 hold) |
| envelope_phase | i8 (0 attack, 1 decay, 2 sustain, 3 release; heuristic) |
| voice_canonical_slot | u8 (slot index under VBO content key) |

### `aggregates/reg_op_val_freq.parquet`

| reg | op | subreg | val | count_global | count_per_cluster_0..N | distinct_sids |

### `aggregates/bigram_atoms.parquet`

| atom_a_reg | atom_a_op | atom_a_subreg | atom_a_val | atom_b_reg | atom_b_op | atom_b_subreg | atom_b_val | count_global | count_per_cluster_0..N |

(One row per distinct ordered pair of consecutive atoms in the raw stream; truncated to pairs that occur ≥10 times to keep the table small.)

### `aggregates/arpeggio_signatures.parquet`

| sid_path | voice | start_frame | period_frames | n_distinct_freqs | freq_set_sha8 | n_periods | step_pattern_sha8 |

Detected via a single forward pass: for each (sid, voice), find frame ranges with cardinality(freq16 values) ≤ 4 and a stable period via autocorrelation; emit one row per detection. Settles "is ArpeggioPass worth shipping?" without any new parse.

### `aggregates/macro_potential.parquet`

| sid_path | macro_name | sites_count | est_atom_savings | est_alphabet_growth |

`macro_name` ∈ {`pwm_preset_b2b`, `arpeggio_pass`, `last_write_wins_ctrl`, `irq_2x_consolidation`, `gate_replay_widen`, `instrument_program_widen`, `slope_pass_widen`, …}. Each macro has a detector function that scans `per_voice_frame` and emits site counts + savings estimate. Add new detectors as future questions arise; re-run aggregator (~20 min) without re-parsing.

## Audits this index retires / accelerates

This table maps each existing audit script to the index columns or aggregates that subsume it. The job is **complete** when every audit listed as `replace` here can be re-implemented as a DuckDB query against the index (≤ 50 lines each) and produces equivalent counts on the smoke list. Audits listed as `accelerate` keep their original parse loop but can lift coarse filtering off the index (cuts wallclock 10-100×).

| audit | mode | consumes |
|---|---|---|
| `irq_audit.py` | replace | `manifest.irq`, `manifest.skipped_by_filter`, `manifest.skip_reason` |
| `irq_digi_spotcheck_audit.py` | replace | `manifest.irq`, `manifest.max_vol_writes_per_frame`, `manifest.has_pcm_voice`, `aggregates/reg_op_val_freq` |
| `digi_audit.py` | replace | `manifest.skipped_by_filter`, `manifest.skip_reason` |
| `hvsc_version_check.py` | accelerate | `manifest.sid_path` enumeration (post-build, no re-parse) |
| `lonely_after_gate_audit.py` | replace | `per_frame.gate_flipped_v*`, `per_frame.intra_frame_op_hist`, `per_frame.n_writes_total` |
| `lonely_op_audit.py` | replace | `per_frame.n_writes_total`, `per_frame.intra_frame_op_hist` |
| `lonely_cooccurrence_audit.py` | replace | `aggregates/bigram_atoms` (filter to (single-row frame, single-row frame) pairs) |
| `gate_slope_adjacency_audit.py` | replace | `per_frame.gate_flipped_v*`, `per_frame.intra_frame_op_hist` on N+1 |
| `hard_restart_layer0_audit.py` | replace | `per_voice_frame.ctrl`, `per_voice_frame.ctrl_writes_this_frame` |
| `set_no_gate_event_audit.py` | replace | `per_frame.intra_frame_op_hist` filtered by `gate_flipped_v* = 0` |
| `slope_macro_audit.py` | replace | `per_voice_frame.step_from_prev_freq` runs |
| `audit_burst_replay_potential.py` | replace | `per_voice_frame.pwm_writes_this_frame`, `per_voice_frame.step_from_prev_pwm` |
| `cluster_coverage_balance.py` | replace | `manifest.engine_fp_cluster` GROUP BY |
| `audit_engine_families.py` | accelerate | `manifest.engine_fp_cluster`, `manifest.engine_fp_canonical` |
| `select_engine_evalb_candidates.py` | accelerate | `manifest.engine_fp_cluster` + intra-cluster centrality from `per_frame` voice-state fps |
| `legato_entropy.py` / `legato_entropy_per_cluster.py` | replace | `per_voice_frame.gate_transition`, `per_voice_frame.freq16`, partition `engine_fp_cluster` |
| `audit_global_instr_reuse.py` | replace | `per_frame.voice_state_fp_v*` GROUP BY |
| `audit_2stage_coverage.py` | replace | `per_voice_frame.step_from_prev_freq` + `aggregates/arpeggio_signatures` |
| `instrument_pass_capture_audit.py` | replace | `per_frame.voice_state_fp_v*` reused across SIDs |
| `audit_engine_fp_palette_eval_encodability.py` | accelerate | `manifest.engine_fp_cluster`, `aggregates/macro_potential` |
| `macro_bit_budget_audit.py` | replace | `aggregates/macro_potential` |
| `bpe_efficiency_audit.py` | replace | `aggregates/bigram_atoms`, `aggregates/trigram_atoms` |
| `alphabet_cooccurrence_audit.py` | keep | post-tokenizer, not parser-state |
| `tokenizer_*_audit.py` (3 scripts) | keep | post-tokenizer, not parser-state |
| `fuzzy_loop_*` (4 scripts) | keep | post-LoopPass, not raw |
| `loop_pass_save_ledger.py` | keep | post-LoopPass |
| `b2_unblock_prototype.py` | keep | needs SubregExtraPass + LoopPass output |
| `sub_frame_timing_characterization.py` | keep | needs intra-frame timing pre-`_consolidate_frames` |
| `parser_pw_stage_trace.py` / `parser_stage_drop_trace.py` | keep | trace tooling, not corpus statistics |
| `palette_pwm_prereq_ab_prototype.py` | keep | A/B prototype, not a corpus probe |
| `galway_sustain_hold_probe.py` | keep | one-off probe |

Coverage: **22 of ~40 corpus audits are retired or accelerated** by this single emit. The remainder need post-encoder-pass or post-tokenizer output that lives in the experiments parquet tree, not this index.

## Queries it answers

Each of these is a 5-30 line DuckDB query against the index, runs in seconds:

1. **Digi-detection threshold tuning** — pick a `max_vol_writes_per_frame` threshold; count how many SIDs would be admitted/rejected, broken down by composer; cross-reference with `has_pcm_voice` (the audibility-based detector from the digi design).
2. **Ctrl-burst admission per simplification candidate** — sum `est_atom_savings` for `macro_name='last_write_wins_ctrl'` over SIDs where `max_ctrl_writes_per_voice_frame ≥ 7`; compare against `est_atom_savings` for `irq_2x_consolidation`.
3. **ArpeggioPass scoping** — count distinct (composer, period_frames, n_distinct_freqs) tuples in `arpeggio_signatures`; estimate vocab-slot cost vs estimated savings.
4. **VoiceBlockOrderPass cross-engine generalisation** — group `voice_order_canonical` distribution by `engine_fp_cluster`; identify clusters where the content key picks identity vs non-identity perms.
5. **PwmPass extension** — count back-to-back `pwm_v0` changes in `per_voice_frame`; same for v1, v2; same for mode-vol; estimate the lonely-PWM_PRESET pair tail beyond the current 6,071-sites-per-500-parquets headline.
6. **Engine-fingerprint cluster validation** — for each cluster, compute the centroid of (waveform usage, hard-restart count, ADSR distribution) and the per-SID Mahalanobis distance; flag misclustered SIDs.
7. **Cross-engine Eval-B re-pin candidate sampling** — find SIDs in clusters not currently represented in Eval-B; rank by intra-cluster centrality; output a candidate list.
8. **`coarsen_pass` re-evaluation** — for each SID, simulate the materialisation savings of every BACK_REF in the LoopPass output (which lives outside this index but is cheap to join).

## Cost

**Parse pass (one-shot, expensive):**
- 77,944 SIDs ÷ 72 cores = ~1,083 SIDs/core. Each parse ≈ 2-4 s of CPU at the no-encoder-pass layer (most encoder passes are skipped here). Per-core wallclock ≈ 1-1.5 hr. Total wallclock: **2-3 hr**.
- Memory: each parser worker peaks ~500 MB; 72 workers × 500 MB = 36 GB. Fogbank fits easily.

**Index emission (concurrent with parse):**
- Each worker writes `per_frame` + `per_voice_frame` rows to a per-SID parquet as the parse finishes. No global IO contention. Output size estimate: 200-400 GB (115M frames × ~30 columns × ~4 bytes ≈ 14 GB compressed, but `per_voice_frame` triples it, plus aggregates).

**Aggregator pass (after index is on disk):**
- Bigram / trigram tables built via DuckDB GROUP BY on the per-frame parquets. ~2-4 hr on 72 cores.
- Arpeggio signature detector: 1 sweep over `per_voice_frame`, ~1 hr.
- Macro-potential detectors: ~30 min each.
- Total aggregator wallclock: **4-8 hr**.

**Total**: **~12-16 hr wallclock** on fogbank, no GPU, ~400 GB disk.

The 24h+ budget the user mentioned gives slack for:
- The full HVSC IRQ audit re-pin (the existing `irq_audit.csv` is ~73k rows; needs refresh).
- Engine fingerprint re-clustering using the new freq/ADSR distributions as features (currently clusters are pinned via `engine_families.json`).
- A first-pass digi audibility detector (FFT-and-blank per the digi design) on the 156-SID prodlike skip set — gives the first datapoints for that design.

## Refutation gates

- If the per-frame parquet schema can't answer 5 of the 8 listed queries without re-parsing or new per-SID computation ⇒ schema undersized; iterate before kicking off the 24h job.
- If aggregator output for `macro_potential` is dominated by a single macro (one detector accounts for >80% of estimated savings) ⇒ the index isn't surfacing diverse macro opportunities; consider that the bottleneck isn't macro design.
- If the index can't be re-generated from a partial parquet (e.g., a corrupted shard requires full re-parse) ⇒ checkpointing was missing; redesign per-cluster partition writes to be independent.

## Implementation sketch

New module `integration_tests/profile/build_corpus_structural_index.py`. Re-uses existing parser worker pool from `regdataset.glob_dumps + parser_worker`, with two changes:

1. Set every encoder pass arg to `False`. The parser yields rows from `_consolidate_frames` directly.
2. Post-yield, each worker runs `_emit_structural_index(df, sid_meta, out_dir)` which walks the df once with a state machine that mirrors `DecodeState` for end-of-frame value capture, then writes the per-SID parquet shards.

Aggregator: standalone DuckDB queries in `integration_tests/profile/aggregate_corpus_index.py`. One function per `aggregates/*.parquet` output; each function is replaceable / re-runnable without touching others.

Macro-potential detectors: one function per macro name, each ≤ 50 lines, all in `integration_tests/profile/macro_potential_detectors.py`. New macros add a new function + a row in the registry tuple.

## Sequencing and routing

- **Wallclock fit**: kicks off on fogbank when `accuracy_push_prodlike` exits and the full-HVSC tokenizer A/B is either complete or paused. The two share `/scratch` IO so don't co-run.
- **Routing**: per the AGENTS.md "CPU compute host" note, this is a fogbank-native job (no GPU). Output lives on shared `/scratch` so the local box and the Jetsons can query the index directly via DuckDB.
- **Re-run cadence**: re-emit when (a) HVSC version bumps, (b) the parser's pre-PASSES stages change (`_consolidate_frames`, `SlopePass`, `PresetPass`, `GateSlopeShiftPass`, `PerRegBurstPass`), or (c) the `_filter` admission predicate changes (so the `skipped_by_filter` column stays calibrated). Encoder-pass changes (PASSES / POST_NORM_PRE_VOICE_PASSES) do **not** invalidate the index because no encoder passes ran.

## Open questions

- Should the index include the full atom stream per SID (every (reg, op, subreg, val, frame) row) or just the per-frame end-state? Full stream is 5-10× the disk; end-state is enough for 8 of the 8 listed queries but loses the *write-order* signal that GateSlopeShiftPass + PerRegBurstPass care about. Compromise: emit `per_voice_frame.intra_frame_writes` as a small RLE-encoded blob (≤ 64 bytes per cell) that captures within-frame write order without materialising every row.
- Should `engine_fp_cluster` re-cluster on-the-fly from the new index, or stay pinned to the existing `engine_families.json` for compatibility with current eval splits? Suggest **both**: emit the existing-cluster column AND a `proposed_cluster` column from a re-cluster, so cluster drift is measurable across HVSC versions.
- Aggregator on DuckDB vs Polars: DuckDB is more SQL-natural and handles the parquet partitioning natively; Polars is faster on the in-memory join steps. Pick DuckDB unless a specific aggregator times out.

## Non-goals

- Live (always-up-to-date) index. The corpus changes monthly via HVSC drops; one big re-emit per HVSC version is fine.
- A web UI. DuckDB CLI queries are the interface.
- Encoder-pass output capture. That's a separate question and already covered by `experiments/run.py`'s per-arm parquet emission.
