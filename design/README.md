# Design notes index

**Orientation:** [`architecture_overview.md`](architecture_overview.md) — which
part lives in which repo (tokens / audio / framework / xpt / aug) and **why**, the
dependency layering, and how to **derive the release process** (PyPI-tag vs
image-VERSION; the public-PyPI-propagation gotcha). Read it before any cross-repo
change or release.

**Per-repo architecture references** (all docs live here in preframr-xpt; the code
they describe lives in the sibling repos under `/scratch/anarkiwi/`):
- [`tokens_architecture.md`](tokens_architecture.md) — **the parsing reference**:
  preframr-tokens parse→pass→tokenize→decode pipeline, register/atom/op model,
  `combine_reg` settled-freq, the 3-layer pass-framework registration, fidelity,
  Corpus/blocks/df-map, and the invariants/gotchas. Consult before touching parsing.
- [`audio_architecture.md`](audio_architecture.md) — preframr-audio render pipeline
  + fidelity oracle.
- [`framework_architecture.md`](framework_architecture.md) — preframr train/predict/
  model + data path + generation gotchas.
- [`backlog_tokens_hardening.md`](backlog_tokens_hardening.md) — **precise file-level
  implementation instructions** for the queued tokens hardening: dead-wood removal,
  real-pipeline structural/balance tests (catch the synthetic-df false-green class), and
  driver-truth fixtures with RESID≈0 as the completeness metric.

Organized by **research axis** (the project's priority order), not by lifecycle:
status is a per-row column so refuted/landed work sits next to the live work in
the same thread. Refuted hypotheses also have one-paragraph stubs at
`preframr_experiments/data/refuted/<exp>.md` (this repo) — read those before
reopening a rejected direction.

## How this index is organized (read before adding a doc)

**Five axes** (top-level grouping — mirrors AGENTS.md "Project goal" + priority).
Add a new doc under its **primary** axis; cross-link a secondary axis in prose if
it spans two:

1. **Generalization** (primary goal) — predicting unseen continuations across
   composers/engines: the content objective, representation/tokenization, the
   metrics that detect it.
2. **Correctness & fidelity** — round-trip/decode exactness, audio equivalence,
   eval-leak + version gates, coverage bugs.
3. **Efficiency & deploy envelope** — token budget, training memory/wallclock,
   predict-host (Orin) throughput.
4. **Runner / experiment infra & process** — the docker runner, resume/abort/
   parallelism, repo scope.
5. **Data & corpus** — tier definitions, engine-fingerprint eval sets,
   augmentation, dataset coverage.

**Status column + lifecycle.** Every doc starts with a one-line `**Status:**`
header, and edits update it before the body. Lifecycle:

- **Draft** / **Scoping** → **Pending impl** / **Drafted, pending trigger** →
  **In flight** (spec + impl shipped; running/awaiting verdict) →
- **Landed**: move the file to [`landed/`](landed/) and add a row to
  [`landed/README.md`](landed/README.md) (archive is grouped by *kind*). The
  axis table keeps an inline `Landed` row pointing at `landed/…` so the thread
  stays whole.
- **Refuted**: keep the doc in place, set status to `Refuted`, and add a "do not
  revisit without …" stub at `preframr_experiments/data/refuted/<exp>.md`.
- **Deferred**: reviewed but parked on an external condition (e.g. cloud rental).
- **Reference**: positioning/strategy docs with no single impl.

Directory stays **flat** (no per-axis subdirs — docs span axes and moves break
the relative `[..](..)` / `../data/refuted/` links). Grouping lives here, in the
index.

## 1. Generalization (primary goal)

The content-ceiling thread: the per-token content objective (refuted across the
board) → the pivot to **representation/tokenization**, where the confirmed win
lives (`full_macros`; FREQ_TRAJ landed).

| Doc | Summary | Status |
|---|---|---|
| [`melody_data_gap_ladder.md`](melody_data_gap_ladder.md) | Why real Bach generalizes through this encoding (held-out 0.513 > 0.456 ceiling) but real-SID melody doesn't (0.35 ≈ 0.30 ceiling) at the SAME size/model/encoding. Marginal onset entropy is similar (~5b); the gap is CONDITIONAL predictability → V0 onset is ornament-polluted (vibrato/slide/arp), not the musical note. Proposes a progressive data-simplification ladder (single-voice → gate-anchor → de-arp → semitone → rhythm-grid) on existing mini data to localise the gap; gate-anchoring predicted as the key rung. | Draft / proposed program |
| [`melody_channel_factorization.md`](melody_channel_factorization.md) | Reopens "encoding sufficient / data-limited" on the gap it never tested: melody was only ever measured IN ISOLATION (ladder destroys ornament); under the deployment condition note + ornament COEXIST, and the deployed V0-onset sits ~0 vs the melody-alone ceiling 0.247 — the cost of MULTIPLEXING the note target with the trivially-predictable (0.559) ornament (P3 within-voice). Proposes factoring into a melody-skeleton channel + a PARAMETRIC ornament channel (maintains ornamentation while surfacing melody; additive, not subtractive). Decisive cheap probe specified (`audit.melody_channel_probe` + `extract_sid_melody --channels`): interleaved-skeleton-acc vs skeleton-only. **EXECUTED ×3 seeds: +0.032 (always positive but small) — strong form REFUTED** (interleaved 0.336 ≈ data ceiling, not ≈0; deployed ~0 is cross-voice/structural multiplexing, not within-voice). Minor factorization bonus only → reopen as a token-budget play, not a melody bet. | **EXECUTED — strong hypothesis refuted, minor bonus confirmed** |
| [`ornament_transfer.md`](ornament_transfer.md) | Make ornament a learnable/transferable channel. Measured: a model under-generates ornament because the per-frame encoding is the rendered output of a low-entropy driver program; and ornament IS overwhelmingly a small note-relative offset-set (wavetable/arp: median 4 distinct pitches, 44% ≤3) + slides — note/transposition-invariant. Proposes a **per-note parametric ornament descriptor** (offset-set/depth/slide-target + rate, relative to the note) aligned to the skeleton; scored by emission-rate + distributional + audition (P6), audition-gated (lossy). Cheap A/B probe (parametric vs raw per-frame) specified. **EXECUTED ×3 seeds: parametric REFUTED as a transfer lever** — RAW per-note already emits ornament at ≈corpus rate (0.18–0.22 vs 0.23) and transfers its type distribution; the earlier under-generation was a per-frame *interleaving* artifact, fixed by per-note ALIGNMENT (de-multiplexing), not parametrization. Parametric reverts to a token-budget question. Follow-up proposal: mined **WAVETABLE codebook** for the expressive RESID tail (Hubbard/DRAX/Goto80 wide arps) — measured 83% cleanly-cyclic, per-composer banks tiny (Hubbard 6 tables/13× reuse), top-128 cover 84%; encode `(table-id, rate, phase)` ordered + note-relative, with a raw escape; fidelity+budget(+within-composer transfer) play, probe-ready. | **EXECUTED (parametric refuted) + codebook proposal** |
| [`unified_pitch_encoding.md`](unified_pitch_encoding.md) | **PROPOSAL** synthesising the melody/ornament arc: a **skeleton** (level-change∪gate segmentation; interval-V0 semitone) + a **pitch-ornament** channel (one descriptor/note: PLAIN / OCTAVE-ARP / ARP-codebook / VIBRATO / SLIDE / RESID), both over **one semitone note→freq LUT**. Cents revisit: notes are clean semitones (measured median 1.5c) → note=semitone index, cents move to vibrato depth. Universal (driver-mapped), learnable (per-note aligned + key-invariant), token-efficient (~2 tokens/note, PLAIN-dominated, arps codebook'd). Honest non-claims: not a melody-accuracy bet (P6); parametric-vs-raw is budget not transfer. **EXECUTED: cents revisit validated (notes 1.5c from semitone), level∪gate segmentation +42% notes (PLAIN 0.33→0.58), encoder/decoder built + round-trips, generalization test passes — skeleton held-out next-interval 0.518 (beats 2-gram ceiling 0.407, ≫ prior gate-anchored 0.225), ornament emits at corpus rate (JS~0.05), WAVs rendered.** | **EXECUTED — probed, implemented, generalization-tested** |
| [`sid_driver_ornament_reference.md`](sid_driver_ornament_reference.md) | **Background reference** (cite, don't re-derive): how C64 SID drivers generate per-frame ornament across **pitch, pulse-width, and filter**. Two mechanisms: (A) note-index semitone offset cycles = arps (codebook); (B) parametric/table sweeps in the value domain = vibrato/portamento/PW/filter. Filter is **global** (one controller voice + per-voice routing); PW/filter sweeps persist across notes (not note-aligned); gate-on ≠ note boundary for held-gate/legato drivers (use intrinsic level-change); Hubbard arp is octave-only. Sources: defMON, SID Wizard, Hubbard *Commando* + **C=Hacking #5**, Galway Ocean drivers. | Reference |
| [`percussion_stamp_encoding.md`](percussion_stamp_encoding.md) | **PROPOSAL** (RESID=0 iter-2): percussion = an EXACT recurring write-series stamped on a rhythmic grid, not a waveform. Encode as **inline, redefinable packed stamp blocks** (streaming dictionary; drums drift) carrying the drum's FULL footprint (freq/ctrl/PW/ADSR + **consistency-attributed** drum-scoped filter; the global filter is folded in only if identical every hit — measured ~1/3 is) + voice-agnostic **backref pointers** with a global **drum-character** tag (kick/hat/snare…). Lossless per-tune defs + transferable character vocab; definitions laid out as regular atoms so **Unigram sub-tokenizes/clusters** similar drums. Measured (rung-0): **85–90% of remaining RESID notes are recurring stamps, ~85% gridded**, incl. pitched drums a noise rule misses; byte-exact. Probes `audit/probes/resid_percussion.py`, `resid_drum_footprint.py`. | **Proposal — measured-validated, not built** |
| [`patch_preamble_encoding.md`](patch_preamble_encoding.md) | **PROPOSAL** (twin of the drum codebook): a melodic note = PITCH (skeleton+ornament) × TIMBRE (a reusable **instrument PATCH** = pitch-invariant ADSR + control-register articulation sequence + PW). Define the patch inline (redefinable), make it AMBIENT via a `PATCH_SET` program-change so new notes just carry pitch; Unigram clusters similar patches (same ctrl-progression/ADSR → shared sub-tokens). The driver-native instrument bank made explicit. Measured (prototype `audit/probes/resid_patch_codebook.py`, rung-0): **top-8 patches cover ~84% of a tune's notes**; dropping PW collapses the codebook **69%** (ctrl-prog+ADSR = the stable core). Factors the stream into pitch × timbre × percussion. | **Proposal — measured-validated, not built** |
| [`IMPLEMENTATION_resid_zero_tokens.md`](IMPLEMENTATION_resid_zero_tokens.md) | **BUILD SPEC** for a `preframr-tokens`-only agent: the full RESID→0 mechanism stack under the speculative pipeline. **Every RESID note is a documented engine mechanism** (no irreducible floor); fitter-per-mechanism accounting (`audit/probes/resid_final_accounting.py`): STAMP_abs/rel/wild + ARP + ARP_accent + SWEEP(+glide/loop) + PERC + SEGMENT/DECOMP → unaccounted **~0.7%** (held-gate concatenations of known mechanisms; literal 0 = the encoder's segment-then-fit, NOT loosening the probe). Build order: pipeline framework → STAMP codebook → patch preamble+mutations → held-ARP/SLIDE/SWEEP/re-seg → arbiter competing claims. **Testing: run ALL tests in DOCKER; full CI-equivalent `docker build` run before any PR (§8.0).** Includes the auto-RE profiler as the acceptance instrument (drive per-engine UNRESOLVED→0). | **Build spec — capstone, hands off to tokens agent** |
| [`speculative_encoding_pipeline.md`](speculative_encoding_pipeline.md) | **PROPOSAL** (architecture): replace the strict-order DESTRUCTIVE macro-pass chain with **claims + arbitration** — passes are non-destructive PROPOSERS reading an immutable source, emitting scored `Claim`s (writes consumed, replacement tokens, fidelity/learnability/budget score); an **arbiter** picks a lossless PARTITION maximizing the objective, per region AND per tune. Fixes one-pass-destroys-another (the `_df_sink` workaround proves it) + lets drum/skeleton/sweep/patch COMPETE so the best encoding wins (RESID becomes the true floor, not a pass-order artifact). Speculative = overlapping/alternative claims; per-tune mode selection = "most appropriate encoding for a tune". Migration generalizes the existing `drop_idx`/`new_rows` splice. | **Proposal — architecture, not built** |
| [`encoding_principles.md`](encoding_principles.md) | The orienting rubric for SID stream encoding: **fidelity × context-efficiency × learnability** (priority order), the learnability sub-principles (separability / locality / no-multiplexing / alphabet≠learnability / right-yardstick), and a per-change checklist. Distilled from the melody-onset arc (de-merge win + voice/op48/semitone results). Other encoding designs check against this. | Reference |
| [`voice_encoding_reference.md`](voice_encoding_reference.md) | How the 3 SID voices are carried in the token stream: voice is **packed into the FRAME (−128) val** (low 6 bits, base-4 digits = voice+1, one slot per FRAME/VOICE marker), NOT in the VOICE (−126) token (whose val is zeroed in the trained stream). `_add_voice_reg` canonicalises reg to 0–6 + emits VOICE delimiters; `remove_voice_reg` inverts. Melody onsets are multiplexed across voices by this header — the model must read it to attribute writes. | Reference |
| [`superframe_voice_lane_design.md`](superframe_voice_lane_design.md) | Reorganize the frame-major interleave into **voice-major lanes inside the superframe** (the existing `super_frame` scaffold's blank intra-block layout): each voice's line contiguous, with voice→register-class sub-lanes so PW/filter re-admit without re-fragmenting melody. Rationale = the interleave taxes melody (fragmentation / multiplexed target / cross-voice merges). Distinct from refuted `voice_trajectory` (features) and `sequence_order_normalization` (order). | **Draft — deferred behind the bare-melody-stream work**; two open prereqs (original intent; why N≥2 stalled) |
| [`melody_learnability.md`](melody_learnability.md) | The open content frontier: melody onset is predictable (trigram 0.79–0.82), ~13.4% of stream (fragmented across op0-SET-freq / op45 V0 / op47 NUDGE pitch), but **0.000 at mini across 5 A/Bs** — converged diagnosis = scale-bound hard cross-song prediction (only prodlike has shown signal, op45=0.067). Consolidates the active arc + landed encoding/loss stack + open frontier (prodlike full-stack arbiter vs distributional/perceptual pivot). | **ACTIVE research arc** — `freq_onset_channel_mini` in flight; prodlike full stack is the deferred decisive test |
| [`landed/trajectory_anchoring.md`](landed/trajectory_anchoring.md) | Gate/sweep-anchored FREQ_TRAJ origins (the original encoding hypothesis + full mini-A/B history). | Landed (tokens 0.25.0, preframr 0.2.6) |
| [`landed/onset_loss_prioritization.md`](landed/onset_loss_prioritization.md) | `--onset-loss-weight`: up-weight rare FREQ V0-onset in CE to force capacity onto the rare-and-ignored melodic onset. | Landed (preframr 0.2.8; W=10 mini nudge 0→0.002, no all-tier cost) |
| [`landed/freq_trajectory_anchoring_impl.md`](landed/freq_trajectory_anchoring_impl.md) | Tokens-side impl spec for `TrajectoryAnchorPass` (the two-pass intrinsic anchor detector + segment-boundary integration). | Landed (tokens 0.25.0) |
| [`landed/freq_v0_interval.md`](landed/freq_v0_interval.md) | Tokens-side impl spec for `--freq-v0-interval`: encode V0 as a signed interval from the previous voice onset. | Landed (tokens 0.26.0) |
| [`landed/freq_onset_channel.md`](landed/freq_onset_channel.md) | Tokens-side impl spec for `--freq-onset-pass` (FREQ_ONSET op48): re-tag residual op0 SET on TRAJ_REGS → 1-token onset; SET only carries control/ADSR/routing. | Landed (tokens 0.27.0) |
| [`op48_freq_onset_residual.md`](op48_freq_onset_residual.md) | Why op48 stays 0.000 while de-merge lifts op45 V0 0.012→0.338: op48 is the **lead-voice (100% voice 0) sub-frame ornament residual** — absolute high-cardinality freq-lo writes the per-frame trajectory pass can't fold (41% in frames with 4+ writes on the reg), and a rare op-tag the model emits 1.6% of the time (op-level starvation). Not mis-routed skeleton; scale/re-routing won't fix it — the lever is the ornament codebook. | Diagnosed 2026-05-29 |
| [`landed/melody_merge_split.md`](landed/melody_merge_split.md) | Tokens-side impl spec for `--melody-merge-split`: post-Unigram-encode pass that splits cross-melody-boundary merges so pitch is a separable target. | Landed (tokens 0.28.0); `melody_merge_split_mini` in flight |
| [`motif_pass_design.md`](motif_pass_design.md) | Corpus-mined, per-block, lossless motif pass (tokens 0.20.0): collapses cross-composer motifs into loss-tier-zero `MOTIF_OP` atoms. ~11.4% fewer tokens at deployment vocab; the A/B tests **learnability**. | **Refuted 2026-05-27** (content-tier neutral-to-negative vs no-motif full_macros; `data/refuted/motif_pass.md`) |
| [`compound_token_design.md`](compound_token_design.md) | Approach D: compound-token tokenizer + parallel-attribute heads (CompoundWord / OctupleMIDI). Multi-attribute-per-token reorganization. Also an efficiency bet (token budget). | Draft, design review pending |
| [`motif_templates_v2_impl_design.md`](motif_templates_v2_impl_design.md) | Implementation design for value-slotted motif templates (MotifDict v2): template token + content slot(s), shape-keyed mining, lossless expand. De-fragments the motif vocab (~10 templates vs 6260 (shape,value) variants) and exposes motif-carried content to the content tier. Tokenizer-only Phases 0–2 (no model change); P3 = compound tokens. | **Refuted 2026-05-27** (built + A/B'd; content-tier did not beat no-motif; `data/refuted/motif_pass.md`) |
| [`audio_equivalence_normalization_design.md`](audio_equivalence_normalization_design.md) | Tokenizer-side normalization collapsing `(op, reg, val)` tuples that render perceptually-equivalent SID output into canonical forms. | Draft 2026-05-23 |
| [`sequence_order_normalization_design.md`](sequence_order_normalization_design.md) | Sequence-level sibling of the above: collapse the inaudible per-frame write-order DoF. Audit decomposed the +0.169 SET→TUPLE gap into −0.123 **multiplicity** + −0.046 order; audio-safe (voice-respecting) reorder recovers only +0.009 (~5%), and 84% of repeated writes are genuine sub-frame modulation content. Reorder is inaudible (render corr 1.0) but low-value. Audit `audit/audit_seq_order_norm.py` kept as instrument. | **Refuted as a lever 2026-05-27** (stub `data/refuted/sequence_order_normalization.md`) |
| [`per_voice_aux_supervision_design.md`](per_voice_aux_supervision_design.md) | Per-voice auxiliary classification heads on the body's hidden state. | Scoping |
| [`generalize_min_val_acc_floor_design.md`](generalize_min_val_acc_floor_design.md) | Calibrate `GENERALIZE_MIN_VAL_ACC` as 2/3 × median val_acc once 2-3 canonical baselines run (the generalization gate). | Pending impl (env hook exists, default 0/off) |
| [`model_loss_queue.md`](model_loss_queue.md) | Back-pocket queue + decision tree for picking the next content bet from audit evidence. | Reference |
| [`tokenization_vs_music_llms.md`](tokenization_vs_music_llms.md) | preframr's register-event + macro Unigram tokenization vs symbolic/MIDI, audio-codec, VQ; argues the content ceiling is tokenization-induced. | Reference (positioning) |
| [`music_llm_landscape_and_fail_fast_plan.md`](music_llm_landscape_and_fail_fast_plan.md) | Cross-LLM idea survey + ranked cheap fail-fast probes. | Reference (strategy) |
| [`landed/unified_oscillation_primitive_design.md`](landed/unified_oscillation_primitive_design.md) | Unified `FREQ_TRAJ` op + `FREQ_NUDGE` — the representation win; now the live deployment tokenizer. | Landed (tokens 0.16/0.17) |
| [`multi_modal_objective_design.md`](multi_modal_objective_design.md) | Umbrella framing of the per-token CE bottleneck on the multi-modal content tier. | Refuted (B/C/A all rejected) |
| [`per_tier_heads_design.md`](per_tier_heads_design.md) | Shared body + 4 tier heads + router; MoS-NLL content head (Approach C). | Refuted at prodlike (router saturates) |
| [`content_diffusion_design.md`](content_diffusion_design.md) | D3PM absorbing-state discrete-diffusion content head (Approach A). | Refuted (sampling-side; no CE change) |
| [`cluster_conditional_content_head_design.md`](cluster_conditional_content_head_design.md) | Cluster-conditional content head (queue item 2). | Refuted (same ceiling, diversity ~1.0–1.2) |

## 2. Correctness & fidelity

| Doc | Summary | Status |
|---|---|---|
| [`start_seq_rotation_audit_design.md`](start_seq_rotation_audit_design.md) | `predict_load` hard-codes rotation 0; ≥50% of rotations unreachable at `max_perm>1`. Flat-indexing fix + coverage probe. | Pending impl |
| [`audio_driver_split_design.md`](audio_driver_split_design.md) | preframr-audio: split the 1499-LoC `audio_driver.py` into a `render.py` core (what fidelity/fingerprint/batch import) vs `live.py` (alsa/ASID/MIDI/CLI). | Drafted, pending review |
| [`landed/tokenizer_profiling_tooling_design.md`](landed/tokenizer_profiling_tooling_design.md) | Torch-free tokenizer profiling (`tokenizer_profile` + `audit_primitives` reductions). | Landed (tokens 0.20.0) |
| `preframr-tokens:pipeline_trace.py` | Torch-free pass-by-pass pipeline tracer (no design doc): given a pipeline spec + a dump parquet, runs the real `RegLogParser` encode path with every `MacroPass` + parser stage instrumented and reports, per stage, which flag gated it, whether it fired, and the op-mix delta. `--isolate FLAG` re-runs with the flag off for a counterfactual proof of its effect; warns loudly on unrecognized spec names (the silent `slope`→`freq_trajectory` no-op class). Test what a spec does / verify a flag took effect without trusting docs. `python3 -m preframr_tokens.pipeline_trace`. | In flight (PR #17 → tokens 0.24.0) |
| [`landed/audio_fidelity_helper_design.md`](landed/audio_fidelity_helper_design.md) | Shared render-and-compare helper (`compare_renders`). | Landed (preframr-audio `fidelity.py`) |
| [`landed/tokenizer_alphabet_coverage_bug.md`](landed/tokenizer_alphabet_coverage_bug.md) | RegTokenizer alphabet-coverage bug. | Landed |
| [`landed/hvsc_version_pinning_design.md`](landed/hvsc_version_pinning_design.md) | Runner HVSC-version check wired into preflight. | Landed |

Open (not a design doc): the **~100-song round-trip audio CI gate** (≥95% within
tolerance) — see AGENTS.md "Land any time" (`compare_renders` helper + unit tests
landed; corpus-scale gate pending).

## 3. Efficiency & deploy envelope

| Doc | Summary | Status |
|---|---|---|
| [`streaming_unembed_ce_design.md`](streaming_unembed_ce_design.md) | Stream `output(chunk) + ce_chunk` in one checkpoint; eliminates the 8.6 GiB chunk slab, restores `batch_size=4` + prodlike wallclock. | Pending impl |
| [`orin_inference_optimization_design.md`](orin_inference_optimization_design.md) | Predict-host throughput: vocab shrink + GPU-resident constrained-decode (Orin ~4% GPU util at predict). | Pending impl |

Cross-axis: [`compound_token_design.md`](compound_token_design.md) (token budget;
primary in Generalization). Vocab shrink (tkvocab ~8× to 4096) is queued under
AGENTS.md "Predict-host envelope" (deferred).

## 4. Runner / experiment infra & process

Serves the Generalization axis: `generalization_metric_tracking_design.md` wires
the decisive content-tier audit + scorecard + cross-run ledger.

| Doc | Summary | Status |
|---|---|---|
| [`generalization_metric_tracking_design.md`](generalization_metric_tracking_design.md) | Make the decisive content-tier `per_class` audit a runner stage (not run by hand), add a generalization scorecard (per-eval_b-family content acc + spread + loop/prompt) to the report, and a tokenizer-hash-keyed cross-run ledger that auto-flags confounded comparisons. Reuses existing audits + the metric registry. | Drafted, pending impl (land with no run in flight; tokenizer-health metrics landed) |
| [`runner_iteration_efficiency_design.md`](runner_iteration_efficiency_design.md) | Per-run overhead: (1) cache key on parse/tokenize cargs — **landed**; (2) symlink-farm + RO dump mount (no 2.7 GB copy) — **landed**; (3) drop the post-step chown container — pending. | #1+#2 landed; #3 pending |
| [`flag_stage_routing_design.md`](flag_stage_routing_design.md) | `FLAG_STAGES` registry + `add_stage_args` for stage-aware flag forwarding (parse/tokenize/train). | Pending impl |
| [`auto_early_abort_design.md`](auto_early_abort_design.md) | Spec-declared `decision_rule` evaluated after each (arm, seed); writes a refutation stub on falsification. | Deferred (cloud-rental prereq) |
| [`max_parallel_arms_design.md`](max_parallel_arms_design.md) | `concurrent.futures` slot allocator with flock; refuses N>1 on single-GPU hosts. | Deferred (cloud-rental prereq) |
| [`resume_design.md`](resume_design.md) | Per-stage `_resume.json` manifest for partial-run recovery (dataset cache already covers parse+tokenize). | Deferred (partial coverage landed) |
| [`repo_focus_cleanup_scope.md`](repo_focus_cleanup_scope.md) | Plan to keep `preframr` (main) = core framework, moving orchestration + audits + tier data + design docs to `preframr-xpt`. | Largely executed (see [`landed/experiments_extraction_design.md`](landed/experiments_extraction_design.md)) |

The container split + extraction landings (experiments / tokens / audio
extraction, train/inference split, regdataset decomposition) are archived in
[`landed/README.md`](landed/README.md) under "Library extractions".

## 5. Data & corpus

Active work here is currently in specs/audits, not design docs; the design record
is archived. New tier / eval-set / augmentation / dataset-coverage docs go here.

| Doc | Summary | Status |
|---|---|---|
| [`landed/prodlike_tier_design.md`](landed/prodlike_tier_design.md) | Prodlike data tier (~4.4K train + 385 Eval-A + 8 Eval-B families). | Landed |
| [`landed/engine_fingerprint_evalb_design.md`](landed/engine_fingerprint_evalb_design.md) | 8 cross-engine Eval-B families + engine-family map. | Landed |
| [`landed/corpus_structural_index_design.md`](landed/corpus_structural_index_design.md) | One-shot CPU structural index of full HVSC. | Landed |
| [`landed/stage_dumps_basename_fix_design.md`](landed/stage_dumps_basename_fix_design.md) | Composer-subdir staging; recovers 50 prodlike dumps lost to basename collisions. | Landed |
| `preframr-aug:design/melody_transfer_augmentation_design.md` | Offline corpus-expansion (inaudible perturbation, voice permutation, cross-song transfer). | Moved to preframr-aug |

## Landed (archived)

Reference docs whose implementations are in HEAD, indexed by *kind* (not axis) in
[`landed/README.md`](landed/README.md): library extractions, data tiers +
infrastructure, audit + validation tooling, tokenizer/encoding (FREQ_TRAJ +
profiling), and bug fixes.

## Decision rules

- **Status header.** One-line status bullet on every doc; update it before the
  body.
- **Promotion thresholds.** 3σ-on-val_acc to flip a default; capacity-attenuation
  refuses if prodlike Δ < ¼ × mini Δ; per-Eval-B-* breakouts confirm
  cross-composer transfer.
- **Refuted alternatives.** Detailed evidence + a "do not revisit without …"
  condition go in `preframr_experiments/data/refuted/<exp>.md`; the design doc
  retains only the `Refuted` status header pointing there.
