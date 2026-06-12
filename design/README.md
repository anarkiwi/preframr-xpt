# Design notes index

**Orientation:** [`architecture_overview.md`](references/architecture_overview.md) — which
part lives in which repo (tokens / audio / framework / xpt / aug) and **why**, the
dependency layering, and how to **derive the release process** (PyPI-tag vs
image-VERSION; the public-PyPI-propagation gotcha). Read it before any cross-repo
change or release.

**Per-repo architecture references** (all docs live here in preframr-xpt; the code
they describe lives in the sibling repos under `/scratch/anarkiwi/`):
- [`tokens_architecture.md`](references/tokens_architecture.md) — **historical parsing
  reference (superseded 2026-06-11)**: documents the retired (op,reg,subreg,val) substrate;
  dump format + register map still correct. The current tokenizer is the event model — see
  gen2 `events/STATUS.md` + the updated
  [`sid_render_fidelity_contract.md`](references/sid_render_fidelity_contract.md) /
  [`verification_and_audits.md`](references/verification_and_audits.md).
- [`audio_architecture.md`](references/audio_architecture.md) — preframr-audio render pipeline
  + fidelity oracle.
- [`framework_architecture.md`](references/framework_architecture.md) — preframr train/predict/
  model + data path + generation gotchas.
- [`backlog_tokens_hardening.md`](encoding/backlog_tokens_hardening.md) — the **testing
  discipline** for preframr-tokens: real-pipeline structural/balance tests (catch the
  synthetic-df false-green class) + the SID-fixture policy + the pass-framework model. (The
  old RESID-completeness / dead-wood items are OBE — subsumed by the generator pipeline.)

## North star: LEARNABILITY (read this first)

The headline goal is **learnability** — a SID tokenization/ordering is good insofar as a
bounded (~TC⁰) autoregressive transformer can *cheaply represent its next-token map*:
minimize causal-state size + dependency horizon, prefer induction-head-expressible
(DEF→REF copy) structure over implicit per-frame counters, order tokens by the driver's
causal DAG. The lens doc — read before proposing any representation/encoding work — is
[`learnability_token_ordering_theory.md`](references/learnability_token_ordering_theory.md) (theory +
the training-free triage `audit/learnability_triage.py` that ranks encodings *without a
run*). **Everything else is subordinate:** correctness/byte-exactness is the *gate* that
lets re-encoding happen; compression/token-budget and parse-perf/runner/deploy are *infra*
that buy faster or cheaper learnability experiments — none is an end in itself, and a doc
that argues "compression" or "fidelity" should say which it is and how it serves the north
star. The decisive empirical lesson behind this: **model-side content interventions were
refuted at a ~0.13 ceiling that tokenizer-side `full_macros` then lifted** — learnability
is won on the *representation* side, and the architecture is already exonerated
(`framework_arch_test`).

This north-star **priority order** (learnability → correctness → efficiency → infra) is
the *lens* for deciding what to work on next. It is **orthogonal to the physical layout
below**, which groups docs by *subject* so the directory is browsable. Each doc still
carries a one-line `**Status:**` header so live / refuted / landed state is visible
in-thread. Refuted hypotheses also have one-paragraph stubs at
`preframr_experiments/data/refuted/<exp>.md` — read those before reopening a rejected
direction.

## How this index is organized (read before adding a doc)

Files live in **six subject-theme subdirs** — the top-level grouping. Add a new doc
under its **primary** theme; cross-link a secondary theme in prose if it spans two.

1. **[`references/`](references/)** — orientation maps + domain/literature background.
   Stable, "read-first" docs (architecture, the SID/driver/render references, the
   learnability lens, positioning surveys, the release/build playbook).
2. **[`encoding/`](encoding/)** — tokenization & representation research. The core
   thread: the generator-MDL encoding, the pitch model, melody/voice/role layers,
   the arbitration pipeline, codebooks.
3. **[`measurement/`](measurement/)** — generalization metrics, measurement plans,
   the val-acc gate calibration.
4. **[`performance/`](performance/)** — parse/tokenize throughput, predict-host
   (Orin) inference envelope, and training memory/wallclock.
5. **[`infra/`](infra/)** — the experiment runner: iteration-efficiency wins, the
   deferred cloud-rental program (resume/abort/parallelism), flag routing, the
   predict-rotation fix, and the audio-driver refactor.
6. **[`refuted/`](refuted/)** — hypotheses tested and rejected; kept in place for the
   record (status `Refuted`, with the evidence stub under `data/refuted/` where one exists).

(There is no `model/` theme: the model-side loss/head arc is closed — refuted in favour
of tokenizer-side representation — so those docs live in `refuted/`, and the one live
training optimization is a memory/wallclock doc under `performance/`.)

Plus **[`landed/`](landed/)** — archived docs whose implementation is in HEAD, indexed
by *kind* in [`landed/README.md`](landed/README.md).

**Cross-doc links are relative** within `design/` (e.g. `../theme/doc.md`). When you
move a doc between themes, fix the inbound links (and any `design/<theme>/<doc>.md`
references from specs/tests/AGENTS.md).

**Status header + lifecycle.** Every doc starts with a one-line `**Status:**` header,
updated before the body. Lifecycle:

- **Draft** / **Scoping** → **Pending impl** / **Drafted, pending trigger** →
  **In flight** (spec + impl shipped; running/awaiting verdict) →
- **Landed**: move the file to [`landed/`](landed/) and add a row to
  [`landed/README.md`](landed/README.md).
- **Refuted**: move the file to [`refuted/`](refuted/), set status to `Refuted`, and add
  a "do not revisit without …" stub at `preframr_experiments/data/refuted/<exp>.md`.
- **Deferred**: reviewed but parked on an external condition (e.g. cloud rental).
- **Reference**: positioning/strategy docs with no single impl.

## references/ — orientation & domain background

| Doc | Summary | Status |
|---|---|---|
| [`architecture_overview.md`](references/architecture_overview.md) | The map for deciding *which repo a change belongs in* and *deriving the release process* from the dependency layering. Read before any cross-repo change. | Reference |
| [`tokens_architecture.md`](references/tokens_architecture.md) | The retired (op,reg,subreg,val) parser + tokenizer: atom/op model, pass framework. Dump format + register map still valid; the shipped tokenizer is the event model (gen2 `events/STATUS.md`). | Reference (superseded 2026-06-11) |
| [`audio_architecture.md`](references/audio_architecture.md) | preframr-audio render pipeline (parsed DataFrame → PCM via resid-fp) + the fidelity-comparison oracle + fingerprinting. | Reference |
| [`framework_architecture.md`](references/framework_architecture.md) | The torch layer: train/predict/model wrapping tokens (parse+tokenize) and audio (render) with a torchtune body + lightning; the `anarkiwi/preframr` image. | Reference |
| [`learnability_token_ordering_theory.md`](references/learnability_token_ordering_theory.md) | **The north-star lens** (read before any representation work): theory of cheap next-token representability + the training-free `learnability_triage` that ranks encodings *without a run* (mini mode-collapses regardless of vocab, so it cannot pick direction). | Reference + tool |
| [`encoding_principles.md`](references/encoding_principles.md) | The orienting rubric for SID stream encoding: **fidelity × context-efficiency × learnability** (priority order), the learnability sub-principles (separability / locality / no-multiplexing / alphabet≠learnability / right-yardstick), and a per-change checklist. Other encoding designs check against this. | Reference |
| [`sid_render_fidelity_contract.md`](references/sid_render_fidelity_contract.md) | **Cite, don't re-derive:** SID render timing + the complete envelope mechanism (the ADSR bug is compare-change associated; the (phase × nibble) write-liveness matrix; gate-edge position is content) + the v3 preserved-vs-canonicalized split, each fact citing its preframr-audio test (the 2026-06-11 canonical reference suites). Oracle = `stream.canonical_writes`. | Reference (updated 2026-06-11) |
| [`verification_and_audits.md`](references/verification_and_audits.md) | **THE how-to-verify reference (v3).** Two properties: **canonical fidelity** → `stream.encode(verify=True)` self-check + events roundtrip suites; **canonicalization soundness** → the chip-semantics suites + the perceptual raw-vs-canonical A/B (productizing corpus-wide = open follow-up). Old `parse_audit`/`cb_div_audit`/residual-census = retired substrate. | Reference (authoritative, rewritten 2026-06-11) |
| [`voice_encoding_reference.md`](references/voice_encoding_reference.md) | How the 3 SID voices are carried in the token stream: voice is **packed into the FRAME (−128) val** (low 6 bits, base-4 digits = voice+1), NOT in the VOICE (−126) token. `_add_voice_reg` canonicalises reg + emits VOICE delimiters; `remove_voice_reg` inverts. Melody onsets are multiplexed across voices by this header. | Reference |
| [`sid_driver_ornament_reference.md`](references/sid_driver_ornament_reference.md) | **Background reference:** how C64 SID drivers generate per-frame ornament across pitch, pulse-width, and filter. Two mechanisms: (A) note-index semitone-offset cycles = arps (codebook); (B) parametric/table sweeps = vibrato/portamento/PW/filter. Filter is global; PW/filter sweeps persist across notes; gate-on ≠ note boundary. Sources: defMON, SID Wizard, Hubbard, Galway, C=Hacking #5. | Reference |
| [`digi_detection_reference.md`](references/digi_detection_reference.md) | Digi techniques + detection (C=Hacking #20; Mahoney's *Musik Run/Stop*), written to refine `dump_meta.is_digi` (which misses PWM digis) and correct a row-count exclusion process error. | Reference |
| [`release_build_cache.md`](references/release_build_cache.md) | **The one place** for release/build/test/cache: which host runs what (fogbank for non-GPU work; defroster for training), the proxpi cache + how to bust it after a PyPI release, per-repo release procedure (PyPI `v*` tag vs Docker VERSION/base bump), and the build-locally-in-parallel rule. | Reference (authoritative) |
| [`related_work.md`](references/related_work.md) | **Related-work survey** (deep-research, adversarially verified): close cousins per facet (NES-MDB/LakhNES, YM2413-MDB, desidulate, OctupleMIDI/CP/MMT/MMM, Meredith COSIATEC/MDL, GTTM, interval pitch) but the combination — generative LM over the RAW register stream + MDL DEF→REF codebook + learnability-theory ordering — is unaddressed; facet 3 (chip control-stream LM) is the thin gap. | Reference (positioning) |
| [`tokenization_vs_music_llms.md`](references/tokenization_vs_music_llms.md) | preframr's register-event + macro Unigram tokenization vs symbolic/MIDI, audio-codec, VQ; argues the content ceiling is tokenization-induced. | Reference (positioning) |

## encoding/ — tokenization & representation

**THE current direction — the generator-MDL pipeline.**
[`generator_mdl_representation.md`](encoding/generator_mdl_representation.md) is the canonical encoding and
**supersedes the former per-pass pitch/ornament/melody/residual-SET stack** (unified-pitch-encoding,
ornament-transfer, sweep-oscillation, the melody-channel/skeleton/gap-ladder/learnability designs, the
macro-zoo triage+review+consolidation, the residual-SET workorders, and the voice/role-lane designs — **all
removed 2026-06-05**; the generator model subsumes them). One self-verifying generator decomposition
`{HOLD,ACCUM,SWEEP,TABLE}` over every channel + a unified per-tune semitone LUT + a block-local DEF→REF
codebook: **lossless + residual-zero by construction, provenance-invariant.** Implementation is handed to an
agent in preframr-tokens: **`preframr-tokens/AGENT_TASK_generator_pipeline.md`** (see AGENTS.md "Current arc").
**Melody learnability is a layered stack** (the generator pipeline gives only layer 1): layer 1 =
de-ornamentation (generator); **layer 2 = interval-skeleton** (key-invariant onsets, 0.52 held-out > 0.41
ceiling); **layer 3 = de-multiplex AND causally order the lanes** — the *dominant* lever (deployed melody-onset
≈ 0 vs ~0.34 per-voice ceiling is cross-voice multiplexing). Layer 3's real lever is **causal-DAG ordering:
accompaniment roles before the melody role** (so melody is predicted with its harmony in-context, P4) — which
makes ROLE identification the mechanism, not a follow-up; plain physical voice-lanes can backfire. **Layer 4
(deferred hypothesis):** surface rhythmic/harmonic determinants + scale-degree anchoring (lossy; open only if
layer 3 plateaus). The self-directing work order [`melody_skeleton_impl.md`](encoding/melody_skeleton_impl.md) builds
**layers 2 AND 3** (BLOCKED on the generator pipeline). Layer-3 designs:
[`superframe_voice_lane_design.md`](encoding/superframe_voice_lane_design.md) (lane mechanics) +
[`role_lane_factorization.md`](encoding/role_lane_factorization.md) (role/causal-order mechanism). Layer 3 is
theory+measurement-motivated but **untested at deployment** → triage (lane-order variants + no other-content
regression) + one canonical run gate it.

| Doc | Summary | Status |
|---|---|---|
| [`generator_mdl_representation.md`](encoding/generator_mdl_representation.md) | THE encoding (above): `{HOLD,ACCUM,SWEEP,TABLE}` generators over all channels + per-tune pitch LUT + DEF→REF bank. Byte-exact + residual-zero on 1580 corpus tunes, every historically-hard engine, and SID-Wizard (91) + defMON (9) player output. Two of three ops already exist (`SWEEP_OP`=ACCUM, the osc-cycle=TABLE); only the triangle SWEEP + tuning/codebook are new. | **LANDED + released (deployed default; zoo deleted; shipped 0.45.0 → 0.46.x). Measurement: `measurement/generator_measurement_readiness.md`** |
| [`universal_multiresolution_pitch.md`](encoding/universal_multiresolution_pitch.md) | **The learnable+lossless+universal pitch model** (supersedes per-tune-LUT / content-tier-residual). Shared NOTE INDEX (semitone grid, note 49≈C5 — the universal prediction target) + **per-voice recovered note→freq TABLE** (~20/voice). Static notes are PURE (**83% of frames**); residual is nonzero ONLY for genuine modulation (~17%). Plus per-voice tuning + modulation in CENTS + intervals Δnote (`MELODY_INTERVAL`). EXACT validation on SWM/defMON/Hubbard/Galway. Foundation on `feat/universal-pitch-grid`; active wiring = the gated default-OFF `universal_pitch` flag. | **Active — validated exact on 4 trackers; gated wiring in progress** |
| [`melody_skeleton_impl.md`](encoding/melody_skeleton_impl.md) | NEXT layer on the generator freq channel: note segmentation (level-change∪gate) + **interval-from-previous** onset encoding (key-invariant) + within-note ornament as note-relative generator atoms. Held-out next-interval 0.52 > cross-tune ceiling 0.41. **Now also builds LAYER 3** (`voice_lane` de-mux into contiguous lanes). **SELF-DIRECTING** (§A start-gate polls tokens `origin/main`). | **Pending impl — executable handed to `preframr-tokens/AGENT_TASK_melody_skeleton.md`; auto-gated on the generator pipeline landing** |
| [`superframe_voice_lane_design.md`](encoding/superframe_voice_lane_design.md) | LAYER 3 (voice form): reorder the frame-major stream into voice-major lanes so the melody line is contiguous (short horizon, P3). Lossless permutation w/ byte-exact render-order inverse. The **voice-form** sister of `role_lane_factorization.md` (role-form); `melody_skeleton_impl.md` §4B builds on both. | **Reinstated — live (was wrongly deleted); untested at deployment, triage-gated** |
| [`role_lane_factorization.md`](encoding/role_lane_factorization.md) | LAYER 3 (role form — the truer target): roles HOP voices, so factor by musical role not physical voice (voice-lanes can split a melodic line). Harder follow-up after voice-lanes. | **Reinstated — live (was wrongly deleted); follow-up to voice-lanes** |
| [`speculative_encoding_pipeline.md`](encoding/speculative_encoding_pipeline.md) | **PROPOSAL** (architecture): replace the strict-order DESTRUCTIVE macro-pass chain with **claims + arbitration** — passes are non-destructive PROPOSERS emitting scored `Claim`s; an **arbiter** picks a lossless PARTITION maximizing the objective, per region AND per tune. Fixes one-pass-destroys-another + lets drum/skeleton/sweep/patch COMPETE so the best encoding wins (RESID becomes the true floor). | **Proposal — architecture, not built** |
| [`compound_token_design.md`](encoding/compound_token_design.md) | Approach D: compound-token tokenizer + parallel-attribute heads (CompoundWord / OctupleMIDI). Multi-attribute-per-token reorganization. Also an efficiency bet (token budget). | Draft, design review pending |
| [`instrument_program_codebook_design.md`](encoding/instrument_program_codebook_design.md) | Instrument-program codebook: collapse the note-associated macro cluster (the per-note `(waveform,AD,SR)` program) into codebook references; distinguishes note-associated vs not-note-associated driver operations. Supersedes the older instrument-state-codebook. | Design 2026-06-04 |
| [`log_to_swm_recompiler_design.md`](encoding/log_to_swm_recompiler_design.md) | A tool (does NOT exist yet) compiling a preframr **register log → SID-Wizard SWM** that re-renders to the SAME output (lossless = re-render-equivalence). Reuses the generator-MDL IR; pysidwizard `build_swm`; the player is the verifier. Path A brute-force wavetable → Path B structured/editable; reports the SID-Wizard-inexpressible residue. Makes generated tunes editable in a real tracker. | Design 2026-06-06 |
| [`audio_equivalence_normalization_design.md`](encoding/audio_equivalence_normalization_design.md) | Tokenizer-side normalization collapsing writes that render perceptually-equivalent into canonical forms. | Superseded — realized as the v3 canonical contract (gen2 `canonical_writes`, chip-measured rules; 2026-06-11) |
| [`backlog_tokens_hardening.md`](encoding/backlog_tokens_hardening.md) | preframr-tokens testing discipline: real-pipeline structural/balance tests (catch the synthetic-df false-green class) + SID-fixture policy + the 3-layer pass-framework model. | Pending impl (real-pipeline harness); rest OBE |

## measurement/ — metrics & generalization

| Doc | Summary | Status |
|---|---|---|
| [`generator_measurement_readiness.md`](measurement/generator_measurement_readiness.md) | **THE active measurement plan.** Generator pipeline LANDED + released (tokens 0.45.0 → 0.46.x). What to run: §1 static learnability triage (the cheap go/no-go); §2 op→tier wiring; §3 residual-in-key refragmentation; §4 the decisive canonical generator-vs-atomic A/B; §5 generalization-metric automation; §6 the melody caveat. | **Active — cheap reads runnable now; canonical run no longer release-blocked (needs image rebuild + re-cut + launch)** |
| [`generalization_metric_tracking_design.md`](measurement/generalization_metric_tracking_design.md) | Make the decisive content-tier `per_class` audit a runner stage (not run by hand), add a generalization scorecard to the report, and a tokenizer-hash-keyed cross-run ledger that auto-flags confounded comparisons. Reuses existing audits + the metric registry. | Drafted, pending impl (tokenizer-health metrics landed) |
| [`generalize_min_val_acc_floor_design.md`](measurement/generalize_min_val_acc_floor_design.md) | Calibrate `GENERALIZE_MIN_VAL_ACC` as 2/3 × median val_acc once 2-3 canonical baselines run (the generalization gate). | Pending impl (env hook exists, default 0/off) |

## performance/ — parse-perf & deploy envelope

| Doc | Summary | Status |
|---|---|---|
| [`parse_perf_proposal.md`](performance/parse_perf_proposal.md) | The parse-perf plan: synthesizes the cProfile scoping (parsing ~8.7 s/song; the bottleneck is the arbiter's per-claim `validate=True` fallback decode, not the `register_state` memo) into a prioritized, correctness-gated plan. Companion deliverable: the tokens test gate now runs under pytest-xdist. | Proposal 2026-06-03 |
| [`orin_inference_optimization_design.md`](performance/orin_inference_optimization_design.md) | Predict-host throughput: vocab shrink + GPU-resident constrained-decode (Orin ~4% GPU util at predict). | Pending impl |
| [`streaming_unembed_ce_design.md`](performance/streaming_unembed_ce_design.md) | Training memory: stream the unembed projection + per-chunk CE inside one gradient checkpoint so the unembed-chunk slab never fully materialises. Recomputed at `tkvocab=32768` the slab is ~2 GiB (not 8.6) and batch=4 already fits 24 GiB — so likely unnecessary now; back-pocket for if vocab/batch/seq grow. | Deferred (likely moot at 32768) |

Vocab shrink (tkvocab ~8× to 4096) is queued under AGENTS.md "Predict-host
envelope" (deferred). [`compound_token_design.md`](encoding/compound_token_design.md) is
also a token-budget bet (primary in encoding/).

## infra/ — runner, process & deploy

| Doc | Summary | Status |
|---|---|---|
| [`runner_iteration_efficiency_design.md`](infra/runner_iteration_efficiency_design.md) | Per-run overhead: (1) cache key on parse/tokenize cargs — **landed**; (2) symlink-farm + RO dump mount (no 2.7 GB copy) — **landed**; (3) drop the post-step chown container — pending. | #1+#2 landed; #3 pending |
| [`flag_stage_routing_design.md`](infra/flag_stage_routing_design.md) | `FLAG_STAGES` registry + `add_stage_args` for stage-aware flag forwarding (parse/tokenize/train). | Pending impl |
| [`start_seq_rotation_audit_design.md`](infra/start_seq_rotation_audit_design.md) | `predict_load` hard-codes rotation 0; ≥50% of rotations unreachable at `max_perm>1`. Flat-indexing fix + coverage probe. | Pending impl |
| [`audio_driver_split_design.md`](infra/audio_driver_split_design.md) | preframr-audio: split the 1499-LoC `audio_driver.py` into a `render.py` core (what fidelity/fingerprint/batch import) vs `live.py` (alsa/ASID/MIDI/CLI). | Drafted, pending review |
| [`cloud_rental_runner_design.md`](infra/cloud_rental_runner_design.md) | The deferred cloud-rental runner program (one doc, three parts): **§1 `--resume`** (reuse on-disk parse/tokenize), **§2 auto early-abort** (spec `decision_rule` replaces human `pkill`), **§3 `--max-parallel-arms`** (concurrent arms across GPUs). Land order §1→§2→§3. | Deferred (cloud-rental prereq) |

## refuted/ — tested & rejected

Kept for the record; where a bet was actually run, detailed evidence + the "do not
revisit without …" condition live in `preframr_experiments/data/refuted/<exp>.md`.

| Doc | Summary | Status |
|---|---|---|
| [`multi_modal_objective_design.md`](refuted/multi_modal_objective_design.md) | Umbrella framing of the per-token CE bottleneck on the multi-modal content tier. **Carries the model-side anti-queue** (the do-not-re-attempt list salvaged from the retired `model_loss_queue`). | Refuted (B/C/A all rejected) |
| [`per_voice_aux_supervision_design.md`](refuted/per_voice_aux_supervision_design.md) | Per-voice auxiliary supervision heads (gate/pitch/waveform/ADSR) forcing musical state into the body's hidden activations. | Refuted by class (model-side content intervention); never run |
| [`per_tier_heads_design.md`](refuted/per_tier_heads_design.md) | Shared body + 4 tier heads + router; MoS-NLL content head (Approach C). | Refuted at prodlike (router saturates) |
| [`content_diffusion_design.md`](refuted/content_diffusion_design.md) | D3PM absorbing-state discrete-diffusion content head (Approach A). | Refuted (sampling-side; no CE change) |
| [`cluster_conditional_content_head_design.md`](refuted/cluster_conditional_content_head_design.md) | Cluster-conditional content head (queue item 2). | Refuted (same ceiling, diversity ~1.0–1.2) |
| [`motif_templates_v2_impl_design.md`](refuted/motif_templates_v2_impl_design.md) | Value-slotted motif templates (MotifDict v2): template token + content slot(s), shape-keyed mining, lossless expand. De-fragments the motif vocab and exposes motif-carried content to the content tier. (The corpus-mined motif-pass v1 design was deleted; this v2 doc is the surviving record of the refuted motif direction.) | **Refuted 2026-05-27** (built + A/B'd; content-tier did not beat no-motif; `data/refuted/motif_pass.md`) |
| [`sequence_order_normalization_design.md`](refuted/sequence_order_normalization_design.md) | Collapse the per-frame write-order DoF that is audio-safe to canonicalise. Audit: only the canonical voice-respecting reorder is inaudible (it is the order dumps already use — a near-no-op); 84% of repeated writes are genuine sub-frame modulation content. Audit `audit/audit_seq_order_norm.py` kept as instrument. | **Refuted as a lever 2026-05-27** (`data/refuted/sequence_order_normalization.md`) |

## landed/ — archived (impl in HEAD)

Reference docs whose implementations shipped, indexed by *kind* in
[`landed/README.md`](landed/README.md): library extractions, data tiers +
infrastructure, audit + validation tooling, tokenizer/encoding (FREQ_TRAJ +
profiling), and bug fixes. Notable threads:

| Doc | Summary | Status |
|---|---|---|
| [`landed/unified_oscillation_primitive_design.md`](landed/unified_oscillation_primitive_design.md) | Unified `FREQ_TRAJ` op + `FREQ_NUDGE` — the representation win; the live deployment tokenizer before the generator pipeline. | Landed (tokens 0.16/0.17) |
| [`landed/trajectory_anchoring.md`](landed/trajectory_anchoring.md) | Gate/sweep-anchored FREQ_TRAJ origins (the original encoding hypothesis + full mini-A/B history). | Landed (tokens 0.25.0, preframr 0.2.6) |
| [`landed/freq_trajectory_anchoring_impl.md`](landed/freq_trajectory_anchoring_impl.md) | Tokens-side impl spec for `TrajectoryAnchorPass`. | Landed (tokens 0.25.0) |
| [`landed/freq_v0_interval.md`](landed/freq_v0_interval.md) | `--freq-v0-interval`: encode V0 as a signed interval from the previous voice onset. | Landed (tokens 0.26.0) |
| [`landed/freq_onset_channel.md`](landed/freq_onset_channel.md) | `--freq-onset-pass` (FREQ_ONSET op48): re-tag residual op0 SET on TRAJ_REGS → 1-token onset. | Landed (tokens 0.27.0) |
| [`landed/melody_merge_split.md`](landed/melody_merge_split.md) | `--melody-merge-split`: split cross-melody-boundary merges so pitch is a separable target. | Landed (tokens 0.28.0) |
| [`landed/onset_loss_prioritization.md`](landed/onset_loss_prioritization.md) | `--onset-loss-weight`: up-weight rare FREQ V0-onset in CE. | Landed (preframr 0.2.8) |
| [`landed/tokenizer_profiling_tooling_design.md`](landed/tokenizer_profiling_tooling_design.md) | Torch-free tokenizer profiling (`tokenizer_profile` + `audit_primitives`). | Landed (tokens 0.20.0) |
| [`landed/audio_fidelity_helper_design.md`](landed/audio_fidelity_helper_design.md) | Shared render-and-compare helper (`compare_renders`). | Landed (preframr-audio `fidelity.py`) |
| [`landed/tokenizer_alphabet_coverage_bug.md`](landed/tokenizer_alphabet_coverage_bug.md) | RegTokenizer alphabet-coverage bug. | Landed |
| [`landed/hvsc_version_pinning_design.md`](landed/hvsc_version_pinning_design.md) | Runner HVSC-version check wired into preflight. | Landed |
| [`landed/prodlike_tier_design.md`](landed/prodlike_tier_design.md) | Prodlike data tier (~4.4K train + 385 Eval-A + 8 Eval-B families). | Landed |
| [`landed/engine_fingerprint_evalb_design.md`](landed/engine_fingerprint_evalb_design.md) | 8 cross-engine Eval-B families + engine-family map. | Landed |
| [`landed/corpus_structural_index_design.md`](landed/corpus_structural_index_design.md) | One-shot CPU structural index of full HVSC. | Landed |
| [`landed/stage_dumps_basename_fix_design.md`](landed/stage_dumps_basename_fix_design.md) | Composer-subdir staging; recovers 50 prodlike dumps lost to basename collisions. | Landed |

The container split + extraction landings (experiments / tokens / audio extraction,
train/inference split, regdataset decomposition) are archived in
[`landed/README.md`](landed/README.md) under "Library extractions".

**Elsewhere (not in this repo):**
- `preframr-tokens:pipeline_trace.py` — torch-free pass-by-pass pipeline tracer (no
  design doc): runs the real encode path with every `MacroPass` + parser stage
  instrumented and reports, per stage, which flag gated it, whether it fired, and the
  op-mix delta; `--isolate FLAG` for a counterfactual. (In flight → tokens 0.24.0.)
- `preframr-aug:design/melody_transfer_augmentation_design.md` — offline corpus
  expansion (inaudible perturbation, voice permutation, cross-song transfer). Moved
  to preframr-aug.

Open (not a design doc): the **corpus-scale canonicalization-soundness audit** — productize
the perceptual raw-vs-canonical A/B render (currently a gen2 tmp probe on the 5 drivers) as a
corpus-wide check. The old `cb_div_audit.py` register-equivalence gate belonged to the retired
substrate; under v3 the per-encode self-verify covers canonical fidelity and this audit covers
the canonicalization rules (see [`references/verification_and_audits.md`](references/verification_and_audits.md)).

## Decision rules

- **Status header.** One-line status bullet on every doc; update it before the body.
- **Promotion thresholds.** 3σ-on-val_acc to flip a default; capacity-attenuation
  refuses if prodlike Δ < ¼ × mini Δ; per-Eval-B-* breakouts confirm cross-composer
  transfer.
- **Refuted alternatives.** Move the doc to [`refuted/`](refuted/); detailed evidence +
  a "do not revisit without …" condition go in
  `preframr_experiments/data/refuted/<exp>.md`; the design doc retains only the
  `Refuted` status header pointing there.
