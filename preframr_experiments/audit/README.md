# Audit suite

Two layers, deliberately separated:

- **Reusable, tested modules** (this dir, importable + CLI) — the measurements and reads
  the program relies on repeatedly. Each has a unit test in `tests/`. Use these; do not
  re-derive a verdict from a one-off `/scratch/tmp` script.
- **`probes/`** — committed reference snapshots of one-shot diagnostics that back a
  documented (live or refuted) finding. Verbatim, no test, not parameterised. Kept so the
  claims stay reproducible after `/scratch/tmp` is cleaned; promote one to a tested module
  only if it becomes a recurring read.

## Reusable modules (tested)

| module | what it decides | test |
|---|---|---|
| `audit_checkpoint_per_class.py` | forward eval blocks through a ckpt → `audit_per_class.json` (per-class by token-id, per-tier, content_over_structural) | (covered via `tier_accuracy` in preframr-tokens) |
| `per_class_acc_audit.py` | per-class + per-tier accuracy from prediction CSVs | `test_per_class_acc_audit.py` |
| **`content_tier_report.py`** | **the decisive content gate**: per-tier content/structural acc (pooled + per-seed mean±std, Δ vs baseline), `eval_b_*` per-family stratification, and a **by-op** breakdown within the content tier (spotlight op, default FREQ_TRAJ op45). Auto-discovers arms+seeds under `--results-root`. Reproduces the STAGE 2 prodlike numbers exactly. | `test_content_tier_report.py` |
| `ordinal_tolerance_audit.py` | tolerance-band accuracy for a contiguous-bin op (e.g. FREQ_TRAJ val 0..256): is a low exact-match acc near-miss noise or a real miss? | `test_ordinal_tolerance_audit.py` |
| `loop_detection_audit.py` | generation loop/collapse detection | `test_loop_detection_audit.py` |
| `sid_register_plausibility.py` | hardware-grounded model-quality signal: per-frame `(n,25)` register state → counts nonsensical SID usage a poor generalizer emits (gate-on with no waveform; ring-mod off an idle source; noise+tone LFSR-lock; oscillator stuck in test), each calibrated to 0 on the real-tune corpus (which *defines* plausible). Verdict PASS/WARN/IMPLAUSIBLE on the per-frame violation rate. Complements `melody_features` (musicality) with "would this even sound". | `test_sid_register_plausibility.py` |
| `prompt_conditioning_audit.py` | prompt-conditioning (does output track the prompt) | `test_prompt_conditioning_audit.py` |
| `engine_fingerprint.py`, `digi_audit.py`, `irq_audit.py`, `audit_eval_leak.py`, `audit_macro_fidelity_probe.py`, `audit_seq_order_norm.py`, `seq_budget_coverage.py`, `build_content_clusters.py`, `build_prodlike_4x_list.py`, `aggregate_corpus_index.py`, `augment_voice_permutation.py`, `audition_cohort_render.py`, `generate_for_audit.py` | corpus/data + generation audits wired into the runner or used as staged steps | mixed |
| `ablate_pwfilter.py` | freq-core ablation (drop PW+filter, inject PW midpoint) for `freq_core_ablation_mini` | **test pending** |
| `extract_sid_melody.py` | per-(dump,voice) melodic onset sequences for the melody data-gap ladder (L1/L2/L3); `--channels` emits the interleaved skeleton+ornament stream for the channel-factorization probe | `test_melody_channel_probe.py` (channels) |
| `melody_channel_probe.py` | channel-factorization verdict (design/melody_channel_factorization.md): is melody predictability stolen by multiplexing ornament into the prediction position? interleaved vs skeleton_only held-out skeleton accuracy (`--seeds N` for mean delta) | `test_melody_channel_probe.py` |
| `melody_multiplex_probe.py` | cross-voice multiplexing verdict: held-out skeleton acc on the ALL-3-voice frame-multiplexed stream (`extract_sid_melody --multiplex`), pooled + per-voice, vs the single-voice reference. `--render-dir`+`--dumps` renders a polyphonic GT + model-prediction audition. | (probe; numbers self-checking) |
| `unified_pitch.py` | unified pitch encoder/decoder (design/unified_pitch_encoding.md): semitone note→freq LUT, level-change∪gate `segment_notes`, `_fit_descriptor` (PLAIN/OCTAVE/ARP/SLIDE/VIB/RESID), `encode_voice`/`decode_notes` (round-trip; PLAIN LUT-exact). `--out` extracts per-note {skel,desc} for the probe. | `test_unified_pitch.py` |
| `unified_pitch_audition.py` | faithful A/B of the encoding on a REAL tune: renders `<tune>_raw.wav` (all original register writes) vs `<tune>_unified.wav` (same tune, per-voice freq replaced by unified encode→decode on the real timeline; timbre/rhythm kept). The representative ground-truth audition. Runs in the image. | (render demo) |
| `unified_pitch_probe.py` | generalization test on the unified encoding: held-out skeleton next-interval acc + cross-tune 2-gram ceiling + ornament emission/JS(type); `--render-dir` decodes GT vs model free-run to WAV. Result: skeleton 0.518 (beats ceiling, ≫ prior 0.225); ornament emits at corpus rate. | (probe; numbers self-checking) |
| `ornament_transfer_probe.py` | ornament-transfer A/B (design/ornament_transfer.md): RAW per-frame vs PARAM per-note-descriptor ornament — emission rate + JS(type) distributional transfer, free-run over held-out tunes; `--render-dir` renders gt/raw/param ornament auditions. Verdict: parametric refuted; per-note alignment is the lever. Pairs with `extract_sid_melody --ornament`. | (probe; numbers self-checking) |
| `melody_channel_render.py` | audition render for the channel/multiplex probes: triangle-voice WAVs of ground-truth melody, model continuations, ornament-preserved demo, 3-voice polyphony (GT + multiplex prediction). Runs in the GPU image. | (render demo; verified via probe) |

## Probe triage (the `/scratch/tmp` consolidation, 2026-05-27)

`/scratch/tmp` is ephemeral; these were one-off research probes. Disposition below.
**PROMOTED** → a tested module above. **ARCHIVED** → `probes/` (committed reference).
**DROP** → left in `/scratch/tmp` until cleanup; the conclusion lives in the cited
doc, so the code is not preserved (recoverable from the description here if ever needed).
**→aug** → belongs to the augmentation thread (preframr-aug), relocated there.

| probe | backs | disposition |
|---|---|---|
| `freqtraj_distribution.py` | trajectory_anchoring (by-op read) | **PROMOTED** → `content_tier_report.py` |
| `parse_per_class.py` | AGENTS STAGE 2 read | **PROMOTED** → `content_tier_report.py` |
| `audit_freqtraj_tolerance.py` | AGENTS, trajectory_anchoring | **PROMOTED** → `ordinal_tolerance_audit.py` |
| `gate_anchor_probe.py` | trajectory_anchoring (gate-anchor hypothesis) | **ARCHIVED** |
| `raw_gate_anchor_confirm.py` | trajectory_anchoring (raw-dump confirm) | **ARCHIVED** |
| `intrinsic_anchor_probe.py` | trajectory_anchoring (intrinsic origin recovery) | **ARCHIVED** |
| `freqtraj_interval_probe.py` | trajectory_anchoring (interval reducibility) | **ARCHIVED** |
| `interval_from_dataset.py` | trajectory_anchoring (post-transform interval) | **ARCHIVED** |
| `raw_atom_diag.py` | trajectory_anchoring (raw atom histogram) | **ARCHIVED** |
| `inspect_frames.py` | trajectory_anchoring / intra-frame reorder | **ARCHIVED** |
| `adf_probe.py` | audio-df intra-frame sensitivity | **ARCHIVED** |
| `perf_probe.py` | orin_inference_optimization_design | **ARCHIVED** |
| `motif_v1_hang_repro.py` | refuted/motif_pass (known regression repro) | **ARCHIVED** |
| `compare_motif_v2_vs_prior.py` | refuted/motif_pass | DROP (refuted; conclusion in stub) |
| `motif_tail_scan.py` | motif designs | DROP (refuted) |
| `parse_motif.py`, `parse_motif_v2.py` | refuted/motif_pass | DROP (refuted) |
| `motif_compression.py` | motif designs | DROP (refuted) |
| `motif_dryrun_soundness.py` | motif designs | DROP (refuted) |
| `structure_content_probe.py` | motif structure/content balance | DROP (refuted) |
| `decompose_order.py` | sequence_order_normalization | DROP (refuted; order norm) |
| `structure_seq_probe.py` | sequence_order_normalization | DROP (refuted; order norm) |
| `engine_content_norm.py` | cross-engine content divergence | DROP (one-off) |
| `evalb_stratify.py` | eval_b stratification | DROP (subsumed by `content_tier_report`) |
| `op_coverage_probe.py` | how each content dim is tokenized | DROP (one-off) |
| `llm_contrast_probe.py` | tokenization_vs_music_llms design | DROP (conclusion in design) |
| `tok_dist.py` | token distribution | DROP (one-off) |
| `tokenize_only.py` | parse+tokenize both arms (hardcoded) | DROP (one-off) |
| `ablate_pwfilter.py` | freq_core_ablation | DROP (stale copy; canonical is `audit/ablate_pwfilter.py`) |
| `gen_mt_wavs.py` | melody transfer | **→aug** |
| `mt_adsr.py` | melody-transfer ADSR eligibility | **→aug** |
| `mt_audition.py` | melody-transfer audition | **→aug** |
| `percussion_audio.py` | percussion onset audio | **→aug** |
| `percussion_probe.py` | percussion onset detection | **→aug** |
