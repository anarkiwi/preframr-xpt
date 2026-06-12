# Generation quality gate — scoring what the model *makes*, not just what it predicts

**Status:** Design (2026-06-12). Every decisive gate today is token-prediction accuracy
(content-tier per-class, val_acc); the only generation-side checks are pathology flags (loop
collapse, prompt-diversity) and an audition smoke test whose pass condition is a non-empty WAV. A
run could pass content-tier and still emit unmusical output — nothing would notice. This doc defines
the missing layer: a **standard generation cohort + scorecard + promotion gate**, built almost
entirely from existing primitives (`detect_tail_cycle`, `distinct_n`, `ngram_overlap`,
`engine_fingerprint`, `events.generate`, `render_to_wav`). Torch-free except generation itself;
CPU-runnable audits (`--device cpu` works for audits during live training).

## Principle

Prediction metrics are necessary, not sufficient. P6 (encoding principles) already concedes that
multi-modal targets must be scored "distributionally + by audition" — this gate is that scoring,
productized. It is **necessary-not-sufficient alongside the content tier**: promotion of any
default (encoding, sampling regime, conditioning treatment) requires both.

## The cohort (fixed, versioned, cheap)

- **Prompts:** a pinned set of ~24 held-out prompts (eval_a + eval_b composers, mid-song
  KEYFRAME-aligned), plus — once the [prompt interface](prompt_interface_design.md) exists — a
  pinned phrase-prompt set. Pinned like a tier list (`data/` list file), so cohorts are comparable
  across runs.
- **Sampling grid:** greedy + temperature {0.7, 0.9, 1.0} × top_k {0, 50} per prompt (one seed each;
  the grid is the diversity). Strict-grammar decode; an invalid emission triggers bounded resample
  and is **counted** (`invalid_rate` is itself a scorecard row).
- **Outputs:** ~1–2 windows per continuation for the standard cohort; 3–5 chained windows for the
  long-horizon reads ([`long_range_structure.md`](long_range_structure.md)). Decode all to writes;
  render a fixed subset to WAV.

## The scorecard

**1. Pathology flags (exist — promote to every canonical run, not on-request):**
`loop_collapse_rate` (`detect_tail_cycle`), prompt-conditioning `diversity_ratio` (real-vs-random ≥
1.2), `invalid_rate`, decoded-fraction (`event_gate` already computes `fully_clean_rate`).

**2. Write-domain structure metrics (new, cheap — computed on decoded writes / settled grids,
compared distributionally to a held-out reference set, JS divergence or KS):**
note-onset rate; NI interval histogram (step/leap mix); pitch-class entropy + tonal concentration;
gate/rest ratio; voice utilization; event-kind mix (NI/FD/PW/FLD/G proportions); DT (tempo)
histogram; repetition profile (onset-stream autocorrelation peak lags + strengths). Each row:
generated-vs-reference distance, alongside reference-vs-reference spread (the calibration band).

**3. Chip-native audio distance (new — the FAD analogue without an external embedding model):**
render → `preframr_tokens.engine_fingerprint.compute_fingerprint` per continuation → distributional
distance (mean pairwise + Fréchet over the fingerprint feature space) generated-vs-held-out, with
held-out-vs-held-out as the floor. Reuses the engine-clustering feature space that already encodes
"what kind of SID music this sounds like". External embeddings (CLAP etc.) are explicitly out of
scope v1 (envelope + dependency discipline); revisit only if the fingerprint space proves
insensitive.

**4. Memorization audit (new — required before any "the model composed this" claim):**
build a token-n-gram index over the training corpus (once per tokenizer hash, cached next to the
dataset cache). Per continuation: novel-fraction at n=8/16, **longest verbatim training match**
(in tokens and in frames), and nearest-training-neighbor id. Gate: longest-match and novel-fraction
thresholds calibrated from the first 2–3 baselines (corpus tunes quote each other — measure the
natural base rate before setting the bar; the same calibrate-then-floor discipline as
`GENERALIZE_MIN_VAL_ACC`). The phrase-prompt cohort reads this both ways: the continuation must not
be a corpus copy, and must not merely parrot the prompt.

**5. Human audition protocol (minimal, promotion-candidates only):**
blind A/B sheet — N=10 clip pairs (generated vs length-matched real, same render chain), two
questions per pair: which is real (discrimination), which do you prefer (preference). 2+ raters,
results committed next to the run report. This is deliberately small: its job is to catch
"scorecard green but obviously bad to ears," not to be a MOS study.

## Sampling regime (the gate's subject, and one open port)

The scorecard is computed **per sampling configuration** — its first scientific use is picking the
default regime (the grid above) instead of folklore ("constrained + low temp"). Open item folded in
here: **port the sampling-time validity mask to the event grammar** (`constrained_decode.StreamState`
speaks the parse-domain space; the event path currently relies on strict decode + resample). The
event grammar is self-delimiting and simple — kind-led bodies, disjoint token ranges — so the mask
is a small numpy state machine in `preframr_tokens.events`. Until it lands, `invalid_rate` measures
how much it is missed. Loop-escape policies (on detected tail cycle: raise T / re-anchor at a fresh
KEYFRAME) are v2, measured by the same scorecard.

## Wiring + lifecycle

Runs as a post-train runner stage beside the content-tier audit
(`generalization_metric_tracking_design.md` §1 — same hook, same image, `--device cpu` capable);
emits `generation_scorecard.json` merged into `metrics.json`; `report.py` shows it under the
content-tier headline. Land order: (1) cohort + flags + write-domain metrics (pure reuse + numpy);
(2) fingerprint distance; (3) memorization index; (4) event-grammar mask port; (5) audition sheet
template. Calibrate all thresholds on the first 2–3 event-model canonical runs, then freeze as
gates.
