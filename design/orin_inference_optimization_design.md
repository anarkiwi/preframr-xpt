# orin_inference_optimization — design note

**Status (2026-05-19):** open design; no implementation yet. Captures
the bottlenecks profiled during the `accuracy_push_prodlike_4x`
audition on Orin NX, so future work has a concrete baseline.

**Update (2026-05-25):** two diagnoses below still hold, but the
vocab picture changed materially after the FREQ_TRAJ tokenizer rework
(preframr-tokens 0.16–0.18). Two independent problems: (1) *throughput*
— GPU 4% util from a per-token Python loop + `.item()` CUDA→CPU sync;
(2) *full-context memory wall* — 512 MB logit slab at vocab=32768.
Code paths moved: decode loop is now
`preframr/inference/predict.py::_predict_constrained` (loop at
:198, sync at :194/:211); `StreamState` is now
`preframr_tokens.constrained_decode` (torch-free numpy). Re-sequenced
plan + refreshed vocab numbers below.

## Measured baseline (Orin NX, anarkiwi/preframr-jetson)

- **Wallclock:** ~5–8 min per audition WAV (`PROMPT_SEQ_LEN=512`,
  `MAX_SEQ_LEN=1024`, compile + Triton cache warm via
  `/scratch/preframr/inductorcache`).
- **GPU `GR3D_FREQ`:** mean **4.1%**, peak **69%** across a 209-sample
  tegrastats window (2 s interval) during a full render.
- **RAM:** peak 4.3 GB of 15.6 GB unified memory (27%).
- **VDD_IN power:** mean 6.4 W, peak 9.4 W (idle baseline ~5.8 W).
- **Full context blocked:** `PROMPT=2048 / MAX=8192` audition is gated
  on a vocab shrink — logit head at vocab=32,768 is
  `1 × 8192 × 32768 × 2` ≈ **512 MB per forward**.

## Where the time goes

The decode loop in
`preframr/inference/predict.py::_predict_constrained` is a
Python for-loop over `n` tokens. Each iteration does:

1. `model(x, input_pos=..., mask=...)` (KV-cache step forward)
2. `_last_token_logits(...)` slice
3. `state.mask_logits(logits)` — constrained-decode validator
4. `_tt_sample(masked, temperature, top_k, q=_q())`
5. `state.update(tok.item())` — **forces CUDA→CPU sync per token**
6. `generated = torch.cat([generated, tok], dim=-1)` — host-side append

Per-token GPU work is ~10–20 ms; the rest of the per-token budget is
Python + the `tok.item()` sync. At 512 generated tokens × ~700 ms
total wallclock per token, the GPU is idle 96% of the time.

## Optimisation ladder

Each row is independent; later rows build on earlier ones but most
can ship on their own.

| # | Change | Expected gain | Risk |
|---|---|---|---|
| 1 | **Vocab shrink** (`--tkvocab 16384` or smaller) — see [vocab analysis] | 2–4× logit-head compute + memory; unblocks full-context audition | Tokenizer re-train; need UNK-rate audit (see `melody_transfer` doc's framing of atom-vs-merge IDs) |
| 2 | **CUDA-graph capture** of the per-token decode step | 5–10× per-token throughput (eliminates Python overhead between forwards) | `state.mask_logits` is data-dependent; capture must be partitioned around it OR the mask must move to GPU-side |
| 3 | **GPU-resident constrained-decode state** | enables (2) end-to-end; removes `tok.item()` sync | `StreamState` is currently pure-Python + numpy; needs Triton kernel or torch.compile-friendly rewrite |
| 4 | **Speculative decoding** with a tiny draft model | 2–3× tokens/s if accept rate >40% | Draft-model train cost; constrained-decode acceptance interaction |
| 5 | **Streaming-window KV cache** (cap attention to last N tokens at predict time) | Mainly memory; small wallclock gain unless N << MAX | Quality loss past N; needs A/B on val_acc-at-position |
| 6 | **Block-wise generation** (predict K tokens then validate) | 1.5–2× if K=4 acceptable | Constrained-decode is a hard constraint, not soft — needs rejection sampling |
| 7 | **Inductor max-autotune** (current Orin run uses `--no-max-autotune`) | 10–30% per-step | Autotune wallclock cost on cold cache; amortise via shared NFS cache |

## Vocab shrink interaction (cross-doc)

**Pre-rework (2026-05-19):** atom alphabet 10,890; tkmodel 32,768
slots; 8,358 distinct IDs used, all id < 8,192.

**Post-rework (2026-05-25, `full_macros_prodlike` 0.18.0 train split):**
the FREQ_TRAJ rework roughly halved the alphabet and the used set:
- atom alphabet **5,492**; tkmodel still 32,768 slots.
- only **2,929 distinct IDs used (8.9%)**, usage-weighted **1.23
  atoms/token** (top tokens are single atoms — compression is the
  atom-level macros, not Unigram merges).
- **Correction to the pre-rework "all used IDs < 8,192" claim:** the
  used IDs now span the *whole* range (max used id **32,766**); only
  660 are < 4,096 (66.7% of token mass), 1,105 < 8,192 (69.8%). So a
  naive cap (ID truncation) would drop ~1/3 of token mass —
  **vocab shrink REQUIRES a tokenizer re-train** at the smaller cap
  (Viterbi reassigns IDs by score), not truncation.

For inference optimisation purposes:
- Only ~2,929 IDs ever fire, so **`tkvocab` ∈ {8,192, 4,096}** are the
  candidate caps; ≤ 8,192 fits the logit slab and embedding table in
  unified memory and unblocks full-context Orin. The floor is an
  empirical question for the UNK audit (the 5,492-atom alphabet means
  a cap < alphabet drops rare atoms → UNK; only the audit over all
  admitted SIDs decides if that's safe).
- **Pre-shrink audit required** — re-tokenize 65k admitted SIDs at the
  candidate cap and assert UNK rate ≈ 0 before training.

**Probe 0b preliminary (2026-05-25, analytical from full_macros_prodlike
train `tokens.csv` per-atom counts):** alphabet 5491 atoms. **tkvocab 8192 →
UNK 0** (trivially: 8192 > 5491, every atom slots; safe 4× trim). **tkvocab 4096
→ ~0.033% UNK floor** (top-4096 atoms cover 99.967%; 1395 rare atoms dropped) —
promising 8× trim but the floor is a train-atoms-only lower bound; confirm with
the full re-tokenize incl. held-out eval coverage before committing. **tkvocab
2048 → 1.47% UNK, ruled out.** So the safe default is 8192; 4096 is the
aggressive option pending the confirmatory re-train.

## Decode-loop GPU-resident rewrite (item 3)

The blocker on items 2 and 6 is the Python-side state machine in
`preframr/core/constrained_decode.py::StreamState`. It enforces:

1. Frame-budget invariant (sval ≤ irq per frame).
2. Voice-block atomicity (a frame's voice-reg group is contiguous).
3. Op-bigram restrictions (per `CTRL_BIGRAM_TABLE` and similar).

All three can be expressed as 1-D index gathers over a precomputed
mask table indexed by `(state_class, vocab_id)`. With state_class on
GPU and the mask table resident, `mask_logits` becomes a single
`logits.masked_fill_(~mask_table[state], NEG_INF)` per step. State
update is one `state = transition_table[state, tok]` lookup.

Once both are GPU-resident, the whole decode step is a single
captured CUDA graph: forward → mask → sample → state-transition,
zero CPU sync per token.

## Predict-host envelope check

The Orin NX is the deployment target. The training-host (RTX 4090)
isn't constrained by these issues — its 24 GB VRAM holds the
32,768-vocab logit head trivially. So **every optimisation here is
an Orin-specific or-better win**:

- Vocab shrink: predict-host win + training-host smaller logit head
  (proportional to vocab, helpful for streaming-unembed-CE).
- CUDA-graph decode: predict-host only (training doesn't autoregress).
- Speculative decode: predict-host only.

The training-host can already saturate batch-size to GPU memory at
the existing context budget; further gains on the predict host
unblock larger seq_len at deployment without re-training.

## Phase plan (re-sequenced 2026-05-25)

Two cheap probes decide how much of the expensive work is needed; run
them on the post-rework `full_macros_prodlike` checkpoint + tokenizer.

| Phase | Scope | Pass gate |
|---|---|---|
| 0a. **Constrained-decode necessity probe** | Instrument `mask_logits` over the finished `full_macros` ckpt: how often does the mask reject the model's top-1? (GPU; run after both arms so it doesn't contend with training) | If top-1 rejection is rare → unconstrained decode + post-hoc validity removes the sync with ~zero engineering, and Phase 3 is unnecessary. Decides Phase 3. |
| 0b. **UNK-rate audit** | Re-tokenize 65k admitted SIDs at `--tkvocab 8192` then `4096` (CPU; route to fogbank, no GPU contention) | smallest cap with UNK rate ≈ 0 |
| 1. **Vocab shrink — fold into the baseline re-run wave** | Bake the chosen `tkvocab` into the "re-run ALL baselines from scratch" arc (it re-trains anyway → zero extra cost; smaller logit head also helps streaming-unembed-CE) | val_acc parity within σ vs 32768 |
| 2. Orin full-context smoke | Shrunk model, predict at `PROMPT=2048 / MAX=8192` | OOM-free, wallclock < 10 min/wav |
| 3. CUDA-graph decode rewrite (only if 0a says masking is load-bearing) | Move `StreamState` to GPU; mask table indexed by `(state_class, vocab_id)`; capture graph. Precursor: confirm the 3 invariants collapse to a finitely enumerable state_class space | tokens/s ≥ 3× baseline at full context |
| 4. Speculative decode (optional) | Draft model + acceptance under constrained-decode | tokens/s ≥ 2× phase 3 |
| 5. Inductor `max-autotune` (near-free) | Flip off `--no-max-autotune`; re-warm shared NFS inductor cache once | 10–30% per-step, no regression |

## Open questions

- **Constrained-decode necessity at predict.** The mask enforces
  structural admissibility; relaxing it (or moving to soft loss
  during training) would simplify the decode path significantly.
  Audit: how often does `mask_logits` reject the model's top-1 in
  practice? If rare, an unconstrained decode + post-hoc validity
  check might be cheaper.
- **Mid-render checkpoint races.** The 2026-05-19 audition retry
  failed when Lightning rewrote `best-*.ckpt` mid-load. Audition
  wrappers should copy the ckpt to a stable path before launching.
  Cheap fix; orthogonal to the perf ladder.
- **Audition song eligibility.** Of 323 eval-A entries, only 200
  have `.0.blocks.npy` on disk (61.9%). Audition wrappers should
  pre-filter by block-file existence rather than scanning
  sequentially. Orthogonal but caught two failed renders.

## Update (2026-05-26): instrumented XPU measurement — sync is NOT the bottleneck

Profiled the real decode path on **vek-x (Intel Arc, Meteor Lake-P, i915)**
with a current post-rework `mini_body_large` baseline checkpoint
(`voice_permutation_mini_body_large/baseline/seed0`, **vocab 8192**) via
`/scratch/tmp/perf_probe.py` (loads the ckpt, self-generates a structurally
valid prompt from the constrained decoder, then times steady-state per-token
decode). XPU vs CPU is the *same* image (`anarkiwi/preframr-xpu` on the new
8.5 GB slim base); CPU = run without `--device /dev/dri`.

This is XPU, not Orin/CUDA, but it's the **identical** `_predict_constrained`
path, so the structural conclusions carry. (The 2026-05-19 "~700 ms/token,
4% util" figure was at vocab=32,768 — the 512 MB logit slab dominated; at
vocab 8192 the picture below is very different.)

**Headline (ms/token, constrained decode, compile on):**

| | CPU | XPU | XPU/CPU |
|---|---|---|---|
| constrained | 9.1 | 19.5 | 2.1× |
| unconstrained (torchtune `generate`) | 11.2 | 23.0 | 2.05× |

**The per-token `tok.item()` sync is not the dominant cost at this vocab.**
Two independent proofs:
1. The XPU/CPU ratio is ~2× **with and without** constrained decode — if the
   sync (only present in the constrained path) were the cause, the ratios
   would diverge. They don't.
2. Holding the constrained path (same `.item()` count) and only shrinking the
   KV cache `--max-seq-len` 8192→512 cut CPU 9.1→3.7 and XPU 19.5→12.5. The
   sync count is identical across those two runs; the 7 ms came purely from
   attention width.

**Where the time actually goes (XPU profiler, 64 steps):** fused attention
`micro_sdpa` **48%**, `aten::mm` 18%, `aten::copy_` **15% (~94 copies/token)**.
The fused SDPA processes the **full allocated cache width every step**
(torchtune does not slice k/v to `cache_pos`), so at MAX=8192 every decode
step attends 8192-wide from the first generated token.

**Cost vs cache width (XPU, clean; CPU noisy under host load):**

| cache L | CPU | XPU |
|---|---|---|
| 512 | 3.7 | 12.5 |
| 1024 | 6.0 | 12.4 |
| 2048 | 7.2 | 13.7 |
| 4096 | (noisy) | 15.2 |
| 8192 | ~9–11 | 19.2 |

XPU fits `cost(L) ≈ 12.0 + 0.00087·L` ms/token: a **~12 ms fixed floor**
(small per-token GEMMs + ~94 copies/token + overhead) plus attention ~linear
in cache width.

### Implications for the optimisation ladder

- **Item 5 (streaming-window / right-sized KV) is undervalued, not "small
  wallclock gain."** Because per-step attention scales with the *allocated*
  cache width, windowing to the filled length is a real wallclock win on
  **both** platforms. Production projection (PROMPT=2048/MAX=8192, gen 6144,
  avg ctx ~5120): current 19.2 → windowed `cost(5120)` ≈ 16.5 ms/tok ≈ **~14%
  on XPU**; up to ~35% for short generations. **Cheapest variant, zero code:**
  set `--max-seq-len` to the run's actual total (cost tracks cache width
  directly), e.g. `--max-seq-len 4096` → 15.2 vs 19.2 = 21%. Dynamic
  per-step windowing only adds value when MAX must stay large but most steps
  sit at lower context.
- **Items 2+3 (CUDA-graph + GPU-resident state to kill the `.item()` sync)
  give ≈0 at this vocab.** constrained ≈ unconstrained and the cache-width
  proof both show the sync isn't load-bearing here. Revisit only if a future
  config makes per-token GPU work tiny again (e.g. far smaller model), or
  fold into Phase 0a's "is masking load-bearing" decision. Note this only
  re-weights the ladder for the *current* small-vocab regime; at vocab 32,768
  the old logit-slab story still dominates (→ item 1 vocab shrink first).
- **No software change closes the 2× XPU<CPU gap.** It's the ~12 ms fixed
  floor (XPU) vs ~3–7 ms (CPU) — the iGPU is just slower at the tiny
  seq_len=1 GEMMs and the per-token copies. Windowing/sync/compile shave a
  fraction off both but don't change the ratio. For this small-model
  single-token decode, **CPU is the faster predict host**; the GPU only wins
  once per-token work is large (big vocab/model or batched/speculative).
- **New cheap find:** RMSNorm logs `input dtype=float, weight dtype=BFloat16,
  Cannot dispatch to fused implementation` — an fp32-activation / bf16-weight
  mismatch forcing the non-fused norm on both platforms. Worth a
  dtype-consistency pass (cheap, both-platform); also trim per-token clones
  (`masked.clone()`, the mask `from_numpy().to(device)` H2D).

Harness + raw logs: `/scratch/tmp/perf_probe.py`, `/scratch/tmp/sweep.sh`,
`/scratch/tmp/vekx-*.log` on vek-x. No code changes made (measurement only).

## Cross-reference

- Vocab shrink details: see `accuracy_push_prodlike_4x` AGENTS.md
  block + this doc's "Vocab shrink interaction" section.
- Streaming-unembed-CE (training-side): existing design at
  `integration_tests/design/streaming_unembed_ce_design.md`.
- The atom-vs-merge tokenizer-ID analysis: produced 2026-05-19
  during accuracy_push_prodlike_4x audition; data in
  `/scratch/tmp/preframr_experiments/results/accuracy_push_prodlike_4x/apush4x/seed0/tkmodel.json`.
