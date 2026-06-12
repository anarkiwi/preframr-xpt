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
  atom-level macros, not Unigram merges). This light-merge regime is
  load-bearing for the motif pass: because Unigram barely merges at
  deployment scale, a motif dictionary is NOT redundant with it
  (measured ~11.4% fewer tokens at vocab 8192). The mini dry-run's 0.6% was an over-merged
  small-corpus regime, not this one.
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
`preframr_tokens.constrained_decode` (torch-free numpy; see the Update above). It enforces:

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

## Update (2026-05-26): CUDA/Orin confirm — width is NOT the lever here, copies are

Ported the identical `perf_probe.py` harness to the **actual deployment target**
(Jetson Orin NX, `anarkiwi/preframr-jetson:latest`, `--runtime=nvidia`, CUDA)
and ran the same width sweep on the **same vocab-8192 mini checkpoint**
(`voice_permutation_mini_body_large/baseline/seed0`). Harness/log:
`/scratch/tmp/orin_sweep.sh`, `/scratch/tmp/orin-sweep.log`. Measurement only.

**The XPU "SDPA-over-allocated-width dominates" finding does NOT carry to CUDA.**

| cache L | CUDA ms/tok | XPU ms/tok |
|---|---|---|
| 1024 | 22.19 | 12.4 |
| 2048 | 22.75 | 13.7 |
| 4096 | 23.20 | 15.2 |
| 8192 | 23.68 | 19.2 |

CUDA fits `≈ 21.9 + 0.0002·L`: an **~22 ms fixed floor** with **near-zero width
dependence** — 8× the allocated cache width costs only **+6.7%** (vs +55% on
XPU). So **dynamic KV-windowing (ladder item 5) and the `--max-seq-len`
right-size stopgap give ≈0 on the Orin.** They were an XPU/i915-specific win
(the i915 fused `micro_sdpa` processed the full allocated width); CUDA's
attention does not, so there is nothing to reclaim by windowing on the target.

**Where the time actually goes on CUDA (profiler, 64 steps, L=8192, Self CUDA %):**

| op | Self CUDA % | notes |
|---|---|---|
| `aten::copy_` | **48.0%** | 6215 calls ≈ **97 copies/token** — the dominant cost |
| `aten::bmm` | 18.4% | attention; flat in L (confirms width isn't the lever) |
| `aten::mul` | 14.4% | |
| `aten::mm` | 4.4% | per-token GEMMs |
| `aten::add_` | 2.8% | |
| `aten::_to_copy` | (15.6% CPU, 400 ms CUDA total) | dtype/device conversions |
| `aten::rms_norm` | 0.7% CUDA / 366 µs-per-call CPU dispatch | non-fused (fp32-act/bf16-wt) |

The XPU profile's `aten::copy_` (15% there, ~94/token) is the **same ~97
copies/token**, but on CUDA they are now #1 at 48% because the GEMMs/attention
are comparatively cheaper. The heavy `aten::_to_copy` + non-fused `rms_norm`
corroborate the fp32-activation / bf16-weight mismatch flagged above.

**Revised ladder for the Orin/CUDA target (this vocab/model regime):**
1. **Per-token copy elimination — now the headline win (48% of CUDA time).**
   Trim the per-step clones/H2D: `masked.clone()` (predict.py:185), `x =
   tok.clone()` / `generated = torch.cat(...)` host appends (predict.py:204-212),
   and the mask `from_numpy().to(device)` H2D. Keep the GPU-resident generated
   buffer instead of `torch.cat` per token.
2. **RMSNorm dtype-consistency fix — doubly indicated.** Restores the fused
   norm AND removes the `aten::_to_copy` conversions it forces. Cheap,
   both-platform.
3. **KV-windowing / `--max-seq-len` right-size — DROP for Orin.** ~0 here;
   keep only as an XPU-host note.
4. CUDA-graph / GPU-resident `StreamState` — still ≈0 at this vocab (the floor
   is copies + tiny GEMMs, not the `.item()` sync); unchanged from the XPU read.

Caveat: this is the mini body at vocab 8192. `bmm`/`mm` being cheap is partly
small-model/small-vocab; at a larger body or vocab the GEMM/attention share
rises and width could re-matter. Re-confirm on the `full_macros_prodlike`
checkpoint (vocab 8192, canonical body) once STAGE 2 finishes.

## Update (2026-05-26): surgical fixes landed + measured — ~5–7% on Orin, parity-clean

Implemented the two surgical items (1+2 above) in
`preframr/inference/predict.py` and re-ran the identical Orin sweep with the
edited source bind-mounted over the baked package. Edits:
`_keep_norms_fp32` (keep RMSNorm `scale` fp32 after the bf16 cast); dropped the
redundant `masked.clone()` (`mask_logits`' `masked_fill` already returns a fresh
tensor); preallocated the `generated` buffer (write-in-place vs O(n²) per-token
`torch.cat`); `x = tok` vs `tok.clone()` in the caches path. Left the KVCache
`index_put` (torchtune-internal, compile-deliberate) and the `mask_logits` H2D
(in the `preframr_tokens` package).

**Correctness: byte-identical.** Greedy (top_k=1) 128-token continuation from an
identical prompt — baked sha1 `4df915ef12e4` == edited sha1 `4df915ef12e4`,
finite. The fp32-norm-weight change did not flip a single argmax. Framework
`tests/predict` 26/26 pass on the edited source.

**Perf (Orin NX, vocab 8192, constrained, ms/token):**

| L | baked | edited | Δ |
|---|---|---|---|
| 1024 | 22.19 | 20.64 | −7.0% |
| 2048 | 22.75 | 21.02 | −7.6% |
| 4096 | 23.20 | 21.96 | −5.3% |
| 8192 | 23.68 | 22.62 | −4.5% |

**Mechanism confirmed by the profiler (edited, L=8192):** `aten::_fused_rms_norm`
now dispatches (8.9 µs/call CUDA, 66 ms CPU total) — the non-fused `rms_norm` at
**366 µs/call / 304 ms CPU dispatch is gone**, no "cannot dispatch" warning. That
CPU-dispatch saving is most of the wallclock win. `aten::copy_` is still ~49%
(6153 calls, −62 vs baked) — confirming the residual copies are the **KVCache
`index_put` (~2/layer × 16)**, which only the custom-decode rewrite reaches.
`aten::_to_copy` barely moved (the `x.float()` upcast inside torchtune's
fp32-norm is inherent).

**Read:** the cheap, both-platform tier is done and banked (~5–7%, zero quality
cost — the fused-norm restore also helps XPU and de-risks the autocast
fp32-promotion trap). **XPU confirmed (2026-05-26, vek-x Arc, mini ckpt):** the
fp32-norm fix removes the "Cannot dispatch to fused implementation" RMSNorm
warning on XPU and gives 12.98→12.12 ms/tok @1024 (−6.6%) / 19.03→18.80 @8192
(−1.2%) — same shape as Orin (bigger at short context where norm/copy is a
larger fraction; SDPA-over-width dominates at 8192). Parity holds by the Orin
argument (device-agnostic, byte-identical there). Note: the XPU *decoder still
runs eager* (cudagraph is CUDA-only; compiling the XPU decoder is an untested
follow-up) and CPU remains the faster predict host for this small model.
Going further is the **custom llama3_2 decode runtime**
(in-place KV write + fused attention), gated on a logits-parity test and the
architecture freezing post-prodlike. Harness: `/scratch/tmp/parity_probe.py`,
`/scratch/tmp/orin_validate.sh`, `/scratch/tmp/orin-edited-sweep.log`.

## Update (2026-05-26): CUDA-graph capture — 1.9–3.1× on Orin, parity-clean (the real win)

**Root cause of the ~22 ms floor: the decoder ran eager at inference.**
`factory.cuda_compile` only adds `triton.cudagraphs` when
`accumulate_grad_batches == 1` (a *training* condition), and the decode loop
calls `self.model.model` directly — bypassing the compiled Lightning wrapper's
`forward`. So the per-token forward executed eager: ~454 `cudaLaunchKernel`/token
= a launch-bound floor.

**Fix (sandbox `predict_fast.py`):** compile the decoder explicitly for the CUDA
inference path —
`torch.compile(model.model, options={"epilogue_fusion": True, "triton.cudagraphs": True})`.
torchtune's KVCache is cudagraph-safe by design (tensor `cache_pos` advances
under replay), so capture is valid.

**Parity: byte-identical** — greedy sha1 `4df915ef12e4` at every width (== the
committed-predict baseline). **Perf (Orin NX, vocab 8192, ms/token):**

| L | eager (committed predict) | cudagraph | speedup |
|---|---|---|---|
| 1024 | 20.64 | 6.64 | 3.1× |
| 2048 | 21.02 | 8.66 | 2.4× |
| 4096 | 21.96 | 10.91 | 2.0× |
| 8192 | 22.62 | 12.14 | 1.9× |

At the production 8192 width that halves per-token cost (a 6k-token render
~136 s → ~73 s).

**This reverses the earlier "width is not the lever on CUDA" call — *for the
cudagraph regime*.** Eager was launch-bound, so width was masked (flat-in-L).
With launches gone, the cudagraph path is strongly **width-dependent**
(6.64→12.14, +83%): the SDPA-over-allocated-width compute is now the critical
path. So the copy-attrib full-width GQA work was real, just hidden behind launch
latency in eager. **Net: after cudagraphs lands, `--max-seq-len` right-sizing /
dynamic cache-windowing / the custom in-place-KV forward all re-open as genuine
CUDA wins** (e.g. capping width 8192→1024 is 12.14→6.64, ~45%).

**Recommendation:** land the cudagraph compile in production `predict.py`, gated
to CUDA inference (independent of the training `accumulate_grad_batches` arg);
keep a disable path for configs that don't capture cleanly. First token per run
pays the capture/compile warmup (~30–60 s), then steady-state is the table
above. Then pursue width right-sizing as the next lever.

Caveat: mini body; the launch-bound→cudagraph win should hold or grow at
prodlike (more layers = more launches eliminated), but re-confirm on the
`full_macros_prodlike` checkpoint once STAGE 2 finishes. Harness:
`/scratch/tmp/fast_probe.py`, `/scratch/tmp/orin_cudagraph_sweep.sh`,
`/scratch/tmp/orin-cudagraph-sweep.log`.

## Update (2026-05-26): roofline — how far from the hardware ceiling, and the big levers

Measured the memory-bandwidth roofline + a forward-vs-overhead decomposition on
the Orin (mini, cudagraph). Harness: `/scratch/tmp/roofline_probe.py`,
`/scratch/tmp/orin-roofline.log`. Achievable LPDDR5 BW (copy_ microbench)
**72–75 GB/s** (~72% of the 102 GB/s spec).

| L | bytes/tok | mem-BW floor | forward-only (GPU) | **% of BW peak** | full step | CPU overhead |
|---|---|---|---|---|---|---|
| 1024 | 19.2 MB | 0.26 ms | 2.66 ms | **9.9%** | 6.47 ms | 3.81 ms (59%) |
| 8192 | 44.0 MB | 0.59 ms | 7.39 ms | **8.0%** | 19.5 ms | 12.1 ms (62%) |

Two structural findings:
1. **The GPU forward runs at only ~8–10% of the memory-bandwidth ceiling.** At
   batch-1 the small model under-occupies the GPU (tiny seq_len=1 GEMMs don't
   fill the 1024 cores) — we are **occupancy-bound, not bandwidth-bound**. ~90%
   of the LPDDR is idle.
2. **~40–60% of the full step is CPU overhead** (`mask_logits` numpy +
   `_tt_sample` + `tok.item()` sync + `StreamState.update`, all OUTSIDE the
   cudagraph; the share varies with prompt/content). So **full-step
   GPU-resident decode (lever 3) can roughly halve the per-token time**,
   bringing the full step toward the forward-only floor (~7.4 ms @8192 mini).

### The levers that GREATLY improve (ranked), vs the small-beer

Because batch-1 leaves ~90% of the hardware idle, the multiplicative wins are
about *filling* the GPU, not shaving the single stream:

1. **Batching / speculative decode (multiplicative).** Render an audition cohort
   in parallel, or use a tiny draft model verified K-at-a-time — amortizes the
   per-token weight read across many tokens/sequences. With ~10× idle headroom
   this is the only path to a large (≫2×) throughput gain.
2. **Quantization (int4 weights / int8 KV)** — the lever for the *memory-bound
   prodlike* regime (weights 226 MB + KV 134 MB/tok @8192 = ~360 MB → floor
   ~4.5 ms): ~4× fewer weight bytes, ~2× fewer KV bytes raises the floor 2–3×.
3. **Full-step GPU-resident decode (lever 3)** — eliminate the ~40–60% CPU
   overhead by moving the constrained mask + sampling + state onto the GPU and
   capturing the whole step. ~halves the single-stream time.
4. Windowing / `--max-seq-len` right-size — small-beer; deprioritized.

Caveat: all numbers are the **mini** body. The deployment model is **prodlike**
(~113M params, ~360 MB/tok @8192, mem-BW floor ~4.5 ms, ~3–4× the forward work);
re-run `roofline_probe.py` + `tokens_per_frame.py` on the `full_macros_prodlike`
checkpoint once STAGE 2 finishes for the deployment-accurate ceiling and the
real-time verdict.

### Real-time verdict

Token→audio compression measured over 105 real song blocks (≈62 min of audio,
`tokens_per_frame.py`): **4.25 tokens/frame**, 50.12 frames/s (PAL) → **213
tokens/s of audio**. So staying ahead of real-time needs **≤ 4.70 ms/token**
(≥213 tok/s).

| stage | mini @8192 | mini @1024 | clears RT? |
|---|---|---|---|
| eager | 22.6 ms (44 t/s) | — | no (0.2×) |
| cudagraph | 12.1 ms (83 t/s) | 6.6 ms (152 t/s) | no (0.4–0.7×) |
| + lever-3 (GPU-resident) | 7.4 ms (135 t/s) | **2.7 ms (370 t/s)** | only @short ctx (1.7×) |

So even on mini, only **lever-3 at short context** clears real-time; nothing
single-stream clears at the full-8192 audition width. **Prodlike** (mem-BW floor
~4.5 ms ≈ the 4.70 ms threshold) cannot clear real-time single-stream at bf16 —
at 30–60% BW efficiency it lands ~8–15 ms/token (1.5–3× too slow). **The lever
that makes prodlike real-time is quantization:** int8 weights+KV → ~180 MB/tok →
floor ~2.25 ms, below threshold with margin (int4 weights lower still), so
lever-3 + windowing can then reach it. Batching/speculative gives >real-time for
offline cohort rendering.

### Prodlike (deployment) roofline — measured, and the corrected real-time gap

Re-ran `roofline_probe.py` on the finished `full_macros_prodlike/full_macros/seed0`
checkpoint (107M params, 16 layers, d768) — deployment-accurate, replacing the
mini extrapolation. Achievable BW 75 GB/s.

| L | bytes/tok | mem-floor | forward-only | % BW peak | full step | tok/s |
|---|---|---|---|---|---|---|
| 2048 | 247.6 MB | 3.30 ms | 10.6 ms | 31.2% | 23.1 ms | 43 |
| 8192 | 348.2 MB | 4.64 ms | 29.9 ms | 15.5% | 42.9 ms | 23 |

Findings:
- **Prodlike is meaningfully memory-bound** (15–31% of BW peak vs mini's ~8%) —
  bigger GEMMs use the GPU better, so **quantization should help here** (unlike
  mini, where it wouldn't).
- **The forward is strongly width-dependent under cudagraph** (10.6→29.9 ms as
  2048→8192): with launch overhead gone, SDPA over the full allocated cache
  width dominates. So **windowing / cache right-sizing re-opens as a genuine
  lever on prodlike** (it was ~0 on mini, but mini was launch-bound).
- **Real-time gap is ~9×, not 1.5–3×.** Full step @8192 = 42.9 ms (23 tok/s) vs
  the 4.70 ms (213 tok/s) threshold. Stacking the levers:
  lever-3 (−13 ms CPU overhead) → ~29.9 ms (6.4× over); + windowing (avg ctx
  ~5120 → forward ~18 ms) → ~18 ms (3.8× over); + quant (measuring; optimistic
  2–3× on the memory-bound part) → ~8–12 ms (still ~2× over).
- **Verdict: prodlike single-stream real-time on Orin is NOT reachable** with
  these levers stacked (~9× gap, levers give ~4–6× combined). Live/interactive
  real-time would need a smaller/distilled model. **Offline auditions are fine:**
  at 42.9 ms/tok × ~9137 tokens/song ≈ 6.5 min/song, within the <10 min target —
  and that is *with* the cudagraph win already (eager would be ~2× worse).
- Throughput (offline cohort) scales with **batching** — orthogonal to the
  single-stream latency wall above.

### Quantization measured (prodlike @8192) — does NOT help single-stream; needs batching

torchao 0.17 in the jetson image, applied to the decoder before the cudagraph
compile, measured vs bf16 (33.5 ms/token in this probe; greedy token-match as a
quality proxy). `/scratch/tmp/quant_probe.py`, `/scratch/tmp/orin-quant-sweep.log`.

| quant | ms/token | quality | outcome |
|---|---|---|---|
| bf16 | 33.5 | — | reference |
| int8 weight-only | **64.3** | **100% token match** | composes + lossless-greedy, but **2× SLOWER** |
| int4 weight-only | — | — | unavailable (`ImportError: Requires mslk >= 1.0.0`) |
| int8 dyn-act+wt | — | — | fails: `self.size(0) needs to be > 16, but got 1` |

**Quantization does not help single-token (batch-1) decode on this Orin/torchao
stack**, and the failure modes say why:
- int8 weight-only is **2× slower** (quality perfect) — at M=1 there is no fast
  int8×bf16 GEMM kernel, so torchao dequantizes to bf16 and the dequant overhead
  exceeds the weight-bandwidth savings. The savings need a large GEMM.
- int8 dynamic-activation **requires M ≥ 16** — i.e. it only works **batched**.
- int4 needs a kernel lib absent from the image.

This confirms the roofline headline: **low-precision speedups require batching**
(M≥16), the same regime as the multiplicative throughput win. For single-stream
latency, **cudagraph is the ceiling** (the committed win); quantization is only
worth it if/when batched decode (cohort rendering or speculative) is built.
Offline single-stream auditions are already within budget (~6.5 min/song).

**Net inference-opt conclusion:** landed = surgical fixes + cudagraph
(single-stream prize). Further *single-stream* gains are marginal (windowing
~17% on prodlike; lever-3 ~CPU-overhead removal) and none reach real-time.
**Real-time and large throughput both require batching** (then int8 dyn-act
quant applies); that is a deployment-model change, deferred as a decision.

### Speculative decoding (single-song latency) — Step 0 accept-rate measured

For single-song latency (cohort batching does not help one song), the lever is
speculative decoding: a draft proposes K tokens, the target verifies all K in
one forward (≈free given the idle compute), advancing 1+accepted tokens/forward.
Greedy (top_k=1) is **lossless by construction** (only tokens the target would
itself emit are accepted) — same parity gate as cudagraph.

Step 0 (no model, no GPU): simulated **prompt-lookup (n-gram) drafting** on 300
real prodlike `full_macros` token blocks (~2.65M tokens), `/scratch/tmp/spec_accept.py`:

| n-gram | K | speedup (tokens/forward) | mean accepted | ≥1 hit |
|---|---|---|---|---|
| 1 | 4 | 1.69× | 0.69 | 34% |
| 2 | 4 | 1.99× | 0.99 | 40% |
| 2 | 8 | 2.23× | 1.23 | 37% |
| 3 | 8 | 2.24× | 1.24 | 31% |

**~2.2× single-stream, zero training, lossless** — SID repetition makes the
n-gram draft effective (best n=2). Caveats: (1) verify-K compute trims high K
(prodlike 15.5% of peak → K=8 verify ~1.2–1.5× a 1-token forward, not free; K=4
stays ~free) → realistic **~1.8–2.0×**; (2) the sim uses the *real* stream as a
proxy for the target's argmax — the target's greedy output (val_acc 0.38) is
likely *more* self-repetitive, so ~2× is probably a conservative floor (confirm
by running the target greedily and re-measuring on its own output).

**Verdict: worth building** — biggest remaining single-song lever, lossless,
exploits the idle compute the roofline exposed, stacks on cudagraph (prodlike
~33–43 → ~16–22 ms/token). Does NOT reach real-time (4.7 ms) alone. Build:
prompt-lookup draft (no training) + greedy verify with KV rollback + the
existing `StreamState` mask driving both draft and verify + fixed-K cudagraph;
parity gate = sha1 identical to non-speculative greedy. ~1–2 weeks. Cheap next
step before committing: **Step 0b** — confirm the accept rate on the *target's
own* greedy output (GPU, prodlike ckpt) rather than the real-stream proxy.

**Step 0b result (2026-05-26): prompt-lookup speculative is a NO-GO.** Generated
4096 tokens from the prodlike target (`gen_dump.py`) and re-ran the accept sim on
its OWN output:
- **greedy → distinct=1** (single repeated token — total loop collapse; unusable
  audio, and its 5–9× accept is a degenerate artifact).
- **sampled (temp=1.0, distinct=638, the realistic deployment mode) → ~1.0–1.18×**
  (n=1: 15.5% hit / 1.18×; n=2: 6%; n=3: 2%).

The Step 0 ~2.2× was a **proxy artifact**: real songs are deterministic/repetitive,
but the model is deployed with **temperature sampling** (greedy collapses), and
high-entropy sampled output is unpredictable by an n-gram draft (speculative
speedup is bounded by how close the draft distribution is to the target's, and
n-gram is far from a temp-sampled high-entropy target). **Do not build
prompt-lookup speculative.** A *trained* draft model could do better (~1.5–2×,
uncertain) but needs draft training + the full build (~2–3 weeks) and the
high-entropy / val_acc-0.38 target makes it harder than typical LLM speculative.

**Single-song latency — final standing:** the **cudagraph win (1.9–3.1×, landed)
is the realistic ceiling**. Remaining options: **lever-3** (GPU-resident decode,
removes the ~13 ms CPU overhead → ~1.4× @8192 / ~2.2× @2048, lossless, moderate
effort) is the best achievable next step; trained-draft speculative is a bigger
uncertain bet. None reach real-time (4.70 ms). Confirm-before-build (Step 0b)
saved ~1–2 weeks here.

### Lever-3 Phase 0 result (2026-05-26): NO-GO — the "CPU overhead" was a probe artifact

Phase 0 (2), state analysis: the constrained-decode mask/update are vectorized
array-arithmetic over per-vocab tables + ~6 scalar state values (`pending_slot`∈8,
`pending_overlay_slot`∈3, `frame_count`, `frame_budget`, `remaining_steps`,
`current_dist_hi`) — **no state-space blowup**; a GPU port would be a direct
numpy→torch translation. So lever-3 is *feasible*. But Phase 0 (1) kills the
*motivation*: a controlled micro-benchmark of every per-token component
(`overhead_probe.py`) sums to **~0.43 ms/token**, NOT the ~13 ms the roofline's
full-minus-forward-only implied:

| numpy mask | H2D | masked_fill | sample | item() | update | total |
|---|---|---|---|---|---|---|
| 0.062 | 0.057 | 0.081 | 0.193 | 0.031 | 0.003 | **~0.43 ms** |

The ~13 ms "CPU overhead" (and the earlier "40–60% of step") was a
**`roofline_probe` artifact** — its full-step timing is systematically inflated
vs the clean sweep + quant probes (mini: roofline full 19.5 ms vs sweep 12.1;
prodlike: roofline full 42.9 vs quant-bf16 33.5). The real per-token CPU work is
~0.43 ms (~1–3% of the ~30 ms forward).

**Airtight NO-GO:** lever-3 can only remove the CPU ops it eliminates (≤0.43 ms),
and it cannot recover anything more — autoregressive decode is **inherently
serial** (token N+1 depends on token N), so there is no lost pipelining for a
no-sync/full-step-capture rewrite to reclaim. ~1% gain for ~1–3 weeks → drop it.

**FINAL single-song conclusion:** the **forward IS the cost** (~30 ms/token,
memory/occupancy-bound at 15.5% of BW peak at batch-1), and **cudagraph (landed)
is the single-song ceiling.** Every other single-song lever is dead or marginal
(windowing ~0 on CUDA; quant 2× slower at M=1; speculative ~1.1× under sampling;
lever-3 ~1%). Cutting the forward further requires a **smaller/quantized model**
or **batching** (throughput, not latency) — both deployment-model changes, not
decode-path tweaks. Inference-opt is concluded; the banked win is surgical +
cudagraph (1.9–3.1×), and the design is now a complete measured map of the dead
ends so this isn't re-explored.

## Cross-reference

- Vocab shrink details: this doc's "Vocab shrink interaction" section (the original
  `accuracy_push_prodlike_4x` audition that sourced the baseline is since retired).
- Streaming-unembed-CE (training-side): design removed 2026-06-12 — moot at event-model vocab
  scale (slab ≤0.5 GiB at tkvocab 8192); git history.
- The atom-vs-merge tokenizer-ID analysis: produced 2026-05-19 during that audition
  (raw `tkmodel.json` lived under a since-cleared `/scratch/tmp/...` results dir).
