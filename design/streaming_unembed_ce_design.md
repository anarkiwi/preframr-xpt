# streaming unembed-CE — design note

Architectural fix that would recover the 2x wallclock cost paid in
commit 231ba88 (batch_size 4→2 to fit the prodlike body in 24 GiB).
Interleaves the unembed projection with per-chunk CE inside a single
gradient checkpoint so the 8.6 GiB list of unembed chunks never
materialises during forward.

**Blocked on `loop_lookahead_prodlike` completion** — touches
`preframr/model.py`, which the AGENTS.md mid-run-edit rule forbids
while an orchestrator is in flight. Design only this commit.

## Motivation

At prodlike scale (`B=4`, `S=8192`, `V=131072`, bf16) the unembed
output is:

    B * S * V * 2 bytes = 4 * 8192 * 131072 * 2 = 8.6 GiB

`torchtune.modules.TransformerDecoder.chunked_output` chunks the seq
dim into `num_output_chunks=8` slabs and returns them as a Python
list:

```python
# torchtune/modules/transformer.py:497
def chunked_output(self, last_hidden_state):
    return [
        self.output(chunk)
        for chunk in last_hidden_state.chunk(self.num_output_chunks, dim=1)
    ]
```

The list comprehension materialises **all 8 chunks concurrently**.
Total: 8 * (4, 1024, 131072) bf16 = 8.6 GiB — the same as the
unchunked output. The current chunked-CE only saves the fp32
log_softmax intermediate (recomputed per-chunk in backward via
`_torch_checkpoint.checkpoint`), not the unembed output itself.

Empirical: at batch=4 the training_step peak is ~22.7 GiB (8.6 GiB
unembed chunks + 1.6 GiB transformer activations + 0.8 GiB params +
3.2 GiB Adam state + 2 GiB CE upcast workspace + misc). OOMs the 24
GiB RTX 4090. At batch=2 the unembed slab is 4.3 GiB and training_step
peak is ~18 GiB — fits with ~5.5 GiB headroom. Wallclock doubles
because we now run 2x micro-batches per optimiser step.

**Streaming the unembed-CE coupling eliminates the 8.6 GiB slab.**
At any moment, at most one chunk's unembed output is alive (~1 GiB);
backward recomputes via checkpoint from the small hidden-state input
(48 MiB per chunk). Peak drops from ~22.7 GiB to ~14-15 GiB at
batch=4 (estimate; needs validation). Restores batch=4 →
accumulate_grad_batches=8 → ~36-66 hr wallclock for prodlike.

## Method

Replace the current two-step path (model returns chunk list → CE
function iterates) with a fused per-chunk loop that runs the unembed
**inside** the gradient checkpoint.

Current path (`preframr/model.py:training_step` + `_chunked_list_cross_entropy`):

```python
# training_step:
preds = self.model(x)                       # list of 8 chunks (8.6 GiB alive)
per_tok = chunked_cross_entropy(preds, y, label_smoothing=...)

# _chunked_list_cross_entropy:
for logit_chunk, tgt_chunk in zip(logit_chunks, target_chunks):
    parts.append(
        _torch_checkpoint.checkpoint(
            _cross_entropy_logit_chunk,    # only fp32 cast is checkpointed
            logit_chunk, tgt_chunk, label_smoothing,
            use_reentrant=False,
        )
    )
```

Proposed path:

```python
# training_step:
h = self._forward_no_unembed(x)             # (B, S, E); ~0.8 GiB
per_tok = streaming_unembed_ce(
    self._unembed_module,                    # decoder.output (the tied F.linear)
    h, y,
    num_chunks=self.num_output_chunks,
    label_smoothing=self.args.label_smoothing,
)

# streaming_unembed_ce:
def _unembed_and_ce(unembed_fn, h_chunk, y_chunk, label_smoothing):
    """Inside the checkpoint: do unembed AND CE in one shot. The
    logit chunk is allocated locally and dropped at function return
    (forward) or end of backward recompute. The saved-for-backward
    is h_chunk (small)."""
    logit_chunk = unembed_fn(h_chunk)        # (B, S/N, V) bf16, ~1 GiB
    return _cross_entropy_logit_chunk(logit_chunk, y_chunk, label_smoothing)

def streaming_unembed_ce(unembed_fn, h, y, num_chunks, label_smoothing):
    h_chunks = list(h.chunk(num_chunks, dim=1))
    y_chunks = list(y.chunk(num_chunks, dim=1))
    use_ckpt = torch.is_grad_enabled() and h.requires_grad
    parts = []
    for h_chunk, y_chunk in zip(h_chunks, y_chunks):
        if use_ckpt:
            parts.append(
                _torch_checkpoint.checkpoint(
                    _unembed_and_ce,
                    unembed_fn, h_chunk, y_chunk, label_smoothing,
                    use_reentrant=False,
                )
            )
        else:
            parts.append(_unembed_and_ce(unembed_fn, h_chunk, y_chunk, label_smoothing))
    return torch.cat(parts, dim=1)
```

Memory savings during forward:

| Path | Peak unembed slab |
|---|---|
| Today (list of N chunks) | N * 1 GiB = 8 GiB |
| Streaming (1 chunk alive at a time) | 1 GiB |
| Saved | 7 GiB |

Backward (with checkpoint):
- saved-for-backward per chunk: `h_chunk` = (B, S/N, E) bf16 = 48 MiB
- N chunks: ~400 MiB total saved
- recomputation cost: each chunk re-runs the unembed + fp32 upcast
  during backward → roughly 2x compute for the unembed step alone.

Net training step peak estimate at batch=4:
- params 0.8 GiB
- activations 1.6 GiB (transformer)
- saved h_chunks 0.4 GiB
- per-chunk live: unembed output 1 GiB + fp32 upcast 2 GiB = 3 GiB
- grad of params 0.8 GiB
- Adam state 3.2 GiB
- workspace ~1 GiB
- **total ~11 GiB peak (down from 22.7 GiB at batch=4)**

Fits in 24 GiB with ~13 GiB headroom. Could even raise batch_size to 8
if the prodlike runtime needs faster wallclock.

## Wiring: `_forward_no_unembed`

The model is a `torchtune.modules.TransformerDecoder`. Today the
training_step does `preds = self.model(x)` which calls torchtune's
`forward(tokens, ...)`, which runs the embed → layers → norm → unembed.
We need access to the hidden state BEFORE unembed.

Three options:

### Option 1: Monkey-patch `chunked_output` to be a no-op

```python
def _identity_chunked_output(self, h):
    return h  # return the hidden state, not a chunked list

self.model.chunked_output = types.MethodType(_identity_chunked_output, self.model)
```

Then `self.model(x)` returns `h` (shape `(B, S, E)`). Downstream
`streaming_unembed_ce` runs the unembed externally.

**Pros:** smallest surface, just one method swap.
**Risks:** other torchtune internals may assume `chunked_output`
returns a list (validators, debug prints). Need to audit.

### Option 2: Subclass TransformerDecoder

Create a `PreframrTransformerDecoder` that overrides `unembed` to
return the hidden state unchanged. Reuse the existing model
constructors (`get_llama3_2` etc.) with the subclass.

**Pros:** explicit, no monkey-patching.
**Cons:** requires plumbing through `MODEL_GETTERS` in `preframr/model.py`
to instantiate the subclass. Touches every model entry.

### Option 3: Forward-hook on the final norm layer

Register a forward hook on `self.model.norm` that captures the output;
skip the unembed call by setting `self.model.output = nn.Identity()`.

**Pros:** preserves the torchtune API surface.
**Cons:** Identity-substituted output may break torchtune internals
that expect a vocab-dim output (e.g. `set_num_output_chunks` validators).

**Recommendation:** Option 1 (monkey-patch). Smallest surface, easiest
to revert. Audit torchtune's references to `chunked_output` /
`num_output_chunks` to verify nothing else reads the chunked list.

## Edge cases

- **Eval-mode (validation_step):** today runs `with torch.no_grad():
  preds = self.model(x)`; preds is the chunk list. Argmax + CE handle
  the list. Under streaming-unembed-CE, the model would return `h`
  not a list; validation_step needs to compute argmax + CE chunk-wise
  too. Refactor needed: same streaming primitive, but with argmax
  emitted per chunk and concatenated, no checkpoint (no grad).
- **`num_output_chunks=0` (legacy single-tensor path):** retain for
  small-vocab cases. The streaming wrapper short-circuits when
  num_chunks=1 (degenerates to unchunked).
- **`structural_loss_lambda > 0`:** the structural loss reads `preds`
  directly. Under streaming, no full `preds` tensor exists; the
  structural loss would need its own chunked variant, or run on the
  re-assembled logits (which costs the 8.6 GiB anyway).
  **Decision:** if structural loss is on, fall back to today's path
  (no streaming). The current prodlike spec sets
  `structural_loss_lambda=0`, so streaming applies.
- **`torch.compile` interaction:** the streaming loop is in plain
  Python; Inductor traces through it. The `chunked_output` torchtune
  method has `@torch.compiler.disable` decorator on it
  (transformer.py:480) — same disabling needed on `streaming_unembed_ce`
  to avoid compile-time issues with checkpoint + variable-length
  loops.
- **Pred IDs for validation accuracy:** today `pred_ids =
  torch.cat([c.argmax(dim=-1) for c in preds], dim=1)` — relies on
  the chunked list. Under streaming, validation_step would compute
  argmax inside the streaming loop, never materialising full logits.

## Wallclock impact

- **Forward:** ~same (one full unembed + CE pass either way).
- **Backward:** ~10-20% slower (extra unembed recompute per chunk
  via checkpoint). The transformer body backward dominates total
  step time, so the marginal cost is small.
- **Net:** estimated +5-15% wall per step. Combined with restoring
  batch=4 (which halves the step count vs batch=2), net wallclock
  improvement is ~40-50% vs the current batch=2 config.

For prodlike at the current ~72-130 hr budget, this brings wallclock
back to **~36-77 hr** — close to the original ~36-66 hr estimate.

## Validation strategy

Layered, in order:

**L0 — unit (`tests/test_model.py`):**
- synthetic (B=2, S=64, V=512, num_chunks=8) input.
- Reference: today's path (chunked_output → chunked_cross_entropy).
- Streaming output: `streaming_unembed_ce(...)`.
- Assert: per-token CE within 1e-5 absolute (fp32 reduction).
- Backward: grads on h_chunk + lm_head weight within 1e-5.

**L1 — peak memory (`profile/streaming_unembed_smoke.py`):**
- Sibling smoke at prodlike body. Run two trainer.fit's:
  one with the current path, one with streaming.
- Report peak_alloc delta. Expected ~7 GiB savings at batch=4.

**L2 — train step OOM smoke (`train_prodlike_oom_smoke.py` extension):**
- `--streaming-unembed` CLI flag.
- At `--batch-size 4`: current path OOMs; streaming path passes at
  ~14-15 GiB peak.

**L3 — short prodlike spec re-run on smoke tier:**
- 1 arm, 1 seed, fast_dev_run-like budget.
- Smoke + streaming together: assert train loss matches the current
  path within seed σ (deterministic seed; same data, same model
  init).

**L4 — full mini-tier `loop_lookahead` re-run:**
- 2 arms × 2 seeds, mini tier (~12-20 min/arm).
- Assert Δval_acc within seed σ of the reference 2026-05-11 result.
- Confirms no training-dynamics drift.

Pre-A/B gate: L0-L4 all green before flipping prodlike to streaming.

## Effort

- `streaming_unembed_ce` function in `preframr/model.py`: **~0.5 day.**
- `_forward_no_unembed` wiring (Option 1 monkey-patch + audit):
  **~0.5 day.**
- `validation_step` refactor to stream argmax + CE: **~0.5 day.**
- L0-L2 tests: **~0.5 day.**
- L3-L4 validation runs: **~1 day (mini-tier wallclock).**

Total: **~3 days.** Lands after `loop_lookahead_prodlike` completes.

## Order of operations

1. Land this design (reviewer pass).
2. Implement `streaming_unembed_ce` + monkey-patch wiring + L0 unit
   tests in one commit.
3. Implement validation_step streaming variant in a sibling commit
   (so the train-side change can land/be reverted independently).
4. Land L1 peak-memory smoke + extend the prodlike OOM smoke with
   `--streaming-unembed`.
5. L3-L4 validation runs; fold result into AGENTS.md Resolved.
6. If green: flip `prodlike_train_args` to `batch_size=4` +
   `accumulate_grad_batches=8` (the pre-2026-05-12 config); rebuild
   the loop_lookahead_prodlike result on the recovered wallclock.

## Out of scope

- **Different chunking strategy** (e.g. batch-dim chunking instead
  of seq-dim). Seq-dim is what torchtune already supports; batch-dim
  would need its own infra.
- **Switching off `tie_word_embeddings`.** Untied weights would
  double the parameter count of the unembed (a separate (V, E)
  matrix) — meaningful memory regression. Not worth the gain.
- **Activation checkpointing on transformer layers.** Orthogonal
  optimization; saves ~1.6 GiB of transformer activations at
  ~30-50% extra compute. Could be layered on top of streaming
  unembed-CE if more headroom is needed (e.g. for frontier-tier
  exploration), but not required for the prodlike envelope.

## Risks & open questions

- **Loss-curve drift.** Per-chunk fp32 CE accumulates differently
  from batched fp32 CE if any non-associativity matters (e.g. when
  reduction='none' is used and the parts are concatenated, the final
  sum/mean is identical — should be byte-equal). L0 unit test must
  verify byte-equality, not just numerical closeness.
- **Compile interaction.** Inductor may not trace through the
  streaming loop cleanly (variable-length iteration + checkpoint).
  `@torch.compiler.disable` on `streaming_unembed_ce` is the safe
  baseline; revisit if compile is critical.
- **Per-chunk grad scaling under bf16-mixed.** Lightning's
  MixedPrecisionPlugin applies the scaler after `.backward()` on the
  full loss. Streaming doesn't change this — the per-chunk parts
  are concatenated into a single per-token tensor before the
  reduction, so the loss gradient flows back through the parts
  identically.
- **Structural-loss compatibility.** Streaming and
  `structural_loss_lambda > 0` are mutually exclusive in this
  design. Document explicitly; future work could chunk the
  structural loss too.

This change is the highest-value memory optimization on the table:
the only one that recovers prodlike wallclock without sacrificing
training dynamics or sample efficiency.
