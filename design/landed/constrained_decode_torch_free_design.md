## Status

**Pending impl** (2026-05-21). Rearrange `constrained_decode.py` so
torch is needed only at the boundary (caller wraps logits and masks).
Preconditions a future `preframr-tokens` move; valuable on its own as
cleanup separating state-machine logic from tensor plumbing.

## Problem

`preframr/inference/constrained_decode.py` (1090 LoC) imports torch
heavily but uses it for nothing computational — only for boundary
conversion:

| use site | what torch does | what it could do |
|---|---|---|
| `precompute_subtoken_arrays`, `precompute_vocab_arrays` (~50 lines) | wrap numpy arrays → torch tensors + `.to(device)` | return numpy dicts; caller converts |
| `_mask_logits_atomic`, `_mask_logits_subtoken` | `torch.zeros + |= bool tensor` ops on n_vocab=32K bool arrays | identical math in numpy with `np.zeros(..., dtype=np.bool_)` |
| `_unstick_and_fill::torch.nonzero` (one site) | find frame markers | `np.nonzero` |
| `mask_logits(logits)::masked_fill` (one site per call) | apply mask to torch logits | stays at boundary — single line of caller-side glue |

The current code already maintains `_cpu` (numpy) mirrors of every
array because the `_update_*` per-token bookkeeping path already runs
on CPU. So the file is already half-numpy: the torch path exists only
to keep `_mask_logits_*` running on the GPU side.

The bitwise OR of an n_vocab=32K bool array runs in ~5 µs on numpy
(measured: ~33K elements). GPU-side it's still O(n_vocab) plus a
kernel launch, so there's no perf reason to keep it on the device.

## Goal

After this commit:

- `constrained_decode.py` has **zero** `import torch` at module level.
  `mask_logits(self, logits)` lazily imports torch inside the function
  to apply the final `masked_fill`; that's the only torch surface.
- `precompute_subtoken_arrays(tokens_df, regtokenizer, pad_id=0)`
  loses its `device` parameter; returns dicts of numpy arrays.
- `precompute_vocab_arrays(tokens_df)` loses its `device` parameter;
  returns a dict of numpy arrays.
- `StreamState.__init__(self, vocab_arrays, ...)` receives the numpy
  dict; no torch tensors stored.
- `_mask_logits_atomic` / `_mask_logits_subtoken` /
  `_unstick_and_fill` return numpy bool arrays of shape `(n_vocab,)`.
- `mask_logits(self, logits)` (public, on StreamState) computes
  numpy invalid mask, then:
  ```python
  import torch  # lazy
  return logits.masked_fill(
      torch.from_numpy(invalid).to(logits.device), float("-inf")
  )
  ```
- The `_cpu` suffix on array keys is dropped (no longer
  distinguishing — every array is numpy now).
- All 22 callers in `tests/predict/test_constrained_decode.py`,
  `preframr/inference/predict.py`, `preframr/train/structural_loss.py`,
  `integration_tests/profile/predict.py` adapt: drop the `device`
  argument; tests construct numpy dicts directly where they were
  building torch tensors.

## Non-goals

- **Not** moving `constrained_decode.py` into `preframr-tokens` in
  this commit. That's a separate design + extraction. This commit is
  the precondition.
- **Not** changing the public API of `StreamState` (the methods
  `mask_logits` / `update` keep the same signatures). Existing
  callers don't need to know the internals went numpy.
- **Not** removing `torch` as a transitive dependency of
  `constrained_decode` consumers — they still use torch for logits.
  Only the module itself stops importing torch.
- **Not** optimising the masking further (e.g. caching reused
  numpy masks across steps). Pure structural cleanup.

## Concrete changes

### `preframr/inference/constrained_decode.py`

1. Drop module-level `import torch`. The `mask_logits` method
   inside `StreamState` does `import torch` lazily inside the
   function body, before the single `masked_fill` call.
2. `precompute_vocab_arrays(tokens_df, device)` →
   `precompute_vocab_arrays(tokens_df)`. Every line of the form
   `torch.from_numpy(x).to(device)` becomes just `x` (or
   `x.astype(np.bool_)` where the type assertion already exists).
   Drop the redundant `_cpu` key mirrors — single source of truth.
3. `precompute_subtoken_arrays(tokens_df, regtokenizer, device,
   pad_id=0)` → `precompute_subtoken_arrays(tokens_df, regtokenizer,
   pad_id=0)`. Same numpy-only treatment.
4. `_mask_logits_atomic(self, logits)` →
   `_compute_invalid_atomic(self) -> np.ndarray`. Replace
   `torch.zeros(a["n_vocab"], dtype=torch.bool, device=logits.device)`
   with `np.zeros(a["n_vocab"], dtype=np.bool_)`. Bitwise OR/AND/~
   stays identical (numpy and torch both support `|=`, `&=`, `~`).
   Returns the bool array.
5. `_mask_logits_subtoken` same treatment →
   `_compute_invalid_subtoken`.
6. `_unstick_and_fill(self, invalid, logits, frame_marker)`:
   - Takes numpy `invalid`; numpy `frame_marker`; torch `logits`.
   - `torch.nonzero(frame_marker_tensor, as_tuple=False)` →
     `np.flatnonzero(frame_marker)`.
   - Returns numpy `invalid` (no logits operation here).
7. `mask_logits(self, logits)` (public): one path now, applies the
   torch glue at the boundary:
   ```python
   def mask_logits(self, logits):
       import torch  # lazy; lets the module load torch-free
       if self.subtoken_mode:
           invalid_np = self._compute_invalid_subtoken()
       else:
           invalid_np = self._compute_invalid_atomic()
       invalid_np = self._maybe_unstick(invalid_np)
       invalid = torch.from_numpy(invalid_np).to(logits.device)
       return logits.masked_fill(invalid, float("-inf"))
   ```
8. `_update_atomic` / `_update_subtoken` already use the `_cpu`
   numpy arrays. Drop the `_cpu` suffix references; they now read
   from the single numpy dict.

### Callers

- `preframr/inference/predict.py`: lines that call
  `precompute_subtoken_arrays(dataset.tokenizer.tokens, dataset.tokenizer, device)`
  / `precompute_vocab_arrays(dataset.tokenizer.tokens, device)`
  drop the `device` argument.
- `preframr/train/structural_loss.py`: same.
- `integration_tests/profile/predict.py`: same.
- `tests/predict/test_constrained_decode.py`: existing fixtures
  build torch tensors directly for mocked vocab_arrays. Switch to
  numpy bool arrays. ~10–15 test sites; mechanical.

### Tests

Existing 14 tests in `tests/predict/test_constrained_decode.py`
cover the masking semantics. Adapt fixtures (`fake_vocab_arrays`
helper) to return numpy. The semantic assertions are unchanged: a
test that asserts `is_pad` positions get `-inf` keeps the same
expected output.

Add one new test:

```python
def test_constrained_decode_module_torch_free():
    """`preframr.inference.constrained_decode` imports without torch."""
    import importlib, sys
    sys.modules.pop("torch", None)
    sys.modules.pop("preframr.inference.constrained_decode", None)
    importlib.import_module("preframr.inference.constrained_decode")
    assert "torch" not in sys.modules
```

This pins the load-bearing structural property: future commits can't
accidentally re-add a module-level torch import.

## Risks

- **Hot-path perf regression.** The per-step boundary copy is one
  `torch.from_numpy(bool[32K]).to(device)` per decode step. At
  cuda, that's ~8 µs (32 KiB host→device). vs the current path
  which keeps the mask on-device permanently and pays per-step
  bitwise-op overhead. Net: probably within noise; if it shows up,
  the fix is to cache the device tensor on `StreamState` between
  steps with a dirty bit. Measure before optimising.
- **Test fixture churn.** 14 existing tests touch `vocab_arrays`
  fixtures with torch tensors. Mechanical conversion to numpy.
  Risk of an edge case where the torch fixture relied on torch
  semantics differing from numpy (e.g. bool autocasting). Verified
  by running the existing test suite after conversion.
- **`structural_loss.py` integration.** This file uses StreamState
  during *training* — confirms the numpy mask works under autograd.
  StreamState doesn't participate in autograd (mask is constructed
  fresh per step, no params). Risk is low; verified by
  `tests/train/test_per_tier_heads.py::TestPerTierModelSmoke`.

## Success criteria

1. `python3 -c "import sys; assert 'torch' not in sys.modules;
   import preframr.inference.constrained_decode; assert 'torch' not in sys.modules"`
   exits 0.
2. New test `test_constrained_decode_module_torch_free` green.
3. Existing 14 `tests/predict/test_constrained_decode.py` green
   after fixture conversion.
4. `./run_tests.sh` 393+ tests green inside docker image.
5. `./build.sh` green; all three images rebake.
6. Phase 3 prodlike continues running unaffected (this commit
   doesn't touch the bind-mounted `preframr/train/` path; it only
   touches `preframr/inference/`, which by the train/inference
   split landed in `2bfbaeb` is **not** bind-mounted into the
   running training containers — they use the baked copy).

## Effort

~half-day:

- File rewrite: ~2 hours (drop torch from precompute + 2 mask
  functions + 1 unstick helper).
- Test fixture conversion: ~1 hour.
- Caller-side adjustments (4 files, mostly `device=` argument
  removal): ~30 min.
- Docker rebake + test run: ~10 min.
- Commit: ~10 min.

## Next-after-this-lands

When this commit is in, a *separate* design doc covers the actual
`preframr-tokens` move:

- Decision: does `constrained_decode` move to `preframr-tokens` or
  to a new `preframr-decode` package?
- API versioning + dual-version stagger.
- Train-side `structural_loss.py` migration to new import path.

That design is deferred until Phase 3 prodlike returns a verdict —
no point investing the migration cost on a design that might be
about to refute at scale.

## References

- `train_inference_split_design.md` — the inference-side bind-mount
  this design rides on.
- `model_regdataset_decomposition_design.md` — sibling refactor,
  related goals.
- `../../AGENTS.md` — preframr-tokens "torch-free" property.
