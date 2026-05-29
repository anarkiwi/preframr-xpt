# preframr architecture

The **torch** layer: train / predict / model. Wraps `preframr-tokens` (parse +
tokenize, torch-free) and `preframr-audio` (render) with a torchtune transformer
body + pytorch-lightning. Ships as the `anarkiwi/preframr` docker image (no PyPI);
floors `preframr-tokens` / `preframr-audio` versions in `requirements.txt`.

## Entrypoints (`/preframr/...` in the image)

| script | role |
|---|---|
| `parse.py` | dumps → parsed `*.N.parquet` (drives `preframr_tokens` parse). |
| `stftokenize.py` | parsed dfs → tokenized blocks + vocab (`Corpus`/`RegTokenizer`). |
| `train/trainer.py` | the training run (Lightning `Trainer.fit`). |
| `inference/predict.py` | checkpoint → generated continuation → WAV (+`--predict-dump`). |
| `mine_motifs.py` | mine the motif codebook artifact (`mine_dict_from_dumps`). |
| `inference/export_weights.py` | export weights for slim/Jetson predict images. |

All share **`args.py`** — `add_args(parser)` is the central CLI; a
`--pipeline-spec` JSON is expanded onto args by `apply_pipeline_spec_to_args`
(this is how a spec's transform list + flags reach the parser/tokenizer). New
tokenizer flags (e.g. a pass's `--<flag>`) are declared here as
`BooleanOptionalAction` and floored against the tokens version.

## Data path

```
dumps ─ preframr_tokens.Corpus (parse+tokenize+cache, torch-free) ─┐
                                                                   ▼
                              preframr.train.regdataset.RegDataset (torch Dataset adapter)
                                  + block_mapper.BlockMapper (slice into seq_len blocks)
                                                                   ▼
                                         DataLoader → Lightning Model
```

- **`RegDataset`** (`train/regdataset.py`) wraps `preframr_tokens.Corpus` +
  `BlockMapper`. `Corpus` owns the torch-free state (RegTokenizer, reg_widths,
  n_vocab, tokenize metadata) and yields `(kind, blocks_path, seq_meta)`;
  RegDataset routes those into per-subset BlockMappers and exposes the torch
  Dataset protocol. `preload(tokens=, tkmodel=)` loads vocab (recovered from the
  ckpt at predict time). `getseq(rotation_i, block_j)` returns a prompt sequence.
- **`BlockMapper`** (`train/block_mapper.py`) cuts block arrays into `seq_len`
  training windows. Eval/predict subsets get their own mappers.
- `df-map.csv` / `.blocks.npy` semantics live in tokens (see tokens
  `ARCHITECTURE.md`); the **df-map staging-path gotcha** (paths are canonical
  labels, blocks live in the work_dir) bites `predict.py` when forwarding a run's
  checkpoint — rewrite the df-map to the work_dir or glob blocks directly.

## Model

- **`train/model/bodies.py`** — torchtune body factories (gemma / llama2 /
  **llama3_2** / mistral / phi3 / qwen2), `MODEL_GETTERS` dispatch on
  `args.model`, `MODEL_PRECISION`, schedule-free `OPTIMIZER`. Pure torchtune.
- **`train/model/factory.py`** — `get_model`, device dispatch (`get_device`),
  `cpu_compile`/`cuda_compile`, and `SchedulerFreeModelCheckpoint` (calls the
  schedule-free optimizer's `.eval()`/`.train()` around checkpoint save).
- **`train/model/lightning.py`** — `Model(LightningModule)`: owns the body
  (`MODEL_GETTERS[args.model]`), the **optional per-tier heads**, loss
  aggregation, and train/val steps. The only file here importing
  pytorch-lightning. hparams persist `args, n_vocab, tokens, tkmodel, metadata,
  reg_widths` → recoverable from the `.ckpt`.
- **`train/model/heads.py`** — `PerTierHeads` + `MoSHead` (mixture-of-softmaxes
  for the content tier when K>0; plain Linear otherwise). `per_tier_unified_log_p`
  combines per-tier head outputs + router posterior into a unified `(B,T,V)`
  log-prob via disjoint-partition scatter (each vocab id belongs to exactly one
  tier — the tier map comes from tokens `vocab_signature`/`tier_classify`).
  `heads_cluster.py` / `heads_diffusion.py` + `losses_diffusion.py` are
  **refuted** experiment heads (kept for the record).
- **Autocast fp32 trap:** any new head `Module` must cast `log_softmax`/`logsumexp`
  back to the input dtype, or per-position buffers stay fp32 and OOM at prodlike
  (pinned by `tests/train/test_per_tier_heads.py::test_bf16_input_preserves_*`).

## Training

`train/trainer.py:main` builds the dataset + model, optionally attaches the
**`GeneralizationGate`** callback (`train/generalization_gate.py`, `GateThresholds`
— early-aborts arms that don't generalize), and runs `pl.Trainer.fit`.
Checkpoints land at `tb_logs/preframr/version_0/checkpoints/` as
`best-epoch=..-val_loss=...ckpt` (+ last). `structural_loss.py` carries the
structural-tier loss term (load-bearing — masking it collapses diversity).

## Predict

`inference/predict.py` (run via `predict-nv.sh` = `docker run --gpus=all …
/preframr/inference/predict.py …`):
1. **`load_model`** — load the `.ckpt`; **recover `pipeline_spec`, `reg_widths`,
   `tokens`, `tkmodel` from the checkpoint** (so you mostly only pass
   `--model-state` + the df-map/dataset/token CSVs + `--tkvocab`).
2. **`get_prompt`** (`train/regdataset`) — pull a prompt from the predict set
   (`--eval-reglogs` / df-map `kind=val`) at `--start-seq`.
3. **`Predictor`** — autoregressive decode; `--constrained-decode` masks logits to
   structurally-valid next atoms (precompute_vocab_arrays / subtoken_arrays);
   `--temperature` / `--top-k` sampling.
4. **Render** — decode tokens → register df → `preframr_audio.render_to_wav` →
   `--wav`; `--predict-dump` saves the prediction-window `df_audio` parquet.
   A safety net (`validate_back_refs` / `validate_pattern_overlays`) rejects
   malformed streams; `--min-acc 0` logs-but-continues.

**Generation gotchas (learned):** `--no-compile` avoids a CUDAGraphs
"overwritten tensor" crash in incremental decode; unconstrained `temperature 1.0`
over the de-merged base-atom vocab emits runaway DELAY/empty-frame tokens → near-
silent drone (use `--constrained-decode` + low temp).

## Deploy envelope

Train: single RTX 4090 (24 GB). Predict: Jetson Orin NX (15.6 GB) at
PROMPT=2048 / MAX=8192; slim images `anarkiwi/preframr-{predict,xpu,jetson}`
(predict-only, built from `Dockerfile.predict`).

## Build / release

`build.sh` builds `anarkiwi/preframr` (+ slim variants) from `Dockerfile`; the
build **runs `./run_tests.sh`** (black / pytest / pylint / pyright / coverage≥77)
before exporting, so the image is test-gated. `VERSION` file → image tag
(`:latest` + `:VERSION`) on push to main / `v*` tag. Local build:
`docker build --build-arg PIP_OPTS="" -f Dockerfile . -t anarkiwi/preframr:<v>`
(empty `PIP_OPTS` = public PyPI; the proxpi mirror in `.env` is host-network only).
Floor the tokens/audio versions in `requirements.txt` + bump `VERSION` to ship a
tokenizer change.
