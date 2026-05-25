# Tokenizer alphabet-coverage bug (2026-05-20)

## Symptom

`RegTokenizer.merge_token_df` raises `KeyError: missing token op=X reg=Y subreg=Z val=W; alphabet does not cover this row` during `train_tokenizer`. The alphabet (`self.tokens` DataFrame) has NO rows with the offending `(op, reg, subreg)` combination, so the substitution-by-nearest-val fallback at `regtokenizer.py:303-318` raises.

## Reproducible failure modes

| pipeline | failing token | tier where seen |
|---|---|---|
| includes `loop` | op=22 (`PATTERN_REPLAY_OP`) reg=-125 subreg=2 various vals | micro_mini, mini |
| includes `hard_restart` | op=25 (`HARD_RESTART_OP`) reg=4 subreg=-1 val=1 | micro_mini |
| includes `ctrl_bigram` | op=42 (`CTRL_BIGRAM_OP`) reg=4 subreg=-1 val=89 | micro_mini |
| includes pre-norm parser passes | op=28 reg=4 subreg=-1 val=6 | mini (without explicit `loop`/`ctrl_bigram` in spec, but parser-builtin still fired) |

The pattern: every transform that emits a custom op-code triggers a
KeyError on first SID parse where that op's val space exceeds what
`accumulate_tokens` enumerated. Workaround: pass all macros OFF via
CLI (`--no-hard-restart-pass --no-loop-pass --no-ctrl-bigram-pass ...`).
This may not fully bypass: parser pre-norm passes (slope / preset)
still fire and may still emit unenumerated ops -- under test in
`contrastive_mini`.

## Hypothesis

The alphabet is built from parsed dfs via `RegTokenizer.accumulate_tokens` (called per-SID during dataset preload), then `merge_token_df` is called on those SAME dfs during `train_tokenizer`. The two paths should be self-consistent by construction — alphabet must contain every (op, reg, subreg, val) tuple in the data.

The bug must be one of:

1. **Alphabet enumeration races the parse output.** If accumulate_tokens runs over a sample / cache that's missing some emissions while merge_token_df processes the full set, coverage gap. Most likely cause.
2. **Decomposition mutates the df.** `_decompose_missing_via_registry` could produce new tokens that don't exist in the alphabet but are also not handled by the substitution path.
3. **Pre-norm parser passes emit ops the Transform registry doesn't claim.** Per AGENTS.md "hardcoded pre-norm passes (slope / preset / gate_slope_shift / hard_restart / etc.) surfaced into the non_set_regs_so_far accumulator" — these parser-builtin emissions may not register their full emission space with the tokenizer alphabet builder.

## Reproduction

Run `contrastive_micro_mini.py` (16 SIDs) or `contrastive_mini.py` (196 SIDs) with `_BASE_TRANSFORMS` including `loop`:
```bash
python3 -m integration_tests.experiments.run contrastive_micro_mini --root /scratch/tmp/preframr_experiments
```
Tokenize phase fails within ~30s with `KeyError`.

## Workaround

Drop `loop` from the pipeline. Fails on a different op (`HARD_RESTART_OP` at reg=4) for micro_mini but appears to work on mini (verify with current contrastive_mini run).

## Why mini PASS earlier didn't hit this

`voice_traj_distributed_set_diff_freq_mini` baseline arm ran with this exact 7-op pipeline on this same 196-SID data and produced val_acc 0.0874 per AGENTS.md. Possibilities:

- HVSC version drift between then and now (verified same v84 via preflight).
- Subtle state pollution: the failed `voice_traj_distributed_set_diff_freq_prodlike` aborted mid-flight may have left bad parsed parquets in the dump cache.
- The earlier run used `set_to_diff` + `voice_trajectory` downstream which restructured the bad emissions into something the alphabet covered.

## Actual root cause (post-investigation)

Two separate issues compounding:

1. **Dtype mismatch on TOKEN_KEYS join.** `accumulate_tokens` and `_merged_and_missing` were merging on TOKEN_KEYS columns with inconsistent integer dtypes (Int8 vs Int64 vs default int). pandas merge silently fails to match across dtypes. Fixed by casting `TOKEN_KEYS` to `Int64` in both code paths.

2. **Substitution gate too strict.** Even after the alphabet is built consistently, the parser/transforms can produce a `val` for a known `(op, reg, subreg)` triple that wasn't seen during alphabet build (parser may have non-determinism on rare paths). `merge_token_df`'s substitution-by-nearest-val logic was gated on `op in _substitutable_ops()`, which excluded `PATTERN_REPLAY_OP`, `HARD_RESTART_OP`, `CTRL_BIGRAM_OP` etc. — those ops failed loudly even when same-(op, reg, subreg) entries existed in the alphabet and a perfectly fine nearest-val substitution was available. Fixed by relaxing the check: substitute whenever same-(op, reg, subreg) rows exist; only raise when the alphabet has no rows at all for that combination.

## Original hypothesis (deprecated)

Was signed/unsigned int dtype mismatch on `reg` column

Pass 1 (`accumulate_tokens`): `reg` column may be stored as `uint8` (0-255).
Pass 2 (`merge_token_df`): `reg` may be cast to `int8` (-128 to 127).
Reg values >= 128 then read as negative (e.g., 131 ↔ -125). The
`df.merge(tokens, on=TOKEN_KEYS, how="left")` at `regtokenizer.py:268`
would fail to match across the signed/unsigned representations.

The first error already shows reg=-125, which IS the int8 view of
uint8 reg=131. Suspicious.

Validation: dump dtypes of `tokens["reg"]` vs `df["reg"]` immediately
before the merge in `_merged_and_missing`. Cast both to a common
signed wide type (`int32`) before merging.

## Action items

1. Validate the dtype-mismatch hypothesis: instrument `_merged_and_missing` to log dtypes + ranges.
2. If confirmed: cast `reg` to `int32` consistently in `accumulate_tokens`, `make_tokens`, and `merge_token_df`. Lossless fix.
3. If refuted: trace which transform actually produces the new (op, reg) tuple via parser-side logging.
4. Audit `_decompose_missing_via_registry` for whether it mutates dfs in pass 2 in ways pass 1 didn't see.
5. Once fixed, re-enable transforms in `contrastive_mini` and confirm the full pipeline tokenizes cleanly.
