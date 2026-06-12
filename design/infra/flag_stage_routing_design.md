# flag stage routing — design note

**Status:** Pending impl. Touches `preframr_experiments/base.py`. (2026-06-12: macro flags are
semantically inert under the unconditional event encoding — the live stage-specific flags are
`--tkvocab` and train-side knobs; routing remains worthwhile, the macro-flag rows are legacy.)

Framework follow-up from AGENTS.md §Framework follow-ups:
`run_arm` (`preframr_experiments/base.py:786`) forwards
`arm.extra_cargs` and `cargs` to every stage (parse, tokenize,
train); train-only flags need stage-aware routing.

Today this works because all stages share `preframr.args.add_args`
— every flag is recognised by every stage's argparse, even when
it has no effect. The cost is conceptual (operators must remember
which flags do something per-stage) and forward-compatibility
(new stage-specific flags could collide if not partitioned).

Cloud-rental prereq adjacent: the `cloud_rental_runner_design.md` features
(`--resume` / auto early-abort / `--max-parallel-arms`) all assume a clean spec
contract. Stage-clear routing reduces the surface for "this flag had no effect
because the stage doesn't read it" debugging.

## Current state

`preframr/args.py` defines 84 flags via one shared `add_args`
parser. Stages call `add_args(argparse.ArgumentParser())` +
`parser.parse_args()`:

- `parse.py`: lines 42-43.
- `stftokenize.py`: lines 11-12.
- `train.py`: lines 142-143.
- `predict.py`: lines 541-542.

`run_arm` (`preframr_experiments/base.py:786`) constructs `cargs` once and
`shlex.split()`s it onto each stage's docker command line:

```python
cargs = f"--no-require-pq --seq-len {spec.seq_len} ... {arm.extra_cargs}"
# Stage 1: parse
... *shlex.split(arm.extra_cargs), ...
# Stage 2: tokenize
... *shlex.split(cargs), ...
# Stage 3: train
... *shlex.split(cargs), *shlex.split(spec.effective_train_args()), *shlex.split(train_overrides_str), ...
```

Train gets cargs + train_args + training_overrides. Parse gets only
`arm.extra_cargs`. Tokenize gets cargs. The asymmetry is intentional
(parse doesn't care about seq_len, tkvocab) but not declared
anywhere — it's emergent from the run_arm body.

## Failure modes

What does NOT fail today (because argparse accepts any flag):
- Train-only `--learning-rate` passed to parse → parsed, ignored.
- Encoder-only `--loop-lookahead 3` passed to train → parsed,
  ignored (train doesn't read it).

What CAN fail in the future:
- A new flag named identically across stages with different
  defaults. argparse accepts the last one; silent precedence bug.
- Removing a flag from `add_args` (because the using stage no
  longer needs it). Other stages still see the flag IF the operator
  passed it; argparse errors with "unrecognized argument" only when
  the flag isn't in the parser AT ALL. Removing the flag from
  add_args trips every stage simultaneously.
- A flag that's deprecated for one stage but kept for another.
  No mechanism to express this.

Slower-failure modes today:
- Operator passes `--learning-rate 1e-3` in `arm.extra_cargs`
  thinking it's an arm-level setting; it goes to parse and
  tokenize too (ignored) but the train_args effective lr is the
  one from `prodlike_train_args()`. The arm's lr does nothing.
  No error. Surprise.
- Spec author adds `arm.extra_cargs="--my-new-encoder-flag"` after
  adding `--my-new-encoder-flag` to args.py. The flag goes to
  train too, which doesn't read it. No issue today, but if the
  same name is later given to a train flag, silent override.

## Approach

Tag every flag with its target stage(s). At `run_arm` build time,
filter `arm.extra_cargs` per stage so only stage-relevant flags
reach each subprocess. Unknown flags fail loud at run start.

### Stage taxonomy

Five stages identified from the existing run_arm:

| Stage | Purpose | Reads |
|---|---|---|
| `parse` | `parse.py` | reglog flags, macro pass toggles, audit flags |
| `tokenize` | `stftokenize.py` | encoder flags (cents, max-perm), tkvocab |
| `train` | `train.py` | model body, optimiser, training hyperparams |
| `predict` | `predict.py` | sampling, decode (constrained-decode, top-k) |
| `audit` | validate_branches.sh | parse subset + macro toggles |

Some flags are in multiple stages (e.g. `--cents` affects parse +
tokenize). The taxonomy expresses a set, not a unique mapping.

### Tagging mechanism

Per-flag metadata via a registry next to `args.py`:

```python
# preframr/args.py
FLAG_STAGES = {
    # Data tier / I/O
    "reglog": {"parse", "tokenize", "train", "predict"},
    "eval-reglogs": {"tokenize", "train"},
    "df-map-csv": {"parse", "tokenize", "train", "predict"},
    "dataset-csv": {"tokenize", "train"},
    "token-csv": {"tokenize", "train", "predict"},
    "tkmodel": {"train", "predict"},
    # Model body (train + predict; predict reads checkpoint shape)
    "layers": {"train", "predict"},
    "heads": {"train", "predict"},
    "kv-heads": {"train", "predict"},
    "embed": {"train", "predict"},
    "intermediate": {"train", "predict"},
    # Encoder passes (parse + tokenize)
    "loop-pass": {"parse", "tokenize"},
    "loop-lookahead": {"parse", "tokenize"},
    "instrument-pass": {"parse", "tokenize"},
    "cents": {"parse", "tokenize"},
    "max-perm": {"parse", "tokenize", "train"},
    # ... ~84 entries total
}
```

The registry lives alongside `add_args` so adding a new flag
naturally prompts tagging. Validation at `add_args` end:

```python
def add_args(parser):
    ...
    # End-of-fn validator: every flag we registered also has a stage tag.
    parser_flags = {a.dest.replace('_', '-') for a in parser._actions if a.dest != 'help'}
    untagged = parser_flags - set(FLAG_STAGES.keys())
    if untagged:
        raise RuntimeError(f"flags without stage tag: {sorted(untagged)}")
    return parser
```

### Per-stage parser construction

Stages get a filtered parser instead of the full `add_args`:

```python
def add_stage_args(parser, stage: str):
    """Add only flags whose FLAG_STAGES set contains ``stage``."""
    full = argparse.ArgumentParser(add_help=False)
    add_args(full)
    for action in full._actions:
        if action.dest == 'help':
            continue
        flag_name = action.dest.replace('_', '-')
        if stage in FLAG_STAGES.get(flag_name, set()):
            parser._add_action(action)
    return parser
```

Stages call `add_stage_args(parser, "parse")` (etc.). Unknown flags
in that stage's argv → argparse error.

### `run_arm` stage filtering

`base.py:run_arm` filters `arm.extra_cargs` before forwarding:

```python
def _filter_cargs_for_stage(cargs_str, stage: str) -> str:
    """Drop flags from cargs_str whose FLAG_STAGES set doesn't
    include ``stage``. Logs dropped flags at INFO so operators see
    the no-op cargs that argparse would otherwise silently accept."""
    ...
```

Called once per stage construction. Drop list is logged so an
operator can see "you passed --learning-rate to parse; it doesn't
apply at that stage."

## Backward compatibility

Filter is permissive: unknown flags pass through (so we don't
break existing scripts that pass deprecated or out-of-tree flags).
The strict mode is engaged only when the spec validates against
`FLAG_STAGES`.

`Arm.extra_cargs`: today is one string. After the change, still
one string but the runner partitions it. No spec changes needed
for existing arms.

`training_overrides`: today is a dict of train-only key/value
pairs. The new contract enforces this — `training_overrides` keys
must have `"train" in FLAG_STAGES[key]`. Validation at spec
__post_init__.

## Validation strategy

**L0 — unit (`tests/test_flag_stages.py`):**
- Every flag in `args.py` is in `FLAG_STAGES` (no missing tags).
- Every key in `FLAG_STAGES` is in `args.py` (no orphan tags).
- Stage filtering produces the expected dropped-flag list for a
  test cargs string.

**L1 — integration (`tests/test_run_arm_stage_routing.py`):**
- Fixture arm with `extra_cargs = "--learning-rate 1e-3
  --loop-lookahead 3"`. Assert: parse stage docker cmd contains
  `--loop-lookahead 3` but NOT `--learning-rate`; train stage has
  both.

**L2 — smoke regen:** re-run `run_memorize_int_test.sh` end-to-end.
Assert: pass with byte-identical output (no flags were dropped
that previous behaviour relied on).

**L3 — error mode test:** spec with `training_overrides={
"loop_lookahead": 3}` (an encoder flag in train overrides). Assert
spec construction raises `ValueError`.

## Risks

- **Untagged out-of-tree flag.** A 3rd party fork that adds a flag
  without registering will trigger the validator. Mitigation: the
  validator only fires when explicitly enabled (off by default in
  Phase 1; on by default in Phase 2 once internal flags are
  tagged).
- **Multi-stage flag ambiguity.** A flag like `--max-perm` appears
  in 3 stages. If its meaning is the same across stages, fine.
  If it diverges (e.g. parse uses it as a parse limit, train as a
  dataloader bound), the divergence is invisible. Mitigation: doc
  comment per flag in `FLAG_STAGES` registry; runner logs the
  resolved value per stage on the docker cmd line.
- **Spec authors forgetting to tag new flags.** L0 validator
  catches at startup. No silent regressions.

## Effort

- `FLAG_STAGES` registry initial pass + `add_stage_args` wrapper:
  **~0.5 day** (84 flags × ~30s thinking each).
- `_filter_cargs_for_stage` in `base.py` + run_arm wiring:
  **~0.3 day**.
- Per-stage entrypoints in `parse.py` / `stftokenize.py` /
  `train.py` / `predict.py` switched to `add_stage_args`:
  **~0.2 day**.
- L0-L3 tests: **~0.5 day**.
- AGENTS.md note + docstring updates: **~0.1 day**.

Total: **~1.5 days**.

## Phased delivery

Phase 1 (lower-risk):
- Land `FLAG_STAGES` registry + `add_stage_args` + L0 unit test.
- DO NOT change stage entrypoints; they keep using full
  `add_args`.
- DO NOT enforce the dest-vs-tag completeness check at startup.
- Effect: registry exists in-tree, future flag additions can be
  tagged voluntarily, but no behaviour change.

Phase 2 (behaviour change):
- Switch stage entrypoints to `add_stage_args`.
- Enable startup completeness check.
- `_filter_cargs_for_stage` in run_arm.
- L1-L3 tests + smoke regen.

Land Phase 1 first; soak for a session or two; then Phase 2.

## Out of scope

- **Refactoring `args.py` into per-stage modules.** The single
  `add_args` is convenient for the long tail of arguments; the
  tagging mechanism is sufficient discipline.
- **Auto-deriving FLAG_STAGES from grep'ing each stage's body.**
  Tempting but fragile; some flags are read indirectly through
  `args`-passing layers. Manual tagging is explicit and reviewable.
- **`run_*_int_test.sh` script flag splits.** The legacy bash
  harnesses are thin wrappers; their flag handling is independent.
  Future work could route through the new contract.

## Order of operations

1. Land this design (reviewer pass).
2. Phase 1: registry + tests.
3. Phase 2: enforcement.
5. AGENTS.md update: move §Framework follow-up entry to Resolved.

## Connection to cloud-rental prereqs

`auto_early_abort` (sibling design) declares a `decision_rule` on
`ExperimentSpec`; the rule reads metrics by name. Cleaner spec
contracts (stage-tagged flags) make the rule's input space
auditable — when the rule says "compute Δ on val_acc_at_best_loss,"
we know that metric came from a train-stage metric extractor and
isn't accidentally a parse-time artefact.

`--resume` (sibling design) keys parse-stage cache on
`arm.extra_cargs`. With stage filtering, the parse key only
hashes flags in `parse`-tagged set — invalidation is precise to
the encoder pass change, not noised by unrelated train-only
flag drift. Reduces unnecessary re-parses.

`--max-parallel-arms` (sibling design) is orthogonal but benefits
from the same clarity: parallel arms with different train flags
but identical encoder flags reuse parse caches via `--resume`.
