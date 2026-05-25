## Status

**Pending impl** (2026-05-21). Extract ~560 LoC of torch-free corpus
orchestration from main-repo `RegDataset` into a new `Corpus` class
in `preframr-tokens` 0.5.0. Main repo's `RegDataset` becomes a thin
torch.utils.data.Dataset wrapper composing a `Corpus` + a
`BlockMapper`.

## Problem

After 0.4.0's `preframr_tokens.blocks` extraction, main repo's
`preframr/train/regdataset.py` is 753 LoC. **~560 LoC of that is
torch-free orchestration** -- corpus loading, tokenizer building,
disk caching, df-map metadata routing -- bundled inside the
`RegDataset(torch.utils.data.Dataset)` class because it grew there
organically. Only ~150 LoC genuinely needs torch (BlockMapper
management + DataLoader interface + get_prompt's torch tensor
return).

The torch-bound vs torch-free LoC split (numbers measured against
HEAD `preframr/train/regdataset.py`):

| stays in main repo (torch-bound) | LoC |
|---|---|
| `RegDataset.__init__`, `val_block_mapper`, `_val_subset_for` | 30 |
| `__len__`, `__getitem__`, `getseq` (Dataset interface) | 20 |
| `LowMemoryRandomSampler`, `_get_loader`, `get_loader`, `get_val_loader` | 60 |
| `get_prompt` (returns torch tensors) | 42 |
| **subtotal** | **~150** |

| could move to preframr-tokens (torch-free) | LoC |
|---|---|
| `load_dfs` (concurrent.futures + parser_worker; yields numpy) | 73 |
| `make_tokens` (alphabet build) | 95 |
| `_encode_and_save_cached_blocks` (writes .blocks.npy) | 32 |
| `_try_preload_from_disk` (reads tokens.csv / tkmodel.json) | 50 |
| `preload` (orchestrator) | 87 |
| `load` (top-level dispatcher) | 9 |
| `_try_load_from_metadata` (df-map.csv prefetch) | 48 |
| `_load_via_reparse` (re-parse fallback) | 58 |
| `predict_load` (predict-time lean load) | 108 |
| **subtotal** | **~560** |

These methods touch only torch-free state -- `self.args`,
`self.logger`, `self.tokenizer` (RegTokenizer), `self.reg_widths`
(dict), `self.n_vocab` (int), `self.n_words` (int),
`self._tokenize_meta` (dict). The only torch-state touchpoint is
the `block_mapper.add(blocks_path, seq_meta)` call, which slots into
the train-side BlockMapper.

## Goal

After this design lands (in `preframr-tokens` 0.5.0 + main repo
cutover):

- `preframr_tokens.corpus.Corpus` owns the torch-free state +
  orchestration. ~560 LoC of method bodies relocate to the sibling.
- Main repo `preframr/train/regdataset.py` shrinks 753 → ~200 LoC.
  RegDataset class is ~120 LoC of torch.utils.data.Dataset adapter
  composing a `Corpus` + a `BlockMapper`.
- `from preframr.train.regdataset import RegDataset, get_loader, ...`
  keeps working unchanged (back-compat via the same public surface).
- Phase 3 mos4 retry continues running without disturbance --
  `preframr/train/regdataset.py` is bind-mounted; this refactor
  must land *between* experiments, not during one.

## Non-goals

- **Not** moving `BlockMapper` (`preframr/train/block_mapper.py`,
  77 LoC). It holds torch tensors; stays in main repo.
- **Not** moving `get_prompt`. Returns torch tensors; stays.
- **Not** restructuring the DataLoader factories. They're already
  thin (`get_loader`, `get_val_loader`) and torch-bound.
- **Not** changing on-disk artefact formats (tokens.csv,
  *.blocks.npy, df-map.csv, _reg_widths.json sidecar). Round-trip
  identical before and after.

## API: `preframr_tokens.corpus.Corpus`

```python
class Corpus:
    """Torch-free corpus orchestration: parse + tokenize +
    disk-cache .blocks.npy + load metadata. Owns the RegTokenizer
    + corpus-wide state (reg_widths, n_vocab, n_words, tokenize
    metadata). Main-repo torch.utils.data.Dataset adapters compose
    a Corpus + a BlockMapper to expose the train-side interface."""

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.tokenizer = RegTokenizer(args, tokens=None, logger=logger)
        self.reg_widths: dict[int, int] = {}
        self.n_vocab: int = 0
        self.n_words: int = 0
        self.val_subset_names: list[str] = []
        self._tokenize_meta: dict | None = None

    # ------ parse + tokenize stage ------

    def load_dfs(self, reglogs=None, dump_files=None, max_perm=99,
                 encode=True):
        """Yield (dump_file, i, df, seq, irq, blocks). Parallel
        parser_worker; tokenizer.encode if tokens are already built
        and encode=True."""

    def make_tokens(self, reglogs, eval_reglogs=""):
        """Build the token alphabet from train + eval blocks.
        Mutates self.tokenizer; populates self._tokenize_meta;
        returns (train_files, val_files, cached_blocks)."""

    def encode_and_save_cached_blocks(self, cached_blocks):
        """Write .blocks.npy for each cached voiced-block group
        using the now-finalised tokenizer."""

    def try_preload_from_disk(self) -> bool:
        """Hydrate tokenizer from args.token_csv (+ args.tkmodel if
        tkvocab>0). Returns True iff both files exist and have content."""

    def preload(self, tokens=None, tkmodel=None):
        """Top-level tokenize-stage orchestrator. Either: (a) explicit
        tokens passed in -> tokenizer.load; (b) try_preload_from_disk
        succeeds; (c) make_tokens + write tokens.csv + write
        dataset.csv + train Unigram if tkvocab>0 +
        encode_and_save_cached_blocks. Side-effects on disk match the
        previous RegDataset.preload behaviour bit-for-bit."""

    # ------ train-stage block routing ------

    def iter_block_seqs(self):
        """Yield (kind, blocks_path, seq_meta) tuples for the train
        stage. Tries metadata-fast-path first (`df-map.csv` with the
        new `irq` / `n_rotations` columns + `_reg_widths.json`
        sidecar); falls back to re-parsing the corpus. Caller routes
        each yielded tuple to the matching BlockMapper.

        Sets `self.n_vocab`, `self.n_words`, `self.reg_widths` as a
        side effect."""

    def iter_predict_block_seqs(self):
        """Lean predict-time iterator. Yields (kind, blocks_path,
        seq_meta) for the single target file selected by
        --predict-set / --start-seq. May parse just that one file if
        no cached blocks exist."""

    # ------ utility ------

    def reg_widths_sidecar_path(self) -> str:
        """`preframr_tokens.blocks.reg_widths_path(args.df_map_csv)`."""
```

State semantics (what each attr means after each call):

- After `__init__`: tokenizer is empty (no alphabet); reg_widths is
  `{}`; n_vocab=0.
- After successful `try_preload_from_disk`: tokenizer populated from
  disk; reg_widths still `{}` (set later by `iter_block_seqs`).
- After `make_tokens`: tokenizer.tokens populated; `_tokenize_meta`
  carries irq_by_file, rotations_by_file, kind_by_file, reg_widths,
  val_subsets. cached_blocks is the in-memory cache to be written by
  `encode_and_save_cached_blocks`.
- After `preload(tokens=...)`: tokenizer loaded; no I/O.
- After `preload()` (no tokens arg): full tokenize stage; tokens.csv
  + dataset.csv + .blocks.npy + df-map.csv + _reg_widths.json all
  written to args paths.
- After `iter_block_seqs()` is exhausted: n_vocab, n_words,
  reg_widths populated. block_mappers populated by the caller from
  the yielded tuples.

## Main repo `RegDataset` after extraction

```python
class RegDataset(torch.utils.data.Dataset):
    def __init__(self, args, logger=logging):
        self.corpus = Corpus(args, logger)
        self.block_mapper = BlockMapper(args.seq_len)
        self.val_block_mappers = OrderedDict()
        self._empty_val_mapper = None

    # back-compat attribute access for code that reads e.g.
    # ``dataset.tokenizer`` or ``dataset.args``
    args = property(lambda self: self.corpus.args)
    logger = property(lambda self: self.corpus.logger)
    tokenizer = property(lambda self: self.corpus.tokenizer)
    reg_widths = property(lambda self: self.corpus.reg_widths)
    n_vocab = property(lambda self: self.corpus.n_vocab)
    n_words = property(lambda self: self.corpus.n_words)

    @property
    def val_block_mapper(self):
        if LEGACY_EVAL_SUBSET_NAME in self.val_block_mappers:
            return self.val_block_mappers[LEGACY_EVAL_SUBSET_NAME]
        if self.val_block_mappers:
            return next(iter(self.val_block_mappers.values()))
        if self._empty_val_mapper is None:
            self._empty_val_mapper = BlockMapper(self.corpus.args.seq_len)
        return self._empty_val_mapper

    def _val_subset_for(self, name):
        if name not in self.val_block_mappers:
            self.val_block_mappers[name] = BlockMapper(self.corpus.args.seq_len)
        return self.val_block_mappers[name]

    def preload(self, tokens=None, tkmodel=None):
        self.corpus.preload(tokens=tokens, tkmodel=tkmodel)

    def load(self):
        for kind, blocks_path, seq_meta in self.corpus.iter_block_seqs():
            if kind == "train":
                target = self.block_mapper
            else:
                target = self._val_subset_for(kind)
            target.add(blocks_path, seq_meta)
        self.block_mapper.finalize()
        for m in self.val_block_mappers.values():
            m.finalize()

    def predict_load(self):
        for kind, blocks_path, seq_meta in self.corpus.iter_predict_block_seqs():
            if kind == "train":
                target = self.block_mapper
            else:
                target = self._val_subset_for(kind)
            target.add(blocks_path, seq_meta)
        self.block_mapper.finalize()
        for m in self.val_block_mappers.values():
            m.finalize()

    def __len__(self):
        return len(self.block_mapper)

    def __getitem__(self, index):
        return self.block_mapper[index]

    def getseq(self, rotation_i, block_j=0):
        return self.block_mapper.getseq(rotation_i, block_j)
```

Resulting `regdataset.py` is ~200 LoC: RegDataset class (~120),
LowMemoryRandomSampler (~10), `_get_loader` / `get_loader` /
`get_val_loader` (~25), `get_prompt` (~42).

## State-routing contract

The Corpus needs to tell the caller which `BlockMapper` each block
file belongs to. The yielded triple is `(kind: str, blocks_path:
str, seq_meta: SeqMeta)`:

- `kind == "train"`: goes into `self.block_mapper`.
- `kind == "eval_a"`, `"eval_b_*"`, or legacy `"eval"`: goes into
  `self.val_block_mappers[kind]`.

`SeqMeta` is currently defined in
`preframr/train/block_mapper.py` (next to BlockMapper).
For Corpus to construct one, **`SeqMeta` also has to move** to
preframr-tokens (e.g. `preframr_tokens.blocks.SeqMeta` -- it's a
NamedTuple of `(irq, df_file, i)`, pure data, no torch). Add to
0.5.0. Main repo's `BlockMapper` imports it from preframr-tokens.

## Reg_widths sidecar

`preframr_tokens.blocks.reg_widths_path(df_map_csv_path)` is the
sidecar JSON path helper. Already in 0.4.0. Corpus uses it to read
+ write the `_reg_widths.json` file alongside `df-map.csv`.

## Risks

- **State semantics divergence**: getting which method sets which
  attribute right is the load-bearing detail. Documented in the
  state-table above; tested by a back-compat integration test that
  parses the same corpus through old and new code and asserts
  identical tokens.csv, dataset.csv, df-map.csv, *.blocks.npy.
- **`getattr(self.args, "...", default)` everywhere**: many places
  use defensive attribute access on args. Corpus copies that.
- **Logging output divergence**: `self.logger.info(...)` calls move
  to Corpus but the level + content matches; tests don't assert on
  log content.
- **predict_load fallback**: when blocks aren't cached and only one
  file needs parsing, it does a focused load_dfs(dump_files=[...]).
  Preserved in Corpus.iter_predict_block_seqs.
- **No mid-experiment landing**: bind-mount rule still applies to
  `preframr/train/regdataset.py`. Land between experiments.
- **Sibling test coverage**: comprehensive tests for Corpus need
  real RegTokenizer + parser fixtures, which only the main repo
  has. Sibling gets smoke tests (constructor, basic round-trip on
  a tiny fixture); main repo's existing `tests/train/test_regdataset.py`
  + `test_regdataset_helpers.py` + `test_regdataset_unit.py`
  continue exercising the full end-to-end paths through RegDataset
  (which now delegates to Corpus).

## Success criteria

1. preframr-tokens 0.5.0 published with `preframr_tokens.corpus`
   module + `preframr_tokens.blocks.SeqMeta` move.
2. Main repo `preframr/train/regdataset.py` <= 220 LoC after
   cutover.
3. Existing `tests/train/test_regdataset*.py` (~50 tests) pass
   unchanged in the rebaked image.
4. `run_memorize_int_test.sh` runs to completion on smoke tier --
   end-to-end runtime validation that the orchestration moves
   didn't break the parse → tokenize → train pipeline.
5. Phase 3 mos4 retry (currently in flight) completes against the
   pre-refactor code; refactor lands after.

## Effort

~1 day:

- Sibling Corpus class + `SeqMeta` move: ~3 hr.
- Sibling smoke tests + lint + pyproject coverage omit: ~30 min.
- 0.5.0 release (your end, GH Actions OIDC publish on `v0.5.0`
  tag).
- Main repo cutover: refactor regdataset.py to use Corpus +
  property forwarding, ~2 hr.
- Update `tests/train/test_regdataset_unit.py` mock targets where
  they patch `preframr.train.regdataset.RegLogParser` (those calls
  live in preframr_tokens.blocks now -- already cutover for
  parser_worker in 0.4.0; same pattern for the methods that move).
  ~30 min.
- Rebake + smoke test: ~10 min.

## References

- `model_regdataset_decomposition_design.md` -- the earlier in-place
  split design; this design supersedes it for the regdataset half.
- `preframr-tokens` v0.4.0 release (`preframr_tokens.blocks`) -- the
  precedent for the helper-extraction shape this design follows.
