# preframr-tokens architecture

**Status:** Pointer (2026-06-12). The API reference for the current tokenizer — the
dump parquet format, SID register map, the 127-atom v3 event alphabet, stream grammar,
the `canonical_writes` fidelity contract, the parse-domain (`RegLogParser`) output
schema and marker registers, and the corpus/dataset entry points — now lives in the
**[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens)**.

Historical note: earlier revisions of this doc (git history) describe the RETIRED
(op,reg,subreg,val) substrate — the atom model, op-code tables, FREQ_TRAJ, macro pass
ordering, and the unigram tokenize path. That pipeline was superseded 2026-06-11 by
the event model (`preframr_tokens/events/`); read the old revisions only to interpret
old designs/results, never as a description of the shipped pipeline. The chip facts
that license the v3 canonical form are indexed in the
[preframr-audio README](https://github.com/anarkiwi/preframr-audio); the verification
tool map is [`verification_and_audits.md`](verification_and_audits.md).
