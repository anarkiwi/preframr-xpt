# preframr-tokens architecture

**Status:** Pointer. The API reference for the current codec — the BACC step/tracker codec
(VOCAB=34), the BACC primitive (subsumes VIB/SLIDE/ARP/PWM/ADSR/sweeps), `recover_from_sid`
(sid-only recovery via the deterministic `preframr-sidtrace` binary → `.sidwr.bin` + `.bus.bin`),
the per-driver backends (hubbard/goattracker/lft, DMC in progress), `program_to_ids`/`measure`/
`ids_to_program`, the residual-zero fidelity contract, the absolute A440/12-TET note grid + backward
Transpose op, and the corpus/dataset entry points — now lives in the
**[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens)**.

Historical note: TWO retirements precede the current codec. (1) the `(op,reg,subreg,val)` substrate —
the atom model, op-code tables, FREQ_TRAJ, macro pass ordering, unigram tokenize path — was superseded
by (2) the event model (`preframr_tokens/events/`; 127-atom alphabet, `canonical_writes` contract,
`RegLogParser`), which is in turn superseded by (3) the BACC codec (current). Read the old revisions
(git history) only to interpret old designs/results, never as a description of the shipped pipeline.
The chip facts that license the residual-zero form are indexed in the
[preframr-audio README](https://github.com/anarkiwi/preframr-audio); the verification
tool map is [`verification_and_audits.md`](verification_and_audits.md).
