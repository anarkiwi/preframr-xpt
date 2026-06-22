# Generic per-axis SID decompiler — STATUS (honest, with numbers)

**Status as of 2026-06-22.** This is the consolidation status of the generic
per-axis SID decompiler work (design: `generic_sid_decompiler.md`). It records
exactly what is **PROVEN** by green tests in the integration branches and what is
**OPEN / a known limit** — with measured numbers, not adjectives. It supersedes the
archetype "zoo" (`unified_generic_recovery.md`) and the format-assuming identity-lift
path (`sidtrace_program_recovery.md` / `generic_tracker_decompile.md`) for the
generic and generative cases; those docs are retained here for the superseded
context and the oracle (yes/no) interface they define.

The two integration branches this status describes:

- **preframr-sidtrace** `integration/sid-decompiler` (from `feature/siddf-bm`) — the
  bounded tracer artifact (SDST: ACMP/SNAP/SIDW/IDXR + SIDDF/STSQ/SDCU + `.sidwr.bin`).
  Clean build (`make distclean && submodule update && make`) + **21 tracer tests green**.
- **preframr-xpt** `integration/sid-decompiler` (from `feat/sid-decompiler-stage3b`,
  merged `feat/sid-decompiler-stage4`) — the host recovery + corpus harness.
  **43 decompiler tests green** (stage0 + stage2 + stage3 + stage3b + stage4).

The **sole correctness gate, everywhere, is residual-zero re-execution** (design §2.7):
a recovered subset must reproduce the byte-exact SID register stream by re-execution.
The gate is **never faked to zero** — the whole-tune residual is reported honestly and
lives entirely in the surfaced (unrecovered) axes.

---

## PROVEN (green tests, measured)

### 1. Length-independence — the central property (design §1.4), measured slope ≈ 0

The recovered object `{G_A}` is **fixed-size**, independent of how long the tune plays.
This is the property the prior output-shaped catalog failed (its Goto80 generic catalog
grew **10.5k → 34k tokens** as the capture window grew).

Measured on the depacker-less Goto80 corpus (`stage4_corpus_results.json`, 8 tunes,
`test_recovered_size_flat_vs_fit_window`):

| tune | driver | recovered `{G_A}` size (bytes) | flat over windows 512→full? | recovered-subset residual (full stream) |
|---|---|---|---|---|
| jch_10        | JCH NewPlayer | 9 | yes (min==max==9) | 0 |
| jch_honolulu  | JCH NewPlayer | 5 | yes | 0 |
| jch_robinson  | JCH NewPlayer | 4 | yes | 0 |
| jch_truth     | JCH NewPlayer | 3 | yes | 0 |
| dm_bfp        | DefMon        | 5 | yes | 0 |
| dm_acxq7      | DefMon        | 3 | yes | 0 |
| dm_bokl0v     | DefMon        | 3 | yes | 0 |
| dm_italic     | DefMon        | 3 | yes | 0 |

The recovered program is **3–9 bytes** and **does not grow with the fit window**
(`recovered_size_min == recovered_size_max` for every tune). The fixed-size object
derived from a **512-frame** window renders **byte-exact over the full stream**
(7,537 frames JCH / 15,062–15,063 frames DefMon) on the axes it claims
(`full_recovered_subset_residual == 0` for all 8). This refutes the old
10.5k→34k output-RLE blowup directly, on real depacker-less drivers, with **no depacker
oracle** — the byte-exact register stream *is* the oracle.

(The artifact size itself has a small nonzero slope, e.g. 0.02–0.62 bytes/frame, because
STATESEQ keeps bounded per-cell samples; the *recovered program* slope is 0, which is the
claim. Both are reported in the JSON.)

### 2. Generative auto-recovery on the hardest case — A Mind Is Born (no tables)

The `{G_A}` recurrence model recovers A Mind Is Born **byte-exact over all 8,190 frames**
(`test_residual_zero_against_white_box_oracle`, `test_residual_zero_against_frozen_fingerprint`)
under the documented don't-cares. This is the case that defeats every "find the table"
approach: the tune has **no note-table / pattern / orderlist** — everything is arithmetic
from one counter. Proven properties:

- **Length-independence test** (`test_length_independence`): the same fixed-size program
  renders a 512-frame slice and the full 8,190 frames both byte-exact; state size is
  identical across windows {128, 512, 2048, 8190} (slope exactly 0).
- **Program size** (`test_program_size_is_small_and_flat`): ~the 254-byte image, i.e.
  **< 0.05 program-bytes/frame** over 8,190 frames — far below storing the output.
- **Generic axis classes auto-recovered** as one procedure (constant / closed-form
  accumulator / table-K identity-lifted / counter-driven): the master counter `C += 2`,
  the closed-form filter `7 + 8·(C>>7)`, the table-K lift (`reg8` lifts `table_f7` at
  base `$F7`, 8 bytes `00 18 26 20 12 24 13 10`, by identity from the widened SNAP —
  `test_amind_snap_widened_captures_zp_generator_table`,
  `test_amind_sdcu_update_dags_present`).

### 3. The seed-image crutch is retired; no-branch-reexec is enforced structurally

Stage 2 retired the seed-image white-box crutch; the host derives `{G_A}` from the SDST
artifact alone (`test_no_seed_image_dependency`, `test_recover_from_psid_identity`). The
**no-branching-re-exec ruling** is enforced by an **AST guard** over `generalize.py`
(`test_closures_are_closed_form_not_branch_reexec`): every recovered axis's `render` is a
pure closed-form / numpy expression over the recovered counter — it must not step a 6502
core, re-run `SRE` in a loop, or execute a branching micro-program. The forbidden tokens
(`step_cpu`, `run_player`, `execute_opcode`, `sre_loop`, `AmindRecurrence`, `cpu6502`,
`LftBackend`) are asserted absent; the BM helper is verified to be typed linear algebra
(no opcode dispatch, no `STA`, no player re-execution).

### 4. Residual-zero is the sole gate and is never faked

For every corpus tune and for A Mind, the recovered subset is byte-exact
(`*_recovered_subset_residual == 0`) and the **whole-tune residual is nonzero and lives
ENTIRELY in the surfaced (unrecovered) axes** (`test_amind_whole_tune_residual_is_exactly_the_surfaced_axes`,
`test_whole_tune_residual_is_exactly_the_surfaced_axes`). e.g. A Mind: the recovered axes
contribute 0 to the residual; all residual is in the surfaced SMC holdouts. The gate is
the byte-exact stream, never a stored patch.

---

## OPEN / known limits (honest)

### A. Whole-tune coverage is low (the surfaced majority)

The recovered subset is byte-exact, but it is a **minority of axes** on the
tracker-driven depacker-less tunes. Measured `frac_axes_closed` (`n_recovered_axes` /
total writing axes) over the corpus:

| driver | frac axes closed | n recovered axes (of ~24–25 surfaced) |
|---|---|---|
| JCH NewPlayer | **4% – 28%** (jch_truth 0.04, robinson 0.08, honolulu 0.12, jch_10 0.28) | 1–7 |
| DefMon        | **4% – 12%** (acxq7/bokl0v/italic 0.04, bfp 0.12) | 1–3 |

The rest are **surfaced as unrecovered** (length-proportional fallback, flagged), never
faked. The whole-tune residual is therefore large (e.g. JCH 120k–163k, DefMon 320k–354k
register-byte mismatches) and **all of it sits in the surfaced columns**. The gap is the
**sequencer cursor** (the nested orderlist→pattern→note traversal — Grid_Runner's freq
axes lift their K-table by identity but the cursor does not close as a fixed-size
recurrence: `test_grid_freq_axes_lift_k_table_by_identity`,
`test_grid_freq_cursor_surfaced_not_faked`) and **DefMon capture** (DefMon currently
yields no SDCU/STSQ state cells in these artifacts: `n_sdcu_cells == 0`,
`n_stateseq_cells == 0` for all 4 DefMon tunes — `test_defmon_has_no_state_cell_sections_REPORTED_LIMITATION`).
These are the **closable gaps** and are owned by the concurrent Stage 5 work.

### B. A Mind reg1 / reg8 / reg10 are PROVABLY irreducible under strict no-branch-reexec

The three fast mid-call SMC accumulators — **reg1 (v0 freq-lo), reg8 (v1 freq-hi),
reg10 (v1 PW-hi)** — are surfaced as unrecovered, with a **data-grounded** reason, not a
hand-wave (`test_amind_smc_axes_surfaced_not_faked`, `test_amind_smc_reason_is_dag_level_and_k_lifted`,
`test_smc_holdouts_reason_cites_bm_verdict`):

- They are driven by cell **`$14`**, an LFSR (`LDA #$b8; SRE $14`) **interleaved** with a
  section-indexed self-modifying store, **branch-selected by the section counter**.
- RULING 1 permits closing an LFSR **only** via Berlekamp–Massey recovering the feedback
  taps — **not** by re-running `SRE` in a loop. BM is run honestly over `$14`'s **genuine
  mid-call value sequence** (the new SDCU `valSeq`, ≥ 256 samples, ≥ 32 distinct values —
  the call-boundary STATESEQ is a useless near-constant for a fast SMC cell:
  `test_sdcu_valseq_present_and_genuine`).
- **Verdict:** BM reports **high linear complexity, L ≈ n/2 in every bit-plane**
  (`test_berlekamp_massey_14_is_not_a_typed_lfsr`: `L ≥ n/3`, every plane `≥ n/4`). `$14`
  is therefore **NOT a low-complexity typed LFSR** — the branch selector co-evolves with
  the LFSR state, so there is no single fixed-size typed recurrence. The reasons cite
  "Berlekamp-Massey" + "NOT a typed LFSR" + "no-branching-re-exec" explicitly.

This is a **genuine limit under the user's rulings**, not a missing feature: closing it
would require branching re-execution, which is forbidden (RULING 1 & 2).

### Limit taxonomy (design §5), as it bites here

- Heavy/whole-sequence SMC where the slice shape changes frame-to-frame (no fixed `U`).
- Data-dependent control flow co-evolving with state (the `$14` case — provably so via BM).
- True high-linear-complexity state (BM `L ≈ n/2` ⇒ not a typed LFSR).
- Deep cross-axis coupling / nested-cursor grammars (the sequencer-cursor gap).

In every case the axis (or frame-span) is **surfaced as unrecovered, length-proportional,
with the reason** — never faked, never hidden in a patch.

---

## User rulings (govern this work)

1. **No branching re-execution.** An axis closes ONLY as a *typed* recurrence
   (constant / closed-form accumulator / table-K identity / BM-recovered LFSR). It must
   never be "recovered" by stepping a 6502 core or re-running a branching micro-program.
   Enforced by the AST guard (§PROVEN.3).
2. **Surface, never fake.** A non-zero residual is reported as an unrecovered axis with a
   data-grounded reason; residual-zero stays the gate. The whole-tune residual is always
   exactly the surfaced axes.
3. **Supersedes the archetype zoo.** The length-proportional output-fit
   (`unified_generic_recovery.md`) and the format-assuming identity-lift
   (`sidtrace_program_recovery.md`, `generic_tracker_decompile.md`) are superseded for the
   generic/generative case; retained for superseded context + the oracle yes/no interface.

---

## SDST section catalog (the bounded tracer artifact)

The tracer emits one compact **SDST** container per tune (a few KB, bounded — keyed by
code site / state cell, NOT by frame count):

| section | tag | what (design ref) |
|---|---|---|
| Access-type map     | **ACMP** | RLE per-address EXEC/READ/WRITE × INIT/PLAY bits — the state/constant/SMC classifier (§2.3–2.4). |
| Post-init RAM snap  | **SNAP** | verbatim post-init RAM image; where array `K` is lifted by identity (§2.4). Widened to capture zero-page generator tables. |
| SID-write summary   | **SIDW** | PC-tagged per-(PC,reg) write summary (count, lastVal). |
| Indexed-read VSA    | **IDXR** | per-PC strided interval `(base, stride, idxMin, idxMax)` — the table sub-case primitive (§2.4). |
| Per-write data-flow | **SIDDF** (SDDF) | bounded backward-slice / DAG summary per (PC,reg): slice PCs, leaves {immediate/ram_read/state_cell/exogenous}, op_seq (§3.1). SDDF carries `nValSeq=0`. |
| Inter-frame samples | **STATESEQ** (STSQ) | bounded per-flagged-cell sample sequence across frames for Daikon/BM (§3.2). |
| Per-cell update DAG  | **SDCU** | the state cell's UPDATE DAG + its **mid-call value sequence** (`valSeq`) — the genuine generator state stream fed to Berlekamp–Massey (§2.5/2.7). The Stage-3b addition. |
| `.sidwr.bin`        | —        | timestamped SID-write stream — the residual-zero gate. |

The acronyms SNAP/SIDW/IDXR (and the legacy SDST artifact "SNAP/SIDW/IDXR" sections)
originate in `sidtrace_program_recovery.md`; SIDDF/STATESEQ/SDCU are the additions this
work specifies (design §3.1–3.2) and `feature/siddf-bm` implements.

### Format note (consolidation)

The SDST header is **v1** in both the pre-BM and post-BM emitters; Stage 3b appended the
per-entry SDCU `valSeq` without a version bump. The depacker-less Goto80 fixtures
(`goto80_jch`, `goto80_defmon`) were distilled before that field existed. The vendored
reader (`preframr_experiments/sid_decompiler/sdst.py`) therefore **disambiguates the
optional `valSeq` field structurally** (parse-with-valSeq, else re-parse-without, accept
only the layout that lands on a valid next section tag / EOF). This changes no recovery
logic and no test; legacy entries get `val_seq=[]` (identical to the new layout's SDDF
`nValSeq=0`).

---

## Staged branch history (none merged; all intact for Stage 5)

**preframr-sidtrace** (linear): `feature/siddf-dataflow` (8c6f5a0) → `feature/stateseq`
(3553277) → `feature/siddf-dag` (8993587) → `feature/siddf-bm` (a5f8a7a, tip — contains
everything). Integration branch `integration/sid-decompiler` is `feature/siddf-bm` under
a new name; 21 tests green from a clean build.

**preframr-xpt** (branching): `…-stage0-amind` (a02efb7) → `…-stage2` (f085677) →
`…-stage3` (1e2caf6) → two siblings off stage3:
- `…-stage3b` (87f37ea): BM closure in `generalize.py` + SDCU reader + new stage3 tests.
- `…-stage4` (621d66f): corpus harness (`stage4_corpus_scaling.py`) + results JSON +
  stage4 tests + goto80 fixtures (did NOT modify recovery algorithms).

Integration branch `integration/sid-decompiler` = `…-stage3b` with `…-stage4` merged
(disjoint apart from the SDST reader format note above) + the design docs + this STATUS;
43 tests green.

All stage branches are left untouched for the concurrent Stage 5 recovery work.
