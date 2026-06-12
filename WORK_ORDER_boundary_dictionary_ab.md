# WORK ORDER: boundary-dictionary cascade + triage + canonical A/B (execute when tokens lands, then delete this file)

**Mission:** consume the preframr-tokens event-boundary-respecting dictionary (its
`WORK_ORDER_event_boundary_dictionary.md`, executed and deleted on tokens `main`): release the
stack, run the kill-cheap static triage, then (if it survives) ONE canonical A/B run, decide via
the gates in `design/encoding/event_boundary_dictionary_proposal.md`, and write back. This file is
the runbook for that proposal — read it plus `design/encoding/encoding_density_frontier.md`
(§1a/§7 especially) and AGENTS.md before starting. Work on the /scratch side: **fogbank** for
CPU/builds, **defroster** for the one training run. This work order constitutes the operator's
authorization for the release cascade in P1.

**Decision-metric law (carry-over, non-negotiable):** all-tier `val_acc` AND raw
cross-tokenization content-tier argmax are CONFOUNDED across tokenizations (population +
granularity — frontier §1a). Decisions use **bits/canonical-atom** and **position-matched argmax**
only, **full-eval** (`--max-blocks 0`) only — the sampled-vs-full gap measured 2.4× on eval_a
(0.077 sampled vs 0.187 full at the same ckpt). Never quote a sampled number without labeling it.

## Reference numbers (everything you compare against; all artifact-backed in `data/audit/`)

- Corpus: 856 dumps single-speed; **~50k atoms/tune median / ~85k mean** (862-tune survey);
  unconstrained BPE-2048 compressed **2.73×** (~18k/31k tokens). `.atoms.zst` atom caches are
  tkvocab-independent and MUST survive (no codec change — verify, P0).
- Atoms-only baseline (tkvocab=0): ckpt `/scratch/tmp/v3c_final.ckpt` (ep99, val_acc 0.561).
  **bits/canonical-atom 1.931 / 2.001 / 2.221** (eval_a / eval_b_daglish / eval_b_follin);
  full-eval content 0.515 / 0.516 / 0.479.
- Unconstrained BPE-2048 (the refuted-as-measured arm): root
  `/scratch/tmp/preframr_experiments/unigram_canonical_v4`; matched-steps endpoint
  `version_2/best-epoch=299-val_loss=4.7437.ckpt`, bits/atom **2.390 / 2.616 / 2.681**
  (= **1.24 / 1.31 / 1.21×** vs atoms). The boundary dictionary must beat THIS to justify itself:
  same-ballpark compression at materially better bits/atom.
- Wallclock anchors: atoms epoch ~1.5 min; dictionary epochs faster by ~compression; tokenize-only
  sweeps are cheap (atom cache + the tokens 0.50.0 parallel block pass).

## P0 — preflight

1. Verify tokens `main`: the boundary-dictionary work is merged, its work-order file is DELETED,
   `tests/test_dictionary_segmentation.py` exists and CI is green. If not → STOP, report.
2. Verify **no codec change**: `stream.EVENT_FORMAT_VERSION` and `events/dataset.ATOM_CACHE_VERSION`
   unchanged vs 0.50.0. If changed → STOP (violates the tokens work order; the corpus re-encode
   decision is the operator's).
3. `git pull` all repos; xpt tests green.

## P1 — release cascade (authorized by this work order)

Authority for mechanics: `design/references/release_build_cache.md`. Outline:

1. preframr-tokens: tag `v0.51.0` on merged main → `release.yml` (tag-triggered) publishes to
   PyPI. Verify the wheel installs and imports on fogbank.
2. preframr framework: bump floor `preframr-tokens>=0.51.0` in `requirements.txt`,
   `predict-requirements.txt`, `jetson/predict-requirements.txt` → merge to main (release.yml
   fires on main-push: Docker Hub `:0.2.30`+`:latest`) + `git tag -a v0.2.30`. Build locally in
   parallel with the push (standing rule); pull the image to defroster.
3. xpt: `ARG BASE=anarkiwi/preframr:0.2.30`, push (docker.yml gates).

## P2 — static triage (minutes, fogbank) — THE KILL GATE

Fresh root `/scratch/tmp/preframr_experiments/unigram_boundary_v1` (the old v4 root's cached
tokenization is unconstrained-merge output under the same tkvocab keys — never reuse it).

1. Corpus pre-encode check: `.atoms.zst` sidecars present (`preencode_corpus.sh --only-missing`
   on fogbank if HVSC changed); the sweep then only retrains dictionaries.
2. Train boundary dictionaries at **tkvocab {1024, 2048, 4096}** (tokenize stage only). Per vocab
   report: **compression** (atoms/tokens, mean AND median per tune — state the unit, the 30k-vs-85k
   confusion was a wrong-unit bug), tunes-per-8192-window, **live-vocab %** (unconstrained 2048 ran
   ~98% — watch it stay near), and `audit/learnability_triage.py` metrics (h_k, induction-copy).
3. **Merge-table classification** (cheap, do it here): decode every vocab piece to atoms; classify
   each as within-value digit run / head+payload / DT / header / marker singleton / other; report
   piece-count and frequency-weighted-savings shares + top-50 pieces. If within-value digits +
   head pairs dominate, the deterministic packs (frontier §3/§4) capture most of the win — feeds
   the PARTIAL gate.
4. **KILL: compression < 1.5× at every vocab** → skip P3/P4, go to P5 with outcome
   REJECT-at-triage (fallback = deterministic packs; context arc proceeds on atoms-only).

## P3 — canonical A/B (defroster, one run)

`generalize --tkvocab <triage knee, default 2048>` on the 0.2.30 stack, fresh root, same
corpus/holdouts as ever. Every §7 lesson applies:

- **Matched steps by construction:** `max_epochs ≈ ceil(100 × compression)` (atoms baseline =
  ~100 epochs; dictionary epochs have ~compression× fewer optimizer steps). Set it deliberately —
  **early-stop NEVER fires under AdamWScheduleFree** (AGENTS gotcha).
- **Endpoint capture:** `save_top_k=1` is per-Lightning-version — note which version dir each
  fit() writes; verify the final-epoch-region best ckpt exists on disk before declaring anything
  uncaptured (the §7C mistake).
- `nohup … & disown`, don't poll; ~2.5 h total at matched steps.
- Optional trajectory: sampled per-class audit every ~25 epochs (label SAMPLED; decisions
  full-eval only).

## P4 — decision audit

1. **Land the permanent tool first**: `preframr_experiments/audit/bits_per_canonical_atom.py` —
   per eval dump, teacher-forced total NLL of the token sequence ÷ the dump's **canonical atom
   count** (= its atoms-only `dump_token_ids` length, available from the atom cache); reports
   per-subset mean + per-tune detail, full-eval. **Self-check before use:** it must reproduce the
   committed v3c/v4 numbers (atoms 1.931/2.001/2.221; v4 ep174 ≈2.71–2.76 — two prior script
   variants differ ~2%, match either, state which) within tolerance. This tool is the standing
   decision metric from now on; commit it regardless of outcome.
2. Run full-eval on the boundary endpoint ckpt: **bits/canonical-atom** (vs atoms AND vs v4 ep299)
   and **position-matched argmax** (§7B methodology: base = atom positions covered by length-1
   tokens; score the atoms-only ckpt restricted to those positions vs the dictionary's base-token
   accuracy on the same positions).
3. Content-tier per-KIND map (full-eval) for the record — expect melody (NI stepwise) to stay
   ~0.11–0.15 regardless; that is NOT a gate (high-entropy, frontier §2).

## P5 — gates + writeback (the established discipline)

Apply `design/encoding/event_boundary_dictionary_proposal.md` gates verbatim:

- **ADOPT** — bits/canonical-atom ≤ atoms × ~1.05 at matched steps AND compression ≥ 1.8×: the
  dictionary becomes the default tokenization for the context arc. Re-run frontier §5's window
  arithmetic in dictionary tokens (median tune at 16384 ≈ 1–2 windows); update AGENTS NEXT so the
  `seq_len`/aligned-windows re-cut is specified in dictionary tokens.
- **PARTIAL** — parity only at 1.3–1.8×: weigh against the deterministic packs using the P2
  merge-table shares; prefer whichever captures the gain more cheaply; record the decision.
- **REJECT** — bits cost > ~1.1× at every vocab: deterministic packs are the fallback (frontier
  §3 per-lane byte-pack ~12% — FD_STEP 1.62 / PW_STEP 1.53 / PW_RAMP 1.52 / FD_RAMP 1.49 /
  NI_STEP 1.46 digits-per-value; §4 head-amortization, ceiling 17.4%); context arc proceeds on
  atoms-only.

Writeback, all outcomes: artifacts (triage report, merge-table classification, audit JSONs) →
`data/audit/`; update the proposal doc Status, frontier §5/§6 (+ a §7-style results addendum),
the AGENTS.md resolved log and NEXT ordering, and `design/generation/long_range_structure.md`
window math if ADOPT. Then `git rm WORK_ORDER_boundary_dictionary_ab.md` (this file must not
survive), commit per repo conventions, push (branch + PR per the last two writebacks, or direct
to main per operator instruction).

## Gotchas (every one has bitten before)

- Fresh run root; never reuse v4's. The dataset cache keys on tkvocab, not on merge semantics.
- NFS hygiene: fogbank IS the /scratch server; cap pools, no lingering `tail -f`, stop
  `preframr_tb` before deleting tb_logs.
- `RUST_MIN_STACK` is already plumbed in `_docker_run` (likely unnecessary now — per-unit words
  shrink the trainer lattice — but leave it).
- Audits support `--device cpu` if defroster is busy. Don't bind-mount unbaked code into runs.
- Quote eval subsets in the fixed order eval_a / eval_b_daglish / eval_b_follin and label
  sampled vs full-eval everywhere.
