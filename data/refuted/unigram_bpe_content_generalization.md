# Unigram BPE (tkvocab>0) harms content generalization — REFUTED as the context lever (true gap ~1.4× bits/atom)

**2026-06-12 (magnitude RETRACTED same day after the §7 de-confounding audit — see
`design/encoding/encoding_density_frontier.md` §1a/§7/§8 and `data/audit/deconfound_summary.md`).**
The canonical learnability run (`generalize`, 14M body) compared atoms-only (tkvocab=0) vs unigram
BPE (tkvocab=2048, ~2.73× stream compression), same corpus/holdouts.

**Raw cross-tokenization argmax (matched ~epoch 100) — now known confounded, kept for record:**

| eval subset | BPE-2048 | atoms-only |
|---|---|---|
| eval_a | 0.049 | 0.479 |
| eval_b_daglish | 0.088 | 0.559 |
| eval_b_follin | 0.039 | 0.416 |

The raw table read ~6–11×. **That magnitude is RETRACTED.** All three suspected confounds were
**confirmed** by the §7 audit, so the table compared unlike things:
- **(a) population.** The BPE column scored only *surviving base atoms*; the atoms-only column
  scored ALL content atoms. Restricting atoms-only to base positions drags it **0.51→0.26** (§7B) —
  different populations by construction.
- **(b) granularity.** Merged-token argmax is joint over k atoms; per-token argmax doesn't compare
  across tokenizations.
- **(c) training.** Matched epochs gave BPE ~3× fewer steps. Extending BPE-2048 174→300 (§7C; the
  endpoint ckpt `version_2/best-epoch=299-val_loss=4.7437.ckpt` WAS saved — `save_top_k=1` is
  per-version) had monitored **val_loss descend 5.57→4.74**, still descending at ep299. At the
  matched-steps endpoint the bits/atom gap shrinks **1.4×→~1.2–1.3×** and content rises
  0.187→0.226 / 0.147→0.190 / 0.141→0.182 (full eval) — gap real but closing. Counter-signal: the
  content/structural accuracy ratio FELL ep174→ep299, so the gain skews structural.

**De-confounded result (the decisive measures):**

| measure | gap | source |
|---|---|---|
| bits per canonical atom (A, full-eval, tokenization-invariant, ep174) | **~1.4×** (1.93→2.71) | `v4_audit_bits_per_atom.json` |
| same, at matched-steps endpoint (C, ep299) | **~1.2–1.3×** (ratio 1.24/1.31/1.21) | `v4_audit_ep299.json` |
| position-matched base-atom content argmax (B) | **~2–4×** (atoms-only 0.264 / 0.323 / 0.245) | `v4_audit_posmatched.json` |

**Direction survives, magnitude retracted:** BPE genuinely costs ~1.4× more bits per canonical atom
at ep174 and ~2–4× on position-matched content argmax — well above the ~5–9% confound-(a) bar — but
the 6–11× headline was confound-dominated, and confound (c) shrinks even the 1.4× to ~1.2–1.3× at
the matched-steps endpoint (still closing). All-tier val_acc and raw cross-tokenization content-tier are confounded; only A/B-style
measures are decision-grade.

**Operational conclusion:** atoms-only (tkvocab=0) stays the default; the "BPE dial is THE context
lever" framing is refuted — even de-confounded the dial buys no content worth its cost. Scoped ban:
*unconstrained cross-event-boundary merges*. An **event-boundary-respecting dictionary is now
PROMOTED to a live lever** (frontier §6) — the audit's confirmation that the harm is welding across
boundaries means a boundary-respecting variant is the untested upside, not deprioritized. Remaining
density: head-amortization (recoverable head 17.4%, §7F) + radix (live ~11–12% per-lane polish
lever, P1-scoped, §7E). Melody (NI_*): the §7D split was by interval size (large-interval/arpeggio
0.21–0.34 vs stepwise 0.11–0.15), **not** phrase position — the |Δ|≥5 bucket is 63–79% of NI_STEP
with median |Δ|=12 semitones (the arpeggio class). Both high-entropy; the claim does not narrow to
anchors; a true anchor split (first onset after rest / voice start) remains untested. Corpus: no
truncation (BlockMapper tiles all atoms); tunes ~50k atoms median / 85k mean (F survey over 862
tunes; A/B/C/D audits over the 856-dump eval corpus).

**Baseline note:** at full eval the atoms-only content baseline is **0.515 / 0.516 / 0.479**
(eval_a / daglish / follin, `v4_audit_posmatched.json` `atoms_content_overall`); the canonical
0.479 / 0.559 / 0.416 came from a noisy 24-block sample — "daglish beats in-distribution" does NOT
hold at full eval (the three subsets are ~level).

Audit artifacts in `data/audit/`: `v4_audit_bits_per_atom.json` (A), `v4_audit_posmatched.json`
(B), `v4_audit_ep299.json` + `v4_audit_ep174_fulleval.json` (C, matched-steps endpoints),
`v4_audit_nistep.json` (D), `deconfound_windows_and_digits_per_value.txt` (E/F),
`deconfound_summary.md` (full writeup).
