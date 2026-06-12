# Unigram BPE (tkvocab>0) harms content generalization — REFUTED as the context lever

**2026-06-12.** The canonical learnability run (`generalize`, 14M body) compared atoms-only
(tkvocab=0) vs unigram BPE (tkvocab=2048), same corpus/holdouts.

**Result (content-tier accuracy, maturity-matched ~epoch 100):**

| eval subset | BPE-2048 | atoms-only |
|---|---|---|
| eval_a | 0.049 | 0.479 |
| eval_b_daglish | 0.088 | 0.559 |
| eval_b_follin | 0.039 | 0.416 |

BPE is **~6–11× worse** on content, including held-out composers, at matched maturity (v4 kept
learning: val_loss 7.08→6.27). Not a training-maturity artifact.

**Mechanism (localized):** merged BPE tokens are ~1% predictable (base atoms 4–8%); BPE welds
content atoms into multi-atom merges across event boundaries, which are not predictable, and the
welding degrades the surviving base atoms too. All-tier val_acc is confounded across tokenizations
(bigger vocab → higher per-token entropy); content-tier is the verdict.

**Conclusion:** unigram/BPE is the wrong lever — it trades content learnability for sequence
compression. The "BPE dial is THE context lever" framing is refuted. The encoding-density levers
(parametric ramps, per-voice note-table pitch) are already shipped (tokens 0.16/0.17, 0.47.0);
remaining density is structural (head overhead), not value-encoding. Melody (NI_STEP) is intrinsically
high-entropy next-token (score by audition, not argmax) — a known pitch-model property, not a gap.

Audit artifacts: `/scratch/tmp/v4_audit*.json`, per-KIND map in session log.
