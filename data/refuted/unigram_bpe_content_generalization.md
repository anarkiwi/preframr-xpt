# Unigram BPE (tkvocab>0) harms content generalization — REFUTED as the context lever (magnitude provisional)

**2026-06-12 (evidence re-scoped same day after a confound review — see
`design/encoding/encoding_density_frontier.md` §1a/§7/§8).** The canonical learnability run
(`generalize`, 14M body) compared atoms-only (tkvocab=0) vs unigram BPE (tkvocab=2048, ~2.6×
stream compression), same corpus/holdouts.

**Result (content-tier accuracy, matched ~epoch 100):**

| eval subset | BPE-2048 | atoms-only |
|---|---|---|
| eval_a | 0.049 | 0.479 |
| eval_b_daglish | 0.088 | 0.559 |
| eval_b_follin | 0.039 | 0.416 |

BPE is **~6–11× worse on content as measured** — but the magnitude is **PROVISIONAL**: (a) the BPE
column scores only *surviving base atoms* (the rare tail unigram didn't merge) vs ALL content atoms
for the baseline — different populations by construction; (b) merged-token argmax is joint over k
atoms (parity for a 2–3-atom merge ≈ 0.11–0.23, not 0.48) — per-token argmax does not compare
across tokenizations; (c) matched epochs gave BPE ~3× fewer optimizer steps and v4 stopped
mid-descent (val_loss 7.08→6.27) with the plateau→steep-drop transition unexcluded. The frontier §7
de-confounding audit (bits/canonical-atom, position-matched scoring, matched-steps extension)
hardens or revises this entry per the §8 falsifier mapping.

**Mechanism (localized; direction plausibly real):** merged BPE tokens ~1% argmax-predictable —
below even the parity-expected ~0.11–0.23 — consistent with BPE welding content atoms into merges
across event boundaries. All-tier val_acc is confounded across tokenizations (bigger vocab →
higher per-token entropy); per the above, content-tier is too.

**Operational conclusion (stands regardless of magnitude):** atoms-only (tkvocab=0) is the default;
the "BPE dial is THE context lever" framing is refuted — even at per-atom parity the dial buys no
content, and the welding mechanism scales WITH vocab (closing the planned vocab sweep). Scoped ban:
*unconstrained cross-event-boundary merges*; an event-boundary-respecting dictionary is untested and
deprioritized, not banned (frontier §6). Remaining density is structural (head-amortization
~10–15%); the context levers are seq_len/windowing + register-domain chaining. Melody (NI_*, the
interval lane) is high-entropy at this regime — anchor-vs-step split pending (frontier §2/§7D);
score by audition/distribution, not argmax.

Audit artifacts: `/scratch/tmp/v4_audit*.json`, per-KIND map in session log — **ephemeral until
copied into `data/audit/` (frontier §7G)**.
