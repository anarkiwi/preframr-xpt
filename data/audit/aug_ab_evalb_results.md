# Augmentation dosage A/B — results (generalize_aug_ab, 2026-06-16)

Atoms-only baseline vs reduce/instrument write-domain augmentation (preframr-aug v0.1.0).
All metrics on the **same held-out eval-B blocks** (Daglish+Follin), n-blocks 24 (copy_novel,
n≈7908 content tokens) / 8 (free_running_gap, n=1464). Run root
`/scratch/tmp/preframr_experiments/aug_ab_v1`.

| arm | train Δ | val_acc | copy_novel novel-content | free-run content acc |
|---|---|---|---|---|
| baseline | — | 0.552 | 0.152 | **0.062** |
| reduce_25 | +197 | 0.596 | 0.151 | 0.062 |
| reduce_full | +749 | 0.641 | 0.162 | 0.047 |
| instrument_25 | +197 | 0.579 | 0.167 | 0.068 |
| instrument_full | +276 | 0.593 | **0.192** | 0.062 |

## Verdict: Tier-3 augmentation does NOT fix free-running (escalate to Tier-4)

- **Teacher-forced gains are real and dose-dependent**, strongest for **instrument transplant**:
  copy_novel novel-content 0.152→**0.192** (+26%), TF acc 0.416→**0.520**. Supports the cross-composer
  melody×timbre debinding hypothesis *for teacher-forced generalization*.
- **But free-running content accuracy is FLAT (~0.05–0.07) across all arms and doses** — no arm beats
  the baseline 0.062 floor (spread within ~1–3 SE of 0.06). The best teacher-forced arm
  (instrument_full) shows **zero** free-run gain. The free-running gap *widened* (TF rose, free-run did
  not).
- This is the textbook **exposure-bias (M1)** signature: the model learns the data distribution better
  but the train↔inference mismatch persists. **Changing the corpus (Tier-3, attacks M4 copy-dominance)
  improves teacher-forced metrics but does not cure free-running collapse.** Per the remediation ladder:
  no free-run lift at any dose → **escalate to Tier-4 DAgger** (model-side, re-canonicalization oracle).
- Caveat: free_running_gap n-blocks=8 (n=1464) is a modest sample, but the signal is robust — free-run
  flat while TF varies by +0.10, and the strongest-TF arm gains nothing free-running.
- Secondary value: instrument augmentation's teacher-forced eval-B gain may still be worth keeping for
  general quality/recombination; it is just not the free-running fix.
