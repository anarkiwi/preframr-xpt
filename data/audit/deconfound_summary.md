# §7 de-confounding audit — results (2026-06-12)

Stack tokens 0.50.0 / preframr 0.2.29; canonical generalize 14M body.
atoms-only ckpt /scratch/tmp/v3c_final.ckpt (ep99); BPE-2048 best ckpt ep174 (val_loss 5.5728).

## A. bits per canonical atom (DECISIVE, full eval, GPU) — v4_audit_bits_per_atom.json
| subset | ATOMS | BPE (ep174) | ratio |
|---|---|---|---|
| eval_a | 1.931 | 2.713 | 1.40x |
| eval_b_daglish | 2.001 | 2.972 | 1.49x |
| eval_b_follin | 2.221 | 3.039 | 1.37x |
=> BPE ~1.4x worse in bits/atom, NOT 6-11x. The argmax headline was confound-dominated.

## B. position-matched argmax (full eval) — v4_audit_posmatched.json
atoms-only content at BPE-base positions: 0.264 / 0.323 / 0.245 (overall 0.515/0.516/0.479).
BPE base-atom content ~0.05-0.13. => atoms-only ~2-4x BPE at matched positions; confound (a) real
(base positions drag atoms-only 0.51->0.26) but does not explain the gap (atoms-only still wins).

## C. matched-steps extension — v4_audit_ep{100,174}.json + trajectory
content-tier trajectory: ep100 0.049/0.088/0.039 -> ep174 0.077/0.126/0.075 (roughly doubled).
Trained 174->300: val_loss descended 5.57 -> 4.74 (STILL improving, no plateau). Run only retained
ep174 (save_top_k=1 on val_loss), so exact ep300 content not captured. => confound (c) REAL and
unbounded at ep300; matched-steps gap <=1.4x (A's ep174), likely smaller. Pin with a save_last re-run.

## D. NI_STEP split (full eval) — v4_audit_nistep.json
phrase-initial (|dnote|>=5, median ~12): 0.21/0.34/0.33 ; within-phrase: 0.15/0.15/0.11.
Both LOW, anchors slightly higher (not "anchors drag"). => keep high-entropy melody read, regime-conditioned.

## E. digits-per-value (corpus) — deconfound_windows_and_digits_per_value.txt
mean 1.364 digits/value (1-digit 66.6%, 2-digit 30.5%, 3-digit 2.9%). Byte-pack (saved=d-ceil(d/2))
saves 8.79M atoms = 11.9% of stream (24% of varint content). Per-KIND juice: FD_STEP 1.62, PW_STEP 1.53,
PW_RAMP 1.52, FD_RAMP 1.49, NI_STEP 1.46; FLD_CTRL/FLD_SR dead (1.001). => radix POLISH-GRADE ~12%
lever (not dead), P1-scoped to multi-digit varint lanes, stackable with head-amortization (~1.3x density).

## F. full-corpus survey (862 tunes)
atoms/tune mean 85k median 50k; BPE tokens mean 31k median 18k (= atoms/2.73x). BlockMapper TILES all
blocks -> NO truncation (model trains on every atom). AGENTS.md "4.16 windows/tune" = MEAN BPE windows
(31k/8192), not atoms. Composition: content 69.0% (varint 50.1 + nibble 18.9), head 28.5% (KIND 15.9 /
VOICE 11.1 / reg 1.5); recoverable head = KIND+reg = 17.4% (VOICE ~0.7/event already amortized).
