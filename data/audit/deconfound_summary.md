# §7 de-confounding audit — results (2026-06-12)

Stack tokens 0.50.0 / preframr 0.2.29; canonical generalize 14M body.
atoms-only ckpt /scratch/tmp/v3c_final.ckpt (ep99); BPE-2048 ckpts: ep174 (version_0,
val_loss 5.5728) and the matched-steps endpoint ep299 (version_2, val_loss 4.7437).
[Amended 2026-06-12 post-merge: C/D corrected, digit mean fixed, ep299 endpoint audited — see
the de-confounding-audit-followup commit.]

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

## C. matched-steps extension — v4_audit_ep299.json + v4_audit_ep174_fulleval.json
CORRECTION (the earlier writeup wrongly said "only ep174 retained, save_last re-run needed"): the
174->300 extension's endpoint ckpt EXISTS at version_2/best-epoch=299-val_loss=4.7437.ckpt
(save_top_k=1 is PER-VERSION; version_2 saved its own best). Monitored val_loss (NOT train loss)
descended 5.5728 (ep174) -> 4.7437 (ep299), STILL descending at ep299 (last 12 evals monotonic
4.787->4.744, no plateau). Full-eval audit at both endpoints:
  bits/atom BPE: ep174 2.762/3.009/3.084 -> ep299 2.390/2.616/2.681
  ratio to atoms-only (1.931/2.001/2.221): ep174 1.43/1.50/1.39 -> ep299 1.24/1.31/1.21
  content-tier (full eval): ep174 0.187/0.147/0.141 -> ep299 0.226/0.190/0.182
(my ep174 full re-run reads 2.762/3.009/3.084 vs the committed §7A bits_per_atom.json 2.713/2.972/
3.039 -- ~2% run/script variance, same ckpt/block-size; both ~1.4x. The decisive shrinkage uses the
self-consistent ep174_full<->ep299 pair, same script.)
=> confound (c) REAL; matched-steps gap ~1.2-1.3x and still closing. NO save_last re-run needed.
COUNTER-SIGNAL: content/structural ratio FELL ep174->ep299 (eval_a 1.70->1.20) -- extended training
favored structure over content, so the content gain partly reflects better structural prediction.
(24-block per-class trajectory, for context: ep64 0.043/0.048/0.031 -> ep99 0.049/0.088/0.039 ->
ep174 0.077/0.126/0.075; content/struct ratio fell ep100->ep174 0.61->0.55 / 1.80->1.29 / 1.60->0.64.)

## D. NI_STEP split (full eval) — v4_audit_nistep.json
CORRECTION: the split is by INTERVAL SIZE (|dnote|>=5 vs stepwise), NOT phrase position. The
|dnote|>=5 bucket is 63-79% of NI_STEP with median |d|=12 semitones = the SID ARPEGGIO / large-
interval class, not phrase onsets. large-interval(arpeggio) 0.21/0.34/0.33 ; stepwise 0.15/0.15/0.11.
Both high-entropy => keep high-entropy melody read; claim does NOT narrow to anchors. A true anchor
split (first onset after rest / voice start) remains untested.

## E. digits-per-value (corpus) — deconfound_windows_and_digits_per_value.txt
mean 1.351 digits/value (36755978 digits / 27196187 values). Byte-pack (saved=d-ceil(d/2))
saves 8.79M atoms = 11.9% of stream (24% of varint content). Per-KIND juice: FD_STEP 1.62, PW_STEP 1.53,
PW_RAMP 1.52, FD_RAMP 1.49, NI_STEP 1.46; FLD_CTRL/FLD_SR dead (1.001). => radix POLISH-GRADE ~12%
lever (not dead), P1-scoped to multi-digit varint lanes, stackable with head-amortization (~1.3x density).

## F. full-corpus survey (862 tunes)
atoms/tune mean 85k median 50k; BPE tokens mean 31k median 18k (= atoms/2.73x). BlockMapper TILES all
blocks -> NO truncation (model trains on every atom). AGENTS.md "4.16 windows/tune" = MEAN BPE windows
(31k/8192), not atoms. Composition: content 69.0% (varint 50.1 + nibble 18.9), head 28.5% (KIND 15.9 /
VOICE 11.1 / reg 1.5); recoverable head = KIND+reg = 17.4% (VOICE ~0.7/event already amortized).
