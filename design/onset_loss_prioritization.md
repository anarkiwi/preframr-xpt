# Onset loss prioritization — force capacity onto the rare melodic onset

**Status:** Design, ready to implement (framework-only; preframr). Opt-in `--onset-loss-weight W`
(default 1.0 = off). Pairs with the interval/anchor onset encoding. Cross-ref:
`trajectory_anchoring.md` (the converged diagnosis) + `freq_v0_interval.md`.

## Problem (converged diagnosis, 2026-05-28)

Across **four** mini conditions (full_macros / anchored / anchored+interval / freq_core) the
model's **FREQ V0-onset acc is 0.000**, while:
- the onset line is **trigram-0.79–0.82 predictable** (not aleatoric),
- removing timbral dilution (freq_core) lifts **SET content 0.078→0.419** — so the model has
  ample mini capacity and *does* learn whatever token dominates,
- representation fixes (anchor, interval) did not move the onset.

So the onset is **rare and the model isn't compelled to spend capacity on it**: it maximises
mean CE by nailing the dominant SET/DELTA tokens and ignoring the sparse onset (~1–2% of all
tokens). The one thing that has ever moved it off 0 is **scale** (prodlike absolute op45 0.067).

## The lever

Up-weight the **FREQ V0-onset** token classes in the per-token CE loss so the rare onset
contributes comparably to the dominant tokens. Onset vids = decode(vid)'s first base atom has
`op == FREQ_TRAJ_OP (45)`, `reg ∈ FREQ_TRAJ_REGS (0,7,14)`, `subreg ∈ {V0_HI(1), V0_LO(2)}`.

This rides the **existing** per-vocab-id loss-weight path
(`lightning.py`: `loss = (per_tok * vocab_frame_weight[y] * vocab_class_weight[y]).sum()/…`) —
add a third factor `vocab_onset_weight[y]`.

**Distinct from the refuted `weighted_token_loss`:** that was generic frequency/audio-frame
weighting over all classes; this is a **single, semantically-targeted class** (the melodic
onset), on the **good (anchored+interval) onset encoding**, motivated by the precise
rare-and-ignored diagnosis. The A/B settles whether the distinction matters.

## Implementation (framework only)

- **`preframr/args.py`**: `--onset-loss-weight` (float, default 1.0). Opt-in, not in
  `_PIPELINE_NAME_TO_FLAG` (a loss knob, not a parse transform).
- **`preframr/train/model/tier_map.py`**: `_build_vocab_onset_weight(args, n_vocab, tokens,
  tkmodel)` → float tensor, `W` on freq-onset vids else `1.0`; returns ones when `W == 1.0`
  (skip work) or `tokens` empty. Mirrors `_build_vocab_class_weight`: `RegTokenizer(args,
  tokens).load(tkmodel, tokens)`, then per vid `rt.decode([vid])` → first base atom →
  `tokens.iloc[bid]` op/reg/subreg test (constants from `preframr_tokens.stfconstants`:
  `FREQ_TRAJ_OP`, `FREQ_TRAJ_REGS`, `FT_SUBREG_V0_HI`, `FT_SUBREG_V0_LO`).
- **`preframr/train/model/lightning.py`**: register `vocab_onset_weight` buffer (like
  `vocab_frame_weight`); multiply into `weights` in the loss. Default W=1.0 ⇒ ones ⇒
  byte-identical loss to today.
- **Test**: `_build_vocab_onset_weight` returns W on a synthetic freq-onset vid and 1.0
  elsewhere; W=1.0 ⇒ all ones.

`W` is a knob (start 10; sweepable). Too low = no effect; too high = destabilises (model
over-focuses onset, structural/content regress) — watch all-tier val_acc doesn't collapse.

## A/B + gate

`onset_loss_weight_mini`: base = full_macros + anchor + interval (the good onset encoding,
`:0.2.8`); target arm `--onset-loss-weight 10`, baseline W=1. 3 seeds, mini body=large.
**Decisive gate:** `content_tier_report --onset` — does **V0-onset acc move off 0.000**
(and does all-tier not collapse)? A clear move at mini → sweep W + fold into a prodlike
interval+onset-weight run. Flat at W=10 → try higher / conclude the onset needs scale.
