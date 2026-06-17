# Representation-abstraction probe — local vs structural (the free-running root, refined)

**Status: RAN (2026-06-17).** Training-free probe of the trained model's *representation* (not its
behaviour), prompted by the question "does the model fail to generate because it can't *see what is
genuinely similar across tracks* — an abstraction failure?" Answer: **yes, but specifically a long-range
STRUCTURAL abstraction failure — local abstraction is strong.** This unifies the free-running findings
and re-points the fix axis. Script: `/scratch/tmp/abstraction_probe.py` (instrument_full ckpt; embed =
centered mean-pooled final-layer hidden state, hook on `model.model.norm`; cosine distance).

## Method

For real passages (first ~1536 atoms), apply matched-surface transforms in the dump domain, re-encode,
and measure how far the model's representation moves (centred — the raw embedding is dominated by a
common direction, norm 17.5 vs per-passage std 0.025, so uncentred cosine is ~0 for everything and
useless):
- **transpose +5 semitones** — PRESERVES melody+form, changes ~92% of atoms.
- **voice-permute** — PRESERVES music (rotates SID voices).
- **block-reverse** — PRESERVES local content, DESTROYS form (reverse the order of 4 contiguous
  frame-segments), changes ~86% of atoms.
- **random ±5 per note** — DESTROYS local content (intervals), matched ~94% surface change.
- **different passage** — baseline (everything different).

## Result (n=10, Daglish_Ben, matched surface ~0.9)

| transform | preserves | rep-dist |
|---|---|---|
| transpose +5 | local + form | **0.035** |
| voice-permute | music | 0.064 |
| **block-reverse** | **local only (form destroyed)** | **0.144** |
| random ±5 | nothing (local destroyed) | **0.861** |
| different passage | — | 1.080 |

- **Local abstraction STRONG:** random/transpose = **25.7** — destroying local content moves the rep 26×
  more than a matched music-preserving change; the model is transposition- and voice-invariant and
  interval-sensitive. It is **not** a surface-token memoriser. (The simple "can't see local similarity"
  hypothesis is refuted.)
- **Structural abstraction WEAK:** block-reverse moves the rep only **0.144** — 4× the invariant floor
  but **6× less** than the matched local-content edit, and only **13%** of a different tune. Reorder the
  sections and the model mostly shrugs → the representation is **largely a bag of local content**; it
  weakly encodes long-range order/form.

## Why this is the unifying diagnosis

- **effective context ≈ 1024 atoms** (`effective_context_audit`) — the model can't span a section
  (sections > its window); bag-of-local is the *consequence*.
- **copy_novel novel-token 0.194** — the right novel token depends on position-in-form; with no form
  representation, only local continuation is possible.
- **free-running drone** — generation needs a long-range plan (repeat/answer/resolve); no structural
  scaffold → wander → collapse.
- **Tier-3 augmentation +26% local TF but flat free-running** — it improved *local* invariance, never
  *structure*. Same for the (triaged-out) lane-demux/role tweaks.

So the binding constraint is **long-range structural representation**, not training dynamics
(Tier-4/DAgger, exhausted) and not local-invariance data (Tier-3, flat). The model sees what's
genuinely similar across tracks *locally*; it cannot hold the *form*.

## Fix axis this opens (project-aligned, genuinely new)

Make long-range structure **local / explicit** so the strong local abstractor can grasp it:
**repetition references (DEF→REF for a recurring section), phrase/section boundary tokens, form labels.**
A repeated section becomes a short reference token instead of a >1024-atom span the window can't hold.
This is exactly the learnability north-star (`induction-head DEF→REF > implicit counters; make structure
explicit`) and a different axis from everything exhausted. Note: just raising `seq_len` won't help — the
model already fails to use context past ~1024 (a learnability limit, not a window-size limit).

## Caveats / next

n=10, one composer, one checkpoint; block-reverse keeps long contiguous segments (so "model sees
mostly-intact local content" is the mechanism — which *is* the bag-of-local reading). Before betting:
(1) replicate on 2–3 more composers + the atoms-only baseline ckpt; (2) finer-grained shuffles
(smaller blocks) to map at what scale order-sensitivity drops off; (3) a positive control — does the rep
detect a *genuine* internal repeat (`[A][A]` vs `[A][B]`)? Then scope the explicit-structure encoding
experiment.
