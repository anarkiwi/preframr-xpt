# Representation-abstraction probe — local vs structural (the free-running root, refined)

**SUPERSEDED (2026-06-20):** the structural-locality diagnosis is addressed at the representation level
by the step/tracker codec (sparse, generator-level). Kept as the probe-method + finding record.

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
mostly-intact local content" is the mechanism — which *is* the bag-of-local reading).

## Replication + positive control (2026-06-17) — verdict holds

- **Replicated across composers + checkpoints.** Local ratio (random/transpose): Daglish 24.0
  (instrument_full), 23.6 (**atoms-only baseline ckpt**), Hubbard 29.4 — strong everywhere, so it's the
  model class, not the augmented model. Structural (block-reverse vs different-tune): Daglish 0.11–0.24,
  Hubbard 0.36–0.43 — always far below local, but **composer-dependent**, which exposes a confound:
  block-reverse also redistributes the *local* content each window sees (more for a varied composer), so
  it is not a clean structure-*only* probe.
- **K-sweep:** even K=32 (≈48-atom blocks, inside the window) moves the rep only ~0.25 of a
  different-tune distance — order-sensitivity rises weakly/gradually with finer disruption, never
  approaching "different music." Bag-of-local at all scales.
- **Positive control (the cleaner test): `[A][A]` literal-musical-repeat vs `[A][B]` novel, TF accuracy
  on the second segment by repeat distance (instrument_full, Daglish):** lift +0.10 @235 atoms → +0.10
  @351 → +0.08 @576 → +0.07 @978 → **+0.04 @1852**. The model **weakly** exploits a section-level repeat
  (a strong structural model would lift ~0.5 on a literal repeat), and the benefit **decays past ~1024**.
  Contrast `copy_novel` copyable-acc 0.535 (local n-gram induction is strong). So: **local copy strong,
  section-repeat exploitation weak and short-ranged** — converges with the representation probe.

**Verdict (multi-composer, multi-checkpoint, multi-method): strong local abstractor, weak long-range
structural abstractor.** Confirmed.

## Front-loaded-instrument hypothesis (operator, 2026-06-17) — the first explicit-structure experiment

Hypothesis: instruments are currently defined *just-in-time* (the onset program re-emitted inline at
every note); instead **define all instruments up front (a header / bank) and have the body reference
them by id** — how trackers actually author this music. This is a concrete DEF→REF instance of the fix
axis. Why it should help: (1) it matches the data's *true generative process* (composers used instrument
tables) — a strong learnability prior; (2) it's a clean DEF→REF the induction head handles natively
(define once, reference; the model's *strong local abstraction* grasps a reference token instead of
re-deriving a ~15-atom program it currently treats as long-range repeated content); (3) it **compresses
the body** — census (Daglish): **timbre = ~38% of the stream** (`FLD_CTRL/AD/SR/PW`), ~1153 onsets/tune
but few distinct instruments, so front-load+reference reclaims most of that 38% → ~1.5× more musical
*form* per ~1024-atom window, attacking the diagnosed bottleneck directly. Constrained decode enforces
define-then-reference (cannot reference an undefined instrument), exactly like the existing grammar mask.
**Honest scope:** this is the *instrument* half of the tracker structure; the *pattern* half (melodic
phrase references) is the complement and the harder part — instruments are the right FIRST step (biggest
immediate win, validates DEF→REF) but may not alone fix *melodic* form. The ~2% onset-program variation
needs a per-use residual/override (or accept small loss). It is a real tokens-side codec redesign with
the byte-exact round-trip invariant, aligned with the existing instrument-bank design
(`transplant_augmentation_design.md` P0). **Triage before the build:** confirm onset-program recurrence
exactness (how lossy pure-reference is); `learnability_triage` on a sample tune encoded reference-style
vs inline (does induction-copy rise / h_k drop?). Then the encode + train A/B on `free_running_gap` + the
structural probe + the `[A][A]` repeat lift.
