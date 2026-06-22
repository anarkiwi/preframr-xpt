# Melody/timbre factorization — a compact, contiguous melodic representation (track-major)

**Status: TRIAGE RAN (2026-06-17) — decisive GO.** Tool:
`preframr_experiments/audit/melody_factor_triage.py` (torch-free; reuses `learnability_triage`
metrics). It re-emits each tune melody-factored (a reordering shim over `encode`'s internal event
lists — no byte-exact decode needed, the model sees only tokens) and compares to the shipped
frame-major stream, windowed block-local at the model's context scale.

**Result (3 composers, seq_len 1024, n 51/74/80):** every signal moves the right way, far exceeding
voice-form's null — and the de-confounded reachability read (the ungameable decider) rises 27–59%.

| composer | induction-copy | **reach k=3 (de-confounded)** | h_inf bits/token | h_inf bits/frame | tokens/frame |
|---|---|---|---|---|---|
| Daglish | 0.804→0.904 | **0.292→0.465 (+59%)** | 1.279→0.930 | 9.76→6.94 | 7.63→7.46 |
| Hubbard | 0.775→0.872 | **0.240→0.356 (+49%)** | 1.458→1.220 | 10.37→8.39 | 7.11→6.88 |
| Follin  | 0.793→0.890 | **0.365→0.462 (+27%)** | 1.334→1.009 | 7.65→5.79 | 5.73→5.73 |

`reach k=3` = fraction of **non-trivial** melodic 3-note motifs (held-note runs collapsed out) whose
earlier copy is within one window — i.e. real `[A][A]` repeats becoming locally reachable. Its rise
proves the induction-copy lift is genuine structural locality, **not** trivial-adjacency inflation
from contiguity (the confound I flagged). `tokens/frame` is flat-to-down → the seq-length tradeoff did
NOT bite; the factored stream is no longer. **Verdict: GO** — the strongest learnability signal in the
arc, dead on the diagnosed failure (long-range melodic structure → local), where voice-form
(induction-copy flat 0.946) and instrument DEF→REF (de-confounded null) both missed.

Caveats (honest): training-free proxies *predicted* learnability before; the true arbiter is the GPU
A/B on de-confounded `copy_novel` novel-content. And **duration currently rides in the NOTE_ON timbre
body**, not the melody track — a v1 build should move duration into the melody unit for a truly compact
melodic token (pitch+duration together), which would strengthen the effect further.

---

**Original proposal (2026-06-17, operator-led).** The on-target reframe of the structural-abstraction
fix: instead of *referencing* recurring content in the existing frame-major stream (instrument
DEF→REF = within-noise null; phrase DEF→REF = modest/risky per `phrase_def_ref_triage.md`),
**change the stream ORDER** so all melodic information is colocated, contiguous, and free of
instrument writes — exactly what the symbolic-music-LLM literature converged on.

## The operator's question, and why it's right

> Other music LLMs all have a compact melodic representation free of instruments. Why can't we have a
> representation that compactly colocates all the melodic information and separates it from instruments?

We can — and our current codec is the *opposite* of the field's standard:

- **MMM (Multi-Track Music Machine, arXiv 2008.06048):** *"Unlike previous work that represents
  musical material as a single time-ordered sequence where musical events from different tracks are
  interleaved, MMM creates a time-ordered sequence of musical events for each track and concatenates
  several tracks."* Track-major; instrument declared once per track token.
- **REMI-z (track-aware):** notes grouped into per-instrument track sequences, each beginning with a
  single instrument token.
- **Compound Word / OctupleMIDI:** a note is a compact super-token / tuple; instrument is a **single
  field**, never re-emitted as a program.

The unifying principle: **a note is compact melodic info (pitch + duration + time); instrument is
declared ONCE (per track / as one attribute), never re-emitted inline; melodic continuity is
prioritized (track-major, not time-major interleave).**

Our v3 event stream violates all three: it is **frame-major** (`<DT> (<VOICE_v> <events>*)*`), the
onset instrument program (CTRL/AD/SR/PW) is **re-emitted at every note** (~38% of atoms), and a
voice's melodic line is **sliced across frames**, interleaved with its own timbre and the other
voices. So melodic *form* is scattered — which is exactly the failure the probe found.

## Why this is the right lever for the diagnosis (and differs from the two nulls)

The representation probe (`../generation/representation_abstraction_probe.md`): the model is a **strong
LOCAL abstractor, weak long-range structural** — a bag of local content bounded by ~1024 effective
context; induction-copy already ~0.94. An `[A][A]` melodic repeat that spans >1024 *interleaved*
atoms (melody + timbre + other voices) lands outside the effective window, so the strong local
abstractor never sees it as a local copy.

**Melody-contiguous, instrument-free ordering converts that long-range structure into LOCAL
structure:** pull timbre out and de-interleave, and the melodic line for a section collapses to a
fraction of the atoms, so `[A][A]` fits inside the local window the model is already strong at copying.
This is the learnability north-star (make structure local for a bounded local abstractor), achieved by
**reordering** — no new reference grammar, no segmentation, riding the already-strong induction-copy.

Why the prior negative results do **not** refute it:
- **Instrument DEF→REF (within-noise):** compressed timbre but kept the **frame-major interleave** —
  melody stayed scattered. Reordering is the missing variable.
- **Voice-form lane-demux (flat induction-copy 0.9457→0.9469, `lane_demux_hypothesis.md`):** separated
  *voices* but kept **melody+timbre interleaved within each voice lane** — never produced a pure
  melody stream. The untested variable is *removing timbre from the melody lane*.

This proposal is the lane-demux **content** axis in its cleanest form — and crucially, **melody vs
timbre is a HARDWARE-EXPLICIT partition** (freq regs vs CTRL/AD/SR/PW regs), needing **no latent role
inference** (the hard, lossy part that gated lane-demux's role-form). The encoder *already* computes
melody (`_freq_layer` → NI/FD) and timbre (`_note_layer`/bank, pw, globals) as **separate event
lists**; frame-major emission (`by_f`) is a final-step choice that re-interleaves them. The factored
representation is recoverable by changing that emission to track/content-major while keeping each
event's frame stamp for a deterministic merge-sort inverse.

## The representation

```
preamble : per-voice headers (TUNING, NOTE_TABLE, TICK) + instrument bank DEF        [v3 already has this]
MELODY   : per voice, all NI/FD events contiguous, frames delta-coded WITHIN the track
TIMBRE   : per voice, all onset programs contiguous (INSTR_REF <id> per onset, or inline tail)
GLOBAL   : filter/cutoff/res/routing/vol bus (its own track — filter is global, lane_demux constraint)
```

Decode = **merge-sort all tracks' events by (frame, canonical within-frame key)** → the exact
frame-major canonical write order → byte-exact. The within-frame total order already exists in
`encode` (`sorted(by_f[f], key=...)`), so the inverse is deterministic.

## Hard constraints (carried from `lane_demux_hypothesis.md` — still binding)

- **Sync/ring (ctrl bits 1/2):** voice N's audio ties to voice N−1's oscillator; keep the carrier↔
  modulator edge explicit, keyed by physical voice index — never split a modulator from its carrier
  into independent absolute-freq lanes. Melody-by-voice contiguity is fine as long as the edge stays.
- **Filter is global:** its own track, owner-keyed by the reg-23 routing bits.
- **Never drop a silent-but-modulating voice.**

## Honest priors / risks

- **Sobering prior:** every representation-reorder bet so far (voice-form, `sequence_order_normalization`,
  `voice_trajectory`) was flat or ~5%. This is the one axis untested (content-factored + timbre-free),
  hardware-explicit, and literature-backed — but "reorder helps" is not a safe prior. **Triage first.**
- **Sequence-length tradeoff:** frame-major shares one DT across a frame's events; track-major needs
  per-track frame stamps → more DT/frame atoms. Net atom count may RISE even as *melodic-form per
  window* rises. The relevant metric is melodic form per window, not raw length — measure both.
- Does **not** fix role-hopping (a melodic line crossing voices); that's a strictly-later refinement.

## De-risking triage (cheap, training-free — DO THIS BEFORE the byte-exact build)

A reordering shim over `encode`'s internal event lists (no byte-exact decode needed yet — the triage
needs only the token stream, exactly how voice-form was triaged). Then `audit/learnability_triage.py`
(seq_len 8192, window mode), comparing frame-major vs melody-factored:
1. **Induction-copy must RISE** (vs voice-form's flat 0.9457→0.9469) and **per-frame h_k drop** — the
   lane-demux gate, now on the timbre-free melody stream.
2. **Melodic form per window:** melody-atoms per 1024/8192-atom window — how much more melodic
   structure fits once timbre is pulled out.
3. **`[A][A]` locality:** fraction of corpus melodic repeats whose two copies now fall within one
   1024-atom window (vs frame-major) — the direct test of "long-range structure became local."

GO if induction-copy rises + more melodic form per window + more `[A][A]` fits locally → byte-exact
build (merge-sort inverse, `encode(verify=True)`) + GPU A/B on de-confounded `copy_novel`
novel-content + the structural probe. Flat like voice-form → decisive evidence the binding constraint
is corpus-inherent copy-dominance, not representation.

## Sources

MMM (arXiv 2008.06048); REMI / Compound Word / Octuple / REMI-z via MidiTok tokenization docs.
