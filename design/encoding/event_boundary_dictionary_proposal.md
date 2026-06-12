# Event-boundary-respecting dictionary — the promoted §6 lever, specified

**Status: PROPOSAL (2026-06-12; triage-gated).** Implements the lever
`encoding_density_frontier.md` §6 promoted after the §7 de-confounding audit. One sentence: train
the existing unigram BPE-as-dictionary with **merges constrained to never cross event boundaries**,
because the audit showed (a) unconstrained BPE's true cost is only **~1.2–1.3× bits/canonical-atom
at matched steps** (ep299, still closing — not 6–11×), and (b) the harm mechanism is **welding
content across event boundaries** — remove the mechanism, keep most of the compression. This is the
**only encoding-side lever left with context-scale upside** (~2× predicted at ≈parity quality); the
deterministic packs (§3 radix ~12%, §4 head-amortization ~10–15%, stack ~1.3×) are its fallback.
**Unconstrained merges stay banned** (frontier §6); this doc does not reopen them.

## Why it should work (evidence, not hope)

- **The cost ceiling is known.** Even merges that freely weld across boundaries cost only
  1.24/1.31/1.21× bits/atom at the matched-steps endpoint (`data/audit/v4_audit_ep299.json`). A
  variant that cannot weld should land below that — plausibly ≈parity.
- **The gain floor is known.** Unconstrained 2048-vocab compresses 2.73×. Within-event merging
  keeps every merge that matters for the modal stream: 67% of events are 2–4 atoms (modal 3 =
  kind + ~2 payload), so head+payload → 1–2 tokens covers most events; only the long ramp-delta
  tails stay atom-ish. Realistic estimate **~1.8–2.5×**.
- **Context stakes** (frontier §7F): median tune 50k atoms ≈ 6 windows at seq_len 8192. At ~2×
  dictionary compression the median tune is ~25k tokens ≈ 3 windows; **combined with `seq_len`
  16384 the median tune fits 1–2 windows**, and the Orin PROMPT=2048 carries ~2× more music.
  Neither deterministic pack can reach this scale (~1.3× combined).
- **No codec change.** The dictionary sits *on top of* the frozen 127-atom alphabet (groupings of
  atoms; decode = ungroup). Fidelity contract, `EVENT_FORMAT_VERSION`, `.atoms.zst` caches all
  untouched — unlike the fallback packs, which bump both versions and re-encode the corpus.

## Mechanism (the actual change)

1. **Training corpus segmentation:** emit the unigram-training stream as **per-event words** (the
   encoder knows event spans — `stream.encode` produces them; insert a separator the pre-tokenizer
   splits on, or feed per-event atom lists). The `tokenizers` UnigramTrainer then cannot form a
   merge spanning two events. Same `RegTokenizer` + BPE-as-dictionary machinery, same `tkvocab`
   dial, same dataset/KEYFRAME windowing.
2. **Encode path:** apply the same per-event segmentation at encode time (pre-tokenizer split),
   so runtime tokenization matches training. Decode is unchanged (dictionary id → atom ids).
3. **Split of work:** tokens-side = expose per-event spans in the training-corpus emission +
   encode segmentation; framework-side = feed event-bounded words to the trainer. Sketch-level;
   exact seam belongs to the implementer with both repos open.
4. **Operational bonus:** per-event words shrink the unigram lattice from 35K–150K-atom sentences
   to ≤~100-atom words — this likely **removes the UnigramTrainer SIGSEGV class entirely** (the
   `RUST_MIN_STACK=2GB` workaround exists only because of long-sentence lattice drops).

## Experiment (canonical discipline; every §7 lesson applied)

1. **Static triage first (minutes, fogbank):** `learnability_triage` on boundary-respecting merged
   streams at `tkvocab` {1024, 2048, 4096}: compression ratio, frames-per-window, h_k,
   induction-copy, live-vocab %. **Kill here if compression < 1.5× at every vocab.**
2. **Canonical A/B (one run):** `generalize` 14M, same corpus/holdouts; boundary-respecting
   tkvocab=2048 (or the triage knee) vs the done atoms-only baseline. **Matched steps by
   construction:** set `max_epochs ≈ 100 × measured compression` (early-stop never fires under
   schedule-free — AGENTS.md gotcha); `save_top_k` is per-Lightning-version — capture endpoints
   deliberately.
3. **Decision metrics — bits/canonical-atom and position-matched argmax ONLY** (frontier §1a:
   raw cross-tokenization argmax is confounded by population and granularity; per-token accuracy
   will read lower at quality parity *by construction*). Promote the §7A bits/canonical-atom
   computation from session script to a permanent `preframr_experiments/audit/` tool as part of
   this work — it is now the standing decision metric for any tokenization comparison.
4. **Merge-table inspection:** classify learned merges (within-value digit pairs / head `(kind,reg)`
   pairs / payload runs / mixed). If the table is dominated by the first two classes, the
   deterministic packs capture most of the win without dictionary training — that feeds the
   PARTIAL gate below.

## Decision gates

- **ADOPT** — bits/canonical-atom ≤ atoms-only × ~1.05 at matched steps AND compression ≥ 1.8×:
  make the boundary-respecting dictionary the default tokenization for the context arc; re-run
  frontier §5's window arithmetic (median tune then fits 1–2 windows at 16384).
- **PARTIAL** — parity only at 1.3–1.8× compression: weigh against the deterministic packs (same
  density without dictionary infra, but codec bump + corpus re-encode); prefer whichever the
  merge-table inspection says captures the gain more cheaply.
- **REJECT** — bits/atom cost > ~1.1× at every vocab: fall back to the deterministic packs —
  (1) per-lane digit byte-pack (~12%: FD_STEP 1.62, PW_STEP 1.53, PW_RAMP 1.52, FD_RAMP 1.49,
  NI_STEP 1.46 digits/value; single-nibble lanes out of scope per P1), (2) head-amortization
  (`(kind,reg)` combos + grammar kind-elision, ceiling 17.4%). Both: byte-exact
  `encode(verify=True)`, bump `EVENT_FORMAT_VERSION` + `ATOM_CACHE_VERSION`, compare pre/post in
  bits/canonical-atom + content-tier only.
- Any outcome: frontier §5 stands — context remains a windowing/chaining problem; this lever only
  changes how much music one window buys.

## Sequencing note (interacts with AGENTS.md NEXT #1)

Run the triage **before** the `seq_len` 16384 dataset re-cut: a winning dictionary changes the
window math (16384 *dictionary tokens* ≈ a whole median tune), and re-cutting twice wastes the
wallclock the atom cache was built to save.

## Risks / watch

- Encode-time per-event segmentation throughput (the thread-parallel block pass exists; measure
  against the ~33 min full-corpus encode anchor).
- Rare-merge sparsity at higher vocab (unconstrained 2048 ran ~98% live — watch it stay there).
- Don't re-propose the measured-and-rejected tokens items (§8.4 joint freq/note DP, §2.7
  mixed-radix ORDER-DT, DT-in-ticks, POLY degree cap, mid-note R-only NOTE_ON fold).
- Melody stays high-entropy (stepwise NI 0.11–0.15, frontier §2/§7D) — no dictionary fixes that;
  score by audition/distribution, out of scope here.
