# augmentation — design note

**Status (2026-05-24):** melody-transfer prototype landed at
`integration_tests/profile/augment_melody_transfer.py` (Phase 0 smoke
pending); voice-permutation drafted. New: a third, cheaper family —
**verified-inaudible macro perturbation** — is unlocked by the current
recognized-macro vocabulary (`preframr-tokens>=0.14.1`) + the
`preframr-audio` fidelity tooling (`compare_renders` / `per_frame_rel_rms`).

**Update (2026-05-25 — FREQ_TRAJ re-frame + result-gated deployment):** the
tokenizer rework (preframr-tokens 0.16→0.18, unified `FREQ_TRAJ` op 45) landed
after this doc and changes both the mechanics and the timing.

*Op vocabulary.* The separate pitch-macro ops named below — `SLOPE_FREQ_*`,
`FREQ_RUN`, `FREQ_VIBRATO`, `FREQ_NUDGE` — are now folded into one `FREQ_TRAJ`
op (SUBTYPE MONOTONE_RAMP/OSCILLATE/RUN; nudge → 2-atom delta). Wherever the
mechanism sections list those as the pitch stream, read `FREQ_TRAJ (+subtypes)`.
Floor is now `preframr-tokens>=0.18.0`; the inaudibility gate is the same
per-frame oracle (`test_full_pipeline_fidelity.py`, byte-exact zero/structural/mid)
+ `compare_renders`. **The full voice-register partition table and "Five things
that bite" #5 need an op-name pass against the 0.18.0 op set at implementation —
do not trust the pre-rework op names there verbatim** (whether PW/FC slopes are
inside FREQ_TRAJ or still separate must be confirmed against
`unified_oscillation_primitive_design.md`).

*Inaudible-perturbation surface shrinks.* The stricter, more canonical tokenizer
has ALREADY absorbed much non-audible variation (content-tier binning collapses
sub-bin differences to a single token), so perturbing a binned content-tier delta
yields the *same* token stream — no augmentation. Scope this family to the
dimensions the tokenizer represents EXPLICITLY and non-degenerately: FREQ_TRAJ
subtype/endpoint choice, PRESET grid neighbors, ADSR/PWM within JND, op
segmentation (CTRL_BIGRAM↔CTRL_TRIPLE). Smaller surface than the 2026-05-24
framing assumed, but every admitted perturbation is now *verified* inaudible.

*Timing — which family is actionable depends on the re-arc result.* The three
families attack different failure modes; the post-rework re-arc + content-tier
per_class audit diagnoses which (if any) to deploy. **Drafted-but-dormant until
then — do not deploy augmentation as a substitute for an unsolved generalization
approach.**

| Re-arc / audit signature | Diagnosis | Lever |
|---|---|---|
| Learns well, but held-out (eval_a/eval_b) plateaus below train and the gap widens with scale | data/diversity-limited | **Melody transfer** — distribution expansion (~30× novel content) |
| High train, brittle to representation — sensitive to voice assignment or exact tokenization | invariance-overfit | **Voice permutation + inaudible perturbation** — content-preserving regularizers (encode the right generalization prior directly) |
| Content-tier generalization fails outright (approach-limited, as with the 5 refuted model-side bets) | NOT data-limited | none — augmentation amplifies a broken approach; fix the approach first |

**Phase-0 viability probe of the inaudible family (2026-05-25,
`integration_tests/profile/augment_inaudible_probe.py`, 8 dumps, perturb raw
voice regs → re-parse → render → `compare_renders`):** the family is **viable
but knob-dependent**, and crucially **inaudible ⟹ token-diverse held in every
cell** (no content-tier canonicalization collapse for these raw-reg knobs — the
inaudible surface is actually usable). Per-knob usable yield (inaudible & token-
diverse, out of 8): **PW_LO 7/8 @±2, 6/8 @±8** (best knob); **FC_HI 4/8 @±2,
2/8 @±32** (robust, moderate); **FREQ_LO 5/8 @±2 → 1/8 @±32** (pitch-sensitive,
small range only); **AD+SR 0/8 at every magnitude — envelope perturbation is
always audible, ruled OUT as a knob**. So the inaudible family has real surface
(PW + FC, small magnitudes) but it is a content-preserving *regularizer*, modest
multiplier — confirms the "invariance-overfit" row, not a data-volume fix.
**BLOCKER found:** preframr-audio 0.5.0 `fidelity._irq_from_df` imports
`preframr_tokens.reglog_helpers`, dropped in tokens 0.18.0 (`read_initial_irq`
moved to `reglogparser`) — so `dfs_render_equivalent` / the round-trip audio
gate are broken in the current image; the probe bypasses via the lower-level
render. One-line fix in preframr-audio + release/rebake needed before the family
(or the queued round-trip gate) can use the library helper.

**Phase-0 audit of melody transfer LANDED + PASSED (2026-05-25,
`integration_tests/profile/audit_melody_transfer.py`):** splice → parse →
render over 10 donor/host pairs (truncated): **parser-admitted 10/10, audible
10/10, not DC-pinned 10/10** — clears the design's Phase-0 gate (parser admits
~100%, audio non-silent), so the splice produces usable end-to-end output. This
validates the mechanism only; the musical-coherence checks (gate-count vs donor,
spectral-centroid in host range, the pitch-range / FC-tracks-gate eligibility
filters from "Five things that bite") are the phase-1 refinement. (Audit uses
the lower-level render to dodge the `_irq_from_df` blocker above.) Net: both
families now have working viability probes — inaudible = content-preserving
regularizer (modest), melody transfer = distribution expansion (Phase-0 clean).

**New eligibility filter — donor↔host ADSR similarity (validated 2026-05-25):**
WAV audition surfaced some splices rendering near-silent ("quiet"). Mechanism:
the splice uses the HOST's ADSR with the DONOR's gate timing, so a donor whose
note rhythm was written for a very different envelope than the host's gets
mangled (host low-sustain + held donor notes → decays to silence). Probe
(`/scratch/tmp/mt_adsr.py`, ADSR 4-vector = mean attack/decay/sustain/release
distance) confirms it: distance **1.37 → splice RMS 2621–3847** (healthy), vs
distance **18.94 → RMS 282** (the quiet symptom) or parse-failure. So add
**ADSR-distance ≤ threshold** to the Phase-1 eligibility set (alongside the
pitch-range Bhattacharyya + FC-tracks-gate filters in "Five things that bite").

## Three augmentation families

| Family | Audio vs original | Validity gate | Corpus ceiling |
|---|---|---|---|
| **Inaudible macro perturbation** (new) | ~identical | `per_frame_rel_rms < ε` (measured) | large, song-local |
| Voice permutation (within-song) | identical (channel relabel) | parser admits | 6× |
| Melody transfer (cross-song) | intentionally different | plausibility test | ~30× |

The first two are *content-preserving* (same musical output, different
token stream — teaches the model what's representation vs content); the
third is *content-changing* (a new coherent song). All three bake to
`.dump.parquet` consumed by the unchanged parse → tokenize → train path.

## Inaudible macro perturbation (new family)

Re-emit a parsed song with macro **parameters** perturbed within a
*measured* inaudible tolerance: the token stream changes, the render
does not. The earlier corpus-expansion plan called this "lossless
permutations (slope ±1 LSB, PRESET grid offset)" and capped it at
"sub-2×" — but it assumed inaudibility rather than measuring it, and
predated most of the macro vocabulary. Two things changed: the pipeline
now emits a rich set of recognized macros (more knobs), and
`per_frame_rel_rms` lets us *verify* each perturbation is below an
audible floor instead of guessing.

Perturbation knobs (each yields a distinct atom stream, same audio):
- **PRESET** (`PWM_PRESET`/`FC_PRESET`): re-snap PW/FC to an adjacent
  table entry; keep only if rel-RMS stays under ε.
- **Content-tier re-quantization**: `slope` endpoint/runtime ±1; the
  `transpose`/`loop_transposed` 16-cent delta bin — any value inside a
  bin is free (decodes identically).
- **Segmentation choice**: `CTRL_BIGRAM`↔`CTRL_TRIPLE` re-grouping of a
  CTRL run; `FREQ_RUN` split points; `FREQ_NUDGE`/`RELEASE_UPDATE`
  vs raw SET on a residual — same per-frame state, different ops.
- **Intra-frame VOICE/write order** where order-independent.

Gate: render original vs perturbed, accept iff
`per_frame_rel_rms(orig, aug) < ε` over the whole song (ε ≈ 0.02, the
floor below which audits found differences inaudible; `CONSTANT/INITIAL/
DRIFTING` shape from `compare_renders` flags pathological cases). This
is the same drift-free per-frame fidelity machinery that caught the
multi-frame decode bug. Cheap (song-local, no cross-song eligibility,
no neural step) and safe (audibly-straying perturbations are rejected,
not shipped). Value: many token streams → one audio output teaches the
representation/content split directly, attacking overfit to one
tokenization.

## What

Offline corpus-expansion tool. Given two parsed SID dumps (donor A,
host B), emit a third dump A⊗B that runs **B's instrument program**
through **A's note schedule**: pitch and gate-on/off transitions come
from A, every other voice-state register and the filter section come
from B. Macros are regenerated by re-running the parser pipeline on
the spliced dump, so HARD_RESTART / LEGATO / CTRL_BIGRAM realign to
the new transition points automatically.

Output suffix `.aug<N>.dump.parquet` lives next to the host's
`.dump.parquet` and is consumed by the standard parse → tokenize →
train pipeline as if it were a fresh song.

## Why

Data-to-params is the binding constraint on the cross-composer KPI:
the in-flight `accuracy_push_prodlike_4x` arm has 1,673 training
blocks against 125M parameters. Lossless permutations (slope ±1 LSB,
PRESET grid offset, etc.) get a sub-2× expansion. Melody transfer is
combinatorially larger: 25,853 train SIDs × K=3 per host with a 10%
eligibility-filter admission rate ≈ ~75k augmented variants, ~30×
corpus.

Tracker-authoring prior (AGENTS.md Layer 4): swapping the instrument
column of a tracker pattern while keeping the note column is exactly
how composers iterate — same melody, different patch. Models trained
on melody-transferred pairs see the note/instrument decoupling
explicitly and across composers.

## Voice-register partition

| Stream | Regs (per voice V, base = V × 7) | From |
|---|---|---|
| Pitch | `FREQ_LO = +0`, `FREQ_HI = +1`, `CTRL bit 0 (gate)` | donor A |
| Pitch macros | `FREQ_TRAJ` (op 45; subsumes the former SLOPE_FREQ / FREQ_RUN / FREQ_VIBRATO / FREQ_NUDGE) | donor A (regenerated) |
| Instrument | `PW_LO = +2`, `PW_HI = +3`, `CTRL bits 1-7`, `AD = +5`, `SR = +6` | host B |
| Instrument macros | `PWM_PRESET`/`FC_PRESET`, `HARD_RESTART`, `CTRL_BIGRAM`/`CTRL_TRIPLE`, `RELEASE_UPDATE`, `SLOPE_PW/FC_*` | host B (regenerated) |
| Filter (shared) | `FC_LO = 21`, `FC_HI = 22`, `RES_FILT = 23`, `MODE_VOL = 24` | host B |
| Structural | `DELAY`, `BACK_REF`, `PATTERN_REPLAY` | donor A (donor sets the tempo) |

CTRL is the only register that splits bit-wise: bit 0 is the note
gate (melody) and bits 1-7 are waveform / sync / ring / test
(instrument). The prototype's splice rule on CTRL is
`(A[CTRL_V] & 0x01) | (B[CTRL_V] & 0xFE)`.

## Pipeline

```
load A.dump.parquet → df_A           (raw register-writes)
load B.dump.parquet → df_B
splice(df_A, df_B) → df_AB           (raw spliced writes, no macros)
write df_AB to B.aug<N>.dump.parquet
parser pipeline (existing)           re-derives macros on next parse
```

The splice operates frame-by-frame after a lightweight frame-bucket
pass: each PAL IRQ tick groups all writes within that frame. For
each frame, the voice-state cells are taken from A or B per the
partition table; the resulting per-frame write batch is emitted into
the output stream with A's `(clock, irq)` timing. No macro logic
runs at splice time.

## Five things that bite

1. **Length mismatch.** Truncate output to `min(N_frames_A,
   N_frames_B)`. Host instrument programs are usually stationary
   enough that this works without phase artifacts; if it does
   matter, future variant: loop B from frame 0 when it runs out.
2. **HARD_RESTART/LEGATO are tied to transitions.** Mechanism: drop
   host's HARD_RESTART/LEGATO tokens at splice time; re-running the
   macro pipeline on the spliced dump regenerates them at A's note
   boundaries using B's restart pattern.
3. **Pitch-range mismatch.** Bass→lead voice transfer is incoherent.
   Eligibility filter (Phase 1+): per-voice FREQ histogram
   Bhattacharyya coefficient ≥ 0.5 between A.V and B.V, else skip
   that voice (keep host's voice untouched in the output).
4. **FC tracks gate** in some host instruments. After splice, gate
   timing changes but the host's FC envelope is unchanged → audible
   artefact. Detector (Phase 1+): cross-correlate `gate_on[V]` with
   `d(FC)/dt > thr` on the host; if r ≥ 0.5, skip the host.
5. **Slope-FREQ is melody, not instrument.** All `SLOPE_FREQ_*` ops
   must travel with the pitch stream even though they share an op
   family with `SLOPE_PW_*` / `SLOPE_FC_*`. The voice-register
   partition table makes this explicit.

## Audio-fidelity gate (melody transfer)

Per-frame render equivalence does **not** apply here — melody transfer
intentionally changes audio (that gate is the inaudible-perturbation
family's job, above). The melody-transfer gate is a **plausibility
test** on the `preframr-audio` render of the spliced dump:

| Check | Reason |
|---|---|
| `RMS(wav_AB) > floor` | not silent (catches stuck-gate / wrong-voice output) |
| `mean(wav_AB) ≈ 0` | not DC-pinned |
| spectral centroid ∈ host's mean ± 2σ | instrument character preserved |
| gate-on transition count matches donor | note schedule faithful |
| no parse errors / over-budget ctrl frames | encoder admits the output |

Implemented by `integration_tests/profile/audit_melody_transfer.py`
(not yet landed; phase 0 task).

## Refutation pressure

- `adsr_equivalence` REFUTED (registry: `integration_tests/data/refuted/`)
  — past evidence that "musically equivalent" augmentation can degrade
  next-token prediction by adding ambiguity. Counter-argument here:
  melody transfer is **not** claiming equivalence; the augmented
  sample is a distinct but musically-coherent song, not a "should
  match" pair.
- `transpose_xframe 2-stage` on HOLD — pitch-space augmentation
  inconclusive at mini. Melody transfer is structurally different
  (full note schedules, not just keys) but the failure mode (the
  model fails to generalize across pitch changes) is shared.
- `coarsen_pass` retained only as a tracker-export tool because lossy
  encoding A/Bs degraded val_acc. Same lesson: don't conflate
  "structurally similar" with "training-equivalent".

## Phase plan

| Phase | Scope | Pass gate |
|---|---|---|
| 0. Smoke audit | 7 SIDs from `smoke.list`, all-pairs (42 pairings × 3 voices), hand-listen 5 random renders | coherent music, no obvious artefacts; parser admits 100% of outputs |
| 1. Mini A/B | 196 SIDs, K=1 transfer per host, fixed seed | +1% val_acc cross-composer on mini |
| 2. Canonical A/B | 901 SIDs, K=2, eligibility filter on | +0.5% val_acc on canonical |
| 3. Prodlike | 25,853 train SIDs × K=3, baked on `fogbank` (72 cores) | net val_acc gain on prodlike vs `accuracy_push_prodlike_4x` baseline |

Audit fields per phase: admission rate, render success rate, FREQ
Bhattacharyya distribution, gate-count delta histogram.

## Wallclock estimate

- Splice: O(N_frames) pandas ops on two ~70k-row dumps; estimated
  ~50 ms per (donor, host) pair.
- Macro re-encode: full parser pipeline ≈ 250-300 ms per dump (the
  same 4 it/s rate observed on prodlike_4x parse). Dominates.
- Prebake budget: 75k variants × 0.3 s ÷ 72 cores ≈ 5 min on
  `fogbank`. Negligible.
- Training-time overhead: zero (augmented dumps look like fresh
  songs to the tokenizer).

## What this is NOT

- Not an online (training-loop) augmentation. Output is baked
  parquet on disk, consumed by the unchanged train pipeline.
- Not a synthesizer. The output is a register-write dump that the
  C64 SID can replay bit-exact; no neural audio generation.
- Not a melody-swap (each side gets the other's). It's a one-way
  transplant per pair; A⊗B and B⊗A are two distinct outputs and
  count as two augmented variants.

## Open questions (deferred to phase 1 readouts)

- Voice rotation: should the donor's voice 0 map to host's voice 0,
  or to whichever host voice has the best FREQ-range overlap?
  Phase 0 uses identity mapping; phase 1 measures gain from a
  rank-based assignment.
- Tempo: structural ops (`DELAY`, `BACK_REF`) come from donor in
  the spec above, but per-frame state-bucket variant might benefit
  from preserving host's `DELAY` to keep instrument envelope timing
  intact. Audit both.
- Cross-engine pairs: if donor and host land in different
  `engine_fingerprint` clusters, eligibility filter may reject them
  all. Decide whether intra-cluster pairing is too restrictive.

## Sibling variant: voice permutation (within-song)

**Status (draft 2026-05-23):** complement to the cross-song splice
above. Smaller corpus expansion (≤ 6×, vs splice's ~30×) but
trivially valid (no eligibility filter, no audio-plausibility check
beyond "is the SID dump still parseable"). Recommended as the
first-launched augmentation experiment because it removes the largest
class of cross-song failure modes from the variable space.

### Mechanism

For each parsed SID dump, generate one variant per non-identity
permutation σ ∈ S_3 (5 variants per song max). σ remaps voice
indices 0..2:

- For each register write `(clock, reg, val)`:
  - If `reg < 21` (voice-specific): `voice = reg // 7`,
    `new_reg = σ(voice) * 7 + (reg % 7)`.
  - Else (filter regs 21-23, master volume 24): pass through.
- For `RES_FILT = 23`, bits 0-2 are voice-filter-routing flags
  (`FILT_VOICE_1/2/3`). These get bit-permuted by σ. Other bits
  unchanged.
- Macros (`HARD_RESTART`, `LEGATO`, etc.) are voice-keyed and
  travel with the voice they reference; re-running the parser
  pipeline on the permuted raw dump regenerates them under the new
  voice assignment.

Output: `B.perm<σ>.dump.parquet` (e.g.,
`SongName.perm120.dump.parquet` for σ = (1,2,0)) next to the input
dump. Consumed by the standard parse → tokenize → train pipeline as
fresh songs.

### Why it works

Tracker-authoring prior (AGENTS.md Layer 4): in real tracker
authoring, "which voice plays which part" is an arbitrary choice
made early in composition. The model currently overfits to that
arbitrary choice (e.g., learns "voice 0 is usually lead") because
every song's voice-role assignment is fixed in training. Permutation
breaks this voice-position coupling without changing musical content,
forcing the model to attend to register-content patterns rather than
register-address patterns.

### Cost

| stage | wallclock |
|---|---|
| splice (raw register remap) | ~5 ms per dump, single core |
| macro re-encode (parser pipeline) | ~250 ms per dump (existing parser throughput) |
| mini (196 SIDs × 5 perm = 980 variants) | ~5 min sequential, ~10 sec on fogbank |
| prodlike (25,853 SIDs × 5 = ~130k variants) | ~10 min fogbank |

Training-time overhead: zero (variants look like fresh dumps).

### Comparison to cross-song splice

| | splice (existing) | permutation (this section) |
|---|---|---|
| corpus expansion ceiling | ~30× | 6× |
| eligibility filter needed | yes (FREQ range, FC-tracks-gate) | no |
| audio plausibility gate needed | yes | no (output is bit-exact a permuted-channel version of the input) |
| failure mode if mis-applied | acoustic chaos (e.g., bass freq through lead voice) | none (parser admits or rejects deterministically) |
| eng risk | medium (5 known biting issues, see above) | low (single integer remap) |
| signal independence | sees cross-composer pairings | sees only intra-song reassignments |
| combine with splice? | yes — permutation-of-permutation × splice gives ~150× ceiling | yes |

Recommend launching permutation first (Phase 1 mini A/B), splice
second. If both pass independently, combined run is the prodlike
escalation.

### Phase plan (permutation variant)

| Phase | Scope | Pass gate |
|---|---|---|
| 0. Smoke | Run on 7 SIDs from `smoke.list` with σ = (1,2,0). Verify 35/35 (5 perm × 7 SID) parse cleanly; spot-check 3 renders are audible. | parser admits 100%, audio non-silent |
| 1. Mini A/B | `voice_permutation_mini_body_large` spec (alongside this doc): baseline vs `voice_permutation_K5` (all 5 non-identity permutations per song). 196 SIDs × 6 = 1176 effective. 3 seeds. | val_acc on eval_a ≥ baseline + 0.005 AND no structural-tier regression > 1σ |
| 2. Canonical A/B | 901 SIDs × 6, same gate as phase 1 at canonical scale | val_acc on eval_a ≥ baseline + 0.003 |
| 3. Prodlike | 25,853 SIDs × 6, single seed | net val_acc gain on prodlike vs the canonical baseline |

### Combination with splice (post-both-pass)

Two independent gains multiply:
- Permutation alone: 6× corpus, gains G_p.
- Splice alone (with eligibility filter at 10% admission): 3× corpus
  expansion at K=3, gains G_s.
- Combined (permutation-of-permutation as splice donors): 6 × 3 = 18×
  effective corpus, gains hopefully ~G_p + G_s if the augmentations
  attack independent generalization failure modes.

Decision tree: launch in order [permutation alone, splice alone,
combined]. Refute either intermediate stage → don't run the
combined-prodlike (saves ~13 hr on the 4090).

## Sibling variant: percussion patch substitution (exploratory, 2026-05-25)

A simpler patch swap than melody transfer: identify a song's **percussion
patch** (the recurring noise/pulse + percussive-envelope + pitch-drop note,
Hubbard-Commando style) and substitute another song's drum patch at the same
rhythm slots — keep the groove, change the kit. Probes (exploratory, not yet
landed): `/scratch/tmp/{percussion_probe,percussion_audio}.py`.

**Register-side identification (what works):**
- **Noise waveform (CTRL bit7)** = high-precision percussion (hats/snares), low
  recall (~few per song).
- **Strict-repeat signature** (full note patch incl. pitch, seen ≥4×) cleanly
  **excludes melody** — melody is the unique-signature tail (~6% of onsets) since
  its pitch varies. But strict-repeat alone is **broad (~77%)**: SID loops mean
  bass/arp repeat too. Likewise **short+repeat ≈ most onsets** — not selective.
- Best register prior so far: **strict-repeat ∩ percussive-timbre (~20%)**.
- Per-song distinct-signature count is small (6–259) → a small **patch
  vocabulary**, so substitution is tractable.

**Acoustic-side (solo render + spectral features, librosa-style via numpy/scipy):**
`render_per_voice` solos a voice; slice the gate-on window; extract centroid /
flatness / rolloff / ZCR / RMS / attack. **The features work** — a
`sustained-rep(bass?)` group separated cleanly (centroid 2182 vs ~3200, flatness
0.074 vs ~0.15 = darker/tonal). **But** the register short+repeat selection is
too broad to isolate drums (its features ≈ melody), and a **gate-on→sample
alignment bug** remains (attack ≈ 0.5 = window not landing on the transient).

**Next:** (1) fix the gate-on→sample alignment (verify against a known hit);
(2) let the **audio features cluster** identify percussion (noise/transient
cluster) rather than register pre-selection, using strict-repeat∩short as a prior;
(3) then swap the percussion patch's register program between songs (splice-like,
but the drum patch instead of the lead). librosa can be added to
`predict-requirements` if this becomes a pipeline.
