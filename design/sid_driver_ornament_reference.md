# SID driver ornament generation — reference

**Status:** Reference (background). How C64 SID music drivers generate per-frame **ornamentation** —
pitch (arpeggio / vibrato / portamento), **pulse-width**, and **filter** — at the register level.
Distilled from a 2026-05-29 read of four drivers (References below). Other designs cite this rather
than re-deriving it; see [`generator_mdl_representation.md`](generator_mdl_representation.md) (the current
encoding that builds on this), [`encoding_principles.md`](encoding_principles.md),
[`landed/trajectory_anchoring.md`](landed/trajectory_anchoring.md),
[`landed/freq_v0_interval.md`](landed/freq_v0_interval.md),
[`voice_encoding_reference.md`](voice_encoding_reference.md).

## The register targets

Ornament is per-frame (50 Hz PAL / 60 Hz NTSC) modulation of three kinds of SID register, with
different scope:

| target | SID regs | width | scope |
|---|---|---|---|
| **pitch** | `$D400/1` (+7,+14) | 16-bit, via a note→freq table indexed `note<<1` | per voice |
| **pulse width** | `$D402/3` (+7,+14) | 12-bit (hi nibble + lo byte) | per voice |
| **filter** | `$D415` cutoff-lo (3 bits), `$D416` cutoff-hi (8) = 11-bit cutoff; `$D417` resonance (hi nibble) + routing (lo 3 bits, one per voice); `$D418` mode LP/BP/HP (bits 4–6) + master volume | **GLOBAL — one filter, all 3 voices** |

The note→frequency table is universal (Hubbard `FREQ_TBL`, Galway `LoFrq/HiFrq`, SID Wizard
`NOTE_FREQ_LO/HI`, defMON `NOTE_PITCH_LO/HI`), indexed by `note<<1`. **Pitch is fundamentally a
semitone index** into it — an encoding in semitone/interval space (interval-V0) is in the driver's
native domain.

## Two value-domain mechanisms, used for every target

Across all drivers and all three targets, ornament is produced by one of two mechanisms:

- **(A) Note-index offset cycle** — only for *pitch*: an ordered, looped table of **note-relative
  semitone offsets**, stepped one entry per frame, added to the note *before* the freq-table lookup.
  Transposition-invariant, low-cardinality, reused per instrument → naturally a **codebook of
  semitone-offset cycles**.
- **(B) Parametric / table sweep in the target's own value domain** — for vibrato, portamento, PW,
  and filter cutoff: a value swept per frame, either by a **program table** of set/sweep entries
  (SID Wizard) or a **parametric bounded sweep / gradient envelope** (defMON, Hubbard, Galway). These
  are *not* note-relative and *not* a small codebook; they are continuous swept values with a few
  parameters (depth/rate/direction/bounds, or staged gradients). Galway notably uses **one** gradient
  envelope structure (`G0..G3` values + `D0..D3` durations + delay) for vibrato, PW *and* filter.

## Pitch ornament

- **Arpeggio (mechanism A).**
  - *General (tracker) form* — an arbitrary note-relative offset cycle: Galway "OFFSET LIST"
    (`FOLA/FOLB` addr/base, `FOLCI` index, `FOLDC` rate; `ADC offset → TAY → LDX LoFrq,Y`); SID Wizard
    `wf_table` `arp_byte` (`$00–$7E` up, `$E0–$FF` down, `$81–$DF` absolute, `$80` NOP, `$7F` chord
    jump, `$FE` jump/loop; paced by `arp_speed_counter`); defMON `TR` transpose steps across reused
    sidTAB rows (paced by cascade `DL`).
  - *Hubbard early routine* — **octave arpeggio only**: "for the first 50th of a second the current
    note is played, and for the next, current note+12, then the current note again" — a hard-coded
    `note / note+12` toggle at 50 Hz, set by instrument fx **byte 7 bit 2**. Not a general table.
    Many Hubbard tunes = octave-arp + skydive.
- **Vibrato (mechanism B, frequency domain, parametric).** Galway: gradient stages `FMG0..FMG3` +
  durations `FMD0..FMD3` + delay `FMDLY`, added to the freq word. SID Wizard: amp/freq nibbles drive a
  triangle via a FREQMOD table. defMON: freq-word LUT. Hubbard: a single **depth** byte (instrument
  byte 5) raises/lowers pitch slightly.
- **Portamento / slide (mechanism B, frequency domain).** Per-frame add/subtract toward a target.
  Hubbard: a per-note byte = speed, with bit 0 = direction (up/down), applied when the note's flag bit
  marks it bent. defMON: LUT accumulator (active-slide) or single-frame step; `GATE_N` zeroes it.
  SID Wizard: 16-bit accumulator toward the target freq. Galway: `PMG/PMD` gradient.
- **Drum / "skydive" (Hubbard fx bits 0/1).** A fast (drum) or slow (skydive) frequency-down done by
  decrementing the **MSB of the frequency** (8-bit, not 16-bit) — a coarse pitch-drop effect, often
  combined with the octave arp.

## Pulse-width ornament

PW is a **per-voice, absolute 12-bit value that is swept** (square ↔ rectangular timbre). Two forms:

- **Table (SID Wizard `pw_table`, 3-byte rows).** Byte 0 mode: `$80–$FF` set-mode (7-bit PW-hi, bit 7
  set) → byte 1 is PW-lo (absolute 12-bit); `$00–$7F` sweep-mode (cycle countdown) → byte 1 is a signed
  per-frame delta accumulated into the 12-bit value; `$FE` jump, `$FF` freeze. Optional keyboard-track
  in byte 2. Walk rate from instrument multispeed flags.
- **Parametric bounded sweep.** defMON `_ps_voice` (PS column): state = `ps_depth` (bit 7 direction,
  bits 0–6 magnitude); adds/subtracts magnitude to the 12-bit PW each frame and **auto-reverses at the
  extremities** (clamp PW-lo at `$F8`/`$01` and flip the direction bit when PW-hi hits `$0F`/`$00`).
  Hubbard "pulsework": ramps between square and very-rectangular at instrument **byte 6 (pulse speed)**,
  reversing at each extremity; the live PW value is stored back in the instrument. Galway: `PMG/PMD`
  gradient stages.

**PW sweeps persist across notes** (held PW) — defMON `ps_depth` and SID Wizard `pw_pos` are *not*
reset by a new note; SID Wizard resets PW only on a fresh note unless instrument control bit 6
(`pulse_reset_off`) is set. So PW modulation is a continuous per-voice sweep, **not note-aligned**.

## Filter ornament

The SID filter is **global**: one cutoff/resonance/mode shared by all voices, with a 3-bit routing
nibble selecting which voices feed it. So filter ornament is **one tune-global channel**, not
per-voice. Drivers therefore use a **filter-controller voice**: at most one voice drives the cutoff/
resonance/mode sweep at a time; the others only set their own routing bit.

- **Cutoff sweep.** SID Wizard `filter_table` (3-byte rows): set-mode (`$80–$FD`) byte 0 carries mode
  (bits 4–6) + resonance (bits 0–3), byte 1 is the 11-bit cutoff-hi (absolute); sweep-mode (`$01–$7F`)
  byte 1 is a signed delta accumulated into the 11-bit cutoff; `$00` = "filtered but not controller",
  `$FE` jump, `$FF` not-filtered/relinquish. defMON ACID column: a 16-bit cutoff accumulator, either
  absolute load or slide (control byte bit 7 = slide, bit 6 = direction → ADC/SBC), with a floor clamp;
  CP column adds an extra. Galway: `FMG/FMD` gradient stages.
- **Resonance / mode / routing** are low-cardinality absolute sets written by the controller:
  resonance = `$D417` hi nibble; mode LP/BP/HP = `$D418` bits 4–6; routing = `$D417` lo 3 bits, one
  bit per voice (defMON RE column does per-voice set/clear; SID Wizard ORs in each filtered voice).
- **Controller arbitration.** SID Wizard: a voice claims the controller when its `filter_table[0]` is
  non-`$00`/`$FF`, relinquishes on `$FF`; a global `filter_sweep_count` is shared so the sweep
  continues across controller swaps. defMON: last voice to write the RE/FV/ACID globals wins. Galway:
  a `FilterChannel` zero-page byte names the controlling voice.

**Filter sweeps persist across notes** too (global accumulator, not reset by `GATE_N` / new note
unless an instrument reset bit). The Hubbard *early* routine barely uses the filter (its expressive
load is octave-arp + PW + portamento); filter sweeps are heavy in Galway and the trackers.

## Note segmentation caveat: gate-on is not the note boundary

Hand-written / legato drivers **hold the gate** and move pitch via freq-table lookups + portamento
rather than re-gating per note. Confirmed in Hubbard's routine (C=Hacking #5): a pattern note's flag
**bit 6 = "appended to the last (no attack)"** — a legato/tie that does **not** re-trigger the gate;
and the octave arp / portamento run under one sustained gate. So **a gate-segment can span several
melodic notes plus their connecting slides and arp** — gate-on transitions under-segment the melody
for these drivers, mis-reading held-gate note changes as "ornament."

The robust note detector is an **intrinsic level-change** detector (sustained pitch-level changes),
not the gate — exactly the landed `TrajectoryAnchorPass` **pass-1** origin detector
([`landed/trajectory_anchoring.md`](landed/trajectory_anchoring.md)), with the gate as a complementary
signal for re-struck same-pitch notes. (This is the mechanism behind the "gate-anchor refuted"
melody-ladder finding: the inflating "ornament" was partly held-gate melody.) Note also that PW and
filter sweeps are **not** note-aligned (they persist across notes), so they should not be segmented on
note boundaries at all.

## The control register disambiguates the freq trajectory (modeling insight, 2026-05-30)

Read from the local reimplementations (`pysidwizard/src/pysidwizard/player.py`,
`pydefmon/pydefmon/defmon_player.py`) — **the freq trajectory cannot be interpreted in isolation; the
control register (gate / test-bit / waveform) is the context that says *why* a freq frame is what it
is.** A freq-only view (as the RESID-archetype survey does) is working blind and mis-reads structured
control-driven frames as RESID. Three concrete mechanisms:

- **Hard-restart (HR) onset window = the "transient" archetype.** SID Wizard primes each note with a
  hard restart: `PRE_HR_LEAD_FRAMES = 2` frames before the row it **clears the gate** and writes
  HR-AD/HR-SR, then `HR_FRAMES = 1` frame with the **TEST bit set** (`CTRL=$09`), before the WF table
  walks. NB SID Wizard combines TWO distinct mechanisms — do not conflate them: (1) the **classic ADSR
  hard-restart** = the gate-based ADSR-bug workaround (gate off + reload AD/SR ~2 frames early so the
  1.7-frame ADSR-bug window elapses and the note attacks from a known state; no TEST bit). Its onset
  frames are gate-off/**release**, where **freq IS audible**. (2) the **TEST bit** = an oscillator-phase
  reset (`CTRL` bit 3); **only on a TEST-bit frame is freq (near-)don't-care**. So every note onset has a
  ~3-frame transient window, **detectable from the control register (test bit / gate-off-then-on), not
  the freq** — but freq is only absorbable on the TEST-bit frame, not the gate-low release frames. Encode HR as a note-onset marker; the HR frames'
  freq is absorbed **to the adjacent note's pitch, not an arbitrary constant** — under the renderer's
  real per-write timing a wild triangle freq leaks through the pre-TEST window, and the HR frame's **PW
  is audible** (see [`sid_render_fidelity_contract.md`](sid_render_fidelity_contract.md), proven in
  preframr-audio 0.5.5). (Also `gateoff_wf/pw/filt`: the release phase swaps waveform/PW/filter
  — more non-note-pitch per-frame variation, again control-flagged.)
- **One-shot chord "pluck" (SID Wizard `$7E` vs looping `$7F`):** a one-shot chord at attack is a
  brief arp transient at note start (vs a continuous arp), another onset ornament — `OCTAVE`/`ARP` but
  short and attack-localised.
- **Gate / waveform flicker mid-note (defMON):** tunes deliberately **flicker the gate and waveform
  mid-note** for character — a short noise phase ⊕ a long pulse tone is one *instrument*, not a drum
  (the user's caution), and rapid gate flicker is an effect. So a noise-waveform frame is
  percussion-*timbre*, not a melodic pitch; the freq there sets drum timbre / skydive.

**Encoding implication (drives the RESID=0 program):** co-read the control register with the freq and
let the control state assign each frame's *role* — `test`/`gate-low` → HR transient (absorb);
`noise waveform` → percussion-timbre (its own primitive, not melodic SKEL); gated pitched frame →
melody. This is the driver-grounded replacement for the freq-only median/heuristic transient detection
in the first-cut `_rebased_note`/`_is_transient_blip` (#16), and likely collapses several freq-survey
archetypes at once. Drums in these editors are **wavetable/sidcall instruments** (noise + freq/PW
manipulation), not a separate driver primitive — so a parametric "percussion/sweep" primitive plus
control-context covers them, no per-driver drum code.

**The precise noise rule — noise is NOT always a drum (concrete: Wiklund *Facemorph*).** A
note-onset noise burst commonly **accents a *pitched* lead** (the "noise-tik": MoN `fx3 $80`,
SF2 driver-13 "add noise in the beginning of note", Hubbard "noise on first vblank"). Facemorph's
voice 0 is, per note: `tri (1f, HR setup) → NOISE @ freq≈note107 (1f accent) → pulse @ note 31/43…
(sustained melody)`. The noise frame's "freq" (107) is **timbre, not pitch** — tracking it as melody
gives a bogus +76-semitone jump; excluding the note as "drum" drops a real melodic note. **Rule: a
noise-waveform frame contributes timbre (already carried by the ctrl/waveform tokens), NOT melodic
pitch.** A note's melodic pitch is taken from its **pitched (non-noise) frames**; the noise frame is
absorbed from the *melody* skeleton (its accent survives in the waveform channel). A note with **no
pitched frames at all** is percussion (its freq = drum-timbre / sweep — its own primitive). So a
noise-accented lead (has pulse frames → a real note) and a pure kick (no pitched frames → percussion)
are separated by **whether the note has any pitched-waveform frame**, decided by the control register,
not the freq.

## Reuse / banks

Ornament definitions are a **small bank referenced by id**, reused across notes/patterns:

- **Hubbard** — 8-byte instrument records (PW-lo/hi, control/waveform, AD, SR, **vibrato depth**,
  **pulse speed**, **fx byte**); one routine + instrument bank serves a whole module (title / in-game /
  game-over) and was reused across ~30 tunes incl. *Commando*, *Monty on the Run*, *Crazy Comets*.
- **SID Wizard** — instruments hold contiguous `wf_table | pw_table | filter_table`, shared per id.
- **defMON** — sidTAB rows referenced by index, shared across voices.
- **Galway** — per-voice parameter banks `D0/D1/D2`.

This is the structural basis for encoding ornament by table-id with per-instrument / per-composer bank
conditioning.

## JCH NewPlayer (20.G4) — table-driven, the JCH/Vibrants lineage

Engine behind a large slice of HVSC (JCH, Vibrants) and modern chiptune — **Goto80's *Baggis*** and
**DRAX's *Camerock*** both run it (identified via `sidid`). Format: JCH editor v3.04 "20.G4" /
NewPlayer (codebase.c64.org "jch_20.g4_player_file_format"); **CheeseCutter** (theyamo) is the modern
reference editor for this exact format, so its in-editor reference is the readable spec. Fully
**table-driven** — same two value-domain mechanisms as the others, with a JCH-specific twist (the
wave-table drives BOTH waveform and pitch).

- **Instrument (8 bytes):** ADSR (A,B); hard-restart-type hi-nibble + wave-program-delay lo-nibble (C);
  HR waveform (D); **filter-table ptr (E)**; **pulse-table ptr (F)**; **wave-table ptr (H)**. HR types:
  `0x` soft gate-off, `4x`, `8x` hard restart, `Ax` Laxity (restart waveform only, keep AD).
- **Wave table (per frame, 2 cols/row) — drives waveform AND pitch.** note/transpose col:
  `00–5F` note-relative transpose (**the arp offsets**, semitone domain); `80–DF` **absolute note**
  (ignores chord/transpose — a melodic note driven straight from the wavetable); `7E` hold (keep prev);
  `7F` jump (next byte = loop index). waveform col: `00` no change; `01–0F` wave-delay override (chord
  timing); `10–DF` SID waveform; `E0–EF` remapped waveforms. One row/frame.
- **Chord/arp table** (the 20.G4 "Arpeggio Table" col1/col2): note-relative offset cycle — `00–3F`
  positive semitone offsets (e.g. `0,4,7` major), `40–7F` negative (`7F`=−1, `7E`=−2), `80–FF`
  loop/wrap index. Row 0 = swing-tempo program (when song speed = 0/1).
- **Pulse table (4 bytes/row):** duration (A; sign = sweep dir), add (B; per-frame, two's-comp), init
  (C; `FF`=keep), jump (D; `00` next, `7F` stop) → PWM sweep. **Filter table:** init rows (A≥`80`:
  type, resonance+voice-mask, cutoff) + sweep rows (duration, add (`FF`=−1), init, jump); 10-bit
  cutoff (4× finer than old JCH players); **global**.
- **Commands (per sequence step):** `2` delta vibrato (A=speed, B=depth); `5` low-fi vibrato
  (speed,depth); `3` detune (16-bit freq offset → sub-semitone); `0`/`1` slide up/down; `7` portamento.

**Note segmentation — the held-gate twist that causes the RESID gap.** Tie-notes (`AA=$90` / `BB` tie;
no gate retrigger) hold the gate across many sequence steps. Two mechanisms then drive pitch *without*
a gate retrigger, so "gate-on ≠ note boundary":
- **Wave-table absolute-note runs (`80–DF`)** = a fast per-frame melodic line under one held gate,
  steps below `MIN_HOLD` → currently falls to RESID → should **segment into notes** (each an absolute
  wavetable note). **This is the dominant gap, and it is NOT JCH-specific** — see below.
- **Portamento (`7`) across tied notes** (slide/vibrato do NOT run on tied notes, but portamento does):
  a portamento armed before a run of tied notes slides *across* them → one long continuous glissando
  under a single held gate → encode as a **SLIDE chain across the re-segmented constituent notes** (the
  held-gate re-segmentation must also cut at portamento transitions, + allow a longer SLIDE span).

**Measured reality (2026-05-30, deterministic test suite #11, skeleton-on, post-resegmentation —
supersedes the earlier "Camerock clean / Baggis gap" framing, which was a pre-resegmentation
artefact):** the remaining RESID across *every* driver is overwhelmingly **fast-melodic-run
under-segmentation** (the absolute-note-run mechanism), **not** genuine glissando:
- **Trap.1** (Daglish, Antony Crowther V3): RESID = **98.8% fast-melodic-run**, 0.5% glissando.
- **Baggis.1** (Goto80, JCH NewPlayer): RESID = **75.6% fast-melodic-run**, 12% glissando, 6% periodic
  long-arp, 6% aperiodic-noise.
- By RESID note-share, **Commando (0.34) and Camerock (0.37) leak as much or *more* than Trap (0.14) /
  Baggis (0.06)** — so there is **no per-driver clean-vs-gap split**; the dominant gap is a single
  **shared segmentation mechanism** common to all four drivers. (This is itself early evidence for the
  collapse hypothesis — see backlog #15.) Portamento-across-tied-notes is a real but *minor* secondary
  component, material only on Baggis.

**✅ CLOSED 2026-05-30 (tokens 0.35.0, #13):** `SkeletonPass._resegment_fast_run` folds fast-melodic-runs
into per-step SKEL notes. New RESID note-share: Trap **0.44→0.01**, Camerock **0.17→0.06**, Baggis
**0.66→0.26**, Commando 0.25→0.24; fast-melodic-run frame-fraction → ~0 (Trap) / 0.009 (Baggis). The
shared fast-run gap is gone. **Baggis's remaining 0.26 is a DISTINCT primitive** — wide/aperiodic
content (span 51–71 semitones, ≤8 distinct: octave-jump wavetable effects / noise), NOT a melodic run
(no melody leaps 4–6 octaves between frames); splitting it would forge spurious giant-interval notes,
so it stays RESID. That wide-aperiodic primitive is the next real-tune gap to characterize.

**Reconciliation to our encoding (what's modelled vs the open gap):** arps (wavetable relative
transpose / chord-table) → **ARP**; vibrato (`2`/`5`) → **VIB** (depth+rate); detune (`3`) →
sub-semitone **cents/VIB**; slide/portamento (`0`/`1`/`7`) → **SLIDE** (target+rate); pulse/filter
tables → PW/filter trajectories (out of pitch scope; ablated). **The open gap is dominated by
fast-melodic-run / absolute-note-run under-segmentation (shared across drivers, extends the held-gate
re-segmentation of #12), with portamento-across-tied-notes as a minor JCH-specific secondary.**

> **Antony Crowther V3** (Trap, Daglish) is a *separate* driver, **not yet documented here** — its
> RESID gap needs its own reverse-engineering (no readable source pulled yet). Tracked as a follow-up.

## Per-driver summary

| | pitch arp | vibrato | portamento | pulse width | filter |
|---|---|---|---|---|---|
| **Hubbard** (early) | octave-only (note/+12 @50Hz), fx bit 2 | depth byte (instr byte 5) | per-note speed+dir byte | bounded sweep @ pulse-speed (instr byte 6) | minimal |
| **SID Wizard** | `wf_table` arp_byte (general offset cycle) | amp/freq-nibble triangle | 16-bit accumulator → target | `pw_table` set/sweep rows | `filter_table` set/sweep + controller voice |
| **defMON** | `TR` steps in reused sidTAB rows | freq-word LUT | LUT accumulator / 1-frame step | `PS` bounded auto-reverse sweep | `ACID` cutoff accumulator + `RE` routing |
| **Galway** | `FOL*` offset list | `FMG/FMD` gradient | `PMG/PMD` gradient | gradient stages | gradient stages + `FilterChannel` |
| **JCH NewPlayer** | wave-table `00–5F` rel-transpose + chord table; `80–DF` absolute note | cmd `2`/`5` speed+depth; cmd `3` 16-bit detune | cmd `7`, slides **across tied notes** | pulse table dur+add (`FF`=keep), jump | filter table dur+add, 10-bit, global |

## Encoding takeaways

- **Arps → a codebook of note-relative semitone offset cycles** (mechanism A), semitone-domain and
  transposition-invariant; add an **octave-arp** special primitive (Hubbard's dominant form).
- **Vibrato / portamento / PW / filter → parametric sweeps** in their own value domains (mechanism B),
  *not* offset codebooks and *not* raw per-frame deltas. One bounded-sweep / gradient primitive family
  generalizes across them (as Galway's shared envelope shows).
- **Filter is global** — encode it as **one tune-level channel** (cutoff sweep + low-cardinality
  resonance/mode/routing sets), not triplicated per voice; this matches `TRAJ_REGS`' treatment of the
  filter cutoff (reg 21) as a single intrinsic, non-gated trajectory.
- **PW and filter sweeps are not note-aligned** (they persist across notes) — segment/encode them as
  continuous per-voice (PW) / global (filter) trajectories, unlike the note-aligned pitch arp.
- **Use intrinsic level-change, not the gate, for note boundaries** on legato/held-gate drivers.

## The common abstraction (universal driver) — #15

**The collapse hypothesis is empirically confirmed, and the universal driver already exists in the
code.** The skeleton+ornament encoder/decoder (`skeleton_pass.py` / `decoders.py`) has **zero
per-driver / per-engine / per-tune branching** — the decoder dispatches purely on `ORN_TYPE`
(`PLAIN`/`OCTAVE`/`ARP`/`SLIDE`/`VIB`/`RESID`) — yet every driver above maps onto that one generic
primitive set. Each driver's per-row table is just a *surface encoding* of the same gesture; the
tokenizer only ever sees the rendered register writes, so provenance (which driver, or hand-written
code) collapses automatically at the register-log input.

**Mechanism → universal primitive (the collapse matrix):**

| universal primitive | the same gesture across drivers |
|---|---|
| `OCTAVE` | Hubbard fx-bit2 note/+12; any 2-step `{0,12}` cycle |
| `ARP` (offset-cycle codebook) | Hubbard fx, SID Wizard `wf_table` arp, defMON `TR`, Galway `FOL*`, JCH chord table |
| `SLIDE` (target+rate) | all drivers' portamento/pitch-slide (incl. JCH cmd `7` across tied notes) |
| `VIB` (depth+rate) | all drivers' vibrato + sub-semitone detune (JCH cmd `2`/`3`/`5`, Galway/defMON gradients) |
| note segmentation | intrinsic level-change ∪ gate-on ∪ **fast-melodic-run split (#13)** — driver-agnostic |
| `RESID` | the lossless escape for content no primitive models YET — every non-zero RESID is an un-modelled engine to trace (the WAVETABLE codebook closes the bulk), not a floor |

**Empirical proof (real HVSC tunes, RESID note-share after #13):** Trap (Antony Crowther V3)
**0.01**, Camerock (JCH) **0.06**, Commando (Hubbard) **0.24**, Baggis (JCH) **0.26** — all through
the *same* code with no per-driver special-casing. **Trap reaching 0.01 with zero Crowther-specific
logic is the proof that Crowther V3 uses the common primitive set** (so #14 — "document Crowther V3
to model its RESID" — is satisfied empirically; there is no Crowther-specific RESID left to model).

**The "wide/aperiodic remainder" (Baggis 0.26 / Commando 0.24) is largely NOISE-timbre, not an
irreducible pitch primitive** — superseded by the control-register finding. Measured: **65% (Commando)
/ 76% (Baggis) of the wide (|≥12 st|) jumps land on the NOISE waveform** — drum hits / noise-accents
whose "freq" is timbre, not pitch (see "The control register disambiguates the freq trajectory"). The
control-aware rule (noise frame = timbre; pitch from pitched frames; no-pitched-frame note =
percussion) absorbs these from the melody; the genuinely-pitched residue is small — codebook-able wavetable ornament (Phase 3), not a floor (some
wide effects / octave-jump wavetable runs). **#15 itself stays a guarantee + a doc** (the
provenance-invariance test #11.4 + this matrix); the wide-RESID reduction is the RESID→0 program's
control-aware work.

## Cross-driver audit (2026-05-30): the primitive set holds, + the novel-mechanism frontier

A source-grounded sweep of seven more drivers (player code / official docs; see References). **The
universal primitive set survives** — and the freq-MSB-decrement **SWEEP** is a *named, first-class*
effect in three independent drivers (Hubbard "skydive", Maniacs-of-Noise "Tonesweep up", SID Factory
II driver-13 "Dive"), strongly validating it as core. Drums everywhere are **wavetable/table-driven**
(noise waveform + a freq-hi table, absolute or relative, often with a per-step gate-mask) — not a
separate engine — so a percussion primitive (noise + freq-table/sweep) + control-context covers them.

| mechanism | GoatTracker2 | Future Composer / Hippel (Amiga) | SID Factory II (d11/d13) | Maniacs of Noise | Hubbard |
|---|---|---|---|---|---|
| arp | wavetable rel/abs notes | sndseq transpose stream | arp table `T3` | tonearp offset table | octave-only (bit2) |
| vibrato | speedtable, triangle | triangle, **octave-scaled**, delay | `T1` triangle | **sine-LUT, delay+length** | triangle (depth byte) |
| portamento | `1/2/3` 16-bit accum; `3` toward-target; `3 00` tie | own 16-bit accum + timed pitch-bend | `T0` raw / `T2` target / `T4` fret-slide | **target+duration** (lands exact) | neg-instr-byte |
| pulse | pulsetable signed steps | n/a (Amiga) | pulse table add-to-pw | auto-sweep + **pulse-arp** | speed+dir |
| filter | filtertable, global+bitmask | n/a | filter table, global+bitmask | filter-program engine | — |
| drum | wavetable **absolute notes** | **PCM sample** | wavetable abs-notes; d13 noise-at-start | **wf-table + freqhi-table + gate-mask**, noise-tik | **noise + freqhi-decrement** |
| SWEEP/skydive | (pitch via speedtable) | timed pitch-bend (period) | d13 **"Dive"** instr flag | **"Tonesweep up"** (freqhi −1/f) | **"skydive"** (freqhi→0) |
| tie/gate | HR-timer `$40` legato, `3 00` tie | nonzero-note retrig; `E8` sustain | tie/gate markers, HR table | typed note bytes; gate-mask | **append note** = no retrig |

**★ NOVEL frontier — mechanisms that do NOT reduce to {PLAIN, OCTAVE, ARP, SLIDE, VIB, SWEEP, seg}**
(these are where encoding RESID stays non-zero; each is a candidate primitive for the RESID=0 program):
1. **Note-onset noise transient ("noise-tik")** — MoN `fx3 $80` / SF2-d13 / Hubbard / *Facemorph*. A
   brief noise burst at a *pitched* note's attack. → handled by the control-aware noise rule above
   (noise frame = timbre, pitch from pitched frames), NOT a new pitch primitive.
2. **Target+duration glide** — MoN computes (target − current)/duration and lands *exactly* on the
   target note after N frames. The rate-only SLIDE can't reproduce this precisely → **SLIDE needs a
   target+duration form**.
3. **Sine / curved vibrato with onset-delay + finite length** — MoN (sine LUT; Cybernoid2 interpolates
   between adjacent note freqs). → **VIB needs delay+length(+shape)**, not just depth+rate.
4. **Pulse-width arpeggio** — MoN `fx3 $08` cycles PW per frame (timbre, not pitch). → the **PW
   channel**, out of pitch scope.
5. **Sample-table-position scrub** — FC/Hippel `E5/E6`: a moving window walks the waveform/sample at a
   signed per-step increment (a timbral wavetable-index sweep). Amiga-side, but the *concept* (a
   per-frame wavetable-index ramp) can appear in SID wavetable runs → a **wavetable-index** descriptor.
6. **Engine-baked "Dive" / auto-triggered sweep** — SF2-d13 `$40`: a SWEEP applied automatically per
   note as an instrument property (no command). → SWEEP with an auto-trigger-at-onset flag.
7. **Cymbal / "dual" FX** — DMC FX high-nibble (UNVERIFIED — primary doc unreachable; flag, don't model).

Reassuringly **inside** the set: Hubbard octave-arp=OCTAVE, skydive=SWEEP, noise+freq-drop drum =
noise + SWEEP; GoatTracker/SF2 wavetable-absolute-note drums = absolute-note waveform runs; all the
triangle vibratos and 16-bit-accumulator portamentos. **Strategy note:** re-validate the novel list
against the full 52k RESID audit when it lands — only items that actually leak to RESID at scale earn
a new primitive; the rest are confirmations.

## SoundMonitor (Chris Hülsbeck "Musicmaster", 1986) — a FREQ-DOMAIN sweep engine

Engine of a large RESID share (Danko_Tomas, Gilmore_Adam; `sidid` "Soundmonitor"). One of the earliest
C64 editors (64'er magazine 10/1986, type-in listing) — **pre-tracker, so its ornament is raw
frequency-register manipulation, not note-relative semitone tables.** Web sources give the editor
structure; the per-frame mechanism is reverse-engineered from the register output (`audit/probes/
resid_trace.py` on `Danko_Tomas/Howard_Jones`):

- **Editor structure (web, medium confidence):** bars linked in a track/step table (per step:
  tempo/length/volume/fade-out); per cell: bar + transpose + instrument; **per note: instrument index
  + a 4-bit flag nibble = {portamento, transpose-disable, arpeggio, soundtranspose}.** Effects:
  transpose, detune, portamento, vibrato, PWM, filter modulation, **arpeggio (the first editor to have
  it)**.
- **The core ornament = a LOOPING freq-domain sweep (empirical, high confidence).** The "arpeggio" and
  drum effects are produced by **decrementing the 16-bit freq register by a CONSTANT step each frame**,
  cycling. Measured: a pitched "arp" runs `fn = 9378,8754,8130,7506,…642` (**−624/frame, exact**) then
  RESETS to 9378 and repeats (period 15) — a freq-domain SAWTOOTH. In semitone space this reads as an
  accelerating descent `[-1,-2,-4,-8,…-48]` that jumps back, which is why every SEMITONE-domain
  primitive (ARP/SLIDE) misses it. Drums = the same constant-Δfreq decrement **on the NOISE waveform**
  (`fn = 25405,24893,23871,…` swept noise) — a snare/tom.
- **Universal-primitive mapping:** a **freq-domain SWEEP primitive `(start, Δfreq/frame, length,
  loop_period)`** captures both the pitched looping arp (loop_period set) and the one-shot noise drum
  (no loop, noise waveform). This is the §6 freq-domain SLIDE/
  SWEEP, EXTENDED with a loop period (the SoundMonitor arp) — and it must run on noise frames too
  (waveform-agnostic). NOT a new family; the freq-domain sweep already on the frontier, with looping.

## System6581 — note-relative arp tables + periodic noise-tik accent

Engine of Moppe's RESID (`sidid` "System6581"); **no surviving web documentation** — reverse-engineered
entirely from register output (`resid_trace.py` on `Moppe/Wow_Man_Dig_That_Funky_Bassline`). Unlike
SoundMonitor, this IS a note-relative (tracker-style) engine:

- **Core ornament = a note-relative ARP offset-cycle (chord), period ~3** — measured `[+5,+2,0]`
  (`fn` cycles `10207,8583,7647` exactly) and `[-5,-8,-10]` (transposed), i.e. chord voicings stepped
  one entry/frame, as in every tracker (mechanism A).
- **The twist (why it leaks to RESID): a periodic gate-retrigger + NOISE-TIK accent interleaved INTO
  the arp cycle.** Each cycle inserts a `gate-off (waveform=none)` frame then a `noise-tik (0:N)` frame
  — measured `5:P 2:P 0:P 5:P 5:- 0:N` repeating: the `5:-` (idle) + `0:N` (noise accent) are the
  engine's per-cycle re-attack/percussion layer (the MoN/SF2 "noise-tik" mechanism). These non-pitched
  frames break the PITCHED-ONLY period detection → the clean period-3 arp is mis-read as
  "wide-irregular". Drum hits (noise + wide freq, e.g. `-26,-41` with `fn` 3034/1275) are also
  interleaved on the same voice.
- **Universal-primitive mapping:** ordinary **ARP** (offset-cycle codebook) — the fix is **control-aware
  arp detection that treats the periodic gate-off/noise-tik frames as part of the cycle's accent layer**
  (carry them, detect the period over the full cycle incl. accents), NOT a new primitive. Confirms the
  noise-inclusive + control-aware ARP fix (§3/§7 of the impl spec) and the noise-tik primitive.

## Auto-profiled engines (residue trace targets) — reverse-engineered from register output

Profiled with `audit/probes/resid_engine_profile.py` (fits each RESID note to the parametric model
library, aggregates per `sidid` engine). No/thin published docs for these — the technique is read from
the register output; confidence medium (per-note shapes, corroborated by recurrence). **Every one
reduces to the existing universal primitives — no new family.**

| engine | composer(s) | recognised technique(s) | universal primitive |
|---|---|---|---|
| **SoedeSoft** | Danko, Moppe | 2-step arp `p=2` + freq-sweep `d≈-1024` + noise drums | ARP / SWEEP / PERC |
| **Music_Assembler** | Bakker | arp `p=1/2` + up-sweep `d≈+4096` + drums | ARP / SWEEP / PERC |
| **AMP** | Bakker | drum-dominated (noise stamps), small arp-accent | PERC (stamp) / ARP |
| **DMC** (Demo Music Creator) | Bakker | **slow** freq-sweep `d≈-136` + arp `p=2` + arp-accent | SWEEP / ARP |
| **GMC/Superiors** (Game Music Creator) | Dalton | octave/2-step arp `p=2` + drums | OCTAVE-ARP / PERC |
| **Adam_Gilmore** (custom) | Gilmore | **octave-arp `[0,12,-12]` with a DRIFTING wide element** (`-51→-48→-44…` a slid 4th entry) + perc-sweep | ARP + per-element SLIDE / PERC |
| **RoMuzak, Electrosound, SidTracker64, Groovy_Bits** | various | small samples in the rung; arp/perc dominant (profile to confirm) | ARP / PERC |

Notable: **Adam_Gilmore's wide-arp-with-drift** is the clearest "new-looking" case but is just an
octave-arp whose one wide wavetable entry is portamento-slid across triggers — encode as an ARP whose
element carries a SLIDE, or a wildcarded stamp (the gesture recurs, the wide element varies). **MoN /
FutureComposer** (Dalton, Moppe, Tron) is already in the frontier: **target+duration glide** (asymptotic
freq approach — the `[46,-12,-14,-17,-21,…]` ramps), noise-tik, sine vibrato. The auto-profiler is the
standing instrument: run after each primitive, drive per-engine `UNRESOLVED`→0; a persistently-high
engine = an un-RE'd technique to add a fitter (and a primitive) for.

**All confirm the collapse hypothesis: no new primitive families.** SoundMonitor →
freq-domain SWEEP with a loop period; System6581 → control-aware ARP that includes the noise-tik/
retrigger accent. Both were "genuinely-irregular/wide" only because the detectors gated on
semitone-domain + pitched-only frames; the engines' own abstractions are a looping freq-sweep and a
chord-arp-with-accent. (Sourcing: SoundMonitor editor structure from c64-wiki / vgmpf / namelessalgorithm;
per-frame mechanism for both from register-output reverse-engineering — System6581 has no other source.)

## References

- **SoundMonitor** (Chris Hülsbeck "Musicmaster", 1986; 64'er magazine 10/1986 type-in listing):
  editor structure from C64-Wiki <https://www.c64-wiki.de/wiki/Soundmonitor>, VGMPF
  <https://www.vgmpf.com/Wiki/index.php?title=Soundmonitor>, namelessalgorithm
  <https://www.namelessalgorithm.com/computer_music/blog/soundmonitor/>, CSDb release
  <https://csdb.dk/release/?id=59929> (no byte-level format published). Per-frame mechanism
  (looping freq-domain sweep) reverse-engineered from register output: `Danko_Tomas/Howard_Jones.sid`.
- **System6581**: no surviving documentation found (web search empty); format reverse-engineered
  entirely from register output: `Moppe/Wow_Man_Dig_That_Funky_Bassline.sid` (note-relative arp + per-cycle
  noise-tik). `sidid` config `/scratch/anarkiwi/sidid/sidid.cfg`.
- **C=Hacking Issue 5** (1993), *"Rob Hubbard's Music: Disassembled, Commented and Explained"* by
  Anthony McSweeney — a commented disassembly of Hubbard's first routine (used in ~30 tunes incl.
  *Commando*, *Monty on the Run*, *Crazy Comets*). Authoritative primary source for the Hubbard family:
  50 Hz pattern format, the 4-byte note spec (length + flag bits incl. **bit 6 legato/appended**,
  optional instrument-or-portamento byte, semitone pitch), the 8-byte instrument (PW, control, ADSR,
  **vibrato depth**, **pulse speed**, **fx byte**: bit 0 drum, bit 1 skydive, bit 2 **octave arp**),
  and the per-frame NoteWork/SoundWork loop (vibrato, pulse sweep, portamento, fx).
  <http://www.ffd2.com/fridge/chacking/c=hacking5.txt> (local copy: `/scratch/tmp/chacking5.txt`).
- **defMON** (tracker; Python reimplementation): `/scratch/anarkiwi/pydefmon` —
  `pydefmon/pydefmon/defmon_player.py` (`_pitch_slide_voice`, `_ps_voice` PW sweep,
  `_cutoff_slide_step` + ACID/RE/FV/CP columns, sidTAB cascade), `defmon.py` (note/freq tables, row
  format), tests under `pydefmon/tests/`.
- **SID Wizard** (tracker; Python reimplementation): `/scratch/anarkiwi/pysidwizard` —
  `src/pysidwizard/player.py` (`_tick_wf_table`, `_tick_vibrato`, `_tick_portamento`, `_tick_pw_table`,
  `_tick_filt_table` + filter-controller logic, `_compose_sid_writes`), `model.py` (instrument layout:
  `wf_table | pw_table | filter_table`), tests incl. `test_player_filter_controller.py`.
- **Rob Hubbard — *Commando*** (hand-written 6502; reverse-engineered disassembly):
  <https://gitlab.com/ricardoquesada/c64-commando-2084/-/blob/orig/src/music.asm> (`FREQ_TBL`, per-frame
  freq writes, portamento). A `f####`-labelled disassembly — treat internal detail as lower-confidence
  than C=Hacking #5 / the trackers.
- **Martin Galway — Ocean drivers**: <https://github.com/MartinGalway/C64_music> (e.g. `wizball.asm`,
  `greenberet.asm`) — `LoFrq/HiFrq`, the `FOL*` offset-list arpeggio, the shared gradient-envelope
  mechanism (`*G0..G3` values + `*D0..D3` durations + delay) reused for vibrato / PW / filter, and the
  `FilterChannel` global-filter controller. (Label-level detail via summary; treat as corroborating
  the structural patterns, not exact byte layouts.)
- **JCH NewPlayer (20.G4)**: file-format/memory-layout + sequence `AA/BB` commands at
  <https://codebase.c64.org/doku.php?id=base:jch_20.g4_player_file_format> (incomplete on table
  mechanics). Authoritative readable spec is **CheeseCutter** (the modern editor for this exact format):
  <https://github.com/theyamo/CheeseCutter> + guide <https://carol6502.neocities.org/c6_ccutter_guide>
  (wave/chord/pulse/filter table byte ranges, command numbers). Original JCH editor v3.04 + source:
  CSDb <https://csdb.dk/release/?id=14037> (`v-c64ed.zip`). Table-format corroboration also in
  Chordian's SID Factory II (JCH/Laxity drivers). Engine of *Baggis* (Goto80) and *Camerock* (DRAX) —
  confirmed via `sidid` (config `/scratch/anarkiwi/sidid/sidid.cfg`; `SIDIDCFG=… sidid <dir>`).
- **GoatTracker 2** (official format docs): <https://github.com/leafo/goattracker2/blob/master/morphos/goattracker.guide> · GT2 doc mirror <https://github.com/jpage8580/GTUltra> — wavetable rel/abs notes, speedtable vibrato, `1/2/3` portamento, pulse/filter tables, wavetable-absolute-note drums, HR-timer `$40` legato / `3 00` tie.
- **Future Composer / Jochen Hippel** (Amiga sample engine — NOT SID; period-domain): <https://github.com/mschwendt/libtfmxaudiodecoder> (`src/Jochen/FC.cpp`, `Instrument.cpp`, `Vibrato.cpp`, `Portamento.cpp`). Novel: `E5/E6` sample-table-position scrub; octave-scaled triangle vibrato. (NB "TFMX" here = Hippel's, distinct from Hülsbeck's TFMX.)
- **SID Factory II** (Laxity/JCH lineage; official driver notes): <https://github.com/Chordian/sidfactory2> (`dist/documentation/notes_driver11.txt`, `notes_driver13.txt`, `notes_driver14.txt`) — d11 `T0–T4` slide/porta/arp/vib/fret-slide; d13 "The Hubbard Experience" adds the **"Dive"** sweep flag + **noise-at-start** flag; wavetable `80–df` absolute notes "for e.g. drums".
- **Maniacs of Noise** (Jeroen Tel / Charles Deenen; commented disassemblies): <https://github.com/realdmx/c64_6581_sid_players> (`Tel_Jeroen_MON/...Cybernoid2.asm`, `Deenen_Charles_MON/...SFX_Player.asm`). Novel: **target+duration glide** (lands exact), **sine-LUT vibrato** (delay+length), **pulse-arpeggio** (`fx3 $08`), **noise-tik** onset transient (`fx3 $80`), **"Tonesweep up"** (freqhi −1/frame). Same repo: Hubbard *Monty* (`instrfx` bit0 drum / bit1 skydive / bit2 octave-arp) and many more disassemblies (Galway, Fred/Matt Gray, Whittaker, Ouwehand, …) for future expansion.
- **SID wavetable drum technique** (general): <http://www.ucapps.de/howto_sid_wavetables_1.html>.
- **DMC (Demo Music Creator)** — secondary only; primary docs (tnd64) blocked from this environment. FX high-nibble reportedly has "cymbal"/"dual" effect flags (UNVERIFIED — needs a DMC4 player disasm).
- **Local reimplementations** (read for the hard-restart / control-register insights): `/scratch/anarkiwi/pysidwizard/src/pysidwizard/player.py` (`HR_FRAMES`, `PRE_HR_LEAD_FRAMES`, gateoff wf/pw/filt, chord `$7E/$7F`), `/scratch/anarkiwi/pydefmon/pydefmon/defmon_player.py` (gate/waveform flicker mid-note, PS sweep, TR transpose).
- **Concrete noise-accent example**: Wiklund *Facemorph* (`/scratch/preframr/hvsc/MUSICIANS/W/Wiklund/Facemorph.sid`) — per-note `tri-setup → noise-tik accent → pulse melody`.
