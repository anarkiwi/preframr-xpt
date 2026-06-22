# GoatTracker Decompiler Codec — complete generator-recovery design (all lanes)

**Status: REFUTED / SUPERSEDED (2026-06-20).** The GoatTracker-as-target direction is a dead end (one
driver's 96-note grid + tempo quantization + 255-row tables = the wrong cage; freq ~85% off-grid,
residual ~0.84). The codec that landed instead is the OWN step/tracker representation (`../encoding/
sid_player_decompiler.md` "HOW IT LANDED"). Kept for the driver-survey record only.

**Original (DESIGN — for subagent execution, 2026-06-17).** This replaces the event codec's per-frame
ornament handling with a **generator-recovery** model: decompile each SID register log into a **GoatTracker
2 `Song`** (instruments + wave/pulse/filter/speed tables + a sparse note score / orderlist) that renders
**byte-exact** (via `pygoattracker`'s bit-exact player) and is **note-rate sparse**. **Read §0.5
FOUNDATION first — the decoder/forward-model and editable endpoint already exist as the operator's
`pygoattracker`; this codec is its missing inverse (register log → `Song`).** Motivation, evidence, and the decision to commit are in
`ornament_generator_recovery.md` (read it first). Scope: single-speed, non-digi (~92% of corpus); the
fidelity target is the existing `canonical_writes(ow)` oracle (chip-equivalent, validated at the reSID
noise floor). "Replace whatever's needed" — keep only what is genuinely right (the canonical-writes
oracle, note-index/note-table pitch, the byte-exact self-verify gate, the constrained-decode idea);
everything about how ornaments are represented is redesigned.

## 0. Principles (in priority order)

1. **Lossless / byte-exact.** `decode(encode(ow)) == canonical_writes(ow)`, self-verified at encode,
   fail-loud. Non-negotiable. A lossless **residual lane** guarantees this during development; the design
   goal is to drive residual → 0.
2. **Sparse to note rate.** The token body sits at ~10–14% of frames (one small group per *note*, not per
   frame). Ornaments are *referenced*, never spelled per-frame.
3. **Universal, no per-driver branching.** One instruction set (the virtual tracker) compiles every
   in-scope tune. No Crowther/driver-specific code (consistent with the standing "universal driver"
   principle).
4. **Learnable.** Every token is a meaningful musical/structural choice; structure fits in a ~1024 window.
   This is the whole point — the LLM learns the *program*, which is the sparsest and most musical form.
5. **MDL-principled.** The encoder minimizes total description length = instrument-bank bits + score bits
   + residual bits. The "best parse" is the shortest lossless program.

## 0.5 FOUNDATION (decisive) — target GoatTracker 2 via `pygoattracker`

**Do not invent a virtual tracker. The canonical model IS GoatTracker 2, and the operator's own
`pygoattracker` (https://github.com/anarkiwi/pygoattracker) already provides the bit-exact forward model
and the editable endpoint.** This collapses most of the risk in this design:

- **`pygoattracker` is a pure-Python, tick-for-tick BIT-EXACT GoatTracker 2 playroutine** (sequencer,
  funktempo, wave/pulse/filter table execution, **speedtable vibrato + portamento incl. note-INDEPENDENT
  speeds**, gateoff timer, hard restart) — single-speed, non-digi, exactly our scope. It is torch-free,
  tested (85% cov).
- **The DECODER already exists — it is `pygoattracker.Player`.** We do NOT write or invent a decoder:
  `Player(song).play_frame()` → one PAL frame's register writes in register order; `iter_register_writes(
  song, until_loop=True)` → the whole log; it clocks each write at the in-frame offsets a register log
  uses. So our fidelity target becomes: a `Song` whose bit-exact playback equals the input log.
- **The free-running phase question (§1.4) is ANSWERED** — GoatTracker's vibrato/portamento are driven by
  the **speedtable with note-independent speeds**, already implemented bit-exact. We match GoatTracker's
  table semantics rather than reverse-engineering a phase rule; correctness is checked by playback.
- **The data model is GoatTracker's typed `Song`** — `Song, Instrument, Pattern, Row, Wavetable,
  Pulsetable, Filtertable, Speedtable, Orderlist` (+ `constants.note_value`), constructible in Python. This
  is already sparse + tracker-native (pattern rows = notes; per-frame programs live in the 4 tables).
- **The endpoint is free** — `write_sng(song)` emits a real GTS5 `.SNG` the operator can open and edit in
  GoatTracker; the generation goal ("phrase → arranged SID tune") lands as an editable module.
- **The only missing piece is the INVERSE** — `pygoattracker` has read/write/play/render and `gt_to_nt2`,
  but **NO register-log → Song importer**. That importer (the *decompiler*) is exactly this codec's job.

**So this codec = the missing inverse of `pygoattracker.Player`**: register log → GoatTracker `Song`
(+ our learnable token serialization of that Song), such that `Player(Song)` reproduces the log.
`pygoattracker` is BOTH the decoder we target AND an independent oracle. Everything below — the model,
the lanes, the encoder, the residual, the phasing — is re-anchored to GoatTracker's `Song`. (The bespoke
"virtual tracker" / SID-Wizard framing in earlier drafts is SUPERSEDED by this where they differ; SID-
Wizard `.swm` export via `log_to_swm_recompiler_design.md` remains an optional secondary export.)

**Caveat (the same expressibility boundary as any recompiler):** our corpus tunes were authored in *other*
drivers (Hubbard/Daglish/Follin, not GoatTracker), so log → `Song` is a re-render *equivalence*
(many-to-one), not the original program. GoatTracker's 4 tables + effects are highly expressive, so the
expressible subset is large; anything outside it goes to the **residual lane** and is REPORTED (never
silently approximated). Residual fraction by tune/driver is the headline quality metric.

## 1. The model: GoatTracker's `Song` (decoder = `pygoattracker.Player`)

The model is GoatTracker 2's `Song`; the **decoder is `pygoattracker.Player` (given, bit-exact)**; the
**encoder inverts it** (the build); the **token stream serializes the `Song`**. The structures below map
1:1 onto pygoattracker's typed classes — build those objects, don't reinvent them.

### 1.1 Program structure
```
PROGRAM = {
  header:   per-voice { tuning, note_table },          # pitch grid (keep existing recovery)
  instruments: [ INSTRUMENT, ... ],                     # the bank (DEF), per tune
  globals:  GLOBAL_PROGRAM,                             # filter/volume automation
  score:    per-voice [ NOTE, ... ],                    # the sparse body
  residual: [ (frame, reg, value), ... ],               # EXTERNAL post-playback patch (see note)
}
# RESIDUAL ARCHITECTURE (clarified by Phase 0): GoatTracker has NO arbitrary-write escape hatch — the
# Player only renders the Song. So the residual is NOT inside the Song; it is an EXTERNAL corrective
# write-stream applied AFTER playback: decode = canonicalize(pygoattracker.iter_register_writes(song))
# THEN overlay residual -> canonical_writes(log). This preserves byte-exactness for the genuinely
# GoatTracker-inexpressible bits (PW LSB &0xFE / odd PW, vol automation beyond DXY, any freq porta/
# speedtable can't hit exactly, control bytes >=0xE0 colliding with wavetable command ranges). The
# residual is a small learnable side-channel in the token stream; the .SNG export drops it (lossy for
# human editing, exact for our codec). Headline science = drive the residual patch toward empty.
INSTRUMENT = {
  wave:  TABLE over the control/waveform register,      # wavetable walk (waveform+test+sync/ring bits)
  ad, sr: envelope values + hard-restart prep,          # onset-static + HR
  pitch: { glide?: RAMP, vibrato?: LFO },               # freq ornament programs
  pw:    { base, pwm?: LFO|TABLE },                      # pulse-width program
  arp?:  TABLE of note-index offsets,                   # arpeggio
}
NOTE = { note_index, duration, instrument_ref, overrides? }
```

### 1.2 The universal primitive: indexed loopable TABLE
Every ornament is one primitive — a table read by a per-frame counter:
```
TABLE = { values:[...], loop_start, advance, phase_model, delay }
  value at step k:  k < delay -> inactive (no contribution / hold)
                    else let j = k - delay
                         idx = j if j < len else loop_start + (j-len) % (len-loop_start)
                         -> values[idx]
  advance:     +1 per frame (default); other rates allowed (rare).
  phase_model: NOTE_RESET (counter=0 at onset) | FREE_RUNNING (counter from a global rule, §1.4)
```
This single primitive expresses: **wavetable** (target = waveform/ctrl bits, mode SET), **arpeggio**
(target = note-index, mode ADD-to-note), **vibrato** (target = freq, mode ADD; periodic table = LFO),
**PWM** (target = pw, mode ADD or SET), **filter sweep** (target = cutoff, mode SET/ADD). An **LFO** is a
TABLE whose values loop with period = rate and amplitude = depth; recognized + stored parametrically when
that is shorter (`{shape, rate, depth, delay}`), else as raw table values. Same decode either way.

### 1.3 Per-frame render (the decoder = `pygoattracker.Player`, NOT reimplemented)
**The decoder is `pygoattracker.Player.play_frame()` / `iter_register_writes(song)` — bit-exact, given.**
Do not reimplement it. The encoder's job is to produce a `Song` whose playback equals the log. The
GoatTracker render semantics it implements (kept here only so the *encoder* knows what it must invert) —
per frame `f`, per voice `v`, with `k = f - onset(currentnote)`:
```
note_idx_eff = note_index + arp.value(k)                      # arp shifts the note (note-grid)
freq = note_freq(note_idx_eff, tuning, note_table)
     + glide.value(k)                                         # attack portamento, freq delta
     + vibrato.value(phase(f))                                # LFO, freq delta
pw   = pw_base + pwm.value(phase(f))    | pwm.value(k) if TABLE/SET
ctrl = wave.value(k)                    # waveform nibble + test/sync/ring bits
       with gate bit = note gate state (on for [onset, onset+duration), off after)
ad,sr = instrument envelope (written at onset; hard-restart writes at onset-hr_k, value-exact)
```
Globals (filter cutoff/res/routing, mode/vol) render from `GLOBAL_PROGRAM` (same TABLE primitive on the
global lane). Same-value-rewrite dropping + ordering follow the existing `canonical_writes` rules so the
output is the exact canonical write set. **Cross-voice wiring** (sync/ring, ctrl bits 1/2): the bit lives
in `wave` (ctrl) and is keyed by physical voice index — never separated from its carrier; a modulating-
but-silent voice is never dropped.

### 1.4b Grounding — see §0.5 (GoatTracker via pygoattracker is the target)
The model above maps onto GoatTracker's `Song` (§0.5); `pygoattracker.Player` is the bit-exact decoder +
oracle, `write_sng` the editable endpoint. Adopt the recompiler's two-path phasing: **Path A = a brute
`Wavetable` walking the exact per-frame sequence** (lossless-but-ugly, the residual floor, Phase 0/1) →
**Path B = structured recovery** (instruments/arps/slides/vibrato/sweeps via the proper tables + effects,
Phase 2+), choosing B per span where exact, else A — verified by playback, not guessed. SID-Wizard `.swm`
export (`log_to_swm_recompiler_design.md`) is an optional *secondary* endpoint, not the primary target.

### 1.4 Free-running phase (the load-bearing detail)
Vibrato/PWM are often **free-running** — phase tied to a global counter, not reset per note (measured:
two notes 192 frames apart share phase; `f mod period` alone does NOT fit). The decoder maintains, per
instrument program flagged FREE_RUNNING, a counter advancing by a **recovered rule** (origin + advance,
possibly gated to "advance only while a note using this instrument is sounding"). `phase(f)` for a note =
that counter's value at the note's active frames. The encoder recovers the rule that reproduces every
note's trace losslessly with **zero per-note phase bits**. NOTE_RESET programs reset to 0 at
onset(+delay). The encoder picks, per program, whichever phase_model is lossless and shorter. **Largely
ANSWERED by GoatTracker's `Speedtable` (note-independent vibrato/porta speeds), already bit-exact in
`pygoattracker` (§0.5): match its semantics, verify by playback — don't reverse-engineer from scratch.
Still the area to validate first on the vibrato lane (Phase 2).**

## 2. The data the encoder must recover (per lane)

| lane | register(s) | program | notes |
|---|---|---|---|
| pitch-note | freq (settled) | `note_index` + `note_table` | keep existing recovery |
| pitch-arp | freq | `arp` TABLE (note-index offsets) | freq hops between exact note-grid values, periodic |
| pitch-glide | freq | `glide` RAMP | attack transient, one-shot, before vibrato delay |
| pitch-vibrato | freq | `vibrato` LFO/TABLE | periodic residual after note+arp+glide removed; delay+depth+phase |
| pulsewidth | pw (12-bit) | `pw.base` + `pwm` | PWM periodic; `_PMAX` cap removed (PWM period up to ≥64) |
| timbre | ctrl/waveform | `wave` TABLE | per-frame waveform walk + test/sync/ring bits + gate |
| envelope | ad, sr | onset values + hard-restart | mostly static per instrument |
| global | cutoff/res/route/vol | `GLOBAL_PROGRAM` TABLE(s) | filter sweeps = same primitive |

## 3. Encoder algorithm (trace → GoatTracker `Song`) — the decompiler, the whole build

The encoder builds `pygoattracker` typed objects (`Instrument`, `Wavetable`/`Pulsetable`/`Filtertable`/
`Speedtable`, `Pattern`/`Row`, `Orderlist`, `Song`) such that `iter_register_writes(song)` ==
`canonical_writes(log)`. Lane → GoatTracker table mapping:

| our lane (§2) | GoatTracker mechanism |
|---|---|
| waveform/ctrl walk | `Wavetable` (waveform + note/arp bytes) |
| arpeggio | `Wavetable` note-relative bytes, or `0XY` arpeggio effect |
| vibrato | `Speedtable` + `4XY` vibrato (note-independent speed = free-running) |
| glide / portamento | `1/2/3XY` porta effects (+ `Speedtable`) |
| pulse-width / PWM | `Pulsetable` |
| filter sweep | `Filtertable` |
| envelope + hard restart | `Instrument` (ADSR, wavetable ptr, gateoff timer, hard restart) |
| notes / duration / order | `Pattern` `Row`s + `Orderlist` |

Per voice, then global. Greedy-then-refine MDL; each step has a lossless residual fallback (a `Wavetable`
that walks the exact per-frame sequence is the Path-A floor — always expressible, always exact).

1. **Note segmentation.** Gate 0→1 = onset; note span = [onset, next-onset or gate-off]. `note_index` at
   onset from the freq lane (existing pitch recovery). Reuse `_typed_cas` onset/duration logic.
2. **Per-note lane extraction.** For each note, the per-frame settled (freq, pw, ctrl, ad, sr).
3. **Freq decomposition (ordered — this is the composite the prototype missed):**
   a. **arp**: detect frames where freq equals an *exact note-grid* entry offset from the base — a periodic
      offset pattern → `arp` TABLE. Remove (work in note-index space).
   b. **glide**: the leading attack transient (monotone approach from the previous pitch, before steady
      state) → `glide` RAMP. Remove.
   c. **vibrato**: the remaining steady periodic freq residual → `vibrato` LFO/TABLE {delay, depth, table,
      phase_model}. Whatever is left after a/b/c is the residual (target 0).
4. **PW**: `pw_base` + periodic residual → `pwm`. Remove `_PMAX=32` cap.
5. **Waveform/ctrl**: the per-frame ctrl sequence (mask off gate) over time-since-onset → `wave` TABLE +
   loop. Sync/ring/test bits travel in the table, voice-keyed.
6. **Envelope**: onset AD/SR + hard-restart prep (existing `_fold_envelope` logic, generalized).
7. **Instrument clustering / bank.** Two notes share an instrument iff their *programs* (wave, vibrato,
   pwm, arp, envelope — modulo the note's own pitch/duration/phase) are identical. Build the per-tune bank;
   a note → `instrument_ref`. (Generalizes the v3 front-loaded instrument bank to carry the per-frame
   programs, not just the static onset patch.) Use MDL: define a program once, reference it ≥2×.
8. **Free-running phase recovery (§1.4).** For each periodic program, fit the global counter rule (origin
   + advance + gating) that makes all referencing notes' phases consistent; flag NOTE_RESET vs
   FREE_RUNNING by which is lossless + shorter.
9. **Globals.** Same recovery on the filter/volume lane → `GLOBAL_PROGRAM`.
10. **Residual.** Any (frame,reg,value) the rendered program doesn't reproduce exactly → residual lane.
    **Encode self-verifies `render(program) == canonical_writes`; residual makes that always pass.**
    Track residual fraction per tune as the headline quality metric (target → 0).

MDL objective: minimize bank-bits + score-bits + residual-bits. Start greedy (recover per-lane, cluster,
measure); add the optimal-parse DP later only where greedy leaves residual.

## 4. Token grammar / serialization (sparse, learnable, byte-exact)

```
HEADER    : per-voice [tuning][note_table]
BANK      : [INSTR_DEF n] then n instruments, each = its programs serialized
            (TABLE = [target][mode][phase_model][delay][len][loop][values...]; LFO = [shape][rate][depth][delay])
GLOBALS   : filter/vol programs
BODY      : per-voice note score. NOTE = [note_index Δ][duration][instr_ref][overrides?]
            Body order: TRACK-MAJOR (each voice's notes contiguous) — proven to localize melodic
            structure (`melody_timbre_factorization.md`); merge-sort by onset at decode.
RESIDUAL  : [RESID k] then k explicit (Δframe, reg, value) writes  (target: absent)
```
Atom design: typed nibbles + varints, BE, as today. Vocab stays small + fixed; ids positional. A note is
now ~3–5 atoms; a whole tune is ~note-count × that ≈ note-rate sparse. `EVENT_FORMAT_VERSION`/
`ATOM_CACHE_VERSION` bump (fresh format). Constrained-decode grammar mask enforces define-before-
reference + the program/score/residual structure (extend `EventStreamState`).

## 5. Fidelity contract + gates (every phase)

- **HARD: byte-exact via the real player** — `canonicalize(pygoattracker.iter_register_writes(
  encode_to_song(ow))) == canonical_writes(ow)` on the 5 reference drivers (grid_runner, commando,
  camerock, trap, baggis) + a corpus sample. Two independent renderers (pygoattracker's player and our
  canonical-writes oracle) must agree. A single non-exact tune fails. The residual lane (a brute
  `Wavetable`) guarantees exactness; the *science* is residual → 0. Token round-trip is separate and also
  exact: `deserialize(serialize(song)) == song`.
- **Residual fraction** (frames needing explicit writes / total) — the headline quality metric per lane;
  report it, drive it down. A lane is "recovered" when its residual ≈ 0 across the corpus.
- **Sparsity**: event-frame density of the body (`event_sparsity.py`-style) → toward the 10–14% note floor;
  report per lane as it lands.
- **Raw-vs-canonical render** at the reSID noise floor (unchanged — same oracle).
- Repo lint (no narrative `#`, docstrings ≤5 lines, black, xdist green).

## 6. Build order (phased; each phase byte-exact + residual-measured before the next)

Implement in **preframr-tokens**, new module `events/` (or a fresh `vtracker/`), off `main`. The residual
lane is built FIRST so every intermediate state is byte-exact (residual just starts at 100% and shrinks).

- **Phase 0 — skeleton + contract (NO decoder to write — pygoattracker is it).** Wire `pygoattracker`
  (read/write/Player/iter_register_writes); build the round-trip harness `canonicalize(iter_register_writes(
  song)) == canonical_writes(ow)`; build the **Path-A brute-force `Wavetable`** encoder — one instrument per
  voice whose wavetable walks the EXACT per-frame (waveform, note+detune) sequence, pulsetable/filtertable
  walk exact PW/filter, one pattern row per note. Trivially exact, exercises the whole pipeline. Gate:
  byte-exact on 5 drivers via the real player. This nails the contract + the dependency + token
  serialization of a `Song`.
  **DONE 2026-06-17 (`feat/gt-decompiler` @ `3c159a4`, `preframr_tokens/gtcodec/`, 16 tests green).
  Findings:** ctrl/waveform ~97% byte-exact, pw ~75% (residual = PW LSB `&0xFE`/odd PW), filter ~85%
  (residual = vol automation > DXY), adsr median 0.024 (onset/HR framing → P1), **freq 0% exact — 85% of
  active frames off GoatTracker's 96-note PAL grid** (vibrato/detune between grid notes + some sub-grid
  bass, e.g. trap v0 freq 19–33 vs table floor 279) = the headline, brute-wavetable hits only grid notes
  → Phase 2's speedtable/porta is the fix. Hard limits: **`MAX_TABLELEN=255` shared table store** caps
  brute per-frame encoding to ~84 frames (~1.7s) on ~80% of tunes (confirms Path A is only a floor;
  structured recovery mandatory); **control bytes ≥0xE0** (e.g. 0xFF all-waveforms) collide with wavetable
  command/jump ranges (8/40 tunes) → need explicit handling/residual; render with `optimize_pulse=False,
  optimize_realtime=False` for the gate. Token round-trip `deserialize(serialize(song))==song` exact.
- **Phase 1 — notes + envelope + static waveform.** Recover note segmentation, pitch (note-index/table),
  static onset patch + envelope, single-step waveform. Move those out of residual. Gate: byte-exact,
  residual drops to the per-frame-ornament fraction.
- **Phase 2 — vibrato (FD) + free-running phase.** The hardest/most valuable lane. NAIL the composite
  decomposition (arp/glide/vibrato §3.3) and the free-running phase rule (§1.4) empirically first, then
  recover. Gate: FD residual ≈ 0; FD density → note rate.
- **Phase 3 — arpeggio (NI) + glide.** Note-index TABLE + attack ramps.
- **Phase 4 — PWM (PW)** (remove `_PMAX` cap) **+ wavetable (ctrl)**: the two remaining dense lanes.
- **Phase 5 — globals (filter sweeps) + cross-voice wiring + instrument clustering polish.**
- **Phase 6 — constrained-decode grammar mask + version bump + full corpus byte-exact + density report.**

Each phase: `encode(verify=True)` green on 5 drivers + corpus sample, residual + density reported, lint/
tests green. Do NOT train (that's the operator's GPU A/B afterward). Commit per phase.

## 7. Hard parts / risks / de-risking

- **Composite freq decomposition (arp+glide+vibrato).** The prototype's single-cell model failed here.
  De-risk in Phase 2 with a focused empirical study (real per-note FD segments) BEFORE coding the
  recovery; the residual lane means partial recovery is still byte-exact.
- **Free-running phase rule (§1.4).** `f mod period` failed; the rule is gated/origin-shifted. Empirically
  fit per instrument; fall back to per-note phase (small, still sparse) if a global rule isn't found.
- **Instrument identity under truncation.** Same program, different note durations cut tables at different
  points — cluster on the *program*, not the realized per-note pattern.
- **Wavetable vs envelope coupling.** Hard-restart writes interleave with waveform walks at onset; keep
  the existing `_fold_envelope` claiming logic.
- **Universality.** Validate on all 5 drivers each phase (different drivers stress different programs);
  the residual fraction by driver flags any un-modeled mechanism (the "residual = unmodeled mechanism"
  principle).

## 8. What is kept vs replaced + dependencies

- **New dependency: `pygoattracker`** (the operator's repo; pure-Python, torch-free) — the bit-exact
  decoder/forward-model, the independent render oracle, the `Song` data model the encoder builds, and the
  `.SNG` editable-export endpoint. Verify it's installable/importable in Phase 0; pin a version.
- **Kept:** `canonical_writes` oracle (the comparison normalizer), `pitch_grid` note-index/note-table
  recovery (maps to GoatTracker notes), `_typed_cas`/`_fold_envelope` onset+envelope logic (informs
  Instrument recovery), the byte-exact self-verify gate, the residual-as-unmodeled-mechanism principle,
  the constrained-decode concept.
- **Replaced:** the per-series `cover()`/`mdl_parse` signal-fitting as the ornament representation (at
  most an internal helper for table detection); the per-frame STEP/RAMP event emission; the bespoke
  virtual-tracker decoder (use `pygoattracker.Player`); the v4 track factoring as top-level structure
  (subsumed — the body is GoatTracker patterns/orderlist).

## 9. After the codec: the payoff experiment (operator-run)

Re-encode the corpus, train the atoms-only baseline (same config), and read the SAME deciders that
diagnosed the problem — de-confounded `copy_novel` novel-content + `free_running_gap` — now on a
note-rate-sparse stream where a window holds real musical structure. Plus the generation-quality gate
(audition + distributional metrics). This is the test of whether sparsity-at-the-generator-level finally
lets the model generate coherent, structured SID music. M0 (notes-only, already coherent) is the floor;
this is the full-character version.

**The generation endpoint is now concrete:** a generated token stream → `Song` → `write_sng()` → an
editable GoatTracker `.SNG` the operator opens in GoatTracker, and `render_wav()` to audition. The
"phrase → arranged SID tune" goal (`../generation/prompt_interface_design.md`) lands as a real,
human-editable tracker module — the input side compiles a phrase to a prompt, the output side is a `.SNG`.
```
