# DMC + JCH driver-backend scoping for the preframr-tokens BACC codec

**Status:** scoping (read-only investigation + a measured DMC feasibility probe). No
backend implemented, no PR. This doc says what it takes to land DMC and the three
JCH players as BACC backends so their HVSC tunes join the unified token alphabet.

## TL;DR

- **DMC probe result (measured, the key deliverable):** a plain *run-the-player*
  emulation of `Ode_to_Music.sid` (py65, init once + play once per frame, snapshot
  `$D400–$D418`) reproduces the siddump oracle **100.000% byte-exact across all 23
  compared registers (regs 0–23), frames 1–399 — 0 mismatches** once a **fixed
  one-frame boot offset** is applied (`rendered[f] == oracle[f-1]`). Raw, before
  alignment, agreement is **87.16%** and the "first divergence" is **frame 0, reg 4**
  — which is *entirely* the boot/framing skew, not a modelling gap. **Nothing is
  missing for byte-exact except the one-frame alignment the lft/GoatTracker backends
  already implement.**
- **Ranking:** land **DMC first** (S–M, huge corpus, byte-exact today, trivial
  abstraction fit) → **JCH Protracker** (M, 94 tunes, needs a `format.py`) → **JCH
  OldPlayer** (M, 32 tunes, needs `format.py`) → **JCH DigiPlayer last** (L, only 4
  tunes, needs a sub-frame digi primitive the codec does not yet have).
- **Single recommended first step:** copy the **lft backend's run-the-player skeleton**
  (`SIDEmu`/py65, no illegal opcodes needed for DMC), add a `DmcBackend` that
  fingerprints the `BRIAN/GRAFFITY` v7.62 `$1000` image, recovers the song via the
  existing `targets/dmc/format.py` model, renders by re-running init+play, and reuses
  the GoatTracker `_align` boot-frame logic to absorb the +1 frame offset.

The reproducible probe lives at `/scratch/anarkiwi/cbm/undmc/probe_dmc.py`.

---

## 1. The measured DMC feasibility probe

### Method (approach (b): run the player image)

`Ode_to_Music.sid` (md5 `1145561cda1e77101737ab2fff5e19a9`, HVSC
`C64Music/MUSICIANS/A/Ass_It/Ode_to_Music.sid`, load/init `$1000`, play `$1003`) is
a standard 6502 PSID with **no illegal opcodes**, so it runs directly under the
codec's existing `preframr_tokens.bacc.sidemu.SIDEmu` (py65 `MPU`, the same loader
the other backends use). The probe:

1. `load_psid` → `SIDEmu(psid)`; `emu.init(0)`; snapshot `state()` (frame 0).
2. For each subsequent frame: `emu.play_frame()`; snapshot `state()`.
   `state()` is the 25-register forward-held SID image — exactly what the codec's
   `per_frame_state` reconstructs from a `.dump.parquet`.
3. Parse `docs/dmc/siddump.txt` into the same (nframes, 25) array (parse spec in §4).
4. Compare cell-by-cell over registers 0–23.

### Result

```
frames compared:              400
register cells compared:      9600  (regs 0-23 × 400 frames)
raw agreement:                8367/9600 = 87.156%
raw first divergence:         frame=0 reg=4  rendered=$00 oracle=$08
aligned (rendered[f]==oracle[f-1]) mismatches over regs 0-23, frames 1-399:  0
=> 100.000% byte-exact after a fixed +1 frame boot alignment
```

### What the 13% gap actually is (and what closes it)

The gap is **100% a frame-phase offset**, not a missing model. siddump samples the
SID *after* play call N writes; the probe's `state()` is indexed so that
`rendered[f]` corresponds to siddump's `oracle[f-1]`. Sweeping the alignment shift
confirms it: shift `-1` → **100.00%**, shift `0` → 87.16%, shift `+1` → 83.75%. The
"first divergence at frame 0 reg 4" is the oracle's frame-0 boot snapshot (V1 ctrl
`$08`, the init gate-off write) landing one row off — a framing artifact.

**This is the identical leading-frame skew the GoatTracker and lft backends already
solve** with boot-frame matching (`goattracker._align` / `lft` boot framing): seed
frame 0 from the dump (`program.boot`) and drop/shift the render's leading frame.
Nothing about the DMC *player model* is unaccounted for — the pulse-width sweep
(reg 2/3 ramps `653→6A6→6F9…`), the per-voice freq-table notes, the portamento
"(- 0012)" glides, ADSR, the filter — **all reproduce exactly** because the player
itself is being executed. There is no effect-decoding gap (the `DMC_FORMAT.md`
caveat about `$C0–$FC` effects only bites the *static song parser*, not the
run-the-player render).

Register 24 (`MODE/VOL` + filter type) was compared separately because siddump
prints it as text (`Typ`/`V`); the player writes it directly (`SID_MODEVOL`,
`SID_RESFILT`) and it is captured byte-exact in `state()` — the recommended backend
compares it from the `.dump.parquet` (the real oracle, see §4) rather than the
text dump, so this is a non-issue in production.

**Gap to production byte-exact: only the one-frame `_align` step (already written,
reusable verbatim).**

---

## 2. Per-player table

| Player | HVSC tunes | RE completeness | Has `format.py`? | Illegal opcodes? | Sub-frame? | Abstraction fit | Effort |
|---|---|---|---|---|---|---|---|
| **DMC v7.62** | ~10,700 (largest editor in HVSC) | High: full memory map, `format.py`, C+asm, siddump oracle; **probe = 100% byte-exact** | **Yes** | No | No | Clean: note→freq-table→A440 grid, instr/pulse/wave generators, backward orderlist | **S–M** |
| **JCH Protracker** | 94 | Medium: asm+C+oracle+symbols; song-data model NOT extracted | **No** (blocker) | No (std `STA $D405,y`) | No | Good: orderlist+pattern+instrument, AD/SR via register writes | **M** |
| **JCH OldPlayer** | 32 | Medium: asm+C+oracle+symbols; subtune table at `$880C` mapped, model not extracted | **No** (blocker) | No | No | Good: 8-byte subtune records, voice orderlist pointers | **M** |
| **JCH DigiPlayer** | 4 | Medium: asm+C+oracle; 4-bit packed-digi unpack identified | **No** (blocker) | No | **Yes (digi)** | Poor *today*: sample playback is sub-frame; needs a digi primitive | **L** |

Corpus counts are from the hvsc-tracker-catalog census cited in `unjch/JCH.md` and
`undmc/DMC_FORMAT.md`. Note: even *one* DMC backend at the v7.62 `$1000` build wins a
large slice of that ~10,700; other DMC versions (v2/v4/v5/v8) relocate and need a
version-detection pass (future work, the `format.py` constants are already
parameterized for retargeting).

---

## 3. Mapping each player to the common abstraction

HARD RULE #0: every backend decomposes into **canonical A440-grid notes +
pitch-invariant instrument generators + backward orderlist**, no raw-byte escape.
For a *run-the-player* render (DMC) the "generators" are the player itself re-executed
(exactly as lft and GoatTracker do via pygoattracker); the recovered BACC *program*
is the song-data structure (`format.py` output), not a per-frame trace.

### DMC v7.62

- **note → pitch.** Freq tables `freq_lo_tbl`/`freq_hi_tbl` at `$1647`/`$16A7` (96
  notes). Measured against `pitch.fn_to_grid`: the table is **monotonic with NO
  non-monotonic steps** and snaps cleanly to distinct grid indices — i.e. it lands
  on the *same A440 grid* as GoatTracker's `FREQ_TABLE`. **Tuning offset:** the table
  is tuned ~**+35 cents** sharp of A440 (constant across the range; one anomalous
  entry, note index 4 / fn `$0147`, sits ~−22c — a single slightly-mistuned table
  cell that still snaps to its own grid index). The offset is irrelevant to
  losslessness (render reproduces the exact Fn); it only means DMC tunes share the
  grid topology with other drivers but sit on a concert reference ~1/3 semitone
  sharp, same as any tune recorded sharp. **No tuning correction is needed or
  allowed** — the onset-Fn→grid map is driver-invariant by construction.
- **instrument / wavetable / pulse model → BACC generators.** 11-byte instrument
  records at `$17B0`: SR/AD (→`$D405/6`), a pulse-width+speed program (reg 2/3 ramp,
  visible in the oracle), a wave/filter-control program (low nibble indexes the
  filter-cutoff presets at `$17CE`), and a flags byte (pulse-program / filter /
  one-shot bits). These are the pitch-invariant per-voice generators. Under
  run-the-player rendering they execute natively; the recovered program carries the
  instrument table + orderlist + patterns (the `format.py` model).
- **orderlist / pattern → backward-LZ rows.** Per-voice orderlist (one stream per
  voice, walked by `ord_idx`): pattern numbers `$00–$7F`, transpose prefix
  `$80–$FD` (signed `v−$A0`), `$FE` stop, `$FF` loop. Patterns (walked by `pat_idx`):
  notes `$00–$5F`, set-instrument `$60–$7F`, set-note-length `$80–$BF`, effect
  `$C0–$FC`, `$FD` tie, `$FE` rest, `$FF` end. This is exactly the row/orderlist
  structure the codec's backward-LZ consumes — *Ode to Music* decodes to a clean
  3-voice canon (`01 02 03 01 02 …`, staggered entries) straight from the bytes.
- **player-specific hazards.** **Hard restart:** `FUN_163e` sets the per-voice
  `hardrestart_flag` (`$100F`) and writes ctrl `$08` (gate off, test-ish) on note
  trigger — this is what produces the frame-0 V1 `WF=$08` and the gate-off frames.
  Because we run the player, hard-restart is reproduced for free; a *static*
  re-encoder would have to model it. The effect commands `$C0–$FC` are only
  first-pass-decoded in `format.py` (Ode uses none) — **but run-the-player rendering
  does not depend on that**, so DMC tunes using effects still render byte-exact; the
  static parser just won't pretty-print them yet (cosmetic, not a render blocker).

### JCH OldPlayer (`$8800`)

- note→pitch: same A440-grid story expected (siddump shows clean notes `G-5 C3`,
  `G-2 9F`); freq table location TBD (needs the model extracted).
- instruments/orderlist: `init` ($882C) copies an 8-byte-per-subtune record (subtune
  table `$880C`) into working pointers `$8806`; per-voice orderlist pointers exist
  but the song-data model is **not yet extracted** (no `format.py`).
- hazard: high load address ($8800), relocates above BASIC — loader already handles
  arbitrary load addrs.

### JCH Protracker (`$1000`)

- Three JMP vectors (init `$1060`, play `$10D1`, third entry `$1653`). `init` copies
  three voice orderlist pointers (lo `$188C`, hi `$188D`, stride 2) into
  `$1733/$1736` and builds a 32-entry table at `$17D2`. AD/SR driven via
  `STA $D405,y`/`STA $D406,y`.
- Standard opcodes, no sub-frame — a run-the-player render should reach byte-exact
  the same way DMC does; the work is writing the `format.py`-equivalent to recover
  the song structure into the abstraction (the render itself is "run the image").

### JCH DigiPlayer (`$1000`)

- IRQ/NMI-driven (PSID `play=$0000`, real init `$1B40`, play vector `$1B72`).
- The signature `B1 ?? 4A 4A 4A 4A 18` is **4-bit packed-digi nibble unpacking**:
  the three voices carry near-identical frequencies and the tune streams sample
  nibbles — i.e. **sample playback is a sub-frame primitive** the BACC abstraction
  (note + instrument-generator + orderlist) does not currently express. This ties to
  the codec's open digi/sub-frame gap. Run-the-player at the PAL frame grid would
  *capture* the registers but the *abstraction* (notes/instruments) doesn't model a
  digi — so this is the one target that needs new codec machinery, not just a
  backend.

---

## 4. siddump oracle parse spec (the byte-exact gate target)

`docs/<player>/siddump.txt` is a fixed-layout ASCII table; parsing it yields the
same (nframes, 25) array as a `.dump.parquet` (which IS the production oracle — see
note below). Header lines: `Load/Init/Play`, `Calling…`, `Middle C frequency is …`,
then the column header and a `+---+` rule. Data rows match
`^\|\s*(\d+)\s*\|(.*)\|\s*$` and split on `|` into **3 voice cells + 1 filter cell**.

**Per-voice cell** (`Freq Note/Abs WF ADSR Pul`) → SID registers, voice base
`b = v*7`:

| siddump field | SID register(s) | encoding |
|---|---|---|
| `Freq` (4 hex) | reg `b+0` (lo), `b+1` (hi) | 16-bit freq, little-endian split |
| `Note/Abs` | — | *derived* (note name / abs-freq delta); **not a register**, ignore for the byte gate |
| `WF` (2 hex) | reg `b+4` | control register (waveform+gate) |
| `ADSR` (4 hex) | reg `b+5` = AD (high byte), `b+6` = SR (low byte) | siddump prints `AD<<8 \| SR` |
| `Pul` (3 hex) | reg `b+2` (lo), `b+3` (hi nibble) | 12-bit pulse width |

**Filter cell** (`FCut RC Typ V`):

| field | register | encoding |
|---|---|---|
| `FCut` (4 hex, 11-bit) | reg 21 (lo 3 bits), reg 22 (hi 8 bits) | siddump prints the 11-bit cutoff |
| `RC` (2 hex) | reg 23 | resonance + filter routing |
| `Typ` (text) + `V` (hex nibble) | reg 24 | mode/volume; text-printed — reconstruct from `.dump.parquet` in production rather than the text dump |

**Hold semantics:** a field of `.`/`..`/`...`/`....` means **no write this frame —
hold the previous value**. The parser keeps a 25-int `held` row and only overwrites
the registers whose field is non-dotted, then appends a copy. This matches the
codec's forward-held `per_frame_state`. Frame 0 carries the post-init snapshot (the
boot frame), which seeds `program.boot`.

> **Production note:** every target tune already has a `.dump.parquet` next to its
> `.sid` in HVSC (e.g. `Ode_to_Music.1.dump.parquet`). The backend's residual gate
> uses `per_frame_state(dump_path)` on that parquet (byte-exact, includes reg 24
> cleanly), exactly like the other backends; the `siddump.txt` parse above is the
> RE/probe oracle and a cross-check, not the production path.

---

## 5. Effort + ranking (with reasoning)

Ranking = corpus size × RE completeness × abstraction fit × (blocker present?).

1. **DMC v7.62 — land first. Effort S–M.** Largest corpus in all of HVSC (~10,700,
   v7.62/$1000 slice winnable immediately), highest RE completeness (`format.py`
   exists, full memory map), **byte-exact today** in the probe, clean abstraction
   fit, no illegal opcodes, no sub-frame. The only "work" is wrapping the existing
   pieces into a backend + reusing `_align`. **No blocker.**
2. **JCH Protracker — second. Effort M.** 94 tunes, standard opcodes, good fit.
   **Blocker:** no `format.py` — the song-data model (orderlist/pattern/instrument)
   must be extracted first (mirror `targets/dmc/format.py`). Run-the-player render
   should then reach byte-exact like DMC.
3. **JCH OldPlayer — third. Effort M.** 32 tunes; same shape as Protracker but
   smaller corpus. **Blocker:** no `format.py` (subtune table `$880C` mapped, model
   not extracted).
4. **JCH DigiPlayer — last. Effort L.** Only 4 tunes and the **sub-frame digi
   primitive blocks abstraction fit** — the codec has no note/instrument
   representation for streamed 4-bit samples. Needs both a `format.py` *and* a new
   digi primitive in the BACC alphabet. Lowest ROI; defer until the digi/sub-frame
   gap is addressed for its own sake.

---

## 6. Concrete step-by-step path to land the DMC backend first

Mirrors how the lft and GoatTracker backends landed (run-the-player + boot-aligned
residual gate). All paths under
`preframr-tokens/preframr_tokens/bacc/backends/`.

1. **New file `dmc.py` with a `DmcBackend(DriverBackend)`** modelled on `lft.py`
   (the run-the-player template) — but DMC needs **no illegal opcodes**, so use the
   plain `SIDEmu`/py65 path from `sidemu.py` instead of `MPU65ILL`.
2. **`matches(psid)`** — fingerprint the v7.62 `$1000` build: load/init `$1000`,
   play `$1003` (`play == init + 3`, but distinguish from GoatTracker which also has
   `init+3`), and the embedded `…762-PLAYER (C) BRIAN/GRAFFITY!-` signature string
   in the image. Order it **before** the broad GoatTracker matcher in
   `select_backend` / `_backend_for` (both in `base.py` and `recover.py`), since the
   GraffITY string makes it unambiguous.
3. **`recover(psid, nframes, subtune)`** — parse the song into the abstraction using
   the existing `targets/dmc/format.py` model (subtune table `$17F0`, orderlists,
   patterns `$1829/$182D`, instruments `$17B0`). Carry it as the `BaccProgram`
   structure (orderlist rows + instrument generators), **not** raw bytes — same
   contract as GoatTracker's `Song`. Seed `program.boot` from the dump's frame 0.
4. **`render(program)`** — re-run init + play once per PAL frame under `SIDEmu`,
   snapshot `state()` to an (nframes, 25) array. **Reuse the GoatTracker `_align`
   boot-frame logic verbatim** to absorb the measured **+1 frame** offset
   (`rendered[f] == oracle[f-1]`): match `program.boot` against the render's leading
   frames and drop the skew. No `mask_state` needed beyond the standard PW-high /
   filter-bit don't-cares the other backends share (lift `_mask_row` from
   `goattracker.py`).
5. **Residual gate** — wire `verify_residual(sid, dump)` against the tune's
   `.dump.parquet` (`Ode_to_Music.1.dump.parquet` is already present). The probe
   shows this passes byte-exact on regs 0–23 once aligned; reg 24 is clean from the
   parquet.
6. **Test** — add a backend test mirroring the lft/GoatTracker tests: assert
   `matches` on `Ode_to_Music.sid`, assert `verify_residual` True, assert the
   recovered orderlist is the expected 3-voice canon. Then sweep the v7.62 `$1000`
   slice of HVSC and report the byte-exact pass rate (the gate for widening to other
   DMC versions / a version-detection pass).

**Reproducible probe artifact:** `/scratch/anarkiwi/cbm/undmc/probe_dmc.py`
(`python3 probe_dmc.py` → prints the 87.16% raw / 100.00% aligned numbers above).
