# VICE CPU bus-trace / observation-graph (revice `libs/bustrace`)

Status: **SHIPPED**. Core + adapter landed in revice (PR
[anarkiwi/revice#7](https://github.com/anarkiwi/revice/pull/7)); asid-vice
adapter wired and **on-emulator validated** (PR
[anarkiwi/asid-vice#39](https://github.com/anarkiwi/asid-vice/pull/39) merged
2026-06-21, merge commit `037ca4c`). Determinism / dump-unchanged / completeness
all PASS on Monty + Grid_Runner — measured numbers in the runbook below.

## Why

We dropped `preframr-sidtrace` / the libsidplayfp `sidtrace` tool: its
`.sidwr.bin` varied run-to-run on the *same* tune (observed 1.78 MB / 464 B /
524 B — sometimes capturing almost nothing), so it was non-deterministic and
broke our byte-exact gates. VICE is the trusted, deterministic, byte-exact
ground-truth emulator (it already produces our `.dump.parquet` via the
`sounddump`/`dump2` path), so the bus observation belongs **there**.

The bus trace is the **provenance substrate for generic BACC recovery**:
- accumulators → rate/dwell (watch a zero-page cell incremented every frame),
- table-walks → orderlists (an index stepping through a contiguous read region),
- indexed-read → freq-write → the A440 note map (a table read feeding a
  `$d4xx` frequency write),
- per-voice generators (group writes by the PC that issued them).

It **replaces the dropped sidtrace `.bus.bin`** with a deterministic,
self-validating equivalent.

## Architecture (revice core + thin asid-vice adapter)

Follows the established revice pattern: pure-C core (no VICE headers, host-ops
via callback, CTest with no VICE build) + a thin VICE adapter compiled from the
submodule.

```
libs/bustrace/
  include/revice_bustrace.h   core API + the documented binary record format
  src/bustrace_core.c         pure serializer (no alloc / time / RNG / float)
  tests/test_bustrace.c       assert-based CTest: golden bytes + determinism + CRC
  vice/soundbustrace.c        VICE adapter: -bustrace resource, file I/O, hook glue
  vice/soundbustrace.h        adapter interface
  vice/{07,08}*.patch         (staged in integration/vice/patches/) the two hooks
```

- **Core** ingests `{cycle, addr, val, rw, pc}` per access and hands finished
  byte spans to an `emit` callback. It holds only a running cycle baseline (for
  delta encoding), a record counter, and a CRC accumulator.
- **Adapter** registers the `BusTraceFile` resource + `-bustrace <file>`
  cmdline option, opens the file lazily on first traced access, feeds the core
  from the maincpu bus hook (reading `maincpu_clk` and `reg_pc` itself), and
  writes the trailer at `machine_specific_shutdown`.

### Where the hook lives

The per-access call sits in `src/c64/vsidcpu.c`'s `FEATURE_CPUMEMHISTORY`
`memmap_mem_store` / `_store_dummy` / `memmap_mem_read` / `_read_dummy`
functions — the single point where VICE already observes **every** 6510 read
and write. This is faithful (real bus values: the stored byte on writes, the
returned byte on reads) and low-perturbation:

- For reads the value is captured *after* the underlying read returns it.
- The opcode-fetch bit is read from `memmap_state & MEMMAP_STATE_OPCODE`
  *before* `memmap_mark_read()` clears it.
- Dummy/internal accesses are tagged (bit 2) so they are kept for fidelity but
  filterable by a reader.

Because it reuses `FEATURE_CPUMEMHISTORY`, the trace build configures the tree
with `--enable-cpuhistory` (the existing fork tree already has it: `config.h`
carries `#define FEATURE_CPUMEMHISTORY`).

## Trace record format (version 1)

A trace file is `header(16) · record(10)* · trailer(16)`, every multi-byte
field little-endian, written by hand so the bytes are identical on any host.

Header (16 B):

| off | size | field | value |
|---|---|---|---|
| 0 | 4 | magic | `"RBT1"` |
| 4 | 2 | version | 1 |
| 6 | 1 | rec_size | 10 |
| 7 | 1 | flags | bit0 = PC present (1) |
| 8 | 8 | cycles_per_sec | CPU clock (PAL/NTSC/Drean), 0 if unknown |

Record (10 B, one per access, in cycle order):

| off | size | field | meaning |
|---|---|---|---|
| 0 | 4 | cycle_delta | cycles since previous record (first is from 0); sum to recover absolute cycle |
| 4 | 2 | addr | 16-bit bus address |
| 6 | 1 | val | byte on the bus (read result or stored) |
| 7 | 1 | rw_flags | bit0 write, bit1 opcode-fetch, bit2 dummy/internal |
| 8 | 2 | pc | program counter of the accessing instruction |

Trailer (16 B):

| off | size | field | meaning |
|---|---|---|---|
| 0 | 4 | magic | `"RBTe"` |
| 4 | 8 | rec_count | number of records |
| 12 | 4 | crc32 | CRC-32 (IEEE, reflected) over the record bytes |

Fixed-width, append-only: a reader strides by `rec_size`, reconstructs absolute
cycles by summing `cycle_delta`, and validates completeness with
`rec_count` + `crc32`.

## Determinism guarantee

The core does **no** allocation, **no** time/RNG/environment reads, and **no**
floating point; all serialization is explicit little-endian. Given an identical
access sequence it emits a byte-identical stream — independent of host
endianness/word size. This is exactly the property sidtrace lacked. The unit
test asserts it directly (two independent runs of a 256-access sequence produce
identical bytes) and pins the CRC polynomial with the standard
`"123456789" → 0xCBF43926` vector.

Because asid-vice's emulation is itself deterministic (it already produces our
byte-exact dumps), the same tune fed through the same build yields the same
access sequence, hence the same trace file.

## Dump-unchanged (additive) guarantee

The feature is additive. With no `-bustrace` file set, `bustrace_observe_access`
is a single null-pointer test and returns — the SID-register dump path
(`-sounddev dump`, the `.dump.parquet` ground truth) is untouched. The hook is
in the CPU memmap functions, which already call `monitor_memmap_store` under
`FEATURE_CPUMEMHISTORY`; we only *read* bus values and append to a separate
file, never altering emulation or the dump.

## How it feeds generic BACC recovery

`vsiddump.py` keeps producing `.dump.parquet` (the SID-write ground truth)
exactly as today. Alongside it, a `-bustrace tune.bus.bin` run yields the full
bus provenance. A recovery pass joins them: every SID-register write in the
dump appears in the trace with matching cycle/addr/val, and the trace
additionally carries the *reads* and the *PC* that produced each write — the
information generic BACC needs to discover accumulators, table-walks, the note
map, and per-voice generators without per-driver hand disassembly.

## Validation runbook (on-emulator) — VALIDATED 2026-06-21

The real build is the **headlessvice container recipe + `--enable-cpuhistory`**.
The prior runbook's `--with-alsa` / `libasound2-dev` / glib / sdl deps were
**wrong**: `/scratch/anarkiwi/cbm/headlessvice/Dockerfile` builds headless VSID
with only `file make autoconf gcc g++ flex bison dos2unix xa65
libcurl4-openssl-dev pkg-config zlib1g-dev` and the flags below — no
alsa/glib/sdl. The one required addition for bustrace is `--enable-cpuhistory`,
because the per-access hook lives in the `FEATURE_CPUMEMHISTORY` `vsidcpu.c`
memmap functions (stock headlessvice does not enable it).

Build (Dockerfile modelled on headlessvice/Dockerfile):

```bash
# clone the PR branch with the revice submodule (pinned 914a33b, has libs/bustrace)
git clone --recursive -b feature/bustrace https://github.com/anarkiwi/asid-vice
cd asid-vice && git submodule update --init --recursive
bash src/revice/integration/vice/apply-wiring.sh    # libvsid bustrace block + patches 07/08
aclocal && autoheader && autoconf && automake --force-missing --add-missing && ./autogen.sh
./configure --enable-headlessui --enable-cpuhistory --disable-pdf-docs \
    --without-pulse --without-alsa --without-png --disable-dependency-tracking \
    --disable-realdevice --disable-rs232 --disable-ipv6 --disable-native-gtk3ui \
    --disable-sdlui --disable-sdlui2 --disable-ffmpeg
make -C src/monitor mon_parse.h mon_parse.c mon_lex.c
make -j"$(nproc)" all && make install        # make install is REQUIRED: it
                                             # installs the C64 ROMs/data dir;
                                             # without it vsid exits 255 at
                                             # initcmdline_check_args (sysfile
                                             # load fails), NOT a bustrace bug.
```

Run (the `dump` sound driver needs `-limitcycles` and a **FIFO** for `-soundarg`,
exactly like `vsiddump.py`; a plain regular-file soundarg segfaults the driver —
unrelated to bustrace, true on the stock headlessvice image too). Mount HVSC +
a work dir; `os.makedirs('/root/.local/state/vice/')` first.

```bash
# DETERMINISM: trace the same tune twice via the vsiddump.py FIFO mechanism
vsid -console -logfile /dev/null +logtofile +logtostdout -debug -warp -sound \
  -soundwarpmode 1 -sounddev dump -soundarg <fifo_a> -bustrace a.bus.bin \
  -tune 1 -limitcycles 29557440 Monty_on_the_Run.sid          # ditto -> b.bus.bin
cmp a.bus.bin b.bus.bin    # MUST be byte-identical

# DUMP-UNCHANGED: same run without -bustrace; cmp the two raw dumps -> identical
# COMPLETENESS: stride the 10-byte records, filter rw bit0 & addr 0xd400..0xd418,
#   confirm the (addr,val) write sequence matches the dump's and differs only by a
#   single constant cycle offset, and recompute CRC-32 over the record region ==
#   trailer crc32.
```

### Measured results (2026-06-21, container `bustrace-validate`, PAL 985248 Hz, 30 s)

| tune | determinism | dump-unchanged | completeness | trace size | SID writes | trailer CRC-32 |
|---|---|---|---|---|---|---|
| Monty_on_the_Run.sid (Hubbard, tune 1) | PASS (byte-identical) | PASS (byte-identical) | PASS | 5 523 262 B (552 323 recs) | 16 428 | `0x1fbc79b4` |
| Grid_Runner.sid (Jammer, tune 1) | PASS (byte-identical) | PASS (byte-identical) | PASS | 4 686 642 B (468 661 recs) | 37 576 | `0x7591e663` |

For both tunes the trace's SID-write `(addr, val)` sequence is **identical** to
the `-sounddev dump`'s, the CRC-32 verifies, and the only difference is a single
**constant +64-cycle** origin offset (the dump driver and the bus trace timestamp
from different fixed baselines — not missing or divergent data). Determinism (the
property libsidplayfp sidtrace lacked) holds byte-exactly. **PR
[anarkiwi/asid-vice#39](https://github.com/anarkiwi/asid-vice/pull/39) merged**
on the strength of this runtime proof (merge commit `037ca4c`).

(The determinism property is also proven in the core unit test, which needs no
VICE build.)

## Status / next step

- revice `libs/bustrace` core + tests: **landed** (CTest green: 6/6 incl.
  bustrace's 8 sub-tests).
- asid-vice adapter: implemented; `soundbustrace.c` and `bustrace_core.c`
  **compile clean** against real VICE headers (headless arch, `-Wall -Wextra`);
  the `vsidcpu.c` / `vsid.c` wiring patches apply cleanly to the fork.
- Next: run the validation runbook on a box with the full VICE build deps
  (alsa/glib/xa65) to capture the determinism + dump-unchanged + completeness
  numbers on Monty / Grid_Runner, then point the BACC recovery pass at the
  `.bus.bin` next to each `.dump.parquet`.
