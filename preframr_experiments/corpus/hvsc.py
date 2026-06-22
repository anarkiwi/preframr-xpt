"""HVSC metadata helpers: per-subtune frame budgets (Songlengths.md5) and the
ground-truth tracker catalog (hvsc-tracker-catalog ``results.csv``).

``subtune_frames`` mirrors headlessvice's ``vsiddump.py`` cycle math so the frame
budget handed to the codec's sid-only ``recover_from_sid`` matches the budget the
codec's own gate fixtures use. Paths are HVSC-relative to the ``C64Music`` root
(``DEMOS/...``, ``MUSICIANS/...``), the convention the catalog and the pinned
.list files share.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import os

PAL_PHI = 985248  # PAL CPU clock (Hz)
NTSC_PHI = 1022727  # NTSC CPU clock (Hz)
PAL_CPF = 19656  # PAL cycles/frame (raster)
NTSC_CPF = 17095  # NTSC cycles/frame (raster)

# Catalog player-id substrings -> codec backend family. The gate is residual-zero
# (the census), not the tracker; this only labels/prioritises rows and is allowed
# to be approximate.
BACKEND_PLAYERS = (
    ("goattracker", "GoatTracker"),
    ("dmc", "DMC"),
    ("hubbard", "Rob_Hubbard"),
    ("lft", "LFT"),
)


def sid_md5(sid_path) -> str:
    """Lowercase hex MD5 of the .sid bytes (the Songlengths.md5 key)."""
    with open(sid_path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest().lower()


@functools.lru_cache(maxsize=8)
def _songlengths_index(songlengths_path: str) -> dict[str, list[str]]:
    """Map md5 -> per-subtune time tokens from a Songlengths.md5 file (cached)."""
    index: dict[str, list[str]] = {}
    with open(songlengths_path, encoding="utf-8") as handle:
        for line in handle:
            if "=" not in line or line.startswith((";", "[")):
                continue
            md5, _, times = line.strip().partition("=")
            index[md5.lower()] = times.split()
    return index


def _token_seconds(token: str) -> float:
    """Parse one HVSC songlength token (``[H:]M:S[.mmm]``) to seconds, matching
    headlessvice: the fractional part after ``.`` is read as milliseconds, the
    colon-separated fields are sexagesimal (S / M:S / H:M:S)."""
    base = token
    fraction = 0.0
    if "." in token:
        base, ms = token.split(".", 1)
        fraction = float(ms) / 1e3
    total = 0
    for part in base.split(":"):
        total = total * 60 + int(part)
    return total + fraction


def subtune_seconds(sid_path, subtune: int, songlengths_path) -> float:
    """Length in seconds of ``subtune`` (1-based) from Songlengths.md5."""
    times = _songlengths_index(str(songlengths_path)).get(sid_md5(sid_path))
    if not times:
        raise KeyError(f"no Songlengths entry for {os.path.basename(str(sid_path))}")
    if subtune < 1 or subtune > len(times):
        raise IndexError(f"subtune {subtune} out of range (1..{len(times)})")
    return _token_seconds(times[subtune - 1])


def subtune_frames(sid_path, subtune: int, songlengths_path, ntsc: bool = False) -> int:
    """Frame budget for ``subtune`` (1-based): ``cycles // cpf`` where
    ``cycles = phi * seconds`` (mirrors headlessvice ``vsiddump.py``)."""
    phi, cpf = (NTSC_PHI, NTSC_CPF) if ntsc else (PAL_PHI, PAL_CPF)
    cycles = int(phi * subtune_seconds(sid_path, subtune, songlengths_path))
    return cycles // cpf


def subtune_count(sid_path, songlengths_path) -> int:
    """Number of subtunes for a .sid (length of its Songlengths token list)."""
    times = _songlengths_index(str(songlengths_path)).get(sid_md5(sid_path))
    return len(times) if times else 0


@functools.lru_cache(maxsize=4)
def load_tracker_map(results_csv: str) -> dict[str, str]:
    """Map HVSC-relative .sid path -> player id from the catalog ``results.csv``."""
    out: dict[str, str] = {}
    with open(results_csv, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header: path,player
        for row in reader:
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def backend_family(player: str | None) -> str | None:
    """Codec backend family for a catalog player id, or None if unmapped."""
    if not player:
        return None
    for family, needle in BACKEND_PLAYERS:
        if needle.lower() in player.lower():
            return family
    return None
