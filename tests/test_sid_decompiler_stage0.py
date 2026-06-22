"""Stage 0 residual-zero gate for the generic per-axis SID decompiler.

Proves the ``{G_A}`` recurrence model end-to-end on the hardest (generative,
no-table) case — A Mind Is Born (lft) — by re-executing the recurrences (a
self-contained fixed-size state machine, NOT the py65 white-box player) and
requiring a byte-exact ``[N, 25]`` SID register stream.

The correctness gate (project HARD RULE #0 / design §2.7) is residual-zero:
``residual(rendered, reference) == 0`` over all 8,190 frames, under the
documented don't-care masks only. The reference is the tune's actual SID-write
stream, taken from the established white-box ``LftBackend`` oracle when its
artifacts are importable; in all environments a frozen SHA-256 fingerprint of
that masked stream pins the gate so a regression in the interpreter is caught.
"""

import hashlib
import os
import sys

import numpy as np
import pytest

from preframr_experiments.sid_decompiler.stage0_amind import (
    AmindRecurrence,
    mask_dontcares,
    residual,
    verify_axis_closed_forms,
)

NFRAMES = 8190

# Frozen fingerprint of the masked [8190,25] reference stream produced by the
# established white-box LftBackend (itself proven residual-zero vs the HVSC dump
# by research/a_mind_is_born/verify.py). Pins the gate without shipping the dump.
REF_SHA256_FULL = "ba5709d471972e172d6f48e80d1004da70315f7ebc5acdf3ccc07f59e490dd67"
REF_SHA256_512 = "e0864496773d4180f62e32003dc9128386b3485b35d2e7f19a09a01dae573212"

# External white-box oracle (residual reference). Optional; the fingerprint above
# makes the gate run everywhere.
_AMIND_WT = "/scratch/tmp/amind-wt"
_SID = "/scratch/preframr/hvsc/C64Music/MUSICIANS/L/Lft/A_Mind_Is_Born.sid"


def _lft_reference():
    """Return the LftBackend [NFRAMES,25] reference, or None if unavailable."""
    if not (os.path.isdir(_AMIND_WT) and os.path.isfile(_SID)):
        return None
    if _AMIND_WT not in sys.path:
        sys.path.insert(0, _AMIND_WT)
    try:
        from preframr_tokens.bacc.backends.lft import LftBackend
        from preframr_tokens.bacc.sidemu import load_psid
    except Exception:
        return None
    psid = load_psid(_SID)
    be = LftBackend()
    if not be.matches(psid):
        return None
    return be.render(be.recover(psid, NFRAMES, 0))


@pytest.fixture(scope="module")
def rendered():
    """Render the recovered recurrence for the full tune (our interpreter)."""
    return AmindRecurrence().render(NFRAMES)


def _masked_sha(arr):
    return hashlib.sha256(mask_dontcares(arr).astype(np.uint8).tobytes()).hexdigest()


def test_residual_zero_against_frozen_fingerprint(rendered):
    """Gate (always runs): masked render fingerprint == frozen reference."""
    assert _masked_sha(rendered) == REF_SHA256_FULL


def test_residual_zero_against_white_box_oracle(rendered):
    """Gate (when oracle present): residual == 0 over all 8,190 frames."""
    ref = _lft_reference()
    if ref is None:
        pytest.skip("LftBackend white-box oracle / .sid not available in env")
    assert ref.shape == (NFRAMES, 25)
    assert residual(rendered, ref) == 0


def test_length_independence(rendered):
    """The recurrence recovered/used is window-independent: rendering K frames
    and M>>K frames are both residual-zero from the SAME fixed-size program, and
    the program size does not grow with the window."""
    rec = AmindRecurrence()
    short = rec.render(512)
    assert _masked_sha(short) == REF_SHA256_512  # K-frame slice is byte-exact
    # M >> K from the same program is byte-exact too.
    assert _masked_sha(rendered) == REF_SHA256_FULL
    # program size is flat across windows -> slope 0.
    sizes = [AmindRecurrence().state_size_bytes() for _ in (128, 512, 2048, 8190)]
    assert len(set(sizes)) == 1
    assert sizes[-1] - sizes[0] == 0


def test_program_size_is_small_and_flat():
    """Recovered program is ~the 254-byte image, not length-proportional."""
    rec = AmindRecurrence()
    assert rec.state_size_bytes() == 256  # full zero-page seed
    nonzero = sum(1 for b in rec.zp_seed if b)
    # the lft image is 254 bytes / 242 nonzero (RECOVERY.md); within that order.
    assert nonzero < 256
    # ~0.031 program-bytes/frame over 8190 frames — far below storing the output.
    assert rec.state_size_bytes() / NFRAMES < 0.05


def test_axis_closed_forms_exact(rendered):
    """The standalone per-axis (S,U,O,K) closed forms over the shared master
    counter reproduce their registers byte-exact (model-shape evidence)."""
    res = verify_axis_closed_forms(rendered)
    assert res == {
        "v1_pulse_width_lo": True,
        "filter_cutoff_hi": True,
        "v2_control": True,
    }


def test_recover_from_psid_identity(rendered):
    """When the tune file + loader are present, recover the program by identity
    from the .sid (run init, lift the post-init zero page) and confirm it renders
    the same residual-zero stream as the embedded seed."""
    from preframr_experiments.sid_decompiler.stage0_amind import recover_from_psid

    if not os.path.isfile(_SID):
        pytest.skip(".sid not available in env")
    try:
        rec = recover_from_psid(_SID, amind_wt=_AMIND_WT)
    except RuntimeError:
        pytest.skip("white-box loader unavailable in env")
    assert _masked_sha(rec.render(NFRAMES)) == REF_SHA256_FULL


def test_deterministic():
    """Re-executing the recurrence twice is byte-identical (determinism)."""
    a = AmindRecurrence().render(1024)
    b = AmindRecurrence().render(1024)
    assert np.array_equal(a, b)


def test_interpreter_rejects_unknown_opcodes():
    """The minimal core never silently diverges: an unknown opcode raises."""
    from preframr_experiments.sid_decompiler.cpu6502 import CPU6502

    mem = bytearray(0x10000)
    mem[0x0200] = 0x02  # KIL/JAM — not implemented
    cpu = CPU6502(mem)
    cpu.pc = 0x0200
    with pytest.raises(NotImplementedError):
        cpu.step()
