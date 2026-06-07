# Voice encoding reference — how the 3 SID voices are carried in the token stream

**Status:** Reference. Documents the shipped encoding (preframr-tokens `RegLogParser`)
so the FRAME-val voice-order packing doesn't get re-derived. Not an experiment.

**Learnability framing.** The FRAME-val voice multiplex is the cross-voice causal-state the de-mux lever targets — see [`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md) and the generator-MDL pipeline [`generator_mdl_representation.md`](generator_mdl_representation.md).

## The trap (read this first)

In the token stream, **`VOICE` (reg −126) markers carry `val=0` and tell you nothing
about which voice is active.** Reading voice off the `VOICE` token makes every write
look like voice 0. **Voice identity is packed into the `FRAME` (reg −128) token's
`val`**, and the `VOICE` markers are positional delimiters that *advance* through the
order that `FRAME` declared.

## The encoding

Raw SID register writes encode voice **by register number**: reg = `voice*7 + field`,
with `VOICE_REG_SIZE = 7` fields per voice (`0 freq_lo, 1 freq_hi, 2 pw_lo, 3 pw_hi,
4 control, 5 attack_decay, 6 sustain_release`). So V0 = regs 0–6, V1 = 7–13, V2 =
14–20; regs 21–24 are global filter/volume. (`VOICE_CTRL_REG = {v: v*7+4}` in
`stfconstants.py`.)

`RegLogParser._add_voice_reg` (`reglogparser.py:558`) rewrites this for the model:

1. **Canonicalises the register** — `reg %= VOICE_REG_SIZE`, collapsing all three
   voices onto the same 0–6 field range. After this, the register alone no longer
   tells you the voice (this is why op45/op48 melodic writes all show up on reg 0).
2. **Inserts a `VOICE` (−126) marker** before each voice's writes, as a delimiter.
3. **Packs the per-frame voice order into the `FRAME` (−128) token's `val`** (`svt`,
   lines 590–617): for the k-th FRAME/VOICE marker in the frame (k = 0,1,2…),
   contribute `(voice+1) << (2*k)` and sum. Result = a **base-4 little-endian number,
   one 2-bit digit per voice slot, digit = voice+1.**

So per frame the layout is:

```
FRAME(val=order)  [writes for order[0]]  VOICE(0) [writes for order[1]]  VOICE(0) [writes for order[2]] …
```

The first voice's writes follow `FRAME` directly (no `VOICE` marker — slot 0 is the
FRAME marker itself); each subsequent `VOICE` marker switches to the next voice in the
declared order. A redundant `VOICE` immediately after `FRAME` is dropped (line 583).

### Decode (authoritative inverse)

`remove_voice_reg` (`reglogparser.py:215`) recovers it:

```python
sval = FRAME.val & 0b111111            # low 6 bits = the packed order (3 slots × 2 bits)
# fn = index of this FRAME/VOICE marker within the frame (0,1,2…)
voice = ((sval >> (fn*2)) & 0b11) - 1  # forward-filled to the writes that follow
reg   = canonical_reg + voice*7        # un-canonicalise back to the real SID register
```

`VALID_VOICEORDERS` (`reglogparser.py:107`) is the 15 legal `val`s = every permutation
of voices {0,1,2} taken 1, 2, or 3 at a time (P(3,1)+P(3,2)+P(3,3) = 3+6+6). A FRAME
val outside this set is a decode error (asserted at encode, line 618).

### Worked example (real stream, first training song)

| FRAME val | binary | 2-bit digits (lo→hi) | digit−1 = voice order |
|---|---|---|---|
| 57 | `111001` | 1, 2, 3 | **[0, 1, 2]** — all three voices |
| 54 | `110110` | 2, 1, 3 | **[1, 0, 2]** — reordered |
| 6  | `000110` | 2, 1 | **[1, 0]** — two voices this frame |
| 1  | `000001` | 1 | **[0]** — single voice |
| 2  | `000010` | 2 | **[1]** |

Frames reorder voices freely; a frame writes only the voices it touches.

## The `zero_voice_reg` switch

`_add_voice_reg(zero_voice_reg=True)` (the training/canonical path, line 595) sets every
`VOICE` token `val=0` — voice lives **only** in the FRAME val. With
`zero_voice_reg=False` (lines 597–614), the `VOICE` token's val instead carries a
per-voice **meta byte** = `freqmeta` (top `META_FREQ_BITS=4` bits of the voice's last
freq) + `ctrlmeta` (the voice's control high nibble). That branch is for analysis/
inspection, not the trained stream. **In any tokens.csv / dataset.csv the model trains
on, `VOICE.val == 0`.**

## Implications for modeling (why this matters for melody)

- The melodic onsets are **multiplexed across voices**: consecutive `op48`/`op45`
  onsets belong to different voices, set by the FRAME order — they are not one line.
  Predicting the next onset requires tracking *which voice's* line is being continued.
- That voice identity rides in a **structural token** (FRAME val, low 6 bits), which
  also marks the time tick. The FRAME class is therefore load-bearing for content, not
  just scaffolding — its per-class accuracy is worth reading alongside V0-onset acc
  (if the model can't predict the voice-order header, voice attribution is broken
  upstream of pitch).
- `per_voice_aux_supervision_design.md` (per-voice aux heads) and any voice-trajectory
  work depend on this: the supervision target is the FRAME-derived voice, not a VOICE
  token field.

## Code anchors

- `preframr_tokens/reglogparser.py:558` `_add_voice_reg` (encode + FRAME-val packing)
- `preframr_tokens/reglogparser.py:215` `remove_voice_reg` (decode/inverse)
- `preframr_tokens/reglogparser.py:107` `_build_valid_voiceorders` (the 15 legal vals)
- `preframr_tokens/stfconstants.py` `VOICE_REG_SIZE`, `VOICE_REG`, `FRAME_REG`,
  `VOICE_CTRL_REG`, `META_FREQ_BITS`
