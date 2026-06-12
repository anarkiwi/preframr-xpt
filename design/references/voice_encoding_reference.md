# Voice encoding reference — how the 3 SID voices are carried in the token stream

**Status:** Pointer (2026-06-12). The encoding mechanics — FRAME-val base-4 voice-order
packing, the `VOICE` (reg −126) delimiter trap (`val=0`, carries no voice identity),
`remove_voice_reg` decode, the 15 legal `VALID_VOICEORDERS`, `zero_voice_reg` — are
documented in the **[preframr-tokens README](https://github.com/anarkiwi/preframr-tokens)**
(parse-domain section) and pinned by the code anchors there.

**Learnability framing (xpt-internal).** The FRAME-val voice multiplex is the
cross-voice causal-state the de-mux lever targets — see
[`learnability_token_ordering_theory.md`](learnability_token_ordering_theory.md) and
[`generator_mdl_representation.md`](../encoding/generator_mdl_representation.md).

## Implications for modeling (why this matters for melody)

- Melodic onsets are **multiplexed across voices**: consecutive onsets belong to
  different voices, set by the FRAME order — they are not one line. Predicting the next
  onset requires tracking *which voice's* line is being continued.
- Voice identity rides in a **structural token** (FRAME val, low 6 bits) that also
  marks the time tick: the FRAME class is load-bearing for content, not just
  scaffolding — its per-class accuracy is worth reading alongside onset accuracy.
- `per_voice_aux_supervision_design.md` and any voice-trajectory work depend on this:
  the supervision target is the FRAME-derived voice, not a VOICE token field.
