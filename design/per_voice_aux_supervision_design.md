# Per-voice multi-target auxiliary supervision — design

**Status (2026-05-20):** scoping. No code yet. Motivated by the
exhaustion of token-form interventions (`weighted_token_loss_mini`
refuted, `learnable_class_loss_mini` refuted, `set_to_diff_mini`
refuted via alphabet inflation +90% on motion-reg DIFF expansion).

## The problem

Per-class audit on `accuracy_push_prodlike_4x` (2026-05-19): the
model predicts structural tokens well (FRAME 42%, HARD_RESTART 68%,
SUBREG_FLUSH 56%) but fails on musical content (FREQ_LO 5.94%,
PWM_PRESET 4-14%). Under greedy decode the model collapses to
fixed-point loops on Wizball.1 even on train-set songs — source
render via the audio pipeline sounds correct, ruling out pipeline.

The pattern across refuted attempts is consistent: the model fails
to maintain an internal representation of "what musical event is
currently happening" — gate state, pitch, waveform, envelope. Token
re-arrangement (loss tier weighting, gate-anchored SET→DIFF, etc.)
doesn't help because the model isn't being asked to *predict* musical
state — it's only asked to predict the next token, and the next
token's CE gradient gets dominated by easy structural patterns.

## Hypothesis

Adding auxiliary supervision targets at every per-voice token
position — gate / pitch / waveform / envelope phase / amplitude —
forces the model to encode musical state in its hidden representation
as an explicit objective. The hidden state is then better-conditioned
to predict the next-token content as a side effect.

Contrast with `voice_trajectory`: that approach injects derived
features as INPUT tokens; the model is free to ignore them. Aux
supervision uses similar derivations as OUTPUT targets; the model
must encode them in hidden state or take a loss hit. Information-
theoretically stronger.

## Auxiliary head set

Five heads attached to the final hidden state. Heads fire only at
`VOICE_REG` token positions (each represents "voice V is about to
write at frame F"). All other positions are masked out.

| Head | Target | Loss | Mask |
|---|---|---|---|
| `gate_on_next` | gate bit at end of frame F for voice V (bool) | BCE | all VOICE_REG positions |
| `pitch_class` | (round(12·log2(freq/freq_C0)) mod 12) ∈ 0..11 | 12-way CE | gate_on_next=1 |
| `pitch_octave` | (round(12·log2(freq/freq_C0)) // 12) ∈ 0..10 | 11-way CE | gate_on_next=1 |
| `waveform` | dominant bit of CTRL[4..7] ∈ {triangle, saw, pulse, noise} | 4-way CE | gate_on_next=1 |
| `adsr_phase` | derived phase ∈ {attack, decay, sustain, release} | 4-way CE | gate_on_next=1 |

**Why pitch as class+octave instead of single 128-way CE:** keeps
per-head logit size small (12 + 11 = 23 categorical outputs vs
128); the model factors octave (slow-changing) from pitch class
(fast-changing).

**Why drop `amp_log` from v1:** computing the analytic ADSR envelope
amplitude at each frame requires emulating the SID envelope state
machine. Moderate complexity; defer to v2 if the categorical heads
help.

**Why no explicit "note onset" head:** captured implicitly by the
gate_on_next transition at consecutive (frame, voice) positions.

## Label derivation (parser-side)

Computed offline during parse and stashed as new columns in the
parquet next to existing (op, reg, val, ...) columns. Schema:

```
{
    "aux_gate_on": pd.Int8Dtype(),     # 0/1, -1 = undefined
    "aux_pitch_class": pd.Int8Dtype(),  # 0..11, -1 = undefined
    "aux_pitch_octave": pd.Int8Dtype(), # 0..10, -1 = undefined
    "aux_waveform": pd.Int8Dtype(),     # 0..3, -1 = undefined
    "aux_adsr_phase": pd.Int8Dtype(),   # 0..3, -1 = undefined
}
```

`-1` is the standard ignore_index for CE losses and the mask
sentinel for BCE.

**Label generation algorithm** (run once per song during parse,
after `_add_voice_reg`):

1. Walk the df row-by-row, tracking per-voice state: `cur_freq`,
   `cur_ctrl`, `cur_ad`, `cur_sr`, `frames_since_gate_change`.
2. At each `VOICE_REG` row for voice V at frame F: compute targets
   from the state AS OF THE END OF FRAME F (after all writes in F
   apply). Stash in the aux columns.
3. At all non-VOICE_REG rows: aux columns = -1.

For frames where voice V doesn't write, the state carries forward
from the previous frame (via running per-voice state). VOICE_REG
markers always exist post-`_add_voice_reg` regardless of whether
the voice writes — so labels are always defined per (frame, voice).

**ADSR phase derivation:**
- frame just after a gate-off→on transition: `attack`
- N frames after gate-on, where N depends on AD value: transition
  to `decay`, then `sustain`
- frame just after gate-on→off transition: `release`

Approximation: use frames-since-gate-change vs the AD/SR decode
table. Don't emulate the SID envelope analog circuit; the
categorical phase is enough for supervision.

## Dataset wiring

`preframr/train/regdataset.py`'s training-block construction needs to
emit the aux labels alongside the existing input/target token
arrays:

```python
# existing
inputs:  (n_blocks, seq_len+1)   # int32 token ids
targets: (n_blocks, seq_len)     # int32 token ids (shifted)

# new
aux_gate:    (n_blocks, seq_len)  # int8, -1 = ignore
aux_pitch_c: (n_blocks, seq_len)  # int8, -1 = ignore
aux_pitch_o: (n_blocks, seq_len)  # int8, -1 = ignore
aux_wave:    (n_blocks, seq_len)  # int8, -1 = ignore
aux_adsr:    (n_blocks, seq_len)  # int8, -1 = ignore
```

Stored as `*.aux.npy` next to `*.blocks.npy` per song.

## Model integration

`preframr/core/model.py`:

```python
class PreframrLM(LightningModule):
    def __init__(...):
        ...
        self.aux_heads = nn.ModuleDict({
            "gate":    nn.Linear(embed_dim, 1),
            "pitch_c": nn.Linear(embed_dim, 12),
            "pitch_o": nn.Linear(embed_dim, 11),
            "wave":    nn.Linear(embed_dim, 4),
            "adsr":    nn.Linear(embed_dim, 4),
        })

    def forward(self, x):
        hidden = self.transformer(x)        # (B, L, D)
        next_logits = self.lm_head(hidden)  # existing next-token head
        aux_logits = {name: head(hidden) for name, head in self.aux_heads.items()}
        return next_logits, aux_logits
```

Loss composition:

```python
loss_next = F.cross_entropy(next_logits, targets, ignore_index=PAD)
aux_loss = 0
for name, logits in aux_logits.items():
    if name == "gate":
        valid = (aux_labels[name] != -1)
        if valid.any():
            aux_loss = aux_loss + F.binary_cross_entropy_with_logits(
                logits[valid].squeeze(-1), aux_labels[name][valid].float()
            )
    else:
        aux_loss = aux_loss + F.cross_entropy(
            logits.transpose(1, 2), aux_labels[name], ignore_index=-1
        )

loss = lambda_next * loss_next + lambda_aux * aux_loss
```

Default weights: `lambda_next=1.0, lambda_aux=0.25` (each of 5 aux
heads contributes ~0.05; total aux gradient ~25% of next-token
gradient). Hand-tuned; could anneal.

## Mini A/B plan

| Arm | Spec | Lambda_aux | Description |
|---|---|---|---|
| `aux_off` (control) | base + no aux heads | 0.0 | baseline |
| `aux_quarter` | base + 5 heads | 0.25 | low-weight aux |
| `aux_full` | base + 5 heads | 1.0 | equal-weight aux |

Same base pipeline_spec as the corrected `set_to_diff_mini` (slope +
preset + hard_restart + legato c2/c4 + voice_block_order + ctrl_bigram
+ loop). 2 seeds, mini tier.

**Metrics:**
- Standard: `alphabet_size`, `val_loss_best`, `val_acc_at_best_loss`,
  `epochs_to_best_val_loss`, `wallclock_train_min`
- New per-head: `aux_gate_acc`, `aux_pitch_class_acc`,
  `aux_pitch_octave_acc`, `aux_waveform_acc`, `aux_adsr_acc`
- Token-class breakdown on next-token val_acc, especially FREQ_LO

**Pass gate:**
- next-token val_acc(aux_quarter or aux_full) ≥ aux_off + 0.005
- AND aux head accuracies > 0.5 (heads are learning meaningful state)
- AND alphabet_size unchanged (aux supervision shouldn't touch tokens)

## Audio render impact

**None.** Aux supervision is supervision-only. The token stream is
unchanged. Existing audio pipeline (`prepare_df_for_audio` →
`remove_voice_reg` → `expand_ops` → renderer) operates on the same
token shape; aux columns are dropped or ignored.

## Tier

`audio_bit_exact` for the encoder (no change). The model gains aux
heads but its forward signature stays compatible (just emits more
logits that callers ignore unless training).

## Risks

1. **Multi-task loss balancing.** Earlier `learnable_class_loss_mini`
   showed learnable per-tier weights drifted near-uniform; same trap
   possible here. Mitigation: start with hand-tuned `lambda_aux=0.25`,
   ablate via the three-arm A/B. If both `aux_*` arms refute, drop
   aux supervision; the multi-task loss is the wrong intervention.
2. **Hidden-state capacity.** 5 extra heads × ~50 dims each = ~250
   extra params; trivial. But the model is asked to encode more in
   the same hidden state; could reduce next-token capacity. Mini
   capacity diag suggests body=large mini still has headroom.
3. **ADSR phase derivation imprecision.** First-cut uses
   frames-since-gate-change heuristics, not the SID envelope state
   machine. If `aux_adsr_acc` plateaus low, the label is too noisy;
   drop that head from v2 or invest in the proper envelope sim.
4. **Eval-time silence.** Aux heads consume parameters but don't
   contribute to audio output. Validate that predict-time model
   ignores them cleanly (`predict.py` only uses `lm_head`).
5. **Parser-side label compute cost.** ~5× new columns at parse
   time, computed via a per-voice state walk. Estimated <10% parse
   wallclock impact. Negligible.

## Implementation surface

1. `preframr/core/macros/aux_labels.py` (new): per-voice state-walk
   function returning the 5 label arrays for a df.
2. `preframr/core/reglogparser.py`: call aux label generator after
   `_add_voice_reg`, stash columns in df, persist in `*.0.parquet`.
3. `preframr/train/regdataset.py`: load aux columns from parquet,
   build block-aligned aux arrays alongside inputs/targets, write
   `*.aux.npy`.
4. `preframr/core/model.py`: `PreframrLM` gains `aux_heads` ModuleDict
   and aux loss composition in `training_step` / `validation_step`.
5. `preframr/core/train.py`: dataloader exposes aux tensors; passed
   into model.
6. `preframr/core/args.py`: `--aux-supervision-weight FLOAT`
   (default 0.0 = disabled, backward compatible).
7. `preframr_experiments/specs/aux_supervision_mini.py`: 3-arm
   A/B spec.
8. Tests:
   - Label derivation unit tests (gate transitions, pitch class
     computation, ADSR phase heuristic)
   - Multi-head forward pass shape test
   - Round-trip: aux columns don't affect audio render
   - Mini integration test exercising the full chain

**Estimated effort:** 1-2 days for the impl scaffolding, 1 day for
testing, 1 day for the mini A/B run + analysis. Mid-size project.

## Cross-references

- Per-class audit:
  `accuracy_push_prodlike_4x` audition findings in AGENTS.md
- Refuted alternatives that motivate this:
  `weighted_token_loss_design.md`, `learnable_class_loss_mini`
- Adjacent encoder approach: `voice_trajectory_design.md`
  (input-side annotation; this design is the output-side counterpart)
- Token-class accuracy infra: `profile/token_class_accuracy.py`
- Audio fidelity guardrails: `audio_fidelity.py`
- Constrained-decode landscape: `design/orin_inference_optimization_design.md`

## Out of scope (v1)

- Analytic ADSR envelope amplitude regression head (`amp_log`)
- Cross-voice aux targets (e.g., harmony / interval prediction
  between voices 0 and 1)
- Auxiliary targets at non-VOICE_REG positions
- Predict-time use of aux heads (e.g., as a consistency check or
  for constrained decode)
- Annealing schedule for `lambda_aux`
- Larger model capacity to absorb additional task burden

## Decision rubric

After the mini A/B:

| Outcome | Verdict |
|---|---|
| `aux_quarter` or `aux_full` next-token val_acc ≥ baseline +0.005 AND alphabet unchanged AND aux head accs > 0.5 | **PASS** — promote to canonical, then prodlike |
| Aux head accs > 0.5 but next-token val_acc within ±0.005 of baseline | **PARTIAL** — model learns the targets but transfer to next-token is weak; revisit head wiring (e.g., share representation) |
| Aux head accs near chance | **REFUTE** — labels too noisy or task too hard; drop or redesign |
| Next-token val_acc strictly worse than baseline | **REFUTE** — multi-task interference; aux supervision isn't the right intervention at this scale |
