"""Codebook distribution read: the skeleton+codebook pipeline vs production
``full_macros``, on the byte-exact rebake (preframr-tokens 0.42.0), Unigram OFF.

This is the payoff read for the whole 0.42 (PW/filter SWEEP) + unified-macro-flag
plumbing arc -- the codebook+sweep pipeline became reachable from a spec only with
framework 0.2.17 (``--macro-flags`` resolving off the tokens registry). The read is
the op DISTRIBUTION, NOT val_acc: mini training mode-collapses regardless of vocab
(``loop_collapse_rate`` ~1.0, established by ``learnability_full_macros_mini``), so
val_acc inside the collapsed regime is not trustworthy. What we can settle at mini
is whether swapping to the codebook substrate moves the encoding where we expect.

Arms (target first, baseline last):

- codebook: the skeleton+codebook pipeline. ``skeleton_pass`` (the freq substrate
  WavetablePass needs) plus the STAMP/PATCH/SWEEP/held-ARP/WAVETABLE codebooks and
  the PW/filter SWEEP sub-flags, over a shared structural base. NOTE: this is a SWAP
  for ``freq_trajectory_pass``, not an add-on -- ``resolve_flags`` rejects
  skeleton+freq_trajectory (alternative freq substrates), so this arm drops the
  FREQ_TRAJ op (~49% of full_macros' mini stream) for SKEL+ORN+codebooks.
- full_macros: ``macro_config="full_macros"`` (``REGISTERED_MACROS``) -- the current
  production vocab, FREQ_TRAJ-based, codebooks ~0% (stamp/patch/wavetable are
  experimental, not in REGISTERED_MACROS). The reference distribution.

Decisive read (post-hoc, not val_acc): per arm-seed run ``audit_checkpoint_per_class``
in the xpt image, then host-side ``content_tier_report``. Confirm the two encoding
payoffs the arc predicts:
  (a) the PW/filter blowup -- full_macros spends SET/PWM_PRESET/FC_PRESET on PW &
      filter ramps (+16/+19/+6pp vs the FREQ_TRAJ substrate); the codebook arm should
      collapse those to SWEEP.
  (b) STAMP/WAVETABLE codebooks should now REGISTER (were ~0% under full_macros).
This proves the encoding payoff before spending the canonical-tier budget on the
real learnability go/no-go (``learnability_full_macros_mini`` generalised to canonical).
"""

from __future__ import annotations

from preframr_experiments.base import Arm, ExperimentSpec, mini_train_args

_IMAGE = "anarkiwi/preframr:0.2.17"

# Shared structural base both substrates carry (preset / hard-restart / legato /
# voice-block / ctrl-bigram / loop). Excludes any freq substrate -- the codebook
# arm uses skeleton_pass, full_macros uses freq_trajectory_pass.
_BASE = (
    "preset_pass",
    "hard_restart_pass",
    "legato_pass_c2",
    "legato_pass_c4",
    "voice_canonical_block_order",
    "ctrl_bigram_pass",
    "loop_pass",
    "loop_transposed",
)

# The skeleton+codebook substrate: skeleton freq + STAMP/PATCH/WAVETABLE/held-ARP
# codebooks + PW/filter cutoff SWEEP. Conflicts with freq_trajectory_pass.
_CODEBOOK = (
    "skeleton_pass",
    "held_arp",
    "zero_plain",
    "slide_wide",
    "slide_landing",
    "stamp_pass",
    "sweep_pass",
    "sweep_loop",
    "pw_sweep",
    "filter_sweep",
    "wavetable_pass",
    "wt_short",
    "wt_oneshot",
    "patch_pass",
)

# Gate ON for parity with learnability_full_macros_mini (per-op gate/op_acc comes
# free); the decisive read is still the post-hoc content_tier_report distribution.
_TRAIN_ARGS = (
    mini_train_args(body="large").replace("--max-epochs 160", "--max-epochs 60")
    + " --generalization-gate"
)


spec = ExperimentSpec(
    name="codebook_distribution_mini",
    doc=(
        "Codebook distribution read: skeleton+codebook+PW/filter-SWEEP pipeline vs "
        "production full_macros (REGISTERED_MACROS) on the byte-exact (tokens "
        "0.42.0) rebake, --tkvocab 0 (Unigram OFF), --generalization-gate. Read = "
        "the op DISTRIBUTION (mini collapses regardless of vocab): confirm the "
        "PW/filter SET/PWM_PRESET/FC_PRESET blowup becomes SWEEP and STAMP/WAVETABLE "
        "codebooks register. Payoff read for the 0.42 + unified-macro-flag arc, "
        "before the canonical-tier learnability go/no-go. 1 seed, mini body=large."
    ),
    tier="mini",
    image=_IMAGE,
    arms=[
        Arm(label="codebook", macro_flags=_BASE + _CODEBOOK),
        Arm(label="full_macros", macro_config="full_macros"),
    ],
    metrics=[
        "alphabet_size",
        "encoded_tokens_per_song",
        "val_loss_best",
        "val_acc_at_best_loss",
        "epochs_to_best_val_loss",
        "wallclock_train_min",
    ],
    seeds=1,
    seq_len=4096,
    tkvocab=0,
    max_perm=1,
    train_args=_TRAIN_ARGS,
)
