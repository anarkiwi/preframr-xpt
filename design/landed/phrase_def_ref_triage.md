# Phrase/pattern DEF→REF — recurrence triage (go/no-go before the build)

**SUPERSEDED (2026-06-20):** phrase recurrence landed as the inline backward orderlist in the
step/tracker codec (`../encoding/sid_player_decompiler.md`). Kept for the recurrence-census record.

**Status: TRIAGE RAN (2026-06-17). Verdict = CONDITIONAL GO.** Tool:
`preframr_experiments/audit/phrase_census.py` (torch-free, fogbank). This is the cheap, training-free
de-risking step gating the *pattern* half of tracker DEF→REF — the complement to the instrument half
(`front_loaded_instrument_encoding.md`), which RAN and was within-noise on de-confounded novel
generation. The diagnosis (`../generation/representation_abstraction_probe.md`) says melodic *patterns*
(the `[A][A]` form) are the hard long-range content the model can't represent; the instrument result
proved DEF→REF doesn't help when the referenced content is *locally easy*. So before committing the
hard byte-exact phrase build, measure: **do exact melodic phrases recur enough to reference, and how
much of that recurrence is genuinely-structural vs locally-easy?**

## Method

Per tune, extract per-voice melodic note-event sequences `[(note_index, duration)]` at gate-on onsets
(`note_index` = the encoder's recovered NI series; `duration` = gate-on frame span — the faithful
melodic abstraction a phrase reference would target). Pool n-grams (len 3–16 notes) across voices;
greedily tile each voice with the **longest** n-gram that recurs (count ≥2 = referenceable). Report:
- **covered** — gross fraction of note events inside a recurring phrase.
- **net_of_def** — covered minus one inline DEFINITION per distinct phrase (a REF only saves repeats).
- **trivial/cov** — fraction of coverage from phrases with ≤2 distinct note values (constant/oscillating
  runs — locally-easy, the **instrument trap**: high recurrence but no structural content).
- **non-trivial net** — net_of_def × (1 − trivial). The genuinely-structural reclaimable fraction.
- **EXACT** (absolute pitch+dur) vs **TRANSP** (interval-contour+dur) keys — does transposition matter?

## Results (3 canonical composers, single-speed, limit 120)

| composer | n | mean notes | EXACT covered | net_of_def | trivial/cov | **non-trivial net** | TRANSP non-triv net |
|---|---|---|---|---|---|---|---|
| Daglish  | 74 | 851 | 56.1% | 20.7% | 19.6% | **16.7%** | 19.7% |
| Hubbard  | 86 | 984 | 68.9% | 25.9% | 27.4% | **18.8%** | 22.3% |
| Follin   | 104 | 585 | 73.4% | 30.7% | 53.5% | **14.2%** | 16.4% |

## Verdict — CONDITIONAL GO

1. **The "melody too entropic to reference" null is REFUTED.** Exact melodic phrases recur heavily
   (gross 56–73%); a byte-exact reference-or-inline phrase scheme has ample coverage to build. The
   §7D "melody high-entropy" finding was about the interval distribution *within* phrases, not
   phrase-level recurrence — phrases recur even though their intervals are individually high-entropy.
2. **Exact-match is enough for v1 — do NOT build relative/transposed references.** TRANSP adds only
   ~+3pp non-trivial net over EXACT. The harder relative-phrase encoding is not justified.
3. **The genuinely-structural payoff is MODEST: ~14–19% net of the note stream** — smaller than the
   instrument program's 38% gross. The case for building rests *entirely* on this being the **hard
   structural content** the probe identified, not on compression magnitude. The instrument DEF→REF
   reclaimed MORE and was a de-confounded null — so this triage cannot promise phrase DEF→REF beats
   that null; it confirms only that the premise (referenceable structural recurrence exists) holds.
4. **A large slice of raw recurrence is the instrument trap.** 20–53% of coverage (Follin worst, half)
   is trivial constant/oscillating runs — locally-easy, exactly what made instruments cosmetic. **The
   build MUST reference only non-trivial phrases** (≥3 distinct note values, length ≥ a DEF-worth) or
   it will reproduce the instrument null by referencing locally-easy material.

## If GO — build conditions (carry into the preframr-tokens work-order)

- **Reference non-trivial phrases only** (≥3 distinct notes, min length so the REF saves > the DEF
  costs); leave trivial runs and singletons inline. Per-tune phrase bank (~25–40 distinct), define in
  the preamble (extends the v3 instrument-bank preamble), `PHRASE_REF <id>` in the body, byte-exact
  reference-or-inline (no residual in v1). Phrase boundaries fall on note onsets — the segmentation is
  over the NI/duration note stream, NOT atoms.
- **Alphabet change → EVENT_FORMAT_VERSION v3→v4 + ATOM_CACHE_VERSION bump** (`.atoms.zst` recompute);
  3-repo change at release, but the experiment runs from source mounts.
- **HARD gate: `encode(verify=True)` byte-exact** on the 5 reference drivers + corpus sample. The
  hard part is the same `FLD_NOTE_ON` instrument/duration split already solved for instruments, now
  also factoring pitch (NI) and duration into the referenceable phrase unit.
- **DECISION METRIC for the A/B = de-confounded `copy_novel` novel-content with `PHRASE_REF` excluded
  from the content-atom set** (the exact trap the instrument arc fell into — refs are copyable/easy and
  inflate aggregate `free_running_gap`). Train atoms-only same config (tkvocab 0, seq_len 8192, ep100);
  also read the structural probe ([A][A]-repeat lift) and eval-B content.

## Reproduce

```
PYTHONPATH=.:/scratch/anarkiwi/preframr-tokens python3 -m preframr_experiments.audit.phrase_census \
  --dumps-glob '/scratch/preframr/hvsc/MUSICIANS/H/Hubbard_Rob/*.dump.parquet' --limit 120
```
