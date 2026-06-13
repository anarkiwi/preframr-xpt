#!/usr/bin/env python3
"""Tier-0 free-running pathology probe: teacher-forced vs free-running accuracy over the horizon.

Two reads on a saved checkpoint, both from held-out ``.blocks.npy`` blocks:

  A. teacher-forced top-1 accuracy bucketed by distance-from-block-start — is the
     learned next-token map long-horizon? flat/rising = healthy; decaying = the
     model's GT-conditioned prediction itself fails far from the start (short
     effective context, or a position/cache bug — NOT exposure bias).

  B. teacher-forced vs greedy free-running accuracy aligned at the SAME predicted
     token, off the same prompt boundary, as a gap curve vs horizon. A large,
     widening gap with a FLAT read-A curve is the assumed exposure-bias pathology
     ("good first token, poor afterward").

This is the cheap go/no-go gate for the remediation ladder in
``design/generation/free_running_pathology_remediation_design.md`` (Tier 0). The
measurement math (alignment, bucketing, verdict) is pure-Python and unit-tested in
CI (``tests/test_free_running_gap_audit.py``); the checkpoint forward + greedy
decode reuse ``predict.load_model`` / ``Predictor`` (event_gate.py's proven path)
and run on a GPU host.

CAVEAT (read before acting on a gap): free-running here is single-reference greedy,
so a large gap is *necessary-not-sufficient* for the pathology — multi-modal
targets (encoding_principles P6) depress single-reference match even for a healthy
model. Confirm distributionally with the generation quality gate before promoting
any fix.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# Verdict heuristics (calibrate-then-freeze on the first 2-3 canonical baselines,
# same discipline as GENERALIZE_MIN_VAL_ACC; defaults are starting points).
DEFAULT_THRESHOLDS = {
    "tf_floor": 0.30,  # read-A far-bucket acc below this ...
    "tf_decay": 0.20,  # ... AND this much below the near bucket => short-context/bug
    "gap_hi": 0.15,  # read-B far-horizon gap at/above this => exposure-bias
    "gap_lo": 0.05,  # gap at/below this (near AND far) => healthy
}


# --------------------------------------------------------------------------- #
# Pure measurement (torch-free; unit-tested)
# --------------------------------------------------------------------------- #
def _h_bucket(h: int) -> tuple[int, int]:
    """Log2 horizon bucket [lo, hi] (inclusive) for 1-based distance ``h`` — isolates
    h=1 (the first predicted token) in its own bucket, then doubles."""
    b = max(h, 1).bit_length() - 1
    return (1 << b, (1 << (b + 1)) - 1)


def _bump(acc: dict, h: int, hit: int) -> None:
    e = acc.setdefault(h, [0, 0])
    e[0] += hit
    e[1] += 1


def tf_start_events(block, tf_pred, pad_id: int = 0):
    """Read A: ``(h, hit)`` for each predicted token ``block[j]`` (h = j, the 1-based
    distance from block start), where ``tf_pred[j-1]`` is its teacher-forced argmax
    (``tf_pred`` = argmax of ``model(block[:-1])``)."""
    out = []
    n = min(len(tf_pred), len(block) - 1)
    for i in range(n):  # i = j - 1
        tgt = block[i + 1]
        if tgt == pad_id:
            continue
        out.append((i + 1, 1 if tf_pred[i] == tgt else 0))
    return out


def gap_events(block, tf_pred, fr_gen, prompt_len: int, pad_id: int = 0, tier_of=None):
    """Read B: ``(h, tf_hit, fr_hit, tier)`` aligning teacher-forced and greedy
    free-running predictions at the SAME predicted token ``block[j]``, with
    ``j = prompt_len + (h-1)`` (h >= 1; ``fr_gen[h-1]`` predicts ``block[j]``).
    ``tier`` is ``tier_of[j]`` when a per-id tier list is supplied, else ``None``."""
    out = []
    for k in range(len(fr_gen)):  # k = h - 1
        j = prompt_len + k
        if j >= len(block) or j - 1 >= len(tf_pred) or j < 1:
            break
        tgt = block[j]
        if tgt == pad_id:
            continue
        tier = tier_of[j] if (tier_of is not None and j < len(tier_of)) else None
        out.append(
            (
                k + 1,
                1 if tf_pred[j - 1] == tgt else 0,
                1 if fr_gen[k] == tgt else 0,
                tier,
            )
        )
    return out


def cache_consistency(gen, recompute) -> dict:
    """Compare incremental (KV-cached) greedy ``gen`` against a from-scratch full-recompute
    argmax over the same realized sequence. They MUST match if the decode loop's cache /
    positions are correct; a mismatch is a decode bug, not the pathology. Returns match
    count + consistent fraction + first divergence index (None if fully consistent)."""
    n = min(len(gen), len(recompute))
    if not n:
        return {"n": 0, "match": 0, "consistent_frac": None, "first_divergence": None}
    first = None
    match = 0
    for i in range(n):
        if gen[i] == recompute[i]:
            match += 1
        elif first is None:
            first = i
    return {
        "n": n,
        "match": match,
        "consistent_frac": match / n,
        "first_divergence": first,
    }


def aggregate_cache(checks) -> dict:
    """Pool per-block ``cache_consistency`` dicts. A consistent_frac < 1.0 means the
    incremental decode diverges from full recompute => suspect a cache/position bug before
    trusting any gap verdict."""
    checks = [c for c in checks if c["n"]]
    if not checks:
        return {"checked": True, "n_blocks": 0, "consistent_frac": None}
    total_n = sum(c["n"] for c in checks)
    total_m = sum(c["match"] for c in checks)
    firsts = [
        c["first_divergence"] for c in checks if c["first_divergence"] is not None
    ]
    return {
        "checked": True,
        "n_blocks": len(checks),
        "consistent_frac": total_m / total_n,
        "inconsistent_blocks": len(firsts),
        "earliest_divergence": min(firsts) if firsts else None,
    }


def finalize_curve(acc: dict, fine_max: int = 8) -> dict:
    """{h:[hits,tot]} -> per-h fine rows (h<=fine_max) + log2-bucketed rows."""
    fine = [
        {"h": h, "acc": acc[h][0] / acc[h][1], "n": acc[h][1]}
        for h in sorted(acc)
        if h <= fine_max and acc[h][1]
    ]
    buckets: dict = {}
    for h, (hits, tot) in acc.items():
        b = buckets.setdefault(_h_bucket(h), [0, 0])
        b[0] += hits
        b[1] += tot
    blist = [
        {"h_lo": lo, "h_hi": hi, "acc": hits / tot, "n": tot}
        for (lo, hi), (hits, tot) in sorted(buckets.items())
        if tot
    ]
    return {"fine": fine, "buckets": blist}


def finalize_gap(tf_acc: dict, fr_acc: dict, fine_max: int = 8) -> dict:
    """Merge teacher-forced + free-running {h:[hits,tot]} (identical key sets — bumped
    together) into gap = tf_acc - fr_acc, per-h and per-bucket."""
    a = finalize_curve(tf_acc, fine_max)
    b = finalize_curve(fr_acc, fine_max)
    fine = [
        {
            "h": x["h"],
            "tf_acc": x["acc"],
            "fr_acc": y["acc"],
            "gap": x["acc"] - y["acc"],
            "n": x["n"],
        }
        for x, y in zip(a["fine"], b["fine"])
    ]
    buckets = [
        {
            "h_lo": x["h_lo"],
            "h_hi": x["h_hi"],
            "tf_acc": x["acc"],
            "fr_acc": y["acc"],
            "gap": x["acc"] - y["acc"],
            "n": x["n"],
        }
        for x, y in zip(a["buckets"], b["buckets"])
    ]
    return {"fine": fine, "buckets": buckets}


def _near_far(buckets, key):
    """First and last bucket value for ``key`` (chronological in horizon)."""
    vals = [b[key] for b in buckets if b.get("n", 1)]
    if not vals:
        return None, None
    return vals[0], vals[-1]


def classify(read_a: dict, read_b: dict, thresholds=None):
    """Heuristic verdict from read A (TF-by-startpos) + read B (gap-by-horizon).
    Returns ``(verdict, reason)``. Order matters: a decaying teacher-forced curve is
    diagnosed first (it is a different, worse failure than exposure bias)."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    tf_near, tf_far = _near_far(read_a.get("buckets", []), "acc")
    gap_near, gap_far = _near_far(read_b.get("buckets", []), "gap")
    if tf_far is None or gap_far is None:
        return "inconclusive", "insufficient data"
    if tf_far < th["tf_floor"] and (tf_near - tf_far) >= th["tf_decay"]:
        return (
            "short_context_or_bug",
            f"teacher-forced accuracy decays with position ({tf_near:.2f}->{tf_far:.2f}); "
            "GT-conditioned prediction itself fails far from start — suspect effective "
            "context or a position/cache bug, not exposure bias",
        )
    if gap_far >= th["gap_hi"]:
        return (
            "exposure_bias",
            f"teacher-forced holds (far={tf_far:.2f}) but free-running gap widens to "
            f"{gap_far:.2f} — assumed pathology (single-reference; confirm vs quality gate)",
        )
    if gap_far <= th["gap_lo"] and (gap_near is None or gap_near <= th["gap_lo"]):
        return (
            "healthy",
            f"free-running tracks teacher-forced (gap<={th['gap_lo']}) across the horizon",
        )
    return "inconclusive", f"far-horizon gap {gap_far:.2f} between thresholds"


def _scalars(result: dict) -> dict:
    a = result["read_a_tf_by_startpos"]["buckets"]
    rb = result["read_b_gap_by_horizon"]
    src_name = "content" if "content" in rb else "all"
    src = rb[src_name]
    tf_near, tf_far = _near_far(a, "acc")
    _, gap_far = _near_far(src["buckets"], "gap")
    return {
        "tf_acc_near": tf_near,
        "tf_acc_far": tf_far,
        "gap_h1": src["fine"][0]["gap"] if src["fine"] else None,
        "fr_acc_h1": src["fine"][0]["fr_acc"] if src["fine"] else None,
        "gap_far": gap_far,
        "verdict_source": src_name,
    }


CAVEAT = (
    "Free-running is single-reference greedy: a large gap is necessary-not-sufficient "
    "for the pathology (multi-modal targets / P6 depress single-reference match even "
    "when healthy). Confirm distributionally via generation_quality_gate before acting."
)


def build_result(a_acc, b_tf, b_fr, tier_tf, tier_fr, config) -> dict:
    """Assemble the JSON result from the per-read accumulators (pure; testable)."""
    rb = {"all": finalize_gap(b_tf, b_fr)}
    for tier in sorted(tier_tf):
        rb[tier] = finalize_gap(tier_tf[tier], tier_fr.get(tier, {}))
    result = {
        "config": config,
        "caveat": CAVEAT,
        "read_a_tf_by_startpos": finalize_curve(a_acc),
        "read_b_gap_by_horizon": rb,
    }
    ver_src = rb.get("content", rb["all"])
    verdict, reason = classify(result["read_a_tf_by_startpos"], ver_src)
    result["verdict"] = verdict
    result["verdict_reason"] = reason
    result["scalars"] = _scalars(result)
    return result


# --------------------------------------------------------------------------- #
# Checkpoint forward + greedy decode (torch; GPU host) — lazy-imported
# --------------------------------------------------------------------------- #
def _iter_blocks(blocks_glob: str, n_blocks: int, min_len: int, logger):
    """First qualifying (nonzero) block per file, deduped by song, up to n_blocks."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    out = []
    for path in sorted(glob.glob(blocks_glob, recursive=True)):
        arr = np.load(path)
        for row in np.atleast_2d(arr):
            nz = row[row > 0]
            if len(nz) < min_len:
                continue
            out.append((path, [int(t) for t in nz]))
            break
        if len(out) >= n_blocks:
            break
    logger.info("probe: %u blocks from %s", len(out), blocks_glob)
    return out


def run_probe(args, logger) -> int:
    import torch  # pylint: disable=import-outside-toplevel

    from preframr.inference.predict import (  # pylint: disable=import-outside-toplevel
        Predictor,
        load_model,
    )

    dataset, model, device, _ = load_model(args, logger)
    model = model.to(device)
    model.eval()
    predictor = Predictor(
        args, dataset, model, device, vocab_arrays=None, logger=logger
    )

    tier_of_id = None
    try:
        from preframr.train.model import (  # pylint: disable=import-outside-toplevel
            build_tier_map,
        )

        tier_of_id = build_tier_map(args, model.n_vocab, model.tokens, model.tkmodel)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("tier map unavailable (gap reported all-tier only): %s", exc)

    prompt_len = args.prompt_seq_len
    min_len = prompt_len + args.gen_tokens + 1
    blocks = _iter_blocks(args.blocks_glob, args.n_blocks, min_len, logger)
    if not blocks:
        print("FREE_RUNNING_GAP_RESULT " + json.dumps({"error": "no blocks"}))
        return 1

    a_acc: dict = {}
    b_tf: dict = {}
    b_fr: dict = {}
    tier_tf: dict = {}
    tier_fr: dict = {}
    cache_checks: list = []
    for path, block in blocks:
        x = torch.tensor(block[:-1], dtype=torch.long).unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model.model(x)
        if isinstance(logits, list):
            tf_pred = torch.cat([c.argmax(dim=-1) for c in logits], dim=1)
        else:
            tf_pred = logits.argmax(dim=-1)
        tf_pred = tf_pred.squeeze(0).cpu().tolist()

        if model.model.caches_are_enabled():
            model.model.reset_caches()
        prompt = torch.tensor(block[:prompt_len], dtype=torch.long)
        gen = (
            predictor.predict(prompt, args.gen_tokens, temperature=1.0, top_k=1)
            .cpu()
            .tolist()
        )

        if args.verify_cache:
            seq = block[:prompt_len] + gen[:-1]
            xr = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            if model.model.caches_are_enabled():
                model.model.reset_caches()
            with torch.inference_mode():
                rlogits = model.model(xr)
            if isinstance(rlogits, list):
                rpred = torch.cat([c.argmax(dim=-1) for c in rlogits], dim=1)
            else:
                rpred = rlogits.argmax(dim=-1)
            rpred = rpred.squeeze(0).cpu().tolist()
            recompute = rpred[prompt_len - 1 : prompt_len - 1 + len(gen)]
            cache_checks.append(cache_consistency(gen, recompute))

        for h, hit in tf_start_events(block, tf_pred):
            _bump(a_acc, h, hit)
        tier_of = (
            [tier_of_id.get(int(t), "_unknown") for t in block]
            if tier_of_id is not None
            else None
        )
        for h, tfh, frh, tier in gap_events(
            block, tf_pred, gen, prompt_len, tier_of=tier_of
        ):
            _bump(b_tf, h, tfh)
            _bump(b_fr, h, frh)
            if tier is not None:
                _bump(tier_tf.setdefault(tier, {}), h, tfh)
                _bump(tier_fr.setdefault(tier, {}), h, frh)
        logger.info("probe: scored %s (len=%u)", path, len(block))

    config = {
        "blocks_glob": args.blocks_glob,
        "n_blocks": len(blocks),
        "prompt_len": prompt_len,
        "gen_tokens": args.gen_tokens,
        "sampling": "greedy(top_k=1,temperature=1.0)",
    }
    result = build_result(a_acc, b_tf, b_fr, tier_tf, tier_fr, config)
    result["cache_check"] = (
        aggregate_cache(cache_checks) if args.verify_cache else {"checked": False}
    )
    text = json.dumps(result, indent=2, default=str)
    if args.out is not None:
        Path(args.out).write_text(text)
    print(text)
    logger.info(
        "probe verdict=%s tf_far=%.3f gap_far=%.3f (%s)",
        result["verdict"],
        result["scalars"].get("tf_acc_far") or float("nan"),
        result["scalars"].get("gap_far") or float("nan"),
        result["scalars"]["verdict_source"],
    )
    return 0


def add_probe_args(parser):
    parser.add_argument(
        "--blocks-glob",
        type=str,
        default="/scratch/preframr/train/**/*.blocks.npy",
        help="Held-out blocks to probe (same source as event_gate).",
    )
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=256,
        help="Free-running horizon per block (prompt is --prompt-seq-len).",
    )
    parser.add_argument(
        "--verify-cache",
        action="store_true",
        help="Also recompute each free-run from scratch (no KV cache) and assert it "
        "matches the incremental decode — rules out a cache/position bug masquerading "
        "as the pathology.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main():
    from preframr.args import add_args  # pylint: disable=import-outside-toplevel
    from preframr.utils import get_logger  # pylint: disable=import-outside-toplevel

    parser = add_probe_args(
        add_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    )
    args = parser.parse_args()
    logger = get_logger("INFO")
    sys.exit(run_probe(args, logger))


if __name__ == "__main__":
    main()
