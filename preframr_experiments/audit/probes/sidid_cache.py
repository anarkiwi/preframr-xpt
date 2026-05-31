"""Corpus-wide sidid engine-label cache: build once, read many.

sidid fingerprints are deterministic per .sid file, so the resid probes should not
re-shell `sidid` per directory on every run. `build()` does one recursive pass over the
corpus (~2 min for HVSC) into a parquet keyed by absolute .sid path; `load()`/`by_dir()`
read it. Rebuild when the corpus or sidid.cfg changes."""
import os
import subprocess

import pandas as pd

CORPUS = "/scratch/preframr/hvsc"
CACHE = os.path.join(CORPUS, "sidid_labels.parquet")
SIDID_CFG = "/scratch/anarkiwi/sidid/sidid.cfg"


def _parse(text, corpus):
    rows = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) >= 2 and parts[0].lower().endswith(".sid"):
            rows.append((os.path.join(corpus, parts[0]), " ".join(parts[1:])))
    return pd.DataFrame(rows, columns=["path", "engine"])


def build(corpus=CORPUS, cache=CACHE, cfg=SIDID_CFG, raw=None):
    """One recursive `sidid -u` pass (or parse a captured `raw` text dump) -> parquet."""
    if raw and os.path.exists(raw):
        with open(raw, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = subprocess.run(
            ["sidid", "-u", corpus], capture_output=True, text=True, check=True,
            env={**os.environ, "SIDIDCFG": cfg}).stdout
    df = _parse(text, corpus)
    df.to_parquet(cache)
    return df


def load(cache=CACHE):
    """{absolute .sid path: engine}."""
    df = pd.read_parquet(cache)
    return dict(zip(df["path"], df["engine"]))


def by_dir(cache=CACHE):
    """{directory: {tune_basename_lower: engine}} -- the shape the resid probes consume."""
    out = {}
    for path, eng in load(cache).items():
        out.setdefault(os.path.dirname(path), {})[os.path.basename(path)[:-4].lower()] = eng
    return out
