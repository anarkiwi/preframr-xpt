# preframr-xpt image: layers the experiment runner + (post-migration) audits on
# top of the preframr framework image. Because it FROMs anarkiwi/preframr, the
# audits/probes that `import preframr` + torch resolve against the base, so they
# run AND are tested inside this image -- the main repo keeps just the framework.
# Pinned to a released framework version (override BASE to track :latest).
ARG BASE=anarkiwi/preframr:0.3.0
FROM ${BASE}

ARG PIP_OPTS=""

WORKDIR /xpt
COPY pyproject.toml README.md LICENSE ./
COPY preframr_experiments ./preframr_experiments
COPY tests ./tests

# Editable install, --no-deps: the base image already provides torch, numpy,
# pandas, scipy, tokenizers, pytest/pylint/black and the preframr package.
# pyproject declares no deps, so PIP_OPTS only matters once audits add any.
RUN pip install ${PIP_OPTS} --no-deps --break-system-packages -e .

# Audit-only deps: muspy + pretty_midi power preframr_experiments.audit.
# melody_features. Drop --no-deps so the small closure (music21,
# importlib_resources, bidict, mido, etc.) installs cleanly; the upgrade
# strategy keeps the base image's torch/numpy in place.
RUN pip install ${PIP_OPTS} --upgrade-strategy only-if-needed --break-system-packages \
    muspy==0.5.0 pretty_midi==0.2.11 "pytest-xdist>=3.5"

# Validate the runner (+ audits once migrated) against the base's preframr.
# test_src_bind_gate exercises host docker --bind-src plumbing (no DinD in the
# build); it is covered host-side by `pytest tests`.
RUN python3 -m pytest tests -q -n auto --dist worksteal -p no:cacheprovider --ignore=tests/test_src_bind_gate.py

# Smoke: the runner CLI resolves and preframr is importable from the base.
RUN preframr-experiments-run --help >/dev/null \
    && python3 -c "import preframr, preframr_experiments; print('xpt layered on preframr OK')"
