# preframr-xpt image: layers the experiment runner + (post-migration) audits on
# top of the preframr framework image. Because it FROMs anarkiwi/preframr, the
# audits/probes that `import preframr` + torch resolve against the base, so they
# run AND are tested inside this image -- the main repo keeps just the framework.
# Pinned to a released framework version (override BASE to track :latest).
ARG BASE=anarkiwi/preframr:0.2.2
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

# Validate the runner (+ audits once migrated) against the base's preframr.
# test_src_bind_gate exercises host docker --bind-src plumbing (no DinD in the
# build); it is covered host-side by `pytest tests`.
RUN python3 -m pytest tests -q -p no:cacheprovider --ignore=tests/test_src_bind_gate.py

# Smoke: the runner CLI resolves and preframr is importable from the base.
RUN preframr-experiments-run --help >/dev/null \
    && python3 -c "import preframr, preframr_experiments; print('xpt layered on preframr OK')"
