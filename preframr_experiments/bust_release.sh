#!/usr/bin/env bash
# Bust the proxpi mirror cache for a package and poll until <version> is served,
# so a docker/pip rebuild picks up a fresh PyPI release (the mirror caches a
# stale index after a release). With no <version>: bust once, print the latest
# version the mirror serves.
#   bust_release.sh <pkg> [version] [mirror]
# Env: PROXPI_URL (default http://192.168.5.1:5001), BUST_TIMEOUT (default 300s).
set -uo pipefail

pkg="${1:?usage: bust_release.sh <pkg> [version] [mirror]}"
version="${2:-}"
mirror="${3:-${PROXPI_URL:-http://192.168.5.1:5001}}"
norm="${pkg//-/_}"
vre="${version//./\\.}"
timeout="${BUST_TIMEOUT:-300}"
start=$SECONDS

latest() {
  grep -oE "${norm}-[0-9]+\.[0-9]+\.[0-9]+" |
    sort -t. -k1,1n -k2,2n -k3,3n -u | tail -1
}

while :; do
  curl -fsS -X DELETE "$mirror/cache/$pkg" >/dev/null 2>&1 || true
  idx="$(curl -fsS "$mirror/index/$pkg/" 2>/dev/null || true)"
  if [ -z "$idx" ]; then
    echo "WARN: empty index from $mirror/index/$pkg/ (mirror reachable?)" >&2
  fi
  if [ -z "$version" ]; then
    echo "busted $pkg; mirror latest: $(printf '%s' "$idx" | latest)"
    exit 0
  fi
  if printf '%s' "$idx" | grep -qE "${norm}-${vre}([.-])"; then
    echo "OK: $pkg $version served by $mirror"
    exit 0
  fi
  if ((SECONDS - start >= timeout)); then
    echo "TIMEOUT: $pkg $version not in $mirror after ${timeout}s" \
      "(latest: $(printf '%s' "$idx" | latest))" >&2
    exit 1
  fi
  echo "waiting for $pkg $version (re-busting, $((SECONDS - start))s elapsed)..."
  sleep 10
done
