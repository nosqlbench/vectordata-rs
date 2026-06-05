# Shared setup for the vecd rate-limits tutorial — sourced by every step.
#
# This tutorial CONTINUES the vecd-end-to-end demo: it reuses that demo's
# running server, its `datasets` namespace, and alice's push token. Run the
# vecd-end-to-end tutorial first (steps 01–04) and leave the daemon up.
#
# Isolation is identical and shared with that demo (same throwaway dir), so
# your real ~/.config/vectordata and ~/.cache are still never touched:
#
#   VECD_CONFIG     — vecd's config dir; its DB + objects live in data/ under it
#   VECTORDATA_HOME — the client's config AND cache

set -euo pipefail

# Reuse the vecd-end-to-end demo's state by default (same server + client).
# Override: DEMO=/path bash 01-big-dataset.sh (export it for all steps).
: "${DEMO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../vecd-end-to-end" && pwd)/vecd-demo}"
export DEMO

# Run the freshly-built binaries straight from the checkout (cargo-installed
# copies on PATH win; otherwise fall back to the workspace target/).
_repo="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"
[ -n "$_repo" ] && PATH="$_repo/target/release:$_repo/target/debug:$PATH"

# The same two isolation knobs the base demo uses — pointed at its dir.
export VECD_CONFIG="$DEMO/server"
export VECTORDATA_HOME="$DEMO/client"

# The base URL the running daemon actually bound (published by `vecd start`).
vecd_base() { echo "http://$(cat "$VECD_CONFIG/data/vecd.addr")"; }

# Pretty step banners.
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

# Wait for the daemon to answer /healthz (used after each `vecd restart`).
wait_healthy() {
  for _ in $(seq 1 60); do
    curl -fsS "$(vecd_base)/healthz" >/dev/null 2>&1 && return 0
    sleep 0.25
  done
  echo "vecd did not become healthy at $(vecd_base)"; vecd status || true; return 1
}

# Download the bigvec dataset over HTTP with a given chunk-download
# concurrency, from a COLD cache, and print the wall-clock milliseconds.
# VECTORDATA_DOWNLOAD_CONCURRENCY is the client knob: the number of parallel
# ranged GETs (one worker per in-flight 8 MiB chunk). We evict the cached
# copy first so every run is a real network fetch, and route the live
# progress meter to stderr so only the timing reaches stdout.
download_ms() {
  local concurrency="$1" t0 t1
  vectordata cache prune --dataset bigvec >/dev/null 2>&1 || true
  t0=$(date +%s%3N)
  VECTORDATA_DOWNLOAD_CONCURRENCY="$concurrency" \
    vectordata datasets precache "$(vecd_base)/datasets/bigvec/dataset.yaml" >&2
  t1=$(date +%s%3N)
  echo $(( t1 - t0 ))
}

# "<a.b>x" speedup from two millisecond counts, with integer math only.
speedup() { local s="$1" p="$2" t; t=$(( s * 10 / (p < 1 ? 1 : p) )); echo "$((t / 10)).$((t % 10))x"; }
