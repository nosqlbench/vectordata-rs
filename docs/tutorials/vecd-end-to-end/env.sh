# Shared setup for the vecd end-to-end tutorial — sourced by every step.
#
# The whole demo lives under one throwaway directory ($DEMO) and is fully
# isolated by just two environment variables, so your real configuration
# and cache are never touched:
#
#   VECD_CONFIG     — vecd's config dir; its DB + objects live in data/ under it
#   VECTORDATA_HOME — the client's config AND cache
#
# The endpoint is HTTPS: vecd terminates TLS in-process with a self-signed cert
# (generated in step 01), and the client trusts it via the **relaxed**
# `trust_self_signed` settings key — dev-only, see step 03. Everything else in
# the steps is a plain literal you can read inline.
#
# This file is meant to be SOURCED (by the numbered scripts, or by you for the
# helper vars). It deliberately does NOT run `set -euo pipefail`: that belongs
# in the executed scripts, which each set it themselves. Putting `errexit` here
# would leak into your interactive shell when you source this file, so the next
# command that fails would kill your whole terminal. Keep shell options out of
# sourced files.

# One throwaway root for the entire demo. Override: DEMO=/path bash 01-...
: "${DEMO:=$(pwd)/vecd-demo}"
export DEMO

# Run the freshly-built binaries straight from the checkout without
# installing them. If you've `cargo install`ed vecd/veks/vectordata, those
# copies on PATH win; otherwise these fall back to the workspace target/.
_repo="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "${_repo:-}" ] && PATH="$_repo/target/release:$_repo/target/debug:$PATH"

# The two isolation knobs. Both binaries read these from the environment.
export VECD_CONFIG="$DEMO/server"      # vecd state → $DEMO/server/{data,...}
export VECTORDATA_HOME="$DEMO/client"  # client config + cache, all isolated

# Pinned loopback bind, and the HTTPS origin it serves. Pinning the bind keeps
# the origin a known literal — needed for `trust_self_signed`, which step 01
# writes into the client's settings.yaml before the daemon is even up.
export VECD_BIND="127.0.0.1:18443"
export VECD_URL="https://$VECD_BIND"

# The endpoint base URL. (`vecd daemon start` also publishes the real bound
# address to data/vecd.addr; with the pinned bind above the two always agree.)
vecd_base() { echo "$VECD_URL"; }

# Pretty step banners.
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
