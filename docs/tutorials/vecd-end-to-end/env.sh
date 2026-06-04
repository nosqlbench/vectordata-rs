# Shared setup for the vecd end-to-end tutorial — sourced by every step.
#
# The whole demo lives under one throwaway directory ($DEMO) and is fully
# isolated by just two environment variables, so your real configuration
# and cache are never touched:
#
#   VECD_CONFIG     — vecd's config dir; its DB + objects live in data/ under it
#   VECTORDATA_HOME — the client's config AND cache
#
# Everything else in the steps is a plain literal you can read inline.

set -euo pipefail

# One throwaway root for the entire demo. Override: DEMO=/path bash 01-...
: "${DEMO:=$(pwd)/vecd-demo}"
export DEMO

# Run the freshly-built binaries straight from the checkout without
# installing them. If you've `cargo install`ed vecd/veks/vectordata, those
# copies on PATH win; otherwise these fall back to the workspace target/.
_repo="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"
[ -n "$_repo" ] && PATH="$_repo/target/release:$_repo/target/debug:$PATH"

# The two isolation knobs. Both binaries read these from the environment.
export VECD_CONFIG="$DEMO/server"      # vecd state → $DEMO/server/{data,...}
export VECTORDATA_HOME="$DEMO/client"  # client config + cache, all isolated

# The base URL the running daemon actually bound. `vecd start` publishes its
# real address to data/vecd.addr, so this is the single source of truth — no
# host/port variables to keep in sync.
vecd_base() { echo "http://$(cat "$VECD_CONFIG/data/vecd.addr")"; }

# Pretty step banners.
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
