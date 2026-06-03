# Shared configuration for the vecd end-to-end tutorial.
#
# Source this from every step:  source ./env.sh
#
# Everything the tutorial creates lives under a single self-contained
# directory ($DEMO_DIR) so the whole thing is trivially disposable and
# never touches your real ~/.config/vectordata or ~/.cache.

set -euo pipefail

# ----------------------------------------------------------------------
# Where everything lives. Override by exporting DEMO_DIR before sourcing.
# ----------------------------------------------------------------------
: "${DEMO_DIR:=$(pwd)/vecd-demo}"
export DEMO_DIR

# vecd control-plane state (SQLite DB, pidfile, logs, vecd.addr).
export VECD_DATA_DIR="$DEMO_DIR/vecd/data"
# vecd config directory (vecd.conf). VECD_CONFIG is read by the vecd binary.
export VECD_CONFIG="$DEMO_DIR/vecd/config"
# The object backing store — a plain local directory. This is the
# "backing store in a subdirectory" the daemon writes blobs into.
export VECD_OBJECTS="$DEMO_DIR/vecd/objects"

# Where the daemon listens. Loopback + a high port keeps it private.
export VECD_HOST="127.0.0.1"
export VECD_PORT="18443"
export VECD_BASE="http://$VECD_HOST:$VECD_PORT"

# Catalog root namespace on the server, and the dataset under it.
export NS_ROOT="datasets"            # holds catalog.yaml
export DATASET_NAME="toy"            # logical dataset name
export NS_DATASET="$NS_ROOT/$DATASET_NAME"

# Local workspace where veks builds the toy dataset before upload.
export WORK_DIR="$DEMO_DIR/work"
export DATASET_DIR="$WORK_DIR/$DATASET_NAME"

# Isolate the vectordata client entirely inside the demo: HOME drives
# ~/.config/vectordata (catalogs.yaml, credentials, settings) and the
# cache. Pointing HOME here means the tutorial leaves your real client
# config untouched.
export DEMO_HOME="$DEMO_DIR/client-home"

# Files holding the tokens minted during setup (written by step 01).
export ROOT_TOKEN_FILE="$DEMO_DIR/root.token"
export ALICE_TOKEN_FILE="$DEMO_DIR/alice.token"

# ----------------------------------------------------------------------
# Binaries. If you've `cargo install`ed them (or they're on PATH) these
# names resolve directly. Otherwise we fall back to the debug builds in
# this checkout's target/ so the tutorial runs from a fresh clone.
# ----------------------------------------------------------------------
_repo_root() { git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null; }
_pick_bin() {
  local name="$1" found
  # `type -P` only matches an executable on PATH — never a shell function
  # or alias — so the wrapper functions below can safely share the name.
  found="$(type -P "$name" 2>/dev/null || true)"
  if [ -n "$found" ]; then echo "$found"; return; fi
  local root; root="$(_repo_root)"
  for build in release debug; do
    if [ -n "$root" ] && [ -x "$root/target/$build/$name" ]; then
      echo "$root/target/$build/$name"; return
    fi
  done
  echo "$name"   # last resort: let the shell report "command not found"
}
export VECD_BIN="${VECD_BIN:-$(_pick_bin vecd)}"
export VEKS_BIN="${VEKS_BIN:-$(_pick_bin veks)}"
export VECTORDATA_BIN="${VECTORDATA_BIN:-$(_pick_bin vectordata)}"
export VECD_CONFIG   # the vecd binary reads this from the environment

# Wrappers that pin the daemon's config/data dirs and the client's HOME.
# (Not exported: each step sources this file, so they're always in scope;
# exporting them would let `type`/`command` see them as the binary name.)
vecd()       { "$VECD_BIN" --data-dir "$VECD_DATA_DIR" "$@"; }
veks()       { "$VEKS_BIN" "$@"; }
vectordata() { HOME="$DEMO_HOME" "$VECTORDATA_BIN" "$@"; }

# Pretty step banners.
say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
