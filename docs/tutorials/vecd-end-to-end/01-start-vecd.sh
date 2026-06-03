#!/usr/bin/env bash
# Step 01 — stand up a private vecd server with a local backing store,
# then create the principals, namespace, and tokens the tutorial needs.
#
# Run:  bash 01-start-vecd.sh
#
# Idempotent-ish: re-running after teardown starts clean. Running twice
# without teardown will refuse to re-init (vecd protects its DB).
source "$(dirname "$0")/env.sh"

mkdir -p "$VECD_DATA_DIR" "$VECD_CONFIG" "$VECD_OBJECTS" "$DEMO_HOME"

say "init control-plane DB + mint a superuser token"
# `init` creates the SQLite control-plane DB under --data-dir and prints
# a one-time superuser token. We capture it for later admin calls.
init_out="$(vecd init --superuser root)"
echo "$init_out"
echo "$init_out" | sed -n 's/^[[:space:]]*token:[[:space:]]*\(vd_[A-Za-z0-9]*\).*/\1/p' > "$ROOT_TOKEN_FILE"
test -s "$ROOT_TOKEN_FILE" || { echo "failed to capture root token"; exit 1; }

say "register a local object backend (the backing store)"
# 'local:<dir>' stores blobs on the filesystem under $VECD_OBJECTS.
vecd backends add store \
  --kind local \
  --endpoint "local:$VECD_OBJECTS" \
  --active

say "create a user 'alice' who will own the dataset namespace"
vecd users add alice --level user

say "create the '$NS_ROOT' namespace, owned by alice, on that backend"
# A namespace is a path prefix governed by one backend + owner. Objects
# at  $NS_ROOT/<anything>  (including  $NS_DATASET/...  and the catalog
# file) live under it.
vecd ns add "$NS_ROOT" --owner alice --backend-config store --active

say "grant access: alice curates the namespace; the public can read it"
vecd bind --to alice  --role curate --ns "$NS_ROOT"
vecd bind --to PUBLIC --role reader --ns "$NS_ROOT"

say "mint a push token for alice (used to upload the dataset)"
tok_out="$(vecd tokens create --user alice --description "tutorial push key" --expires 30d)"
echo "$tok_out"
echo "$tok_out" | sed -n 's/^[[:space:]]*token:[[:space:]]*\(vd_[A-Za-z0-9]*\).*/\1/p' > "$ALICE_TOKEN_FILE"
test -s "$ALICE_TOKEN_FILE" || { echo "failed to capture alice token"; exit 1; }

say "start the daemon in the background on $VECD_BASE"
vecd start --bind "$VECD_HOST:$VECD_PORT"

# Wait for /healthz before handing off to the next step.
say "wait for the server to answer /healthz"
for _ in $(seq 1 60); do
  if curl -fsS "$VECD_BASE/healthz" >/dev/null 2>&1; then ok=1; break; fi
  sleep 0.25
done
test "${ok:-}" = 1 || { echo "vecd did not become healthy"; vecd status || true; exit 1; }

vecd status
echo
echo "vecd is up at $VECD_BASE"
echo "  namespaces:"; vecd ns list
echo "  push token saved to $ALICE_TOKEN_FILE"
