#!/usr/bin/env bash
# Step 01 — stand up a private vecd server with a local backing store, then
# create the principals, namespace, and tokens the tutorial needs.
#
# Run:  bash 01-start-vecd.sh
#
# Re-running after teardown starts clean. Running twice without teardown
# refuses to re-init (vecd protects its DB).
source "$(dirname "$0")/env.sh"

mkdir -p "$VECD_CONFIG"

say "tell vecd to bind a private loopback port (its only operator setting)"
# vecd reads operator settings from $VECD_CONFIG/vecd.conf. Putting the bind
# address here means `vecd start` needs no flags. The DB and pidfile live in
# data/ under this same dir automatically.
printf 'bind = 127.0.0.1:18443\n' > "$VECD_CONFIG/vecd.conf"

say "init the control-plane DB + capture a superuser token"
# `init` creates the SQLite DB and mints a one-time superuser token; --quiet
# prints just the token so we can capture it for later admin calls.
vecd init --superuser root --quiet > "$DEMO/root.token"

say "register a local object backend (the backing store)"
# 'local:<dir>' stores blobs on the filesystem under $DEMO/objects.
vecd backends add store --kind local --endpoint "local:$DEMO/objects" --active

say "create a user 'alice' who will own the dataset namespace"
vecd users add alice --level user

say "create the 'datasets' namespace, owned by alice, on that backend"
# A namespace is a path prefix governed by one backend + owner. Everything at
# datasets/* — the dataset (datasets/toy/…) and the catalog file — lives here.
vecd ns add datasets --owner alice --backend-config store --active

say "grant access: alice curates the namespace; the public can read it"
vecd bind --to alice  --role curate --ns datasets
vecd bind --to PUBLIC --role reader --ns datasets

say "mint a push token for alice (used to upload the dataset)"
vecd tokens create --user alice --description "tutorial push key" --expires 30d --quiet \
  > "$DEMO/alice.token"

say "start the daemon in the background (bind comes from vecd.conf)"
vecd start

# Wait for /healthz before handing off to the next step.
say "wait for the server to answer /healthz"
for _ in $(seq 1 60); do
  if curl -fsS "$(vecd_base)/healthz" >/dev/null 2>&1; then ok=1; break; fi
  sleep 0.25
done
test "${ok:-}" = 1 || { echo "vecd did not become healthy"; vecd status || true; exit 1; }

vecd status
echo
echo "vecd is up at $(vecd_base)"
echo "  namespaces:"; vecd ns list
echo "  push token saved to $DEMO/alice.token"
