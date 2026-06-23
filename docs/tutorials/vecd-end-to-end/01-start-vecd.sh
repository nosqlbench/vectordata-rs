#!/usr/bin/env bash
set -euo pipefail   # fail-fast in this script; NOT in env.sh (would kill a sourcing shell)
# Step 01 — stand up a private vecd server (self-signed HTTPS) with a local
# backing store, then create the principals, namespace, and tokens the
# tutorial needs.
#
# Run:  bash 01-start-vecd.sh
#
# Re-running after teardown starts clean. Running twice without teardown
# refuses to re-init (vecd protects its DB).
source "$(dirname "$0")/env.sh"

mkdir -p "$VECD_CONFIG" "$VECTORDATA_HOME"

say "configure vecd — a private loopback bind (vecd requires a config to run)"
# `vecd config` manages vecd.conf under $VECD_CONFIG; `set` creates it on first
# use. (For a fresh box you'd usually run `vecd config auto` for safe defaults,
# then tweak — here we pin the demo's port directly.) The DB + pidfile live in
# data/ under this same dir automatically.
vecd config set bind "$VECD_BIND"

say "enable TLS — generate a self-signed cert + key, in-process (no openssl)"
# Writes tls_cert/tls_key into vecd.conf; the daemon then serves HTTPS on the
# bind host (127.0.0.1 + localhost). The CLIENT side trusts this self-signed
# cert later — in step 03, as part of the single `config catalog add` (its
# `--trust-self-signed` flag records the relaxed, dev-only TLS posture). So no
# client-side URL or settings.yaml is touched here.
vecd config tls generate
vecd config get

say "init the control-plane DB + capture a superuser token"
# `init` creates the SQLite DB and mints a one-time superuser token; --quiet
# prints just the token so we can capture it for later admin calls.
vecd init --superuser root --quiet > "$DEMO/root.token"

say "register a local object backend (the backing store)"
# 'local:<dir>' stores blobs on the filesystem under $DEMO/objects.
vecd store backends add store --kind local --endpoint "local:$DEMO/objects" --active

say "create a user 'alice' who will own the dataset namespace"
vecd access users add alice --level user

say "create the 'datasets' namespace, owned by alice, on that backend"
# A namespace is a path prefix governed by one backend + owner. Everything at
# datasets/* — the dataset (datasets/toy/…) and the catalog file — lives here.
vecd store ns add datasets --owner alice --backend-config store --active

say "grant access: alice curates the namespace; the public can read it"
# Drop the PUBLIC bind to make reads auth-required — see the README's step 5
# (provide alice's token via the explorer's config view to read it).
vecd access bind --to alice  --role curate --ns datasets
vecd access bind --to PUBLIC --role reader --ns datasets

say "mint a push token for alice (used to upload the dataset)"
vecd access tokens create --user alice --description "tutorial push key" --expires 30d --quiet \
  > "$DEMO/alice.token"

say "start the daemon and wait until it answers /healthz"
# `daemon start` returns once the process is up; `daemon status --wait` then
# BLOCKS until the HTTPS endpoint actually serves /healthz (or its timeout
# elapses, exiting non-zero) — so the next step never races a not-yet-listening
# server. It resolves the scheme/bind itself and self-checks over the
# self-signed cert; no curl needed.
vecd daemon start
vecd daemon status --wait
echo
echo "vecd is up at $VECD_URL (self-signed TLS)"
echo "  namespaces:"; vecd store ns list
echo "  push token saved to $DEMO/alice.token"
