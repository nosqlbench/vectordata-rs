#!/usr/bin/env bash
set -euo pipefail   # fail-fast in this script; NOT in env.sh (would kill a sourcing shell)
# Step 04 — find and fetch the dataset back, addressing the catalog entirely BY
# NAME (`toy-vecd`, registered in step 03) — no URLs. Facets are fetched over
# HTTPS, authenticated by the login stored in step 03.
#
# Run:  bash 04-explore.sh   (after 01–03)
#
# $VECTORDATA_HOME (set in env.sh) isolates ALL client state — catalogs,
# credentials, settings, AND the cache (under $VECTORDATA_HOME/cache) — so your
# real ~/.config/vectordata and ~/.cache stay untouched.
#
# A separate consumer (their own $VECTORDATA_HOME) would do the same one-time
# `config catalog add … --name toy-vecd --trust-self-signed` + `login toy-vecd`
# from step 03, then run exactly the name-based commands below.
source "$(dirname "$0")/env.sh"

say "confirm the catalog + login from step 03 (addressed by name)"
vectordata config catalog list
vectordata whoami toy-vecd

say "list datasets (uses the configured catalogs — no URL)"
vectordata datasets list

say "describe the dataset (profiles, attributes, every facet)"
vectordata datasets describe "toy:default"

say "ping every facet (verifies remote readability over HTTPS)"
vectordata datasets ping toy --profile default

say "precache the profile (download all facets into the local cache)"
vectordata datasets precache "toy:default"

say "show what was cached"
vectordata cache list

echo
echo "explored 'toy-vecd' — all facets readable, addressed by name."
