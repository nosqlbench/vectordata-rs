#!/usr/bin/env bash
# Step 04 — register the vecd namespace as a vectordata catalog and explore
# the dataset through the client, fetching facets over HTTP from vecd.
#
# Run:  bash 04-explore.sh   (after 01–03)
#
# $VECTORDATA_HOME (set in env.sh) isolates ALL client state — catalogs,
# credentials, settings, AND the cache (it lands in $VECTORDATA_HOME/cache by
# default) — so your real ~/.config/vectordata and ~/.cache stay untouched.
source "$(dirname "$0")/env.sh"

say "add the vecd namespace as a catalog"
vectordata config catalog add "$(vecd_base)/datasets/"

say "list datasets in the catalog"
vectordata datasets list

say "describe the dataset (profiles, attributes, every facet)"
vectordata datasets describe "toy:default"

say "ping every facet (verifies remote readability over HTTP)"
vectordata datasets ping toy --profile default

say "precache the profile (download all facets into the local cache)"
vectordata datasets precache "toy:default"

say "show what was cached"
vectordata cache list

say "the vecd CLI is also a client — log in and check access with it"
# `vecd login`/`whoami` mirror vectordata's: the token is stored under
# $VECD_CONFIG (keyed by origin) and used automatically. --token accepts the
# token file directly, and once you're logged in the endpoint commands
# (whoami, logout, …) default to that endpoint — so no URL needed.
vecd login "$(vecd_base)/" --token "$DEMO/alice.token"
vecd whoami

echo
echo "explored toy from $(vecd_base)/datasets/ — all facets readable."
