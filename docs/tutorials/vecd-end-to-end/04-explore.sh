#!/usr/bin/env bash
# Step 04 — register the vecd namespace as a vectordata catalog and explore
# the dataset through the client, fetching facets over HTTP from vecd.
#
# Run:  bash 04-explore.sh   (after 01–03)
#
# All client state (catalogs.yaml, settings.yaml, cache) lives under the
# demo's isolated HOME, so your real ~/.config/vectordata is untouched.
source "$(dirname "$0")/env.sh"

mkdir -p "$DEMO_HOME"

say "point the client cache at the demo directory (isolated)"
vectordata config set-cache "$DEMO_DIR/cache"

say "add the vecd namespace as a catalog"
vectordata config add-catalog "$VECD_BASE/$NS_ROOT/"

say "list datasets in the catalog"
vectordata datasets list

say "describe the dataset (profiles, attributes, every facet)"
vectordata datasets describe "$DATASET_NAME:default"

say "ping every facet (verifies remote readability over HTTP)"
vectordata datasets ping "$DATASET_NAME" --profile default

say "precache the profile (download all facets into the local cache)"
vectordata datasets precache "$DATASET_NAME:default"

say "show what was cached"
vectordata cache list

echo
echo "explored $DATASET_NAME from $VECD_BASE/$NS_ROOT/ — all facets readable."
