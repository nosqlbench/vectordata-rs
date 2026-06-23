#!/usr/bin/env bash
set -euo pipefail   # fail-fast in this script; NOT in env.sh (would kill a sourcing shell)
# Step 03 — register the vecd namespace as a NAMED catalog, then generate a
# catalog for the toy dataset and push both into vecd in one authenticated
# `vectordata` operation, so a client can discover it.
#
# Run:  bash 03-upload.sh   (after 01 and 02)
source "$(dirname "$0")/env.sh"

TOKEN="$(cat "$DEMO/alice.token")"   # minted for alice in step 01

say "register the vecd namespace as a named catalog — the ONLY URL reference"
# This is the single place the endpoint URL appears on the client side. After
# this, address the catalog as `toy-vecd` (or its index) everywhere — login,
# push --to, list, describe, ping, precache.
#   --no-verify        : the namespace is still empty (we publish to it below),
#                        so skip the parse+ping gate and just register it.
#   --trust-self-signed : accept vecd's self-signed cert for this origin without
#                        verification (records `trust_self_signed` in
#                        settings.yaml). Insecure — dev/throwaway only. For
#                        production, export the cert (`vecd config tls export`)
#                        and list it under `trusted_ca_certs:` instead, keeping
#                        verification ON.
vectordata config catalog add "$(vecd_base)/datasets/" \
  --name toy-vecd \
  --no-verify \
  --trust-self-signed

say "authenticate BY NAME with vectordata login"
# `login` stores alice's token for this catalog's endpoint under
# $VECTORDATA_HOME. Reads and pushes then use it automatically — no --token
# needed below. `login`/`whoami`/`logout`/`push --to` accept a configured
# catalog name (or 1-based index), resolved to its URL for you — no URL repeat.
# (`vectordata login toy-vecd --user alice --password …` does a password grant
# instead; here we already hold a minted token.)
vectordata login toy-vecd --token "$TOKEN"
vectordata whoami toy-vecd

say "generate a catalog that lists the dataset"
# A vecd namespace becomes a *catalog* when it serves a catalog.json /
# catalog.yaml. `veks prepare catalog generate` walks a publish tree and writes
# one entry per .publish-marked dataset, embedding each dataset's
# profiles/facets. We mark the dataset publishable and point the catalog root's
# .publish_url at the namespace, then generate — leaving catalog.{json,yaml}
# next to toy/ under $DEMO/work.
: > "$DEMO/work/toy/.publish"                  # mark toy publishable
rm -f "$DEMO/work/toy/.publish_url"            # the catalog root owns the binding
printf '%s\n' "$(vecd_base)/datasets/" > "$DEMO/work/.publish_url"
veks prepare catalog generate "$DEMO/work"

say "push the catalog directory (dataset + catalog) to the 'toy-vecd' catalog"
# `vectordata datasets push` understands a CATALOG directory: this single,
# bearer-authenticated push uploads the dataset (→ datasets/toy/…) AND the
# catalog files (→ datasets/catalog.{json,yaml}) in one go — no curl. `--to`
# takes the catalog NAME (resolved to its URL); the stored login supplies the
# token.
#
# `--raw` pushes the tree as-is. The quick 02 dataset isn't a fully-pipelined
# "known-good" dataset (no dedup/zero-vector attestation), so a plain push would
# refuse it; after the full 02-generate-dataset.sh + `veks check` you can drop
# --raw for a provenance-verified push (its own SHA256SUMS round-trip).
vectordata datasets push "$DEMO/work" \
  --to toy-vecd \
  --raw \
  --yes

echo
echo "uploaded the dataset + catalog to the 'toy-vecd' catalog ($(vecd_base)/datasets/)"
echo "  (catalog at $(vecd_base)/datasets/catalog.yaml)"
