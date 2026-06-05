#!/usr/bin/env bash
# Step 01 — add ONE large vector facet to the running vecd demo, big enough
# to span many 8 MiB download chunks so concurrent chunk downloads have
# something to parallelize. Reuses the vecd-end-to-end server, its `datasets`
# namespace, and alice's push token.
#
# Run:  bash 01-big-dataset.sh   (after the vecd-end-to-end demo, with it up)
source "$(dirname "$0")/env.sh"

curl -fsS "$(vecd_base)/healthz" >/dev/null 2>&1 || {
  echo "vecd demo not reachable at $(vecd_base)."
  echo "Run the vecd-end-to-end tutorial first (steps 01–04) and leave it up."
  exit 1
}
TOKEN="$(cat "$DEMO/alice.token")"   # minted for alice in the base demo's step 01

# Keep this dataset in its OWN dir (a sibling of the base demo's work/, which
# carries the catalog's .publish_url) so push records an independent binding.
SRC="$DEMO/bigvec-src"
say "generate a ~64 MiB example dataset (131072 random 128-dim vectors)"
# `veks generate example-dataset` writes the base_vectors.fvec facet AND a
# minimal dataset.yaml in one step. --facets B keeps it base-only — all this
# demo needs is a big object to download. fvec layout = a 4-byte dim header +
# 128 f32 per vector = 516 B; × 131072 ≈ 64 MiB. With no .mref sidecar the
# client reads it over the chunked, TLS-trusted Storage::Http path, which
# fetches 8 MiB ranges — so ~9 chunks the workers can pull in parallel.
veks generate example-dataset \
  --target "$SRC" \
  --name bigvec \
  --facets B \
  --base-count 131072 \
  --dimension 128 \
  --seed 11

say "authenticate, then push it to $(vecd_base)/datasets/bigvec/"
# Log in once (the base demo already did; this is idempotent), then push with
# the stored credential — no --token. Upload is not rate-limited here (only the
# download caps are), so the push runs at full speed. It lands beside the toy
# dataset but is NOT added to the catalog — we read it by direct URL. --no-check
# skips the deeper known-good validation: this base-only example has no
# catalog or merkle sidecar, which is all the download demo needs.
vectordata login "$(vecd_base)/" --token "$TOKEN"
vectordata datasets push "$SRC" \
  --to "$(vecd_base)/datasets/bigvec/" \
  --no-check \
  --yes

echo
echo "bigvec uploaded: a $(du -h "$SRC/base_vectors.fvec" | cut -f1) facet at $(vecd_base)/datasets/bigvec/"
echo "next: bash 02-connection-limit.sh"
