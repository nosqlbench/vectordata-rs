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
mkdir -p "$SRC"

say "generate ~64 MiB of random 128-dim vectors (131072 of them)"
# fvecs layout = a 4-byte dim header + 128 f32 per vector = 516 B; × 131072
# ≈ 64 MiB. With no .mref sidecar the client reads this over the chunked,
# TLS-trusted Storage::Http path, which fetches 8 MiB ranges — so ~9 chunks
# the workers can pull in parallel.
veks pipeline generate vectors \
  --workspace "$SRC" \
  --output "$SRC/base_vectors.fvecs" \
  --dimension 128 \
  --count 131072 \
  --seed 11

say "write a minimal single-facet dataset.yaml"
# One facet, no pipeline, no metadata — all this demo needs is a big object
# to download. (The toy dataset from the base demo still has every facet.)
cat > "$SRC/dataset.yaml" <<'YAML'
name: bigvec
attributes:
  distance_function: L2
profiles:
  default:
    base_vectors: base_vectors.fvecs
YAML

say "push it to $(vecd_base)/datasets/bigvec/"
# Upload is not rate-limited in this tutorial (only the download caps are),
# so the push runs at full speed. It lands beside the toy dataset in the same
# namespace but is NOT added to the catalog — we read it by direct URL.
# --no-check skips the known-good dataset validation (this hand-written,
# metadata-free dataset isn't a veks-verified one — it's just a big object).
vectordata datasets push "$SRC" \
  --to "$(vecd_base)/datasets/bigvec/" \
  --token "$TOKEN" \
  --no-check \
  --yes

echo
echo "bigvec uploaded: a $(du -h "$SRC/base_vectors.fvecs" | cut -f1) facet at $(vecd_base)/datasets/bigvec/"
echo "next: bash 02-connection-limit.sh"
