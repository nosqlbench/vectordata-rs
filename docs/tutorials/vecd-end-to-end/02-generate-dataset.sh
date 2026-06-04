#!/usr/bin/env bash
# Step 02 — use veks to build a small, fully-synthetic toy dataset that
# exercises EVERY core facet, then materialize and verify it locally.
#
# Facets produced (veks facet codes):
#   B base_vectors                  — the searchable vectors
#   Q query_vectors                 — query set (extracted via --self-search)
#   G neighbor_indices              — exact KNN ground-truth indices
#   D neighbor_distances            — exact KNN ground-truth distances
#   M metadata_content              — per-vector integer metadata fields
#   P metadata_predicates           — synthesized filter predicates
#   R metadata_results              — per-predicate matching base ordinals
#   F prefiltered_neighbor_indices  — filtered KNN ground truth (+distances)
#   E postfiltered_neighbor_indices — post-filter KNN ground truth (+distances)
# plus the optional metadata_layout (the field schema, as a standalone slab).
#
# The dataset is built under $DEMO/work; nothing here needs the server.
#
# Run:  bash 02-generate-dataset.sh
source "$(dirname "$0")/env.sh"

mkdir -p "$DEMO/work"

say "generate 2000 random 16-dim base vectors (no external data needed)"
# --workspace keeps this ad-hoc command's bookkeeping (its variable cache,
# .cache/) inside the demo's work dir instead of the current directory.
veks pipeline generate vectors \
  --workspace "$DEMO/work" \
  --output "$DEMO/work/base.fvecs" \
  --dimension 16 \
  --count 2000 \
  --seed 7

say "bootstrap a dataset.yaml requesting ALL facets, with synthetic metadata"
# --self-search          derive the query set from the base vectors
# --synthesize-metadata  generate M, and from it P, R, F, E
# --metadata-fields 3    three integer metadata fields per vector
# --selectivity 0.05     each predicate matches ~5% of base vectors; this
#                        sets the value range so predicate matches stay dense
#                        and the strict predicate cross-verification passes
# --required-facets      pin the full facet set
veks prepare bootstrap \
  --name toy \
  --output "$DEMO/work/toy" \
  --base-vectors "$DEMO/work/base.fvecs" \
  --self-search \
  --query-count 50 \
  --metric L2 \
  --neighbors 10 \
  --seed 7 \
  --synthesize-metadata \
  --metadata-fields 3 \
  --selectivity 0.05 \
  --predicate-count 12000 \
  --synthesis-mode simple-int-eq \
  --synthesis-format ivec \
  --required-facets "BQGDMPRF" \
  --force
# --predicate-count drives the size of the R (metadata_results) facet:
# ~12000 predicates × ~100 matching ordinals × 4 bytes ≈ a multi-MB object.
# vecd now *streams* each upload straight to storage (no whole-object
# buffering, no request-body cap), so large facets ride through unimpeded —
# the per-namespace quota (50 TiB default) is the only size gate. See README
# "A note on object sizes".

say "run the pipeline to materialize and cross-verify every facet"
veks run "$DEMO/work/toy/dataset.yaml" --output batch

say "verify file-format integrity and facet-spec conformance"
veks check "$DEMO/work/toy" --check-integrity

say "the produced facet files"
find "$DEMO/work/toy/profiles" -type f ! -name '*.provenance.json' ! -name '*.mref' \
  | sort | sed "s#$DEMO/work/toy/##"
echo
echo "toy dataset ready at $DEMO/work/toy"
