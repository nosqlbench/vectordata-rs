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
# Run:  bash 02-generate-dataset.sh
source "$(dirname "$0")/env.sh"

mkdir -p "$WORK_DIR"
BASE_VECS="$WORK_DIR/base.fvecs"

say "generate 2000 random 16-dim base vectors (no external data needed)"
veks pipeline generate vectors \
  --output "$BASE_VECS" \
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
  --name "$DATASET_NAME" \
  --output "$DATASET_DIR" \
  --base-vectors "$BASE_VECS" \
  --self-search \
  --query-count 50 \
  --metric L2 \
  --neighbors 10 \
  --seed 7 \
  --synthesize-metadata \
  --metadata-fields 3 \
  --selectivity 0.05 \
  --predicate-count 1000 \
  --synthesis-mode simple-int-eq \
  --synthesis-format ivec \
  --required-facets "BQGDMPRF" \
  --force
# --predicate-count keeps the R (metadata_results) facet small. A larger
# count is fine for real datasets, but note vecd currently buffers each
# uploaded object in memory with a default request-body cap (~2 MB), so a
# toy stays well under it. See README "A note on object sizes".

say "run the pipeline to materialize and cross-verify every facet"
veks run "$DATASET_DIR/dataset.yaml" --output batch

say "verify file-format integrity and facet-spec conformance"
veks check "$DATASET_DIR" --check-integrity

say "the produced facet files"
find "$DATASET_DIR/profiles" -type f ! -name '*.provenance.json' ! -name '*.mref' \
  | sort | sed "s#$DATASET_DIR/##"
echo
echo "toy dataset ready at $DATASET_DIR"
