#!/usr/bin/env bash
# Step 02 (SHORTCUT) — the one-command alternative to 02-generate-dataset.sh.
#
# The full 02-generate-dataset.sh builds a dataset that exercises EVERY core
# facet (queries, exact + filtered KNN ground truth, metadata, predicates)
# via `veks prepare bootstrap` + `veks run`. That's the right choice when you
# want to see every facet type flow through vecd.
#
# This shortcut instead scaffolds a TRIVIAL base-vectors-only dataset in a
# single command. The rest of the tutorial is identical — 03 uploads it and
# publishes a catalog, 04 explores it over HTTP — there are simply fewer
# facets to describe / ping / precache. Use it when you just want to watch
# vecd serve a dataset without the full pipeline.
#
# Run:  bash 02-quick-dataset.sh   (instead of 02-generate-dataset.sh)
source "$(dirname "$0")/env.sh"

say "scaffold a trivial toy dataset (2000 base + 100 query 16-dim vectors)"
# `veks generate example-dataset` writes the source facets (base + query
# vectors, the default --facets BQ) AND a minimal dataset.yaml together — no
# external data, no pipeline. --name toy keeps the dataset name the later
# steps expect; --force lets you re-run it. (Want every facet — KNN ground
# truth, metadata, predicates? Use 02-generate-dataset.sh instead, which
# bootstraps + runs the full pipeline on top of source files like these.)
veks generate example-dataset \
  --target "$DEMO/work/toy" \
  --name toy \
  --base-count 2000 \
  --query-count 100 \
  --dimension 16 \
  --seed 7 \
  --force

echo
echo "toy dataset ready at $DEMO/work/toy (base + query vectors)"
echo "continue with: bash 03-upload.sh"
