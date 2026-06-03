#!/usr/bin/env bash
# Step 03 — upload the toy dataset into vecd, then publish a catalog so a
# vectordata client can discover it.
#
# Run:  bash 03-upload.sh   (after 01 and 02)
source "$(dirname "$0")/env.sh"

TOKEN="$(cat "$ALICE_TOKEN_FILE")"

say "push the dataset to $VECD_BASE/$NS_DATASET/"
# vectordata speaks vecd's object-REST protocol: a versioned PUT of every
# facet file under the namespace, bearer-authenticated. The first push to an
# empty namespace creates version 1.
vectordata datasets push "$DATASET_DIR" \
  --to "$VECD_BASE/$NS_DATASET/" \
  --token "$TOKEN" \
  --yes

say "generate a catalog that lists the dataset"
# A vecd namespace becomes a *catalog* when it serves a catalog.json /
# catalog.yaml. `veks prepare catalog generate` walks a publish tree and
# writes one entry per .publish-marked dataset, embedding each dataset's
# profiles/facets. We mark the dataset publishable and point the catalog
# root's .publish_url at the namespace, then generate.
: > "$DATASET_DIR/.publish"                                  # mark toy publishable
[ -n "$DATASET_DIR" ] && rm -f "$DATASET_DIR/.publish_url"   # push wrote this; the root owns it
printf '%s\n' "$VECD_BASE/$NS_ROOT/" > "$WORK_DIR/.publish_url"
veks prepare catalog generate "$WORK_DIR"

say "upload the catalog to the '$NS_ROOT' namespace root"
# The resolver probes a directory URL for catalog.json then catalog.yaml, so
# placing them at the namespace root makes the whole namespace a catalog.
# Each entry's `path` (toy/dataset.yaml) resolves relative to this location.
for c in catalog.json catalog.yaml; do
  curl -fsS -X PUT "$VECD_BASE/$NS_ROOT/$c" \
    -H "Authorization: Bearer $TOKEN" \
    --data-binary @"$WORK_DIR/$c" -o /dev/null \
    -w "  PUT $c -> HTTP %{http_code}\n"
done

echo
echo "uploaded the dataset to $VECD_BASE/$NS_DATASET/"
echo "published a catalog at $VECD_BASE/$NS_ROOT/catalog.yaml"
