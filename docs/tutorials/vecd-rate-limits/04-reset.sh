#!/usr/bin/env bash
# Step 04 — restore the demo to its un-limited state and tidy up after this
# tutorial. The vecd-end-to-end demo (server, toy dataset, catalog) is left
# exactly as it was; only the rate limits and this tutorial's cached bigvec
# download are cleared.
#
# Run:  bash 04-reset.sh
source "$(dirname "$0")/env.sh"

say "restart vecd with no rate limits (back to the base demo's behavior)"
# No --ratelimit-* flags and none in vecd.conf → unlimited, zero overhead.
vecd restart
wait_healthy

say "evict the bigvec download from the client cache"
vectordata cache prune --dataset bigvec || true

echo
echo "Rate limits cleared; the vecd-end-to-end demo is untouched and still up."
echo "The bigvec object still lives on the server at $(vecd_base)/datasets/bigvec/"
echo "(and its source under \$DEMO/bigvec-src). To remove everything, tear the"
echo "base demo down and delete \$DEMO — see vecd-end-to-end/99-teardown.sh."
