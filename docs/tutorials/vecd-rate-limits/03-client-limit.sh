#!/usr/bin/env bash
# Step 03 — cap the CLIENT's *total* download bandwidth (summed across all of
# its connections), then show that concurrency no longer helps: the aggregate
# is bounded no matter how many connections the client opens. This is the
# mirror image of step 02.
#
# Run:  bash 03-client-limit.sh   (after 01; order vs 02 doesn't matter)
source "$(dirname "$0")/env.sh"

CAP=8MiB   # per-client download cap (bytes/sec)

say "restart vecd with a per-CLIENT download cap of $CAP/s"
# Keyed by the remote host (IP): every connection from this client shares one
# bucket, so the cap is on the client's *aggregate*, not each connection.
# Persistent form: `ratelimit_client_download = 8MiB` in vecd.conf.
vecd restart --ratelimit-client-download "$CAP"
wait_healthy

say "download with ONE connection (VECTORDATA_DOWNLOAD_CONCURRENCY=1)"
seq_ms=$(download_ms 1)

say "download with EIGHT connections (VECTORDATA_DOWNLOAD_CONCURRENCY=8)"
par_ms=$(download_ms 8)

echo
echo "per-CLIENT cap $CAP/s:"
echo "    1 connection : ${seq_ms} ms"
echo "    8 connections: ${par_ms} ms   →  $(speedup "$seq_ms" "$par_ms") (≈ no gain)"
echo
echo "All eight connections draw from one shared cap, so opening more buys"
echo "nothing — the opposite of the per-connection case in step 02. Use this"
echo "to bound a single client's total footprint regardless of how hard it"
echo "tries to parallelize."
echo "next: bash 04-reset.sh"
