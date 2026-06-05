#!/usr/bin/env bash
# Step 02 — cap EACH CONNECTION's download bandwidth, then show that fanning
# out across more connections multiplies aggregate throughput. This is the
# "auto-concurrent streaming scales to the available bandwidth" demo: the
# server limits any one connection, but the client opens one connection per
# in-flight chunk, so more concurrency = more total bandwidth.
#
# Run:  bash 02-connection-limit.sh   (after 01)
source "$(dirname "$0")/env.sh"

CAP=8MiB   # per-connection download cap (bytes/sec)

say "restart vecd with a per-CONNECTION download cap of $CAP/s"
# The cap shapes each TCP connection independently, keyed by the remote
# socket pair (IP:port). Uploads stay unlimited. `restart` re-reads vecd.conf
# for the bind address; the flag overrides config. The equivalent persistent
# setting is `ratelimit_connection_download = 8MiB` in vecd.conf.
vecd restart --ratelimit-connection-download "$CAP"
wait_healthy

say "download with ONE connection (VECTORDATA_DOWNLOAD_CONCURRENCY=1)"
# One worker → one chunk in flight at a time → throughput ≈ a single capped
# connection.
seq_ms=$(download_ms 1)

say "download with EIGHT connections (VECTORDATA_DOWNLOAD_CONCURRENCY=8)"
# Eight workers → up to eight capped connections in flight → ~8× the cap.
par_ms=$(download_ms 8)

echo
echo "per-CONNECTION cap $CAP/s:"
echo "    1 connection : ${seq_ms} ms"
echo "    8 connections: ${par_ms} ms   →  $(speedup "$seq_ms" "$par_ms") faster"
echo
echo "Each connection is throttled, but the client parallelizes chunk fetches"
echo "across many connections, so aggregate download speed scales with"
echo "concurrency — straight up to whatever bandwidth is available."
echo "next: bash 03-client-limit.sh"
