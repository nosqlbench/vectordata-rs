# Rate limits & concurrent-chunk download scaling

A follow-on to **[vecd-end-to-end](../vecd-end-to-end/)**. It shows vecd's two
bandwidth rate limits and how the `vectordata` client's concurrent chunk
downloads scale to the available bandwidth:

- a **per-connection** cap throttles each TCP connection (keyed by the remote
  `IP:port`). Because the client downloads a large object as many parallel
  8 MiB chunks — one connection each — fanning out across more connections
  multiplies the aggregate rate. This is **auto-concurrent streaming scaling
  to the available bandwidth.**
- a **per-client** cap throttles a client's *total* throughput (keyed by the
  remote `IP`, summed across all its connections). Here concurrency buys
  nothing: the aggregate is bounded no matter how many connections open.

Both directions (download and upload) have independent caps; this tutorial
exercises the download caps, where the scaling story is visible.

## Prerequisites

**Run [vecd-end-to-end](../vecd-end-to-end/) first and leave the daemon up.**
This tutorial reuses that demo's running server, its `datasets` namespace, and
alice's push token — it does not stand up its own. The same two environment
variables (`VECD_CONFIG`, `VECTORDATA_HOME`, set in `env.sh`) keep all state
under that demo's throwaway `./vecd-demo` directory, so your real
`~/.config/vectordata` and `~/.cache` are still never touched.

If you installed the binaries with `cargo install`, those win; otherwise the
scripts fall back to the workspace `target/{release,debug}`. Make sure the
build is current — these scripts use the `--ratelimit-*` flags:

```bash
cargo build --release -p vecd -p veks -p vectordata --bins
```

## Run it

```bash
cd docs/tutorials/vecd-rate-limits
bash 01-big-dataset.sh        # add a ~64 MiB single-facet dataset (many chunks)
bash 02-connection-limit.sh   # per-connection cap → concurrency SCALES
bash 03-client-limit.sh       # per-client cap → concurrency is BOUNDED
bash 04-reset.sh              # clear limits + cached bigvec; base demo untouched
```

Each script sources `env.sh`, which points `DEMO` at the vecd-end-to-end
demo's directory by default (override with `DEMO=/path bash 01-…`).

## What you'll see

Step 02 and 03 download the same 64 MiB object twice — once with
`VECTORDATA_DOWNLOAD_CONCURRENCY=1`, once with `=8` — from a cold cache each
time (the script evicts it between runs with `vectordata cache prune`).

With an **8 MiB/s per-connection** cap, the contrast is stark — one capped
connection vs eight in parallel:

```
per-CONNECTION cap 8MiB/s:
    1 connection : 8510 ms        # ≈ 7.8 MiB/s — one capped connection
    8 connections: 1543 ms        # ≈ 48  MiB/s — eight capped connections
                   → 5.5x faster
```

With an **8 MiB/s per-client** cap, the eight connections share one bucket, so
concurrency makes no difference:

```
per-CLIENT cap 8MiB/s:
    1 connection : 8502 ms        # ≈ 7.8 MiB/s
    8 connections: 8303 ms        # ≈ 8.0 MiB/s — same aggregate
                   → 1.0x (≈ no gain)
```

(Exact milliseconds vary with your machine; the *ratio* is the point.)

## What each step shows

### 01 — a big object to download
A real concurrent-chunk download needs an object that spans many chunks. The
toy dataset's facets are all under one 8 MiB chunk, so this step generates
~64 MiB of random 128-dim vectors with `veks` and pushes them as a minimal
single-facet `bigvec` dataset into the same `datasets` namespace. No `.mref`
sidecar is published, so the client reads it over the chunked, TLS-trusted
`Storage::Http` path — 8 MiB ranged GETs, ~9 chunks. The push uses
`--no-check` because this hand-written, metadata-free dataset isn't a
veks-validated one; it's just a big blob to stream. `bigvec` is **not** added
to the catalog — the steps read it by direct URL, so the toy catalog from the
base demo is left exactly as it was.

### 02 — per-connection cap → concurrency scales
`vecd restart --ratelimit-connection-download 8MiB` caps each connection at
8 MiB/s and leaves uploads unlimited. One download worker keeps a single
connection busy (~8 MiB/s); eight workers keep eight capped connections busy
(~8× that), because the client fetches disjoint 8 MiB chunks in parallel. This
is the headline result: **a per-connection limit lets a client soak up the
available bandwidth by parallelizing, exactly as auto-concurrent streaming is
meant to.**

### 03 — per-client cap → concurrency bounded
`vecd restart --ratelimit-client-download 8MiB` caps the client's *aggregate*
at 8 MiB/s (one token bucket per remote IP, shared across all its
connections). Now one connection and eight connections finish in the same
time — opening more connections can't exceed the shared cap. Use this to bound
any single client's footprint regardless of how hard it parallelizes.

### 04 — reset
`vecd restart` with no `--ratelimit-*` flags returns the server to unlimited
(zero overhead when no cap is set), and `vectordata cache prune --dataset
bigvec` evicts the downloaded copy. The vecd-end-to-end demo — server, toy
dataset, catalog — is untouched and still up.

## Configuring the limits

Every cap is reachable from **both** the CLI and `vecd.conf` — flags override
the file, per axis. The four knobs (bytes/sec; suffixes `KiB`/`MiB`/`GiB`,
`KB`/`MB`/`GB`; `0` or unset = unlimited):

| Concern                | CLI flag (`vecd serve`/`start`/`restart`) | `vecd.conf` key                 |
|------------------------|-------------------------------------------|---------------------------------|
| per-connection, down   | `--ratelimit-connection-download`         | `ratelimit_connection_download` |
| per-connection, up     | `--ratelimit-connection-upload`           | `ratelimit_connection_upload`   |
| per-client, down       | `--ratelimit-client-download`             | `ratelimit_client_download`     |
| per-client, up         | `--ratelimit-client-upload`               | `ratelimit_client_upload`       |

For a persistent setup, put the keys in `$VECD_CONFIG/vecd.conf` instead of
passing flags:

```
ratelimit_connection_download = 8MiB
ratelimit_client_download     = 50MiB
```

The **upload** caps shape `PUT` / resumable `PATCH` request bodies the same
way; this tutorial leaves them at `0` so the step-01 push runs at full speed.

## How it works

vecd shapes each transfer with a token bucket — tokens accrue at the
configured bytes/sec up to a small burst, and a chunk waits until its bytes
are covered. A transfer subject to *both* a per-connection and a per-client
cap waits on both, so the tighter one governs. Buckets are created lazily per
key (`IP:port` for connections, `IP` for clients) and swept when idle, so an
unlimited server pays nothing. On the client side,
`VECTORDATA_DOWNLOAD_CONCURRENCY` sets how many chunk-fetch workers (and thus
parallel connections) run; it defaults high enough to saturate a multi-Gbps
link and is what turns a per-connection-capped server into full aggregate
bandwidth.
