# End-to-end: a private vecd server, a toy dataset, and vectordata

This tutorial walks the full loop with runnable scripts:

1. **Stand up a private `vecd` server** with a local backing store in a
   subdirectory, and create its principals, namespace, and tokens.
2. **Generate a toy dataset** with `veks` that exercises every core facet.
3. **Upload** the dataset into `vecd` and **publish a catalog** for it.
4. **Add the catalog** to a `vectordata` client and **explore** the
   dataset — listing, describing, pinging, and precaching it over HTTP.

Everything the tutorial creates lives under a single throwaway directory
(`./vecd-demo` by default) and the `vectordata` client is pointed at an
isolated `HOME`, so **your real `~/.config/vectordata` and `~/.cache` are
never touched.**

## Prerequisites

You need the three binaries — `vecd`, `veks`, and `vectordata` — plus
`curl` and `python3`-free (no Python needed). Either install them:

```bash
cargo install --path vecd
cargo install --path veks
cargo install --path vectordata
```

…or just build the workspace and run from the checkout — the scripts fall
back to `target/release/<bin>` or `target/debug/<bin>` automatically:

```bash
cargo build -p vecd -p veks -p vectordata --bins
```

(If you have older copies on your `PATH`, the scripts prefer those; export
`VECD_BIN` / `VEKS_BIN` / `VECTORDATA_BIN` to force specific binaries.)

## Run it

```bash
cd docs/tutorials/vecd-end-to-end
bash 01-start-vecd.sh        # init DB, backend, user, namespace, tokens; start daemon
bash 02-generate-dataset.sh  # build + verify the toy dataset locally
bash 03-upload.sh            # push dataset to vecd + publish a catalog
bash 04-explore.sh           # add the catalog and explore over HTTP
bash 99-teardown.sh          # stop the daemon (prints how to delete the dir)
```

Each script sources `env.sh`, which defines the shared paths, ports, and
binary-resolving `vecd`/`veks`/`vectordata` shell wrappers. Override the
location with `DEMO_DIR=/path bash 01-start-vecd.sh` (export it for all
steps).

## What each step shows

### 01 — the server and its AAA
- `vecd init` creates the SQLite control-plane DB under `--data-dir` and
  mints a one-time superuser token.
- `vecd backends add … --kind local --endpoint local:<dir>` registers the
  **backing store** — a plain directory blobs are written into.
- `vecd users add`, `vecd ns add`, `vecd bind`, `vecd tokens create` set up
  a user who **owns** the `datasets` namespace, a **public read** grant, and
  a **push token**. The namespace is a path prefix, so it governs both the
  dataset (`datasets/toy/…`) and the catalog file (`datasets/catalog.yaml`).
- `vecd start` daemonizes; the script waits for `/healthz`.

### 02 — the toy dataset (all facets)
`veks pipeline generate vectors` makes 2000 random vectors; `veks prepare
bootstrap --synthesize-metadata` then produces a single-profile dataset with
**every core facet**: base/query vectors, KNN ground-truth indices and
distances, metadata content, predicates, predicate results, and the
pre-/post-filtered KNN ground truth — plus the optional metadata-layout
(schema) facet. `veks run` materializes and cross-verifies them; `veks check
--check-integrity` confirms every facet file is well-formed.

`--selectivity 0.05` is the knob that makes synthetic predicates match a
dense ~5% of the corpus (it sets the metadata/predicate value range), so the
predicate-evaluation cross-check passes on a small toy.

### 03 — upload + catalog
`vectordata datasets push` uploads every facet to the namespace as a new
version (bearer-authenticated PUTs over vecd's object-REST protocol). Then
`veks prepare catalog generate` writes a `catalog.{json,yaml}` listing the
dataset (with its facets embedded), which we PUT to the namespace root — that
is what turns the namespace into a discoverable **catalog**.

### 04 — explore
`vectordata config add-catalog <namespace-url>` records the catalog; then
`datasets list` / `describe` / `ping` / `precache` browse and read the
dataset, fetching facets over HTTP from vecd (anonymously, via the public
read grant). `ping` reports every facet readable; `precache` pulls them into
the isolated cache.

## A note on object sizes

`vecd` currently **buffers each uploaded object in memory** and applies a
default request-body cap of ~2 MB (per-namespace *quotas* — 50 TiB by
default — are the intended size control, but the body cap bites first). The
toy stays well under this; `--predicate-count` in step 02 keeps the largest
facet (`metadata_results`) small. For real datasets with multi-MB/GB facets,
that cap and the in-memory buffering need to be raised/streamed server-side —
a known limitation, not a property of the protocol.

## A note on the facets

The toy uses a single metadata field range derived from `--selectivity`.
`veks` supports richer metadata; the synthetic-predicate path is calibrated
by `--selectivity` / `--predicate-count` (both settable on `veks prepare
bootstrap`). For real workloads you would import actual metadata instead of
synthesizing it.
