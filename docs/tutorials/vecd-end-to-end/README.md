# End-to-end: a private vecd server, a toy dataset, and vectordata

This tutorial walks the full loop with runnable scripts:

1. **Stand up a private `vecd` server** with a local backing store in a
   subdirectory, and create its principals, namespace, and tokens.
2. **Generate a toy dataset** with `veks` that exercises every core facet.
3. **Upload** the dataset into `vecd` and **publish a catalog** for it.
4. **Add the catalog** to a `vectordata` client and **explore** the
   dataset — listing, describing, pinging, and precaching it over HTTP.

Everything the tutorial creates lives under a single throwaway directory
(`./vecd-demo` by default). Two environment variables — set once in
`env.sh` — isolate all state under it, so **your real `~/.config/vectordata`
and `~/.cache` are never touched:**

- `VECD_CONFIG` → the server's config dir (its DB and objects live under it),
- `VECTORDATA_HOME` → all client state: config (catalogs, credentials,
  settings) and the cache (`$VECTORDATA_HOME/cache` by default).

Everything else in the steps is a plain literal — the namespace `datasets`,
the dataset `toy`, the loopback bind `127.0.0.1:18443` — that you can read
right where it's used instead of chasing it through a wall of `$VARS`.

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

`env.sh` prepends the workspace `target/{release,debug}` to `PATH`, so the
steps just call `vecd` / `veks` / `vectordata` by name. If you have copies
already on your `PATH` (e.g. from `cargo install`), those win.

## Run it

```bash
cd docs/tutorials/vecd-end-to-end
bash 01-start-vecd.sh        # init DB, backend, user, namespace, tokens; start daemon
bash 02-generate-dataset.sh  # build + verify the toy dataset locally
bash 03-upload.sh            # push dataset to vecd + publish a catalog
bash 04-explore.sh           # add the catalog and explore over HTTP
bash 99-teardown.sh          # stop the daemon (prints how to delete the dir)
```

Each script sources `env.sh`, which sets `DEMO` (the throwaway root),
prepends the build dir to `PATH`, and exports the two isolation variables.
Override the location with `DEMO=/path bash 01-start-vecd.sh` (export it for
all steps).

## What each step shows

### 01 — the server and its AAA
- A one-line `vecd.conf` (just `bind = 127.0.0.1:18443`) under `$VECD_CONFIG`
  is vecd's only operator setting; the DB and pidfile live in `data/` under
  that dir automatically, so no `--data-dir` is threaded through the steps.
- `vecd init --quiet` creates the SQLite control-plane DB and mints a
  one-time superuser token, printing just the token for `$(…)` capture.
- `vecd backends add … --kind local --endpoint local:<dir>` registers the
  **backing store** — a plain directory blobs are written into.
- `vecd users add`, `vecd ns add`, `vecd bind`, `vecd tokens create --quiet`
  set up a user who **owns** the `datasets` namespace, a **public read**
  grant, and a **push token**. The namespace is a path prefix, so it governs
  both the dataset (`datasets/toy/…`) and the catalog (`datasets/catalog.yaml`).
- `vecd start` daemonizes (bind comes from `vecd.conf`), publishes its real
  address to `data/vecd.addr`, and the script waits for `/healthz`.

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

`vecd` **streams each uploaded object straight to storage** in bounded
chunks — it never buffers the whole object in memory and imposes no
request-body cap, so a facet can be any size. The only size gate is the
per-namespace **quota** (50 TiB by default). Step 02 deliberately sets
`--predicate-count 12000`, which makes the `metadata_results` facet a
multi-MB object, to exercise that large-object path end-to-end. (vecd also
exposes the IETF resumable-upload endpoints — `POST`/`PATCH`/`HEAD` with
sparse, parallel, resumable chunks — for clients that want resume-on-failure
or parallel uploads; the plain streaming `PUT` the push engine uses needs
none of that.) Integrity is the client's job: `vectordata push` verifies
content via its separate `SHA256SUMS` round-trip and treats vecd's
envelope ETags as opaque — vecd itself never hashes content.

## A note on the facets

The toy uses a single metadata field range derived from `--selectivity`.
`veks` supports richer metadata; the synthetic-predicate path is calibrated
by `--selectivity` / `--predicate-count` (both settable on `veks prepare
bootstrap`). For real workloads you would import actual metadata instead of
synthesizing it.
