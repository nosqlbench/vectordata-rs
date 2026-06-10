# End-to-end: a private vecd server, a toy dataset, and vectordata

This tutorial walks the full loop, **one copy-pasteable command block per step**:

1. **Stand up a private `vecd` server** with a local backing store, and create
   its principals, namespace, and tokens.
2. **Generate a toy dataset** with `veks` (a one-liner, or the full all-facets
   build).
3. **Upload** the dataset into `vecd` and **publish a catalog** for it.
4. **Add the catalog** to a `vectordata` client and **explore** the dataset —
   listing, describing, pinging, and precaching it over **verified HTTPS**.

Run the blocks **in order, in one shell session** (the variables set in
[Set up your shell](#set-up-your-shell) carry through). Each block is
self-contained copy-pasta; the prose under it explains what just happened.

> Prefer to run it hands-off? The numbered scripts in this directory
> (`bash 01-start-vecd.sh`, `02-…`, `03-…`, `04-…`, `99-teardown.sh`) automate
> the same flow, but currently over **plain HTTP** (port 18443, no TLS) — this
> README is the canonical, **TLS-enabled** walk-through. The TLS steps below
> (`vecd tls generate`/`export`, `trusted_ca_certs`, `https://` URLs) are the
> delta the scripts don't yet carry.

## Isolation: nothing touches your real config

Everything the tutorial creates lives under one throwaway directory
(`./vecd-demo`). Two environment variables isolate all state under it, so
**your real `~/.config/{vecd,vectordata}` and `~/.cache` are never touched:**

- `VECD_CONFIG` → the server's config dir (its DB and objects live under it),
- `VECTORDATA_HOME` → all client state: config (catalogs, credentials,
  settings) **and** the cache.

Everything else is a plain literal you can read right where it's used — the
namespace `datasets`, the dataset `toy`, the loopback bind `127.0.0.1:18443`.

## Prerequisites

You need the three binaries — `vecd`, `veks`, and `vectordata` — plus `curl`.
Build them from the workspace root:

```bash
cargo build -p vecd -p veks -p vectordata --bins
```

…or `cargo install --path vecd` (and `veks`, `vectordata`) to put them on your
`PATH`. The shell setup below prepends the workspace `target/` dirs to `PATH`,
so a plain `cargo build` is enough; installed copies win if present.

## Set up your shell

Run this once. Every later block depends on these variables.

```bash
cd docs/tutorials/vecd-end-to-end

export DEMO="$PWD/vecd-demo"            # one throwaway root for everything
export VECD_CONFIG="$DEMO/server"       # vecd's DB + objects live under here
export VECTORDATA_HOME="$DEMO/client"   # all client config + cache, isolated
export VECD_URL="https://127.0.0.1:18443"      # real TLS (configured in step 1)
export VECD_CA="$VECTORDATA_HOME/vecd-ca.pem"  # vecd's cert, exported for the client to trust

# Run the freshly-built binaries straight from the checkout (no-op if you
# installed them on PATH):
export PATH="$(git rev-parse --show-toplevel)/target/release:$(git rev-parse --show-toplevel)/target/debug:$PATH"

mkdir -p "$VECD_CONFIG"
```

## Step 1 — start a private vecd server

```bash
# 1a. Configure: vecd requires a config before it will run. `config set`
#     creates vecd.conf under $VECD_CONFIG on first use; the DB + pidfile live
#     in data/ under it automatically.
vecd config set bind 127.0.0.1:18443

# 1b. Real TLS. Generate a self-signed cert for the bind host (127.0.0.1 +
#     localhost) and configure vecd to serve HTTPS (writes tls_cert/tls_key into
#     vecd.conf). No openssl needed — vecd does it in-process.
vecd tls generate

# 1c. Export that cert and have the vectordata client trust it — verification
#     stays ON (this is NOT "accept any cert"). cache_dir falls back to
#     $VECTORDATA_HOME/cache, so settings.yaml needs only this line.
mkdir -p "$VECTORDATA_HOME"
vecd tls export "$VECD_CA"
printf 'trusted_ca_certs:\n  - %s\n' "$VECD_CA" > "$VECTORDATA_HOME/settings.yaml"

# 1d. Create the control-plane DB and mint a one-time superuser token (--quiet
#     prints just the token, so we can capture it).
vecd init --superuser root --quiet > "$DEMO/root.token"

# 1e. Register a local object backend — a plain directory blobs are written to.
#     The stored endpoint MUST be local:<absolute-dir>.
vecd backends add store --kind local --endpoint "local:$DEMO/objects" --active

# 1f. A user who will own the namespace.
vecd users add alice --level user

# 1g. The 'datasets' namespace: a path prefix served by the backend, owned by
#     alice. Everything at datasets/* — the dataset and the catalog — lives here.
vecd ns add datasets --owner alice --backend-config store --active

# 1h. Access: alice curates the namespace; the public can read it.
vecd bind --to alice  --role curate --ns datasets
vecd bind --to PUBLIC --role reader --ns datasets

# 1i. A push token for alice (used to upload in step 3).
vecd tokens create --user alice --description "tutorial push key" --expires 30d --quiet \
  > "$DEMO/alice.token"

# 1j. Start the daemon (bind + TLS come from vecd.conf) and wait for health.
#     `--cacert` makes curl verify vecd's cert — the same trust the client uses.
vecd start
for _ in $(seq 1 40); do curl -fsS --cacert "$VECD_CA" "$VECD_URL/healthz" >/dev/null 2>&1 && break; sleep 0.25; done

vecd status
vecd ns list
```

You now have a running **HTTPS** server at `$VECD_URL` (real TLS, cert verified),
a writable namespace `datasets` (active, backed by `store`), and alice's push
token in `$DEMO/alice.token`. `vecd ns list` shows `datasets` as `active` with
`backend=store`.

> **TLS note:** vecd terminates TLS in-process with rustls. The client trusts
> vecd's cert because step 1c added it to `trusted_ca_certs` — verification is on,
> so a wrong/forged cert is rejected. (The insecure alternative, for throwaway
> dev only, is `trust_self_signed: [https://127.0.0.1:18443]` in `settings.yaml`.)

> **Re-running:** start clean after `99-teardown.sh` + removing `vecd-demo`.
> Running 1b twice without a teardown refuses to re-init — vecd protects its DB.

## Step 2 — generate the toy dataset

Pick **one** of the two blocks. Both leave `$DEMO/work/toy/dataset.yaml`, so
steps 3–4 are identical either way.

### Option A — quick (base + query vectors only)

```bash
veks generate example-dataset \
  --target "$DEMO/work/toy" \
  --name toy \
  --base-count 2000 \
  --query-count 100 \
  --dimension 16 \
  --seed 7 \
  --force
```

One command scaffolds the source facets (base + query vectors) and a minimal
`dataset.yaml`. Use this when you just want to watch vecd serve a dataset.

### Option B — full (every core facet)

```bash
# 2b-1. 2000 random 16-dim base vectors (no external data). --workspace keeps
#       this command's bookkeeping inside the demo dir.
veks pipeline generate vectors \
  --workspace "$DEMO/work" \
  --output "$DEMO/work/base.fvecs" \
  --dimension 16 --count 2000 --seed 7

# 2b-2. Bootstrap a dataset.yaml requesting ALL facets, with synthetic metadata.
#       --self-search derives the query set; --synthesize-metadata generates the
#       metadata + predicates + filtered KNN ground truth; --selectivity 0.05
#       keeps predicate matches dense so the cross-verification passes.
veks prepare bootstrap \
  --name toy \
  --output "$DEMO/work/toy" \
  --base-vectors "$DEMO/work/base.fvecs" \
  --self-search --query-count 50 \
  --metric L2 --neighbors 10 --seed 7 \
  --synthesize-metadata --metadata-fields 3 \
  --selectivity 0.05 --predicate-count 12000 \
  --synthesis-mode simple-int-eq --synthesis-format ivec \
  --required-facets "BQGDMPRF" \
  --force

# 2b-3. Materialize + cross-verify every facet, then check integrity.
veks run "$DEMO/work/toy/dataset.yaml" --output batch
veks check "$DEMO/work/toy" --check-integrity
```

This exercises **every core facet**: base/query vectors, exact + filtered KNN
ground truth, metadata, predicates, and predicate results. `--predicate-count
12000` deliberately makes the `metadata_results` facet a multi-MB object to
exercise vecd's large-object streaming (see [object sizes](#a-note-on-object-sizes)).

## Step 3 — authenticate, upload, publish a catalog

```bash
TOKEN="$(cat "$DEMO/alice.token")"   # minted for alice in step 1

# 3a. Log in: stores alice's token for this endpoint under $VECTORDATA_HOME,
#     keyed by origin. Pushes and reads then use it automatically — no --token.
vectordata login "$VECD_URL/" --token "$TOKEN"
vectordata whoami "$VECD_URL/"

# 3b. Push the dataset — a versioned, bearer-authenticated PUT of every facet
#     under datasets/toy/. The first push to an empty namespace creates v1.
vectordata datasets push "$DEMO/work/toy" --to "$VECD_URL/datasets/toy/" --yes

# 3c. Generate a catalog that lists the dataset. Mark toy publishable, point the
#     catalog root's .publish_url at the namespace, then generate.
: > "$DEMO/work/toy/.publish"                       # mark toy publishable
rm -f "$DEMO/work/toy/.publish_url"                 # push wrote this; the root owns it
printf '%s\n' "$VECD_URL/datasets/" > "$DEMO/work/.publish_url"
veks prepare catalog generate "$DEMO/work"

# 3d. Upload the catalog to the namespace ROOT — that is what makes the whole
#     namespace a discoverable catalog (the resolver probes for catalog.json/yaml).
#     `--cacert "$VECD_CA"` so curl verifies vecd's TLS cert.
for c in catalog.json catalog.yaml; do
  curl -fsS --cacert "$VECD_CA" -X PUT "$VECD_URL/datasets/$c" \
    -H "Authorization: Bearer $TOKEN" \
    --data-binary @"$DEMO/work/$c" -o /dev/null \
    -w "  PUT $c -> HTTP %{http_code}\n"
done
```

> **Catalog is now optional:** vecd also *synthesizes* a live `catalog.json` for a
> namespace from the datasets stored under it, so after just the `push` the
> dataset is already discoverable. Steps 3c–3d publish an explicit, curated
> catalog (which takes precedence) — skip them and `datasets list` still finds
> `toy` via the dynamic catalog.

> **Tip:** with a stored login, `--to datasets` (a bare namespace name) also
> works — it resolves to `$VECD_URL/datasets/toy/` for you. The explicit URL is
> used here to show the structure.

## Step 4 — explore over HTTPS

```bash
# 4a. Register the vecd namespace as a catalog, then browse it. The catalog and
#     every facet are fetched over verified HTTPS (the client trusts vecd's cert
#     via trusted_ca_certs from step 1c).
vectordata config catalog add "$VECD_URL/datasets/"
vectordata datasets list
vectordata datasets describe "toy:default"

# 4b. Read every facet over HTTPS (anonymously, via the public read grant).
vectordata datasets ping toy --profile default
vectordata datasets precache "toy:default"
vectordata cache list

# 4c. The vecd CLI is a client too. Its HTTP routes through the same vectordata
#     client, so it trusts vecd's cert via the trusted_ca_certs from step 1c —
#     verified HTTPS, no extra setup.
vecd login "$VECD_URL/" --token "$DEMO/alice.token"
vecd whoami
```

`ping` reports every facet readable; `precache` pulls them into the isolated
cache under `$VECTORDATA_HOME/cache`.

> **Heads-up (interactive):** running `vectordata login` yourself will also offer
> to add this endpoint's namespaces as catalogs — say yes and step 4a's
> `config catalog add` is already done for you.

## Teardown

```bash
vecd stop
```

The demo directory is left in place so you can inspect it. To remove it, from
**this tutorial directory** (verify first with `pwd`):

```bash
rm -rf vecd-demo
```

## A note on object sizes

`vecd` **streams each uploaded object straight to storage** in bounded chunks —
it never buffers the whole object in memory and imposes no request-body cap, so
a facet can be any size. The only size gate is the per-namespace **quota**
(50 TiB by default). Option B's `--predicate-count 12000` makes the
`metadata_results` facet a multi-MB object to exercise that path end-to-end.
(vecd also exposes IETF resumable-upload endpoints — `POST`/`PATCH`/`HEAD` with
sparse, parallel, resumable chunks — for clients that want resume-on-failure;
the plain streaming `PUT` the push engine uses needs none of that.) Integrity is
the client's job: `vectordata push` verifies content via its own `SHA256SUMS`
round-trip and treats vecd's envelope ETags as opaque — vecd never hashes content.

## A note on the facets

Option B's toy uses a single metadata field range derived from `--selectivity`.
`veks` supports richer metadata; the synthetic-predicate path is calibrated by
`--selectivity` / `--predicate-count`. For real workloads you would import
actual metadata instead of synthesizing it.
