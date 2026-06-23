# End-to-end: a private vecd server, a toy dataset, and vectordata

This tutorial walks the full loop, **one copy-pasteable command block per step**:

1. **Stand up a private `vecd` server** with a local backing store, and create
   its principals, namespace, and tokens.
2. **Generate a toy dataset** with `veks` (a one-liner, or the full all-facets
   build).
3. **Register the catalog by name** (the single place the URL appears on the
   client side), then **upload** the dataset and **publish a catalog** — login
   and push both **by name**.
4. **Find and fetch it back by name** (list, describe, ping, precache) over
   HTTPS — no URLs at all.
5. **(Interactive)** do the catalog + auth configuration inside the explorer's
   **config view** instead — including authenticating a *private* endpoint.

Steps 1–4 are copy-pasteable command blocks; step 5 is an interactive
keystroke walk-through of the same configuration in the TUI.

Run the blocks **in order, in one shell session** (the variables set in
[Set up your shell](#set-up-your-shell) carry through). Each block is
self-contained copy-pasta; the prose under it explains what just happened.

> Prefer to run it hands-off? The numbered scripts in this directory
> (`bash 01-start-vecd.sh`, `02-…`, `03-…`, `04-…`, `99-teardown.sh`) automate
> the same flow over self-signed **HTTPS** with the relaxed `trust_self_signed`
> posture shown below — the same commands this README walks through in prose.

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

You need the three binaries — `vecd`, `veks`, and `vectordata`.
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
export VECD_URL="https://127.0.0.1:18443"  # self-signed TLS, configured in step 1

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

# 1b. Self-signed TLS. Generate a cert + key for the bind host (127.0.0.1 +
#     localhost) and configure vecd to serve HTTPS (writes tls_cert/tls_key into
#     vecd.conf). No openssl needed — vecd does it in-process. The CLIENT trusts
#     this cert later, in step 3, as part of the single `config catalog add`
#     (its --trust-self-signed flag) — so nothing client-side is written here.
vecd config tls generate

# 1c. Create the control-plane DB and mint a one-time superuser token (--quiet
#     prints just the token, so we can capture it).
vecd init --superuser root --quiet > "$DEMO/root.token"

# 1d. Register a local object backend — a plain directory blobs are written to.
#     The stored endpoint MUST be local:<absolute-dir>.
vecd store backends add store --kind local --endpoint "local:$DEMO/objects" --active

# 1e. A user who will own the namespace.
vecd access users add alice --level user

# 1f. The 'datasets' namespace: a path prefix served by the backend, owned by
#     alice. Everything at datasets/* — the dataset and the catalog — lives here.
vecd store ns add datasets --owner alice --backend-config store --active

# 1g. Access: alice curates the namespace; the public can read it. (Drop the
#     PUBLIC bind to require a token for reads — then step 5 authenticates.)
vecd access bind --to alice  --role curate --ns datasets
vecd access bind --to PUBLIC --role reader --ns datasets

# 1h. A push token for alice (used to upload in step 3).
vecd access tokens create --user alice --description "tutorial push key" --expires 30d --quiet \
  > "$DEMO/alice.token"

# 1i. Start the daemon, then block until it answers /healthz. `daemon status
#     --wait` resolves the scheme/bind itself and self-checks over the self-signed
#     cert — no curl, exits non-zero on timeout — so the next step never races a
#     not-yet-listening server.
vecd daemon start
vecd daemon status --wait
vecd store ns list
```

You now have a running **HTTPS** server at `$VECD_URL` (self-signed TLS), a
writable namespace `datasets` (active, backed by `store`), and alice's push
token in `$DEMO/alice.token`. `vecd store ns list` shows `datasets` as `active`
with `backend=store`.

> **TLS note:** vecd terminates TLS in-process with rustls. The client will
> accept vecd's self-signed cert because step 3's `config catalog add
> --trust-self-signed` records this origin under `trust_self_signed` — the
> **relaxed** posture (no verification), fine for a throwaway local demo. For
> anything real, export the cert (`vecd config tls export <file>`) and trust it
> via `trusted_ca_certs:` instead, which keeps verification ON so a
> wrong/forged cert is rejected.

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

## Step 3 — register the catalog by name, then upload + publish

This is the **only** place the endpoint URL appears on the client side. Register
the catalog with a **name** once; from then on — here and in step 4 — refer to it
by that name (or its 1-based index): `login`, `logout`, `whoami`, `ping`,
`token`, `backup`/`restore`, and `push --to` all accept it, so you never paste
the URL again.

```bash
TOKEN="$(cat "$DEMO/alice.token")"   # minted for alice in step 1

# 3a. Register the vecd namespace as a NAMED catalog — the single client-side URL
#     reference. The namespace is still empty (we publish to it just below), so
#     --no-verify skips the parse+ping gate and simply records it. --trust-self-
#     signed accepts vecd's self-signed cert for this origin WITHOUT verification
#     (writes trust_self_signed into settings.yaml) — insecure, dev/throwaway
#     only. (For production, export the cert with `vecd config tls export <file>`
#     and list it under `trusted_ca_certs:` instead, which keeps verification ON.)
vectordata config catalog add "$VECD_URL/datasets/" \
  --name toy-vecd --no-verify --trust-self-signed

# 3b. Log in BY NAME: stores alice's token for this catalog's endpoint under
#     $VECTORDATA_HOME. Pushes and reads then use it automatically — no --token.
vectordata login toy-vecd --token "$TOKEN"
vectordata whoami toy-vecd

# 3c. Generate a catalog that lists the dataset. Mark toy publishable, point the
#     catalog root's .publish_url at the namespace, then generate — leaving
#     catalog.{json,yaml} next to toy/ under $DEMO/work. (.publish_url is catalog
#     metadata — it records where the published datasets are served from.)
: > "$DEMO/work/toy/.publish"                       # mark toy publishable
rm -f "$DEMO/work/toy/.publish_url"                 # the catalog root owns the binding
printf '%s\n' "$VECD_URL/datasets/" > "$DEMO/work/.publish_url"
veks prepare catalog generate "$DEMO/work"

# 3d. Push the catalog DIRECTORY BY NAME — one bearer-authenticated `vectordata`
#     push uploads the dataset (→ datasets/toy/…) AND the catalog files
#     (→ datasets/catalog.{json,yaml}) together; no curl needed. `--to` takes the
#     catalog name (resolved to its URL). The catalog at the namespace root is
#     what makes the whole namespace discoverable (the resolver probes for
#     catalog.json/yaml). The stored login supplies the token. `--raw` pushes the
#     tree as-is, which the quick (Option A) dataset needs — it isn't a pipelined
#     "known-good" dataset; for Option B's fully-checked dataset you can drop
#     --raw for a provenance-verified push.
vectordata datasets push "$DEMO/work" --to toy-vecd --raw --yes
```

> **Catalog is optional:** vecd also *synthesizes* a live `catalog.json` for a
> namespace from the datasets stored under it, so even pushing just `toy/`
> leaves the dataset discoverable. Generating + pushing the explicit catalog
> (above) publishes a curated one (which takes precedence).

> **By name or index:** `--to toy-vecd` could equally be `--to 1` (the 1-based
> position in `vectordata config catalog list`); a `scheme://…` URL is still
> accepted verbatim. With a stored login, `--to datasets` (a bare namespace
> name) also resolves to the endpoint. The vecd CLI mirrors the name-or-URL
> model — `vecd endpoint login`, `vecd endpoint whoami`.

## Step 4 — find and fetch it back, by name

The catalog `toy-vecd` and the login are already in place from step 3, so this
step touches **no URLs at all** — everything is by name. (A separate consumer,
with their own `$VECTORDATA_HOME`, would first run the same one-time `config
catalog add … --name toy-vecd --trust-self-signed` + `login toy-vecd` from step
3, then exactly the commands below.)

```bash
vectordata config catalog list                 # toy-vecd is registered (#1)
vectordata whoami toy-vecd                      # confirm the stored login
vectordata datasets list
vectordata datasets describe "toy:default"
vectordata datasets ping toy --profile default
vectordata datasets precache "toy:default"
vectordata cache list
```

`ping` reports every facet readable; `precache` pulls them into the isolated
cache under `$VECTORDATA_HOME/cache`.

## Step 5 — configure catalogs & auth in the explorer (interactive)

Steps 3–4 did everything from the command line. The same two things — *adding a
catalog* and *providing its credentials* — are also doable interactively, inside
the explorer's **config view**, which is the natural place to manage them and to
see each catalog's live status. This is also a convenient way to authenticate a
catalog whose endpoint is **private** (no `PUBLIC reader` grant) without dropping
to `vectordata login`.

The relaxed self-signed trust recorded by step 3's `--trust-self-signed` is in
force, so HTTPS verification works inside the explorer too. (The explorer's own
`a` add does **not** record self-signed trust — if you skipped step 3, set it
once first via `config catalog add … --trust-self-signed --no-verify`.)

Launch the explorer (no arguments → the dataset picker):

```bash
vectordata explore
```

Press **`Ctrl-G`** to open the config view. It's a tabbed pane — **`←`/`→`** (or
`Tab`/`Shift-Tab`) move between **Catalogs · Columns · Theme · Maintenance**; on
the **Catalogs** tab:

| Key | Action |
|-----|--------|
| `↑` `↓` | move between catalogs |
| `a` | **add** a catalog — prompts for a **name**, a **URL**, and an optional **token** (`Tab` cycles the fields; the token is masked) |
| `c` | set/clear the catalog's **auth token** (empty = clear) |
| `e` | **edit** the highlighted catalog (name / URL / token) and re-validate |
| `x` / `Del` | **remove** the highlighted catalog (`y` to confirm) |
| `Space` | enable/disable (or, for an unvalidated catalog, **re-check** it) |
| `Esc` / `q` | close the config view (back to the picker) |

From the picker list, **`Esc`** (or **`Ctrl-D`**) twice quits the app, and
**`Ctrl-C`** quits immediately.

Each catalog is **colour-coded** — green = active, dim = disabled, ⚠ amber =
saved-but-unvalidated — and shows a **🔒** when a credential is recorded for it.
The highlighted catalog reveals its URL, dataset count, and auth state below the
list. You'll see **`toy-vecd`** here — green with a **🔒** (added and logged in
during step 3).

To practice the interactive **add** flow on it, **remove** it first (**`x`** →
**`y`**), then press **`a`** and fill in:

- **name:** `toy-vecd`
- **url:** `$VECD_URL/datasets/` — i.e. `https://127.0.0.1:18443/datasets/`
- **token:** paste the contents of `$DEMO/alice.token`

On submit the explorer **verifies** the catalog (parse + ping) over the
relaxed-trusted self-signed HTTPS — using the token you just provided — and, on
success, saves it and shows it green with a 🔒. The dataset `toy` appears in the
picker, and opening it streams its facets over **authenticated** HTTPS, with the
credential resolved from the catalog you configured. (The token is stored in
`credentials.toml` under `$VECTORDATA_HOME`, keyed by the catalog URL and
tagged with the name.)

If verification fails (e.g. a typo'd URL), the catalog is **saved but left
disabled** with an ⚠ marker and a message on the console — fix it with **`e`**
(edit) or re-check with **`Space`**, so a mistyped URL is never lost.

> **Make it truly auth-required:** to prove authenticated *download*, omit the
> `vecd access bind --to PUBLIC --role reader` line in step 1g. Anonymous reads then
> fail, and the dataset only becomes readable once you supply alice's token via
> `a` (token field) or `c` — exactly the flow above.

> **List-format `catalogs.yaml`:** if your file is the legacy `- url` list form
> (which can't store names), naming a catalog prompts you on the console to
> **rewrite it to name-based form** (recommended) — accept to convert (comments
> preserved) or decline to cancel.

## Teardown

```bash
vecd daemon stop
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
