# Find and fetch datasets with the `vectordata` CLI

This is the fastest way to get a vector-search benchmark dataset onto your
machine: point the `vectordata` CLI at a catalog, find a dataset — by eye or by
name, metric, dimension, or size — and fetch it into a local cache, all from the
terminal, no code.

> This tutorial is **CLI only**. To read the same datasets from Rust, see
> [Accessing datasets from Rust](./access-datasets-from-rust.md).

**To start, you need a catalog URL** — an `http(s)://` or `s3://` endpoint (or a
local directory) where datasets have been published. Whoever runs the catalog
gives you that URL; every command below works against any catalog.

We use a dataset named `glove` as the running example — substitute whatever your
catalog actually contains (`vectordata explore` and `vectordata datasets list`
both show you).

## Quick start (Linux)

The whole loop in five commands (Linux x64 — other platforms under [Install](#install)):

```bash
curl -fL -o vectordata https://github.com/nosqlbench/vectordata-rs/releases/latest/download/vectordata-x86_64-unknown-linux-musl
chmod +x vectordata
./vectordata config set cache auto                             # cache → largest writable mount
./vectordata config catalog add https://example.com/datasets/  # your catalog URL
./vectordata explore                                           # browse + visualize the datasets
```

The sections below explain every step (and other platforms) in detail.

## Install

The simplest way to get started: download the latest prebuilt binary for your
platform, make it executable, and put it on your `PATH`. Pick your platform:

```bash
# Linux x64
curl -fL -o vectordata https://github.com/nosqlbench/vectordata-rs/releases/latest/download/vectordata-x86_64-unknown-linux-musl
chmod +x vectordata
```

```bash
# Linux arm64
curl -fL -o vectordata https://github.com/nosqlbench/vectordata-rs/releases/latest/download/vectordata-aarch64-unknown-linux-musl
chmod +x vectordata
```

```bash
# macOS (Apple Silicon)
curl -fL -o vectordata https://github.com/nosqlbench/vectordata-rs/releases/latest/download/vectordata-aarch64-apple-darwin
chmod +x vectordata
```

```powershell
# Windows x64 (PowerShell)
curl.exe -fL -o vectordata.exe https://github.com/nosqlbench/vectordata-rs/releases/latest/download/vectordata-x86_64-pc-windows-msvc.exe
```

Then move it onto your `PATH` (e.g. `sudo mv vectordata /usr/local/bin/`). All
assets and checksums are on the [Releases](https://github.com/nosqlbench/vectordata-rs/releases) page.

Prefer to build from source instead? `cargo install --git https://github.com/nosqlbench/vectordata-rs vectordata`

## Enable shell completions (very helpful)

Tab-completion for `vectordata` is genuinely worth setting up — it completes
subcommands and flags, and even live values like catalog and dataset names. Add
one line to your shell's startup file:

```bash
echo 'eval "$(vectordata completions)"' >> ~/.bashrc   # or ~/.zshrc
```

With no arguments, `vectordata completions` auto-detects your shell from `$SHELL`
and prints a sourceable wrapper (what the `eval` above runs). For other shells,
emit the raw script directly — it supports **bash, zsh, fish, elvish, and
powershell**:

```bash
vectordata completions --shell fish > ~/.config/fish/completions/vectordata.fish
vectordata completions --shell powershell   # then add to your $PROFILE
```

## 1. Configure your cache

Everything you fetch — and everything `explore` samples — lands in a local
cache, so set its location once, up front:

```bash
# Pick a specific directory (room to spare, ideally on fast disk):
vectordata config set cache /mnt/fast/vectordata-cache

# …or let vectordata pick the largest writable mount for you:
vectordata config set cache auto       # → <largest-writable-mount>/vectordata-cache
vectordata config mounts               # the candidate mounts it chooses among

vectordata config get cache            # confirm where it landed
```

If you set nothing, the cache defaults to `$VECTORDATA_HOME/cache`; the fetch
commands also print paste-ready setup commands if no location can be resolved.

### How datasets are stored

The cache is a flat, human-navigable directory — **one folder per dataset**, laid
out exactly the way the catalog published it (no content-addressed hash jumble).
After you fetch `glove` it looks like this:

```text
/mnt/fast/vectordata-cache/        # the cache root you set above
└── glove/                         # one directory per dataset, named as the catalog names it
    ├── origin.json                # the catalog URL this copy came from
    ├── base.fvecs                  # data files, laid out exactly as the catalog publishes them
    ├── base.fvecs.mrkl             # per-facet merkle sidecar (chunk hashes + which chunks are valid)
    ├── query.fvecs
    ├── query.fvecs.mrkl
    ├── neighbor_indices.ivecs
    └── neighbor_indices.ivecs.mrkl
```

So you can navigate to `<cache>/glove/`, inspect or copy the files directly, or
delete the folder to drop the dataset from the cache — it's all plain files.

## 2. Point at a catalog

Register a catalog source once; it's remembered in `catalogs.yaml` and used by
every later command.

```bash
vectordata config catalog add https://example.com/datasets/   # an HTTP(S) URL …
vectordata config catalog add s3://my-bucket/datasets/        # … a public S3 bucket …
vectordata config catalog add /shared/datasets                # … or a local directory
vectordata config catalog list                                # what's registered
```

## 3. Explore — the fastest way in

The quickest way to find a dataset is to *look* at one. Run `explore` with no
arguments and it pops a **catalog picker**, lets you browse every dataset
reachable through your catalogs, and opens a vector-space view — norms,
distances, eigenvalues, and PCA projections — of whichever you select, pulling
only the data it samples, on demand. No dataset names to know up front:

```bash
vectordata explore                   # pops the catalog picker → browse → visualize
vectordata explore --dataset glove   # …or jump straight to a known dataset
vectordata datasets                  # a plainer TUI browser (no visualization)
```

This is the easiest on-ramp. The commands below give you the same discovery
explicitly — for when you already know what you want, or you're scripting.

## 4. Find datasets from the command line

List everything reachable through your catalogs:

```bash
vectordata datasets list
```

The `--matching-*` / `--with-*` filters compose conjunctively, so you can narrow
to exactly what you need. Name filters accept a substring, a regex, or a glob:

```bash
# By name:
vectordata datasets list --matching-name 'glove*'

# By shape — metric, dimensionality, and base-vector count (K/M/B suffixes):
vectordata datasets list --with-metric cosine --with-min-dim 768 --with-min-size 1M

# By element type, grouped, with attributes shown:
vectordata datasets list --with-vtype float32 --group-by metric --verbose

# Datasets that carry a particular facet (e.g. filtered-search ground truth):
vectordata datasets list --with-facet filtered_knn
```

Then inspect one in full — its profiles, every facet, the distance metric, and
profile metadata (`base_count`, `maxk`, partitions):

```bash
vectordata datasets describe glove           # defaults to the `default` profile
vectordata datasets describe glove:default   # or name a profile explicitly
```

Datasets are addressed by **name** (`glove`) or **`name:profile`**
(`glove:default`). A dataset can publish several profiles (e.g. different
sub-sample sizes); `describe` and `list --matching-profile` show them.

## 5. Check access before a big download

`ping` walks every facet the profile declares and reads its first record over an
HTTP range request — a fast confirmation that the dataset is reachable and
readable (and that your credentials, if any, work) before you pull gigabytes:

```bash
vectordata datasets ping glove
vectordata datasets ping glove --profile default
```

It reports success/failure per facet.

## 6. Fetch a full profile

Download and cache every facet of a profile, into the cache you set up in step 1:

```bash
vectordata datasets precache glove:default
```

`precache` renders a live per-facet progress meter, downloads each facet,
**verifies it against its SHA-256 chunk hashes**, and persists it to disk.
Once a facet is fully cached, subsequent reads (and `explore`) are zero-copy
`mmap` — no re-download, no re-verification.

## 7. Manage the cache

```bash
vectordata cache list                         # every cached entry, with origin + size
vectordata cache prune --dataset 'glove*'     # remove cached datasets by name/glob
```

`cache prune --dataset` is required to name what to remove — there is no
"prune everything" shorthand, by design.

## Other handy commands

- **`vectordata datasets curlify <dataset-dir-or-yaml>`** — emit a plain `curl`
  script that downloads a dataset's published files, for environments where you'd
  rather not run `vectordata` at fetch time.
- **`vectordata datasets derive --dataset glove --profile default -o ./glove-standalone`**
  — copy one profile *out* of the cache into a fresh, self-standing dataset
  directory (its own `dataset.yaml`, windowed views flattened into real files),
  with no ties back to the donor.

## Authenticated catalogs

Public catalogs need no credentials — everything above works as-is. A **private**
catalog expects a bearer token. Log in once; `vectordata` stores the credential
per endpoint and attaches it automatically to every later request there.

```bash
# Store a pre-issued token (a literal token, or a path to a token file):
vectordata login https://catalog.example.com/ --token vd_xxxxxxxx

# …or exchange a username + password for one — prompts for the password
# (or reads $VECTORDATA_PASSWORD):
vectordata login https://catalog.example.com/ --user alice
```

Check who you are and what you're allowed to see at an endpoint:

```bash
vectordata whoami https://catalog.example.com/   # identity, access level, visible namespaces
vectordata whoami                                # …the endpoint you last logged in to
```

Manage stored credentials:

```bash
vectordata login --list                          # endpoints you have credentials for
vectordata logout https://catalog.example.com/   # forget one
```

Once you're logged in, `datasets list`, `describe`, `ping`, and `precache`
against that endpoint are authenticated transparently — no extra flags. Logging
in interactively also offers to register the endpoint's namespaces as catalogs,
so `datasets list` immediately shows what's published there.

## Where to go next

- **Read these datasets from Rust** → [Accessing datasets from Rust](./access-datasets-from-rust.md)
