# 3. Catalogs and Publishing

---

## 3.1 Catalog Configuration

Catalogs are lists of locations (HTTP URLs or local paths) where
datasets can be discovered. Configured in
`~/.config/vectordata/catalogs.yaml`:

```yaml
production: https://example.com/datasets/production/
staging: https://internal.example.com/datasets/staging/
local: /mnt/data/local-datasets/
shipped: shipped-catalog.yaml
```

Each key names its catalog; the name is what an ambiguous-dataset
error shows beside each match. The legacy list form (`- <location>`)
is still read, and its entries are named by 0-based index.

Each location must contain (or serve) a `catalog.json` that indexes
the datasets under it, or be a catalog file itself.

- **Relative paths** resolve against the directory holding that
  `catalogs.yaml`, never the process's working directory, so a
  `catalogs.yaml` vendored with a program means the same thing
  wherever it is run from. `~` expands to the home directory.
- **Nesting:** a location that is a directory with its own
  `catalogs.yaml` contributes that file's sources, under that file's
  names. The same holds for a directory passed by path
  (`CatalogSources::add_catalogs`, `--catalogs <dir>`). Any other ad-hoc
  location is named by the next free index.

### When a catalog fails to load

A catalog that cannot be read (the URL does not answer, the file is
gone, it does not parse) never stops a command; the others still load.
Every command says what went wrong on stderr, except shell completion,
whose output would land in the completion list. The explorer shows it
on its status line and marks the catalog with `✗` on the catalogs screen
(Ctrl-G). A catalog named for the run (`--at`, `--catalog`) that fails
is an error: the command exits with status 1.

### CLI management

```bash
veks datasets config catalog add https://example.com/datasets/
veks datasets config catalog list
veks datasets config catalog remove --index 2
```

### Programmatic access

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

let sources = CatalogSources::new().configure_default();
let catalog = Catalog::of(&sources);

for entry in catalog.datasets() {
    println!("{} (profiles: {})", entry.name(), entry.profile_names().join(", "));
}

if let Some(entry) = catalog.find_exact("my-dataset") {
    println!("found at: {}", entry.location());
}

// Glob matching
let matches = catalog.match_glob("my-vectors*");
```

---

## 3.2 Dataset Publishing

Publishing uploads a prepared dataset to a static HTTP server (S3 or
any server supporting Range requests).

### Publish URL binding

A `.publish_url` file in any parent directory sets the publish target:

```
s3://my-bucket/datasets/
```

### Publish workflow

```bash
# Preflight check — verifies everything is ready
veks check

# Publish to the configured S3 destination
veks publish
```

### Nested publish roots

When multiple `.publish_url` files exist in the directory hierarchy,
the most interior (closest to the dataset) root wins. Outer roots
are noted as warnings but do not block publishing.

---

## 3.3 Preflight Checks

`veks check` validates dataset readiness:

| Check | What it verifies |
|-------|-----------------|
| `pipeline-execution` | All pipeline steps completed successfully |
| `pipeline-coverage` | Every publishable file is produced by a pipeline step |
| `dataset-attributes` | Required attributes present (distance_function, zero/dup flags) |
| `publish` | .publish_url is valid with supported transport |
| `merkle` | Every data file has a current .mref hash |
| `integrity` | File geometry is correct (stride, dimensions) |
| `catalogs` | catalog.json exists and is current |
| `extraneous-files` | No unaccounted files in the publish tree |

```bash
veks check                    # run all checks
veks check --check-integrity  # run only integrity
veks check --json             # machine-readable output
veks check --clean            # list extraneous files
veks check --clean-files      # remove extraneous files
```

---

## 3.4 Merkle Integrity

Every published data file has a companion `.mref` file containing a
Merkle hash tree. Consumers verify data integrity chunk-by-chunk
during download.

The pipeline generates merkle trees automatically (`generate-merkle`
step). The `veks check --check-merkle` verifier confirms coverage.

---

## 3.5 Prebuffering and Caching

### Cache location

The cache directory must be configured before any HTTP-backed read.
There is no silent fallback — an unconfigured cache produces an
error with paste-ready setup commands.

Configure via the CLI:

```bash
veks datasets config set cache /mnt/fast-storage/vd-cache
```

Or manually in `~/.config/vectordata/settings.yaml`:

```yaml
cache_dir: /mnt/fast-storage/vd-cache
protect_settings: true
```

Resolution order: `--cache-dir` flag (per-command override) >
`cache_dir:` from `settings.yaml`. If neither is set, every API
that needs the cache (`Storage::open_url`, `veks datasets prebuffer`,
…) fails with a `SettingsError::NotConfigured` whose `Display` impl
prints both the `veks` CLI command and the manual `mkdir`+`echo`
sequence the user can paste.

### Prebuffering

Downloads the facets of the selected profiles for offline access,
verified against merkle hashes. The spec names the profiles with a
selector (see 02-api §2.9): a name, `profile=*` for every profile, or an
expression; a spec with no selector is refused naming the spellings.

```bash
vectordata datasets precache my-dataset:default
vectordata datasets precache my-dataset:profile=*
vectordata datasets precache "tessera:size=10m,predicates=uniform*" --plan
veks datasets precache --dataset my-dataset:default --at https://example.com/datasets/
```

### Cache inspection

```bash
veks datasets cache
```

Shows cached datasets with file counts and sizes.
