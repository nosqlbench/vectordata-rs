# 3. Catalogs and Publishing

---

## 3.1 Catalog Configuration

Catalogs are lists of locations (HTTP URLs or local paths) where
datasets can be discovered. Configured in
`~/.config/vectordata/catalogs.yaml`:

```yaml
- https://example.com/datasets/production/
- https://internal.example.com/datasets/staging/
- /mnt/data/local-datasets/
```

Each location must contain (or serve) a `catalog.json` that indexes
the datasets under it.

### CLI management

```bash
veks datasets config add-catalog https://example.com/datasets/
veks datasets config list-catalogs
veks datasets config remove-catalog 2
```

### Programmatic access

```rust
use vectordata::catalog::sources::CatalogSources;
use vectordata::catalog::resolver::Catalog;

let sources = CatalogSources::new().configure_default();
let catalog = Catalog::of(&sources);

for entry in catalog.datasets() {
    println!("{} (profiles: {})", entry.name, entry.profile_names().join(", "));
}

if let Some(entry) = catalog.find_exact("my-dataset") {
    println!("found at path: {}", entry.path);
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
veks datasets config set-cache /mnt/fast-storage/vd-cache
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

Downloads all facets for offline access, verified against merkle hashes:

```bash
veks datasets prebuffer --dataset my-dataset
veks datasets prebuffer --dataset my-dataset:default --at https://example.com/datasets/
```

### Cache inspection

```bash
veks datasets cache
```

Shows cached datasets with file counts and sizes.
