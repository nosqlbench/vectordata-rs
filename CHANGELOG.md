# Changelog

## Unreleased

### Fixes

- **`generate dataset-json` rewrites a `dataset.json` an older veks
  rendered.** It skipped whenever `dataset.json` was newer than
  `dataset.yaml`, so the 3.0.1 re-run of the finalize steps left an
  older build's `dataset.json` in place. It now renders every time and
  writes when the result differs from the file there; an identical file
  keeps its modification time.
- **A new veks re-runs a dataset's finalize steps.** Under the default
  `--provenance config-only`, a step built by another veks counted as
  fresh. That is right for compute steps, and wrong for the finalize
  steps (`dataset.json`, the catalogs, the docs, the merkle trees):
  they project `dataset.yaml`, so a fix to the code that writes them
  never reached a dataset whose projections an older veks had written.
  Finalize steps now also count the binary under every selector; compute
  steps keep the run's selector.
- **`veks run --explain-staleness`** now reports the finalize steps too,
  after the compute steps, each judged as a run would judge it.
- **`veks run --explain-staleness` agrees with the run.** It named each
  upstream by the node a re-run would record, even for an upstream the
  run keeps, so on a dataset whose compute steps an older veks recorded
  it called their dependents and every finalize step stale while
  `veks run` correctly skipped them. A fresh upstream is now named by the
  record it keeps.
- **`veks run --explain-staleness` writes nothing.** It appended to
  `runlog.jsonl` and `.cache/run.log` and rewrote `variables.yaml`, so
  asking why a step would run changed files a publish carries.
- **Pipeline variables are saved only when one changes.** Every
  `veks run` rewrote `variables.yaml` to set `dataset_name` and
  `distance_function` to the values it already held, which gave an
  unchanged file a new modification time.
- **A publish sends only the checksum files that changed, and reports
  them.** After the content uploads it sent every directory's
  `SHA256SUMS` again, one at a time and silently, and then deleted
  orphans silently: over S3, a change touching 4 directories of 186 sat
  on one line for minutes. It now sends only the sums that differ from
  the remote's, on `--concurrency` streams, with a running count, and
  counts its deletions the same way.

## 3.0.0 — 2026-10-15

Every crate in the workspace moves to 3.0.0 and is released with the
others at one version from now on. The library API changes are in
`vectordata`, and in `veks-core` and `veks-pipeline`, which re-export
its types; the other crates move for the lockstep version and their own
APIs are unchanged.

The release breaks the library API: code built against 2.x needs the
changes under [Upgrading](#upgrading-from-25). A 2.x pin
(`vectordata = "2.5"`) keeps building against 2.5.1 until you bump it.

### Highlights

- **Catalogs, entries and datasets can be built at runtime.** A program
  can describe a dataset in code (`DatasetConfig::builder`), list it in
  a catalog entry (`CatalogEntry::builder`), and register its own
  catalogs beside the configured ones (`CatalogBuilder`,
  `CatalogSources::with`). These behave as configured catalogs do: their
  datasets list, open, and report which catalog they came from. See
  `crates/vectordata/examples/register_catalog.rs`.
- **One description, one inheritance rule.** A program opening a
  dataset, the pipeline, the catalog generator, `dataset.json` and the
  docs all read a `dataset.yaml` the same way. Before, the catalog
  generator had its own rule, and published listings that disagreed
  with what an open read (see [Published data](#published-data)).
- **One catalog generator.** `veks prepare catalog generate` and the
  pipeline's `catalog generate` step now write the same files.
- **`knn_entries.yaml` states only what it can state exactly** (see
  below).

### Fixes

- **Configuration files are replaced whole, never rewritten in place.**
  `catalogs.yaml`, `settings.yaml` and `credentials.toml` were written
  by truncating the file and then writing it, so another process reading
  one in between, such as a second `vectordata` command while the
  explorer edits catalogs, could find it empty. Each is now written to a
  temporary file beside it and renamed over it, so a reader sees the old
  file or the new one.
  - A symlinked configuration file stays a symlink; its target is
    replaced, as before.
  - A configuration file keeps its permissions.
  - `credentials.toml` is owner-only (`0600`) from the moment it is
    created, before any token is written to it. Before, it was made
    owner-only after the tokens were written.
- The writes that already used a temporary file and a rename (publish's
  local files and objects, backup restores) now share one
  implementation, which also removes its temporary file when the rename
  fails.

### Upgrading from 2.5

Each removed or changed call, and what replaces it:

| 2.5 | 3.0 |
|---|---|
| `entry.name`, `.path`, `.dataset_type`, `.layout`, `.catalog_file`, `.catalog_name` (public fields of `CatalogEntry`) | `entry.name()`, `.location()`, `.dataset_type()`, `.profiles()` / `.attributes()` / `.layout()`, `.catalog_file()`, `.catalog_name()`. Use `location()` for where the entry lives; `layout()` is only the listing's summary |
| `CatalogEntry { .. }` struct literal | `CatalogEntry::builder(name)…build()` |
| `entry.dataset_home_url()` returning `String` | Returns `&str`; add `.to_string()` where you need an owned value |
| `vectordata::knn_entries` (`KnnEntries`, `KnnEntry`) | `vectordata::catalog::knn_entries`, or open the directory as a catalog |
| `CatalogSources::required()`, `optional()` | `CatalogSources::iter()`, filtering on `Requirement` |
| `CatalogSources::diagnostics()` returning a slice | Returns a `Vec` |
| `catalog::sources::catalog_file_for`, `ensure_trailing_slash` | `vectordata::location::dir_form` |
| `config::verify_catalog_source(source)` | `config::verify_catalog_source(name, source)` |
| `DataSource` | `Location` (`Location::parse`, `Location::from_path`) |
| `GenericTestDataView::new` / `with_attributes` taking a `DataSource` | Take a `Location` |
| `datasets::build_sources` returning `CatalogSources` | Returns `RunSources`; use `.sources()`, or `.open()` for the catalog |
| `datasets::{describe, ping}::run_args(.., at_extra)` | The `at_extra` argument is gone |

New public fields: `DatasetConfig.header`, `DatasetAttributes.other`,
`CatalogDiagnostic.catalog` and `PrecacheRequest.catalog`. A struct
literal of one of these types needs the field, or
`..Default::default()`.

### Behaviour changes

**Library**

- **Opening a dataset** follows the one inheritance rule. It also reads
  facets keyed by aliases (such as jvector's and HDF5's `base`, `query`,
  `gt`, `indices`, `distances`, `train`, `test`), accepts a `dataset.yaml` with no `name` (the directory
  names it), and expands compact sized profiles (`sized:`, `strata:`).
  Datasets written jvector-style, which 2.5 opened with no base, query
  or ground truth, now open fully.
- **An oracle-partition profile** no longer inherits the default
  profile's `metadata_content` and `metadata_predicates`, which describe
  other rows.
- **A profile's facets** are listed in one order everywhere: the
  facets it shares with the default profile, in the default's order;
  then its own; then what a nearer parent adds. This is the order 2.5
  published.
- **`Catalog::of`** lists catalogs in the order they were added. 2.5
  listed required catalogs first, so a set mixing
  `configure_default()` with `add_catalogs(..)` may list in a different
  order.
- **`Error::AmbiguousDataset`** names the catalog document
  (`…/catalog.json`), not only its directory.
- **A remote catalog directory** is also probed for `catalog.yaml`: one
  more request for a catalog that only has `knn_entries.yaml`.
- **Relative entries in `catalogs.yaml`** resolve against that file's
  directory, not the working directory. Names in a nested
  `catalogs.yaml` are kept.
- **A directory holding only `knn_entries.yaml`** finds local copies as
  jvector does. Each file is read from the entry's `cache_dir`, else
  `$DATASET_CACHE_DIR`, else the catalog's own directory, when it is
  there; otherwise it comes from `_defaults.base_url`. An entry key
  given twice keeps its last copy, with a warning, rather than failing
  the file.

**`knn_entries.yaml`**

jvector's format names three whole files per entry: base, query and
ground truth. The generator now lists a profile only when each of the
three is one whole file, with no window, no multi-file series and no
namespace inside a file.

- Every other profile with all three facets is left out. The generator
  reports each one (`knn_entries.yaml omits <dataset>:<profile>:
  <reason>`).
- A directory with nothing left to list has its old `knn_entries.yaml`
  removed, and that is reported too.
- `merkle create` removes a `.mref` whose file is gone.

2.5 listed a sized profile against the whole base file, beside ground
truth computed over the profile's first rows. It listed a sharded base
as its `NNNN` pattern, which jvector cannot load.

**Pipeline**

- **`veks run` leaves `dataset.yaml` alone** when what it would write
  describes the same dataset. Before, a file not in the renderer's
  exact layout was rewritten on every run. That changed its content key
  and sent every finalize step round again.

**CLI**

- **`--at` and `--catalog`**: a catalog named this way that fails to
  load makes any `datasets` command exit with status 1. A configured
  catalog that fails is reported, and the command continues.
- **`datasets describe` and `datasets ping`**: `--at` replaces the
  configured catalogs, as in every other command.
- **`veks catalog generate`** is the pipeline command (`--source`).
  `veks prepare catalog generate <INPUT>` is the manual one.
- **`veks prepare catalog generate`** writes what the pipeline writes:
  - `knn_entries.yaml` at every level;
  - template-derived views;
  - sized profiles read through `variables.yaml`.
- **`.do_not_catalog`**: nothing below a directory marked with it is
  listed in any catalog, including the catalogs above the marker. The
  manual generator used to list such datasets above it.
  `veks check catalogs` follows the same rule.
- **`dataset.yaml` writers** (`prepare bootstrap`, `datasets derive`,
  `generate example-dataset`, `gen_dataset`, `gen_derive`):
  - their output parses to the same description as before;
  - its layout is the renderer's (for example, a structured `strata:`
    block);
  - it carries the strata and tag-schema comments.

  A script that matches these files by their text layout may need
  updating.
- **The explorer:**
  - reports catalogs that fail to load, on its status line and with `✗`
    on the catalogs screen;
  - opens the catalog once per pass rather than once per action;
  - lists the catalogs screen in the order the catalogs were added;
  - finds a dataset to purge case-insensitively.
- **`vectordata::explore::run_with_sources`** opens the explorer over a
  program's own catalog set.

### Published data

Catalogs and docs generated by 2.5 could disagree with the
`dataset.yaml` they came from. Data files and `dataset.yaml` files were
never affected, and neither were programs that open a dataset.
Listings were: anything that read a profile's files from a catalog
(`catalog.json`, `catalog.yaml`, `dataset.json`, generated docs, or
`knn_entries.yaml`).

- **tessera, `*-unfiltered` profiles.** The 2.5 catalogs listed five of
  the default profile's filtered outputs under each of the 46 unfiltered
  profiles: `metadata_results`, and the pre- and post-filtered neighbour
  indices and distances. These outputs are computed over the full base.
  A filtered-search result for an unfiltered profile, scored with files
  taken from the listing, is invalid. A result from opening the profile
  is not affected.
- **`knn_entries.yaml` with sized or sharded profiles**, tessera's
  included. A result for a sized profile that read its base from
  `knn_entries.yaml` searched the whole base, while its ground truth
  covered only the profile's rows, so it is invalid.

tessera's catalogs and docs will be regenerated with this release.
Its `dataset.yaml` and data files do not change. Its `knn_entries.yaml`
will be removed, because no profile in it can be stated exactly.
