# Changelog

## Unreleased

### Fixes

- **`config catalog add .` saves the directory, absolute, under its
  name.** It saved `0: .`: relative, so it resolved only from that
  working directory, and named by index, so `config catalog remove`
  could not find it by name or path. A path is now saved absolute and
  named after its directory (`stage: /data/stage`), and `remove` finds
  it by either spelling. A URL is saved as written.

## 4.0.2 — 2026-10-20

The `datasets` commands read a dataset however it is named — by
catalog name, directory, `dataset.yaml` or URL — and a program that
embeds them can hand them its own catalogs.

### Highlights

- **`datasets describe` takes a path or URL**, as `precache` does: a
  directory, a `dataset.yaml` or a URL, each with an optional
  `:profile` selector. Catalogs are read only for a name.
  `describe::run_via_catalog` takes such a head too.
- **`--at` names a catalog.** Each value, in order, is a URL or an
  existing path; else a configured catalog's number; else the name of
  a configured or program catalog; else a location. `--at fvec --at
  /mnt/stage` reads the program's `fvec` catalog and the directory and
  nothing from `catalogs.yaml`. `describe --at` and `ping --at` repeat
  like the other commands'. `datasets::at_sources` and
  `CatalogSources::named` are the pieces.
- **`datasets::list::run_via_catalog`** lists a catalog the caller
  already assembled, in every output format, with filters and
  `--select`.

### Fixes

- **A name two catalogs hold is ambiguous**, not missing. `describe`
  and `ping` reported it as not found in any catalog; they now list
  each catalog that holds it and say to choose one with `--at`.
  `cache status` says why it prints no profile summary for one.
- **`describe` counts a series' files.** A uniform `__NNNN` series
  printed `shards 1` with its pattern as the one file; it now prints
  the expanded count, the pattern once and every file.
- **`list --select` returns its exit code** instead of ending the
  process, so a program calling it keeps running. The CLI's exit
  status is unchanged.

## 4.0.1 — 2026-10-20

### Highlights

- **`DistanceFunction`** (in `veks-datatypes`, re-exported as
  `vectordata::DistanceFunction`): L2, Cosine, DotProduct, L1,
  Hamming, Jaccard, with one table of spellings — `euclidean`, `cos`,
  `angular`, `ip`, `inner_product`, `mips`, `manhattan`, `taxicab`
  and the rest, case-insensitive, `-` or space as `_`. `distance()` on
  `TestDataView`, `DatasetConfig` and `DatasetAttributes` returns it,
  `None` when undeclared or unrecognised. `veks_simd::Metric` and
  every pipeline metric option parse through it.
- **`shell::bin_main_with_catalogs(invoked, argv, catalogs)`** runs
  the embedded `vectordata` CLI over its program's catalogs: `datasets
  list`, `describe`, `ping`, `precache`, `derive` and `explore` see
  them after the configured ones and before `--catalog`'s; `--at`
  replaces them all. `CatalogSources::extend` appends one set to
  another.
- **`TestDataGroup::name()`** returns the dataset's cache identity:
  the catalog entry's name, a directory's last segment, or
  `from_config`'s name; `None` at a server's root.

### Fixes

- **`veks prepare --metric angular`** emitted KNN steps without the
  cosine strategy they require; every spelling of cosine now gets it.
- **Publishing 4.0.0 failed**: `veks` and `veks-core` pinned
  `veks-datatypes = "=3.0.4"` as a dev-dependency, which the version
  bump did not rewrite. The dev-dependencies are path-only.
- The 4.0.0 deprecations carry `since = "4.0.0"`, and the rustdoc
  examples and sysref call the `*_as` forms; doctests in `vectordata`,
  `veks-io` and `veks-datatypes` deny deprecated items.

## 4.0.0 — 2026-10-20

One element model for the workspace, and a typed facet access API on
it. A new leaf crate, `veks-datatypes`, holds the twelve element
types (`bf16` is new), the one extension table, the `Element` trait
and the conversion table that says how a facet of one type may be
read as another: natively, widened when lossless, checked per value
across a sign, or refused — never a reinterpretation of the bytes.
`vectordata`, `veks-io`, `veks-core` and `veks-pipeline` are all on
it; their own copies of the element enum, the facet enum and the
extension tables are gone.

This changes the library API: see [Upgrading](#upgrading-from-30).
Every pinned call of 3.0 still compiles, deprecated, beside its
replacement. The design is `docs/design/srd-element-access.md`.

### Highlights

- **Any facet at any type: `view.facet_as::<T>(name)`** and the
  per-facet `base_vectors_as::<T>()`, `query_vectors_as`,
  `neighbor_indices_as`, `neighbor_distances_as`, the filtered six,
  `metadata_results_as` — on `TestDataViewExt`, implemented for every
  `TestDataView`. A `bf16`, `f16` or `u8` base reads as `f32`; `u8`
  codes read as `u8`, zero-copy; an `f32` base asked for as `u8` is
  `IoError::ElementMismatch`. `open_facet_native(name)` opens a facet
  at whatever type it holds (`AnyVectorReader`), `open_vec_as::<T>`
  and `open_vec_native` do the same for a raw path or URL.
- **`VectorReader<T>` says what it is**: `as_slice()` borrows a whole
  header-free column from the mapping, `conversion()` says whether the
  reader converts, `native_element_type()` says what the file holds.
  `ScalarReader<T>` reads a column natively; `ConvertingReader<N, T>`
  carries a widening or cross-sign read and reports the first value
  that does not fit as `IoError::ValueOverflow` at its ordinal.
- **`bf16`** everywhere: `.bf16`, `.bf16vecs`, `.bf16vvecs`, a type of
  its own in the readers, the writers, the facet spec, the pipeline's
  analysis commands and the explorer. `half::{f16, bf16}` are
  re-exported from `veks-datatypes` and `vectordata::typed_access`.
- **Float scalar columns** (`.f32`, `.f64`, `.f16`, `.bf16`) are a
  facet format, `FacetFormat::FloatScalar`.
- **Pre-quantized corpora**: three standard facets — `quant_scale`
  (`.f32`), `quant_zero_point` (`.i32`), `quant_sum` (`.i32`), one
  column each, index-aligned with `base_vectors` and windowed with it
  — and a typed `quantization` attribute (scheme, granularity, code
  type, range). `view.quantized_base::<u8>()` opens the codes and the
  three columns as one thing, checked against each other
  (`Error::FacetCardinality`) and against the attribute. `veks check`
  holds the columns, the attribute and the counts.
- **`veks check` holds every ground truth** — G, F and E: one row per
  query, distances' rows equal to the indices' rows, every index
  under the base count with the `-1` sentinel admitted only in a
  filtered truth's short row. The index scan reports progress.
- **The explorer browses `f16`, `bf16` and integer bases.**
- **`veks-io`** readers are generic over `Element`
  (`MmapReader::<f32>::open`, `ScalarMmapReader::<u8>::open`, with
  `as_slice`) and validate what they map: a negative or zero
  dimension, a truncated file and a wrong element type are refused at
  open, and an unaligned `get_slice` is `None` rather than undefined
  behaviour. `VecFormat` carries `bf16` and the float scalars.
- **`veks-pipeline`'s `dispatch_reader!`** opens an `.i8vecs` as `i8`
  and a `.u16vecs` as `u16`; it opened them as the unsigned or signed
  type of the same width.
- **Every KNN engine and verifier computes over a bf16 or 8-/16-bit
  integer base** by widening it to f32 exactly: `compute knn-metal`,
  `knn-stdarch`, `knn-blas`, `knn-faiss`, `compute prefiltered-knn`,
  `verify knn-consolidated`, `verify knn-groundtruth` and `verify
  dataset-knnutils`. The base is scanned at its own width and widened
  one vector or one streamed chunk at a time, never copied as f32;
  queries may be f32 or any type that widens to it. A 32- or 64-bit
  integer base is refused by name.
- **`transform extract` takes a vector file of every element type**,
  gathering whole records at the source's type for the types without a
  specialized extractor.

### Upgrading from 3.0

Each changed call, and what replaces it. The old spelling stays,
deprecated, unless the row says otherwise.

| 3.0 | now |
|---|---|
| `view.base_vectors()`, `query_vectors()`, `neighbor_indices()`, `neighbor_distances()`, the filtered six (f32 / i32) | `view.base_vectors_as::<f32>()` … on `TestDataViewExt` (import it); the type is yours to choose |
| `view.facet(name)` (f32) | `view.facet_as::<T>(name)` |
| `view.metadata_results()` / `metadata_indices()` | `view.metadata_results_as::<i32>()` |
| `open_facet_typed::<T>(&*view, name)`, `TypedReader<T>`, `TypedAccessError`, `dispatch_typed!` | `view.facet_as::<T>(name)`; `get_value(i)` → `get(i)?[0]`, `get_native(i)` → `as_slice()?[i]` or `get_slice(i)?[0]`, `is_native()` → `conversion() == Native`; `dispatch_element!` |
| `T: VvecElement`, `T: TypedElement` | `T: Element` (the shims remain; `TypedElement::width`/`to_i128` are gone) |
| `XvecReader::<f32>::open("x.mvecs")` — reinterpreted the bytes | `open_vec_as::<f32>("x.mvecs")` — widened; the raw open is now `ElementMismatch` |
| `match ElementType { … }` exhaustively | `ElementType` is `#[non_exhaustive]` and has `BF16`: add a `_` arm |
| `match IoError { … }` exhaustively | `#[non_exhaustive]`; new `ElementMismatch`, `ValueOverflow` |
| `match FacetFormat` / `StandardFacet` exhaustively | new `FloatScalar`; new `QuantScale`, `QuantZeroPoint`, `QuantSum` |
| `TestDataGroup::load` on a `knn_entries.yaml` directory with several entries | opens the entry named after the directory, else the only one, else `Error::AmbiguousDataset` |
| `veks_core::formats::Facet` | `vectordata::dataset::StandardFacet` (removed, no shim) |
| `veks_core::formats::preferred_format(facet, width)` | `preferred_vec_format(StandardFacet, ElementType)` |
| `veks_pipeline::pipeline::ElementType` (its own enum), `ToF64` | the same path, now `veks_datatypes::ElementType`; `Element::as_f64` |
| `veks_io::MmapReader::open_fvec(p)`, `ScalarMmapReader::open_u8(p)` | `MmapReader::<f32>::open(p)`, `ScalarMmapReader::<u8>::open(p)` |
| `veks_io::ScalarType` | `ElementType` |

### Fixes

- **A raw reader refuses a file of another type.** `XvecReader::open`
  and `open_path` compared nothing to the extension, so
  `XvecReader::<f32>::open("x.ivecs")` read integers as floats and a
  `.dvecs` opened as `f32` gave a reader of garbage with the wrong
  count. Every raw open now requires the extension's type to be the
  requested one; a conversion goes through `open_vec_as`.
- **`TypedReader` read every float as zero.** A `.f32` column opened as
  `TypedReader<i32>` passed the width check and returned `0` for every
  value. A float native type is now refused at open.
- **A `dataset.yaml` named as a catalog source** listed a dataset per
  top-level key (`profiles`, `attributes`, …). It is now one entry.
- **`transform convert` f32 → bf16 wrote f16 bits under a bf16
  extension**, and a width pair it had no converter for wrote the
  source bytes into the target unchanged (f32 → u8 a corrupt file;
  u8 → f32 broken; same-width pairs reinterpreted). The conversion is
  chosen from the two element types: widening and cross-sign as the
  conversion table allows, or any float to float; anything else is
  refused naming both.
- **`transform extract` refused `.dvec`, `.svec` and `.bvec` sources**
  its documentation listed.
- **`veks prepare import` of a base-only source never wrote the base
  facet.**
- **`XvecReader::get_slice` gave `None` for every 8-byte element type**;
  every other record of such a file is aligned and now borrows.
- **`veks-io` validated nothing it mapped.** A negative dimension was
  cast to a size, the entry size overflowed, a truncated file read
  past its end, and an 8-byte `get_slice` at an unaligned offset was
  undefined behaviour.
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
- **`veks prepare cache-compress` and `cache-uncompress` stream.**
  Each read its files whole and compressed them in memory, on every
  thread at once, so a cache of large segments ran the host out of
  memory (killed at 765 GB on a 1 TB host). Each file is now compressed
  through a fixed buffer to a temporary file renamed into place, with a
  running count of files and bytes.
- **A compressed predicate-key segment cache is reused.** The
  predicate-key step looked for a `.gz` segment only when it set
  `compress-cache`, so every segment `cache-compress` compressed on a
  dataset without that option was recomputed on its next run. It now
  reads either form, decompressing by streaming rather than in memory,
  as the KNN and dedup caches already did.
- **The `veks run` hint to run `cache-compress`** now recognises a
  pipeline that sets `compress-cache`; it looked for `compress_cache`.

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
