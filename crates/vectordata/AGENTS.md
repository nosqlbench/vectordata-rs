# vectordata for agents

This crate reads vector-search benchmark datasets: it resolves a dataset
by name, fetches the parts a program needs into a local cache with
progress, and serves vectors, ground truth and metadata from there. The
`vectordata` binary is a thin layer over the library — every command
below names the library call it runs, and a program should make that
call rather than reimplement the command.

Paths below are relative to the crate root (`vectordata::…`). The crate
documentation's **Common tasks** table is the same index with examples.

## Do this

- Load the configured catalogs:
  [`Catalog::of`](crate::catalog::Catalog::of)`(&`[`CatalogSources::new`](crate::catalog::CatalogSources::new)`().configure_default())`.
  Loading problems are values on
  [`Catalog::diagnostics`](crate::catalog::Catalog::diagnostics), not
  stderr output.
- Open a dataset from a string a user typed (`name:selector`, a path, or
  a URL): [`Catalog::open_spec`](crate::catalog::Catalog::open_spec), then
  [`DatasetSelection::view`](crate::catalog::DatasetSelection::view) for
  one profile or [`DatasetSelection::views`](crate::catalog::DatasetSelection::views)
  for several.
- Open one profile by name:
  [`Catalog::open_profile`](crate::catalog::Catalog::open_profile).
- Fetch facets into the cache with progress — the call for "download",
  "precache", "prebuffer" or "prefetch":
  [`TestDataView::fetch`](crate::TestDataView::fetch) with
  [`FetchRequest::facets`](crate::fetch::FetchRequest::facets) or
  [`FetchRequest::all`](crate::fetch::FetchRequest::all). Across
  profiles: [`TestDataGroup::fetch`](crate::TestDataGroup::fetch). From a
  spec in one call: [`Catalog::fetch`](crate::catalog::Catalog::fetch).
- Fetch only some records:
  [`FetchRequest::window`](crate::fetch::FetchRequest::window), with a
  window from [`parse_window`](crate::dataset::source::parse_window).
- Learn what a fetch costs before running it:
  [`TestDataView::plan_fetch`](crate::TestDataView::plan_fetch), then
  [`FetchPlan::bytes_to_fetch`](crate::fetch::FetchPlan::bytes_to_fetch)
  and [`FetchPlan::execute`](crate::fetch::FetchPlan::execute).
- Show the CLI's progress meter:
  [`TextMeter::stderr`](crate::fetch::TextMeter::stderr). Hear raw
  progress: pass a closure taking
  [`&FetchEvent`](crate::fetch::FetchEvent). Show nothing:
  [`Silent`](crate::fetch::Silent).
- Read fetched data through vectordata's readers —
  [`TestDataView::base_vectors`](crate::TestDataView::base_vectors) and
  the other facet accessors, or
  [`TestDataView::facet`](crate::TestDataView::facet). Once a facet is
  fetched ([`FacetFetch::complete`](crate::fetch::FacetFetch::complete)),
  [`VectorReader::get_slice`](crate::VectorReader::get_slice) is a
  zero-copy borrow from the mapped cache: there is no second copy to
  avoid, and nothing to gain from opening the files yourself.
- Warm a window in the background while reading:
  [`TestDataView::prefetch_in_background`](crate::TestDataView::prefetch_in_background).
- Read vectors:
  [`TestDataView::base_vectors`](crate::TestDataView::base_vectors) /
  [`query_vectors`](crate::TestDataView::query_vectors) →
  [`VectorReader::get`](crate::VectorReader::get) or the zero-copy
  [`get_slice`](crate::VectorReader::get_slice) per record, and
  [`VectorReader::read_into`](crate::VectorReader::read_into) to copy a
  run of records into a contiguous buffer you own — a whole base set,
  or one window at a time — split across shards and windows for you.
- Read ground truth:
  [`TestDataView::neighbor_indices`](crate::TestDataView::neighbor_indices),
  [`neighbor_distances`](crate::TestDataView::neighbor_distances).
- Read metadata or scalar facets with type widening:
  [`open_facet_typed`](crate::open_facet_typed) →
  [`TypedReader`](crate::TypedReader). Record (slab) facets:
  [`TestDataView::open_facet_records`](crate::TestDataView::open_facet_records).
- List profiles and facets:
  [`TestDataGroup::profile_names`](crate::TestDataGroup::profile_names),
  [`TestDataView::facet_manifest`](crate::TestDataView::facet_manifest).
- Find the cache directory:
  [`settings::cache_dir`](crate::settings::cache_dir). Inspect what is
  cached for a facet:
  [`TestDataView::open_facet_storage`](crate::TestDataView::open_facet_storage)
  → [`FacetStorage::cache_stats`](crate::FacetStorage::cache_stats).
- Publish a dataset: [`push::execute`](crate::push::execute), reporting
  through [`push::ProgressSink`](crate::push::ProgressSink).

## Do not hand-roll

Each of these exists, is tested, and handles cases that are easy to miss.

- **Download planning and progress meters.** `fetch` plans every facet,
  merges ranges, skips resident chunks, checks cache space, sums
  progress across ranges and shards, and clamps it to the plan. Do not
  build a loop over `prefetch_in_background` handles to get a meter.
- **Spec parsing.** Do not split a spec on `:` — URLs have ports and
  Windows paths have drive letters. Use
  [`DatasetSpec::parse`](crate::dataset::selector::DatasetSpec::parse) or
  `open_spec`.
- **Cache-capacity checks.** `fetch` refuses a run the cache directory
  cannot hold, before anything moves
  ([`Error::InsufficientCacheSpace`](crate::Error::InsufficientCacheSpace)).
- **Record-to-byte mapping for windows.** `plan_fetch` does it per
  format, including sharded facets and variable-length files.
- **Facet name aliases.** `FetchRequest::facets` accepts the standard
  aliases (`base`, `gt`, `metadata_indices`, …).
- **Vector file readers.** Do not open the cache's or the dataset's
  files with your own `.fvec`/`.ivec` reader. The facet readers already
  mmap them, and they also know the shard boundaries, record windows,
  element widening and verification state that a raw file does not. A
  cache file is pre-sized and sparse until complete, with its valid
  chunks recorded separately, so its bytes are only meaningful through
  vectordata. `FacetStorage::cache_path` and `local_files` are for
  diagnostics and cache administration, never for reading data.
- **"Did you mean" for unknown datasets.**
  [`Catalog::lookup`](crate::catalog::Catalog::lookup) returns
  [`Error::UnknownDataset`](crate::Error::UnknownDataset) with
  suggestions.

## CLI ↔ library

| Command | Library call |
|---|---|
| `vectordata datasets precache` | [`TestDataView::fetch`](crate::TestDataView::fetch), [`TestDataGroup::fetch`](crate::TestDataGroup::fetch), [`Catalog::fetch`](crate::catalog::Catalog::fetch) |
| `vectordata datasets list` | [`Catalog::datasets`](crate::catalog::Catalog::datasets), [`Catalog::match_glob`](crate::catalog::Catalog::match_glob) |
| `vectordata datasets describe` | [`Catalog::open_spec`](crate::catalog::Catalog::open_spec), [`TestDataView::facet_manifest`](crate::TestDataView::facet_manifest) |
| `vectordata datasets ping` | [`Catalog::open_spec`](crate::catalog::Catalog::open_spec), then a reader per facet ([`TestDataView::facet`](crate::TestDataView::facet)) |
| `vectordata datasets push` | [`push::execute`](crate::push::execute) |
| `vectordata datasets derive` | none — a command that writes a new dataset directory |
| `vectordata datasets curlify` | none — a command that writes a download script |
| `vectordata cache list` | [`cache_admin::list_entries`](crate::cache_admin::list_entries) |
| `vectordata cache prune` | [`cache_admin::prune_by_filter`](crate::cache_admin::prune_by_filter) |
| `vectordata cache prune-legacy` | [`cache_admin::prune_legacy_layout`](crate::cache_admin::prune_legacy_layout) |
| `vectordata config get` | [`settings::setting_value`](crate::settings::setting_value), [`settings::cache_dir`](crate::settings::cache_dir) |
| `vectordata config set` | [`settings::write_setting`](crate::settings::write_setting) |
| `vectordata config catalog` | [`catalog::sources::named_catalog_entries`](crate::catalog::sources::named_catalog_entries) |
| `vectordata config mounts` | [`mounts`](crate::mounts) |
| `vectordata login` | [`endpoint::login_password`](crate::endpoint::login_password), [`credentials`](crate::credentials) |
| `vectordata logout` | [`credentials`](crate::credentials) |
| `vectordata whoami` | [`endpoint::whoami`](crate::endpoint::whoami) |
| `vectordata ping` | [`endpoint::whoami`](crate::endpoint::whoami) |
| `vectordata token issue` | [`endpoint::issue_token`](crate::endpoint::issue_token) |
| `vectordata token revoke` | [`endpoint::revoke_token`](crate::endpoint::revoke_token) |
| `vectordata backup` | [`backup::run_backup`](crate::backup::run_backup) |
| `vectordata restore` | [`backup::run_restore`](crate::backup::run_restore) |
| `vectordata explore` | none — the interactive TUI |
| `vectordata completions` | none — shell integration |

`veks datasets …` dispatches to the same implementations.

## Traps

- **A fetched facet needs no network.** Complete copies open from disk,
  and dataset definitions fall back to a kept copy, so a warmed cache
  works offline. Upstream changes are detected by `fetch` (and
  `datasets ping`), not by opening; a fetch that could not reach the
  server says so in
  [`FacetFetch::upstream_checked`](crate::fetch::FacetFetch::upstream_checked).
  Remote catalogs keep a copy too, under `<cache>/.catalogs/`.
  `VECTORDATA_OFFLINE=1` (or `vectordata config set offline on`) makes
  no request at all; what was never fetched is then an error saying so.
- **Open remote datasets by catalog or `TestDataGroup`, not by file
  URL.** `XvecReader::open_url` and `TypedReader::open_url` are
  deprecated; a file URL opened with the path-or-URL constructors lands
  in its dataset's cache directory when one covers it, but has no
  fetch, profiles or windows around it.
- **Opening a reader does not fetch the facet.** Reads fetch the chunks
  they touch, one at a time. For a scan, `fetch` first and read locally.
- **A remote variable-length facet with no published offset index
  cannot be opened until fetched.** The open is refused with
  [`IoError::OffsetIndexUnavailable`](crate::IoError::OffsetIndexUnavailable)
  instead of downloading the whole file. `fetch` it (no window), or
  publish its `IDXFOR__` sidecar.
- **A server without HTTP range support serves only whole files.** The
  first read downloads everything; `fetch` first to see it with
  progress.
- **A window is fetched by the chunk.** Small windows can be large
  downloads; `plan_fetch` reports the real cost and the overfetch.
- **Some windows cannot be honoured** (parquet; a variable-length file
  without its index). They are refused
  ([`Error::WindowUnresolvable`](crate::Error::WindowUnresolvable))
  unless the request says
  [`allow_whole_facet`](crate::fetch::FetchRequest::allow_whole_facet).
- **The `datasets` module is the command-line layer.** Its functions take
  CLI-shaped strings, print, and return exit codes. Call the library
  function the table above names instead.
- **The `prebuffer_*`, `prefetch` and `prefetch_with_progress` methods
  are deprecated** wrappers over `fetch`; new code should not use them.

## For your own project's agent instructions

Paste this into the `AGENTS.md` or `CLAUDE.md` of a project that depends
on vectordata:

> Before writing download, caching, progress-reporting, dataset-spec
> parsing or vector-streaming code against `vectordata`, read its
> `AGENTS.md` (in the crate's registry source, next to `Cargo.toml`) and
> the Common tasks table in its crate documentation. Use the call they
> name; if a needed capability seems to be missing, check the CLI ↔
> library table first — every `vectordata` command is a thin adapter
> over a library call.
