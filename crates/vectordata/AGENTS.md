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
  [`TextMeter::stderr`](crate::fetch::TextMeter::stderr). Report it
  as log lines instead (files, CI, the `log` crate):
  [`LogMeter`](crate::fetch::LogMeter), with
  [`step`](crate::fetch::LogMeter::step) and
  [`min_interval`](crate::fetch::LogMeter::min_interval). Hear raw
  progress: pass a closure taking
  [`&FetchEvent`](crate::fetch::FetchEvent). Show nothing:
  [`Silent`](crate::fetch::Silent).
- Read fetched data through vectordata's readers —
  [`TestDataViewExt::base_vectors_as`](crate::TestDataViewExt::base_vectors_as) and
  the other facet accessors, or
  [`TestDataViewExt::facet_as`](crate::TestDataViewExt::facet_as). Once a facet is
  fetched ([`FacetFetch::complete`](crate::fetch::FacetFetch::complete)),
  [`VectorReader::get_slice`](crate::VectorReader::get_slice) is a
  zero-copy borrow from the mapped cache: there is no second copy to
  avoid, and nothing to gain from opening the files yourself.
- Warm a window in the background while reading:
  [`TestDataView::prefetch_in_background`](crate::TestDataView::prefetch_in_background).
- Read vectors:
  [`TestDataViewExt::base_vectors_as`](crate::TestDataViewExt::base_vectors_as) /
  [`query_vectors_as`](crate::TestDataViewExt::query_vectors_as) →
  [`VectorReader::get`](crate::VectorReader::get) or the zero-copy
  [`get_slice`](crate::VectorReader::get_slice) per record, and
  [`VectorReader::read_into`](crate::VectorReader::read_into) to copy a
  run of records into a contiguous buffer you own — a whole base set,
  or one window at a time — split across shards and windows for you.
- Read ground truth:
  [`TestDataViewExt::neighbor_indices_as`](crate::TestDataViewExt::neighbor_indices_as),
  [`neighbor_distances_as`](crate::TestDataViewExt::neighbor_distances_as).
- Read any element facet at the type it holds or converted (u8 codes
  in place, bf16 or f16 widened to f32, a metadata column as i32):
  [`TestDataViewExt::facet_as`](crate::TestDataViewExt::facet_as); the
  rules are [`ElementType::conversion_to`](crate::ElementType::conversion_to).
  A whole column borrows with [`VectorReader::as_slice`](crate::VectorReader::as_slice).
  Record (slab) facets:
  [`TestDataView::open_facet_records`](crate::TestDataView::open_facet_records).
- List profiles and facets:
  [`TestDataGroup::profile_names`](crate::TestDataGroup::profile_names),
  [`TestDataView::facet_manifest`](crate::TestDataView::facet_manifest).
- Find the cache directory:
  [`settings::cache_dir`](crate::settings::cache_dir). Keep a cache of
  your own, leaving the user's settings alone: call
  [`settings::set_cache_dir`](crate::settings::set_cache_dir) before
  the first open or fetch. Inspect what is
  cached for a facet:
  [`TestDataView::open_facet_storage`](crate::TestDataView::open_facet_storage)
  → [`FacetStorage::cache_stats`](crate::FacetStorage::cache_stats).
- Publish a dataset: [`push::execute`](crate::push::execute), reporting
  through [`push::ProgressSink`](crate::push::ProgressSink).

- Build a dataset description in code, or write a `dataset.yaml`:
  [`DatasetConfig::builder`](crate::dataset::DatasetConfig::builder), then
  [`build`](crate::dataset::DatasetBuilder::build) and
  [`save`](crate::dataset::DatasetConfig::save). The result passes the
  checks a parsed file passes and equals the parse of what it saves.
  Parse text you hold with
  [`DatasetConfig::parse`](crate::dataset::DatasetConfig::parse).
- Open a dataset you described in code, with no catalog:
  [`TestDataGroup::from_config`](crate::TestDataGroup::from_config)`(name, home, description)`.
  Relative facets resolve against `home`, and a remote home caches under
  `<cache_root>/<name>/`.
- Build a catalog entry in code:
  [`CatalogEntry::builder`](crate::dataset::CatalogEntry::builder), with
  `home` and `dataset` for a whole description, or `descriptor` and a
  summary for a `dataset.yaml` read when opened. Open it with
  [`Catalog::open_entry`](crate::catalog::Catalog::open_entry). Read an
  entry through its accessors (`profiles()`, `attributes()`,
  `dataset_home_url()`); what it holds is
  [`EntryDescription`](crate::dataset::EntryDescription).
- Register a catalog at runtime:
  [`CatalogBuilder::named`](crate::catalog::CatalogBuilder::named) with
  `entry`/`entries` you built, or `content` (catalog JSON/YAML or a
  `knn_entries.yaml` document) and a `base` for its relative paths; add
  it with [`CatalogSources::with`](crate::catalog::CatalogSources::with)
  (or `with_optional`), before or after `configure_default()` (order is
  listing order). To serve entries from elsewhere, implement
  [`CatalogSource`](crate::catalog::CatalogSource). Do not merge entries
  into a catalog by hand: a name in two catalogs must stay ambiguous.
  `examples/register_catalog.rs` runs end to end, offline.
- Open the explorer over a program's catalogs:
  [`explore::run_with_sources`](crate::explore::run_with_sources) with the
  `CatalogSources` you assembled. It lists and opens exactly those, and
  an edit on its catalogs screen re-reads only the configured part
  ([`reload_configured`](crate::catalog::CatalogSources::reload_configured)).
- Embed the `vectordata` command line with a program's catalogs:
  [`shell::bin_main_with_catalogs`](crate::shell::bin_main_with_catalogs)
  — they follow the configured catalogs in every command, and `--at`
  replaces them for a run.
- Run the `vectordata` command line inside a program that carries on
  after it: [`shell::run`](crate::shell::run) with a
  [`shell::Host`](crate::shell::Host) (or
  [`shell::run_with_catalogs`](crate::shell::run_with_catalogs)) returns
  the exit status (0, 1 failed, 2 usage) instead of ending the process.
  The host's own output goes in
  [`Host::after_describe`](crate::shell::Host::after_describe) and
  [`Host::after_list`](crate::shell::Host::after_list), its own precache
  specs (`group:<name>`) in
  [`Host::before_precache`](crate::shell::Host::before_precache), and
  [`Host::owns_completion`](crate::shell::Host::owns_completion) binds
  completion to the host's executable name. Do not parse the argv and
  call the command modules yourself.
- Precache several specs as one run (one plan, one progress display, one
  report): `vectordata datasets precache a:default b:10m`, or
  [`datasets::precache::run_specs`](crate::datasets::precache::run_specs).
- Build the catalogs a `datasets` command runs against:
  [`datasets::build_sources`](crate::datasets::build_sources) (the
  program's catalogs, then the `--configdir` / `--catalog` / `--at`
  trio), then
  [`RunSources::open`](crate::datasets::RunSources::open), which reports
  every problem and fails only for a catalog named for the run. Do not
  assemble `CatalogSources` per command.
- Write the catalog files for a dataset tree:
  [`CatalogTree::scan`](crate::catalog::generate::CatalogTree::scan) then
  [`write`](crate::catalog::generate::CatalogTree::write) over
  [`catalog_dirs`](crate::catalog::generate::CatalogTree::catalog_dirs). It
  is the generator both `veks` commands run; do not walk a tree for
  `dataset.yaml` files and serialize entries yourself.

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
- **`dataset.yaml` text.** Do not format one with `format!`: the
  builder and `save` write the version a dataset needs, quote what YAML
  would misread, and leave out what inheritance restores.
- **Joining a facet or entry path onto a dataset or catalog location.**
  [`Location::resolve`](crate::Location::resolve) is the rule the open
  reads by: a URL or leading `/` kept as written, `file://` read as its
  path, anything else joined by URL rules (so `..` resolves). Formatting
  `"{home}{path}"` yourself names a different file than the open
  reads.
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
| `vectordata datasets ping` | [`Catalog::open_spec`](crate::catalog::Catalog::open_spec), then a reader per facet ([`TestDataViewExt::facet_as`](crate::TestDataViewExt::facet_as)) |
| `vectordata datasets push` | [`push::execute`](crate::push::execute) |
| `vectordata datasets derive` | none — a command that writes a new dataset directory |
| `vectordata datasets init` | [`dataset::infer::infer_dir`](crate::dataset::infer::infer_dir) — writes its `yaml`; `TestDataGroup::load` on the directory reads the same descriptor |
| `vectordata datasets curlify` | none — a command that writes a download script |
| `vectordata cache list` | [`cache_admin::list_entries`](crate::cache_admin::list_entries) |
| `vectordata cache prune` | [`cache_admin::prune_by_filter`](crate::cache_admin::prune_by_filter) |
| `vectordata cache prune-legacy` | [`cache_admin::prune_legacy_layout`](crate::cache_admin::prune_legacy_layout) |
| `vectordata config get` | [`settings::setting_value`](crate::settings::setting_value), [`settings::cache_dir`](crate::settings::cache_dir) |
| `vectordata config set` | [`settings::write_setting`](crate::settings::write_setting) |
| `vectordata config catalog` | [`catalog::sources::named_catalog_entries`](crate::catalog::sources::named_catalog_entries) |
| `vectordata config overlay` | [`overlay::Overlay`](crate::overlay::Overlay), [`CatalogSources::with_overlay`](crate::catalog::CatalogSources::with_overlay), [`CatalogEntry::attribute_source`](crate::dataset::CatalogEntry::attribute_source) |
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
  URL.** A file URL opened with the path-or-URL constructors
  (`XvecReader::open`, `io::open_vec_as`) lands in its dataset's cache
  directory when one covers it, but has no fetch, profiles or windows
  around it.
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
