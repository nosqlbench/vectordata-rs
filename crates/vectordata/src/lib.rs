//! # vectordata — typed access to vector search datasets
//!
//! Find a dataset by name, fetch the parts you need into a local cache
//! with progress, and read vectors, ground truth and metadata from it —
//! locally, over HTTP, or from S3, through one API. For programs that
//! consume benchmark datasets (an ANN harness, a loader, a test suite)
//! and for the `vectordata` command-line tool, which is a thin layer
//! over this library.
//!
//! **Agents and newcomers:** the table below is the index. Each row is
//! the one call to use for that task; other public items that look
//! similar say in their first line what they are and point back here.
//! `AGENTS.md` in this crate's package ([`_agents`]) carries the same
//! index plus a map from every CLI command to the library call behind
//! it.
//!
//! ## Common tasks
//!
//! | Task | Call | Example |
//! |---|---|---|
//! | Load the configured catalogs | [`Catalog::of`](catalog::Catalog::of) with [`CatalogSources::configure_default`](catalog::CatalogSources::configure_default) | below |
//! | Open a dataset from a `name:selector` spec | [`Catalog::open_spec`](catalog::Catalog::open_spec) → [`DatasetSelection::view`](catalog::DatasetSelection::view) | `examples/open_by_spec.rs` |
//! | Open one profile by name | [`Catalog::open_profile`](catalog::Catalog::open_profile) | below |
//! | Parse a spec without opening it | [`DatasetSpec::parse`](dataset::selector::DatasetSpec::parse) | — |
//! | Fetch chosen facets into the cache, with progress | [`TestDataView::fetch`] with [`FetchRequest::facets`](fetch::FetchRequest::facets) | `examples/fetch_with_progress.rs` |
//! | Fetch a whole profile, or several | [`TestDataView::fetch`] with [`FetchRequest::all`](fetch::FetchRequest::all); [`TestDataGroup::fetch`]; [`Catalog::fetch`](catalog::Catalog::fetch) | `examples/fetch_with_progress.rs` |
//! | Fetch only a record window | [`FetchRequest::window`](fetch::FetchRequest::window) | `examples/stream_base_vectors.rs` |
//! | Know what a fetch will cost first | [`TestDataView::plan_fetch`] → [`FetchPlan`](fetch::FetchPlan) | — |
//! | Draw the CLI's progress meter | [`TextMeter`](fetch::TextMeter) | `examples/fetch_with_progress.rs` |
//! | Report fetch progress as log lines | [`LogMeter`](fetch::LogMeter), throttled by [`ProgressStep`](fetch::ProgressStep) | — |
//! | Warm a window in the background while reading | [`TestDataView::prefetch_in_background`] | `examples/stream_base_vectors.rs` |
//! | Read vectors by ordinal | [`TestDataViewExt::base_vectors_as`] → [`VectorReader::get`] | below |
//! | Read a facet at its own element type, or widened (u8 codes, bf16, f16 → f32) | [`TestDataViewExt::facet_as`]; the rules: [`ElementType::conversion_to`] | [`TestDataViewExt`] |
//! | Load or stream records in bulk, into your own buffer | [`VectorReader::read_into`] | `examples/stream_base_vectors.rs` |
//! | Borrow a whole column (quantization parameters, labels) with no copy | [`VectorReader::as_slice`] on a scalar facet | [`TestDataViewExt::facet_as`] |
//! | Open a facet without knowing its type | [`TestDataView::open_facet_native`] → [`AnyVectorReader`] | — |
//! | Read ground truth | [`TestDataViewExt::neighbor_indices_as`], [`TestDataViewExt::neighbor_distances_as`] | [`TestDataViewExt::neighbor_indices_as`] |
//! | Read a metadata or scalar facet | [`TestDataViewExt::facet_as`] (`open_facet_typed` is deprecated) | below |
//! | Read a record (slab) facet | [`TestDataView::open_facet_records`] | — |
//! | List profiles and facets | [`TestDataGroup::profile_names`], [`TestDataView::facet_manifest`] | below |
//! | Find datasets by pattern | [`Catalog::match_glob`](catalog::Catalog::match_glob), [`Catalog::datasets`](catalog::Catalog::datasets) | — |
//! | Build a dataset description in code | [`DatasetConfig::builder`](dataset::DatasetConfig::builder) → [`save`](dataset::DatasetConfig::save) | `examples/register_catalog.rs` |
//! | Open a dataset described in code | [`TestDataGroup::from_config`] | — |
//! | Build a catalog entry in code | [`CatalogEntry::builder`](dataset::CatalogEntry::builder) → [`Catalog::open_entry`](catalog::Catalog::open_entry) | `examples/register_catalog.rs` |
//! | Register a catalog at runtime | [`CatalogBuilder`](catalog::CatalogBuilder) → [`CatalogSources::with`](catalog::CatalogSources::with) | below, `examples/register_catalog.rs` |
//! | Serve catalog entries from elsewhere | implement [`CatalogSource`](catalog::CatalogSource) | — |
//! | Explore a program's catalogs (feature `explore`) | [`explore::run_with_sources`] | — |
//! | Write a tree's catalog files | [`CatalogTree`](catalog::generate::CatalogTree) | — |
//! | Resolve a path against a dataset or catalog location | [`Location::resolve`] | — |
//! | Locate the cache | [`settings::cache_dir`] | — |
//! | Inspect what is cached | [`FacetStorage::cache_stats`] via [`TestDataView::open_facet_storage`] | — |
//!
//! ### Open and read
//!
//! ```no_run
//! use vectordata::catalog::{Catalog, CatalogSources};
//! use vectordata::{TestDataViewExt, VectorReader};
//!
//! # fn main() -> vectordata::Result<()> {
//! // Catalogs come from ~/.config/vectordata/catalogs.yaml.
//! let catalog = Catalog::of(&CatalogSources::new().configure_default());
//! let view = catalog.open_profile("my-dataset", "default")?;
//! let base = view.base_vectors_as::<f32>()?;
//! println!("{} vectors, dim={}", base.count(), base.dim());
//! let v: Vec<f32> = base.get(42)?;
//! # let _ = v; Ok(()) }
//! ```
//!
//! ### Fetch, then read
//!
//! ```no_run
//! use vectordata::catalog::{Catalog, CatalogSources};
//! use vectordata::fetch::{FetchRequest, TextMeter};
//! use vectordata::TestDataViewExt;
//!
//! # fn main() -> vectordata::Result<()> {
//! let catalog = Catalog::of(&CatalogSources::new().configure_default());
//! let view = catalog.open_spec("my-dataset:default")?.view()?;
//! view.fetch(
//!     &FetchRequest::facets(["base_vectors", "query_vectors"]),
//!     &mut TextMeter::stderr("Fetch"),
//! )?;
//! // Every read below is local and zero-copy.
//! let queries = view.query_vectors_as::<f32>()?;
//! # let _ = queries; Ok(()) }
//! ```
//!
//! ### Profiles, facets and typed metadata
//!
//! ```no_run
//! # use vectordata::catalog::{Catalog, CatalogSources};
//! use vectordata::{TestDataViewExt, VectorReader};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let catalog = Catalog::of(&CatalogSources::new().configure_default());
//! let group = catalog.open("my-dataset")?;
//! for name in group.profile_names() {
//!     let view = group.profile(&name).unwrap();
//!     for (facet, desc) in view.facet_manifest() {
//!         println!("{name}:{facet} → {}", desc.source_type.as_deref().unwrap_or("?"));
//!     }
//! }
//! let view = group.profile("default").unwrap();
//! // A u8 metadata facet read as i32: widening, by the conversion table.
//! let labels = view.facet_as::<i32>("metadata_content")?;
//! let label = labels.get(42)?[0];
//! // Or at its own type, borrowed whole from the mapping.
//! let native = view.facet_as::<u8>("metadata_content")?;
//! let all: Option<&[u8]> = native.as_slice();
//! # let _ = (label, all); Ok(()) }
//! ```
//!
//! ### Catalogs a program supplies
//!
//! A program can register catalogs of its own, built from entries it
//! made or content it holds, beside or instead of the configured ones.
//! They list, open and collide exactly as configured catalogs do.
//!
//! ```
//! use vectordata::catalog::{Catalog, CatalogBuilder, CatalogSources};
//! use vectordata::dataset::{CatalogEntry, DatasetConfig, StandardFacet};
//! use vectordata::Location;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let description = DatasetConfig::builder("demo")
//!     .profile("default", |p| p.facet(StandardFacet::BaseVectors, "base.fvec"))
//!     .build()?;
//! let entry = CatalogEntry::builder("demo")
//!     .home(&Location::parse("https://example.com/datasets/demo/")?)
//!     .dataset(description)
//!     .build()?;
//! let shipped = CatalogBuilder::named("shipped")
//!     .base(Location::parse("https://example.com/datasets/")?)
//!     .content("listed:default:\n  base: listed/base.fvec\n  query: listed/query.fvec\n  gt: listed/gt.ivec\n")
//!     .build();
//! let sources = CatalogSources::new()
//!     .with(CatalogBuilder::named("in-code").entry(entry).build())
//!     .with(shipped); // add .configure_default() for the user's own
//! let catalog = Catalog::of(&sources);
//! assert_eq!(catalog.lookup("demo")?.catalog_name(), Some("in-code"));
//! assert_eq!(catalog.lookup("listed")?.catalog_name(), Some("shipped"));
//! # Ok(()) }
//! ```
//!
//! ## Layers
//!
//! From the outside in. Applications use the first four; the rest
//! exist for tools and for inspection.
//!
//! 1. **Catalog** ([`catalog`]) — names to locations. [`Catalog`](catalog::Catalog)
//!    resolves a name or spec; nothing below needs a URL.
//! 2. **Group** ([`TestDataGroup`]) — one dataset: its `dataset.yaml`,
//!    its profiles, and profile selection.
//! 3. **View** ([`TestDataView`]) — one profile: its facets, and the
//!    fetch call.
//! 4. **Reader** ([`VectorReader`], [`VvecReader`], [`AnyVectorReader`],
//!    [`records`]) — ordinal access to one facet.
//! 5. **Storage** ([`FacetStorage`], low-level) — a facet's bytes:
//!    cache state, range fetches. Transport and cache internals beneath
//!    it are private.
//!
//! The `vectordata` binary sits on top of all of this: its commands
//! live in [`datasets`], [`config`] and [`shell`] and call the layers
//! above. Nothing in the library layers prints, prompts or exits;
//! progress reaches the terminal only through a sink the caller passes.
//!
//! ## Traps
//!
//! - **A fetched facet needs no network.** A complete cached copy opens
//!   from disk — no `.mref` fetch, no HEAD — and a dataset's definition
//!   and its catalog fall back to the copies kept when they were last
//!   fetched, so a warmed cache works offline. Whether the upstream has
//!   since changed is asked by [`TestDataView::fetch`] (and `datasets
//!   ping`), not by opening;
//!   [`FacetFetch::upstream_checked`](fetch::FacetFetch::upstream_checked)
//!   says whether it could be. [`settings::offline`] (`VECTORDATA_OFFLINE=1`
//!   or `offline: on`) goes further and makes no request at all.
//! - **Opening a reader does not fetch the facet.** Reads fetch the
//!   chunks they touch, one request at a time. For a scan, fetch first
//!   ([`TestDataView::fetch`]) — in parallel, with progress — and read
//!   locally after.
//! - **A remote variable-length facet with no published offset index
//!   cannot be opened until fetched.** Its record boundaries are only
//!   knowable from every byte, so the open is refused with
//!   [`IoError::OffsetIndexUnavailable`] rather than downloading the
//!   file behind your back. Fetch it, or publish its `IDXFOR__` sidecar.
//! - **A server without HTTP range support serves only whole files.**
//!   The first read of such a file downloads all of it; fetch first to
//!   see that happen with progress.
//! - **A window is fetched by the chunk.** A ten-record window costs at
//!   least one chunk; [`TestDataView::plan_fetch`] reports the real
//!   cost, including overfetch, before anything moves.
//! - **Some windows cannot be honoured.** Parquet, and a variable-length
//!   file without its index, have no record-to-byte mapping, so a
//!   window means the whole facet. That is refused
//!   ([`Error::WindowUnresolvable`]) unless the request says
//!   [`allow_whole_facet`](fetch::FetchRequest::allow_whole_facet).
//! - **Do not split a spec on `:`.** URLs have ports and Windows paths
//!   have drive letters; [`DatasetSpec::parse`](dataset::selector::DatasetSpec::parse)
//!   and [`Catalog::open_spec`](catalog::Catalog::open_spec) know the
//!   grammar.
//! - **`datasets` is the command-line layer.** Its functions take CLI
//!   strings and return exit codes; call the library function each one
//!   names instead.
//!
//! ## Feature flags
//!
//! - **`cli`** (default) — the `vectordata` binary and its shell
//!   ([`shell`]), on the in-tree `veks-completion` parser. The library,
//!   including [`fetch`], [`catalog`] and the [`datasets`] command
//!   implementations, does not need it.
//! - **`explore`** (default) — the `vectordata explore` TUI, adding the
//!   pure-Rust `veks-simd` kernels and `rand`/`rayon`. No C toolchain.
//!
//! `default-features = false` gives the library alone: catalogs, fetch,
//! readers and the cache, with no terminal UI dependencies beyond what
//! the progress meter needs.
//!
//! ## Caching
//!
//! Remote data is cached under the directory [`settings::cache_dir`]
//! names (configured in `~/.config/vectordata/settings.yaml`, or
//! isolated under `$VECTORDATA_HOME`). Chunks published with a `.mref`
//! are merkle-verified before use. Once a file is complete, readers
//! switch to mmap for zero-copy access.

#![warn(missing_docs)]
// An example that calls a deprecated item teaches the old API; only a
// deprecated item's own example may, and it says so with an allow.
#![doc(test(attr(deny(deprecated))))]

// Module docs, each opening with whether the module is library API,
// CLI support or internal, live in the module files themselves.

/// Internal. Merkle-verified download cache for remote datasets;
/// reached through readers and [`fetch`], never directly.
pub(crate) mod cache;
pub(crate) mod chunked_http;
/// Internal. Byte-level storage shared by every reader. Users never see
/// `storage::Storage` — they get reader handles whose transport is
/// chosen for them.
pub(crate) mod storage;
/// Internal. Replace a file so a reader sees the old file or the new
/// one, never a truncated one.
pub(crate) mod atomic_file;
pub mod settings;
pub mod filters;
pub mod mounts;
pub mod config;
pub mod update_check;

#[cfg(feature = "cli")]
pub mod shell;
pub mod datasets;
pub mod push;
pub mod credentials;
pub mod endpoint;
pub mod backup;
pub mod client_cli;
#[cfg(feature = "explore")]
pub mod explore;
pub mod catalog;
pub mod dataset;
pub mod formats;
pub mod records;
pub mod binding;
pub mod merkle;
/// Internal. HTTP transport layer, reached only through readers and the
/// cache.
pub(crate) mod transport;
pub use veks_apicore::model;
pub mod io;
pub mod access;
pub use veks_apicore::metadata_schema;
pub mod typed_access;
pub mod view;
pub mod verify;
pub mod overlay;
pub mod fetch;
pub mod group;
pub use veks_apicore::location;

/// Library API. The guide for coding agents shipped in this package
/// (`AGENTS.md`), rendered here so its links are checked with the rest
/// of the docs.
///
/// Its links spell out full `crate::` paths even where a shorter label
/// would resolve, because the file is also read raw, where the path is
/// the information.
#[doc = include_str!("../AGENTS.md")]
#[allow(rustdoc::redundant_explicit_links)]
pub mod _agents {}

/// Library API. Cache administration: list and prune what is cached
/// under the cache root, for the `vectordata cache` and `veks` tooling.
///
/// Exposes a *minimal* surface for tasks that inspect or scrub the
/// on-disk cache root — the live cache state itself is owned by
/// internal types (`storage::Storage`, `cache::CachedChannel`)
/// that are deliberately not reachable from outside this crate. If you
/// find yourself reaching for more than this module exposes, the right
/// move is to add another targeted re-export, not to widen visibility
/// on the core types.
pub mod cache_admin {
    pub use crate::cache::reader::{
        LEGACY_BLOBS_DIR, LEGACY_HTTP_DIR,
        CacheEntry, CacheListing, PruneFilter, PruneReport,
        is_legacy_layout_dir,
        list_entries, prune_by_filter, prune_legacy_layout,
    };
}

pub use group::{TestDataGroup, PREBUFFER_LARGE_WARNING_BYTES};
pub use location::Location;
pub use model::FacetConfig;
pub use view::{FacetStorage, PrebufferProgress};
pub use veks_apicore::{
    CacheStats, ContentId, ContentIdKind, FacetBytes, FacetDescriptor, NoProgress, PrefetchHandle, PrefetchPlan,
    PrefetchReport, ProgressSink, QuantizedBase, RangeFill, ShardRange, TestDataView, TestDataViewExt,
    WholeFacetFallback,
};
#[allow(deprecated)]
pub use view::open_facet_typed;
pub use io::{
    AnyVectorReader, AnyVvecReader, ConvertingReader, IndexedVvecReader, IoError, ScalarReader,
    VectorReader, VvecReader, XvecReader, open_vec_as, open_vec_native,
};
pub use veks_apicore::InsufficientCacheSpace;
pub use access::AccessMode;
pub use typed_access::{Container, Element, ElementConversion, ElementType, ElementTypeExt};
pub use veks_datatypes::{DistanceFunction, UnknownDistanceFunction};
#[allow(deprecated)]
pub use typed_access::{TypedAccessError, TypedReader};
/// Re-exported so external callers of
/// [`FacetStorage::prebuffer_with_progress`] can name the
/// progress type their callback receives.
pub use veks_apicore::DownloadProgress;

pub use veks_apicore::{Error, Result, TransportError};

