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
//! | Warm a window in the background while reading | [`TestDataView::prefetch_in_background`] | `examples/stream_base_vectors.rs` |
//! | Read vectors by ordinal | [`TestDataView::base_vectors`] → [`VectorReader::get`] | below |
//! | Stream base vectors in order | [`VectorReader::get`] over a fetched window | `examples/stream_base_vectors.rs` |
//! | Read ground truth | [`TestDataView::neighbor_indices`], [`TestDataView::neighbor_distances`] | — |
//! | Read a metadata or scalar facet | [`open_facet_typed`] → [`TypedReader`] | below |
//! | Read a record (slab) facet | [`TestDataView::open_facet_records`] | — |
//! | List profiles and facets | [`TestDataGroup::profile_names`], [`TestDataView::facet_manifest`] | below |
//! | Find datasets by pattern | [`Catalog::match_glob`](catalog::Catalog::match_glob), [`Catalog::datasets`](catalog::Catalog::datasets) | — |
//! | Locate the cache | [`settings::cache_dir`] | — |
//! | Inspect what is cached | [`FacetStorage::cache_stats`] via [`TestDataView::open_facet_storage`] | — |
//!
//! ### Open and read
//!
//! ```no_run
//! use vectordata::catalog::{Catalog, CatalogSources};
//!
//! # fn main() -> vectordata::Result<()> {
//! // Catalogs come from ~/.config/vectordata/catalogs.yaml.
//! let catalog = Catalog::of(&CatalogSources::new().configure_default());
//! let view = catalog.open_profile("my-dataset", "default")?;
//! let base = view.base_vectors()?;
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
//!
//! # fn main() -> vectordata::Result<()> {
//! let catalog = Catalog::of(&CatalogSources::new().configure_default());
//! let view = catalog.open_spec("my-dataset:default")?.view()?;
//! view.fetch(
//!     &FetchRequest::facets(["base_vectors", "query_vectors"]),
//!     &mut TextMeter::stderr("Fetch"),
//! )?;
//! // Every read below is local and zero-copy.
//! let queries = view.query_vectors()?;
//! # let _ = queries; Ok(()) }
//! ```
//!
//! ### Profiles, facets and typed metadata
//!
//! ```no_run
//! # use vectordata::catalog::{Catalog, CatalogSources};
//! use vectordata::{open_facet_typed, TypedReader};
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
//! // A u8 metadata facet read as i32: widening is automatic.
//! let labels: TypedReader<i32> = open_facet_typed(&*view, "metadata_content")?;
//! let label = labels.get_value(42)?;
//! # let _ = label; Ok(()) }
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
//! 4. **Reader** ([`VectorReader`], [`VvecReader`], [`TypedReader`],
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
pub mod model;
pub mod io;
pub mod access;
pub mod knn_entries;
pub mod metadata_schema;
pub mod typed_access;
pub mod view;
pub mod fetch;
pub mod group;

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
pub use model::FacetConfig;
pub use view::{
    CacheStats, FacetDescriptor, FacetStorage, PrebufferProgress, PrefetchHandle, PrefetchPlan,
    PrefetchReport, RangeFill, TestDataView, WholeFacetFallback, open_facet_typed,
};
pub use io::{VectorReader, VvecReader, XvecReader, IndexedVvecReader, IoError};
pub use cache::InsufficientCacheSpace;
pub use access::AccessMode;
pub use typed_access::{ElementType, TypedAccessError, TypedReader};
/// Re-exported so external callers of
/// [`FacetStorage::prebuffer_with_progress`] can name the
/// progress type their callback receives.
pub use transport::DownloadProgress;

use thiserror::Error;

/// Top-level error type for the vectordata crate.
#[derive(Error, Debug)]
pub enum Error {
    /// A vector I/O operation failed (read, mmap, HTTP fetch).
    #[error("Vector IO error: {0}")]
    VectorIo(#[from] crate::io::IoError),
    /// The `dataset.yaml` file could not be read from disk or network.
    #[error("Failed to read dataset configuration: {0}")]
    ConfigIo(#[source] std::io::Error),
    /// The `dataset.yaml` content is not valid YAML or does not match the schema.
    #[error("Failed to parse dataset configuration: {0}")]
    ConfigParse(#[from] serde_yaml::Error),
    /// A URL string could not be parsed.
    #[error("Invalid URL: {0}")]
    UrlParse(#[from] url::ParseError),
    /// An HTTP request to a remote dataset failed.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    /// A required facet (e.g., `base_vectors`) is not defined in the profile.
    #[error("Required facet not defined: {0}")]
    MissingFacet(String),
    /// The facet was opened through a reader for the wrong shape.
    ///
    /// Not a missing capability: the facet is readable, through the
    /// other path. A `metadata_content.slab` holds opaque records and
    /// has no element width, so asking a vector reader for it is asking
    /// a question it cannot answer — and saying "cannot infer element
    /// size" describes the symptom rather than the situation.
    ///
    /// Carries the shape so a caller can branch on it rather than parse
    /// a message; [`crate::dataset::facet::FacetShape`] is also
    /// available up front through `facet_shape()`, which is the way to
    /// avoid the error entirely.
    #[error(
        "facet '{facet}' holds {shape} and was opened as {attempted}; \
         open it with {reader}"
    )]
    WrongFacetShape {
        /// The facet as declared.
        facet: String,
        /// What the facet actually holds.
        shape: crate::dataset::facet::FacetShape,
        /// The shape the caller's reader expects.
        attempted: crate::dataset::facet::FacetShape,
        /// The reader that does open this facet.
        reader: &'static str,
    },
    /// A fetch named facets the profile does not declare. Nothing was
    /// fetched: fetching the facets that do exist and reporting success
    /// would hide the typo.
    #[error(
        "no such facet(s){}: {}; the profile declares: {}",
        .profile.as_deref().map(|p| format!(" in profile '{p}'")).unwrap_or_default(),
        .missing.join(", "),
        .declared.join(", ")
    )]
    UnknownFacets {
        /// The profile, when the fetch spanned several.
        profile: Option<String>,
        /// The names asked for that the profile does not declare, sorted.
        missing: Vec<String>,
        /// Every facet the profile declares, sorted.
        declared: Vec<String>,
    },
    /// A fetch window cannot be resolved for these facets' formats, so
    /// honouring it means fetching them whole, and the request did not
    /// allow that. Refused before anything was fetched.
    #[error(
        "the window cannot be resolved for {}, so honouring it means fetching \
         the whole facet ({bytes} bytes); allow that with \
         FetchRequest::allow_whole_facet (--allow-whole-facet on the command \
         line), or drop the window",
        .facets.iter().map(|f| format!("'{f}'")).collect::<Vec<_>>().join(", ")
    )]
    WindowUnresolvable {
        /// The facets, as `facet` or `profile/facet`.
        facets: Vec<String>,
        /// Their combined size — what allowing it means fetching, less
        /// whatever is already cached. The number the decision turns on.
        bytes: u64,
    },
    /// The cache directory cannot hold what a fetch would download.
    /// Refused before anything was fetched.
    #[error("{0}")]
    InsufficientCacheSpace(#[from] InsufficientCacheSpace),
    /// No configured catalog has a dataset by this name.
    #[error(
        "dataset '{name}' not found{}",
        if .suggestions.is_empty() { String::new() } else { format!("; did you mean {}?", .suggestions.join(", ")) }
    )]
    UnknownDataset {
        /// The name looked up.
        name: String,
        /// Datasets whose names contain it, for a "did you mean".
        suggestions: Vec<String>,
    },
    /// More than one dataset matches this name (case-insensitively).
    #[error("multiple datasets match '{name}': {}", .matches.join(", "))]
    AmbiguousDataset {
        /// The name looked up.
        name: String,
        /// Every dataset it matches.
        matches: Vec<String>,
    },
    /// A dataset spec or profile selector is malformed, or matches no
    /// profile of the dataset.
    #[error("dataset '{dataset}': {message}")]
    Selection {
        /// The dataset the selector was applied to.
        dataset: String,
        /// What is wrong with it.
        message: String,
    },
    /// Catch-all for errors that do not fit other variants.
    #[error("{0}")]
    Other(String),
}

/// A specialized Result type for the library.
pub type Result<T> = std::result::Result<T, Error>;
