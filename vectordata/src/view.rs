// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Profile views for accessing dataset components.
//!
//! Defines the [`TestDataView`] trait and `GenericTestDataView` implementation
//! for uniform access to base vectors, query vectors, ground-truth neighbors,
//! and metadata facets regardless of backing storage (local mmap or HTTP).
//!
//! The [`FacetDescriptor`] type supports discover-then-load patterns by
//! describing available facets without materializing data.

use crate::group::DataSource;
use crate::io::{self, VectorReader, VvecReader, VvecElement};
use crate::model::{FacetConfig, ProfileConfig};
use crate::{Error, Result};
use crate::dataset::facet::StandardFacet;
use crate::io::IoError;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use url::Url;

/// Parse a window string using the canonical dataset-source parser
/// (`crate::dataset::source::parse_window`). Returns the FIRST
/// interval as `(start, end)` with `end` exclusive — the multi-
/// interval form `"[0..1K, 2K..3K]"` is documented but the reader
/// API only handles a single contiguous range; callers wanting a
/// disjoint window should split into separate facet configs.
///
/// `None` for malformed input or empty windows — callers fall back
/// to the unwrapped reader.
fn parse_window_first(s: &str) -> Option<(usize, usize)> {
    let dsw = crate::dataset::source::parse_window(s).ok()?;
    let iv = dsw.0.into_iter().next()?;
    let start = iv.min_incl as usize;
    let end = iv.max_excl as usize;
    if end < start { return None; }
    Some((start, end))
}

/// Reader wrapper that clips access to a sub-range of the underlying
/// reader. `count()` reports the window length; `get(i)` reads from
/// `underlying.get(i + start)` and rejects `i >= window length`.
///
/// Built when a `FacetConfig` carries a `window` field — the documented
/// sub-ordinal suffix model used by sized profiles to express "first
/// `base_count` rows of the shared base file" without having to copy
/// the file or trust every consumer to honor `view.base_count()`.
struct WindowedVectorReader<T> {
    inner: Box<dyn VectorReader<T>>,
    start: usize,
    /// Window length, capped to the underlying file's count.
    len: usize,
}

impl<T> WindowedVectorReader<T> {
    fn new(inner: Box<dyn VectorReader<T>>, start: usize, end: usize) -> Self {
        let total = inner.count();
        let s = start.min(total);
        let e = end.min(total);
        let len = e.saturating_sub(s);
        WindowedVectorReader { inner, start: s, len }
    }
}

impl<T: Send + Sync> VectorReader<T> for WindowedVectorReader<T> {
    fn dim(&self) -> usize { self.inner.dim() }
    fn count(&self) -> usize { self.len }
    fn get(&self, index: usize) -> std::result::Result<Vec<T>, IoError> {
        if index >= self.len {
            return Err(IoError::InvalidFormat(format!(
                "index {} out of range for windowed reader (len {}, start {})",
                index, self.len, self.start,
            )));
        }
        self.inner.get(self.start + index)
    }
}

/// Describes a single facet declared in a dataset profile.
///
/// Returned by `facet_manifest()` for discover-then-load patterns.
/// Does not materialize data — just describes what's available.
#[derive(Debug, Clone)]
pub struct FacetDescriptor {
    /// Facet name as declared in dataset.yaml (canonical key).
    pub name: String,
    /// Source file path or filename.
    pub source_path: Option<String>,
    /// Inferred source format type (e.g., "fvec", "ivec", "mvec", "slab").
    pub source_type: Option<String>,
    /// Matching StandardFacet if this is a recognized standard facet.
    pub standard_kind: Option<StandardFacet>,
}

impl FacetDescriptor {
    /// Returns true if this is a recognized standard facet.
    pub fn is_standard(&self) -> bool {
        self.standard_kind.is_some()
    }

    /// Infer the source type from a file extension.
    fn infer_type(source: &str) -> Option<String> {
        let ext = source.rsplit('.').next()?;
        match ext {
            "fvec" | "ivec" | "mvec" | "slab" | "json" | "parquet" | "npy" => {
                Some(ext.to_string())
            }
            _ => None,
        }
    }
}

/// Interface for accessing the components of a dataset profile.
///
/// This mirrors the Java `TestDataView` interface. Vector and filtered-neighbor
/// methods return `VectorReader`s; metadata accessors return the `FacetConfig`
/// so callers can resolve the underlying resource.
pub trait TestDataView: Send + Sync {
    // -- Vector facets --

    /// Returns a reader for the base (database) vectors.
    fn base_vectors(&self) -> Result<Arc<dyn VectorReader<f32>>>;
    /// Returns a reader for the query vectors.
    fn query_vectors(&self) -> Result<Arc<dyn VectorReader<f32>>>;
    /// Returns a reader for the neighbor indices (ground truth).
    fn neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>>;
    /// Returns a reader for the neighbor distances (ground truth).
    fn neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>>;

    // -- Filtered neighbor facets --

    /// Returns a reader for the filtered neighbor indices.
    fn filtered_neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>>;
    /// Returns a reader for the filtered neighbor distances.
    fn filtered_neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>>;

    // -- Metadata facets --

    /// Returns the facet config for metadata content, if present.
    fn metadata_content(&self) -> Option<&FacetConfig>;
    /// Returns the facet config for metadata predicates, if present.
    fn metadata_predicates(&self) -> Option<&FacetConfig>;
    /// Returns the facet config for predicate result indices, if present.
    fn predicate_results(&self) -> Option<&FacetConfig>;
    /// Returns the facet config for metadata layout, if present.
    fn metadata_layout(&self) -> Option<&FacetConfig>;

    /// Returns a reader for predicate result indices (metadata_indices).
    /// This is typically a variable-length file (ivvec) where each
    /// predicate maps to a different number of matching base ordinals.
    fn metadata_indices(&self) -> Result<Arc<dyn VvecReader<i32>>>;

    // -- Facet discovery --

    /// Returns descriptors for all facets in the profile, without
    /// materializing data. Includes both standard and custom facets.
    fn facet_manifest(&self) -> HashMap<String, FacetDescriptor>;

    /// Materializes and returns the reader for any named facet.
    ///
    /// For standard vector facets, delegates to the typed accessor.
    /// For custom facets or facets with non-standard types, this is
    /// the generic access path. Returns f32 vectors.
    fn facet(&self, name: &str) -> Result<Arc<dyn VectorReader<f32>>>;

    // -- Typed facet access --

    /// Returns the native element type of a named facet, inferred from
    /// the file extension in the profile config.
    ///
    /// ```rust,ignore
    /// let etype = view.facet_element_type("metadata_content")?; // → ElementType::U8
    /// ```
    fn facet_element_type(&self, name: &str) -> Result<crate::typed_access::ElementType>;

    // -- Profile metadata --

    /// Returns the base vector count for this profile, if declared.
    fn base_count(&self) -> Option<u64>;

    // -- Dataset metadata --

    /// Returns the distance function name if declared in attributes.
    fn distance_function(&self) -> Option<String>;

    /// Resolved source string (path or URL) for a named facet, if it
    /// exists. Used by [`open_facet_typed`] so callers holding an
    /// `Arc<dyn TestDataView>` can construct typed readers without
    /// knowing the dataset's transport details.
    fn facet_source(&self, name: &str) -> Option<String>;

    // -- Prebuffer / cache --

    /// Force-download every standard facet of this profile into the
    /// local cache so subsequent reads are zero-copy. Local-mmap
    /// facets and direct-HTTP (no `.mref`) facets no-op silently.
    ///
    /// The default implementation walks `facet_manifest()` and calls
    /// `prebuffer()` on each opened reader. Profile implementations
    /// may override to parallelise or to filter by facet kind.
    fn prebuffer_all(&self) -> Result<()> {
        self.prebuffer_all_with_progress(&mut |_, _| {})
    }

    /// Same as [`prebuffer_all`] with a callback fired per facet
    /// when its underlying download completes. The callback receives
    /// the facet name and a progress snapshot.
    ///
    /// `cb` is taken by `&mut dyn` rather than as a generic parameter
    /// so this trait remains dyn-compatible (consumers receive
    /// `Arc<dyn TestDataView>` from the catalog API).
    fn prebuffer_all_with_progress(&self, cb: &mut dyn FnMut(&str, &PrebufferProgress)) -> Result<()> {
        for (name, _desc) in self.facet_manifest() {
            // Try element-typed open first so scalar metadata facets
            // (e.g., metadata_content.u8) get the same prebuffer path
            // as vector facets.
            let etype = match self.facet_element_type(&name) {
                Ok(e) => e,
                Err(_) => continue, // unknown extension — skip silently
            };
            // Open through TypedReader to get the underlying Storage
            // and call its prebuffer_with_progress, regardless of the
            // facet's data shape (uniform, scalar, or vvec).
            // Width-i64 widening covers every native element type up
            // to 8 bytes — opens succeed even if the consumer only
            // ever reads it as i32 etc.
            match self.open_facet_storage(&name) {
                Ok(reader) => {
                    let _ = reader.prebuffer_with_progress(|p| {
                        cb(&name, &PrebufferProgress {
                            verified_chunks: p.completed_chunks(),
                            total_chunks: p.total_chunks(),
                            verified_bytes: p.downloaded_bytes(),
                            total_bytes: p.total_bytes(),
                        });
                    });
                    let _ = etype; // suppressed
                }
                Err(_) => continue,
            }
        }
        Ok(())
    }

    /// **Crate-internal hook** used by the default `prebuffer_all`
    /// implementation. Returns a handle whose `prebuffer*` methods
    /// drive the underlying [`crate::storage::Storage`]. Implementors
    /// rarely override this — the `GenericTestDataView` default is
    /// usually correct.
    #[doc(hidden)]
    fn open_facet_storage(&self, name: &str) -> Result<FacetStorage>;
}

/// Open a typed reader for a named facet on any [`TestDataView`].
///
/// This is the dyn-compatible companion to
/// [`GenericTestDataView::open_facet_typed`] — `Arc<dyn TestDataView>`
/// from `Catalog::open_profile` works directly:
///
/// ```no_run
/// # use std::sync::Arc;
/// use vectordata::{open_facet_typed, TypedReader, TestDataView};
/// # fn demo(view: Arc<dyn TestDataView>) -> Result<(), Box<dyn std::error::Error>> {
/// let meta: TypedReader<i32> = open_facet_typed(&*view, "metadata_content")?;
/// let label = meta.get_value(42)?;
/// # let _ = label; Ok(())
/// # }
/// ```
///
/// The native element type is inferred from the facet's source
/// extension via [`TestDataView::facet_element_type`]; the transport
/// is inferred from the source string via [`TypedReader::open_auto`].
pub fn open_facet_typed<T: crate::typed_access::TypedElement>(
    view: &dyn TestDataView,
    facet_name: &str,
) -> std::result::Result<crate::typed_access::TypedReader<T>, crate::typed_access::TypedAccessError> {
    let source = view.facet_source(facet_name)
        .ok_or_else(|| crate::typed_access::TypedAccessError::Io(
            format!("facet '{facet_name}' not declared by this view")))?;
    let native = view.facet_element_type(facet_name)
        .map_err(|e| crate::typed_access::TypedAccessError::Io(e.to_string()))?;
    crate::typed_access::TypedReader::<T>::open_auto(&source, native)
}

/// Snapshot of in-progress prebuffer state for a single facet. Passed
/// to the callback registered with [`TestDataView::prebuffer_all_with_progress`].
#[derive(Debug, Clone)]
pub struct PrebufferProgress {
    pub verified_chunks: u32,
    pub total_chunks: u32,
    pub verified_bytes: u64,
    pub total_bytes: u64,
}

/// Live cache fill statistics for a single facet's underlying storage.
/// Returned by [`FacetStorage::cache_stats`]. `None` when the storage
/// is not cache-backed (e.g., local mmap or direct HTTP).
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Verified chunks in the local cache.
    pub valid_chunks: u32,
    /// Total chunks in the file.
    pub total_chunks: u32,
    /// Total content size in bytes.
    pub content_size: u64,
    /// Whether every chunk is verified.
    pub is_complete: bool,
}

/// Opaque handle to a facet's underlying storage. Returned by
/// [`TestDataView::open_facet_storage`] and consumed by the default
/// `prebuffer_all` implementation. There is no public API for
/// reading bytes through this handle — reads go through the
/// shape-aware reader returned by `base_vectors()`, `facet()`, or
/// `open_facet_typed`.
pub struct FacetStorage {
    storage: std::sync::Arc<crate::storage::Storage>,
}

impl FacetStorage {
    pub(crate) fn new(storage: std::sync::Arc<crate::storage::Storage>) -> Self {
        Self { storage }
    }
    pub fn prebuffer(&self) -> std::io::Result<()> { self.storage.prebuffer() }
    pub fn prebuffer_with_progress<F>(&self, cb: F) -> std::io::Result<()>
    where F: FnMut(&crate::transport::DownloadProgress)
    { self.storage.prebuffer_with_progress(cb) }
    pub fn is_complete(&self) -> bool { self.storage.is_complete() }
    pub fn is_local(&self) -> bool { self.storage.is_local() }
    pub fn total_size(&self) -> u64 { self.storage.total_size() }

    /// Live cache-fill statistics. Returns `None` when the storage is
    /// not cache-backed (local mmap or direct HTTP).
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.storage.cached_channel().map(|c| CacheStats {
            valid_chunks: c.valid_count(),
            total_chunks: c.total_chunks(),
            content_size: c.content_size(),
            is_complete: c.is_complete(),
        })
    }
}

/// A generic implementation of `TestDataView`.
///
/// This struct holds the configuration for a profile and the data source location,
/// creating the appropriate `VectorReader` (Mmap or Http) on demand.
#[derive(Debug)]
pub struct GenericTestDataView {
    source: DataSource,
    config: ProfileConfig,
    /// Dataset-level attributes for metadata accessors.
    attributes: HashMap<String, serde_yaml::Value>,
}

impl GenericTestDataView {
    /// Creates a new `GenericTestDataView`.
    pub fn new(source: DataSource, config: ProfileConfig) -> Self {
        Self {
            source,
            config,
            attributes: HashMap::new(),
        }
    }

    /// Creates a new `GenericTestDataView` with dataset attributes.
    pub fn with_attributes(
        source: DataSource,
        config: ProfileConfig,
        attributes: HashMap<String, serde_yaml::Value>,
    ) -> Self {
        Self {
            source,
            config,
            attributes,
        }
    }

    fn resolve_resource(&self, facet: &FacetConfig) -> Result<ResourceLocation> {
        let source_str = facet.source();
        match &self.source {
            DataSource::FileSystem(base_path) => {
                let path = base_path.join(source_str);
                Ok(ResourceLocation::FileSystem(path))
            }
            DataSource::Http(base_url) => {
                let url = base_url.join(source_str)?;
                Ok(ResourceLocation::Http(url))
            }
        }
    }

    /// Open a uniform vector facet with the unified `open_vec` API.
    /// Handles local (mmap) and remote (HTTP) transparently.
    ///
    /// When the facet config carries a `window` field
    /// (`profiles.X.base_vectors.window: "0..N"`), the returned
    /// reader is wrapped so that `count()` reports `N - 0` and
    /// `get(i)` is offset into the underlying file. This is the
    /// documented sub-ordinal suffix model — sized profiles inherit
    /// `base_vectors` from default with `[0..base_count)` so every
    /// consumer reading via the trait gets a clipped reader without
    /// having to honor `view.base_count()` manually.
    fn open_uniform<T: VvecElement>(
        &self,
        facet_opt: Option<&FacetConfig>,
        name: &str,
    ) -> Result<Arc<dyn VectorReader<T>>> {
        let facet = facet_opt.ok_or_else(|| Error::MissingFacet(name.to_string()))?;

        // Two ways the dataset config can carry a sub-ordinal range:
        //   1. `Detailed { source, window }` — explicit `window:` field.
        //   2. `Simple("path[0..N)")` — the documented suffix sugar
        //      that `parse_source_string` understands.
        // Try the suffix form first (so `Simple` strings work without
        // forcing the consumer to use `Detailed`); fall back to the
        // explicit field. Either path produces the same windowed
        // reader behind the trait.
        let raw = facet.source();
        let (path_str, window_from_suffix): (String, Option<(usize, usize)>) =
            match crate::dataset::source::parse_source_string(raw) {
                Ok(parsed) if !parsed.window.is_empty() => {
                    let iv = &parsed.window.0[0];
                    (parsed.path, Some((iv.min_incl as usize, iv.max_excl as usize)))
                }
                _ => (raw.to_string(), None),
            };
        let resolved = self.resolve_path_str(&path_str)?;
        let reader = io::open_vec::<T>(&resolved)?;

        let window = window_from_suffix.or_else(|| {
            facet.window().and_then(parse_window_first)
        });
        if let Some((start, end)) = window {
            return Ok(Arc::new(WindowedVectorReader::new(reader, start, end)));
        }
        Ok(Arc::from(reader))
    }

    /// Resolve a bare path string (no window suffix) against the
    /// data source root. Mirrors `resolve_as_string` but takes a raw
    /// path so callers that pre-parsed the suffix can join it
    /// correctly without round-tripping through `FacetConfig`.
    fn resolve_path_str(&self, path: &str) -> Result<String> {
        match &self.source {
            DataSource::FileSystem(base_path) => {
                Ok(base_path.join(path).to_string_lossy().to_string())
            }
            DataSource::Http(base_url) => {
                let url = base_url.join(path)?;
                Ok(url.to_string())
            }
        }
    }

    /// Open a variable-length vector facet with the unified `open_vvec` API.
    /// Handles local (mmap + index) and remote (HTTP + index) transparently.
    fn open_variable<T: VvecElement>(
        &self,
        facet_opt: Option<&FacetConfig>,
        name: &str,
    ) -> Result<Arc<dyn VvecReader<T>>> {
        let facet = facet_opt.ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        let path_or_url = self.resolve_as_string(facet)?;
        let reader = io::open_vvec::<T>(&path_or_url)?;
        Ok(Arc::from(reader))
    }

    /// Resolve a facet to a path string (local) or URL string (remote).
    fn resolve_as_string(&self, facet: &FacetConfig) -> Result<String> {
        let source_str = facet.source();
        match &self.source {
            DataSource::FileSystem(base_path) => {
                Ok(base_path.join(source_str).to_string_lossy().to_string())
            }
            DataSource::Http(base_url) => {
                let url = base_url.join(source_str)?;
                Ok(url.to_string())
            }
        }
    }

    /// Collect all facets declared in the profile config.
    fn collect_facets(&self) -> HashMap<String, FacetDescriptor> {
        let mut manifest = HashMap::new();

        let standard_facets: &[(&str, Option<&FacetConfig>, StandardFacet)] = &[
            ("base_vectors", self.config.base_vectors.as_ref(), StandardFacet::BaseVectors),
            ("query_vectors", self.config.query_vectors.as_ref(), StandardFacet::QueryVectors),
            ("neighbor_indices", self.config.neighbor_indices.as_ref(), StandardFacet::NeighborIndices),
            ("neighbor_distances", self.config.neighbor_distances.as_ref(), StandardFacet::NeighborDistances),
            ("metadata_content", self.config.metadata_content.as_ref(), StandardFacet::MetadataContent),
            ("metadata_predicates", self.config.metadata_predicates.as_ref(), StandardFacet::MetadataPredicates),
            ("predicate_results", self.config.predicate_results.as_ref(), StandardFacet::MetadataResults),
            ("metadata_layout", self.config.metadata_layout.as_ref(), StandardFacet::MetadataLayout),
            ("filtered_neighbor_indices", self.config.filtered_neighbor_indices.as_ref(), StandardFacet::FilteredNeighborIndices),
            ("filtered_neighbor_distances", self.config.filtered_neighbor_distances.as_ref(), StandardFacet::FilteredNeighborDistances),
        ];

        for (name, facet_opt, kind) in standard_facets {
            if let Some(facet) = facet_opt {
                let source = facet.source().to_string();
                manifest.insert(
                    name.to_string(),
                    FacetDescriptor {
                        name: name.to_string(),
                        source_type: FacetDescriptor::infer_type(&source),
                        source_path: Some(source),
                        standard_kind: Some(*kind),
                    },
                );
            }
        }

        manifest
    }
}

impl GenericTestDataView {
    /// Look up a facet by name and return its FacetConfig.
    fn facet_config_by_name(&self, name: &str) -> Option<&FacetConfig> {
        match name {
            "base_vectors" => self.config.base_vectors.as_ref(),
            "query_vectors" => self.config.query_vectors.as_ref(),
            "neighbor_indices" => self.config.neighbor_indices.as_ref(),
            "neighbor_distances" => self.config.neighbor_distances.as_ref(),
            "metadata_content" => self.config.metadata_content.as_ref(),
            "metadata_predicates" => self.config.metadata_predicates.as_ref(),
            "predicate_results" => self.config.predicate_results.as_ref(),
            "metadata_layout" => self.config.metadata_layout.as_ref(),
            "filtered_neighbor_indices" => self.config.filtered_neighbor_indices.as_ref(),
            "filtered_neighbor_distances" => self.config.filtered_neighbor_distances.as_ref(),
            _ => None,
        }
    }

    /// Open a named facet as a typed reader.
    ///
    /// Fails at open time if T is narrower than the native element type.
    /// Same-width cross-sign (e.g., u8↔i8) is allowed but checked per-value.
    ///
    /// ```rust,ignore
    /// // Open with native type — zero-copy access
    /// let r = view.open_facet_typed::<u8>("metadata_content")?;
    ///
    /// // Open with wider type — always succeeds
    /// let r = view.open_facet_typed::<i32>("metadata_content")?;
    /// ```
    pub fn open_facet_typed<T: crate::typed_access::TypedElement>(
        &self,
        name: &str,
    ) -> std::result::Result<crate::typed_access::TypedReader<T>, crate::typed_access::TypedAccessError> {
        let facet = self.facet_config_by_name(name)
            .ok_or_else(|| crate::typed_access::TypedAccessError::Io(
                format!("facet '{}' not found", name)))?;
        let resource = self.resolve_resource(facet)
            .map_err(|e| crate::typed_access::TypedAccessError::Io(e.to_string()))?;
        match resource {
            ResourceLocation::FileSystem(path) => {
                crate::typed_access::TypedReader::<T>::open(&path)
            }
            ResourceLocation::Http(url) => {
                let native_type = crate::typed_access::ElementType::from_url(&url)
                    .map_err(|e| crate::typed_access::TypedAccessError::Io(e))?;
                crate::typed_access::TypedReader::<T>::open_url(url, native_type)
            }
        }
    }
}

#[allow(dead_code)]
enum ResourceLocation {
    FileSystem(PathBuf),
    Http(Url),
}

impl TestDataView for GenericTestDataView {
    fn base_vectors(&self) -> Result<Arc<dyn VectorReader<f32>>> {
        self.open_uniform(self.config.base_vectors.as_ref(), "base_vectors")
    }

    fn query_vectors(&self) -> Result<Arc<dyn VectorReader<f32>>> {
        self.open_uniform(self.config.query_vectors.as_ref(), "query_vectors")
    }

    fn neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>> {
        self.open_uniform(self.config.neighbor_indices.as_ref(), "neighbor_indices")
    }

    fn neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>> {
        self.open_uniform(self.config.neighbor_distances.as_ref(), "neighbor_distances")
    }

    fn filtered_neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>> {
        self.open_uniform(
            self.config.filtered_neighbor_indices.as_ref(),
            "filtered_neighbor_indices",
        )
    }

    fn filtered_neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>> {
        self.open_uniform(
            self.config.filtered_neighbor_distances.as_ref(),
            "filtered_neighbor_distances",
        )
    }

    fn metadata_content(&self) -> Option<&FacetConfig> {
        self.config.metadata_content.as_ref()
    }

    fn metadata_predicates(&self) -> Option<&FacetConfig> {
        self.config.metadata_predicates.as_ref()
    }

    fn predicate_results(&self) -> Option<&FacetConfig> {
        self.config.predicate_results.as_ref()
    }

    fn metadata_layout(&self) -> Option<&FacetConfig> {
        self.config.metadata_layout.as_ref()
    }

    fn facet_manifest(&self) -> HashMap<String, FacetDescriptor> {
        self.collect_facets()
    }

    fn metadata_indices(&self) -> Result<Arc<dyn VvecReader<i32>>> {
        self.open_variable(self.config.predicate_results.as_ref(), "metadata_indices")
    }

    fn facet(&self, name: &str) -> Result<Arc<dyn VectorReader<f32>>> {
        // Try standard facets first (uniform vector types)
        match name {
            "base_vectors" => return self.base_vectors(),
            "query_vectors" => return self.query_vectors(),
            "neighbor_distances" => return self.neighbor_distances(),
            "filtered_neighbor_distances" => return self.filtered_neighbor_distances(),
            _ => {}
        }

        // For f32-compatible facets, open via unified API
        let facet_config = self.facet_config_by_name(name);
        match facet_config {
            Some(fc) => self.open_uniform::<f32>(Some(fc), name),
            None => Err(Error::MissingFacet(name.to_string())),
        }
    }

    fn facet_element_type(&self, name: &str) -> Result<crate::typed_access::ElementType> {
        let facet = self.facet_config_by_name(name)
            .ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        let source = facet.source();
        crate::typed_access::ElementType::from_extension(
            source.rsplit('.').next().unwrap_or("")
        ).ok_or_else(|| Error::Other(format!("unknown element type for facet '{name}'")))
    }

    fn base_count(&self) -> Option<u64> {
        self.config.base_count
    }

    fn distance_function(&self) -> Option<String> {
        self.attributes
            .get("distance_function")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    fn facet_source(&self, name: &str) -> Option<String> {
        let facet = self.facet_config_by_name(name)?;
        let raw = facet.source();
        let path_str = match crate::dataset::source::parse_source_string(raw) {
            Ok(parsed) => parsed.path,
            Err(_) => raw.to_string(),
        };
        self.resolve_path_str(&path_str).ok()
    }

    fn open_facet_storage(&self, name: &str) -> Result<FacetStorage> {
        let facet = self.facet_config_by_name(name)
            .ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        // Strip any window suffix from the source so Storage opens
        // the underlying file/url, not a half-parsed token.
        let raw = facet.source();
        let path_str = match crate::dataset::source::parse_source_string(raw) {
            Ok(parsed) => parsed.path,
            Err(_) => raw.to_string(),
        };
        let resolved = self.resolve_path_str(&path_str)?;
        let storage = std::sync::Arc::new(crate::storage::Storage::open(&resolved)
            .map_err(|e| Error::Other(format!("storage open '{name}': {e}")))?);
        Ok(FacetStorage::new(storage))
    }
}
