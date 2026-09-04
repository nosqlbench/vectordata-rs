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

use crate::dataset::facet::StandardFacet;
use crate::dataset::source::DSWindow;
use crate::group::DataSource;
use crate::io::IoError;
use crate::io::{self, VectorReader, VvecElement, VvecReader};
use crate::model::{FacetConfig, ProfileConfig};
use crate::{Error, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use url::Url;

/// True for `http://` and `https://` schemes — the only schemes the
/// crate handles as remote.
fn is_absolute_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

/// True when `s` starts with `<scheme>://` for any RFC 3986-style
/// scheme (letter followed by letters/digits/`+`/`-`/`.`). Used to
/// short-circuit base-URL joining for any absolutely-scoped facet
/// path the catalog might publish (`http://`, `https://`,
/// `file://`, `s3://`, `gs://`, etc.) — even when the actual
/// scheme isn't one vectordata's I/O layer speaks. Joining e.g.
/// `s3://x/y` onto a base URL would produce a bogus double-
/// prefixed path that nothing can open; passing the original
/// through lets the storage layer fail with a precise
/// "scheme not supported" error instead.
fn has_absolute_uri_scheme(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphabetic() {
        return false;
    }
    let scheme_end = s.find("://").unwrap_or(usize::MAX);
    if scheme_end == usize::MAX {
        return false;
    }
    s[1..scheme_end]
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

/// True for facet paths that should bypass the catalog's base
/// location and open directly as a local file. Covers absolute
/// filesystem paths (`/abs/foo.fvec`) and `file://` URIs — both
/// are treated identically by the I/O stack. The point is parity
/// with a fully-precached remote facet: no URL joins, no
/// download/cache layer, just mmap.
fn is_local_facet_source(s: &str) -> bool {
    s.starts_with("file://") || s.starts_with('/')
}

/// Strip a `file://` prefix to produce a plain filesystem path
/// string. Pass-through for everything else. Mirrors
/// `crate::storage::local_source_path` but lives at the view
/// layer so we can avoid a cross-module dependency for what is a
/// trivial inline transform.
fn file_uri_to_path(s: &str) -> &str {
    if let Some(rest) = s.strip_prefix("file://") {
        // `file:///abs/path` → `/abs/path`
        // `file://host/abs/path` → `/abs/path` (host dropped)
        if rest.starts_with('/') {
            return rest;
        }
        if let Some(slash) = rest.find('/') {
            return &rest[slash..];
        }
        return rest;
    }
    s
}

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
    if end < start {
        return None;
    }
    Some((start, end))
}

/// Resolve a facet's window bounds from whichever encoding the config
/// carries, mirroring `open_uniform`: the `[start..end)` suffix on the
/// source string takes precedence, else the explicit `window:` field.
///
/// A malformed window is an error, not an absent window. Conflating
/// them is what let `base.fvec[0,1000)` fall through to an unbounded
/// prebuffer and download a whole terabyte in silence.
fn parse_window_bounds(
    raw_source: &str,
    explicit_window: Option<&str>,
) -> Result<Option<(String, u64, u64)>> {
    let parsed = crate::dataset::source::parse_source_string(raw_source)
        .map_err(|e| Error::Other(format!("source '{raw_source}' has a malformed window: {e}")))?;
    let (path_no_window, win_start, win_end): (String, u64, u64) = if !parsed.window.is_empty() {
        let iv = &parsed.window.0[0];
        (parsed.path, iv.min_incl, iv.max_excl)
    } else if let Some(w) = explicit_window {
        let dsw = crate::dataset::source::parse_window(w)
            .map_err(|e| Error::Other(format!("window '{w}' is malformed: {e}")))?;
        match dsw.0.first() {
            Some(iv) => (parsed.path, iv.min_incl, iv.max_excl),
            None => return Ok(None),
        }
    } else {
        return Ok(None);
    };
    // Empty intervals no longer parse, so this is belt-and-braces.
    if win_end <= win_start {
        return Ok(None);
    }
    Ok(Some((path_no_window, win_start, win_end)))
}

/// The single source of a facet, or an error naming sharding as the
/// reason there is not one.
///
/// Every caller of this predates multi-file facets and reads one path.
/// A series has no single path, and a caller written before sharding
/// must fail visibly rather than resolve to its first shard
/// (`docs/design/srd-multifile-facet-shards.md`, SH-74). These call
/// sites go away when the reader layer consumes the realized `Shards`
/// model.
fn single_source<'a>(facet: &'a FacetConfig, what: &str) -> Result<&'a str> {
    facet.source().ok_or_else(|| {
        Error::Other(format!(
            "{what}: this facet declares a multi-file series, which this \
             path does not resolve yet — see \
             docs/design/srd-multifile-facet-shards.md"
        ))
    })
}

/// How many records a file holds — the inverse of
/// [`record_range_to_bytes`], sharing its format dispatch so the two
/// cannot disagree about what a record is.
///
/// Consulted only for a bare series entry, whose length the declaration
/// does not state (SH-63). `None` when the format offers no answer from
/// the file alone.
pub(crate) fn records_in(path: &str, storage: &crate::storage::Storage) -> Option<u64> {
    let ext = path.rsplit('.').next().unwrap_or("");
    let elem_size = crate::io::infer_elem_size(ext);
    if elem_size == 0 {
        return None;
    }
    let total = storage.total_size();
    if crate::io::is_vvec_ext(ext) {
        // Variable-length: only the index knows. A bare entry is
        // local-only, so the walk this may cost is a local walk.
        let offsets =
            crate::io::load_offsets(path, storage, elem_size, crate::io::OffsetSource::Rebuild)
                .ok()?;
        return Some(offsets.len() as u64);
    }
    if crate::io::is_scalar_ext(ext) {
        return Some(total / elem_size as u64);
    }
    // Uniform xvec: one dim header governs the whole file.
    let header = storage.read_bytes(0, 4).ok()?;
    if header.len() != 4 {
        return None;
    }
    let dim = i32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if dim <= 0 || dim > 1_000_000 {
        return None;
    }
    let bpr = 4 + (dim as u64) * (elem_size as u64);
    (bpr > 0).then(|| total / bpr)
}

/// Map a record range to a byte range, for formats where that mapping
/// is computable from the file alone.
///
/// Split out so the mapping can serve
/// a window the *caller* named as well as one a profile declared. A
/// profile window is a convenience — a name for a range someone will
/// want repeatedly — not the only range anyone is allowed to ask for.
///
/// Two mappings, by format:
///   - **xvec** (uniform stride): `4 + dim × elem_size`, with `dim`
///     read from the header at byte 0. That read pulls one chunk on
///     remote storage, which is the same first-chunk read any reader
///     does on first access.
///   - **vvec** (variable length): the sibling `IDXFOR__` offset index,
///     loaded whole. Record `i` begins at `offsets[i]`, so a window is
///     `offsets[start]` to `offsets[end]` — exact, not approximate.
///
/// `None`, meaning "cannot window this", for parquet — whose ordinal
/// windowing is **excluded by design**, since its row-group structure
/// has no record-to-byte mapping a caller could predict — and for
/// unrecognized extensions, a corrupt header, an index that cannot be
/// loaded, or an empty range.
///
/// `None` is not a failure and not a placeholder. It is the answer, and
/// [`WholeFacetFallback`] decides what a caller does with it.
/// A record range resolved to bytes, and what resolving it cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MappedRange {
    /// The shard-qualified byte ranges the record window resolved to.
    ///
    /// One entry for a single-file facet; one per shard the window
    /// touches for a series (SH-14, SH-28).
    pub ranges: Vec<ShardRange>,
    /// Bytes that had to be read before the mapping could be computed
    /// at all — the offset index, for variable-length formats. Zero
    /// where the stride is computable from the header.
    pub prerequisite_bytes: u64,
}

/// Map a record range to shard-qualified byte ranges.
///
/// For a single file this is one range in shard `0` — the old meaning
/// under the new shape. For a series the window is decomposed per shard
/// (SH-14) and each sub-window mapped against **its own file**, in that
/// file's ordinals (SH-64).
pub(crate) fn record_range_to_bytes(
    path_no_window: &str,
    win_start: u64,
    win_end: u64,
    storage: &FacetStorage,
) -> Option<MappedRange> {
    if win_end <= win_start {
        return None;
    }
    match storage.series_ref() {
        None => {
            let m = map_in_file(
                path_no_window,
                win_start,
                win_end,
                &storage.storage,
                &|elem| storage.published_offsets(path_no_window, elem),
            )?;
            Some(MappedRange {
                ranges: vec![ShardRange::whole(m.0, m.1)],
                prerequisite_bytes: m.2,
            })
        }
        Some(series) => {
            let mut ranges = Vec::new();
            let mut prerequisite_bytes = 0u64;
            for w in series.shards().decompose(win_start, win_end) {
                let file = series.file_index_of_shard(w.shard).ok()?;
                let st = series.file(file).ok()?;
                let path = series.shards().entries()[w.shard].source.path.clone();
                let m = map_in_file(&path, w.file.0, w.file.1, &st, &|elem| {
                    series.published_offsets(file, &path, elem)
                })?;
                // Summed, not maxed: each shard's index is a separate
                // read, and a window touching three shards pays for
                // three (SH-29).
                prerequisite_bytes = prerequisite_bytes.saturating_add(m.2);
                ranges.push(ShardRange {
                    shard: w.shard,
                    start: m.0,
                    end: m.1,
                });
            }
            (!ranges.is_empty()).then_some(MappedRange {
                ranges,
                prerequisite_bytes,
            })
        }
    }
}

/// Map a record range **within one file** to `(start, end, prerequisite)`.
///
/// `offsets` is consulted only for variable-length formats, and is a
/// closure so the caller decides where the index comes from: a facet
/// handle's cache for a single file, the series' per-file cache for a
/// shard.
fn map_in_file(
    path_no_window: &str,
    lo: u64,
    hi: u64,
    storage: &crate::storage::Storage,
    offsets: &dyn Fn(usize) -> Option<std::sync::Arc<Vec<u64>>>,
) -> Option<(u64, u64, u64)> {
    if hi <= lo {
        return None;
    }
    // A slab carries its own ordinal index in its tail, so a window
    // maps to the pages it spans. Without this branch the extension
    // fails `infer_elem_size` below and every slab window degrades to a
    // whole-facet fetch — which would leave the planner claiming a cost
    // the reader does not pay, since a record read costs its page.
    if slab_ext(path_no_window) {
        return map_in_slab(path_no_window, lo, hi, storage);
    }

    let ext = path_no_window.rsplit('.').next().unwrap_or("");
    let elem_size = crate::io::infer_elem_size(ext);
    if elem_size == 0 {
        return None;
    }
    let total = storage.total_size();

    if crate::io::is_vvec_ext(ext) {
        // Planning must not move bytes to decide what to move: a remote
        // vvec with no published sidecar returns `None` here and the
        // window degrades to a whole-facet fetch, which the fallback
        // policy then consents to or refuses. Rebuilding the index by
        // walking the file would download all of it *while planning*,
        // defeating the gate.
        let offsets = offsets(elem_size)?;
        if offsets.is_empty() {
            return None;
        }
        let count = offsets.len() as u64;
        if lo >= count {
            return None;
        }
        let byte_start = offsets[lo as usize];
        // A window running past the last record ends at the file, not at
        // a record that does not exist.
        let byte_end = if hi >= count {
            total
        } else {
            offsets[hi as usize]
        };
        if byte_end <= byte_start {
            return None;
        }
        // The index had to be read whole to answer this at all.
        //
        // Counted as **record starts × 8**, not sidecar-file bytes: the
        // published widths differ (`i32` vs `i64`) and the sentinel
        // layout carries one entry more than there are records, so the
        // on-disk size is not a stable measure of the same thing.
        // `offsets` is post-sentinel — see `io::parse_index_bytes` — so
        // both sidecar layouts report the same figure here.
        let prereq = (offsets.len() * std::mem::size_of::<u64>()) as u64;
        return Some((byte_start, byte_end, prereq));
    }

    if crate::io::is_scalar_ext(ext) {
        // Scalar facets are raw packed values with no per-record
        // header — `TypedReader` addresses them at `ordinal *
        // elem_size`. Falling through to the xvec path below would
        // read the first *value* as a dimension: a needless degrade
        // when it fails the sanity check, and silently wrong byte
        // ranges when it happens to pass.
        let stride = elem_size as u64;
        let byte_start = lo.saturating_mul(stride).min(total);
        let byte_end = hi.saturating_mul(stride).min(total);
        return (byte_start < byte_end).then_some((byte_start, byte_end, 0));
    }

    // Uniform xvec: one dim header governs the file.
    let header = storage.read_bytes(0, 4).ok()?;
    if header.len() != 4 {
        return None;
    }
    let dim = i32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if dim <= 0 || dim > 1_000_000 {
        return None; // sanity vs corrupt header
    }
    let bpr = 4 + (dim as u64) * (elem_size as u64);
    let byte_start = lo.saturating_mul(bpr).min(total);
    let byte_end = hi.saturating_mul(bpr).min(total);
    // A uniform-stride mapping costs a 4-byte header read, which every
    // reader pays on first access anyway — nothing to report.
    (byte_start < byte_end).then_some((byte_start, byte_end, 0))
}

/// Whether a locator names a slab, ignoring any `:namespace` suffix.
fn slab_ext(locator: &str) -> bool {
    let path = locator.split(':').next().unwrap_or(locator);
    path.rsplit('.')
        .next()
        .is_some_and(|e| e.eq_ignore_ascii_case("slab"))
}

/// Map an ordinal window in a slab to the byte range of the pages that
/// hold it.
///
/// Pages are laid out in ordinal order, so a contiguous window is a
/// contiguous span of pages and therefore one byte range — the same
/// shape every other branch returns. The index comes from the tail
/// (`records::read_slab_index`), which is a bounded read of two short
/// pages rather than a walk of the file, so planning stays cheap in the
/// sense SH-21 requires: it must not move the bytes it is deciding
/// whether to move.
fn map_in_slab(
    locator: &str,
    lo: u64,
    hi: u64,
    storage: &crate::storage::Storage,
) -> Option<(u64, u64, u64)> {
    let namespace = locator.split_once(':').map(|(_, ns)| ns);
    let index = crate::records::read_slab_index(storage, namespace, locator)
        .ok()
        .flatten()?;
    if index.total() == 0 {
        return None;
    }
    let first = index.page_of(lo)?;
    // A window running past the last record ends at the last page, not
    // at one that does not exist.
    let last = index.page_of(hi - 1).unwrap_or(index.page_count() - 1);
    let byte_start = index.page_offset(first)?;
    let last_start = index.page_offset(last)?;
    // The final page's extent comes from its own header, which is the
    // only place its size is written.
    let size = storage
        .read_bytes(last_start + 4, 4)
        .ok()
        .and_then(|b| b.try_into().ok())
        .map(u32::from_le_bytes)? as u64;
    let byte_end = (last_start + size).min(storage.total_size());
    (byte_start < byte_end).then_some((byte_start, byte_end, index.prerequisite_bytes()))
}

/// The window a facet declares for itself, in facet ordinals: a
/// `[start..end)` suffix on a single source, else the `window:` field
/// (which is where a series carries it, having no single source), else
/// none. This is the window the whole-profile precache honours per
/// facet, and it comes from the same two encodings the reader consults.
/// A malformed window is an error, not an absent one.
pub fn facet_declared_window(desc: &FacetDescriptor) -> Result<DSWindow> {
    match desc.source_path.as_deref() {
        Some(src) => Ok(match parse_window_bounds(src, desc.window.as_deref())? {
            Some((_, start, end)) => DSWindow(vec![crate::dataset::source::DSInterval {
                min_incl: start,
                max_excl: end,
            }]),
            None => DSWindow(Vec::new()),
        }),
        None => match desc.window.as_deref() {
            Some(w) => crate::dataset::source::parse_window(w)
                .map_err(|e| Error::Other(format!("window '{w}' is malformed: {e}"))),
            None => Ok(DSWindow(Vec::new())),
        },
    }
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
        WindowedVectorReader {
            inner,
            start: s,
            len,
        }
    }
}

impl<T: Send + Sync> VectorReader<T> for WindowedVectorReader<T> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn count(&self) -> usize {
        self.len
    }
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
    /// Source file path or filename. Retains any `[start..end)`
    /// sub-ordinal suffix the catalog declared inline.
    pub source_path: Option<String>,
    /// Inferred source format type (e.g., "fvec", "ivec", "mvec", "slab").
    pub source_type: Option<String>,
    /// Explicit `window:` field from a `Detailed` facet config, when
    /// present. The suffix form stays embedded in `source_path`;
    /// consumers that need the effective window must consult both.
    pub window: Option<String>,
    /// Matching StandardFacet if this is a recognized standard facet.
    pub standard_kind: Option<StandardFacet>,
}

impl FacetDescriptor {
    /// Returns true if this is a recognized standard facet.
    pub fn is_standard(&self) -> bool {
        self.standard_kind.is_some()
    }

    /// Infer the source type from a file extension. Accepts any
    /// recognized xvec/vvec/scalar extension (including plural forms
    /// like `fvecs`, `mvecs`) via
    /// [`crate::typed_access::ElementType::from_extension`], plus the
    /// container extensions (`slab`, `json`, `parquet`, `npy`, `hdf5`,
    /// `h5`). The returned string is the lowercase canonical form.
    fn infer_type(source: &str) -> Option<String> {
        // Strip the `[start..end)` window suffix and any `:namespace`
        // before looking at the extension. A window contains dots, so
        // `base.fvec[0..20)` splits to `20)` and the facet reports no
        // format at all — which is invisible while the answer is only
        // shown to a human, and load-bearing the moment anything gates
        // on it. `facet_element_type` has always done this; the
        // manifest did not.
        let path = match crate::dataset::source::parse_source_string(source) {
            Ok(parsed) => parsed.path,
            Err(_) => source.to_string(),
        };
        let ext = path.rsplit('.').next()?;
        let lower = ext.to_lowercase();
        if crate::typed_access::ElementType::from_extension(&lower).is_some() {
            return Some(lower);
        }
        match lower.as_str() {
            "slab" | "json" | "parquet" | "npy" | "hdf5" | "h5" => Some(lower),
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

    // -- Filtered neighbor facets (E / F) --

    /// Returns a reader for the **pre-filter** KNN ground-truth indices (E).
    /// Top-K of `X_p` by distance; full K when `|X_p| ≥ K`.
    fn prefiltered_neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>>;
    /// Returns a reader for the **pre-filter** KNN ground-truth distances (E).
    fn prefiltered_neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>>;

    /// Returns a reader for the **post-filter** KNN ground-truth indices (F).
    /// `G ∩ R` — the unfiltered top-K intersected with the predicate-passing
    /// set; sparse possible. The legacy `filtered_neighbor_indices` YAML key
    /// resolves to this facet for backwards compatibility.
    fn postfiltered_neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>>;
    /// Returns a reader for the **post-filter** KNN ground-truth distances (F).
    fn postfiltered_neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>>;

    // -- Metadata facets --

    /// Returns the facet config for metadata content, if present.
    fn metadata_content(&self) -> Option<&FacetConfig>;
    /// Returns the facet config for metadata predicates, if present.
    fn metadata_predicates(&self) -> Option<&FacetConfig>;
    /// Returns the facet config for predicate result indices, if present.
    fn predicate_results(&self) -> Option<&FacetConfig>;
    /// Returns the facet config for metadata layout, if present.
    fn metadata_layout(&self) -> Option<&FacetConfig>;

    /// Returns a reader for the predicate-result indices (the `R` facet,
    /// canonically `metadata_results`; legacy `metadata_indices` is still
    /// resolved on read). This is typically a variable-length file (ivvec)
    /// where each predicate maps to a different number of matching base
    /// ordinals.
    fn metadata_results(&self) -> Result<Arc<dyn VvecReader<i32>>>;

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

    // -- Precache / cache --

    /// Drive every facet of this profile to fully-resident,
    /// zero-copy state. **Strict contract**: returning `Ok(())`
    /// means every facet is local and mmap-promoted. Local-mmap
    /// facets are already complete (no work). Cached-remote and
    /// direct-HTTP-without-`.mref` facets are downloaded fully and
    /// promoted. Per-facet failure surfaces as `Err` — callers
    /// cannot continue with partial state.
    fn prebuffer_all(&self) -> Result<()> {
        self.prebuffer_all_with_progress(WholeFacetFallback::Refuse, &mut |_, _| {})
    }

    /// Same as [`prebuffer_all`] with a callback fired per facet
    /// after its download completes. The callback receives the
    /// facet name and a progress snapshot.
    ///
    /// **Strict contract**: returning `Ok(())` means every declared
    /// facet is fully resident and zero-copy ready. Per-facet
    /// failure is propagated as `Err`, never swallowed — callers
    /// cannot accidentally continue with partial state.
    ///
    /// Every facet is fetched against the window it declares, a series
    /// included; a declared window the format cannot map is refused
    /// under [`WholeFacetFallback::Refuse`] and fetched whole under
    /// [`WholeFacetFallback::Allow`]. `cb` is taken by `&mut dyn`
    /// rather than as a generic parameter so this trait remains
    /// dyn-compatible (consumers receive `Arc<dyn TestDataView>` from
    /// the catalog API).
    fn prebuffer_all_with_progress(
        &self,
        fallback: WholeFacetFallback,
        cb: &mut dyn FnMut(&str, &PrebufferProgress),
    ) -> Result<()> {
        use std::cell::{Cell, RefCell};

        // One plan per facet, from the planner the selective precache
        // uses, against the window the facet declares for itself. The
        // per-file mapping this replaced answered "no window" for a
        // series, which has no single source path, so a sized profile
        // over a sharded base downloaded the whole base: a "small part"
        // of tessera came to two terabytes. The planner decomposes a
        // facet window across shards (SH-14), so a profile pulls what
        // it can address and nothing more.
        let mut planned: Vec<(String, FacetStorage, PrefetchPlan)> = Vec::new();
        for (name, desc) in self.facet_manifest() {
            if !self.facet_holds_data(&name) {
                continue;
            }
            let storage = self
                .open_facet_storage(&name)
                .map_err(|e| Error::Other(format!("open '{name}' for precache: {e}")))?;
            let window = facet_declared_window(&desc)?;
            let plan = self.prefetch_plan_on(&storage, &name, &window)?;
            // A declared window the format cannot map is refused unless
            // the caller accepted the whole facet, exactly as a
            // requested window is. Widening a profile's own window to
            // its whole base in silence is not a fallback; it is the
            // download the window existed to prevent.
            check_fallback(&name, &plan, fallback)?;
            planned.push((name, storage, plan));
        }
        let bytes_to_fetch: u64 = planned
            .iter()
            .map(|(_, storage, plan)| {
                if plan.degrades_to_full_download {
                    storage
                        .total_size()
                        .saturating_sub(storage.allocated_cache_bytes())
                } else {
                    plan.bytes_to_fetch()
                }
            })
            .sum();
        crate::cache::ensure_cache_capacity(bytes_to_fetch)
            .map_err(|e| Error::Other(e.to_string()))?;

        for (name, storage, plan) in &planned {
            // The "total" a meter tops out at is what this facet will
            // hold when done: the window's bytes for a windowed facet,
            // the file for a whole one.
            let total_for_display: u64 = if plan.degrades_to_full_download {
                storage.total_size()
            } else {
                plan.byte_ranges.iter().map(|r| r.len()).sum()
            };
            {
                let p = PrebufferProgress {
                    verified_chunks: 0,
                    total_chunks: 0,
                    verified_bytes: 0,
                    total_bytes: total_for_display,
                };
                cb(name, &p);
            }

            let cb_cell: RefCell<&mut dyn FnMut(&str, &PrebufferProgress)> = RefCell::new(&mut *cb);
            let fired = Cell::new(false);
            let forward = |p: &crate::transport::DownloadProgress| {
                let progress = PrebufferProgress {
                    verified_chunks: p.completed_chunks(),
                    total_chunks: p.total_chunks(),
                    verified_bytes: p.downloaded_bytes(),
                    total_bytes: p.total_bytes(),
                };
                (cb_cell.borrow_mut())(name, &progress);
                fired.set(true);
            };
            let result = if plan.degrades_to_full_download {
                storage.prebuffer_with_progress(&forward)
            } else {
                plan.byte_ranges
                    .iter()
                    .try_for_each(|r| storage.prebuffer_shard_range(r.shard, r.start, r.end, &forward))
            };
            result.map_err(|e| Error::Other(format!("precache '{name}': {e}")))?;

            if !fired.get() {
                let progress = PrebufferProgress {
                    verified_chunks: 0,
                    total_chunks: 0,
                    verified_bytes: total_for_display,
                    total_bytes: total_for_display,
                };
                (cb_cell.borrow_mut())(name, &progress);
            }
        }
        Ok(())
    }

    fn open_facet_records(
        &self,
        name: &str,
    ) -> std::result::Result<crate::records::RecordFacet, crate::records::RecordError> {
        // The mirror of the vector path's guard: an element-shaped
        // facet brought here would fail as a slab parse error about a
        // footer, which says nothing about what went wrong.
        if self.facet_shape(name).ok() == Some(crate::dataset::facet::FacetShape::Elements) {
            return Err(crate::records::RecordError::WrongShape {
                facet: name.to_string(),
                reader: crate::dataset::facet::FacetShape::Elements.reader(),
            });
        }
        let storage = self.open_facet_storage(name).map_err(|e| {
            crate::records::RecordError::Container(format!("open facet '{name}': {e}"))
        })?;
        // A namespace written into the declaration selects a document
        // inside the container — `metadata_content.slab:layout` — and
        // travels with the facet rather than being passed separately.
        let namespace = self
            .facet_manifest()
            .get(name)
            .and_then(|d| d.source_path.as_deref())
            .and_then(|raw| crate::dataset::source::parse_source_string(raw).ok())
            .and_then(|p| p.namespace);
        crate::records::RecordFacet::open(&storage, name, namespace.as_deref())
    }

    /// How this facet's records are addressed, and therefore which
    /// reader opens it.
    ///
    /// Ask this rather than trying a reader and interpreting the
    /// failure. Some facets hold element runs and some hold opaque
    /// records; that is a property of the data, and a caller handling
    /// both should branch on it explicitly.
    ///
    /// ```rust,ignore
    /// match view.facet_shape("metadata_content")? {
    ///     FacetShape::Elements => { let r = view.facet("metadata_content")?; }
    ///     FacetShape::Records  => { let r = view.open_facet_records("metadata_content")?; }
    /// }
    /// ```
    fn facet_shape(&self, name: &str) -> Result<crate::dataset::facet::FacetShape> {
        let desc = self
            .facet_manifest()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        desc.source_type
            .as_deref()
            .and_then(crate::dataset::facet::FacetFormat::from_extension)
            .map(|f| f.shape())
            .ok_or_else(|| {
                Error::Other(format!(
                    "facet '{name}': the spec names no format for '{}'",
                    desc.source_type.as_deref().unwrap_or("(none)")
                ))
            })
    }

    /// Refuse a facet whose shape this reader cannot address, naming
    /// the one that can.
    ///
    /// A facet of the wrong shape is not unreadable — it is readable
    /// elsewhere — so the error says where rather than describing the
    /// symptom the caller happened to hit.
    fn require_facet_shape(
        &self,
        name: &str,
        attempted: crate::dataset::facet::FacetShape,
    ) -> Result<()> {
        let shape = self.facet_shape(name)?;
        if shape == attempted {
            return Ok(());
        }
        Err(Error::WrongFacetShape {
            facet: name.to_string(),
            shape,
            attempted,
            reader: shape.reader(),
        })
    }

    /// Whether this facet holds data this build can move.
    ///
    /// **Not the same question as "what element width does it have?"**,
    /// which is what the gates here used to ask. For every fixed-width
    /// format the two coincide; for a slab they do not — a slab record
    /// is not a run of elements — and taking the element answer skipped
    /// metadata facets entirely: `prebuffer_all` downloaded the vectors
    /// and silently left the metadata behind, and the catalog path of
    /// `derive` dropped it from its output while the local path kept
    /// it.
    ///
    /// The facet spec is the authority on what formats exist
    /// (`FacetFormat`), so it is what gets asked. That keeps this to
    /// exactly the formats the spec names — a parquet or hdf5 facet is
    /// still out of scope here, as it was.
    fn facet_holds_data(&self, name: &str) -> bool {
        self.facet_manifest()
            .get(name)
            .and_then(|d| d.source_type.as_deref())
            .and_then(crate::dataset::facet::FacetFormat::from_extension)
            .is_some()
    }

    /// **Crate-internal hook** used by the default `prebuffer_all`
    /// implementation. Returns a handle whose `precache*` methods
    /// drive the underlying [`crate::storage::Storage`]. Implementors
    /// rarely override this — the `GenericTestDataView` default is
    /// usually correct.
    #[doc(hidden)]
    fn open_facet_storage(&self, name: &str) -> Result<FacetStorage>;

    // -- Prefetch --------------------------------------------------

    /// What prefetching `window` on `facet` would cost, without
    /// fetching any of it.
    ///
    /// `window` is in **record** coordinates and is the caller's to
    /// choose. A profile's `window:` is a convenience — a name for a
    /// range someone wants repeatedly — not a fence around which ranges
    /// may be asked for.
    fn prefetch_plan(&self, facet: &str, window: &DSWindow) -> Result<PrefetchPlan> {
        let storage = self.open_facet_storage(facet)?;
        self.prefetch_plan_on(&storage, facet, window)
    }

    /// The same, against a handle the caller already holds.
    ///
    /// Reusing one handle is what makes the offset-index cache pay: a
    /// plan and the fetch that follows it both need the index, and
    /// opening a handle each would load it twice.
    fn prefetch_plan_on(
        &self,
        storage: &FacetStorage,
        facet: &str,
        window: &DSWindow,
    ) -> Result<PrefetchPlan> {
        let facet_bytes = storage.total_size();
        let mut plan = PrefetchPlan {
            facet_bytes,
            ..PrefetchPlan::default()
        };

        // Where the bytes live, for the record→byte mapping. Absent a
        // source path there is nothing to map against.
        let desc = self.facet_manifest();
        let path = desc
            .get(facet)
            .and_then(|d| d.source_path.as_deref())
            .and_then(|raw| crate::dataset::source::parse_source_string(raw).ok())
            .map(|p| p.path);
        // A series has no single source path — each shard carries its
        // own, and the mapping below reads them from the shard model.
        // Absent both, there is nothing to map against and the request
        // degrades.
        let path = match (path, storage.series_ref()) {
            (Some(p), _) => p,
            (None, Some(_)) => String::new(),
            (None, None) => {
                plan.degrades_to_full_download = true;
                return Ok(plan);
            }
        };

        // No window is a request for the whole facet, and that is not a
        // fallback from anything — it resolves to the whole byte range
        // whatever the format, with no ordinal mapping needed. Routing
        // it through the mapping would make an unmappable format look
        // like it had degraded when the caller asked for everything in
        // the first place.
        if window.is_empty() {
            // The whole facet, shard by shard — one range per shard, so
            // the plan names real files rather than a byte span that
            // exists in none of them (SH-28).
            for (shard, bytes) in storage.shard_sizes().into_iter().enumerate() {
                let r = ShardRange {
                    shard,
                    start: 0,
                    end: bytes,
                };
                plan.requested_ranges.push(r);
                plan.byte_ranges.push(r);
                if let Some(fill) = storage.shard_range_fill(shard, 0, bytes) {
                    plan.fills.push(fill);
                }
            }
            return Ok(plan);
        }
        let intervals: Vec<(u64, u64)> = window
            .0
            .iter()
            .map(|iv| (iv.min_incl, iv.max_excl))
            .collect();

        for (start, end) in intervals {
            match record_range_to_bytes(&path, start, end, storage) {
                Some(m) => {
                    plan.prerequisite_bytes = plan.prerequisite_bytes.max(m.prerequisite_bytes);
                    plan.requested_ranges.extend(m.ranges.iter().copied());
                }
                None => {
                    // One unmappable interval makes the whole request a
                    // full download; reporting a partial plan beside it
                    // would understate what is about to happen.
                    plan.degrades_to_full_download = true;
                    plan.requested_ranges.clear();
                    plan.byte_ranges.clear();
                    plan.fills.clear();
                    return Ok(plan);
                }
            }
        }

        // Merge before planning the fills, so the chunk accounting
        // describes the requests that will actually be issued rather
        // than the intervals that were asked for.
        let chunk_size = storage.cache_stats().map(|c| c.chunk_size);
        plan.byte_ranges = coalesce_ranges(plan.requested_ranges.clone(), chunk_size);
        for r in &plan.byte_ranges {
            if let Some(fill) = storage.shard_range_fill(r.shard, r.start, r.end) {
                plan.fills.push(fill);
            }
        }
        Ok(plan)
    }

    /// Start fetching `window` of `facet` on another thread.
    ///
    /// The plan is computed before this returns — planning needs the
    /// view, and a caller deserves the cost before committing — so the
    /// `Err` here is a planning failure. Fetch failures arrive through
    /// [`PrefetchHandle::join`], and are logged regardless so an
    /// unwatched prefetch cannot fail silently.
    ///
    /// This is the form for warming ahead of a scan: start it, keep
    /// reading, and let the bytes arrive underneath. Reads that
    /// overtake the prefetch are not wrong, only slower — they fault in
    /// the chunk themselves, and the prefetch skips what is already
    /// resident.
    fn prefetch_in_background(
        &self,
        facet: &str,
        window: &DSWindow,
        fallback: WholeFacetFallback,
    ) -> Result<PrefetchHandle> {
        let storage = self.open_facet_storage(facet)?;
        let plan = self.prefetch_plan_on(&storage, facet, window)?;
        check_fallback(facet, &plan, fallback)?;

        let state = std::sync::Arc::new(PrefetchState::default());
        let worker_state = state.clone();
        let ranges = plan.byte_ranges.clone();
        let degrades = plan.degrades_to_full_download;
        let name = facet.to_string();

        let thread = std::thread::Builder::new()
            .name(format!("prefetch:{name}"))
            .spawn(move || {
                use std::sync::atomic::Ordering;
                let record = |e: std::io::Error| {
                    log::warn!("prefetch of '{name}' failed: {e}");
                    *worker_state.error.lock().unwrap() = Some(e.to_string());
                };

                let outcome = if degrades {
                    storage.prebuffer_with_progress(|p| {
                        worker_state
                            .bytes_fetched
                            .store(p.downloaded_bytes(), Ordering::Relaxed);
                    })
                } else {
                    let mut acc = 0u64;
                    let mut result = Ok(());
                    for r in ranges {
                        if worker_state.cancelled.load(Ordering::Acquire) {
                            break;
                        }
                        result = storage.prebuffer_shard_range(r.shard, r.start, r.end, |p| {
                            worker_state
                                .bytes_fetched
                                .store(acc + p.downloaded_bytes(), Ordering::Relaxed);
                        });
                        if result.is_err() {
                            break;
                        }
                        acc = worker_state.bytes_fetched.load(Ordering::Relaxed);
                        worker_state.ranges_fetched.fetch_add(1, Ordering::Relaxed);
                    }
                    result
                };
                if let Err(e) = outcome {
                    record(e);
                }
                // Set last, and with Release, so a caller that sees
                // `is_done()` also sees the error and the counters.
                worker_state.done.store(true, Ordering::Release);
            })
            .map_err(|e| Error::Other(format!("could not start prefetch thread: {e}")))?;

        Ok(PrefetchHandle {
            plan,
            state,
            thread: Some(thread),
        })
    }

    /// Fetch `window` of `facet` and return when it is resident.
    fn prefetch(
        &self,
        facet: &str,
        window: &DSWindow,
        fallback: WholeFacetFallback,
    ) -> Result<PrefetchReport> {
        self.prefetch_with_progress(facet, window, fallback, &mut |_| {})
    }

    /// Same, with chunk-level progress per range.
    fn prefetch_with_progress(
        &self,
        facet: &str,
        window: &DSWindow,
        fallback: WholeFacetFallback,
        cb: &mut dyn FnMut(&crate::transport::DownloadProgress),
    ) -> Result<PrefetchReport> {
        // One handle for both the plan and the fetch, so the offset
        // index a vvec window needs is loaded once rather than twice.
        let storage = self.open_facet_storage(facet)?;
        let planned = self.prefetch_plan_on(&storage, facet, window)?;
        check_fallback(facet, &planned, fallback)?;

        if planned.degrades_to_full_download {
            storage
                .prebuffer_with_progress(|p| cb(p))
                .map_err(|e| Error::Other(e.to_string()))?;
            return Ok(PrefetchReport {
                planned,
                ranges_fetched: 1,
            });
        }

        let mut fetched = 0;
        for r in &planned.byte_ranges {
            storage
                .prebuffer_shard_range(r.shard, r.start, r.end, |p| cb(p))
                .map_err(|e| Error::Other(e.to_string()))?;
            fetched += 1;
        }
        Ok(PrefetchReport {
            planned,
            ranges_fetched: fetched,
        })
    }
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
) -> std::result::Result<crate::typed_access::TypedReader<T>, crate::typed_access::TypedAccessError>
{
    let native = view
        .facet_element_type(facet_name)
        .map_err(|e| crate::typed_access::TypedAccessError::Io(e.to_string()))?;

    // A series has no single source to open; it reads through the shard
    // model instead. Every shard shares the element type and scalar-ness
    // (SRD invariant 4), so both come from the facet as a whole.
    if let Some(source) = view.facet_source(facet_name) {
        return crate::typed_access::TypedReader::<T>::open_auto(&source, native);
    }
    if !view.facet_manifest().contains_key(facet_name) {
        return Err(crate::typed_access::TypedAccessError::Io(format!(
            "facet '{facet_name}' not declared by this view"
        )));
    }
    let storage = view
        .open_facet_storage(facet_name)
        .map_err(|e| crate::typed_access::TypedAccessError::Io(e.to_string()))?;
    let series = storage
        .series_ref()
        .ok_or_else(|| {
            crate::typed_access::TypedAccessError::Io(format!(
                "facet '{facet_name}' has no single source and no series"
            ))
        })?
        .clone();
    let is_scalar =
        series.shards().entries().first().is_some_and(|e| {
            crate::io::is_scalar_ext(e.source.path.rsplit('.').next().unwrap_or(""))
        });
    crate::typed_access::TypedReader::<T>::from_series(series, native, is_scalar)
}

/// Snapshot of in-progress precache state for a single facet. Passed
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
    /// Transfer-chunk size in bytes. Every chunk except the last is
    /// exactly this size, so consumers can map byte offsets to chunk
    /// ordinals (e.g. for chunk-aligned sampling or prefetch).
    pub chunk_size: u64,
    /// Total content size in bytes.
    pub content_size: u64,
    /// Whether every chunk is verified.
    pub is_complete: bool,
}

/// What fetching a byte range would actually cost.
///
/// Returned by [`FacetStorage::range_fill`]. The unit of fetch is a
/// chunk, not a byte, so a range's real cost is rarely the range's
/// length: a 4 KiB window against 8 MiB chunks is 8 MiB, and a window
/// whose chunks are already resident is free. Both facts have to be
/// visible *before* the fetch, or a caller cannot tell an incremental
/// warm-up from a full download wearing one's clothes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeFill {
    /// First chunk covering the range.
    pub first_chunk: u32,
    /// Last chunk covering the range, inclusive.
    pub last_chunk: u32,
    /// Transfer-chunk size in bytes.
    pub chunk_size: u64,
    /// Chunks covering the range.
    pub chunks: u32,
    /// Of those, how many are already resident. These cost nothing.
    pub chunks_resident: u32,
    /// Byte range the fetch actually spans once widened to chunk
    /// boundaries — always a superset of what was asked for.
    pub aligned_start: u64,
    pub aligned_end: u64,
}

impl RangeFill {
    /// Chunks that still have to be fetched.
    pub fn chunks_to_fetch(&self) -> u32 {
        self.chunks.saturating_sub(self.chunks_resident)
    }

    /// Bytes that will cross the network, at chunk granularity.
    pub fn bytes_to_fetch(&self) -> u64 {
        self.chunks_to_fetch() as u64 * self.chunk_size
    }

    /// Bytes fetched beyond the requested range because chunks are the
    /// granularity. This is the number that tells a caller its
    /// scattered single-record prefetches are really a full download.
    pub fn overfetch_bytes(&self, requested_start: u64, requested_end: u64) -> u64 {
        let requested = requested_end.saturating_sub(requested_start);
        let spanned = self.aligned_end.saturating_sub(self.aligned_start);
        spanned.saturating_sub(requested)
    }

    /// Whether every chunk covering the range is already resident.
    pub fn is_resident(&self) -> bool {
        self.chunks_resident >= self.chunks
    }
}

/// Map a record range to a byte range for a variable-length facet,
/// through its offset index.
///
/// Merge byte ranges whose fetches would overlap.
///
/// The unit of fetch is a chunk, so two ranges landing in the same
/// chunk are already one fetch — issuing them separately asks for the
/// same bytes twice. Ranges in *adjacent* chunks are contiguous on the
/// device, so one request covering both beats two. Ranges with a whole
/// chunk between them are not merged: bridging that gap would fetch a
/// chunk nobody asked for.
///
/// With `chunk_size` `None` — local storage, which has no chunks —
/// merging falls back to plain byte overlap or adjacency.
///
/// Pure, and separated from the plan so the boundaries can be tested
/// without a device. Merging one chunk too eagerly silently inflates
/// every scattered prefetch; merging one too shyly silently doubles the
/// request count. Neither fails visibly.
pub(crate) fn coalesce_ranges(
    mut ranges: Vec<ShardRange>,
    chunk_size: Option<u64>,
) -> Vec<ShardRange> {
    ranges.retain(|r| !r.is_empty());
    if ranges.len() < 2 {
        return ranges;
    }
    // Sort by shard first: merging is only ever *within* a shard,
    // because two ranges in different shards are in different files and
    // cannot share a request (SH-28).
    ranges.sort_unstable_by_key(|r| (r.shard, r.start, r.end));

    // Do two ranges belong in one request?
    let joins = |cur_end: u64, next_start: u64| -> bool {
        match chunk_size.filter(|cs| *cs > 0) {
            // Touching or overlapping in chunk space: the chunk holding
            // the current end, and the one holding the next start, are
            // the same or neighbours.
            Some(cs) => next_start / cs <= (cur_end - 1) / cs + 1,
            None => next_start <= cur_end,
        }
    };

    let mut merged: Vec<ShardRange> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match merged.last_mut() {
            Some(cur) if cur.shard == r.shard && joins(cur.end, r.start) => {
                cur.end = cur.end.max(r.end);
            }
            _ => merged.push(r),
        }
    }
    merged
}

/// What a prefetch would fetch, before any of it moves.
///
/// The unit of fetch is a chunk, not a byte, so a window's real cost is
/// rarely its length: a 4 KiB window against 8 MiB chunks is 8 MiB, and
/// a window whose chunks are resident is free. Both have to be visible
/// up front, or a caller cannot tell an incremental warm-up from a full
/// download wearing its clothes.
#[derive(Debug, Clone, Default)]
pub struct PrefetchPlan {
    /// Byte ranges the window resolved to, one per interval, before
    /// merging. What the caller actually asked for.
    ///
    /// **Shard-qualified** (SH-28): a bare `(start, end)` is meaningless
    /// across a series, because byte 500 exists in every shard. Index 0
    /// for a single-file facet.
    pub requested_ranges: Vec<ShardRange>,
    /// Ranges that will be issued, after merging those whose fetches
    /// would overlap. One request each.
    ///
    /// Merging happens **within** a shard: two ranges in different
    /// shards are in different files and cannot share a request.
    pub byte_ranges: Vec<ShardRange>,
    /// Chunk-level cost of each range. Empty when the facet has no
    /// chunks — local storage, which is free by definition.
    pub fills: Vec<RangeFill>,
    /// Bytes that had to be read before the window could be resolved at
    /// all — the offset index, for variable-length formats. Zero for
    /// uniform-stride formats, whose stride comes from a header read
    /// every reader pays anyway.
    ///
    /// Reported whether or not it was paid this time: the index is
    /// cached on the facet handle, so a plan and the fetch that follows
    /// it load it once between them, and repeated asks against a handle
    /// a caller keeps are free. The number is what the mapping *needs*,
    /// which is what makes one large window cheaper than a hundred
    /// small ones across freshly-opened handles.
    pub prerequisite_bytes: u64,
    /// Set when the window could not be resolved and honouring the
    /// request means fetching the whole facet: a format whose
    /// record→byte mapping this layer cannot compute (vvec without its
    /// index; parquet, excluded by design), or storage with no range
    /// capability.
    pub degrades_to_full_download: bool,
    /// Size of the facet, for reading the degrade case against.
    pub facet_bytes: u64,
}

/// A byte range within a named shard.
///
/// The shard index is not decoration: across a series the same byte
/// offset exists in every file, so a range without one names no bytes
/// (SH-28). A single-file facet is shard `0`, which is exactly the old
/// meaning under a new name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardRange {
    /// Which shard of the facet this range is in.
    pub shard: usize,
    /// Start byte within that shard's file.
    pub start: u64,
    /// End byte (exclusive) within that shard's file.
    pub end: u64,
}

impl ShardRange {
    /// A range in the only shard of a single-file facet.
    pub fn whole(start: u64, end: u64) -> Self {
        Self {
            shard: 0,
            start,
            end,
        }
    }

    /// Bytes this range spans.
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Whether the range names no bytes.
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// A range in the only shard compares equal to the plain `(start, end)`
/// pair it replaced.
///
/// This is the migration made explicit rather than papered over: for a
/// single-file facet the shape changed and the meaning did not, so the
/// old spelling still answers. A range in shard 1 compares unequal to
/// any pair, which is correct — a bare pair names no bytes across a
/// series (SH-28).
impl PartialEq<(u64, u64)> for ShardRange {
    fn eq(&self, other: &(u64, u64)) -> bool {
        self.shard == 0 && self.start == other.0 && self.end == other.1
    }
}

impl PrefetchPlan {
    /// Bytes that will cross the network.
    pub fn bytes_to_fetch(&self) -> u64 {
        if self.degrades_to_full_download {
            return self.facet_bytes;
        }
        self.fills.iter().map(|f| f.bytes_to_fetch()).sum()
    }

    /// Chunks that still have to be fetched.
    pub fn chunks_to_fetch(&self) -> u32 {
        self.fills.iter().map(|f| f.chunks_to_fetch()).sum()
    }

    /// Requests that will be issued. Lower than the interval count when
    /// intervals were merged.
    pub fn requests(&self) -> usize {
        self.byte_ranges.len()
    }

    /// Bytes fetched beyond what was asked for.
    ///
    /// Two sources, counted together: chunks are the granularity, so a
    /// 4 KiB window against 8 MiB chunks drags in most of a chunk; and
    /// merging two nearby intervals bridges the gap between them. Both
    /// are bytes crossing the wire that nobody asked for, and a caller
    /// deciding whether its scattered prefetches are really a full
    /// download needs them in one number.
    pub fn overfetch_bytes(&self) -> u64 {
        let spanned: u64 = self
            .fills
            .iter()
            .map(|f| f.aligned_end.saturating_sub(f.aligned_start))
            .sum();
        let asked: u64 = self.requested_ranges.iter().map(|r| r.len()).sum();
        spanned.saturating_sub(asked)
    }

    /// Whether everything the window covers is already resident, so a
    /// prefetch would do nothing.
    pub fn is_resident(&self) -> bool {
        !self.degrades_to_full_download && self.fills.iter().all(|f| f.is_resident())
    }
}

/// Whether a caller will accept the whole facet when the window it
/// asked for cannot be resolved.
///
/// Some formats have no ordinal-to-byte mapping: parquet, whose
/// windowing is excluded by design, and a vvec whose offset index
/// cannot be loaded. Asking for records 5M..6M of one of those and
/// quietly fetching a terabyte is the exact surprise windowed prefetch
/// exists to prevent, so the fallback is **refused unless the caller
/// says otherwise**.
///
/// This only concerns a window that was actually asked for. A prefetch
/// with no window is a request for the whole facet, and fetching it is
/// not a fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WholeFacetFallback {
    /// An unresolvable window is an error. The default.
    #[default]
    Refuse,
    /// Fetch the entire facet rather than failing. The caller has seen
    /// the size — [`PrefetchPlan::facet_bytes`] — and accepted it.
    Allow,
}

/// Refuse a plan that would fetch the whole facet, unless allowed.
///
/// A free function rather than a trait method: a static method on
/// `TestDataView` would make the trait dyn-incompatible, and every
/// caller in this crate holds it as `&dyn TestDataView`.
///
/// The message carries the size, because the decision the caller has to
/// make is whether that size is acceptable.
fn check_fallback(facet: &str, plan: &PrefetchPlan, fallback: WholeFacetFallback) -> Result<()> {
    if plan.degrades_to_full_download && fallback == WholeFacetFallback::Refuse {
        return Err(Error::Other(format!(
            "facet '{facet}': the requested window cannot be resolved for this format, \
             so honouring it means fetching the whole facet ({} bytes). Pass \
             WholeFacetFallback::Allow to accept that.",
            plan.facet_bytes
        )));
    }
    Ok(())
}

/// A prefetch running on another thread.
///
/// The **plan is computed synchronously**, before this is returned, so
/// a caller learns the cost up front and can decide not to proceed.
/// Only the fetching moves off-thread — which is the part that takes
/// time and the part a scan wants to overlap with.
///
/// Dropping the handle **detaches**: the fetch keeps running and its
/// bytes land in the cache, which is what a caller who has moved on
/// still wants. A failure is logged by the worker whether or not
/// anybody joins, so an unwatched prefetch cannot fail silently.
pub struct PrefetchHandle {
    plan: PrefetchPlan,
    state: std::sync::Arc<PrefetchState>,
    thread: Option<std::thread::JoinHandle<()>>,
}

#[derive(Debug, Default)]
struct PrefetchState {
    cancelled: std::sync::atomic::AtomicBool,
    done: std::sync::atomic::AtomicBool,
    bytes_fetched: std::sync::atomic::AtomicU64,
    ranges_fetched: std::sync::atomic::AtomicUsize,
    error: std::sync::Mutex<Option<String>>,
}

impl PrefetchHandle {
    /// What this prefetch set out to do. Available immediately.
    pub fn plan(&self) -> &PrefetchPlan {
        &self.plan
    }

    /// Whether the worker has finished — successfully, in error, or by
    /// cancellation.
    pub fn is_done(&self) -> bool {
        self.state.done.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Bytes fetched so far.
    pub fn bytes_fetched(&self) -> u64 {
        self.state
            .bytes_fetched
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Ranges completed so far, of [`PrefetchPlan::requests`].
    pub fn ranges_fetched(&self) -> usize {
        self.state
            .ranges_fetched
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Ask the worker to stop.
    ///
    /// **Granular to a range**, not to a byte: a fetch already in
    /// flight runs to completion, because the transport has no way to
    /// abandon one part-way and leave the chunk bitmap honest. With one
    /// large range this cancels nothing; with many small ones it stops
    /// promptly. Ranges already fetched stay in the cache — a cancelled
    /// prefetch is partial work, not undone work.
    pub fn cancel(&self) {
        self.state
            .cancelled
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.state
            .cancelled
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Wait for the fetch and report what it did.
    pub fn join(mut self) -> Result<PrefetchReport> {
        if let Some(t) = self.thread.take() {
            // A worker panic is a bug here, not a fetch failure; say so
            // rather than reporting it as an I/O error.
            t.join()
                .map_err(|_| Error::Other("prefetch worker panicked".into()))?;
        }
        if let Some(e) = self.state.error.lock().unwrap().take() {
            return Err(Error::Other(e));
        }
        Ok(PrefetchReport {
            planned: self.plan.clone(),
            ranges_fetched: self.ranges_fetched(),
        })
    }
}

/// What a prefetch actually did.
#[derive(Debug, Clone, Default)]
pub struct PrefetchReport {
    /// The plan as it stood before fetching.
    pub planned: PrefetchPlan,
    /// Byte ranges handed to the transport. Fewer than the plan's when
    /// ranges were merged.
    pub ranges_fetched: usize,
}

/// Opaque handle to a facet's underlying storage. Returned by
/// [`TestDataView::open_facet_storage`] and consumed by the default
/// `prebuffer_all` implementation. There is no public API for
/// reading bytes through this handle — reads go through the
/// shape-aware reader returned by `base_vectors()`, `facet()`, or
/// `open_facet_typed`.
/// How to open one of a series' files, recorded at handle creation so
/// the open itself can be deferred (SH-26).
///
/// Resolution — catalog anchoring, cache relpath, collision rejection —
/// happens once, up front, because it is string work and its errors
/// belong at handle creation. Only the `Storage::open` is lazy.
#[derive(Debug, Clone)]
pub(crate) enum ShardOpen {
    /// Catalog-anchored: cached under `<root>/<dataset>/<relpath>`.
    Layered {
        resolved: String,
        dataset: String,
        relpath: String,
        home: String,
    },
    /// Plain open by resolved path or URL.
    Plain { resolved: String },
}

impl ShardOpen {
    fn open(&self) -> std::io::Result<std::sync::Arc<crate::storage::Storage>> {
        match self {
            Self::Layered {
                resolved,
                dataset,
                relpath,
                home,
            } => crate::storage::Storage::open_layered(resolved, dataset, relpath, home),
            Self::Plain { resolved } => crate::storage::Storage::open(resolved),
        }
    }

    fn resolved(&self) -> &str {
        match self {
            Self::Layered { resolved, .. } | Self::Plain { resolved } => resolved,
        }
    }
}

/// How many of a series' files may be open at once (SH-59).
///
/// **Derived from the process file-descriptor limit, not hardcoded.** A
/// fixed constant is wrong in both directions: it strands descriptors on
/// a generous host and thrashes on a constrained one, and it is
/// invisible to the operator who raised `ulimit -n` precisely to make
/// this work.
///
/// A quarter of the soft limit, leaving the rest for everything else the
/// process holds open — transport sockets, other facets, sidecars. The
/// fraction is an implementation constant; the limit it applies to is
/// not.
pub fn open_file_cap() -> usize {
    resolve_open_file_cap(
        std::env::var("VECTORDATA_SHARD_FD_CAP").ok().as_deref(),
        fd_soft_limit(),
    )
}

/// The cap, given the two things that decide it.
///
/// Split from [`open_file_cap`] so it is answerable without a process
/// to configure: the environment value and the limit arrive as
/// arguments, which is the only way to test the derivation without
/// mutating the environment out from under a parallel test.
///
/// An explicit `VECTORDATA_SHARD_FD_CAP` wins outright — an operator
/// setting it knows something the fraction cannot. A value that is not
/// a positive integer is ignored rather than fatal: a typo in an
/// environment variable should not stop a dataset from opening.
pub(crate) fn resolve_open_file_cap(env: Option<&str>, soft_limit: Option<u64>) -> usize {
    const FRACTION: u64 = 4;
    const FLOOR: usize = 8;
    if let Some(n) = env
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    let soft = soft_limit.unwrap_or(1024);
    ((soft / FRACTION) as usize).max(FLOOR)
}

/// The process's soft `RLIMIT_NOFILE`, if it can be read.
fn fd_soft_limit() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut lim = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: `getrlimit` writes into a fully-initialised struct we
        // own; no aliasing and no lifetime concerns.
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) } == 0 {
            return Some(lim.rlim_cur);
        }
        None
    }
    #[cfg(not(unix))]
    {
        None
    }
}

/// Whether the open-file cap would bind on fetch concurrency (SH-77).
///
/// The cap is a **backstop, never a throttle**: it is expected to sit
/// above whatever the transport wants to run in parallel. When it does
/// not, that is a misconfiguration and must be *reported* — silently
/// reducing parallelism surfaces as unexplained slowness with no line
/// of output pointing at `ulimit`.
pub(crate) fn warn_if_cap_binds_concurrency() {
    let cap = open_file_cap();
    let want = crate::cache::download_concurrency();
    if cap < want {
        log::warn!(
            "open-file cap {cap} (from RLIMIT_NOFILE) is below the download \
             concurrency {want}: shard fetches will be limited by descriptors \
             rather than by the link. Raise `ulimit -n`, or lower \
             VECTORDATA_DOWNLOAD_CONCURRENCY to match."
        );
    }
}

/// One distinct file a series draws from.
///
/// Held behind a `Mutex<Option<..>>` rather than a `OnceLock` because an
/// open file may be **evicted**: a series larger than the descriptor
/// budget must still be readable, and that means the set of open files
/// changes over the handle's life (SH-59).
#[derive(Debug)]
struct SeriesFile {
    spec: ShardOpen,
    opened: std::sync::Mutex<Option<std::sync::Arc<crate::storage::Storage>>>,
}

/// A facet's shards, and the files behind them.
///
/// **Shards and files are counted separately** (SH-81): the ordinal
/// model has one entry per shard, while storage has one entry per
/// *distinct file*. Two shards sliced from one file share a `Storage`,
/// a cache file, a descriptor, and a fetch — so anything byte-shaped
/// here iterates `files`, and anything ordinal-shaped iterates shards.
// Consumed by the reader and prefetch layers, which arrive in the next
// steps; until then the ordinal side of a series has no caller. Remove
// this attribute when they land — it covers a staged landing, not a
// permanently unused surface.
#[derive(Debug)]
pub struct Series {
    shards: crate::dataset::shards::Shards,
    files: Vec<SeriesFile>,
    /// Shard index → index into `files`.
    file_of_shard: Vec<usize>,
    /// Summed file sizes, computed on first ask. Answering it opens
    /// every file, so a caller that never asks never pays.
    total_bytes: std::sync::OnceLock<u64>,
    /// Open files, most-recently-used last. Bounded by
    /// [`open_file_cap`]; the least-recently-used is closed when the
    /// budget is reached (SH-59).
    ///
    /// This list is the authority on what the series holds: a file is
    /// installed in its slot and recorded here under one lock, so the
    /// two can never disagree and the bound holds under concurrent
    /// readers.
    lru: std::sync::Mutex<std::collections::VecDeque<usize>>,
    /// How many files may be open at once.
    cap: usize,
    /// Per-**file** offset indexes for variable-length shards (SH-82).
    ///
    /// One index per file, not per shard: two shards sliced from one
    /// file read that file's index at their own offsets, and load it
    /// once between them.
    offsets: Vec<std::sync::OnceLock<std::sync::Arc<Vec<u64>>>>,
}

impl Series {
    /// Build from the ordinal model and a per-shard open spec,
    /// collapsing shards that name one file onto one entry.
    pub(crate) fn new(shards: crate::dataset::shards::Shards, per_shard: Vec<ShardOpen>) -> Self {
        Self::with_cap(shards, per_shard, open_file_cap())
    }

    /// As [`Self::new`], with the descriptor budget supplied.
    ///
    /// The budget is a property of the process, so `new` reads it from
    /// the process. Naming it here is what lets eviction be exercised
    /// at a size a test can build: with the real budget a test would
    /// need more shards than the host has descriptors, so the branch
    /// that closes a file would never run.
    pub(crate) fn with_cap(
        shards: crate::dataset::shards::Shards,
        per_shard: Vec<ShardOpen>,
        budget: usize,
    ) -> Self {
        let mut files: Vec<SeriesFile> = Vec::new();
        let mut file_of_shard = Vec::with_capacity(per_shard.len());
        for spec in per_shard {
            let existing = files
                .iter()
                .position(|f| f.spec.resolved() == spec.resolved());
            let idx = match existing {
                Some(i) => i,
                None => {
                    files.push(SeriesFile {
                        spec,
                        opened: std::sync::Mutex::new(None),
                    });
                    files.len() - 1
                }
            };
            file_of_shard.push(idx);
        }
        let offsets = (0..files.len())
            .map(|_| std::sync::OnceLock::new())
            .collect();
        // A series never needs more descriptors than it has files, so
        // the cap is the smaller of the budget and the series.
        if files.len() > budget {
            warn_if_cap_binds_concurrency();
        }
        let cap = budget.min(files.len().max(1));
        Self {
            shards,
            files,
            file_of_shard,
            total_bytes: std::sync::OnceLock::new(),
            offsets,
            lru: std::sync::Mutex::new(std::collections::VecDeque::new()),
            cap,
        }
    }

    /// The published offset index for file `i`, loaded once.
    ///
    /// Restricted to what is already cheap, exactly as the single-file
    /// path is: planning must not walk-download a shard to price it
    /// (SH-21).
    pub(crate) fn published_offsets(
        &self,
        i: usize,
        source: &str,
        elem_size: usize,
    ) -> Option<std::sync::Arc<Vec<u64>>> {
        if let Some(o) = self.offsets.get(i)?.get() {
            return Some(o.clone());
        }
        let storage = self.file(i).ok()?;
        let loaded = crate::io::load_offsets(
            source,
            &storage,
            elem_size,
            crate::io::OffsetSource::Published,
        )
        .ok()?;
        let _ = self.offsets[i].set(std::sync::Arc::new(loaded));
        self.offsets.get(i)?.get().cloned()
    }

    /// The ordinal model.
    pub(crate) fn shards(&self) -> &crate::dataset::shards::Shards {
        &self.shards
    }

    /// How many distinct files back this series — never more than the
    /// shard count, and fewer when shards share a file.
    pub(crate) fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Open (or reuse) the storage for file `i`.
    ///
    /// Returns an owned handle rather than a borrow: a file may be
    /// evicted while a caller still holds its storage, and the `Arc`
    /// keeps that read valid until the caller is done (SH-59).
    pub(crate) fn file(&self, i: usize) -> Result<std::sync::Arc<crate::storage::Storage>> {
        let slot = self
            .files
            .get(i)
            .ok_or_else(|| Error::Other(format!("shard file {i} out of range")))?;

        // The recency list is taken before a slot, and that order is
        // never reversed in this type — an eviction holds the list
        // while it closes its victims, so a slot lock is only ever
        // acquired by a thread that already holds the list or holds
        // nothing.
        {
            let mut lru = self.lru.lock().expect("lru");
            if let Some(s) = slot.opened.lock().expect("shard slot").as_ref() {
                let s = s.clone();
                Self::touch_locked(&mut lru, i);
                return Ok(s);
            }
        }

        // Opening happens outside every lock. It may hit the filesystem
        // or the network, and a series is meant to fetch its shards in
        // parallel — holding the recency list across an open would
        // serialise exactly what sharding exists to spread out. Two
        // threads may therefore open the same file at once; one
        // installs and the other's handle is dropped.
        let opened = slot.spec.open().map_err(|e| {
            Error::Other(format!("open shard file '{}': {e}", slot.spec.resolved()))
        })?;

        // Install, record, and evict as **one** critical section. Any
        // of the three alone lets the open set and the recency list
        // disagree, and the descriptor bound is precisely their
        // agreement: two threads opening different shards would each
        // install before either evicted, and the series would hold
        // more files than its budget.
        //
        // The bound is on files this series *holds*. A handle opened
        // above but not yet installed is a descriptor in flight, and
        // those are bounded by fetch concurrency rather than by the
        // cap — the case `warn_if_cap_binds_concurrency` reports.
        let mut lru = self.lru.lock().expect("lru");
        Self::touch_locked(&mut lru, i);
        while lru.len() > self.cap {
            let Some(victim) = lru.pop_front() else { break };
            // The file just opened is the most recently used, so it is
            // never the victim. Asserting that here rather than relying
            // on it keeps a cap of one from closing what it just
            // opened, which would make the series unable to advance.
            if victim == i {
                lru.push_back(victim);
                break;
            }
            if let Some(slot) = self.files.get(victim) {
                *slot.opened.lock().expect("shard slot") = None;
            }
        }
        // Installed **after** the eviction, so the set of held files
        // never passes through a state above the budget. Installing
        // first and trimming after would hold cap+1 descriptors for the
        // width of the eviction loop. Nothing outside can observe that
        // window — every observer goes through the same list — but on a
        // host where the cap is the real ceiling, the descriptor is
        // claimed whether or not anyone is counting.
        let held = {
            let mut slotted = slot.opened.lock().expect("shard slot");
            slotted.get_or_insert(opened).clone()
        };
        Ok(held)
    }

    /// Mark file `i` as most recently used, under a list already held.
    ///
    /// Takes the guard rather than the lock so a caller can record a
    /// use and act on the result without releasing the list in between.
    fn touch_locked(lru: &mut std::collections::VecDeque<usize>, i: usize) {
        if let Some(pos) = lru.iter().position(|&x| x == i) {
            lru.remove(pos);
        }
        lru.push_back(i);
    }

    /// The byte range shard `s` can address in its file.
    ///
    /// The whole file for a whole-file shard; the window's extent for a
    /// sliced one. This is what makes residency mean **addressable**
    /// bytes rather than every byte of every file (SH-92) — a facet
    /// slicing a tenth of a large file would otherwise report incomplete
    /// forever unless the other nine tenths were downloaded.
    pub(crate) fn shard_byte_extent(&self, s: usize) -> Option<(u64, u64)> {
        let entry = self.shards.entries().get(s)?;
        let i = self.file_index_of_shard(s).ok()?;
        let storage = self.file(i).ok()?;
        let (lo, hi) = (entry.file_base, entry.file_base + entry.len);
        let m = map_in_file(&entry.source.path, lo, hi, &storage, &|elem| {
            self.published_offsets(i, &entry.source.path, elem)
        })?;
        Some((m.0, m.1))
    }

    /// The resolved source of file `i` — the string it was opened
    /// from, which is what classification reads.
    pub(crate) fn file_source(&self, i: usize) -> &str {
        self.files[i].spec.resolved()
    }

    /// How many files are open right now.
    ///
    /// Never exceeds the descriptor budget (SH-59) — that bound is what
    /// this exists to make observable.
    ///
    /// Counted under the recency list, not by walking the slots freely.
    /// A free walk answers from a torn view: it can read one slot
    /// before an eviction closes it and a later slot after the open
    /// that displaced it, and report a total that was never held at any
    /// instant. Holding the list makes the answer a real moment,
    /// because every install and every eviction passes through it.
    pub fn open_file_count(&self) -> usize {
        let _lru = self.lru.lock().expect("lru");
        self.files
            .iter()
            .filter(|f| f.opened.lock().expect("shard slot").is_some())
            .count()
    }

    /// Which file backs shard `s`.
    pub(crate) fn file_index_of_shard(&self, s: usize) -> Result<usize> {
        self.file_of_shard
            .get(s)
            .copied()
            .ok_or_else(|| Error::Other(format!("shard {s} out of range")))
    }

    /// Every file, opened. Used by the whole-facet operations —
    /// precache, residency, total size — which are byte-shaped and so
    /// per file (SH-81).
    fn all_files(&self) -> Result<Vec<std::sync::Arc<crate::storage::Storage>>> {
        (0..self.files.len()).map(|i| self.file(i)).collect()
    }

    /// Summed size of the distinct files, counting a shared file once.
    ///
    /// Fallible on purpose. A declared shard that will not open is a
    /// broken facet, and the infallible answer for one is `0` — which is
    /// indistinguishable from an empty facet and is exactly the silent
    /// shape this design forbids.
    fn try_total_size(&self) -> Result<u64> {
        if let Some(n) = self.total_bytes.get() {
            return Ok(*n);
        }
        let n: u64 = self.all_files()?.iter().map(|s| s.total_size()).sum();
        let _ = self.total_bytes.set(n);
        Ok(n)
    }
}

pub struct FacetStorage {
    storage: std::sync::Arc<crate::storage::Storage>,
    /// The remaining shards, when this facet is a series.
    ///
    /// `None` for a single file — which is every facet written before
    /// multi-file support, and the case that must add no indirection
    /// (SH-73). Every method below branches once on this and then runs
    /// exactly the code it ran before.
    series: Option<std::sync::Arc<Series>>,
    /// Record offsets for variable-length facets, loaded once per
    /// handle.
    ///
    /// Deliberately per-handle rather than process-wide. The index is
    /// eight bytes per record — 8 GB at a billion — so a global cache
    /// would be an unbounded one, and the point at which it should be
    /// dropped is a question only the caller can answer. Holding a
    /// handle is how a caller says "I am going to ask about this facet
    /// repeatedly"; dropping it is how they say they are done.
    offsets: std::sync::OnceLock<CachedOffsets>,
}

/// A loaded offset index, with the element width it was loaded for.
///
/// The width is carried so a handle reused across element types cannot
/// silently answer with offsets computed for a different stride — that
/// would resolve a window to plausible, wrong bytes.
#[derive(Debug)]
struct CachedOffsets {
    elem_size: usize,
    offsets: std::sync::Arc<Vec<u64>>,
}

impl FacetStorage {
    pub(crate) fn new(storage: std::sync::Arc<crate::storage::Storage>) -> Self {
        Self {
            storage,
            series: None,
            offsets: std::sync::OnceLock::new(),
        }
    }

    /// A handle over a multi-file series.
    ///
    /// `storage` is the first shard's file, so every single-file path
    /// below keeps working against something real rather than a
    /// placeholder — but any *whole-facet* answer comes from `series`.
    pub(crate) fn series(first: std::sync::Arc<crate::storage::Storage>, series: Series) -> Self {
        Self {
            storage: first,
            series: Some(std::sync::Arc::new(series)),
            offsets: std::sync::OnceLock::new(),
        }
    }

    /// The series behind this facet, if it has one.
    pub fn series_ref(&self) -> Option<&std::sync::Arc<Series>> {
        self.series.as_ref()
    }

    /// The record-offset index for a variable-length facet, loaded on
    /// first use and reused for the life of this handle.
    ///
    /// Restricted to [`crate::io::OffsetSource::Published`] — a
    /// published sidecar, a persisted rebuild, or a local mmap walk.
    /// This handle exists to answer *planning* questions ("what would
    /// this window cost?"), and the remaining way to build an index for
    /// a remote vvec is to walk the file, which downloads all of it. A
    /// plan that transfers the facet in order to report the cost of
    /// transferring part of it is not a plan. Readers, which will read
    /// the data regardless, load their own offsets and may rebuild.
    ///
    /// `None` when no such index is available, or when this handle
    /// already holds offsets for a different element width — see
    /// [`CachedOffsets`].
    pub(crate) fn published_offsets(
        &self,
        source: &str,
        elem_size: usize,
    ) -> Option<std::sync::Arc<Vec<u64>>> {
        if let Some(cached) = self.offsets.get() {
            return (cached.elem_size == elem_size).then(|| cached.offsets.clone());
        }
        let loaded = crate::io::load_offsets(
            source,
            &self.storage,
            elem_size,
            crate::io::OffsetSource::Published,
        )
        .ok()?;
        // A race here means two loads and one winner, which is wasteful
        // but never wrong — both produce the same offsets. Return what
        // is *cached* rather than what this call built, so every caller
        // observes the same allocation.
        let _ = self.offsets.set(CachedOffsets {
            elem_size,
            offsets: std::sync::Arc::new(loaded),
        });
        let winner = self.offsets.get()?;
        (winner.elem_size == elem_size).then(|| winner.offsets.clone())
    }

    /// Read the first `len` bytes of the facet (clamped to its size).
    /// Crate-internal shape probe: pulls the covering chunk on remote
    /// storage, which is exactly what `datasets ping` wants — after a
    /// ping, the cache survey can derive record counts from the now-
    /// resident header.
    pub(crate) fn read_prefix(&self, len: u64) -> std::io::Result<Vec<u8>> {
        let len = len.min(self.total_size());
        self.storage.read_bytes(0, len)
    }
    /// Drive this facet to fully-resident, zero-copy state.
    /// **Strict**: returns `Err` on any failure — never silently
    /// leaves the facet in a partially-resident state. After
    /// `Ok(())`, every read on every reader against this source
    /// (this `FacetStorage` and any other) takes the zero-copy
    /// path on next access.
    pub fn precache(&self) -> std::io::Result<()> {
        match &self.series {
            None => self.storage.precache(),
            // Drives exactly what the facet can address (SH-92). A
            // whole-file shard asks for the whole file, so the common
            // case is unchanged; a sliced one asks only for its window
            // rather than dragging bytes it can never read.
            Some(s) => {
                for shard in 0..s.shards().entries().len() {
                    match s.shard_byte_extent(shard) {
                        Some((lo, hi)) => {
                            self.prebuffer_shard_range(shard, lo, hi, |_| {})?;
                        }
                        None => {
                            let i = s
                                .file_index_of_shard(shard)
                                .map_err(|e| std::io::Error::other(e.to_string()))?;
                            s.file(i)
                                .map_err(|e| std::io::Error::other(e.to_string()))?
                                .precache()?;
                        }
                    }
                }
                Ok(())
            }
        }
    }
    /// Fetch everything this facet can address, reporting progress.
    ///
    /// **Walks the series**, exactly as [`Self::precache`] does.
    /// `self.storage` is the first shard's, so delegating to it would
    /// fetch one file and return — leaving a facet that reports itself
    /// precached while most of it is still a hole, which is the worst
    /// of the three possible outcomes.
    ///
    /// Progress arrives **per file**: each shard's fetch reports its
    /// own totals, so a meter driven by this advances once per shard
    /// rather than once across the series. That is a display question;
    /// the alternative — synthesising a combined `DownloadProgress`
    /// the transport never produced — would misreport what is actually
    /// in flight.
    pub fn prebuffer_with_progress<F>(&self, mut cb: F) -> std::io::Result<()>
    where
        F: FnMut(&crate::transport::DownloadProgress),
    {
        let Some(s) = &self.series else {
            return self.storage.prebuffer_with_progress(cb);
        };
        for shard in 0..s.shards().entries().len() {
            // Addressable bytes, not whole files (SH-92): a sliced
            // shard pulls its window, not bytes it can never read.
            match s.shard_byte_extent(shard) {
                Some((lo, hi)) => self.prebuffer_shard_range(shard, lo, hi, &mut cb)?,
                None => {
                    let i = s
                        .file_index_of_shard(shard)
                        .map_err(|e| std::io::Error::other(e.to_string()))?;
                    s.file(i)
                        .map_err(|e| std::io::Error::other(e.to_string()))?
                        .prebuffer_with_progress(&mut cb)?;
                }
            }
        }
        Ok(())
    }

    /// Same as [`Self::prebuffer_with_progress`] but only fetches
    /// chunks covering `[byte_start, byte_end)`. Used by windowed
    /// profile precache so a `:1m` window against a 1.3 TiB base
    /// only pulls the chunks for the window's byte range. The
    /// view layer computes the byte range from the window record
    /// count and the format's bytes-per-record.
    ///
    /// **Walks the series.** The facet's byte space is its shards' files
    /// laid end to end (SH-38), and `self.storage` is only the first of
    /// them: delegating to it would fetch a range in shard 0 and
    /// silently do nothing for a range that lives in any later shard.
    /// The explorer's prefetcher found that out on tessera, where every
    /// chunk past the first shard fell back to the reader's serial
    /// on-demand fetch.
    pub fn prebuffer_range_with_progress<F>(
        &self,
        byte_start: u64,
        byte_end: u64,
        mut cb: F,
    ) -> std::io::Result<()>
    where
        F: FnMut(&crate::transport::DownloadProgress),
    {
        let Some(s) = &self.series else {
            return self
                .storage
                .prebuffer_range_with_progress(byte_start, byte_end, cb);
        };
        let mut file_start = 0u64;
        for i in 0..s.file_count() {
            if file_start >= byte_end {
                break;
            }
            let storage = s
                .file(i)
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            let file_end = file_start + storage.total_size();
            if file_end > byte_start && file_start < byte_end {
                let from = byte_start.saturating_sub(file_start);
                let to = byte_end.min(file_end) - file_start;
                storage.prebuffer_range_with_progress(from, to, &mut cb)?;
            }
            file_start = file_end;
        }
        Ok(())
    }

    /// [`Self::prebuffer_range_with_progress`] against a named shard's
    /// file — the byte range is in that file's own coordinates.
    pub(crate) fn prebuffer_shard_range<F>(
        &self,
        shard: usize,
        byte_start: u64,
        byte_end: u64,
        cb: F,
    ) -> std::io::Result<()>
    where
        F: FnMut(&crate::transport::DownloadProgress),
    {
        let storage = match &self.series {
            None => self.storage.clone(),
            Some(s) => {
                let i = s
                    .file_index_of_shard(shard)
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                s.file(i)
                    .map_err(|e| std::io::Error::other(e.to_string()))?
            }
        };
        storage.prebuffer_range_with_progress(byte_start, byte_end, cb)
    }
    /// Whether every byte this facet can address is resident.
    ///
    /// For whole-file shards — every form but a sliced series — that is
    /// the conjunction over the files it draws from. A *sliced* facet
    /// should answer over its addressable range only (SH-92); that
    /// needs byte ranges, which arrive with the reader layer, and no
    /// writer produces a sliced series yet.
    pub fn is_complete(&self) -> bool {
        match &self.series {
            None => self.storage.is_complete(),
            Some(s) => (0..s.shards().entries().len()).all(|shard| {
                // Addressable bytes, not whole files (SH-92). For a
                // whole-file shard the two coincide, so the common case
                // is unchanged; where they diverge, the addressable
                // range is what "complete" has to mean, or a sliced
                // facet can never reach it.
                match s.shard_byte_extent(shard) {
                    Some((lo, hi)) => self
                        .shard_range_fill(shard, lo, hi)
                        .is_none_or(|f| f.is_resident()),
                    // No mapping to work from: fall back to the file,
                    // which is the conservative answer.
                    None => s
                        .file_index_of_shard(shard)
                        .and_then(|i| s.file(i))
                        .is_ok_and(|f| f.is_complete()),
                }
            }),
        }
    }
    /// The storage behind this facet.
    ///
    /// A reader that parses a container of its own reads through this
    /// rather than mapping a file, which is what keeps such a facet
    /// incremental: a chunked, merkle-verified source fetches and
    /// verifies the chunks covering each range as it is asked for.
    pub(crate) fn storage_handle(&self) -> std::sync::Arc<crate::storage::Storage> {
        self.storage.clone()
    }

    /// The access mode this facet promises, which is the weakest among
    /// its files (SH-93).
    ///
    /// A facet's mode is a promise about every read it will serve. A
    /// series whose shards were published differently — one with a
    /// `.mref`, one without, one on a server with no range support —
    /// must answer for the worst of them, or a caller plans against an
    /// average it will not get. Understating a good shard costs
    /// efficiency; overstating a bad one costs correctness.
    ///
    /// Per-file detail stays available to `describe` and the explore
    /// TUI, where it is information rather than a promise (SH-46).
    /// `source` is the facet's declared source, which is all a
    /// single-file facet needs. A series ignores it — it is the `NNNN`
    /// pattern, which names no file — and classifies each shard from
    /// the source it was actually opened from.
    pub fn access_mode(
        &self,
        source: &str,
        cache_dir: &std::path::Path,
    ) -> crate::access::AccessMode {
        let Some(s) = &self.series else {
            return crate::access::AccessMode::classify(source, cache_dir);
        };
        crate::access::AccessMode::weakest(
            (0..s.file_count())
                .map(|i| crate::access::AccessMode::classify(s.file_source(i), cache_dir)),
        )
        .unwrap_or(crate::access::AccessMode::Local)
    }

    pub fn is_local(&self) -> bool {
        match &self.series {
            None => self.storage.is_local(),
            Some(s) => (0..s.file_count()).all(|i| s.file(i).is_ok_and(|f| f.is_local())),
        }
    }
    /// Bytes this facet occupies, summed over the distinct files it
    /// draws from (a shared file counted once).
    ///
    /// Returns `0` for a series with an unopenable shard, having logged
    /// why. That is the only answer an infallible accessor can give, and
    /// it is why [`Self::try_total_size`] exists: every caller that can
    /// report a failure should use that one instead.
    pub fn total_size(&self) -> u64 {
        match self.try_total_size() {
            Ok(n) => n,
            Err(e) => {
                log::error!("facet size unavailable: {e}");
                0
            }
        }
    }

    /// [`Self::total_size`], propagating the reason a series cannot be
    /// sized — a declared shard that is absent or will not open.
    pub fn try_total_size(&self) -> Result<u64> {
        match &self.series {
            None => Ok(self.storage.total_size()),
            Some(s) => s.try_total_size(),
        }
    }

    /// Records this facet holds, when its layout states them.
    ///
    /// `None` for a single file, whose count is the reader's to derive
    /// from the bytes; `Some` for a series, whose declaration states it.
    #[allow(dead_code)]
    pub(crate) fn record_count(&self) -> Option<u64> {
        self.series.as_ref().map(|s| s.shards().count())
    }

    /// Path to the local file that backs this facet, when one
    /// exists. `Some` for `Storage::Cached` (the cache file under
    /// the configured cache root) and for purely-local datasets
    /// where no cache is involved. `None` for direct-HTTP storage
    /// (no `.mref` published — there is no local file).
    ///
    /// Hot-path consumers that need a `&Path` to mmap should call
    /// `precache()` first to ensure the file is fully resident,
    /// then use this path. Consumers that just want to read should
    /// prefer `view.facet(name)` / `view.base_vectors()` and let
    /// the reader handle resident-state for them.
    pub fn cache_path(&self) -> Option<std::path::PathBuf> {
        self.storage.local_path()
    }

    /// Bytes this facet's backing cache file *actually* occupies on
    /// disk right now (`du` semantics — allocated blocks, not the
    /// apparent length the file was sparse-pre-sized to). `0` when no
    /// local file backs the facet yet (nothing downloaded) or its
    /// metadata can't be read. Used by the precache capacity check to
    /// discount already-resident bytes from what a download still
    /// needs to fetch.
    pub(crate) fn allocated_cache_bytes(&self) -> u64 {
        let of = |s: &crate::storage::Storage| -> u64 {
            s.local_path()
                .and_then(|p| std::fs::metadata(&p).ok())
                .map(|m| crate::cache::reader::allocated_size(&m))
                .unwrap_or(0)
        };
        match &self.series {
            None => of(&self.storage),
            Some(s) => (0..s.file_count())
                .filter_map(|i| s.file(i).ok())
                .map(|f| of(&f))
                .sum(),
        }
    }

    /// Live cache-fill statistics. Reports chunk fill for both the
    /// merkle-verified (`.mref`) and merkle-less chunked-HTTP remote
    /// paths, so progress UIs work no matter which transport the
    /// open resolved to. Returns `None` only for local mmap storage,
    /// which is always fully resident.
    /// What fetching `[byte_start, byte_end)` would cost, without
    /// fetching any of it.
    ///
    /// `None` for storage with no chunks — a local mmap is resident by
    /// definition and a fully-downloaded cache file has nothing left to
    /// plan. Callers should treat `None` as "free".
    /// Byte size of each shard, in shard order.
    ///
    /// One entry for a single-file facet — the facet's own size — so a
    /// caller never has to ask which case it is in.
    pub(crate) fn shard_sizes(&self) -> Vec<u64> {
        match &self.series {
            None => vec![self.storage.total_size()],
            Some(s) => (0..s.shards().entries().len())
                .map(|shard| {
                    s.file_index_of_shard(shard)
                        .and_then(|i| s.file(i).map(|f| f.total_size()))
                        .unwrap_or(0)
                })
                .collect(),
        }
    }

    /// [`Self::range_fill`] against a named shard's file.
    pub(crate) fn shard_range_fill(
        &self,
        shard: usize,
        byte_start: u64,
        byte_end: u64,
    ) -> Option<RangeFill> {
        let storage = match &self.series {
            None => self.storage.clone(),
            Some(s) => {
                let i = s.file_index_of_shard(shard).ok()?;
                s.file(i).ok()?
            }
        };
        let (first, last, chunk_size, resident) = storage.range_fill(byte_start, byte_end)?;
        let aligned_start = first as u64 * chunk_size;
        let aligned_end = ((last as u64 + 1) * chunk_size).min(storage.total_size());
        Some(RangeFill {
            first_chunk: first,
            last_chunk: last,
            chunk_size,
            chunks: last - first + 1,
            chunks_resident: resident,
            aligned_start,
            aligned_end,
        })
    }

    pub fn range_fill(&self, byte_start: u64, byte_end: u64) -> Option<RangeFill> {
        let (first, last, chunk_size, resident) = self.storage.range_fill(byte_start, byte_end)?;
        let aligned_start = first as u64 * chunk_size;
        let aligned_end = ((last as u64 + 1) * chunk_size).min(self.total_size());
        Some(RangeFill {
            first_chunk: first,
            last_chunk: last,
            chunk_size,
            chunks: last - first + 1,
            chunks_resident: resident,
            aligned_start,
            aligned_end,
        })
    }

    /// Cache fill for the whole facet.
    ///
    /// **Summed across a series' files** (SH-81). `self.storage` is the
    /// first shard's, so reporting its numbers would say "37% cached"
    /// about a five-file facet on the strength of one file — a
    /// percentage that is right about nothing. Chunk counts and content
    /// bytes add; completeness is the conjunction, since a facet with
    /// one missing shard is not cached.
    ///
    /// `None` when no file is cache-backed (local mmap, direct HTTP).
    /// A series where only some files are cache-backed reports the ones
    /// that are, which is the same partial answer a single file gives
    /// while filling.
    pub fn cache_stats(&self) -> Option<CacheStats> {
        let Some(series) = &self.series else {
            return self.storage.fill_stats().map(
                |(valid_chunks, total_chunks, chunk_size, content_size, is_complete)| CacheStats {
                    valid_chunks,
                    total_chunks,
                    chunk_size,
                    content_size,
                    is_complete,
                },
            );
        };
        let mut acc: Option<CacheStats> = None;
        for i in 0..series.file_count() {
            let Ok(storage) = series.file(i) else { continue };
            let Some((valid, total, chunk_size, content, complete)) = storage.fill_stats() else {
                continue;
            };
            acc = Some(match acc {
                None => CacheStats {
                    valid_chunks: valid,
                    total_chunks: total,
                    chunk_size,
                    content_size: content,
                    is_complete: complete,
                },
                Some(a) => CacheStats {
                    valid_chunks: a.valid_chunks + valid,
                    total_chunks: a.total_chunks + total,
                    // One chunk size across the series; files agreeing
                    // is the normal case, and the larger is the honest
                    // answer if they ever do not.
                    chunk_size: a.chunk_size.max(chunk_size),
                    content_size: a.content_size + content,
                    is_complete: a.is_complete && complete,
                },
            });
        }
        acc
    }
}

/// A uniform-stride reader over a facet spread across several files.
///
/// Presents the **same surface** as a single-file reader (SH-22): a
/// caller that never asks about layout never learns there is one.
/// `count()` is the series total, `dim()` the dimension every shard
/// shares, and `get(o)` resolves through the ordinal model to the file
/// that holds `o`.
///
/// One reader per **file**, not per shard (SH-81) — two shards sliced
/// from one file read through one reader at their own offsets.
pub(crate) struct ShardedXvecReader<T> {
    series: std::sync::Arc<Series>,
    /// Per-file readers, opened on first touch (SH-26).
    readers: Vec<std::sync::OnceLock<crate::io::XvecReader<T>>>,
    dim: usize,
    count: usize,
}

impl<T> std::fmt::Debug for ShardedXvecReader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedXvecReader")
            .field("shards", &self.series.shards().entries().len())
            .field("files", &self.series.file_count())
            .field("dim", &self.dim)
            .field("count", &self.count)
            .finish()
    }
}

impl<T: VvecElement> ShardedXvecReader<T> {
    /// Open a series, verifying that every file agrees on the shape.
    ///
    /// The dimension check is eager and total: a series whose shards
    /// disagree resolves ordinals to records of the wrong width, and
    /// there is no later moment at which that becomes visible
    /// (SH-15). Paying one header read per file at open is what makes
    /// every read after it trustworthy.
    fn open(series: std::sync::Arc<Series>) -> Result<Self> {
        let n = series.file_count();
        let readers: Vec<std::sync::OnceLock<crate::io::XvecReader<T>>> =
            (0..n).map(|_| std::sync::OnceLock::new()).collect();
        let me = Self {
            series,
            readers,
            dim: 0,
            count: 0,
        };
        let first = me.reader(0)?;
        let dim = first.dim();
        for i in 1..n {
            let d = me.reader(i)?.dim();
            if d != dim {
                return Err(Error::Other(format!(
                    "shard file {i} has dim {d}, but the series is dim {dim} —                      every shard of a facet must share one shape"
                )));
            }
        }
        Ok(Self {
            count: me.series.shards().count() as usize,
            dim,
            ..me
        })
    }

    /// The reader for file `i`, opened on first use.
    fn reader(&self, i: usize) -> Result<&crate::io::XvecReader<T>> {
        let slot = self
            .readers
            .get(i)
            .ok_or_else(|| Error::Other(format!("shard file {i} out of range")))?;
        if let Some(r) = slot.get() {
            return Ok(r);
        }
        let storage = self.series.file(i)?.clone();
        let opened = crate::io::XvecReader::<T>::from_storage(storage)
            .map_err(|e| Error::Other(format!("open shard file {i}: {e}")))?;
        let _ = slot.set(opened);
        slot.get()
            .ok_or_else(|| Error::Other("shard reader vanished after set".into()))
    }

    /// Resolve a global ordinal to the reader that holds it and the
    /// ordinal within that reader's file (SH-64).
    fn locate(&self, index: usize) -> Result<(&crate::io::XvecReader<T>, usize)> {
        let at = self
            .series
            .shards()
            .locate(index as u64)
            .ok_or_else(|| Error::Other(format!("ordinal {index} out of range")))?;
        let file = self.series.file_index_of_shard(at.shard)?;
        Ok((self.reader(file)?, at.file_ordinal as usize))
    }
}

impl<T: VvecElement> VectorReader<T> for ShardedXvecReader<T> {
    fn dim(&self) -> usize {
        self.dim
    }
    fn count(&self) -> usize {
        self.count
    }

    fn get(&self, index: usize) -> std::result::Result<Vec<T>, crate::io::IoError> {
        let (reader, at) = self
            .locate(index)
            .map_err(|_| crate::io::IoError::OutOfBounds(index))?;
        crate::io::VectorReader::get(reader, at)
    }

    /// A borrowed slice, when the record lies in a resident,
    /// mmap-promoted file.
    ///
    /// Never synthesized across files (SH-25): a record spans no shard
    /// boundary and a shard spans no file, so the delegation is exact —
    /// but a caller asking for a slice of a file that is not mapped gets
    /// an honest `None` rather than a copy pretending to be a borrow.
    fn get_slice(&self, index: usize) -> Option<&[T]> {
        let (reader, at) = self.locate(index).ok()?;
        reader.get_slice(at)
    }

    fn precache(&self) -> std::io::Result<()> {
        for i in 0..self.series.file_count() {
            self.series
                .file(i)
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .precache()?;
        }
        Ok(())
    }

    fn is_complete(&self) -> bool {
        (0..self.series.file_count()).all(|i| self.series.file(i).is_ok_and(|f| f.is_complete()))
    }
}

/// A variable-length reader over a facet spread across several files.
///
/// Each **file** carries its own `IDXFOR__` sidecar, whose offsets are
/// local to that file (SH-17, SH-82). A sliced shard reads the
/// sub-range its window names; no index is rebuilt or re-based to
/// support slicing, and two shards drawn from one file load its index
/// once between them.
pub(crate) struct ShardedVvecReader<T> {
    series: std::sync::Arc<Series>,
    readers: Vec<std::sync::OnceLock<crate::io::IndexedVvecReader<T>>>,
    /// The source string per file, for the sidecar lookup.
    sources: Vec<String>,
    elem_size: usize,
    count: usize,
}

impl<T> std::fmt::Debug for ShardedVvecReader<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedVvecReader")
            .field("files", &self.series.file_count())
            .field("count", &self.count)
            .finish()
    }
}

impl<T: VvecElement> ShardedVvecReader<T> {
    fn open(series: std::sync::Arc<Series>, elem_size: usize) -> Result<Self> {
        // One source per distinct file, in file order — the sidecar is
        // found beside the file, so the shard's own path is what names
        // it.
        let mut sources = vec![String::new(); series.file_count()];
        for (shard, entry) in series.shards().entries().iter().enumerate() {
            let i = series.file_index_of_shard(shard)?;
            if sources[i].is_empty() {
                sources[i] = entry.source.path.clone();
            }
        }
        let readers = (0..series.file_count())
            .map(|_| std::sync::OnceLock::new())
            .collect();
        let count = series.shards().count() as usize;
        Ok(Self {
            series,
            readers,
            sources,
            elem_size,
            count,
        })
    }

    fn reader(&self, i: usize) -> Result<&crate::io::IndexedVvecReader<T>> {
        let slot = self
            .readers
            .get(i)
            .ok_or_else(|| Error::Other(format!("shard file {i} out of range")))?;
        if let Some(r) = slot.get() {
            return Ok(r);
        }
        let storage = self.series.file(i)?.clone();
        let opened = crate::io::IndexedVvecReader::<T>::from_storage(
            storage,
            &self.sources[i],
            self.elem_size,
        )
        .map_err(|e| Error::Other(format!("open shard file {i}: {e}")))?;
        let _ = slot.set(opened);
        slot.get()
            .ok_or_else(|| Error::Other("shard reader vanished after set".into()))
    }

    fn locate(&self, index: usize) -> Result<(&crate::io::IndexedVvecReader<T>, usize)> {
        let at = self
            .series
            .shards()
            .locate(index as u64)
            .ok_or_else(|| Error::Other(format!("ordinal {index} out of range")))?;
        let file = self.series.file_index_of_shard(at.shard)?;
        Ok((self.reader(file)?, at.file_ordinal as usize))
    }
}

impl<T: VvecElement> VvecReader<T> for ShardedVvecReader<T> {
    fn count(&self) -> usize {
        self.count
    }

    fn dim_at(&self, index: usize) -> std::result::Result<usize, crate::io::IoError> {
        let (r, at) = self
            .locate(index)
            .map_err(|_| crate::io::IoError::OutOfBounds(index))?;
        r.dim_at(at)
    }

    fn get_bytes(&self, index: usize) -> std::result::Result<Vec<u8>, crate::io::IoError> {
        let (r, at) = self
            .locate(index)
            .map_err(|_| crate::io::IoError::OutOfBounds(index))?;
        r.get_bytes(at)
    }

    fn precache(&self) -> std::io::Result<()> {
        for i in 0..self.series.file_count() {
            self.series
                .file(i)
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .precache()?;
        }
        Ok(())
    }

    fn is_complete(&self) -> bool {
        (0..self.series.file_count()).all(|i| self.series.file(i).is_ok_and(|f| f.is_complete()))
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
    /// Dataset name (matches the `name:` field in `dataset.yaml`,
    /// or the catalog entry name for knn_entries-shape catalogs).
    /// When set, [`Self::open_facet_storage`] routes every facet
    /// through [`crate::storage::Storage::open_layered`] so cached
    /// bytes land at `<cache_root>/<dataset_name>/<facet_relpath>`
    /// — stable across catalog moves. `None` falls back to URL-
    /// derived layout (sufficient for direct URL opens that have
    /// no catalog context to anchor on).
    dataset_name: Option<String>,
    /// Catalog source URL recorded in the per-dataset `origin.json`.
    /// Verified on subsequent opens; editable by the user when the
    /// catalog moves but the bytes are the same.
    catalog_source: Option<String>,
}

impl GenericTestDataView {
    /// Creates a new `GenericTestDataView`.
    pub fn new(source: DataSource, config: ProfileConfig) -> Self {
        Self {
            source,
            config,
            attributes: HashMap::new(),
            dataset_name: None,
            catalog_source: None,
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
            dataset_name: None,
            catalog_source: None,
        }
    }

    /// Set the catalog-anchored cache identity. Both must be set
    /// together — `dataset_name` is the per-dataset cache
    /// directory name, `catalog_source` is the URL recorded in
    /// `origin.json` (typically the dataset.yaml URL, or the
    /// knn_entries.yaml URL for that catalog shape).
    pub fn with_catalog_identity(
        mut self,
        dataset_name: impl Into<String>,
        catalog_source: impl Into<String>,
    ) -> Self {
        self.dataset_name = Some(dataset_name.into());
        self.catalog_source = Some(catalog_source.into());
        self
    }

    fn resolve_resource(&self, facet: &FacetConfig) -> Result<ResourceLocation> {
        let source_str = single_source(facet, "resolve")?;
        // Absolute URLs in the YAML override the dataset's base
        // location — supports the case where a local dataset.yaml
        // references remote facets, or a remote catalog references
        // facets in a different bucket.
        if is_absolute_url(source_str) {
            return Ok(ResourceLocation::Http(Url::parse(source_str)?));
        }
        // Local-file short-circuit: `file://` URI or absolute path.
        // Bypass the catalog's base-URL join — the catalog entry
        // is naming a file that already exists locally, and the
        // most efficient open is a direct mmap. Same semantic as a
        // fully-precached remote facet.
        if is_local_facet_source(source_str) {
            return Ok(ResourceLocation::FileSystem(PathBuf::from(
                file_uri_to_path(source_str),
            )));
        }
        // Any other absolute URI scheme (e.g. `s3://`) — pass
        // through verbatim. We can't open it (no transport here),
        // but the storage layer will surface a precise error
        // rather than the bogus double-prefixed path we'd get from
        // joining it against the catalog's base URL.
        if has_absolute_uri_scheme(source_str) {
            return Ok(ResourceLocation::FileSystem(PathBuf::from(source_str)));
        }
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
        // `VectorReader` promises `dim()` and a run of `T` per record.
        // A record-shaped facet has neither, so it is refused here with
        // the reader that does open it — rather than failing further in
        // with "cannot infer element size", which describes the symptom
        // and points nowhere.
        self.require_facet_shape(name, crate::dataset::facet::FacetShape::Elements)?;

        // A series reads through the shard model. The profile window
        // still applies on top, in *facet* ordinals (SH-67) — an entry
        // window inside the series is in file ordinals and has already
        // been folded into the shards.
        if facet.is_series() {
            let storage = self.open_facet_storage(name)?;
            let series = storage
                .series_ref()
                .ok_or_else(|| Error::Other(format!("facet '{name}': series not realized")))?
                .clone();
            let reader: Box<dyn VectorReader<T>> = Box::new(ShardedXvecReader::<T>::open(series)?);
            return Ok(match facet.window().and_then(parse_window_first) {
                Some((start, end)) => Arc::new(WindowedVectorReader::new(reader, start, end)),
                None => Arc::from(reader),
            });
        }

        // Two ways the dataset config can carry a sub-ordinal range:
        //   1. `Detailed { source, window }` — explicit `window:` field.
        //   2. `Simple("path[0..N)")` — the documented suffix sugar
        //      that `parse_source_string` understands.
        // Try the suffix form first (so `Simple` strings work without
        // forcing the consumer to use `Detailed`); fall back to the
        // explicit field. Either path produces the same windowed
        // reader behind the trait.
        let raw = single_source(facet, "open")?;
        // A source string only fails to parse when it *looks* like it
        // carries a `[..]` window suffix and that suffix is malformed —
        // a plain path always parses with an empty window. So an error
        // here is a broken window, not a broken path, and treating the
        // whole string as a filename turns "your window is malformed"
        // into "no such file: base.fvec[0,1000)". Surface the real
        // reason instead.
        let (path_str, window_from_suffix): (String, Option<(usize, usize)>) =
            match crate::dataset::source::parse_source_string(raw) {
                Ok(parsed) if !parsed.window.is_empty() => {
                    let iv = &parsed.window.0[0];
                    (
                        parsed.path,
                        Some((iv.min_incl as usize, iv.max_excl as usize)),
                    )
                }
                Ok(parsed) => (parsed.path, None),
                Err(e) => {
                    return Err(Error::Other(format!(
                        "facet '{name}': source '{raw}' has a malformed window: {e}"
                    )));
                }
            };
        let resolved = self.resolve_path_str(&path_str)?;
        let reader = io::open_vec::<T>(&resolved)?;

        let window = window_from_suffix.or_else(|| facet.window().and_then(parse_window_first));
        if let Some((start, end)) = window {
            return Ok(Arc::new(WindowedVectorReader::new(reader, start, end)));
        }
        Ok(Arc::from(reader))
    }

    /// Resolve a bare path string (no window suffix) against the
    /// data source root. Mirrors `resolve_as_string` but takes a raw
    /// path so callers that pre-parsed the suffix can join it
    /// correctly without round-tripping through `FacetConfig`.
    ///
    /// Absolute URLs (`http://`, `https://`) in the YAML pass
    /// through unchanged regardless of the dataset's base location —
    /// so a local `dataset.yaml` can reference remote facets and a
    /// remote dataset can reference facets hosted elsewhere.
    fn resolve_path_str(&self, path: &str) -> Result<String> {
        if is_absolute_url(path) {
            return Ok(path.to_string());
        }
        if is_local_facet_source(path) {
            return Ok(file_uri_to_path(path).to_string());
        }
        if has_absolute_uri_scheme(path) {
            return Ok(path.to_string());
        }
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
        if facet.is_series() {
            let storage = self.open_facet_storage(name)?;
            let series = storage
                .series_ref()
                .ok_or_else(|| Error::Other(format!("facet '{name}': series not realized")))?
                .clone();
            return Ok(Arc::new(ShardedVvecReader::<T>::open(
                series,
                T::ELEM_SIZE,
            )?));
        }
        let path_or_url = self.resolve_as_string(facet)?;
        let reader = io::open_vvec::<T>(&path_or_url)?;
        Ok(Arc::from(reader))
    }

    /// Resolve a facet to a path string (local) or URL string (remote).
    fn resolve_as_string(&self, facet: &FacetConfig) -> Result<String> {
        let source_str = single_source(facet, "resolve")?;
        if is_absolute_url(source_str) {
            return Ok(source_str.to_string());
        }
        if is_local_facet_source(source_str) {
            return Ok(file_uri_to_path(source_str).to_string());
        }
        if has_absolute_uri_scheme(source_str) {
            return Ok(source_str.to_string());
        }
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

        // Every facet this view can open, whether or not the spec
        // names it. `base_content`, `query_terms` and `query_filters`
        // are declarable in `dataset.yaml` and openable by name, so a
        // manifest that omitted them made them invisible to everything
        // that asks what a profile holds — precache among them.
        let standard_facets: &[(&str, Option<&FacetConfig>, Option<StandardFacet>)] = &[
            (
                "base_vectors",
                self.config.base_vectors.as_ref(),
                Some(StandardFacet::BaseVectors),
            ),
            (
                "query_vectors",
                self.config.query_vectors.as_ref(),
                Some(StandardFacet::QueryVectors),
            ),
            (
                "neighbor_indices",
                self.config.neighbor_indices.as_ref(),
                Some(StandardFacet::NeighborIndices),
            ),
            (
                "neighbor_distances",
                self.config.neighbor_distances.as_ref(),
                Some(StandardFacet::NeighborDistances),
            ),
            (
                "metadata_content",
                self.config.metadata_content.as_ref(),
                Some(StandardFacet::MetadataContent),
            ),
            (
                "metadata_predicates",
                self.config.metadata_predicates.as_ref(),
                Some(StandardFacet::MetadataPredicates),
            ),
            (
                "predicate_results",
                self.config.predicate_results.as_ref(),
                Some(StandardFacet::MetadataResults),
            ),
            (
                "metadata_layout",
                self.config.metadata_layout.as_ref(),
                Some(StandardFacet::MetadataLayout),
            ),
            (
                "prefiltered_neighbor_indices",
                self.config.prefiltered_neighbor_indices.as_ref(),
                Some(StandardFacet::PrefilteredNeighborIndices),
            ),
            (
                "prefiltered_neighbor_distances",
                self.config.prefiltered_neighbor_distances.as_ref(),
                Some(StandardFacet::PrefilteredNeighborDistances),
            ),
            (
                "postfiltered_neighbor_indices",
                self.config.postfiltered_neighbor_indices.as_ref(),
                Some(StandardFacet::PostfilteredNeighborIndices),
            ),
            (
                "postfiltered_neighbor_distances",
                self.config.postfiltered_neighbor_distances.as_ref(),
                Some(StandardFacet::PostfilteredNeighborDistances),
            ),
            ("base_content", self.config.base_content.as_ref(), None),
            ("query_terms", self.config.query_terms.as_ref(), None),
            ("query_filters", self.config.query_filters.as_ref(), None),
        ];

        for (name, facet_opt, kind) in standard_facets {
            if let Some(facet) = facet_opt {
                // A series has no single source path. Reporting `None`
                // makes a windowed plan degrade to a whole-facet fetch,
                // which the fallback gate then refuses — the safe
                // transitional answer until the planner consumes the
                // realized `Shards` model. Reporting the first shard
                // instead would silently map the facet against a
                // fraction of itself.
                let source = facet.source().map(str::to_string);
                // The **format** is answerable for a series even though
                // the path is not: every shard shares one format (SRD
                // invariant 4), so the first declared source names it.
                // Leaving this `None` for a series made a sharded facet
                // look formatless to anything that asks what it is.
                let source_type = facet
                    .sources()
                    .first()
                    .map(|s| s.as_str())
                    .and_then(FacetDescriptor::infer_type);
                manifest.insert(
                    name.to_string(),
                    FacetDescriptor {
                        name: name.to_string(),
                        source_type,
                        source_path: source,
                        window: facet.window().map(|w| w.to_string()),
                        standard_kind: *kind,
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
            // Declared in `ProfileConfig` and therefore declarable in
            // `dataset.yaml`. Without an arm here a dataset can name
            // one and nothing can open it: `facet()` reports it
            // missing, and `derive` — which asks this route for an
            // element type — treats it as a non-data key and drops it
            // from the output.
            "base_content" => self.config.base_content.as_ref(),
            "query_terms" => self.config.query_terms.as_ref(),
            "query_filters" => self.config.query_filters.as_ref(),
            "neighbor_indices" => self.config.neighbor_indices.as_ref(),
            "neighbor_distances" => self.config.neighbor_distances.as_ref(),
            "metadata_content" => self.config.metadata_content.as_ref(),
            "metadata_predicates" => self.config.metadata_predicates.as_ref(),
            // Canonical spec key first (`StandardFacet::basenames`), then
            // the legacy names — the same set the config field accepts
            // as serde aliases. Lookup must know every spelling the
            // reader methods pass, or a facet resolves by one route and
            // not another.
            "metadata_results" | "metadata_indices" | "predicate_results" => {
                self.config.predicate_results.as_ref()
            }
            "metadata_layout" => self.config.metadata_layout.as_ref(),
            // Canonical pre-filter keys + legacy `filtered_*` aliases
            // (legacy on-disk files carry pre-filter shape).
            "prefiltered_neighbor_indices" | "filtered_neighbor_indices" => {
                self.config.prefiltered_neighbor_indices.as_ref()
            }
            "prefiltered_neighbor_distances" | "filtered_neighbor_distances" => {
                self.config.prefiltered_neighbor_distances.as_ref()
            }
            // Canonical post-filter keys.
            "postfiltered_neighbor_indices" => self.config.postfiltered_neighbor_indices.as_ref(),
            "postfiltered_neighbor_distances" => {
                self.config.postfiltered_neighbor_distances.as_ref()
            }
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
    ) -> std::result::Result<
        crate::typed_access::TypedReader<T>,
        crate::typed_access::TypedAccessError,
    > {
        self.require_facet_shape(name, crate::dataset::facet::FacetShape::Elements)
            .map_err(|e| crate::typed_access::TypedAccessError::Io(e.to_string()))?;
        let facet = self.facet_config_by_name(name).ok_or_else(|| {
            crate::typed_access::TypedAccessError::Io(format!("facet '{}' not found", name))
        })?;
        // A series has no single resource to resolve. The free
        // `open_facet_typed` reads one through the shard model, and
        // routing there is what keeps the two entry points answering
        // the same question the same way — a method that errored where
        // the function succeeded would be a bypass in the direction
        // nobody wants (SH-79).
        if facet.is_series() {
            return open_facet_typed::<T>(self, name);
        }
        let resource = self
            .resolve_resource(facet)
            .map_err(|e| crate::typed_access::TypedAccessError::Io(e.to_string()))?;
        match resource {
            ResourceLocation::FileSystem(path) => {
                crate::typed_access::TypedReader::<T>::open(&path)
            }
            ResourceLocation::Http(url) => {
                let native_type = crate::typed_access::ElementType::from_url(&url)
                    .map_err(crate::typed_access::TypedAccessError::Io)?;
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

/// Cache-relative path for a facet inside its dataset's cache
/// directory (`<cache_root>/<dataset>/<relpath>`): the home-relative
/// path when the facet lives under the dataset's home URL, otherwise
/// the URL basename — flat under the dataset directory, per the
/// mandated dataset-keyed layout.
pub(crate) fn facet_cache_relpath<'a>(resolved: &'a str, home_norm: &str) -> &'a str {
    if let Some(rel) = resolved.strip_prefix(home_norm) {
        return rel;
    }
    resolved.rsplit('/').next().unwrap_or(resolved)
}

impl GenericTestDataView {
    /// Hard-stop guard for cache-file collisions: two facets of this
    /// profile must never map to the same `<dataset>/<relpath>` from
    /// different source URLs — chunk sidecars are keyed to the file,
    /// so a shared path with divergent bytes is silent corruption.
    /// Realistically reachable only via basename collisions between
    /// out-of-home facets; cheap to check (≤ a dozen facets).
    fn reject_relpath_collisions(
        &self,
        facet_name: &str,
        resolved: &str,
        relpath: &str,
        home_norm: &str,
    ) -> Result<()> {
        for other_name in self.facet_manifest().keys() {
            if other_name == facet_name {
                continue;
            }
            // Every file the other facet names, not its single source.
            // A series has no single source, so reading one would skip
            // series facets entirely — and a shard is exactly the kind
            // of file that collides, since shard names are derived from
            // a basename many facets could share (SH-33).
            let Some(other) = self.facet_config_by_name(other_name) else {
                continue;
            };
            for raw in other.declared_files() {
                let other_path = match crate::dataset::source::parse_source_string(&raw) {
                    Ok(parsed) => parsed.path,
                    Err(_) => raw.clone(),
                };
                let Ok(other_resolved) = self.resolve_path_str(&other_path) else {
                    continue;
                };
                if other_resolved == resolved {
                    continue;
                }
                if facet_cache_relpath(&other_resolved, home_norm) == relpath {
                    return Err(Error::Other(format!(
                        "cache filename collision in dataset '{}': facets \
                         '{facet_name}' ({resolved}) and '{other_name}' \
                         ({other_resolved}) both cache as '{relpath}'. Rename \
                         one of the source files (or move it under the \
                         dataset's home URL) so each facet has a distinct \
                         filename.",
                        self.dataset_name.as_deref().unwrap_or("?"),
                    )));
                }
            }
        }
        Ok(())
    }
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
        self.open_uniform(
            self.config.neighbor_distances.as_ref(),
            "neighbor_distances",
        )
    }

    fn prefiltered_neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>> {
        self.open_uniform(
            self.config.prefiltered_neighbor_indices.as_ref(),
            "prefiltered_neighbor_indices",
        )
    }

    fn prefiltered_neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>> {
        self.open_uniform(
            self.config.prefiltered_neighbor_distances.as_ref(),
            "prefiltered_neighbor_distances",
        )
    }

    fn postfiltered_neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>> {
        self.open_uniform(
            self.config.postfiltered_neighbor_indices.as_ref(),
            "postfiltered_neighbor_indices",
        )
    }

    fn postfiltered_neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>> {
        self.open_uniform(
            self.config.postfiltered_neighbor_distances.as_ref(),
            "postfiltered_neighbor_distances",
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

    fn metadata_results(&self) -> Result<Arc<dyn VvecReader<i32>>> {
        self.open_variable(self.config.predicate_results.as_ref(), "metadata_results")
    }

    fn facet(&self, name: &str) -> Result<Arc<dyn VectorReader<f32>>> {
        // Try standard facets first (uniform vector types). The legacy
        // `filtered_*` aliases route to the prefilter distances method
        // because legacy on-disk files carry pre-filter shape.
        match name {
            "base_vectors" => return self.base_vectors(),
            "query_vectors" => return self.query_vectors(),
            "neighbor_distances" => return self.neighbor_distances(),
            "prefiltered_neighbor_distances" | "filtered_neighbor_distances" => {
                return self.prefiltered_neighbor_distances();
            }
            "postfiltered_neighbor_distances" => return self.postfiltered_neighbor_distances(),
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
        // A record-shaped facet has no element width, and saying so as
        // "unknown element type" reads like a gap in this build rather
        // than a property of the data. Name the shape and the reader
        // that opens it.
        self.require_facet_shape(name, crate::dataset::facet::FacetShape::Elements)?;
        let facet = self
            .facet_config_by_name(name)
            .ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        // Strip the `[start..end)` window suffix before inspecting
        // the extension — otherwise "base.fvec[0..1000)" splits at the
        // dots inside the window and yields "1000)" as the "extension",
        // which infers to no element type at all. The whole facet then
        // looks unrecognised to the precache iterator and gets silently
        // skipped — exactly the symptom that started this fix.
        // Any entry answers this: every shard of a series shares one
        // format and element type (SRD invariant 4), so the first is not
        // a stand-in for the facet — it is the same answer the others
        // would give.
        let raw = facet
            .sources()
            .first()
            .ok_or_else(|| Error::Other(format!("facet '{name}' declares no source")))?;
        let source = match crate::dataset::source::parse_source_string(raw) {
            Ok(parsed) => parsed.path,
            Err(_) => raw.to_string(),
        };
        crate::typed_access::ElementType::from_extension(source.rsplit('.').next().unwrap_or(""))
            .ok_or_else(|| Error::Other(format!("unknown element type for facet '{name}'")))
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
        // `None` for a series: it has no single path, and answering with
        // the first shard would resolve the facet to a fraction of
        // itself (SH-74).
        let raw = facet.source()?;
        let path_str = match crate::dataset::source::parse_source_string(raw) {
            Ok(parsed) => parsed.path,
            Err(_) => raw.to_string(),
        };
        self.resolve_path_str(&path_str).ok()
    }

    fn open_facet_storage(&self, name: &str) -> Result<FacetStorage> {
        let facet = self
            .facet_config_by_name(name)
            .ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        // A series resolves every shard's file; a single facet resolves
        // one. Both go through the same per-file resolution below, so a
        // shard is anchored, cached, and collision-checked exactly as a
        // lone facet is (SH-32).
        if facet.is_series() {
            return self.open_facet_series(name, facet);
        }
        // Strip any window suffix from the source so Storage opens
        // the underlying file/url, not a half-parsed token.
        let raw = single_source(facet, "open storage for")?;
        let path_str = match crate::dataset::source::parse_source_string(raw) {
            Ok(parsed) => parsed.path,
            Err(_) => raw.to_string(),
        };
        let resolved = self.resolve_path_str(&path_str)?;
        // Catalog-anchored open. The cache layout is fixed by design:
        //
        //   <cache_root>/<dataset_name>/<dataset files>
        //
        // For facets under the dataset's home URL (`catalog_source`)
        // the file keeps its home-relative path ("base.fvec",
        // "profiles/1m/base.fvec") so the cache mirrors the published
        // layout. A facet living OUTSIDE the home URL (legal in
        // knn_entries catalogs — several sized datasets sharing one
        // remote directory of files) still caches under THIS
        // dataset's directory, flat, by URL basename. Datasets that
        // share a remote file each hold their own copy — the layout
        // is dataset-keyed, never URL-keyed. (The previous fallback
        // used the absolute URL as a "relative path", producing
        // `<dataset>/s3:/...` trees whose colon broke Windows.)
        //
        // `facet_cache_relpath` collisions (two facets of this
        // profile mapping to one cache file from different URLs) are
        // rejected before any byte lands — see the guard below.
        let storage = match (&self.dataset_name, &self.catalog_source) {
            (Some(ds_name), Some(home)) => {
                let home_norm = if home.ends_with('/') {
                    home.clone()
                } else {
                    format!("{home}/")
                };
                let file_relpath = facet_cache_relpath(&resolved, &home_norm);
                self.reject_relpath_collisions(name, &resolved, file_relpath, &home_norm)?;
                crate::storage::Storage::open_layered(&resolved, ds_name, file_relpath, home)
            }
            _ => crate::storage::Storage::open(&resolved),
        }
        .map_err(|e| Error::Other(format!("storage open '{name}': {e}")))?;
        Ok(FacetStorage::new(storage))
    }
}

impl GenericTestDataView {
    /// How to open one resolved path, using this view's catalog
    /// anchoring. Shared by the single-file and series paths so a shard
    /// caches exactly where a lone facet would.
    fn shard_open_spec(&self, facet: &str, path_str: &str) -> Result<ShardOpen> {
        let resolved = self.resolve_path_str(path_str)?;
        Ok(match (&self.dataset_name, &self.catalog_source) {
            (Some(ds_name), Some(home)) => {
                let home_norm = if home.ends_with('/') {
                    home.clone()
                } else {
                    format!("{home}/")
                };
                let relpath = facet_cache_relpath(&resolved, &home_norm).to_string();
                self.reject_relpath_collisions(facet, &resolved, &relpath, &home_norm)?;
                ShardOpen::Layered {
                    resolved,
                    dataset: ds_name.clone(),
                    relpath,
                    home: home.clone(),
                }
            }
            _ => ShardOpen::Plain { resolved },
        })
    }

    /// Open a facet declared as a multi-file series.
    ///
    /// The declaration is realized into the ordinal model first, so a
    /// series that disagrees with itself fails here rather than at the
    /// first read. Only the first shard's file is opened; the rest wait
    /// until something reads them (SH-26).
    fn open_facet_series(&self, name: &str, facet: &FacetConfig) -> Result<FacetStorage> {
        // Bare local entries need a record count from the file itself,
        // which means resolving the path and opening it — the local
        // convenience SH-63 permits.
        let probe = |src: &crate::dataset::source::DSSource| -> std::result::Result<u64, String> {
            let spec = self
                .shard_open_spec(name, &src.path)
                .map_err(|e| e.to_string())?;
            let storage = spec.open().map_err(|e| e.to_string())?;
            records_in(&src.path, &storage)
                .ok_or_else(|| format!("cannot derive a record count from '{}'", src.path))
        };
        let shards = crate::dataset::shards::realize(name, &facet.declaration(), &probe)
            .map_err(|e| Error::Other(e.to_string()))?;

        let specs = shards
            .entries()
            .iter()
            .map(|e| self.shard_open_spec(name, &e.source.path))
            .collect::<Result<Vec<_>>>()?;

        // Two shards naming one file is a declared relationship and
        // shares a cache entry by design (SH-66, SH-80). Two *different*
        // files landing on one cache path is a collision, and the
        // cross-facet guard cannot see it — it compares facets, and
        // these are shards of the same one.
        let mut by_relpath: std::collections::HashMap<&str, &str> =
            std::collections::HashMap::new();
        for spec in &specs {
            if let ShardOpen::Layered {
                resolved, relpath, ..
            } = spec
                && let Some(prev) = by_relpath.insert(relpath, resolved)
                && prev != resolved
            {
                return Err(Error::Other(format!(
                    "facet '{name}': shards '{prev}' and '{resolved}' both cache as \
                     '{relpath}'. Two shards may share a file, but two files cannot \
                     share a cache entry — rename one so each has a distinct filename."
                )));
            }
        }
        let series = Series::new(shards, specs);
        let first = series.file(0)?.clone();
        Ok(FacetStorage::series(first, series))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_absolute_url_recognises_http_schemes() {
        assert!(is_absolute_url("http://example.com/data.fvec"));
        assert!(is_absolute_url("https://example.com/data.fvec"));
        assert!(!is_absolute_url("file:///tmp/data.fvec"));
        assert!(!is_absolute_url("/absolute/local/path.fvec"));
        assert!(!is_absolute_url("relative/path.fvec"));
        assert!(!is_absolute_url("data.fvec"));
        assert!(!is_absolute_url(""));
    }

    #[test]
    fn local_facet_source_covers_abs_path_and_file_uri() {
        assert!(is_local_facet_source("/abs/path.fvec"));
        assert!(is_local_facet_source("file:///abs/path.fvec"));
        assert!(is_local_facet_source("file://host/abs/path.fvec"));
        assert!(!is_local_facet_source("relative/path.fvec"));
        assert!(!is_local_facet_source("data.fvec"));
        assert!(!is_local_facet_source("http://x/path.fvec"));
        assert!(!is_local_facet_source(""));
    }

    #[test]
    fn file_uri_to_path_strips_scheme_and_host() {
        assert_eq!(file_uri_to_path("file:///abs/path.fvec"), "/abs/path.fvec");
        assert_eq!(
            file_uri_to_path("file://localhost/abs/path.fvec"),
            "/abs/path.fvec"
        );
        assert_eq!(file_uri_to_path("/abs/path.fvec"), "/abs/path.fvec");
        assert_eq!(file_uri_to_path("relative/path.fvec"), "relative/path.fvec");
    }

    /// When a remote-loaded `TestDataGroup` has a facet whose source
    /// is a `file://` URI (a fully-precached local file), the resolver
    /// must yield a `ResourceLocation::FileSystem` — not try to join
    /// the URI onto the catalog's HTTP base. Symmetric for an
    /// absolute filesystem path. Same semantic as "remote facet that
    /// already lives on disk".
    #[test]
    fn remote_catalog_with_local_facet_short_circuits_to_filesystem() {
        use crate::model::FacetConfig;

        fn empty_profile() -> crate::model::ProfileConfig {
            crate::model::ProfileConfig {
                attributes: Default::default(),
                inherits: None,
                base_count: None,
                maxk: None,
                partition: false,
                base_vectors: None,
                base_content: None,
                query_vectors: None,
                query_terms: None,
                query_filters: None,
                neighbor_indices: None,
                neighbor_distances: None,
                prefiltered_neighbor_indices: None,
                prefiltered_neighbor_distances: None,
                postfiltered_neighbor_indices: None,
                postfiltered_neighbor_distances: None,
                metadata_content: None,
                metadata_predicates: None,
                predicate_results: None,
                metadata_layout: None,
            }
        }

        let mut profile = empty_profile();
        profile.base_vectors = Some(FacetConfig::Simple(
            "file:///tank/share/base.fvec".to_string(),
        ));
        profile.query_vectors = Some(FacetConfig::Simple("/tank/share/query.fvec".to_string()));

        let base_url = Url::parse("https://catalog.example.com/datasets/x/").unwrap();
        let view = GenericTestDataView::new(DataSource::Http(base_url), profile);

        let bv = view.config.base_vectors.as_ref().unwrap();
        let q = view.config.query_vectors.as_ref().unwrap();
        match view.resolve_resource(bv).unwrap() {
            ResourceLocation::FileSystem(p) => {
                assert_eq!(p, PathBuf::from("/tank/share/base.fvec"))
            }
            ResourceLocation::Http(u) => panic!("expected FileSystem, got Http({u})"),
        }
        match view.resolve_resource(q).unwrap() {
            ResourceLocation::FileSystem(p) => {
                assert_eq!(p, PathBuf::from("/tank/share/query.fvec"))
            }
            ResourceLocation::Http(u) => panic!("expected FileSystem, got Http({u})"),
        }

        // resolve_as_string short-circuits the same way (no `https://.../tank/...` join).
        assert_eq!(view.resolve_as_string(bv).unwrap(), "/tank/share/base.fvec");
        assert_eq!(view.resolve_as_string(q).unwrap(), "/tank/share/query.fvec");
    }

    /// Build a minimal uniform `.fvec` (dim-prefixed records) on disk
    /// and wrap it in a `FacetStorage` so the window-byte-range logic
    /// can read the real dim header at byte 0.
    fn fvec_storage(dir: &std::path::Path, dim: i32, records: usize) -> (PathBuf, FacetStorage) {
        let path = dir.join("base.fvec");
        let mut bytes = Vec::new();
        for r in 0..records {
            bytes.extend_from_slice(&dim.to_le_bytes());
            for d in 0..dim {
                bytes.extend_from_slice(&((r as f32) + d as f32).to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes).unwrap();
        let storage = crate::storage::Storage::open_path(&path).unwrap();
        (path, FacetStorage::new(storage))
    }

    /// A facet declares its window in one of two encodings, and the
    /// whole-profile precache reads both, the way the reader does: the
    /// `[start..end)` suffix on a single source, else the `window:`
    /// field. The suffix wins when both are present; neither means no
    /// window. Consulting only the suffix once made the precache pull
    /// the entire base file for every field-form sized profile.
    #[test]
    fn a_declared_window_is_read_from_either_encoding() {
        let desc = |source: Option<&str>, window: Option<&str>| FacetDescriptor {
            name: "base_vectors".into(),
            source_path: source.map(str::to_string),
            source_type: None,
            window: window.map(str::to_string),
            standard_kind: None,
        };
        let bounds = |d: &FacetDescriptor| {
            let w = facet_declared_window(d).unwrap();
            w.0.first().map(|iv| (iv.min_incl, iv.max_excl))
        };
        assert_eq!(bounds(&desc(Some("base.fvec[1..3)"), None)), Some((1, 3)));
        assert_eq!(bounds(&desc(Some("base.fvec"), Some("1..3"))), Some((1, 3)));
        assert_eq!(
            bounds(&desc(Some("base.fvec[1..3)"), Some("0..4"))),
            Some((1, 3)),
            "the path suffix takes precedence over the window field"
        );
        assert_eq!(bounds(&desc(Some("base.fvec"), None)), None);
        // A series has no single source; its window lives in the field.
        assert_eq!(bounds(&desc(None, Some("[0..500]"))), Some((0, 500)));
        assert_eq!(bounds(&desc(None, None)), None);
    }

    /// **A malformed window is an error, not an absent one.**
    ///
    /// Returning `None` here is what a facet with no window returns, and
    /// the caller reads that as "fetch the whole file". Conflating the
    /// two is how `base.fvec[0,1000)` came to download a terabyte in
    /// silence: the reader clipped to zero records, the precache path
    /// saw no window, and neither said why.
    #[test]
    fn a_malformed_window_is_an_error_not_an_absent_window() {
        let tmp = tempfile::tempdir().unwrap();
        let (path, _storage) = fvec_storage(tmp.path(), 2, 4);
        let src = path.to_string_lossy().to_string();

        // And the declared-window helper the whole-profile precache
        // plans from propagates rather than quietly answering "no
        // window" — which would size, and fetch, the whole file.
        let desc = FacetDescriptor {
            name: "base_vectors".into(),
            source_path: Some(format!("{src}[0,1000)")),
            source_type: None,
            window: None,
            standard_kind: None,
        };
        assert!(
            facet_declared_window(&desc).is_err(),
            "a plan built on a malformed window must fail, not size the whole file"
        );
        let series_desc = FacetDescriptor {
            name: "base_vectors".into(),
            source_path: None,
            source_type: None,
            window: Some("[0..500]".into()),
            standard_kind: None,
        };
        let w = facet_declared_window(&series_desc).unwrap();
        assert_eq!((w.0[0].min_incl, w.0[0].max_excl), (0, 500), "a series declares its window in the field");
        let bad_field = FacetDescriptor {
            name: "base_vectors".into(),
            source_path: Some(src.clone()),
            source_type: None,
            window: Some("0,1000".into()),
            standard_kind: None,
        };
        let err = facet_declared_window(&bad_field).expect_err("the window field is checked too");
        assert!(err.to_string().contains("malformed"), "{err}");
        let bad_series = FacetDescriptor {
            name: "base_vectors".into(),
            source_path: None,
            source_type: None,
            window: Some("0,1000".into()),
            standard_kind: None,
        };
        assert!(facet_declared_window(&bad_series).is_err(), "a series' malformed window is an error too");
    }

    // ---- Coalescing -------------------------------------------------

    /// 100-byte chunks, so chunk N covers bytes [N*100, N*100+100).
    const CS: Option<u64> = Some(100);

    /// A range in the only shard — what every single-file facet has.
    fn sr(start: u64, end: u64) -> ShardRange {
        ShardRange::whole(start, end)
    }

    /// A range in a named shard.
    fn sr_at(shard: usize, start: u64, end: u64) -> ShardRange {
        ShardRange { shard, start, end }
    }

    #[test]
    fn coalescing_leaves_a_single_range_alone() {
        assert_eq!(coalesce_ranges(vec![], CS), Vec::<ShardRange>::new());
        assert_eq!(coalesce_ranges(vec![sr(10, 20)], CS), vec![sr(10, 20)]);
    }

    /// Overlapping and touching byte ranges merge under any granularity.
    #[test]
    fn overlapping_and_touching_ranges_merge() {
        assert_eq!(
            coalesce_ranges(vec![sr(0, 50), sr(25, 75)], CS),
            vec![sr(0, 75)]
        );
        assert_eq!(
            coalesce_ranges(vec![sr(0, 50), sr(50, 75)], CS),
            vec![sr(0, 75)]
        );
        assert_eq!(
            coalesce_ranges(vec![sr(0, 50), sr(50, 75)], None),
            vec![sr(0, 75)]
        );
    }

    /// **Ranges in different shards never merge**, however adjacent
    /// their byte offsets look.
    ///
    /// Two shards are two files: byte 90 of shard 0 and byte 10 of
    /// shard 1 are in the same chunk *of different files*, and one
    /// request cannot cover both (SH-28). Without the shard in the key
    /// this is exactly the case that would silently fold together.
    #[test]
    fn ranges_in_different_shards_never_merge() {
        assert_eq!(
            coalesce_ranges(vec![sr_at(0, 10, 20), sr_at(1, 20, 30)], CS),
            vec![sr_at(0, 10, 20), sr_at(1, 20, 30)]
        );
        // Even byte-identical ranges stay separate across shards.
        assert_eq!(
            coalesce_ranges(vec![sr_at(0, 0, 50), sr_at(1, 0, 50)], CS).len(),
            2
        );
        // And within one shard they still merge as before.
        assert_eq!(
            coalesce_ranges(vec![sr_at(2, 10, 20), sr_at(2, 80, 90)], CS),
            vec![sr_at(2, 10, 90)]
        );
    }

    /// **Two ranges in one chunk are already one fetch.** Issuing them
    /// separately asks the device for the same bytes twice.
    #[test]
    fn ranges_sharing_a_chunk_merge() {
        assert_eq!(
            coalesce_ranges(vec![sr(10, 20), sr(80, 90)], CS),
            vec![sr(10, 90)]
        );
        // Without chunking they are plainly disjoint and stay so.
        assert_eq!(
            coalesce_ranges(vec![sr(10, 20), sr(80, 90)], None),
            vec![sr(10, 20), sr(80, 90)]
        );
    }

    /// Adjacent chunks are contiguous on the device, so one request
    /// covering both beats two.
    #[test]
    fn ranges_in_adjacent_chunks_merge() {
        assert_eq!(
            coalesce_ranges(vec![sr(10, 20), sr(150, 160)], CS),
            vec![sr(10, 160)]
        );
    }

    /// **But a whole chunk of gap is not bridged.** Merging across it
    /// would fetch a chunk nobody asked for — the exact failure the
    /// plan exists to make visible rather than commit.
    #[test]
    fn a_chunk_of_gap_is_not_bridged() {
        assert_eq!(
            coalesce_ranges(vec![sr(10, 20), sr(250, 260)], CS),
            vec![sr(10, 20), sr(250, 260)]
        );
        assert_eq!(
            coalesce_ranges(vec![sr(10, 20), sr(350, 360)], CS),
            vec![sr(10, 20), sr(350, 360)]
        );
    }

    /// Input order is not the caller's problem.
    #[test]
    fn ranges_are_sorted_before_merging() {
        // Chunks 0, 1 and 4. The first two are a run; chunks 2 and 3
        // lie untouched between that run and the last range.
        assert_eq!(
            coalesce_ranges(vec![sr(450, 460), sr(10, 20), sr(150, 160)], CS),
            vec![sr(10, 160), sr(450, 460)],
            "sorting happens first, then merging by chunk adjacency"
        );
    }

    /// A chain merges transitively, and an empty range contributes
    /// nothing rather than acting as a bridge.
    #[test]
    fn a_chain_merges_and_empty_ranges_drop_out() {
        assert_eq!(
            coalesce_ranges(vec![sr(0, 10), sr(100, 110), sr(200, 210)], CS),
            vec![sr(0, 210)],
            "chunks 0, 1 and 2 are a run"
        );
        assert_eq!(
            coalesce_ranges(vec![sr(0, 10), sr(50, 50), sr(300, 310)], CS),
            vec![sr(0, 10), sr(300, 310)],
            "an empty range is not a bridge"
        );
    }

    /// A degenerate chunk size must not divide by zero or merge
    /// everything into one range.
    #[test]
    fn a_zero_chunk_size_falls_back_to_byte_adjacency() {
        assert_eq!(
            coalesce_ranges(vec![sr(0, 10), sr(300, 310)], Some(0)),
            vec![sr(0, 10), sr(300, 310)]
        );
    }

    // ---- RangeFill ---------------------------------------------------

    /// `RangeFill`'s derived numbers are what a caller decides on: a
    /// fetch is chunk-granular, and both "already warm" and "this is
    /// secretly a full download" have to be readable off the struct.
    #[test]
    fn range_fill_reports_cost_at_chunk_granularity() {
        let f = RangeFill {
            first_chunk: 0,
            last_chunk: 1,
            chunk_size: 8 << 20,
            chunks: 2,
            chunks_resident: 1,
            aligned_start: 0,
            aligned_end: 16 << 20,
        };
        assert_eq!(f.chunks_to_fetch(), 1);
        assert_eq!(f.bytes_to_fetch(), 8 << 20, "resident chunks are free");
        assert!(!f.is_resident());
        assert_eq!(f.overfetch_bytes(0, 4096), (16 << 20) - 4096);

        let warm = RangeFill {
            chunks_resident: 2,
            ..f
        };
        assert!(warm.is_resident());
        assert_eq!(warm.bytes_to_fetch(), 0);
    }

    /// Local storage has no chunks, so it has no plan — and a caller
    /// must read that as "free", not as "unknown".
    #[test]
    fn local_storage_has_no_range_plan() {
        let tmp = tempfile::tempdir().unwrap();
        let (_, storage) = fvec_storage(tmp.path(), 2, 4);
        assert!(storage.is_local());
        assert_eq!(storage.range_fill(0, 12), None);
    }

    // ---- The offset-index cache --------------------------------------

    /// Build a variable-length file: 4-byte dim then `dim` i32s, so no
    /// two records need be the same size.
    fn ivvec_storage(dir: &std::path::Path, dims: &[i32]) -> (PathBuf, FacetStorage) {
        use std::io::Write as _;
        let path = dir.join("meta.ivvec");
        let mut f = std::fs::File::create(&path).unwrap();
        for &d in dims {
            f.write_all(&d.to_le_bytes()).unwrap();
            for e in 0..d {
                f.write_all(&e.to_le_bytes()).unwrap();
            }
        }
        drop(f);
        let storage = crate::storage::Storage::open_path(&path).unwrap();
        (path, FacetStorage::new(storage))
    }

    /// **The index is loaded once per handle.** Two asks return the
    /// same allocation, not two equal ones — which is the difference
    /// between a cache and a coincidence.
    #[test]
    fn a_handle_loads_its_offset_index_once() {
        let tmp = tempfile::tempdir().unwrap();
        let (path, storage) = ivvec_storage(tmp.path(), &[3, 1, 8, 2]);
        let src = path.to_string_lossy().to_string();

        let first = storage.published_offsets(&src, 4).expect("offsets load");
        let second = storage.published_offsets(&src, 4).expect("offsets load");
        assert!(
            std::sync::Arc::ptr_eq(&first, &second),
            "the second ask must be served from the handle, not reloaded"
        );
        assert_eq!(first.len(), 4);
    }

    /// A handle holding offsets for one element width will not answer
    /// for another. Offsets are computed by walking with a stride, so
    /// serving them for a different width resolves a window to
    /// plausible, wrong bytes.
    #[test]
    fn a_handle_refuses_offsets_for_a_different_element_width() {
        let tmp = tempfile::tempdir().unwrap();
        let (path, storage) = ivvec_storage(tmp.path(), &[3, 1, 8, 2]);
        let src = path.to_string_lossy().to_string();

        assert!(storage.published_offsets(&src, 4).is_some());
        assert!(
            storage.published_offsets(&src, 8).is_none(),
            "a width mismatch degrades to no window rather than wrong bytes"
        );
    }

    /// **The cache is safe to race.** Two threads asking at once may
    /// both load — wasteful, never wrong — but every caller must come
    /// away with the *same* allocation, or "cached on the handle" is a
    /// claim that quietly does not hold under concurrency.
    #[test]
    fn concurrent_asks_all_see_one_allocation() {
        let tmp = tempfile::tempdir().unwrap();
        let (path, storage) = ivvec_storage(tmp.path(), &[3, 1, 8, 2, 5, 9]);
        let src = path.to_string_lossy().to_string();
        let storage = std::sync::Arc::new(storage);

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let storage = storage.clone();
                let src = src.clone();
                std::thread::spawn(move || {
                    storage.published_offsets(&src, 4).expect("offsets load")
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let first = &results[0];
        for (i, r) in results.iter().enumerate() {
            assert!(
                std::sync::Arc::ptr_eq(first, r),
                "thread {i} got a different allocation; the cache did not settle"
            );
        }
        assert_eq!(first.len(), 6);
    }

    // ── the descriptor budget ──────────────────────────────────────

    /// `n` one-file shards, each holding four bytes of the shard's own
    /// index, opened under an explicit budget.
    fn byte_series(dir: &std::path::Path, n: usize, cap: usize) -> Series {
        use crate::dataset::shards::{Entry, Shards};
        let mut entries = Vec::new();
        let mut opens = Vec::new();
        for i in 0..n {
            let path = dir.join(format!("part__{i:04}.u8"));
            std::fs::write(&path, [i as u8; 4]).unwrap();
            entries.push(Entry {
                source: crate::dataset::source::parse_source_string(
                    path.to_str().unwrap(),
                )
                .unwrap(),
                file_base: 0,
                len: 4,
            });
            opens.push(ShardOpen::Plain {
                resolved: path.to_str().unwrap().to_string(),
            });
        }
        Series::with_cap(Shards::new(entries).unwrap(), opens, cap)
    }

    /// Whether the series is currently holding file `i` open. Reads
    /// the slot directly rather than widening `Series`: eviction order
    /// is an internal policy, and this test module is inside the module
    /// that owns it.
    fn is_open(s: &Series, i: usize) -> bool {
        s.files[i].opened.lock().expect("shard slot").is_some()
    }

    /// An explicit setting wins; otherwise a quarter of the limit,
    /// never below the floor, and a junk value is ignored rather than
    /// fatal (SH-59).
    #[test]
    fn the_cap_derivation_answers_from_its_two_inputs() {
        assert_eq!(resolve_open_file_cap(Some("3"), Some(1024)), 3);
        assert_eq!(resolve_open_file_cap(Some(" 12 "), Some(1024)), 12);
        assert_eq!(resolve_open_file_cap(None, Some(1024)), 256);
        // No limit to read: the documented default stands in.
        assert_eq!(resolve_open_file_cap(None, None), 256);
        // The floor keeps a constrained host workable.
        assert_eq!(resolve_open_file_cap(None, Some(4)), 8);
        // Junk, zero, and negatives fall through to the derivation
        // rather than stopping a dataset from opening.
        for junk in ["", "lots", "0", "-1", "8.5"] {
            assert_eq!(
                resolve_open_file_cap(Some(junk), Some(1024)),
                256,
                "{junk:?} should not have been honoured"
            );
        }
    }

    /// **A series never claims more descriptors than it has files.** The
    /// budget is an upper bound, not a target.
    #[test]
    fn the_cap_never_exceeds_the_file_count() {
        let tmp = tempfile::tempdir().unwrap();
        let s = byte_series(tmp.path(), 3, 4096);
        assert_eq!(s.cap, 3);
    }

    /// **Reading past the budget closes the least-recently-used file**
    /// (SH-59). With a real budget this branch needs more shards than
    /// the host has descriptors, which is why the cap is injectable.
    #[test]
    fn a_series_over_its_budget_closes_the_least_recently_used_file() {
        let tmp = tempfile::tempdir().unwrap();
        let s = byte_series(tmp.path(), 8, 3);

        for i in 0..8 {
            let _ = s.file(i).unwrap();
            assert!(
                s.open_file_count() <= 3,
                "after opening {i}, {} files were open",
                s.open_file_count()
            );
        }
        // The budget bound is met exactly, and it is the *oldest* three
        // that were closed rather than an arbitrary three.
        assert_eq!(s.open_file_count(), 3);
        let open: Vec<usize> = (0..8).filter(|&i| is_open(&s, i)).collect();
        assert_eq!(open, vec![5, 6, 7]);
    }

    /// **An evicted file reopens and reads the same bytes.** Eviction
    /// is a descriptor economy, not a loss of data — a second pass over
    /// a series wider than the budget must answer exactly as the first.
    #[test]
    fn an_evicted_file_reopens_and_reads_the_same_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let s = byte_series(tmp.path(), 8, 2);

        let first: Vec<u8> = (0..8)
            .map(|i| s.file(i).unwrap().read_bytes(0, 4).unwrap()[0])
            .collect();
        assert_eq!(first, (0..8).map(|i| i as u8).collect::<Vec<_>>());

        // Every file but the last two has been evicted by now; reading
        // backwards evicts the rest and reopens the front.
        let second: Vec<u8> = (0..8)
            .rev()
            .map(|i| s.file(i).unwrap().read_bytes(0, 4).unwrap()[0])
            .collect();
        assert_eq!(second, (0..8).rev().map(|i| i as u8).collect::<Vec<_>>());
        assert!(s.open_file_count() <= 2);
    }

    /// **Eviction never pulls a file out from under a reader** (SH-59).
    /// `file()` hands back an owned `Arc`, so a caller mid-read keeps
    /// its own claim after the series has closed its slot.
    #[test]
    fn eviction_cannot_invalidate_a_handle_a_caller_still_holds() {
        let tmp = tempfile::tempdir().unwrap();
        let s = byte_series(tmp.path(), 6, 2);

        let held = s.file(0).unwrap();
        for i in 1..6 {
            let _ = s.file(i).unwrap();
        }
        assert!(!is_open(&s, 0), "file 0 should have been evicted");
        // The evicted slot is empty, but the handle taken before the
        // eviction still reads.
        assert_eq!(held.read_bytes(0, 4).unwrap(), vec![0u8; 4]);
    }

    /// **Parallel readers stay inside the budget and see their own
    /// bytes.** The LRU is shared mutable state behind a lock; a series
    /// read from several threads must neither exceed the cap nor return
    /// one shard's bytes for another's.
    #[test]
    fn parallel_readers_respect_the_budget_and_read_correctly() {
        let tmp = tempfile::tempdir().unwrap();
        let s = std::sync::Arc::new(byte_series(tmp.path(), 16, 4));

        std::thread::scope(|scope| {
            for t in 0..8 {
                let s = std::sync::Arc::clone(&s);
                scope.spawn(move || {
                    for pass in 0..40 {
                        let i = (t * 7 + pass * 3) % 16;
                        let bytes = s.file(i).unwrap().read_bytes(0, 4).unwrap();
                        assert_eq!(
                            bytes,
                            vec![i as u8; 4],
                            "thread {t} pass {pass} read shard {i}"
                        );
                        assert!(
                            s.open_file_count() <= 4,
                            "budget exceeded: {}",
                            s.open_file_count()
                        );
                    }
                });
            }
        });
        assert!(s.open_file_count() <= 4);
    }

    /// Separate handles are separate caches — the point of putting the
    /// cache on the handle. A caller wanting reuse holds one; a caller
    /// that does not pays per handle and nothing leaks.
    #[test]
    fn separate_handles_do_not_share_the_index() {
        let tmp = tempfile::tempdir().unwrap();
        let (path, first) = ivvec_storage(tmp.path(), &[3, 1, 8, 2]);
        let src = path.to_string_lossy().to_string();
        let second = FacetStorage::new(crate::storage::Storage::open_path(&path).unwrap());

        let a = first.published_offsets(&src, 4).unwrap();
        let b = second.published_offsets(&src, 4).unwrap();
        assert_eq!(a, b, "same file, same offsets");
        assert!(
            !std::sync::Arc::ptr_eq(&a, &b),
            "but a distinct handle holds its own copy, with its own lifetime"
        );
    }
}
