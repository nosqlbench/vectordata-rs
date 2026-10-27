// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The dataset view contract: one profile of a dataset, its facets by
//! name, its attributes, and fetching and verifying them
//! ([`TestDataView`], with the typed readers of [`TestDataViewExt`]).
//!
//! vectordata implements it (`GenericTestDataView`, behind every
//! `TestDataGroup::profile`); a program codes against it. The plan and
//! report values of a fetch or prefetch are here too, so a caller reads
//! a fetch's cost and outcome in the API's own vocabulary.

use std::collections::HashMap;
use std::sync::Arc;

use crate::dataset::facet::StandardFacet;
use crate::dataset::source::DSWindow;
use crate::model::FacetConfig;
use crate::reader::{IoError, VectorReader, VvecReader};
use crate::{Element, Error, Result};

/// Resolve a facet's window bounds from whichever encoding the config
/// carries, mirroring `open_uniform`: the `[start..end)` suffix on the
/// source string takes precedence, else the explicit `window:` field.
///
/// A malformed window is an error, not an absent window. Conflating
/// them is what let `base.fvec[0,1000)` fall through to an unbounded
/// prebuffer and download a whole terabyte in silence.
pub fn parse_window_bounds(
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
    /// How many files a sharded facet spans; `None` for a single file.
    /// A series has no single source path, and a caller that reports
    /// what opening the facet will cost needs this number without
    /// realising the series first.
    pub shard_count: Option<u32>,
}

impl FacetDescriptor {
    /// Returns true if this is a recognized standard facet.
    pub fn is_standard(&self) -> bool {
        self.standard_kind.is_some()
    }

    /// Infer the source type from a file extension. Accepts any
    /// recognized xvec/vvec/scalar extension (including plural forms
    /// like `fvecs`, `mvecs`) via
    /// [`crate::ElementType::from_extension`], plus the
    /// container extensions (`slab`, `json`, `parquet`, `npy`, `hdf5`,
    /// `h5`). The returned string is the lowercase canonical form.
    pub fn infer_type(source: &str) -> Option<String> {
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
        if crate::ElementType::from_extension(&lower).is_some() {
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
/// This mirrors the Java `TestDataView` interface. Facets are read
/// through the typed readers of [`TestDataViewExt`]; metadata accessors
/// return the `FacetConfig` so callers can resolve the underlying
/// resource. vectordata implements it behind every
/// `TestDataGroup::profile`.
///
/// A second implementation — a test double, a view served by another
/// program — answers the descriptor questions and opens its facets; the
/// typed readers, fetch planning and shape checks come with the trait:
///
/// ```
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use veks_apicore::model::FacetConfig;
/// use veks_apicore::verify::{Normalization, VerifyOptions, VerifyReport};
/// use veks_apicore::{AnyElement, AnyVectorReader, ElementType, Error, FacetBytes, FacetDescriptor, IoError,
///     ProgressSink, Result, TestDataView, TestDataViewExt, VectorReader};
///
/// /// A base of two-dimensional vectors held in memory.
/// struct Rows(Vec<[f32; 2]>);
/// impl VectorReader<f32> for Rows {
///     fn dim(&self) -> usize { 2 }
///     fn count(&self) -> usize { self.0.len() }
///     fn get(&self, i: usize) -> std::result::Result<Vec<f32>, IoError> {
///         self.0.get(i).map(|r| r.to_vec()).ok_or(IoError::OutOfBounds(i))
///     }
/// }
///
/// struct InMemoryView(Arc<Rows>);
///
/// impl TestDataView for InMemoryView {
///     fn metadata_content(&self) -> Option<&FacetConfig> { None }
///     fn metadata_predicates(&self) -> Option<&FacetConfig> { None }
///     fn predicate_results(&self) -> Option<&FacetConfig> { None }
///     fn metadata_layout(&self) -> Option<&FacetConfig> { None }
///     fn facet_manifest(&self) -> HashMap<String, FacetDescriptor> {
///         let base = FacetDescriptor { name: "base_vectors".into(), source_path: Some("base.fvecs".into()),
///             source_type: Some("fvecs".into()), window: None, standard_kind: None, shard_count: None };
///         HashMap::from([("base_vectors".to_string(), base)])
///     }
///     fn facet_element_type(&self, _name: &str) -> Result<ElementType> { Ok(ElementType::F32) }
///     fn base_count(&self) -> Option<u64> { Some(self.0.0.len() as u64) }
///     fn distance_function(&self) -> Option<String> { Some("COSINE".into()) }
///     fn facet_source(&self, _name: &str) -> Option<String> { Some("base.fvecs".into()) }
///     fn open_facet_storage(&self, name: &str) -> Result<Box<dyn FacetBytes>> {
///         Err(Error::Other(format!("'{name}' has no bytes: it is in memory")))
///     }
///     fn open_facet_as_type(&self, name: &str, target: ElementType) -> Result<AnyVectorReader> {
///         match (name, target) {
///             ("base_vectors", ElementType::F32) => Ok(f32::into_any(self.0.clone())),
///             _ => Err(Error::MissingFacet(name.into())),
///         }
///     }
///     fn normalization(&self) -> Result<Normalization> {
///         Err(Error::Other("not measured".into()))
///     }
///     fn verify_report(&self, _options: &VerifyOptions, _progress: &mut dyn ProgressSink) -> VerifyReport {
///         VerifyReport::default()
///     }
/// }
///
/// let view: Arc<dyn TestDataView> = Arc::new(InMemoryView(Arc::new(Rows(vec![[1.0, 0.0], [0.0, 1.0]]))));
/// let base = view.base_vectors_as::<f32>().unwrap(); // the typed reader, from the trait
/// assert_eq!(base.get(1).unwrap(), vec![0.0, 1.0]);
/// assert_eq!(view.distance(), Some(veks_apicore::DistanceFunction::Cosine));
/// ```
pub trait TestDataView: Send + Sync {
    // -- Vector facets --

    // -- Filtered neighbor facets (F / E) --

    // -- Metadata facets --

    /// Returns the facet config for metadata content, if present.
    fn metadata_content(&self) -> Option<&FacetConfig>;
    /// Returns the facet config for metadata predicates, if present.
    fn metadata_predicates(&self) -> Option<&FacetConfig>;
    /// Returns the facet config for predicate result indices, if present.
    fn predicate_results(&self) -> Option<&FacetConfig>;
    /// Returns the facet config for metadata layout, if present.
    fn metadata_layout(&self) -> Option<&FacetConfig>;

    // -- Facet discovery --

    /// Returns descriptors for all facets in the profile, without
    /// materializing data. Includes both standard and custom facets.
    fn facet_manifest(&self) -> HashMap<String, FacetDescriptor>;

    // -- Typed facet access --

    /// Returns the native element type of a named facet, inferred from
    /// the file extension in the profile config.
    ///
    /// ```rust,ignore
    /// let etype = view.facet_element_type("metadata_content")?; // → ElementType::U8
    /// ```
    fn facet_element_type(&self, name: &str) -> Result<crate::ElementType>;

    // -- Profile metadata --

    /// Returns the base vector count for this profile, if declared.
    fn base_count(&self) -> Option<u64>;

    // -- Dataset metadata --

    /// Returns the distance function name if declared in attributes,
    /// as written; [`distance`](Self::distance) is it typed.
    fn distance_function(&self) -> Option<String>;

    /// The declared distance function, read through the one spelling
    /// table ([`DistanceFunction::parse`](crate::DistanceFunction::parse)),
    /// so a catalog's `angular`, `ip` or `euclidean` is the same as
    /// `COSINE`, `DOT_PRODUCT` or `L2`. `None` when none is declared or
    /// the spelling is not one the table holds.
    fn distance(&self) -> Option<crate::DistanceFunction> {
        self.distance_function().as_deref().and_then(crate::DistanceFunction::parse)
    }

    /// One dataset-level attribute by name — a known one or a
    /// consumer's namespaced key — with overlays merged in
    /// ([`crate::overlay`]). `None` when the view does not carry
    /// attributes, as the default does.
    fn dataset_attribute(&self, _key: &str) -> Option<serde_yaml::Value> {
        None
    }

    /// Where [`dataset_attribute`](Self::dataset_attribute)'s value for
    /// `key` came from. `None` when the view does not know.
    fn dataset_attribute_source(&self, _key: &str) -> Option<crate::overlay::AttributeSource> {
        None
    }

    /// The dataset's name in the cache (`<cache_root>/<name>/`), when
    /// it has one there.
    fn cache_name(&self) -> Option<String> {
        None
    }

    /// The cache root the dataset is kept under when its catalog names
    /// one (a jvector catalog's `cache_dir`), in place of the configured
    /// cache. `None` when the configured cache is used.
    fn cache_dir(&self) -> Option<std::path::PathBuf> {
        None
    }

    /// The `quantization` attribute, when the profile or the dataset
    /// declares one: how the codes in `base_vectors` relate to the
    /// vectors they stand for. The profile's wins over the dataset's.
    /// `None` when neither declares it; `Some(Err)` when one does and it
    /// does not parse.
    fn quantization(&self) -> Option<std::result::Result<crate::dataset::Quantization, String>> {
        None
    }

    /// Resolved source string (path or URL) for a named facet, if it
    /// exists, for a caller holding an `Arc<dyn TestDataView>` that
    /// opens the file itself (with veks-io's `open_vec_as`,
    /// say) without knowing the dataset's transport details.
    fn facet_source(&self, name: &str) -> Option<String>;

    // -- Fetch --------------------------------------------------------

    /// Bring facets of this profile into the local cache, with progress.
    /// **The fetch call** — also what to use for "precache", "prebuffer"
    /// or "download" ([`crate::fetch`]).
    ///
    /// `request` picks the facets ([`FetchRequest::all`](crate::fetch::FetchRequest::all)
    /// or [`FetchRequest::facets`](crate::fetch::FetchRequest::facets))
    /// and optionally a record window; `progress` hears every step
    /// ([`Silent`](crate::fetch::Silent), vectordata's `TextMeter`,
    /// or a closure). This is [`plan_fetch`](Self::plan_fetch) followed
    /// by [`FetchPlan::execute`](crate::fetch::FetchPlan::execute).
    ///
    /// Before any byte moves, the whole run is checked: an unknown facet
    /// ([`Error::UnknownFacets`]), a window that would silently become a
    /// whole-facet download ([`Error::WindowUnresolvable`]), and a cache
    /// directory without room ([`Error::InsufficientCacheSpace`]) are all
    /// refused up front. Returning `Ok` means every planned byte is
    /// resident; readers opened afterwards take the zero-copy path.
    ///
    /// ```no_run
    /// # use vectordata::fetch::{FetchRequest, TextMeter};
    /// # fn demo(view: &dyn vectordata::TestDataView) -> vectordata::Result<()> {
    /// let report = view.fetch(
    ///     &FetchRequest::facets(["base_vectors", "query_vectors"]),
    ///     &mut TextMeter::stderr("Fetch"),
    /// )?;
    /// assert!(report.facet("base_vectors").is_some());
    /// # Ok(()) }
    /// ```
    fn fetch(
        &self,
        request: &crate::fetch::FetchRequest,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> Result<crate::fetch::FetchReport> {
        self.plan_fetch(request, progress)?.execute(progress)
    }

    /// Plan a [`fetch`](Self::fetch) without fetching: open each facet
    /// and resolve its window to the ranges and chunks it needs, net of
    /// what is already resident. Read the cost from the returned
    /// [`FetchPlan`](crate::fetch::FetchPlan), then
    /// [`execute`](crate::fetch::FetchPlan::execute) it or drop it.
    ///
    /// `progress` hears a `PlanBegin`/`PlanEnd` pair per facet, because
    /// opening a large remote facet takes long enough to need saying.
    fn plan_fetch(
        &self,
        request: &crate::fetch::FetchRequest,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> Result<crate::fetch::FetchPlan> {
        let mut facets = Vec::new();
        crate::fetch::plan_view(self, None, request, progress, &mut facets)?;
        Ok(crate::fetch::plan_from(facets, request, self.cache_capacity()))
    }

    /// Open a record-shaped facet — a slab of opaque records, such as
    /// `metadata_content` — for ordinal access to its records. A facet
    /// of element runs is refused naming the reader that opens it; ask
    /// [`facet_shape`](Self::facet_shape) to know which to use. A
    /// namespace in the declaration (`metadata_content.slab:layout`)
    /// selects that document inside the container.
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
        storage.open_records(name, namespace.as_deref())
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
    ///     FacetShape::Elements => { let r = view.facet_as::<i32>("metadata_content")?; }
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

    /// Low-level: an open handle on the facet's bytes
    /// ([`FacetBytes`](crate::FacetBytes)), for cache inspection,
    /// content identity and planning. Most callers want
    /// [`fetch`](Self::fetch) to bring bytes in and a reader to read
    /// them. Every implementor provides it; the default fetch and
    /// planning methods are built on it.
    fn open_facet_storage(&self, name: &str) -> Result<Box<dyn crate::FacetBytes>>;

    /// What a fetch checks before moving bytes: that `bytes` more fit
    /// where this view's facets are cached
    /// ([`Error::InsufficientCacheSpace`] otherwise). The default
    /// checks nothing; vectordata's checks its cache directory.
    fn cache_capacity(&self) -> crate::fetch::CapacityCheck {
        crate::fetch::CapacityCheck::unchecked()
    }

    // -- Typed access, erased -------------------------------------

    /// Open an element facet as `target`: at the file's own type when
    /// `target` is it, converted where
    /// [`ElementType::conversion_to`](crate::ElementType::conversion_to)
    /// allows, refused with [`IoError::ElementMismatch`]
    /// otherwise. Dyn-safe; the typed `*_as` methods of
    /// [`TestDataViewExt`] are built on it, and are what a caller with
    /// a `T` in hand uses.
    ///
    /// A view that cannot open typed facets leaves the default, which
    /// says so.
    ///
    /// The refusal is a value, so a caller learns the rule from the
    /// error: an f32 facet asked for as u8 is not narrowed.
    ///
    /// ```
    /// use vectordata::{Error, ElementType, TestDataGroup, TestDataViewExt};
    /// use vectordata::io::IoError;
    /// # use std::io::Write;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let ds = dir.path().join("ds");
    /// # std::fs::create_dir_all(&ds)?;
    /// # let mut f = std::fs::File::create(ds.join("base.fvecs"))?;
    /// # f.write_all(&1i32.to_le_bytes())?; f.write_all(&1.5f32.to_le_bytes())?;
    /// # drop(f);
    /// # std::fs::write(ds.join("dataset.yaml"), "name: floats\nprofiles:\n  default:\n    base_vectors: base.fvecs\n")?;
    /// let view = TestDataGroup::load(ds.to_str().unwrap())?.profile("default").unwrap();
    /// match view.base_vectors_as::<u8>() {
    ///     Err(Error::VectorIo(IoError::ElementMismatch { file, requested, reason })) => {
    ///         assert_eq!((file, requested), (ElementType::F32, ElementType::U8));
    ///         assert_eq!(reason, "no conversion between the two");
    ///     }
    ///     other => panic!("expected a refusal, got {:?}", other.map(|r| r.count())),
    /// }
    /// # Ok(()) }
    /// ```
    fn open_facet_as_type(
        &self,
        name: &str,
        target: crate::ElementType,
    ) -> Result<crate::reader::AnyVectorReader> {
        let _ = (name, target);
        Err(Error::Other("this view does not open typed facets".into()))
    }

    /// Open an element facet at the type its file holds, for a caller
    /// that does not know the type ahead of time.
    ///
    /// ```
    /// use vectordata::{ElementType, TestDataGroup, TestDataView, VectorReader};
    /// use vectordata::io::AnyVectorReader;
    /// # use std::io::Write;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let ds = dir.path().join("ds");
    /// # std::fs::create_dir_all(&ds)?;
    /// # let mut f = std::fs::File::create(ds.join("gt.i64vecs"))?;
    /// # for v in [[0i64, 1 << 40]] { f.write_all(&2i32.to_le_bytes())?; for x in v { f.write_all(&x.to_le_bytes())?; } }
    /// # drop(f);
    /// # std::fs::write(ds.join("dataset.yaml"), "name: wide\nprofiles:\n  default:\n    neighbor_indices: gt.i64vecs\n")?;
    /// let view = TestDataGroup::load(ds.to_str().unwrap())?.profile("default").unwrap();
    /// assert_eq!(view.facet_element_type("neighbor_indices")?, ElementType::I64);   // from the extension, no open
    /// let ids: Vec<i64> = match view.open_facet_native("neighbor_indices")? {
    ///     AnyVectorReader::I32(r) => r.get(0)?.into_iter().map(i64::from).collect(),  // the common case
    ///     AnyVectorReader::I64(r) => r.get(0)?,                                        // a corpus past 2^31 vectors
    ///     other => other.as_type::<i64>()?.get(0)?,                                    // any other integer widens
    /// };
    /// assert_eq!(ids, vec![0, 1 << 40]);
    /// # Ok(()) }
    /// ```
    fn open_facet_native(&self, name: &str) -> Result<crate::reader::AnyVectorReader> {
        let native = self.facet_element_type(name)?;
        self.open_facet_as_type(name, native)
    }

    /// Open a variable-length facet at the type its file holds. A
    /// variable-length facet is never converted.
    fn open_vvec_facet_native(&self, name: &str) -> Result<crate::reader::AnyVvecReader> {
        let _ = name;
        Err(Error::Other("this view does not open typed facets".into()))
    }

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
        self.prefetch_plan_on(&*storage, facet, window)
    }

    /// The same, against a handle the caller already holds.
    ///
    /// Reusing one handle is what makes the offset-index cache pay: a
    /// plan and the fetch that follows it both need the index, and
    /// opening a handle each would load it twice.
    fn prefetch_plan_on(
        &self,
        storage: &dyn crate::FacetBytes,
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
        let path = match (path, storage.is_series()) {
            (Some(p), _) => p,
            (None, true) => String::new(),
            (None, false) => {
                plan.degrades_to_full_download = true;
                plan.resident_bytes = storage.resident_bytes().min(facet_bytes);
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
            match storage.map_records(&path, start, end) {
                Some(m) => {
                    plan.prerequisite_bytes = plan.prerequisite_bytes.max(m.prerequisite_bytes);
                    plan.requested_ranges.extend(m.ranges.iter().copied());
                }
                None => {
                    // One unmappable interval makes the whole request a
                    // full download; reporting a partial plan beside it
                    // would understate what is about to happen.
                    plan.degrades_to_full_download = true;
                    plan.resident_bytes = storage.resident_bytes().min(facet_bytes);
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

    /// Start fetching `window` of `facet` on another thread, returning
    /// a handle to watch, cancel or join it. The background form of
    /// [`fetch`](Self::fetch): use that instead when the caller waits
    /// for the bytes anyway.
    ///
    /// The plan is computed, and the cache directory's free space
    /// checked, before this returns — planning needs the view, and a
    /// caller deserves the cost before committing — so the `Err` here
    /// is a planning or capacity failure. Fetch failures arrive through
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
        let plan = self.prefetch_plan_on(&*storage, facet, window)?;
        check_fallback(facet, &plan, fallback)?;
        self.cache_capacity().check(plan.bytes_to_fetch())?;

        let state = std::sync::Arc::new(PrefetchState::default());
        let worker_state = state.clone();
        let worker_plan = plan.clone();
        let name = facet.to_string();

        let thread = std::thread::Builder::new()
            .name(format!("prefetch:{name}"))
            .spawn(move || {
                use std::sync::atomic::Ordering;
                let outcome = crate::fetch::fetch_planned(
                    &*storage,
                    &worker_plan,
                    &|| worker_state.cancelled.load(Ordering::Acquire),
                    &mut |bytes, _| worker_state.bytes_fetched.store(bytes, Ordering::Relaxed),
                    &mut |ranges| worker_state.ranges_fetched.store(ranges, Ordering::Relaxed),
                );
                if let Err(e) = outcome {
                    log::warn!("prefetch of '{name}' failed: {e}");
                    *worker_state.error.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
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

    // -- Integrity --------------------------------------------------

    /// Whether the base is unit length, and on whose word: the declared
    /// `is_normalized` (the dataset's, an overlay's, or inferred from
    /// file names) checked against a 1000-vector sample, or the sample
    /// alone when nothing declares it. The sample of a remote base is
    /// recorded by its content identity and never re-taken; it fetches
    /// the chunks it touches when the base is not here yet.
    ///
    /// ```
    /// use vectordata::overlay::AttributeSource;
    /// use vectordata::{TestDataGroup, TestDataView};
    /// # use std::io::Write;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let mut f = std::fs::File::create(dir.path().join("base.fvecs")).unwrap();
    /// # for i in 0..10 { let a = i as f32 * 0.3; f.write_all(&2i32.to_le_bytes()).unwrap(); for x in [a.cos(), a.sin()] { f.write_all(&x.to_le_bytes()).unwrap(); } }
    /// # std::fs::write(dir.path().join("dataset.yaml"), "name: n\nprofiles:\n  default:\n    base_vectors: base.fvecs\n").unwrap();
    /// let view = TestDataGroup::load(dir.path().to_str().unwrap()).unwrap().profile("default").unwrap();
    /// let n = view.normalization().unwrap();
    /// assert!(n.value && !n.contradicted);
    /// assert!(matches!(n.source, AttributeSource::Measured { .. }), "nothing declares it: the sample says");
    /// assert_eq!(n.measured.sample, 10);
    /// ```
    fn normalization(&self) -> Result<crate::verify::Normalization>;

    /// Verify that the profile's facets belong together: the checks of
    /// [`crate::verify`] (V1–V9), every failure listed. A profile that
    /// passes is [`Ok`] with what was checked; one that does not is
    /// [`Error::Verification`] naming each check, what it found and the
    /// file. V5 reads every ground-truth index and reports its progress
    /// to `progress`.
    ///
    /// ```
    /// use vectordata::verify::{Check, NoProgress};
    /// use vectordata::{TestDataGroup, TestDataView};
    /// # use std::io::Write;
    /// # fn rows<T: Copy>(path: &std::path::Path, rows: &[&[T]], bytes: fn(T) -> Vec<u8>) {
    /// #     let mut f = std::fs::File::create(path).unwrap();
    /// #     for r in rows { f.write_all(&(r.len() as i32).to_le_bytes()).unwrap(); for &v in *r { f.write_all(&bytes(v)).unwrap(); } }
    /// # }
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # rows::<f32>(&dir.path().join("base.fvecs"), &[&[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]], |v| v.to_le_bytes().to_vec());
    /// # rows::<f32>(&dir.path().join("query.fvecs"), &[&[0.0, 0.9], &[0.9, 0.0]], |v| v.to_le_bytes().to_vec());
    /// # rows::<i32>(&dir.path().join("good.ivecs"), &[&[0, 2], &[1, 2]], |v| v.to_le_bytes().to_vec());
    /// # rows::<i32>(&dir.path().join("bad.ivecs"), &[&[0, 2], &[1, 3]], |v| v.to_le_bytes().to_vec());
    /// # std::fs::write(dir.path().join("dataset.yaml"), "name: v\nprofiles:\n  default:\n    base_vectors: base.fvecs\n    query_vectors: query.fvecs\n    neighbor_indices: good.ivecs\n  stale:\n    base_vectors: base.fvecs\n    query_vectors: query.fvecs\n    neighbor_indices: bad.ivecs\n")?;
    /// let group = TestDataGroup::load(dir.path().to_str().unwrap())?;
    /// let good = group.profile("default").unwrap().verify(&mut NoProgress)?;
    /// assert!(good.checked.contains(&Check::V5TruthIndices));
    ///
    /// // Index 3 names no vector of a 3-vector base: this truth was not computed against it.
    /// let err = group.profile("stale").unwrap().verify(&mut NoProgress).unwrap_err();
    /// assert!(err.to_string().contains("V5 neighbor_indices"), "{err}");
    /// assert!(err.to_string().contains("row 1 column 1 holds 3, which names no base vector of 3 base vectors"), "{err}");
    /// # Ok(()) }
    /// ```
    fn verify(&self, progress: &mut dyn crate::ProgressSink) -> Result<crate::verify::VerifyReport> {
        let report = self.verify_report(&crate::verify::VerifyOptions::default(), progress);
        if report.passed() {
            Ok(report)
        } else {
            Err(Error::Verification { failures: report.failures })
        }
    }

    /// [`verify`](Self::verify) as a report, failures and all, never an
    /// error: for a caller that lists every problem (`veks check`).
    fn verify_report(
        &self,
        options: &crate::verify::VerifyOptions,
        progress: &mut dyn crate::ProgressSink,
    ) -> crate::verify::VerifyReport;
}

/// The warning a whole-file fetch of the unindexed variable-length file
/// at `url`, `bytes` long, logs: what happened, and what its publisher
/// should add so that it does not happen again.
fn unindexed_vvec_warning(url: &str, bytes: u64) -> String {
    let file = url.rsplit('/').next().unwrap_or(url);
    let width = if bytes <= i32::MAX as u64 { "i32" } else { "i64" };
    format!(
        "{url} is a variable-length file published without an offset index, so all {bytes} bytes of it were \
         downloaded to open it. The dataset should be optimized upstream: ask its publisher to add the index \
         IDXFOR__{file}.{width} beside it (`vectordata push` and `veks publish` write it, as does a pipeline's \
         `generate vvec-index` step), so it opens without a download."
    )
}

/// The typed facet readers: every [`TestDataView`] method that serves
/// a vector facet, with the element type a parameter.
///
/// `view.base_vectors_as::<f32>()` is what `view.base_vectors()` was;
/// `view.base_vectors_as::<u8>()` reads a quantized base in place, and
/// `view.base_vectors_as::<f32>()` over a `bf16` base widens it. What
/// each pair of types means is one table,
/// [`ElementType::conversion_to`](crate::ElementType::conversion_to):
/// the file's own type reads in place (zero-copy where the storage is
/// mapped), a lossless conversion widens, a same-width sign change is
/// checked value by value, and anything else is refused at open with
/// [`IoError::ElementMismatch`].
/// The reader says which it is: [`VectorReader::conversion`].
///
/// Implemented for every view, so it works on an `Arc<dyn
/// TestDataView>` as on the concrete one; bring it into scope with
/// `use vectordata::TestDataViewExt`.
///
/// ```
/// use vectordata::{TestDataGroup, TestDataViewExt, VectorReader};
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let ds = dir.path().join("ds");
/// # std::fs::create_dir_all(&ds)?;
/// # let mut f = std::fs::File::create(ds.join("codes.bvecs"))?;
/// # for v in [[1u8, 2, 3], [4, 5, 6]] { f.write_all(&3i32.to_le_bytes())?; f.write_all(&v)?; }
/// # drop(f);
/// # std::fs::write(ds.join("dataset.yaml"), "name: codes\nprofiles:\n  default:\n    base_vectors: codes.bvecs\n")?;
/// let group = TestDataGroup::load(ds.to_str().unwrap())?;
/// let view = group.profile("default").unwrap();
///
/// // Native: the u8 codes as they are, borrowed from the mapping.
/// let codes = view.base_vectors_as::<u8>()?;
/// assert_eq!(codes.get_slice(1), Some(&[4u8, 5, 6][..]));
///
/// // Widening: the same facet as f32, converted on read.
/// let wide = view.base_vectors_as::<f32>()?;
/// assert_eq!(wide.get(1)?, vec![4.0, 5.0, 6.0]);
/// assert_eq!(wide.get_slice(1), None, "nothing of f32 exists to borrow");
/// # Ok(()) }
/// ```
pub trait TestDataViewExt: TestDataView {
    /// Any element facet as `T`; see the trait docs for what `T` may be.
    ///
    /// A `bf16` facet at its own type is zero-copy; as `f32` it widens
    /// exactly, at the cost of one conversion per element read:
    ///
    /// ```
    /// use vectordata::{TestDataGroup, TestDataViewExt, VectorReader, ElementConversion};
    /// use vectordata::typed_access::bf16;
    /// # use std::io::Write;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let ds = dir.path().join("ds");
    /// # std::fs::create_dir_all(&ds)?;
    /// # let mut f = std::fs::File::create(ds.join("base.bf16vecs"))?;
    /// # for v in [[1.5f32, -2.0], [0.25, 8.0]] {
    /// #     f.write_all(&2i32.to_le_bytes())?;
    /// #     for x in v { f.write_all(&bf16::from_f32(x).to_le_bytes())?; }
    /// # }
    /// # drop(f);
    /// # std::fs::write(ds.join("dataset.yaml"), "name: brains\nprofiles:\n  default:\n    base_vectors: base.bf16vecs\n")?;
    /// let view = TestDataGroup::load(ds.to_str().unwrap())?.profile("default").unwrap();
    ///
    /// // Native: the bf16 values borrowed from the mapping, no conversion.
    /// let native = view.facet_as::<bf16>("base_vectors")?;
    /// assert_eq!(native.conversion(), ElementConversion::Native);
    /// assert_eq!(native.get_slice(1).map(|r| r[1].to_f32()), Some(8.0));
    ///
    /// // Widening: every bf16 is an f32, so the values are exact; each
    /// // read converts, and there is no slice of f32 to borrow.
    /// let wide = view.facet_as::<f32>("base_vectors")?;
    /// assert_eq!(wide.conversion(), ElementConversion::Widening);
    /// assert_eq!(wide.get(0)?, vec![1.5, -2.0]);
    /// assert_eq!(wide.get_slice(0), None);
    /// # Ok(()) }
    /// ```
    ///
    /// A column facet (`.u8`, `.f32`, …) is one slice at its own type —
    /// the zero-copy path for quantization parameters and labels:
    ///
    /// ```
    /// use vectordata::{TestDataGroup, TestDataViewExt, VectorReader};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let ds = dir.path().join("ds");
    /// # std::fs::create_dir_all(&ds)?;
    /// # std::fs::write(ds.join("labels.u8"), [3u8, 1, 4, 1, 5])?;
    /// # std::fs::write(ds.join("dataset.yaml"), "name: labels\nprofiles:\n  default:\n    metadata_content: labels.u8\n")?;
    /// let view = TestDataGroup::load(ds.to_str().unwrap())?.profile("default").unwrap();
    /// let labels = view.facet_as::<u8>("metadata_content")?;
    /// assert_eq!(labels.dim(), 1);
    /// let all: &[u8] = labels.as_slice().expect("a local column maps whole");
    /// assert_eq!(all, &[3, 1, 4, 1, 5]);
    /// # Ok(()) }
    /// ```
    fn facet_as<T: Element>(&self, name: &str) -> Result<Arc<dyn VectorReader<T>>> {
        Ok(self.open_facet_as_type(name, T::TYPE)?.into_typed::<T>()?)
    }

    /// The base (database) vectors as `T`.
    ///
    /// At the file's own type, a profile window reads in place through
    /// [`read_into`](VectorReader::read_into) into a buffer you keep —
    /// the bulk path for a chunked build:
    ///
    /// ```
    /// use vectordata::{TestDataGroup, TestDataViewExt, VectorReader, ElementConversion};
    /// # use std::io::Write;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let ds = dir.path().join("ds");
    /// # std::fs::create_dir_all(&ds)?;
    /// # let mut f = std::fs::File::create(ds.join("base.fvecs"))?;
    /// # for i in 0..6i32 { f.write_all(&2i32.to_le_bytes())?; for d in 0..2 { f.write_all(&((i * 2 + d) as f32).to_le_bytes())?; } }
    /// # drop(f);
    /// // The profile reads records 1..5 of the six in the file.
    /// # std::fs::write(ds.join("dataset.yaml"), "name: window\nprofiles:\n  default:\n    base_vectors: base.fvecs[1..5)\n")?;
    /// let view = TestDataGroup::load(ds.to_str().unwrap())?.profile("default").unwrap();
    /// let base = view.base_vectors_as::<f32>()?;
    /// assert_eq!(base.conversion(), ElementConversion::Native);
    /// assert_eq!(base.count(), 4);
    ///
    /// let mut block = vec![0f32; base.dim() * 2];      // two records at a time
    /// let mut start = 0;
    /// while start < base.count() {
    ///     let got = base.read_into(start, &mut block)?;
    ///     if start == 0 { assert_eq!(&block[..4], &[2.0, 3.0, 4.0, 5.0]); }
    ///     start += got;
    /// }
    /// assert_eq!(start, 4);
    /// # Ok(()) }
    /// ```
    fn base_vectors_as<T: Element>(&self) -> Result<Arc<dyn VectorReader<T>>> {
        self.facet_as("base_vectors")
    }

    /// The query vectors as `T`.
    fn query_vectors_as<T: Element>(&self) -> Result<Arc<dyn VectorReader<T>>> {
        self.facet_as("query_vectors")
    }

    /// The neighbor indices (ground truth) as `T`.
    ///
    /// Read at their own type, i32, and checked against the base count
    /// the way a consumer should check any ground truth it is handed:
    ///
    /// ```
    /// use vectordata::{TestDataGroup, TestDataViewExt, VectorReader};
    /// # use std::io::Write;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let ds = dir.path().join("ds");
    /// # std::fs::create_dir_all(&ds)?;
    /// # let mut f = std::fs::File::create(ds.join("base.fvecs"))?;
    /// # for i in 0..3i32 { f.write_all(&1i32.to_le_bytes())?; f.write_all(&(i as f32).to_le_bytes())?; }
    /// # drop(f);
    /// # let mut g = std::fs::File::create(ds.join("gt.ivecs"))?;
    /// # for row in [[0i32, 1], [1, 2]] { g.write_all(&2i32.to_le_bytes())?; for id in row { g.write_all(&id.to_le_bytes())?; } }
    /// # drop(g);
    /// # std::fs::write(ds.join("dataset.yaml"), "name: gt\nprofiles:\n  default:\n    base_vectors: base.fvecs\n    neighbor_indices: gt.ivecs\n")?;
    /// let view = TestDataGroup::load(ds.to_str().unwrap())?.profile("default").unwrap();
    /// let base = view.base_vectors_as::<f32>()?;
    /// let gt = view.neighbor_indices_as::<i32>()?;
    /// for q in 0..gt.count() {
    ///     let ids = gt.get_slice(q).expect("a local i32 file borrows");
    ///     assert!(ids.iter().all(|&id| (id as usize) < base.count()), "query {q} names a base vector");
    /// }
    /// # Ok(()) }
    /// ```
    fn neighbor_indices_as<T: Element>(&self) -> Result<Arc<dyn VectorReader<T>>> {
        self.facet_as("neighbor_indices")
    }

    /// The neighbor distances (ground truth) as `T`.
    ///
    /// ```
    /// use vectordata::{ElementConversion, TestDataGroup, TestDataViewExt, VectorReader};
    /// # use std::io::Write;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let ds = dir.path().join("ds");
    /// # std::fs::create_dir_all(&ds)?;
    /// # let mut f = std::fs::File::create(ds.join("dist.dvecs"))?;
    /// # for v in [[0.0f64, 1e-300]] { f.write_all(&2i32.to_le_bytes())?; for x in v { f.write_all(&x.to_le_bytes())?; } }
    /// # drop(f);
    /// # std::fs::write(ds.join("dataset.yaml"), "name: d\nprofiles:\n  default:\n    neighbor_distances: dist.dvecs\n")?;
    /// let view = TestDataGroup::load(ds.to_str().unwrap())?.profile("default").unwrap();
    /// let d = view.neighbor_distances_as::<f64>()?;                     // the file's type: zero-copy
    /// assert_eq!((d.conversion(), d.get(0)?), (ElementConversion::Native, vec![0.0, 1e-300]));
    /// assert!(view.neighbor_distances_as::<f32>().is_err(), "f64 never narrows: read it as f64");
    /// # Ok(()) }
    /// ```
    fn neighbor_distances_as<T: Element>(&self) -> Result<Arc<dyn VectorReader<T>>> {
        self.facet_as("neighbor_distances")
    }

    /// The pre-filter KNN ground-truth indices (F) as `T`.
    fn prefiltered_neighbor_indices_as<T: Element>(&self) -> Result<Arc<dyn VectorReader<T>>> {
        self.facet_as("prefiltered_neighbor_indices")
    }

    /// The pre-filter KNN ground-truth distances (F) as `T`.
    fn prefiltered_neighbor_distances_as<T: Element>(&self) -> Result<Arc<dyn VectorReader<T>>> {
        self.facet_as("prefiltered_neighbor_distances")
    }

    /// The post-filter KNN ground-truth indices (E) as `T`.
    fn postfiltered_neighbor_indices_as<T: Element>(&self) -> Result<Arc<dyn VectorReader<T>>> {
        self.facet_as("postfiltered_neighbor_indices")
    }

    /// The post-filter KNN ground-truth distances (E) as `T`.
    fn postfiltered_neighbor_distances_as<T: Element>(&self) -> Result<Arc<dyn VectorReader<T>>> {
        self.facet_as("postfiltered_neighbor_distances")
    }

    /// The predicate-result indices (the `R` facet) as `T`, which must
    /// be the type the file holds: a variable-length facet is never
    /// converted.
    ///
    /// ```
    /// use vectordata::{Error, ElementType, TestDataGroup, TestDataViewExt};
    /// use vectordata::io::IoError;
    /// # use std::io::Write;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let ds = dir.path().join("ds");
    /// # std::fs::create_dir_all(&ds)?;
    /// # let mut f = std::fs::File::create(ds.join("matches.i16vvecs"))?;
    /// # for v in [vec![3i16, 1, 2], vec![], vec![-1]] { f.write_all(&(v.len() as i32).to_le_bytes())?; for x in v { f.write_all(&x.to_le_bytes())?; } }
    /// # drop(f);
    /// # std::fs::write(ds.join("dataset.yaml"), "name: r\nprofiles:\n  default:\n    metadata_results: matches.i16vvecs\n")?;
    /// let view = TestDataGroup::load(ds.to_str().unwrap())?.profile("default").unwrap();
    /// let r = view.metadata_results_as::<i16>()?;                       // the file's type: zero-copy index, decoded records
    /// assert_eq!((r.count(), r.dim_at(1)?), (3, 0));
    /// assert_eq!(r.get(0)?, vec![3i16, 1, 2]);
    /// match view.metadata_results_as::<i32>() {                         // wider, but a vvec is never converted
    ///     Err(Error::VectorIo(IoError::ElementMismatch { file, requested, .. })) => assert_eq!((file, requested), (ElementType::I16, ElementType::I32)),
    ///     other => panic!("{:?}", other.map(|r| r.count())),
    /// }
    /// # Ok(()) }
    /// ```
    fn metadata_results_as<T: Element>(&self) -> Result<Arc<dyn VvecReader<T>>> {
        Ok(self.open_vvec_facet_native("metadata_results")?.into_typed::<T>()?)
    }

    /// The variable-length facet `name` as `T`, which must be the type
    /// the file holds, as [`metadata_results_as`](Self::metadata_results_as)
    /// is for `metadata_results`.
    ///
    /// A remote file whose source publishes no `IDXFOR__` offset index is
    /// refused ([`IoError::OffsetIndexUnavailable`]): only a download of
    /// every byte would find its records. To accept that download, open
    /// it with [`vvec_facet_fetching_as`](Self::vvec_facet_fetching_as).
    fn vvec_facet_as<T: Element>(&self, name: &str) -> Result<Arc<dyn VvecReader<T>>> {
        Ok(self.open_vvec_facet_native(name)?.into_typed::<T>()?)
    }

    /// [`vvec_facet_as`](Self::vvec_facet_as), fetching the facet whole
    /// first when its source publishes no offset index: the one case a
    /// lazy open refuses. `progress` hears the fetch, as
    /// [`fetch`](TestDataView::fetch)'s does; an indexed or already
    /// fetched facet opens as `vvec_facet_as` opens it, and nothing is
    /// fetched.
    ///
    /// A whole-file download to open a variable-length file is a gap in
    /// the published dataset, so it is logged as a warning naming the
    /// file and the index its publisher should add: `vectordata push`
    /// and `veks publish` write it, as does a pipeline's
    /// `generate vvec-index` step.
    ///
    /// ```no_run
    /// use vectordata::{TestDataViewExt, fetch::TextMeter};
    /// # fn demo(view: &dyn vectordata::TestDataView) -> vectordata::Result<()> {
    /// let results = view.vvec_facet_fetching_as::<i32>("metadata_results", &mut TextMeter::stderr("Fetch"))?;
    /// # let _ = results;
    /// # Ok(()) }
    /// ```
    fn vvec_facet_fetching_as<T: Element>(
        &self,
        name: &str,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> Result<Arc<dyn VvecReader<T>>> {
        match self.open_vvec_facet_native(name) {
            Err(Error::VectorIo(IoError::OffsetIndexUnavailable { url, bytes })) => {
                log::warn!("{}", unindexed_vvec_warning(&url, bytes));
                self.fetch(&crate::fetch::FetchRequest::facets([name]).allow_whole_facet(), progress)?;
                self.vvec_facet_as(name)
            }
            opened => Ok(opened?.into_typed::<T>()?),
        }
    }

    /// [`metadata_results_as`](Self::metadata_results_as), fetching the
    /// facet whole first when its source publishes no offset index. See
    /// [`vvec_facet_fetching_as`](Self::vvec_facet_fetching_as).
    fn metadata_results_fetching_as<T: Element>(
        &self,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> Result<Arc<dyn VvecReader<T>>> {
        self.vvec_facet_fetching_as("metadata_results", progress)
    }

    /// A quantized base as one thing: the codes at their own type `C`,
    /// the three parameter columns, and the scheme, checked against
    /// each other. See [`QuantizedBase`].
    ///
    /// `C` is the type the `quantization` attribute's `codes` names —
    /// here signed, symmetric codes:
    ///
    /// ```
    /// use vectordata::{ElementType, TestDataGroup, TestDataViewExt, VectorReader};
    /// use vectordata::dataset::QuantScheme;
    /// # use std::io::Write;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = tempfile::tempdir()?;
    /// # let ds = dir.path().join("ds");
    /// # std::fs::create_dir_all(&ds)?;
    /// # let mut f = std::fs::File::create(ds.join("codes.i8vecs"))?;
    /// # for v in [[-128i8, 0, 127], [1, -1, 64]] { f.write_all(&3i32.to_le_bytes())?; for x in v { f.write_all(&x.to_le_bytes())?; } }
    /// # drop(f);
    /// # std::fs::write(ds.join("quant_scale.f32"), [0.5f32, 0.25].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    /// # std::fs::write(ds.join("quant_zero_point.i32"), [0i32, 0].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    /// # std::fs::write(ds.join("quant_sum.i32"), [-1i32, 64].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    /// # std::fs::write(ds.join("dataset.yaml"), "name: q\nattributes:\n  quantization:\n    scheme: symmetric\n    granularity: per-vector\n    codes: i8\n    range: [-127, 127]\nprofiles:\n  default:\n    base_vectors: codes.i8vecs\n    quant_scale: quant_scale.f32\n    quant_zero_point: quant_zero_point.i32\n    quant_sum: quant_sum.i32\n")?;
    /// let view = TestDataGroup::load(ds.to_str().unwrap())?.profile("default").unwrap();
    /// let q = view.quantized_base::<i8>()?;
    /// assert_eq!(q.codes.get_slice(0), Some(&[-128i8, 0, 127][..]));    // the codes, zero-copy
    /// assert_eq!(q.scale.as_slice(), Some(&[0.5f32, 0.25][..]));        // each column, one slice
    /// assert_eq!((q.scheme.scheme, q.scheme.codes), (QuantScheme::Symmetric, ElementType::I8));
    /// assert!(view.quantized_base::<u8>().is_err(), "the scheme names i8; no other type opens the codes");
    /// # Ok(()) }
    /// ```
    fn quantized_base<C: Element>(&self) -> Result<QuantizedBase<C>> {
        QuantizedBase::open(self)
    }
}

/// A pre-quantized base resolved as one dataset: the codes, the
/// per-vector parameters the dequantized distance needs, and the scheme
/// that says how to apply them.
///
/// Opening checks what a consumer would otherwise have to: that the
/// `quantization` attribute names the type the codes are read at, and
/// that every column holds exactly one value per base vector
/// ([`Error::FacetCardinality`] otherwise). On a local single-file
/// dataset each column's [`as_slice`](VectorReader::as_slice) is `Some`,
/// so the three reach a kernel as borrowed slices with no copy.
///
/// ```
/// use vectordata::{TestDataGroup, TestDataViewExt, VectorReader};
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let ds = dir.path().join("ds");
/// # std::fs::create_dir_all(&ds)?;
/// # let mut f = std::fs::File::create(ds.join("base_vectors.bvecs"))?;
/// # for v in [[10u8, 20, 30], [40, 50, 60]] { f.write_all(&3i32.to_le_bytes())?; f.write_all(&v)?; }
/// # drop(f);
/// # std::fs::write(ds.join("quant_scale.f32"), [0.5f32, 0.25].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
/// # std::fs::write(ds.join("quant_zero_point.i32"), [128i32, 100].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
/// # std::fs::write(ds.join("quant_sum.i32"), [60i32, 150].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
/// # std::fs::write(ds.join("dataset.yaml"), "name: q\nattributes:\n  quantization:\n    scheme: affine\n    granularity: per-vector\n    codes: u8\n    range: [0, 255]\nprofiles:\n  default:\n    base_vectors: base_vectors.bvecs\n    quant_scale: quant_scale.f32\n    quant_zero_point: quant_zero_point.i32\n    quant_sum: quant_sum.i32\n")?;
/// let view = TestDataGroup::load(ds.to_str().unwrap())?.profile("default").unwrap();
/// let q = view.quantized_base::<u8>()?;
/// assert_eq!(q.codes.get_slice(1), Some(&[40u8, 50, 60][..]));
/// assert_eq!(q.scale.as_slice(), Some(&[0.5f32, 0.25][..]));
/// assert_eq!(q.zero_point.as_slice(), Some(&[128i32, 100][..]));
/// assert_eq!(q.sum.as_slice(), Some(&[60i32, 150][..]));
/// assert_eq!(q.scheme.range, [0, 255]);
/// # Ok(()) }
/// ```
pub struct QuantizedBase<C: Element> {
    /// The codes: `base_vectors` at its own type.
    pub codes: Arc<dyn VectorReader<C>>,
    /// Per-vector scale (`quant_scale`).
    pub scale: Arc<dyn VectorReader<f32>>,
    /// Per-vector zero point (`quant_zero_point`).
    pub zero_point: Arc<dyn VectorReader<i32>>,
    /// Per-vector sum of the codes (`quant_sum`).
    pub sum: Arc<dyn VectorReader<i32>>,
    /// The scheme, from the `quantization` attribute.
    pub scheme: crate::dataset::Quantization,
}

impl<C: Element> QuantizedBase<C> {
    fn open<V: TestDataView + ?Sized>(view: &V) -> Result<Self> {
        let scheme = match view.quantization() {
            Some(Ok(q)) => q,
            Some(Err(e)) => return Err(Error::Other(format!("the quantization attribute does not parse: {e}"))),
            None => {
                return Err(Error::Other(
                    "no quantization attribute: a quantized base declares its scheme at the dataset or \
                     profile level"
                        .into(),
                ));
            }
        };
        if scheme.codes != C::TYPE {
            return Err(IoError::ElementMismatch {
                file: scheme.codes,
                requested: C::TYPE,
                reason: "the quantization attribute names the type the codes are read at",
            }
            .into());
        }
        let codes = view.base_vectors_as::<C>()?;
        let expected = codes.count();
        let column = |name: &str, count: usize| -> Result<()> {
            if count == expected {
                Ok(())
            } else {
                Err(Error::FacetCardinality {
                    facet: name.to_string(),
                    against: "base_vectors".to_string(),
                    expected,
                    found: count,
                })
            }
        };
        let scale = view.facet_as::<f32>("quant_scale")?;
        column("quant_scale", scale.count())?;
        let zero_point = view.facet_as::<i32>("quant_zero_point")?;
        column("quant_zero_point", zero_point.count())?;
        let sum = view.facet_as::<i32>("quant_sum")?;
        column("quant_sum", sum.count())?;
        Ok(Self { codes, scale, zero_point, sum, scheme })
    }
}

impl<V: TestDataView + ?Sized> TestDataViewExt for V {}

/// A record range resolved to bytes, and what resolving it cost:
/// what [`FacetBytes::map_records`](crate::FacetBytes::map_records)
/// answers for a format whose records map to bytes from the file alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappedRange {
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

/// Live cache fill statistics for a single facet's underlying storage.
/// Returned by [`FacetBytes::cache_stats`](crate::FacetBytes::cache_stats). `None` when the storage
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
/// Returned by [`FacetBytes::range_fill`](crate::FacetBytes::range_fill). The unit of fetch is a
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
    /// Bytes of the resident chunks, each at its real length — the
    /// file's last chunk holds only what is left of the file.
    pub resident_bytes: u64,
    /// Byte range the fetch actually spans once widened to chunk
    /// boundaries — always a superset of what was asked for.
    pub aligned_start: u64,
    /// End of that chunk-aligned span (exclusive).
    pub aligned_end: u64,
}

impl RangeFill {
    /// Chunks that still have to be fetched.
    pub fn chunks_to_fetch(&self) -> u32 {
        self.chunks.saturating_sub(self.chunks_resident)
    }

    /// Bytes that will cross the network, at chunk granularity: the
    /// chunk-aligned span less what is resident. A span ending at the
    /// end of the file ends with a short chunk, charged at its length.
    pub fn bytes_to_fetch(&self) -> u64 {
        self.aligned_end.saturating_sub(self.aligned_start).saturating_sub(self.resident_bytes)
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
pub fn coalesce_ranges(
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
    /// For a plan that degrades to the whole facet, bytes of it already
    /// in the local cache — what an interrupted whole-facet fetch left
    /// behind, which a retry need not fetch again. Zero otherwise.
    pub resident_bytes: u64,
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
            return self.facet_bytes.saturating_sub(self.resident_bytes);
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
pub fn check_fallback(facet: &str, plan: &PrefetchPlan, fallback: WholeFacetFallback) -> Result<()> {
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

    /// Whether [`cancel`](Self::cancel) has been called.
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
        if let Some(e) = self.state.error.lock().unwrap_or_else(|p| p.into_inner()).take() {
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
