// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Profile views for accessing dataset components.
//!
//! Defines the [`TestDataView`] trait and [`GenericTestDataView`] implementation
//! for uniform access to base vectors, query vectors, ground-truth neighbors,
//! and metadata facets regardless of backing storage (local mmap or HTTP).
//!
//! The [`FacetDescriptor`] type supports discover-then-load patterns by
//! describing available facets without materializing data.

use crate::group::DataSource;
use crate::io::{HttpVectorReader, MmapVectorReader, VectorReader};
use crate::model::{FacetConfig, ProfileConfig};
use crate::{Error, Result};
use crate::dataset::facet::StandardFacet;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use url::Url;

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

    // -- Metadata facets (config access) --

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

    // -- Dataset metadata --

    /// Returns the distance function name if declared in attributes.
    fn distance_function(&self) -> Option<String>;
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

    fn open_fvec(
        &self,
        facet_opt: Option<&FacetConfig>,
        name: &str,
    ) -> Result<Arc<dyn VectorReader<f32>>> {
        let facet = facet_opt.ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        let location = self.resolve_resource(facet)?;

        match location {
            ResourceLocation::FileSystem(path) => {
                let reader = MmapVectorReader::open_fvec(&path)?;
                Ok(Arc::new(reader))
            }
            ResourceLocation::Http(url) => {
                let reader = HttpVectorReader::open_fvec(url.clone())?;
                Ok(Arc::new(reader))
            }
        }
    }

    fn open_ivec(
        &self,
        facet_opt: Option<&FacetConfig>,
        name: &str,
    ) -> Result<Arc<dyn VectorReader<i32>>> {
        let facet = facet_opt.ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        let location = self.resolve_resource(facet)?;

        match location {
            ResourceLocation::FileSystem(path) => {
                let reader = MmapVectorReader::open_ivec(&path)?;
                Ok(Arc::new(reader))
            }
            ResourceLocation::Http(url) => {
                let reader = HttpVectorReader::open_ivec(url.clone())?;
                Ok(Arc::new(reader))
            }
        }
    }

    /// Open a facet as an f32 reader regardless of the underlying format.
    ///
    /// Determines the format from the file extension.
    fn open_facet_as_fvec(&self, facet: &FacetConfig) -> Result<Arc<dyn VectorReader<f32>>> {
        let source = facet.source();
        let location = self.resolve_resource(facet)?;

        // Determine format from extension
        let ext = source.rsplit('.').next().unwrap_or("");

        match (ext, location) {
            ("fvec", ResourceLocation::FileSystem(path)) => {
                Ok(Arc::new(MmapVectorReader::open_fvec(&path)?))
            }
            ("fvec", ResourceLocation::Http(url)) => {
                Ok(Arc::new(HttpVectorReader::open_fvec(url)?))
            }
            _ => Err(Error::Other(format!(
                "unsupported format '{}' for generic facet access",
                ext
            ))),
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
        let path = self.resolve_facet_path(name)
            .map_err(|e| crate::typed_access::TypedAccessError::Io(e.to_string()))?;
        crate::typed_access::TypedReader::<T>::open(&path)
    }

    /// Resolve a facet name to a filesystem path.
    fn resolve_facet_path(&self, name: &str) -> Result<PathBuf> {
        let facet = self.facet_config_by_name(name)
            .ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        match self.resolve_resource(facet)? {
            ResourceLocation::FileSystem(path) => Ok(path),
            ResourceLocation::Http(_) => Err(Error::Other(
                format!("typed access not supported for HTTP facets ({})", name))),
        }
    }
}

enum ResourceLocation {
    FileSystem(PathBuf),
    Http(Url),
}

impl TestDataView for GenericTestDataView {
    fn base_vectors(&self) -> Result<Arc<dyn VectorReader<f32>>> {
        self.open_fvec(self.config.base_vectors.as_ref(), "base_vectors")
    }

    fn query_vectors(&self) -> Result<Arc<dyn VectorReader<f32>>> {
        self.open_fvec(self.config.query_vectors.as_ref(), "query_vectors")
    }

    fn neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>> {
        self.open_ivec(self.config.neighbor_indices.as_ref(), "neighbor_indices")
    }

    fn neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>> {
        self.open_fvec(self.config.neighbor_distances.as_ref(), "neighbor_distances")
    }

    fn filtered_neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>> {
        self.open_ivec(
            self.config.filtered_neighbor_indices.as_ref(),
            "filtered_neighbor_indices",
        )
    }

    fn filtered_neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>> {
        self.open_fvec(
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

    fn facet(&self, name: &str) -> Result<Arc<dyn VectorReader<f32>>> {
        // Try standard facets first
        match name {
            "base_vectors" => return self.base_vectors(),
            "query_vectors" => return self.query_vectors(),
            "neighbor_distances" => return self.neighbor_distances(),
            "filtered_neighbor_distances" => return self.filtered_neighbor_distances(),
            _ => {}
        }

        // For other facets, try to find a FacetConfig and open generically
        let facet_config = match name {
            "neighbor_indices" => self.config.neighbor_indices.as_ref(),
            "filtered_neighbor_indices" => self.config.filtered_neighbor_indices.as_ref(),
            "metadata_content" => self.config.metadata_content.as_ref(),
            "metadata_predicates" => self.config.metadata_predicates.as_ref(),
            "predicate_results" => self.config.predicate_results.as_ref(),
            "metadata_layout" => self.config.metadata_layout.as_ref(),
            _ => None,
        };

        match facet_config {
            Some(fc) => self.open_facet_as_fvec(fc),
            None => Err(Error::MissingFacet(name.to_string())),
        }
    }

    fn facet_element_type(&self, name: &str) -> Result<crate::typed_access::ElementType> {
        let path = self.resolve_facet_path(name)?;
        crate::typed_access::ElementType::from_path(&path)
            .map_err(Error::Other)
    }

    fn distance_function(&self) -> Option<String> {
        self.attributes
            .get("distance_function")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
}
