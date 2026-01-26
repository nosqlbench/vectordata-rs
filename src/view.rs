//! Views for accessing dataset components.
//!
//! This module defines traits and structs for accessing the different parts of a
//! vector dataset (base vectors, query vectors, ground truth) uniformly, regardless
//! of the underlying data source (local file or HTTP).

use crate::group::DataSource;
use crate::io::{HttpVectorReader, MmapVectorReader, VectorReader};
use crate::model::{FacetConfig, ProfileConfig};
use crate::{Error, Result};
use std::path::PathBuf;
use std::sync::Arc;
use url::Url;

/// Interface for accessing the components of a dataset profile.
///
/// Each method returns a thread-safe `VectorReader` for the corresponding component.
pub trait TestDataView: Send + Sync {
    /// Returns a reader for the base (database) vectors.
    fn base_vectors(&self) -> Result<Arc<dyn VectorReader<f32>>>;
    /// Returns a reader for the query vectors.
    fn query_vectors(&self) -> Result<Arc<dyn VectorReader<f32>>>;
    /// Returns a reader for the neighbor indices (ground truth).
    fn neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>>;
    /// Returns a reader for the neighbor distances (ground truth).
    fn neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>>;
}

/// A generic implementation of `TestDataView`.
///
/// This struct holds the configuration for a profile and the data source location,
/// creating the appropriate `VectorReader` (Mmap or Http) on demand.
#[derive(Debug)]
pub struct GenericTestDataView {
    source: DataSource,
    config: ProfileConfig,
}

impl GenericTestDataView {
    /// Creates a new `GenericTestDataView`.
    pub fn new(source: DataSource, config: ProfileConfig) -> Self {
        Self { source, config }
    }


    fn resolve_resource(&self, facet: &FacetConfig) -> Result<ResourceLocation> {
        let source_str = facet.source();
        match &self.source {
            DataSource::FileSystem(base_path) => {
                let path = base_path.join(source_str);
                Ok(ResourceLocation::FileSystem(path))
            },
            DataSource::Http(base_url) => {
                let url = base_url.join(source_str)?;
                Ok(ResourceLocation::Http(url))
            }
        }
    }
    
    fn open_fvec(&self, facet_opt: Option<&FacetConfig>, name: &str) -> Result<Arc<dyn VectorReader<f32>>> {
        let facet = facet_opt.ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        let location = self.resolve_resource(facet)?;
        
        match location {
            ResourceLocation::FileSystem(path) => {
                let reader = MmapVectorReader::open_fvec(&path)?;
                Ok(Arc::new(reader))
            },
            ResourceLocation::Http(url) => {
                let reader = HttpVectorReader::open_fvec(url.clone())?;
                Ok(Arc::new(reader))
            }
        }
    }

    fn open_ivec(&self, facet_opt: Option<&FacetConfig>, name: &str) -> Result<Arc<dyn VectorReader<i32>>> {
        let facet = facet_opt.ok_or_else(|| Error::MissingFacet(name.to_string()))?;
        let location = self.resolve_resource(facet)?;
        
        match location {
            ResourceLocation::FileSystem(path) => {
                let reader = MmapVectorReader::open_ivec(&path)?;
                Ok(Arc::new(reader))
            },
            ResourceLocation::Http(url) => {
                let reader = HttpVectorReader::open_ivec(url.clone())?;
                Ok(Arc::new(reader))
            }
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
}
