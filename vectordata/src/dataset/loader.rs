// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Unified dataset loader for the data access layer.
//!
//! [`DatasetLoader`] is the primary entry point for all dataset access.
//! It resolves specifiers (local paths, URLs, `dataset:profile` catalog
//! references) to typed [`DatasetView`] implementations that provide
//! transparent access regardless of data location.

use std::io;
use std::path::{Path, PathBuf};

use crate::dataset::remote::{LocalDatasetView, RemoteDatasetView};
use crate::dataset::view::TypedVectorView;

/// Unified dataset loader.
///
/// Resolves dataset specifiers to typed views. Handles local files,
/// URLs, and catalog `dataset:profile` references.
///
/// # Examples
///
/// ```
/// use std::path::PathBuf;
/// use vectordata::dataset::loader::DatasetLoader;
///
/// let loader = DatasetLoader::new(PathBuf::from("/tmp/cache"));
/// // Load a local file (no catalog needed):
/// // let view = loader.load("vectors.fvec", &[])?;
/// // Load from catalog (requires catalog entries):
/// // let view = loader.load("sift-128:default", &catalog)?;
/// ```
pub struct DatasetLoader {
    cache_dir: PathBuf,
}

/// Result of loading a dataset — either local or remote.
pub enum LoadedDataset {
    Local(LocalDatasetView),
    Remote(RemoteDatasetView),
}

impl LoadedDataset {
    /// Get base vectors view.
    pub fn base_vectors(&self) -> Option<&dyn TypedVectorView> {
        match self {
            LoadedDataset::Local(v) => v.base_vectors(),
            LoadedDataset::Remote(v) => v.base_vectors(),
        }
    }

    /// Get query vectors view.
    pub fn query_vectors(&self) -> Option<&dyn TypedVectorView> {
        match self {
            LoadedDataset::Local(v) => v.query_vectors(),
            LoadedDataset::Remote(v) => v.query_vectors(),
        }
    }

    /// Get neighbor indices view.
    pub fn neighbor_indices(&self) -> Option<&dyn TypedVectorView> {
        match self {
            LoadedDataset::Local(_) => None, // TODO: implement for local
            LoadedDataset::Remote(v) => v.neighbor_indices(),
        }
    }

    /// Get neighbor distances view.
    pub fn neighbor_distances(&self) -> Option<&dyn TypedVectorView> {
        match self {
            LoadedDataset::Local(_) => None, // TODO: implement for local
            LoadedDataset::Remote(v) => v.neighbor_distances(),
        }
    }
}

impl DatasetLoader {
    /// Create a loader with an explicit cache directory.
    pub fn new(cache_dir: PathBuf) -> Self {
        DatasetLoader { cache_dir }
    }

    /// Load a dataset from a specifier.
    ///
    /// The specifier can be:
    /// - A local file path (fvec, mvec, ivec, dvec)
    /// - A local directory containing dataset.yaml
    /// - A `dataset:profile` catalog reference
    /// - A `dataset` name (uses "default" profile)
    ///
    /// For catalog references, the `catalog_entries` must be provided
    /// (loaded by the caller from configured catalog sources).
    pub fn load(
        &self,
        spec: &str,
        catalog_entries: &[crate::dataset::catalog::CatalogEntry],
    ) -> io::Result<LoadedDataset> {
        // 1. Check if it's a local file
        let as_path = Path::new(spec);
        if as_path.exists() {
            if as_path.is_file() {
                // Single vector file — wrap in a minimal view
                return self.load_single_file(as_path);
            }
            if as_path.is_dir() {
                // Directory with dataset.yaml
                return Ok(LoadedDataset::Local(
                    LocalDatasetView::open(as_path, "default")?
                ));
            }
        }

        // 2. Check if it contains a path separator (but doesn't exist)
        if spec.contains('/') || spec.contains('.') {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("file not found: {}", spec),
            ));
        }

        // 3. Parse as dataset:profile catalog reference
        let (dataset_name, profile_name) = if let Some(pos) = spec.find(':') {
            (&spec[..pos], &spec[pos + 1..])
        } else {
            (spec, "default")
        };

        // 4. Look up in catalog
        let entry = catalog_entries.iter()
            .find(|e| e.name.eq_ignore_ascii_case(dataset_name))
            .ok_or_else(|| {
                let mut msg = format!("dataset '{}' not found in catalog", dataset_name);
                // Suggest similar names
                let similar: Vec<&str> = catalog_entries.iter()
                    .filter(|e| e.name.to_lowercase().contains(&dataset_name.to_lowercase())
                        || dataset_name.to_lowercase().contains(&e.name.to_lowercase()))
                    .map(|e| e.name.as_str())
                    .take(5)
                    .collect();
                if !similar.is_empty() {
                    msg.push_str(&format!(". Similar: {}", similar.join(", ")));
                }
                msg.push_str(&format!(" ({} entries loaded)", catalog_entries.len()));
                io::Error::new(io::ErrorKind::NotFound, msg)
            })?;

        Ok(LoadedDataset::Remote(
            RemoteDatasetView::open(entry, profile_name, &self.cache_dir)?
        ))
    }

    /// Load a single vector file as a minimal dataset view.
    fn load_single_file(&self, path: &Path) -> io::Result<LoadedDataset> {
        use crate::dataset::view::LocalVectorView;

        let view = LocalVectorView::open(path, None)?;
        Ok(LoadedDataset::Local(LocalDatasetView::from_single_view(
            "base_vectors",
            view,
        )))
    }
}

impl LocalDatasetView {
    /// Create a minimal view from a single facet.
    pub fn from_single_view(facet: &str, view: crate::dataset::view::LocalVectorView) -> Self {
        let mut result = LocalDatasetView {
            base_vectors: None,
            query_vectors: None,
            neighbor_indices: None,
            neighbor_distances: None,
        };
        match facet {
            "base_vectors" => result.base_vectors = Some(view),
            "query_vectors" => result.query_vectors = Some(view),
            _ => result.base_vectors = Some(view), // default to base
        }
        result
    }
}
