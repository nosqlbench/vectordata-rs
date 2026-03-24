// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Remote dataset view construction.
//!
//! Given a catalog entry (with a base URL) and a profile name, resolves
//! source URLs for each facet, downloads merkle references, creates
//! [`CachedChannel`]s, and wraps them in [`CachedVectorView`]s.

use std::io;
use std::path::Path;

use crate::cache::CachedChannel;
use crate::dataset::catalog::CatalogEntry;
use crate::dataset::view::{
    CachedVectorView, DatasetView, LocalVectorView, TypedVectorView, VecElementType,
};
use crate::merkle::{MerkleRef, MerkleState};
use crate::transport::HttpTransport;

/// A dataset view backed by remote files accessed through CachedChannels.
pub struct RemoteDatasetView {
    base_vectors: Option<Box<dyn TypedVectorView>>,
    query_vectors: Option<Box<dyn TypedVectorView>>,
    neighbor_indices: Option<Box<dyn TypedVectorView>>,
    neighbor_distances: Option<Box<dyn TypedVectorView>>,
}

impl DatasetView for RemoteDatasetView {
    fn base_vectors(&self) -> Option<Box<dyn TypedVectorView>> { None }
    fn query_vectors(&self) -> Option<Box<dyn TypedVectorView>> { None }
    fn neighbor_indices(&self) -> Option<Box<dyn TypedVectorView>> { None }
    fn neighbor_distances(&self) -> Option<Box<dyn TypedVectorView>> { None }

    fn prebuffer_all(&self) -> io::Result<()> {
        // Prebuffer each facet that exists
        if let Some(ref bv) = self.base_vectors {
            bv.prebuffer(0, bv.count())?;
        }
        if let Some(ref qv) = self.query_vectors {
            qv.prebuffer(0, qv.count())?;
        }
        if let Some(ref ni) = self.neighbor_indices {
            ni.prebuffer(0, ni.count())?;
        }
        if let Some(ref nd) = self.neighbor_distances {
            nd.prebuffer(0, nd.count())?;
        }
        Ok(())
    }
}

// The trait needs to return owned views. Since CachedVectorView is not Clone,
// we need a different approach. Store Arc-wrapped views.
// Actually, let's redesign: DatasetView returns references, not owned boxes.

/// A dataset view backed by local files.
pub struct LocalDatasetView {
    pub(crate) base_vectors: Option<LocalVectorView>,
    pub(crate) query_vectors: Option<LocalVectorView>,
    #[allow(dead_code)]
    pub(crate) neighbor_indices: Option<LocalVectorView>,
    #[allow(dead_code)]
    pub(crate) neighbor_distances: Option<LocalVectorView>,
}

impl LocalDatasetView {
    /// Open a local dataset from a directory containing dataset.yaml.
    pub fn open(dir: &Path, profile_name: &str) -> io::Result<Self> {
        use crate::dataset::config::DatasetConfig;

        let yaml_path = dir.join("dataset.yaml");
        let config = DatasetConfig::load(&yaml_path)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let profile = config.profiles.profile(profile_name)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                format!("profile '{}' not found", profile_name),
            ))?;

        let resolve_view = |facet_name: &str| -> Option<LocalVectorView> {
            let view = profile.view(facet_name)?;
            let source_path = dir.join(&view.source.path);
            if !source_path.exists() { return None; }

            let ew = view.effective_window();
            let window = if ew.is_empty() {
                None
            } else {
                ew.0.first().map(|iv| (iv.min_incl as usize, (iv.max_excl - iv.min_incl) as usize))
            };

            LocalVectorView::open(&source_path, window).ok()
        };

        Ok(LocalDatasetView {
            base_vectors: resolve_view("base_vectors"),
            query_vectors: resolve_view("query_vectors"),
            neighbor_indices: resolve_view("neighbor_indices"),
            neighbor_distances: resolve_view("neighbor_distances"),
        })
    }

    pub fn base_vectors(&self) -> Option<&dyn TypedVectorView> {
        self.base_vectors.as_ref().map(|v| v as &dyn TypedVectorView)
    }

    pub fn query_vectors(&self) -> Option<&dyn TypedVectorView> {
        self.query_vectors.as_ref().map(|v| v as &dyn TypedVectorView)
    }
}

impl RemoteDatasetView {
    /// Construct a remote view from a catalog entry and profile.
    ///
    /// `base_url` is the URL prefix for source files (derived from entry.path).
    /// `cache_dir` is the local cache directory for this dataset.
    pub fn open(
        entry: &CatalogEntry,
        profile_name: &str,
        cache_dir: &Path,
    ) -> io::Result<Self> {
        let profile = entry.layout.profiles.profile(profile_name)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                format!("profile '{}' not found in '{}'", profile_name, entry.name),
            ))?;

        // Derive the base URL from the entry path (strip dataset.yaml filename)
        let base_url = entry.path.rsplit_once('/')
            .map(|(base, _)| base)
            .unwrap_or(&entry.path);

        let ds_cache = cache_dir.join(&entry.name);
        std::fs::create_dir_all(&ds_cache)?;

        let resolve_facet = |facet_name: &str| -> Option<Box<dyn TypedVectorView>> {
            let view = profile.view(facet_name)?;
            let raw_source_path = &view.source.path;
            if raw_source_path.is_empty() { return None; }

            // Strip window notation from the source path if present.
            // e.g., "base.fvec(0..10000000)" → "base.fvec"
            // or "base.fvec[0..10000000)" → "base.fvec"
            let source_path = if let Some(bracket) = raw_source_path.find(|c: char| c == '[' || c == '(') {
                &raw_source_path[..bracket]
            } else {
                raw_source_path.as_str()
            };

            let ext = match Path::new(source_path).extension().and_then(|e| e.to_str()) {
                Some(e) => e.to_string(),
                None => {
                    log::warn!("{}: no file extension in '{}'", facet_name, source_path);
                    return None;
                }
            };
            let etype = match VecElementType::from_extension(&ext) {
                Some(t) => t,
                None => {
                    log::warn!("{}: unsupported format '.{}'", facet_name, ext);
                    return None;
                }
            };

            // Resolve full URL
            let full_url = if source_path.starts_with("http://") || source_path.starts_with("https://") {
                source_path.to_string()
            } else {
                format!("{}/{}", base_url, source_path)
            };


            // Resolve merkle reference — dual-mode: check .mrkl first (it
            // embeds the reference hashes), only download .mref if no .mrkl exists.
            // The .mref is never persisted separately; CachedChannel::open
            // creates the .mrkl immediately on first access.
            //
            // On resume, a lightweight Range request fetches just the remote
            // root hash (32 bytes) to detect republished content. If it
            // differs, the local .mrkl and cache are invalidated.
            let mref_url = format!("{}.mref", full_url);
            let content_cache_path_tmp = ds_cache.join(source_path);
            let mrkl_path = content_cache_path_tmp.with_extension(
                format!("{}.mrkl", content_cache_path_tmp.extension()
                    .and_then(|e| e.to_str()).unwrap_or("")),
            );
            let mref = if mrkl_path.exists() {
                // Resume path: load local state, then verify root hash
                // against remote with a single 32-byte Range request.
                match MerkleState::load(&mrkl_path) {
                    Ok(state) => {
                        let local_ref = state.to_ref();
                        // Check remote root hash — if it differs, content was republished
                        match fetch_remote_root_hash(&mref_url) {
                            Ok(remote_root) => {
                                if remote_root != *local_ref.root_hash() {
                                    log::info!("{}: remote content changed, re-downloading .mref", facet_name);
                                    // Invalidate local state and cache
                                    let _ = std::fs::remove_file(&mrkl_path);
                                    let _ = std::fs::remove_file(&content_cache_path_tmp);
                                    // Fall through to full download
                                    match download_mref(&mref_url) {
                                        Ok(m) => m,
                                        Err(e) => {
                                            log::warn!("{}: failed to download .mref: {}", facet_name, e);
                                            return None;
                                        }
                                    }
                                } else {
                                    local_ref
                                }
                            }
                            Err(e) => {
                                // Range request failed — proceed with local state.
                                // This is non-fatal: offline resume or server
                                // doesn't support Range are both acceptable.
                                log::debug!("{}: root hash check unavailable ({}), using cached state", facet_name, e);
                                local_ref
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("{}: failed to load .mrkl: {}", facet_name, e);
                        return None;
                    }
                }
            } else {
                // First access: download .mref (not persisted — .mrkl created by CachedChannel)
                match download_mref(&mref_url) {
                    Ok(m) => m,
                    Err(e) => {
                        log::warn!("{}: failed to download .mref: {}", facet_name, e);
                        return None;
                    }
                }
            };

            // Create transport + cached channel
            let url = match url::Url::parse(&full_url) {
                Ok(u) => u,
                Err(e) => {
                    log::warn!("{}: invalid URL '{}': {}", facet_name, full_url, e);
                    return None;
                }
            };
            let transport = HttpTransport::new(url);
            let content_cache_path = ds_cache.join(source_path);
            if let Some(parent) = content_cache_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            let channel = match CachedChannel::open(
                Box::new(transport),
                mref,
                content_cache_path.parent().unwrap_or(&ds_cache),
                content_cache_path.file_name()?.to_str()?,
            ) {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("{}: failed to open cache channel: {}", facet_name, e);
                    return None;
                }
            };

            // Resolve window from the view
            let ew = view.effective_window();
            let window = if ew.is_empty() {
                None
            } else {
                ew.0.first().map(|iv| (iv.min_incl as usize, (iv.max_excl - iv.min_incl) as usize))
            };

            match CachedVectorView::new(channel, content_cache_path, etype, window) {
                Ok(v) => Some(Box::new(v) as Box<dyn TypedVectorView>),
                Err(e) => {
                    log::warn!("{}: failed to create view: {}", facet_name, e);
                    None
                }
            }
        };

        // Only resolve base_vectors eagerly — the primary facet.
        // Other facets are resolved lazily on first access to avoid
        // unnecessary .mref downloads and HTTP requests.
        Ok(RemoteDatasetView {
            base_vectors: resolve_facet("base_vectors"),
            query_vectors: None,
            neighbor_indices: None,
            neighbor_distances: None,
        })
    }

    pub fn base_vectors(&self) -> Option<&dyn TypedVectorView> {
        self.base_vectors.as_ref().map(|v| v.as_ref())
    }

    pub fn query_vectors(&self) -> Option<&dyn TypedVectorView> {
        self.query_vectors.as_ref().map(|v| v.as_ref())
    }

    pub fn neighbor_indices(&self) -> Option<&dyn TypedVectorView> {
        self.neighbor_indices.as_ref().map(|v| v.as_ref())
    }

    pub fn neighbor_distances(&self) -> Option<&dyn TypedVectorView> {
        self.neighbor_distances.as_ref().map(|v| v.as_ref())
    }
}

/// Download a merkle reference from a URL (not persisted — the `.mrkl` file
/// created by `CachedChannel::open` serves as the durable dual-mode store).
///
/// Downloads on a separate thread to avoid conflicts with tokio
/// runtimes (reqwest::blocking creates its own runtime).
fn download_mref(url: &str) -> io::Result<MerkleRef> {
    let url_owned = url.to_string();
    let bytes = std::thread::spawn(move || -> io::Result<Vec<u8>> {
        let client = reqwest::blocking::Client::new();
        let response = client.get(&url_owned).send()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("failed to download .mref: {}", e)))?;

        if !response.status().is_success() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(".mref not found at {}: HTTP {}", url_owned, response.status()),
            ));
        }

        response.bytes()
            .map(|b| b.to_vec())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))
    }).join().map_err(|_| io::Error::new(io::ErrorKind::Other, "download thread panicked"))??;

    MerkleRef::from_bytes(&bytes)
}

/// Fetch just the root hash (first 32 bytes) of a remote `.mref` file via
/// an HTTP Range request. This is a lightweight check to detect whether the
/// remote content has been republished since our local `.mrkl` was created.
///
/// The root hash is the SHA-256 of the entire merkle tree, so a mismatch
/// means the content (or its chunking) has changed.
fn fetch_remote_root_hash(url: &str) -> io::Result<[u8; 32]> {
    let url_owned = url.to_string();
    std::thread::spawn(move || -> io::Result<[u8; 32]> {
        let client = reqwest::blocking::Client::new();
        let response = client.get(&url_owned)
            .header("Range", "bytes=0-31")
            .send()
            .map_err(|e| io::Error::new(
                io::ErrorKind::Other,
                format!("failed to fetch root hash from .mref: {}", e),
            ))?;

        if !response.status().is_success() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("root hash check failed at {}: HTTP {}", url_owned, response.status()),
            ));
        }

        let bytes = response.bytes()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        if bytes.len() < 32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("root hash response too short: {} bytes", bytes.len()),
            ));
        }

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes[..32]);
        Ok(hash)
    }).join().map_err(|_| io::Error::new(io::ErrorKind::Other, "root hash fetch thread panicked"))?
}
