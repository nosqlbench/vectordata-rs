//! Library API. High-level entry point for loading vector datasets.
//!
//! [`TestDataGroup`] parses a `dataset.yaml` (from a local path or HTTP URL)
//! and exposes named profiles as [`TestDataView`]
//! instances for reading vectors and metadata.

use crate::dataset::selector::{self, ProfileFacts, SelectionError};
use crate::location::{dir_form, Location};
use crate::model::DatasetConfig;
use crate::view::{GenericTestDataView, TestDataView};
use crate::{Error, Result};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use url::Url;

/// Represents a loaded vector dataset configuration.
///
/// Use `TestDataGroup::load` to parse a `dataset.yaml` and prepare for data access.
#[derive(Debug)]
pub struct TestDataGroup {
    /// Where the dataset lives; relative facet paths resolve against it.
    home: Location,
    /// The home as written, recorded in the dataset's
    /// `<cache_root>/<dataset>/origin.json`, so a catalog that moves can
    /// be migrated by editing one file. Never re-spelled by URL parsing.
    origin: String,
    /// The per-dataset cache directory name (`<cache_root>/<name>/`).
    /// `None` for a home with no name to give, a server's root: its
    /// facets cache by URL instead, as no directory can be named.
    name: Option<String>,
    config: DatasetConfig,
    /// The dataset as a catalog lists it, whichever route opened it:
    /// what `datasets describe` renders for a path or URL as it does
    /// for a catalog name.
    entry: crate::dataset::CatalogEntry,
    /// What each file was read as, when the directory had no
    /// descriptor and its files implied one (`dataset::infer`).
    inferred: Option<String>,
    /// The count claims overlays make, for the views to check as their
    /// facets open (OV-5).
    claims: Option<Arc<crate::overlay::Claims>>,
}

/// A dataset resolved for opening: what every route to a
/// [`TestDataGroup`] arrives at before the one constructor
/// ([`TestDataGroup::from_handle`]) builds it.
///
/// The name and origin are the dataset's cache identity. A directory or
/// URL open names the dataset after its last segment and records the
/// directory; a catalog entry gives its own name and
/// [`dataset_home_url`](crate::dataset::CatalogEntry::dataset_home_url).
/// Those are today's cache directory names, kept (SRD §11, item 1).
struct DatasetHandle {
    name: Option<String>,
    home: Location,
    origin: String,
    config: DatasetConfig,
    entry: crate::dataset::CatalogEntry,
    inferred: Option<String>,
}

/// The embedded catalog entry for `description` at `home`, named `name`.
fn embedded_entry(name: &str, home: &Location, description: crate::dataset::DatasetConfig) -> crate::dataset::CatalogEntry {
    crate::dataset::CatalogEntry::builder(name)
        .home(home)
        .dataset(description)
        .build()
        .expect("an entry with a home and a description always builds")
}

impl DatasetHandle {
    /// A dataset at `home` described by `description`, named after its
    /// last segment (or, at a server's root, by the description), whose
    /// origin is `home` as a directory.
    fn at(home: Location, description: crate::dataset::DatasetConfig) -> Self {
        let name = home.last_segment().map(str::to_string);
        let entry = embedded_entry(name.as_deref().unwrap_or(&description.name), &home, description.clone());
        DatasetHandle {
            name,
            origin: dir_form(&home.to_string()),
            home,
            config: DatasetConfig::opened(description),
            entry,
            inferred: None,
        }
    }

    /// A dataset at `home` described by `content`, a descriptor file of
    /// either shape: `dataset.yaml` (it has `profiles:`) or
    /// `knn_entries.yaml`. The shape decides, not the file's name, as in
    /// jvector's `DataSetLoaderSimpleMFD`.
    fn of_content(home: Location, descriptor: &Location, content: &str) -> Result<Self> {
        let value: serde_yaml::Value = serde_yaml::from_str(content)?;
        let is_dataset_yaml = value.as_mapping().is_some_and(|m| m.contains_key("profiles"));
        if is_dataset_yaml {
            Self::of_dataset_yaml(home, descriptor, content)
        } else {
            Self::of_knn_entries(home, content)
        }
    }

    /// A dataset at `home` described by `dataset.yaml` content, read
    /// from `descriptor`: listed as a catalog lists a `dataset.yaml`, by
    /// its descriptor and its layout.
    fn of_dataset_yaml(home: Location, descriptor: &Location, content: &str) -> Result<Self> {
        let description = crate::dataset::DatasetConfig::parse(content)?;
        let handle = Self::at(home, description.clone());
        let name = handle.entry.name().to_string();
        let layout = crate::dataset::CatalogLayout {
            format_version: description.format_version,
            profile_tags: description.profile_tags,
            attributes: description.attributes,
            profiles: description.profiles,
        };
        let entry = crate::dataset::CatalogEntry::builder(name)
            .descriptor(descriptor, layout)
            .build()
            .expect("an entry with a descriptor always builds");
        Ok(Self { entry, ..handle })
    }

    /// A dataset at `home` described by `knn_entries.yaml` content: the
    /// entry named after the directory, or the only entry. A file that
    /// lists several datasets and none named after the directory is
    /// ambiguous ([`Error::AmbiguousDataset`], naming every entry) — a
    /// choice is never made by order, as it is never made between
    /// catalogs. The file is read by the parser its catalog is read by,
    /// so `_defaults` (`base_url`, `cache_dir`) and `${VAR}` apply as
    /// they do there. The dataset keeps the directory's name and origin,
    /// the cache identity a directory open has always had.
    fn of_knn_entries(home: Location, content: &str) -> Result<Self> {
        let entries = crate::catalog::knn_entries::parse_knn_entries_yaml(content, &home.to_string())
            .map_err(Error::Other)?;
        let dir_name = home.last_segment().unwrap_or_default();
        let chosen = match entries.iter().position(|e| e.name() == dir_name) {
            Some(i) => i,
            None if entries.len() > 1 => {
                return Err(Error::AmbiguousDataset {
                    name: dir_name.to_string(),
                    matches: entries
                        .iter()
                        .map(|e| format!("{} (knn_entries.yaml: {})", e.name(), e.dataset_home_url()))
                        .collect(),
                });
            }
            None => 0,
        };
        // A listing with no description of its own is an empty dataset.
        let empty = crate::dataset::DatasetConfig::builder(dir_name).build().map_err(Error::Other)?;
        match entries.get(chosen) {
            Some(entry) => match entry.description() {
                // The entry is the catalog's own listing of it: kept whole.
                crate::dataset::EntryDescription::Embedded(description) => {
                    Ok(Self { entry: entry.clone(), ..Self::at(home, (**description).clone()) })
                }
                crate::dataset::EntryDescription::Referenced { .. } => Ok(Self::at(home, empty)),
            },
            None => Ok(Self::at(home, empty)),
        }
    }
}

/// Say, once per process for each directory, what a directory without
/// a descriptor was read as (LF-2), through `log`, so a library user
/// sees it where they see the crate's other notices.
fn announce_inferred(dir: &Path, inferred: &crate::dataset::infer::Inferred) {
    static SAID: std::sync::Mutex<Vec<std::path::PathBuf>> = std::sync::Mutex::new(Vec::new());
    let key = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.contains(&key) {
        return;
    }
    said.push(key);
    log::info!("inferred dataset.yaml for {}: {}", dir.display(), inferred.summary());
    for note in &inferred.notes {
        log::info!("{}: {note}", dir.display());
    }
}

impl TestDataGroup {
    /// Loads a TestDataGroup from a path string which can be a file path or URL.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vectordata::TestDataGroup;
    ///
    /// // Load from local path
    /// let local_group = TestDataGroup::load("./data")?;
    ///
    /// // Load from remote URL
    /// let remote_group = TestDataGroup::load("https://example.com/data/")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn load(path_or_url: &str) -> Result<Self> {
        match Location::parse(path_or_url)? {
            Location::Path(path) => Self::load_from_path(path),
            remote if remote.is_remote() => Self::load_from_url(&remote.to_string()),
            Location::Url(url) => Err(Error::Other(format!(
                "{url}: the '{}' scheme is not one vectordata reads (http, https, s3)",
                url.scheme(),
            ))),
        }
    }

    /// Loads a TestDataGroup from a local directory path or a YAML
    /// catalog file.
    ///
    /// When `path` names a `.yaml`/`.yml` file, that file *is* the
    /// catalog regardless of its basename — the content is dispatched
    /// by shape (`dataset.yaml`-shaped struct vs `knn_entries`-shaped
    /// map), matching the back-end behaviour of jvector's
    /// `DataSetLoaderSimpleMFD`. When `path` names a directory, the
    /// canonical cascade applies: `dataset.yaml` → `knn_entries.yaml`.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let read = |file: &Path| fs::read_to_string(file).map_err(Error::ConfigIo);

        // Explicit catalog file: dispatch by content shape, no
        // sibling probing.
        if path.is_file() && is_yaml_name(&path.to_string_lossy()) {
            let dir = path.parent().unwrap_or(Path::new("."));
            let handle = DatasetHandle::of_content(Location::from_path(dir), &Location::from_path(path), &read(path)?)?;
            return Ok(Self::from_handle(handle));
        }

        // Directory cascade: dataset.yaml, then knn_entries.yaml, then —
        // for a directory with neither — the descriptor its files imply
        // (`dataset::infer`, LF-1), said once per process.
        let home = Location::from_path(path);
        let knn_path = path.join("knn_entries.yaml");
        let dataset_path = path.join("dataset.yaml");
        let handle = if dataset_path.exists() {
            DatasetHandle::of_dataset_yaml(home, &Location::from_path(&dataset_path), &read(&dataset_path)?)?
        } else if knn_path.exists() {
            DatasetHandle::of_knn_entries(home, &read(&knn_path)?)?
        } else if path.is_dir() && crate::dataset::infer::has_loose_files(path) {
            let inferred = crate::dataset::infer::infer_dir(path).map_err(|e| {
                Error::Other(format!("{}: no dataset.yaml or knn_entries.yaml, and its files make no dataset: {e}", path.display()))
            })?;
            announce_inferred(path, &inferred);
            DatasetHandle { inferred: Some(inferred.summary()), ..DatasetHandle::at(home, inferred.config) }
        } else {
            DatasetHandle::of_dataset_yaml(home, &Location::from_path(&dataset_path), &read(&dataset_path)?)?
        };
        Ok(Self::from_handle(handle))
    }

    /// Loads a TestDataGroup from a URL.
    ///
    /// When the URL path ends in `.yaml`/`.yml`, that file *is* the
    /// catalog regardless of its basename — the content is dispatched
    /// by shape (`dataset.yaml`-shaped struct vs `knn_entries`-shaped
    /// map), matching jvector's `DataSetLoaderSimpleMFD`. When the
    /// URL targets a directory, the canonical cascade applies:
    /// `dataset.yaml` → `knn_entries.yaml`.
    pub fn load_from_url(url_str: &str) -> Result<Self> {
        let url = Url::parse(url_str)?;

        // Explicit catalog file: dispatch by content shape, no
        // sibling probing. Facet paths resolve against the URL's
        // parent directory.
        if is_yaml_name(url.path()) {
            let base_url = url.join(".")?;
            let content = fetch_descriptor(&url, &base_url)?.required(&url)?;
            let handle = DatasetHandle::of_content(Location::Url(base_url), &Location::Url(url.clone()), &content)?;
            return Ok(Self::from_handle(handle));
        }

        // Directory cascade.
        let mut base_url = url;
        if !base_url.path().ends_with('/') {
            base_url.set_path(&(base_url.path().to_owned() + "/"));
        }
        let home = Location::Url(base_url.clone());

        // Try dataset.yaml first. When the server cannot be asked and no
        // copy of it was kept, the dataset may still be a kept
        // `knn_entries.yaml` one: go on to that before giving up.
        let dataset_url = base_url.join("dataset.yaml")?;
        let mut unavailable = None;
        match fetch_descriptor(&dataset_url, &base_url)? {
            Descriptor::Text(content) => {
                return Ok(Self::from_handle(DatasetHandle::of_dataset_yaml(home, &Location::Url(dataset_url), &content)?));
            }
            Descriptor::Absent => {}
            Descriptor::Unavailable(e) => unavailable = Some(e),
        }

        let knn_url = base_url.join("knn_entries.yaml")?;
        let content = match (fetch_descriptor(&knn_url, &base_url)?, unavailable) {
            (Descriptor::Text(text), _) => text,
            // dataset.yaml went unasked: that is the reason, whatever
            // came of knn_entries.yaml.
            (_, Some(e)) | (Descriptor::Unavailable(e), None) => return Err(e),
            (Descriptor::Absent, None) => {
                return Err(Error::Other(format!("{base_url}: no dataset.yaml or knn_entries.yaml")));
            }
        };
        Ok(Self::from_handle(DatasetHandle::of_knn_entries(home, &content)?))
    }

    /// Open a dataset a program described itself: a description built
    /// with [`DatasetConfig::builder`](crate::dataset::DatasetConfig::builder)
    /// or parsed from text it holds, named `name`, living at `home`.
    ///
    /// Relative facet paths resolve against `home`, and a remote home is
    /// what the cache mirrors, under `<cache_root>/<name>/`.
    ///
    /// ```
    /// use vectordata::dataset::{DatasetConfig, StandardFacet};
    /// use vectordata::{Location, TestDataGroup};
    ///
    /// let description = DatasetConfig::builder("demo")
    ///     .profile("default", |p| p
    ///         .facet(StandardFacet::BaseVectors, "base.fvec")
    ///         .facet(StandardFacet::QueryVectors, "query.fvec"))
    ///     .build()
    ///     .map_err(vectordata::Error::Other)?;
    /// let home = Location::parse("https://example.com/datasets/demo/")?;
    /// let group = TestDataGroup::from_config("demo", home, description);
    /// assert_eq!(group.profile_names(), ["default"]);
    /// # Ok::<(), vectordata::Error>(())
    /// ```
    pub fn from_config(
        name: impl Into<String>,
        home: Location,
        description: crate::dataset::DatasetConfig,
    ) -> Self {
        let name = name.into();
        let home = home.as_dir();
        let entry = embedded_entry(&name, &home, description.clone());
        let handle = DatasetHandle { name: Some(name), entry, ..DatasetHandle::at(home, description) };
        Self::from_handle(handle)
    }

    /// The one constructor every open ends in (RC-7).
    fn from_handle(handle: DatasetHandle) -> Self {
        let DatasetHandle { name, home, origin, config, entry, inferred } = handle;
        TestDataGroup { home, origin, name, config, entry, inferred, claims: None }
    }

    /// The view of one profile, anchored to this dataset's cache
    /// identity: what [`profile`](Self::profile) and
    /// [`generic_view`](Self::generic_view) both return.
    fn view_of(&self, name: &str) -> Option<GenericTestDataView> {
        let profile_config = self.config.profiles.get(name)?;
        let view = GenericTestDataView::with_attributes(
            self.home.clone(),
            profile_config.clone(),
            self.config.attributes.clone(),
        )
        .with_overlay_claims(self.claims.clone())
        .with_attribute_sources(
            self.config
                .attributes
                .keys()
                .filter_map(|k| match self.attribute_source(k) {
                    Some(crate::overlay::AttributeSource::Dataset) | None => None,
                    Some(source) => Some((k.clone(), source)),
                })
                .collect(),
        );
        Some(match &self.name {
            Some(dataset) => view.with_catalog_identity(dataset.clone(), self.origin.clone()),
            None => view,
        })
    }

    /// Retrieves a view for the specified profile name.
    ///
    /// Returns `None` if the profile name does not exist in the configuration.
    pub fn profile(&self, name: &str) -> Option<Arc<dyn TestDataView>> {
        self.view_of(name).map(|view| Arc::new(view) as Arc<dyn TestDataView>)
    }

    /// Returns the concrete `GenericTestDataView` for typed facet access.
    ///
    /// Unlike `profile()` which returns a trait object, this returns the
    /// concrete type. Every facet call is on the trait object too,
    /// through [`TestDataViewExt`](crate::TestDataViewExt).
    pub fn generic_view(&self, name: &str) -> Option<GenericTestDataView> {
        self.view_of(name)
    }

    /// The dataset's name: the one its cache directory
    /// (`<cache_root>/<name>/`) and its views carry.
    ///
    /// - A catalog entry's dataset is named by the entry.
    /// - A dataset opened by directory, `dataset.yaml` or URL is named
    ///   after the directory, its last segment, whatever `name:` its
    ///   `dataset.yaml` states (that one is the description's,
    ///   [`dataset::DatasetConfig::name`](crate::dataset::DatasetConfig)).
    /// - [`from_config`](Self::from_config) takes the name it is given.
    /// - `None` for a dataset at a server's root, which has no segment
    ///   to be named after; its facets cache by URL.
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let tmp = tempfile::tempdir()?;
    /// let dir = tmp.path().join("mydata");
    /// # std::fs::create_dir(&dir)?;
    /// std::fs::write(dir.join("dataset.yaml"), "name: stated\nprofiles:\n  default: {}\n")?;
    /// let group = vectordata::TestDataGroup::load(dir.to_str().unwrap())?;
    /// assert_eq!(group.name(), Some("mydata"));
    /// # Ok(()) }
    /// ```
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the names of all available profiles.
    pub fn profile_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.config.profiles.keys().cloned().collect();
        names.sort_by(|a, b| {
            let a_bc = self.config.profiles.get(a).and_then(|p| p.base_count);
            let b_bc = self.config.profiles.get(b).and_then(|p| p.base_count);
            crate::dataset::profile::profile_sort_by_size(a, a_bc, b, b_bc)
        });
        names
    }

    /// Everything a selector can read of every profile, in the order
    /// `profile_names` lists them (PS-7, PS-9).
    pub fn profile_facts(&self) -> Vec<ProfileFacts> {
        self.profile_names()
            .iter()
            .filter_map(|n| self.config.profiles.get(n).map(|p| ProfileFacts::of_profile(n, p)))
            .collect()
    }

    /// The profiles a selector names, size-ordered (PS-9). `None` is
    /// `default`, `profile=*` is every profile, and a selector that
    /// matches nothing is an error listing what was on offer (PS-10).
    pub fn select(&self, selector: Option<&str>) -> std::result::Result<Vec<String>, SelectionError> {
        selector::resolve(selector, &self.profile_facts())
    }

    /// The one profile a selector names, for a surface that takes a
    /// single profile; more than one match is an error (PS-10).
    pub fn select_one(&self, selector: Option<&str>) -> std::result::Result<String, SelectionError> {
        selector::resolve_one(selector, &self.profile_facts())
    }

    /// Retrieves a top-level attribute from the dataset configuration.
    pub fn attribute(&self, name: &str) -> Option<&serde_yaml::Value> {
        self.config.attributes.get(name)
    }

    /// The same dataset read through `overlays` ([`crate::overlay`]):
    /// what they state for it, by its name and its catalog, merged over
    /// its attributes, so [`attribute`](Self::attribute), every view's
    /// [`distance`](crate::TestDataView::distance) and `describe` see
    /// it. A [`Catalog`](crate::catalog::Catalog) applies its overlays
    /// to every dataset it opens; this is for a group opened by path
    /// in code.
    ///
    /// ```
    /// use vectordata::overlay::{AttributeSource, Overlay, Overlays};
    /// use vectordata::{DistanceFunction, TestDataGroup};
    /// # use std::io::Write;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let d = dir.path().join("ada002-100k");
    /// # std::fs::create_dir(&d).unwrap();
    /// # let mut f = std::fs::File::create(d.join("base_vectors.fvec")).unwrap();
    /// # for _ in 0..4 { f.write_all(&2i32.to_le_bytes()).unwrap(); f.write_all(&1f32.to_le_bytes()).unwrap(); f.write_all(&0f32.to_le_bytes()).unwrap(); }
    /// # std::fs::write(d.join("dataset.yaml"), "name: ada002-100k\nprofiles:\n  default:\n    base_vectors: base_vectors.fvec\n").unwrap();
    /// let mut overlays = Overlays::new();
    /// overlays.add_program(Overlay::program("bench", "datasets:\n  ada002-100k:\n    distance_function: DOT_PRODUCT\n    bench.ncentroids: 300\n").unwrap()).unwrap();
    /// let group = TestDataGroup::load(d.to_str().unwrap()).unwrap().with_overlays(&overlays);
    /// assert_eq!(group.profile("default").unwrap().distance(), Some(DistanceFunction::DotProduct));
    /// assert_eq!(group.attribute("bench.ncentroids").and_then(|v| v.as_u64()), Some(300));
    /// assert!(matches!(group.attribute_source("distance_function"), Some(AttributeSource::Overlay { .. })));
    /// ```
    pub fn with_overlays(mut self, overlays: &crate::overlay::Overlays) -> Self {
        let catalog = self.entry.catalog_name().map(str::to_string);
        let name = self.name.clone().unwrap_or_else(|| self.entry.name().to_string());
        let stated = overlays.resolve(catalog.as_deref(), &name);
        self.apply_stated(stated);
        self
    }

    /// Merge what overlays stated over the attributes, the entry's and
    /// the views'.
    fn apply_stated(&mut self, stated: Vec<crate::overlay::Stated>) {
        if stated.is_empty() {
            return;
        }
        for s in &stated {
            self.config.attributes.insert(s.key.clone(), s.value.clone());
        }
        let name = self.name.clone().unwrap_or_else(|| self.entry.name().to_string());
        self.claims = crate::overlay::Claims::of(&name, &stated).map(Arc::new);
        crate::overlay::warn_contradictions(self.entry.apply_overlay(stated));
    }

    /// Where attribute `key`'s value came from: an overlay, the names of
    /// loose files (`dataset::infer`), or the dataset. `None` when
    /// nothing sets it.
    pub fn attribute_source(&self, key: &str) -> Option<crate::overlay::AttributeSource> {
        if let Some(s) = self.entry.overlaid().iter().find(|s| s.key == key) {
            return Some(s.source.clone());
        }
        self.config.attributes.get(key)?;
        Some(if self.inferred.is_some() && matches!(key, "distance_function" | "is_normalized") {
            crate::overlay::AttributeSource::Inferred
        } else {
            crate::overlay::AttributeSource::Dataset
        })
    }

    /// What each file was read as, when the dataset is a directory with
    /// no descriptor read by the rules of [`crate::dataset::infer`]:
    /// `base_vectors=base.fvecs, query_vectors=…`. `None` for a
    /// declared dataset.
    pub fn inferred(&self) -> Option<&str> {
        self.inferred.as_deref()
    }

    /// The dataset as a catalog lists it: the catalog's own entry for a
    /// dataset opened from one, else an entry embedding the description
    /// it was opened with, named and homed as the group is.
    pub(crate) fn catalog_entry(&self) -> &crate::dataset::CatalogEntry {
        &self.entry
    }

    /// Direct read-only access to the underlying parsed
    /// `dataset.yaml`. Used by tooling that needs profile-level
    /// detail not exposed through [`TestDataView`] (e.g.
    /// `vectordata datasets derive` reads per-facet windows so it
    /// can materialize them into a self-standing dataset).
    pub fn config(&self) -> &DatasetConfig {
        &self.config
    }

    /// Fetch facets across several profiles, with progress — the
    /// multi-profile form of [`TestDataView::fetch`].
    ///
    /// `profiles` are fetched in the order given (what a selector
    /// resolved to, through [`select`](Self::select)); a name repeated
    /// is fetched once. `request` applies to each profile. Every event
    /// and report row carries its profile in
    /// [`FacetId::profile`](crate::fetch::FacetId::profile).
    ///
    /// The whole run is planned and checked before anything is fetched,
    /// so an unknown facet in the third profile fails the call before
    /// the first profile's bytes move.
    pub fn fetch(
        &self,
        profiles: &[String],
        request: &crate::fetch::FetchRequest,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> crate::Result<crate::fetch::FetchReport> {
        self.plan_fetch(profiles, request, progress)?.execute(progress)
    }

    /// Plan a [`fetch`](Self::fetch) across `profiles` without fetching.
    pub fn plan_fetch(
        &self,
        profiles: &[String],
        request: &crate::fetch::FetchRequest,
        progress: &mut dyn crate::fetch::FetchProgress,
    ) -> crate::Result<crate::fetch::FetchPlan> {
        let mut seen = std::collections::HashSet::new();
        let mut facets = Vec::new();
        for name in profiles.iter().filter(|p| seen.insert(p.as_str())) {
            let view = self.profile(name).ok_or_else(|| crate::Error::Selection {
                dataset: self.name.clone().unwrap_or_default(),
                message: format!("profile '{name}' not found"),
            })?;
            crate::fetch::plan_view(&*view, Some(name), request, progress, &mut facets)?;
        }
        Ok(crate::fetch::plan_from(
            facets,
            request,
            crate::fetch::CapacityCheck::new(|bytes| Ok(crate::cache::ensure_cache_capacity(bytes)?)),
        ))
    }

    /// Deprecated: use [`fetch`](Self::fetch) over
    /// [`profile_names`](Self::profile_names) with
    /// [`FetchRequest::all`](crate::fetch::FetchRequest::all).
    #[deprecated(
        since = "2.5.0",
        note = "use `group.fetch(&group.profile_names(), &FetchRequest::all(), &mut Silent)` (vectordata::fetch)"
    )]
    pub fn prebuffer_all_profiles(&self) -> crate::Result<()> {
        self.fetch(&self.profile_names(), &crate::fetch::FetchRequest::all(), &mut crate::fetch::Silent)
            .map(|_| ())
    }

    /// Deprecated: use [`fetch`](Self::fetch) over
    /// [`profile_names`](Self::profile_names) with a
    /// [`FetchProgress`](crate::fetch::FetchProgress) sink.
    #[deprecated(
        since = "2.5.0",
        note = "use `group.fetch(&group.profile_names(), &FetchRequest::all(), &mut sink)` (vectordata::fetch)"
    )]
    pub fn prebuffer_all_profiles_with_progress(
        &self,
        fallback: crate::view::WholeFacetFallback,
        progress_cb: &mut dyn FnMut(&str, &str, &crate::PrebufferProgress),
        warn_cb: &mut dyn FnMut(u64),
    ) -> crate::Result<()> {
        #[allow(deprecated)]
        self.prebuffer_profiles_with_progress(&self.profile_names(), fallback, progress_cb, warn_cb)
    }

    /// Deprecated: use [`fetch`](Self::fetch), and compare
    /// [`FetchPlan::bytes_to_fetch`](crate::fetch::FetchPlan::bytes_to_fetch)
    /// from [`plan_fetch`](Self::plan_fetch) against
    /// [`PREBUFFER_LARGE_WARNING_BYTES`] for the large-download notice.
    ///
    /// `progress_cb(profile, facet, prog)` hears each facet; `warn_cb`
    /// hears the planned total once, before anything is fetched, when
    /// it reaches [`PREBUFFER_LARGE_WARNING_BYTES`].
    #[deprecated(
        since = "2.5.0",
        note = "use `group.fetch(&profiles, &FetchRequest::all(), &mut sink)` (vectordata::fetch)"
    )]
    pub fn prebuffer_profiles_with_progress(
        &self,
        profiles: &[String],
        fallback: crate::view::WholeFacetFallback,
        progress_cb: &mut dyn FnMut(&str, &str, &crate::PrebufferProgress),
        warn_cb: &mut dyn FnMut(u64),
    ) -> crate::Result<()> {
        let request = crate::fetch::FetchRequest::all().fallback(fallback);
        let mut sink = crate::view::prebuffer_progress_adapter(progress_cb);
        let plan = self.plan_fetch(profiles, &request, &mut sink)?;
        if plan.bytes_to_fetch() >= PREBUFFER_LARGE_WARNING_BYTES {
            warn_cb(plan.bytes_to_fetch());
        }
        plan.execute(&mut sink).map(|_| ())
    }
}

/// Fetch a dataset's descriptor (`dataset.yaml`, `knn_entries.yaml`, or
/// an explicitly named file) from `url`, keeping a copy in the dataset's
/// cache directory, and falling back to that copy when the server
/// cannot be reached.
///
/// The server is asked first because a descriptor is mutable — a
/// republished dataset can gain profiles — unlike the verified data it
/// describes, whose complete copies open with no network at all. The
/// kept copy is what lets a cached dataset still open when the server
/// is down or the machine is offline. Errors only for a server answer
/// that is neither the file nor "no such file", or a kept copy that
/// cannot be read.
fn fetch_descriptor(
    url: &Url,
    base_url: &Url,
) -> Result<Descriptor> {
    let kept = kept_descriptor_path(url, base_url).filter(|p| p.is_file());
    let read_kept = |path: &std::path::Path| {
        std::fs::read_to_string(path).map(Descriptor::Text).map_err(Error::ConfigIo)
    };
    if crate::settings::offline() {
        return match &kept {
            Some(path) => read_kept(path),
            None => Ok(Descriptor::Unavailable(Error::Other(
                crate::transport::ensure_online(url.as_str()).unwrap_err().to_string(),
            ))),
        };
    }
    // An `s3://` URL is fetched over its HTTPS endpoint; auth and the
    // kept copy stay keyed by the URL as the dataset names it.
    let resp = crate::transport::shared().send(crate::transport::Request::get(url.as_str()));
    match resp {
        Ok(r) if r.is_success() => {
            let text = r.text()?;
            if let Some(path) = kept_descriptor_path(url, base_url) {
                // Best effort: a cache directory that cannot be written
                // only costs the offline fallback.
                let _ = path.parent().map(std::fs::create_dir_all);
                let _ = std::fs::write(path, &text);
            }
            Ok(Descriptor::Text(text))
        }
        Ok(r) if r.status() == 404 => Ok(Descriptor::Absent),
        Ok(r) => Err(Error::Other(format!("{url}: {}", r.status_line()))),
        Err(e) if e.is_unreachable() => match &kept {
            Some(path) => {
                log::warn!("{url} is unreachable ({}); using the copy kept at {}", e.message, path.display());
                read_kept(path)
            }
            None => Ok(Descriptor::Unavailable(Error::Transport(e))),
        },
        Err(e) => Err(Error::Transport(e)),
    }
}

/// What [`fetch_descriptor`] found for one descriptor file.
enum Descriptor {
    /// The file's text: from the server, or the kept copy.
    Text(String),
    /// The server answered that there is no such file.
    Absent,
    /// The server could not be asked — unreachable, or offline mode —
    /// and no copy was kept. The error says which.
    Unavailable(Error),
}

impl Descriptor {
    /// The text of a descriptor the caller cannot do without: absent is
    /// "not found", unavailable is why it could not be asked for.
    fn required(self, url: &Url) -> Result<String> {
        match self {
            Descriptor::Text(text) => Ok(text),
            Descriptor::Absent => Err(Error::Other(format!("{url}: not found"))),
            Descriptor::Unavailable(e) => Err(e),
        }
    }
}

/// Fetch a dataset's `dataset.yaml` from a URL naming the dataset
/// directory or the file itself, through [`fetch_descriptor`]: read
/// auth, a kept copy, and that copy when the server is unreachable. For
/// callers that need the raw text rather than a parsed group.
pub(crate) fn fetch_dataset_yaml(url_str: &str) -> Result<String> {
    let mut url = Url::parse(url_str)?;
    if !url.path().ends_with(".yaml") && !url.path().ends_with(".yml") {
        if !url.path().ends_with('/') {
            url.set_path(&(url.path().to_owned() + "/"));
        }
        url = url.join("dataset.yaml")?;
    }
    let base_url = url.join(".")?;
    fetch_descriptor(&url, &base_url)?.required(&url)
}

/// Where a descriptor fetched from `url` is kept: under the dataset's
/// cache directory, `<cache>/<dataset>/<path below the dataset's base>`
/// — the same place the published layout puts it. `None` without a
/// configured cache.
fn kept_descriptor_path(url: &Url, base_url: &Url) -> Option<std::path::PathBuf> {
    let cache_root = crate::settings::cache_dir().ok()?;
    let dataset = last_segment(base_url)?;
    let rel = url.path().strip_prefix(base_url.path()).unwrap_or(url.path()).trim_start_matches('/');
    if rel.is_empty() || rel.contains("..") {
        return None;
    }
    let dir = crate::cache::layout::dataset_cache_dir(&cache_root, &dataset);
    // The directory belongs to one origin; a same-named dataset from
    // elsewhere neither reads nor overwrites this one's descriptor.
    crate::cache::layout::verify_or_record_origin(&dir, &dir_form(base_url.as_str())).ok()?;
    Some(dir.join(rel))
}

/// Planned download size at which a fetch deserves a "this is a lot of
/// data" notice before it starts: what `vectordata datasets precache`
/// warns at, comparing [`FetchPlan::bytes_to_fetch`](crate::fetch::FetchPlan::bytes_to_fetch)
/// against it.
/// 250 MiB matches the documented operator guidance: a reminder, not
/// a hard limit.
pub const PREBUFFER_LARGE_WARNING_BYTES: u64 = 250 * 1024 * 1024;

impl TestDataGroup {
    /// Open the dataset a [`CatalogEntry`](crate::dataset::CatalogEntry)
    /// names: what [`Catalog::open_entry`](crate::catalog::Catalog::open_entry) does.
    ///
    /// An [embedded](crate::dataset::EntryDescription::Embedded) entry
    /// holds its whole description, so nothing more is fetched. It is
    /// named `entry.name`, and its home is
    /// [`dataset_home_url`](crate::dataset::CatalogEntry::dataset_home_url),
    /// which relative facet paths resolve against, as the catalog
    /// resolves them, and which the cache mirrors under `<cache>/<name>/`.
    ///
    /// A [referenced](crate::dataset::EntryDescription::Referenced)
    /// entry's description is read from its descriptor, as
    /// [`load`](Self::load) reads it, which names the dataset after the
    /// descriptor's directory: today's cache names (SRD §11, item 1).
    pub fn from_catalog_entry(entry: &crate::dataset::CatalogEntry) -> Result<Self> {
        match entry.description() {
            crate::dataset::EntryDescription::Embedded(description) => {
                let origin = entry.dataset_home_url().to_string();
                Ok(Self::from_handle(DatasetHandle {
                    name: Some(entry.name().to_string()),
                    home: Location::parse(&origin)?,
                    origin,
                    config: DatasetConfig::opened((**description).clone()),
                    entry: entry.clone(),
                    inferred: None,
                }))
            }
            crate::dataset::EntryDescription::Referenced { descriptor, .. } => Self::load(descriptor),
        }
        .map(|mut group| {
            // What the catalog's overlays stated for the entry, over the
            // attributes the opened dataset reads.
            group.apply_stated(entry.overlaid().to_vec());
            group
        })
    }
}

/// True for a name ending in `.yaml` or `.yml`: an explicitly named
/// descriptor file rather than a dataset directory.
fn is_yaml_name(name: &str) -> bool {
    name.ends_with(".yaml") || name.ends_with(".yml")
}

/// The last non-empty path segment of a dataset's base URL: its
/// directory name, which names the dataset when nothing else does.
fn last_segment(base_url: &Url) -> Option<String> {
    Location::Url(base_url.clone()).last_segment().map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestDataViewExt;
    use std::io::Write;
    use tempfile::tempdir;

    /// **A `file://` URI opens the local dataset it names**, as the
    /// path itself does. It was once taken for a relative path that
    /// does not exist.
    #[test]
    fn a_file_uri_opens_the_local_dataset() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("dataset.yaml"),
            "name: local\nprofiles:\n  default:\n    base_vectors: b.fvec\n",
        )
        .unwrap();
        let uri = url::Url::from_directory_path(dir.path()).unwrap().to_string();
        let group = TestDataGroup::load(&uri).expect("a file:// URI opens");
        assert_eq!(group.profile_names(), vec!["default".to_string()]);
    }

    /// Three two-dimensional f32 vectors in `.fvec` form.
    fn write_fvec(path: &Path) {
        let mut buf = Vec::new();
        for i in 0..3 {
            buf.extend(&2i32.to_le_bytes());
            buf.extend(&(i as f32).to_le_bytes());
            buf.extend(&(i as f32 + 0.5).to_le_bytes());
        }
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, buf).unwrap();
    }

    /// **A catalog entry's dataset reads a relative facet from where the
    /// catalog lists it**: under the dataset's home, not the catalog's
    /// base. The open once resolved against `entry.path`, which for a
    /// `knn_entries` entry is the catalog directory, so a listed file
    /// was not the one the open looked for.
    #[test]
    fn a_catalog_entry_reads_its_facets_where_the_catalog_lists_them() {
        let dir = tempdir().unwrap();
        let catalog_dir = dir.path().join("catalog");
        write_fvec(&catalog_dir.join("ds").join("b.fvec"));
        let entry: crate::dataset::CatalogEntry = serde_json::from_value(serde_json::json!({
            "name": "ds",
            "path": catalog_dir.to_str().unwrap(),
            "dataset_type": "knn_entries.yaml",
            "layout": {"profiles": {"default": {"base_vectors": "b.fvec"}}},
        }))
        .unwrap();
        assert_eq!(
            Path::new(&entry.resolve_facet_url("b.fvec").unwrap()),
            catalog_dir.join("ds").join("b.fvec"),
        );
        let group = TestDataGroup::from_catalog_entry(&entry).unwrap();
        let base = group.profile("default").unwrap().base_vectors_as::<f32>().expect("the listed file opens");
        assert_eq!(base.count(), 3);
    }

    /// **An entry built in code opens through the catalog's open**, its
    /// relative facets read from the home it was given.
    #[test]
    fn a_built_entry_opens_at_its_home() {
        let dir = tempdir().unwrap();
        write_fvec(&dir.path().join("b.fvec"));
        let description = crate::dataset::DatasetConfig::builder("built")
            .profile("default", |p| p.facet(crate::dataset::StandardFacet::BaseVectors, "b.fvec"))
            .build()
            .unwrap();
        let entry = crate::dataset::CatalogEntry::builder("built")
            .home(&Location::from_path(dir.path()))
            .dataset(description)
            .build()
            .unwrap();
        let group = crate::catalog::Catalog::open_entry(&entry).unwrap();
        assert_eq!(group.name.as_deref(), Some("built"));
        assert_eq!(group.profile("default").unwrap().base_vectors_as::<f32>().unwrap().count(), 3);
    }

    /// **A description built in code opens as a dataset**, its relative
    /// facets read from the home it is given.
    #[test]
    fn a_built_description_opens_at_its_home() {
        let dir = tempdir().unwrap();
        write_fvec(&dir.path().join("profiles/base/b.fvec"));
        let description = crate::dataset::DatasetConfig::builder("built")
            .profile("default", |p| p.facet(crate::dataset::StandardFacet::BaseVectors, "profiles/base/b.fvec"))
            .build()
            .unwrap();
        let group = TestDataGroup::from_config("built", Location::from_path(dir.path()), description);
        assert_eq!(group.name.as_deref(), Some("built"));
        let base = group.profile("default").unwrap().base_vectors_as::<f32>().unwrap();
        assert_eq!(base.count(), 3);
    }

    /// **A directory holding only `knn_entries.yaml` opens the entry
    /// named after the directory**, and the first entry when none is.
    #[test]
    fn a_knn_entries_directory_opens_the_entry_named_after_it() {
        let dir = tempdir().unwrap();
        let ds = dir.path().join("second");
        fs::create_dir_all(&ds).unwrap();
        let content = "first:\n  base: a.fvec\n  query: q.fvec\n  gt: g.ivec\n\
                       second:\n  base: b.fvec\n  query: q.fvec\n  gt: g.ivec\n";
        fs::write(ds.join("knn_entries.yaml"), content).unwrap();
        let group = TestDataGroup::load(ds.to_str().unwrap()).unwrap();
        let base = group.config().profiles["default"].base_vectors.as_ref().unwrap();
        assert_eq!(base.source().map(Path::new), Some(ds.join("b.fvec").as_path()));
        assert_eq!(group.name.as_deref(), Some("second"));

    }

    /// **A directory whose `knn_entries.yaml` lists several datasets,
    /// none named after it, is ambiguous**: opening it names every entry
    /// rather than picking the first, as two catalogs declaring one name
    /// are never resolved by order. One entry opens whatever it is
    /// called.
    #[test]
    fn a_knn_entries_directory_with_two_entries_and_no_match_is_ambiguous() {
        let dir = tempdir().unwrap();
        let content = "first:\n  base: a.fvec\n  query: q.fvec\n  gt: g.ivec\n\
                       second:\n  base: b.fvec\n  query: q.fvec\n  gt: g.ivec\n";
        let other = dir.path().join("other");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("knn_entries.yaml"), content).unwrap();
        match TestDataGroup::load(other.to_str().unwrap()) {
            Err(Error::AmbiguousDataset { name, matches }) => {
                assert_eq!(name, "other");
                assert_eq!(matches.len(), 2, "{matches:?}");
                assert!(matches[0].starts_with("first") && matches[1].starts_with("second"), "{matches:?}");
            }
            Err(e) => panic!("the wrong error: {e}"),
            Ok(group) => panic!("opened {:?} out of two entries", group.name),
        }

        let only = dir.path().join("only");
        fs::create_dir_all(&only).unwrap();
        fs::write(only.join("knn_entries.yaml"), "lone:\n  base: a.fvec\n  query: q.fvec\n  gt: g.ivec\n").unwrap();
        let group = TestDataGroup::load(only.to_str().unwrap()).unwrap();
        let base = group.config().profiles["default"].base_vectors.as_ref().unwrap();
        assert_eq!(base.source().map(Path::new), Some(only.join("a.fvec").as_path()));
    }

    /// **A directory's `knn_entries.yaml` reads its facets where its
    /// `_defaults.base_url` says**, as the same file does when a catalog
    /// lists it. The directory open had its own parser, which ignored
    /// `_defaults` and looked for the files beside the catalog.
    #[test]
    fn a_knn_entries_directory_honours_its_defaults_base_url() {
        let dir = tempdir().unwrap();
        let data = dir.path().join("data");
        write_fvec(&data.join("b.fvec"));
        let ds = dir.path().join("ds");
        fs::create_dir_all(&ds).unwrap();
        let content = format!(
            "_defaults:\n  base_url: {}\nds:\n  base: b.fvec\n  query: b.fvec\n  gt: g.ivec\n",
            data.to_str().unwrap(),
        );
        fs::write(ds.join("knn_entries.yaml"), content).unwrap();
        let group = TestDataGroup::load(ds.to_str().unwrap()).unwrap();
        let base = group.profile("default").unwrap().base_vectors_as::<f32>().expect("the file under base_url opens");
        assert_eq!(base.count(), 3);
        assert_eq!(group.name.as_deref(), Some("ds"), "the directory keeps its cache name");
    }

    /// **A directory's `knn_entries.yaml` reads the files beside it
    /// before its `_defaults.base_url`**, as jvector does: with no
    /// `cache_dir`, the catalog's own directory is where the local
    /// copies are. A published dataset's directory carries its remote
    /// home as `base_url` and its files on disk; reading them from the
    /// remote fetched data that was already there, and failed offline.
    #[test]
    fn a_knn_entries_directory_reads_the_files_beside_it_before_its_base_url() {
        let dir = tempdir().unwrap();
        let ds = dir.path().join("ds");
        write_fvec(&ds.join("b.fvec"));
        let content = "_defaults:\n  base_url: s3://no-such-bucket/ds\nds:\n  base: b.fvec\n  query: b.fvec\n  gt: g.ivec\n";
        fs::write(ds.join("knn_entries.yaml"), content).unwrap();
        let group = TestDataGroup::load(ds.to_str().unwrap()).unwrap();
        let profile = group.profile("default").unwrap();
        assert_eq!(profile.base_vectors_as::<f32>().expect("the local copy opens, with no request").count(), 3);
        let gt = group.config().profiles["default"].neighbor_indices.as_ref().unwrap();
        assert_eq!(gt.source(), Some("s3://no-such-bucket/ds/g.ivec"), "a file not beside it is still the remote's");
    }

    /// **A dataset at a server's root has no cache directory name**, so
    /// its facets cache by URL. Naming it with the empty string put every
    /// such dataset in the cache root, where the second server's origin
    /// was refused as a mismatch.
    #[test]
    fn a_dataset_at_a_server_root_is_not_given_an_empty_name() {
        let empty = crate::dataset::DatasetConfig::builder("").build().unwrap();
        for root in ["http://127.0.0.1:9/", "http://127.0.0.1:10/"] {
            let handle = DatasetHandle::at(Location::parse(root).unwrap(), empty.clone());
            assert_eq!(handle.name, None, "{root}");
            assert_eq!(handle.origin, root);
        }
        let handle = DatasetHandle::at(
            Location::parse("http://127.0.0.1:9/ds/").unwrap(),
            empty,
        );
        assert_eq!(handle.name.as_deref(), Some("ds"));
    }

    /// **`name()` is the cache identity of every route**: a catalog
    /// entry's name, a directory's last segment over the `name:` its
    /// `dataset.yaml` states, `from_config`'s argument, and `None` at a
    /// server's root.
    #[test]
    fn name_is_the_cache_identity_of_every_route() {
        let tmp = tempdir().unwrap();
        let dir = tmp.path().join("mydata");
        write_fvec(&dir.join("b.fvec"));
        std::fs::write(dir.join("dataset.yaml"), "name: stated\nprofiles:\n  default:\n    base_vectors: b.fvec\n").unwrap();
        assert_eq!(TestDataGroup::load(dir.to_str().unwrap()).unwrap().name(), Some("mydata"), "by directory");
        let yaml = dir.join("dataset.yaml");
        assert_eq!(TestDataGroup::load(yaml.to_str().unwrap()).unwrap().name(), Some("mydata"), "by dataset.yaml");

        let description = crate::dataset::DatasetConfig::builder("described")
            .profile("default", |p| p.facet(crate::dataset::StandardFacet::BaseVectors, "b.fvec"))
            .build()
            .unwrap();
        let entry = crate::dataset::CatalogEntry::builder("listed")
            .home(&Location::from_path(&dir))
            .dataset(description.clone())
            .build()
            .unwrap();
        assert_eq!(crate::catalog::Catalog::open_entry(&entry).unwrap().name(), Some("listed"), "by catalog entry");
        let home = Location::from_path(&dir);
        assert_eq!(TestDataGroup::from_config("given", home, description.clone()).name(), Some("given"));

        let root = DatasetHandle::at(Location::parse("http://127.0.0.1:9/").unwrap(), description);
        assert_eq!(TestDataGroup::from_handle(root).name(), None, "a server's root");
    }

    /// **A view's `distance()` reads the declared spelling through the
    /// one table**: a catalog's `ip` is `DotProduct`; an undeclared or
    /// unknown function is `None`, while `distance_function()` keeps the
    /// text as written.
    #[test]
    fn a_views_distance_reads_any_spelling_and_none_when_unknown() {
        use crate::DistanceFunction;
        let tmp = tempdir().unwrap();
        for (attributes, written, typed) in [
            ("attributes:\n  distance_function: ip\n", Some("ip"), Some(DistanceFunction::DotProduct)),
            ("attributes:\n  distance_function: Euclidean\n", Some("Euclidean"), Some(DistanceFunction::L2)),
            ("attributes:\n  distance_function: chebyshev\n", Some("chebyshev"), None),
            ("", None, None),
        ] {
            let dir = tmp.path().join(format!("d{}", attributes.len()));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("dataset.yaml"), format!("name: d\n{attributes}profiles:\n  default:\n    base_vectors: b.fvec\n"))
                .unwrap();
            let group = TestDataGroup::load(dir.to_str().unwrap()).unwrap();
            let view = group.profile("default").unwrap();
            assert_eq!(view.distance_function().as_deref(), written, "{attributes:?}");
            assert_eq!(view.distance(), typed, "{attributes:?}");
        }
    }

    /// **A scheme vectordata does not read is named as such**, not
    /// reported as a missing local file.
    #[test]
    fn an_unsupported_scheme_is_named_in_the_error() {
        let err = TestDataGroup::load("gs://bucket/ds/").unwrap_err().to_string();
        assert!(err.contains("'gs' scheme"), "{err}");
    }

    #[test]
    fn dataset_home_url_canonical_strips_dataset_yaml_basename() {
        // Build a CatalogEntry with the canonical `dataset.yaml` shape
        // and assert that the home URL we record as catalog_source is
        // the *directory containing* the dataset.yaml, not the URL of
        // the YAML itself. View facet URLs strip this prefix to get
        // a clean file_relpath like "base.fvec".
        let layout = crate::dataset::CatalogLayout {
            format_version: crate::model::FORMAT_VERSION_BASE,
            profile_tags: Default::default(),
            attributes: None,
            profiles: Default::default(),
        };
        let entry = crate::dataset::CatalogEntry::from_listing(
            "vecs1m".to_string(),
            "https://example.com/datasets/vecs1m/dataset.yaml".to_string(),
            "dataset.yaml".to_string(),
            layout,
        )
        .unwrap();
        assert_eq!(entry.dataset_home_url(), "https://example.com/datasets/vecs1m/");
    }

    #[test]
    fn dataset_home_url_knn_entries_appends_dataset_name() {
        // For knn_entries-shape catalogs, entry.path is the catalog
        // *base* (no dataset segment). The home URL needs the
        // dataset name appended so the strip-prefix in
        // view.open_facet_storage yields the right relpath
        // ("base.fvec", not the absolute URL).
        let layout = crate::dataset::CatalogLayout {
            format_version: crate::model::FORMAT_VERSION_BASE,
            profile_tags: Default::default(),
            attributes: None,
            profiles: Default::default(),
        };
        let entry = crate::dataset::CatalogEntry::from_listing(
            "emb-002-100k".to_string(),
            "s3://vector-datasets-public/datasets-clean".to_string(),
            "knn_entries.yaml".to_string(),
            layout,
        )
        .unwrap();
        let group = TestDataGroup::from_catalog_entry(&entry).unwrap();
        assert_eq!(group.origin, "s3://vector-datasets-public/datasets-clean/emb-002-100k/");
        assert_eq!(group.name.as_deref(), Some("emb-002-100k"));
    }

    #[test]
    fn test_load_from_path_success() {
        let dir = tempdir().unwrap();
        let yaml_path = dir.path().join("dataset.yaml");
        let mut file = fs::File::create(&yaml_path).unwrap();
        writeln!(file, r#"
attributes:
  dimension: 128
profiles:
  default:
    base_vectors: base.fvec
"#).unwrap();

        let group = TestDataGroup::load(dir.path().to_str().unwrap()).unwrap();
        assert!(group.profile("default").is_some());
        assert!(group.profile("nonexistent").is_none());
        
        let dim = group.attribute("dimension").unwrap();
        assert_eq!(dim.as_u64().unwrap(), 128);
    }

    #[test]
    fn test_load_from_path_file_success() {
        let dir = tempdir().unwrap();
        let yaml_path = dir.path().join("dataset.yaml");
        let mut file = fs::File::create(&yaml_path).unwrap();
        writeln!(file, "profiles: {{}}").unwrap();

        let group = TestDataGroup::load(yaml_path.to_str().unwrap()).unwrap();
        assert!(group.config.profiles.is_empty());
    }

    #[test]
    fn test_load_from_path_missing_file() {
        let dir = tempdir().unwrap();
        let result = TestDataGroup::load(dir.path().to_str().unwrap());
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::ConfigIo(_) => (), // Expected
            e => panic!("Expected ConfigIo error, got {:?}", e),
        }
    }
    
    #[test]
    fn test_load_from_path_invalid_yaml() {
        let dir = tempdir().unwrap();
        let yaml_path = dir.path().join("dataset.yaml");
        let mut file = fs::File::create(&yaml_path).unwrap();
        writeln!(file, "invalid_yaml: [").unwrap();

        let result = TestDataGroup::load(dir.path().to_str().unwrap());
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::ConfigParse(_) => (), // Expected
            e => panic!("Expected ConfigParse error, got {:?}", e),
        }
    }
}
