// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Catalog source discovery — locates catalog files from config directories,
//! explicit paths, and URLs.
//!
//! Mirrors the upstream Java `TestDataSources` class. The resolution chain:
//!
//! 1. A config directory (default `~/.config/vectordata/`) contains a
//!    `catalogs.yaml` file listing catalog locations.
//! 2. Each location is a directory path, file path, or HTTP(S) URL.
//! 3. Directory paths are resolved to `catalog.json` within that directory.
//! 4. Each location becomes a [`LocatedCatalog`], required or optional,
//!    in a [`CatalogSources`] set, beside any catalog a program adds with
//!    [`CatalogSources::with`].
//! 5. The `disabled_catalogs` setting in the `settings.yaml` beside that
//!    `catalogs.yaml` names catalogs the set holds but does not assemble
//!    ([`CatalogSources::iter`]); naming one for the run reads it.

use std::path::Path;
use std::sync::Arc;

use super::CatalogSource;
use super::source::{LocatedCatalog, Requirement};

/// Default configuration directory for catalog discovery.
pub const DEFAULT_CONFIG_DIR: &str = "~/.config/vectordata";

/// The catalogs a [`Catalog`](super::Catalog) is made of, in priority
/// order: any [`CatalogSource`], each required or optional.
///
/// [`with`](Self::with) adds one a program supplies, such as a
/// [`CatalogBuilder`](super::CatalogBuilder)'s catalog. The `configure*`
/// and `add_*catalogs` methods add [`LocatedCatalog`]s, from a
/// `catalogs.yaml` or from locations given. A required catalog that
/// gives nothing is an error; an optional one is not. Registered and
/// located catalogs are alike in every other way: a dataset name in two
/// of them is ambiguous, whichever kinds they are.
///
/// A catalog the user disabled (`disabled_catalogs` in the
/// `settings.yaml` of a configured directory, as `config catalog
/// disable` writes it) stays in the set but is not assembled: it is not
/// in [`iter`](Self::iter), so no [`Catalog`](super::Catalog) of the set
/// reads it. The setting names catalogs, so it disables a program's
/// catalog of that name as well as a configured one. A catalog named
/// for the run is read all the same: [`named`](Self::named) finds a
/// disabled catalog, and a location added with
/// [`add_catalogs`](Self::add_catalogs) or
/// [`add_named_catalog`](Self::add_named_catalog) is never disabled.
#[derive(Clone, Default)]
pub struct CatalogSources {
    sources: Vec<Slot>,
    /// The names the `disabled_catalogs` setting of each configured
    /// directory lists: catalogs by these names are not assembled.
    disabled: std::collections::BTreeSet<String>,
    /// Each `configure*` call, so [`reload_configured`](Self::reload_configured)
    /// can read its `catalogs.yaml` again.
    configured: Vec<Configured>,
    /// Problems met resolving locations given directly.
    diagnostics: Vec<super::CatalogDiagnostic>,
    /// The attribute overlays every entry and opened dataset is read
    /// through: the program's, and the user's from a configured
    /// directory (`crate::overlay`).
    overlays: crate::overlay::Overlays,
}

/// One catalog in the set.
#[derive(Clone)]
struct Slot {
    source: Arc<dyn CatalogSource>,
    requirement: Requirement,
    /// The `configure*` call it came from, by index into `configured`.
    from_config: Option<usize>,
    /// A location given directly ([`CatalogSources::add_catalogs`],
    /// [`CatalogSources::add_named_catalog`]): named for the run, so
    /// never disabled.
    given: bool,
}

/// One `configure*` call: which directory's `catalogs.yaml`, how its
/// catalogs are required, and what reading it reported.
#[derive(Clone)]
struct Configured {
    dir: String,
    requirement: Requirement,
    /// `configure_optional`: a directory with no `catalogs.yaml` says nothing.
    quiet_if_absent: bool,
    diagnostics: Vec<super::CatalogDiagnostic>,
}

impl std::fmt::Debug for CatalogSources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatalogSources")
            .field(
                "sources",
                &self
                    .iter()
                    .map(|(s, r)| (s.name().to_string(), s.origin().to_string(), r))
                    .collect::<Vec<_>>(),
            )
            .field("disabled", &self.disabled)
            .field("diagnostics", &self.diagnostics())
            .finish()
    }
}

impl CatalogSources {
    /// Create an empty source set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure from the default config directory ([`config_dir`] —
    /// `$VECTORDATA_HOME`, else `~/.config/vectordata`).
    ///
    /// Composes the env-aware [`config_dir`] resolver with the dir-taking
    /// [`Self::configure_optional`] loader; both are tested independently, so
    /// this is correct by construction and needs no env-mutating test.
    pub fn configure_default(self) -> Self {
        self.configure_optional(&config_dir())
    }

    /// Load `catalogs.yaml` from an explicit config directory as *optional*
    /// sources. Having no `catalogs.yaml` is the ordinary case and says
    /// nothing; one that cannot be read or parsed records a warning in
    /// [`diagnostics`](Self::diagnostics). The env-free seam behind
    /// [`Self::configure_default`].
    pub fn configure_optional(mut self, config_dir: &str) -> Self {
        self.add_configured(Configured {
            dir: expand_tilde(config_dir),
            requirement: Requirement::Optional,
            quiet_if_absent: true,
            diagnostics: Vec::new(),
        });
        self
    }

    /// Configure from a specific config directory.
    ///
    /// Loads `catalogs.yaml` from the directory. If the file cannot be
    /// read, records a warning in [`diagnostics`](Self::diagnostics) and
    /// continues.
    pub fn configure(mut self, config_dir: &str) -> Self {
        self.add_configured(Configured {
            dir: expand_tilde(config_dir),
            requirement: Requirement::Required,
            quiet_if_absent: false,
            diagnostics: Vec::new(),
        });
        self
    }

    /// Read a `configure*` call's `catalogs.yaml` and add its catalogs,
    /// its `overlays.yaml` as the user's overlay, and the
    /// `disabled_catalogs` of its `settings.yaml` as catalogs not to
    /// assemble.
    fn add_configured(&mut self, mut configured: Configured) {
        match crate::overlay::Overlay::load_user(Path::new(&configured.dir)) {
            Some(Ok(overlay)) => {
                for w in overlay.warnings() {
                    configured.diagnostics.push(super::CatalogDiagnostic::warning(w.clone()));
                }
                self.overlays.set_user(Some(overlay));
            }
            Some(Err(e)) => configured.diagnostics.push(super::CatalogDiagnostic::warning(e)),
            None => {}
        }
        let index = self.configured.len();
        let absent = !Path::new(&configured.dir).join("catalogs.yaml").is_file();
        if !(absent && configured.quiet_if_absent) {
            match load_config(&configured.dir, &mut configured.diagnostics) {
                Ok(located) => self.extend_located(located, configured.requirement, Some(index)),
                Err(e) => configured.diagnostics.push(super::CatalogDiagnostic::warning(e)),
            }
        }
        if let Ok(settings) = std::fs::read_to_string(Path::new(&configured.dir).join(crate::settings::SETTINGS_FILE)) {
            self.disabled.extend(crate::settings::disabled_catalogs_in(&settings));
        }
        self.configured.push(configured);
    }

    /// The same set with every catalog assembled, the disabled ones
    /// included: what a browser that turns catalogs on and off in the
    /// session reads, so a catalog turned back on has its datasets at
    /// hand. Every other reader uses the set as it is.
    pub fn including_disabled(&self) -> Self {
        CatalogSources { disabled: std::collections::BTreeSet::new(), ..self.clone() }
    }

    /// The same set with the catalogs named `name` assembled though the
    /// user disabled them: what `--catalog <name>` reads.
    pub(crate) fn enabling(mut self, name: &str) -> Self {
        self.disabled.remove(name);
        self
    }

    /// Whether `slot` is assembled: a location given directly always
    /// is; any other catalog unless `disabled` names it. Pure over the
    /// names given.
    fn assembled(slot: &Slot, disabled: &std::collections::BTreeSet<String>) -> bool {
        slot.given || !disabled.contains(slot.source.name())
    }

    /// The same set with every `catalogs.yaml` it was configured from read
    /// again, as after an edit to one: configured catalogs are replaced
    /// where they stood, and every other catalog (a program's, or one
    /// given by location) is kept as it is.
    pub fn reload_configured(&self) -> Self {
        let mut overlays = self.overlays.clone();
        overlays.set_user(None);
        // The disabled names are read again with each configuration.
        let mut out = CatalogSources { diagnostics: self.diagnostics.clone(), overlays, ..CatalogSources::default() };
        let mut reread = vec![false; self.configured.len()];
        for slot in &self.sources {
            match slot.from_config {
                None => out.sources.push(Slot { from_config: None, ..slot.clone() }),
                Some(k) if !reread[k] => {
                    reread[k] = true;
                    out.add_configured(Configured { diagnostics: Vec::new(), ..self.configured[k].clone() });
                }
                Some(_) => {}
            }
        }
        // A configuration that gave no catalogs before may give some now.
        for (k, configured) in self.configured.iter().enumerate() {
            if !reread[k] {
                out.add_configured(Configured { diagnostics: Vec::new(), ..configured.clone() });
            }
        }
        out
    }

    /// An attribute overlay the program states ([`crate::overlay`]),
    /// over the datasets' own attributes and under the user's. A key
    /// another program overlay already sets for the same dataset is
    /// refused, naming both.
    ///
    /// ```
    /// use vectordata::catalog::{CatalogBuilder, CatalogSources};
    /// use vectordata::overlay::Overlay;
    /// let sources = CatalogSources::new()
    ///     .with_overlay(Overlay::program("bench", "datasets:\n  ada002-100k:\n    distance_function: DOT_PRODUCT\n").unwrap())
    ///     .unwrap();
    /// assert!(sources
    ///     .with_overlay(Overlay::program("other", "datasets:\n  ada002-100k:\n    distance_function: COSINE\n").unwrap())
    ///     .is_err());
    /// ```
    pub fn with_overlay(mut self, overlay: crate::overlay::Overlay) -> Result<Self, String> {
        self.overlays.add_program(overlay)?;
        Ok(self)
    }

    /// The overlays in force.
    pub fn overlays(&self) -> &crate::overlay::Overlays {
        &self.overlays
    }

    /// The same set with `overlays` in force instead: what `--at`
    /// keeps when it replaces the catalogs, as it names which catalogs
    /// to read and not what a host knows of datasets.
    pub fn with_overlays_from(mut self, other: &CatalogSources) -> Self {
        self.overlays = other.overlays.clone();
        self
    }

    /// A catalog the program supplies, required: one built with
    /// [`CatalogBuilder`](super::CatalogBuilder), a [`LocatedCatalog`],
    /// or the program's own [`CatalogSource`]. It takes its place in
    /// order after the catalogs already added.
    pub fn with(mut self, source: impl CatalogSource + 'static) -> Self {
        self.sources.push(Slot { source: Arc::new(source), requirement: Requirement::Required, from_config: None, given: false });
        self
    }

    /// A catalog the program supplies, optional: one that gives no
    /// catalog at all is not an error.
    pub fn with_optional(mut self, source: impl CatalogSource + 'static) -> Self {
        self.sources.push(Slot { source: Arc::new(source), requirement: Requirement::Optional, from_config: None, given: false });
        self
    }

    /// This set followed by every catalog of `other`, in its order, each
    /// as required as it was there. `other`'s configured catalogs stay
    /// configured, so [`reload_configured`](Self::reload_configured)
    /// re-reads them where they now stand, and its diagnostics come
    /// along. A name either set disables is disabled in the whole.
    ///
    /// ```
    /// use vectordata::catalog::{CatalogBuilder, CatalogSources};
    /// let program = CatalogSources::new().with(CatalogBuilder::named("mine").build());
    /// let run = CatalogSources::new().with(CatalogBuilder::named("first").build()).extend(program);
    /// let names: Vec<&str> = run.iter().map(|(s, _)| s.name()).collect();
    /// assert_eq!(names, ["first", "mine"]);
    /// ```
    pub fn extend(mut self, other: CatalogSources) -> Self {
        let offset = self.configured.len();
        self.configured.extend(other.configured);
        self.disabled.extend(other.disabled);
        self.sources.extend(other.sources.into_iter().map(|slot| Slot {
            from_config: slot.from_config.map(|k| k + offset),
            ..slot
        }));
        self.diagnostics.extend(other.diagnostics);
        for overlay in other.overlays.iter() {
            match overlay.kind() {
                crate::overlay::OverlayKind::User => self.overlays.set_user(Some(overlay.clone())),
                crate::overlay::OverlayKind::Program(_) => {
                    if let Err(e) = self.overlays.add_program(overlay.clone()) {
                        self.diagnostics.push(super::CatalogDiagnostic::error(e));
                    }
                }
            }
        }
        self
    }

    /// The one catalog of this set named `name`, as a set of its own and
    /// required: what `--at <name>` reads. `Ok(None)` when no catalog
    /// has the name; an error naming each when several do, since a
    /// choice between catalogs is never made by order. A disabled
    /// catalog is found too, and is assembled in the set returned:
    /// naming it is a choice to read it.
    ///
    /// The set holds the catalog alone, with no overlays: the overlays
    /// belong to the set they were registered on, and
    /// [`at_sources`](crate::datasets::at_sources) carries them once
    /// for every catalog it names. A program overlay carried by each of
    /// two named sets would be registered twice when they are joined
    /// with [`extend`](Self::extend), which refuses it as two programs
    /// setting one key (OV-2).
    pub fn named(&self, name: &str) -> Result<Option<CatalogSources>, String> {
        let matches: Vec<&Slot> = self.sources.iter().filter(|slot| slot.source.name() == name).collect();
        match matches.as_slice() {
            [] => Ok(None),
            [slot] => Ok(Some(CatalogSources {
                sources: vec![Slot { requirement: Requirement::Required, from_config: None, ..(*slot).clone() }],
                ..CatalogSources::default()
            })),
            several => Err(format!(
                "catalog name '{name}' names {} catalogs: {}",
                several.len(),
                several.iter().map(|s| s.source.origin().to_string()).collect::<Vec<_>>().join("; "),
            )),
        }
    }

    /// The catalogs assembled, in order, each with whether it must
    /// load: every catalog of the set but those the user disabled
    /// ([`including_disabled`](Self::including_disabled) has them all).
    pub fn iter(&self) -> impl Iterator<Item = (&dyn CatalogSource, Requirement)> + '_ {
        self.sources
            .iter()
            .filter(|slot| Self::assembled(slot, &self.disabled))
            .map(|slot| (slot.source.as_ref(), slot.requirement))
    }

    fn extend_located(&mut self, located: Vec<LocatedCatalog>, requirement: Requirement, from_config: Option<usize>) {
        self.sources.extend(located.into_iter().map(|l| Slot {
            source: Arc::new(l) as Arc<dyn CatalogSource>,
            requirement,
            from_config,
            given: from_config.is_none(),
        }));
    }

    /// Problems met while reading configuration and resolving
    /// locations, such as a directory with no catalog file in it.
    /// [`Catalog::of`](super::Catalog::of) carries these forward into
    /// its own [`diagnostics`](super::Catalog::diagnostics).
    pub fn diagnostics(&self) -> Vec<super::CatalogDiagnostic> {
        self.configured
            .iter()
            .flat_map(|c| c.diagnostics.iter().cloned())
            .chain(self.diagnostics.iter().cloned())
            .collect()
    }

    /// Add required catalog locations.
    ///
    /// Each path can be a directory (containing `catalog.json` or
    /// `catalogs.yaml`), a file path, or an HTTP(S) URL. Sources from
    /// a directory's `catalogs.yaml` keep the names that file gives
    /// them; any other path gets the next free stringified index.
    pub fn add_catalogs(self, paths: &[String]) -> Self {
        self.add_paths(paths, Requirement::Required)
    }

    fn add_paths(mut self, paths: &[String], requirement: Requirement) -> Self {
        for path in paths {
            let expanded = expand_tilde(path);
            for found in resolve_catalog_path(&expanded, &mut self.diagnostics) {
                let located = match found {
                    Found::Configured(located) => located,
                    Found::Bare(location) => LocatedCatalog::as_written(self.synthesize_name(), location, None),
                };
                self.extend_located(vec![located], requirement, None);
            }
        }
        self
    }

    /// Add the catalog at `path` under `name`, required. A directory
    /// holding its own `catalogs.yaml` adds the catalogs that file names,
    /// under its names, as [`add_catalogs`](Self::add_catalogs) does.
    pub fn add_named_catalog(mut self, name: &str, path: &str) -> Self {
        for found in resolve_catalog_path(&expand_tilde(path), &mut self.diagnostics) {
            let located = match found {
                Found::Configured(located) => located,
                Found::Bare(location) => LocatedCatalog::as_written(name, location, None),
            };
            self.extend_located(vec![located], Requirement::Required, None);
        }
        self
    }

    /// Next made-up name for a source whose name can't be determined
    /// (ad-hoc CLI locations): the next free stringified index,
    /// matching the promotion rule for legacy list files.
    fn synthesize_name(&self) -> String {
        let taken: std::collections::HashSet<&str> = self.sources.iter().map(|slot| slot.source.name()).collect();
        let mut n = self.sources.len();
        loop {
            let candidate = n.to_string();
            if !taken.contains(candidate.as_str()) {
                return candidate;
            }
            n += 1;
        }
    }

    /// Add optional catalog locations (failures are silently ignored).
    pub fn add_optional_catalogs(self, paths: &[String]) -> Self {
        self.add_paths(paths, Requirement::Optional)
    }

    /// Returns true if no catalog is assembled: none is configured or
    /// given, or every one is disabled.
    pub fn is_empty(&self) -> bool {
        self.iter().next().is_none()
    }
}

/// Read raw catalog entries from `catalogs.yaml` without resolving them.
///
/// Returns the list of strings exactly as written in the file.
pub fn raw_catalog_entries(config_dir: &str) -> Vec<String> {
    named_catalog_entries(config_dir).into_iter().map(|n| n.location).collect()
}

/// Pure core of the numbered catalog shortcut (`--at 2` → second
/// configured catalog). `None` when `value` isn't a positive integer
/// (it's a literal URL/path); `Some(Ok(url))` when the 1-based index
/// resolves against `configured`; `Some(Err(n))` when the index is
/// out of range.
pub fn lookup_catalog_index(value: &str, configured: &[String]) -> Option<Result<String, usize>> {
    let n: usize = value.parse().ok().filter(|n| *n >= 1)?;
    Some(configured.get(n - 1).cloned().ok_or(n))
}

/// Resolve a `--at`/`--catalog` value: a positive integer is a 1-based
/// index into the configured catalog list (see `config catalog list`);
/// anything else passes through as a literal URL/path. An out-of-range
/// index is an error naming it.
///
/// Idempotent: resolving an already-resolved URL is a no-op, so
/// callers layered above (e.g. binary dispatch arms) may resolve
/// early without harm.
pub fn resolve_catalog_value(value: &str) -> crate::Result<String> {
    let configured = raw_catalog_entries(&config_dir());
    match lookup_catalog_index(value, &configured) {
        None => Ok(value.to_string()),
        Some(Ok(url)) => Ok(url),
        Some(Err(n)) => Err(crate::Error::Other(format!(
            "catalog #{n} not found; list configured catalogs with \
             the `config catalog list` subcommand."
        ))),
    }
}

/// [`resolve_catalog_value`] over a slice — the seam the shared
/// `datasets` run paths use so numbered shortcuts work identically
/// from every binary.
pub fn resolve_catalog_values(values: &[String]) -> crate::Result<Vec<String>> {
    values.iter().map(|v| resolve_catalog_value(v)).collect()
}

/// A configured catalog source: symbolic name + location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedCatalogSource {
    /// Symbolic name — the map key in the map form of
    /// `catalogs.yaml`, or a stringified 0-based index for the legacy
    /// list form (and any other entry whose name can't be
    /// determined).
    pub name: String,
    /// Catalog URL or path. [`named_catalog_entries`] gives it
    /// verbatim from the file; [`CatalogSources`] holds it resolved,
    /// with a relative path joined onto the directory of the
    /// `catalogs.yaml` that lists it.
    pub location: String,
}

/// Read `catalogs.yaml` from a config directory as named sources.
/// Missing/unreadable/malformed files yield an empty list.
pub fn named_catalog_entries(config_dir: &str) -> Vec<NamedCatalogSource> {
    let catalogs_yaml = std::path::Path::new(config_dir).join("catalogs.yaml");
    if !catalogs_yaml.is_file() { return Vec::new(); }
    let content = match std::fs::read_to_string(&catalogs_yaml) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    parse_named_catalogs(&content).unwrap_or_default()
}

/// Parse `catalogs.yaml` content into named sources. Two supported
/// shapes:
///
/// - **Map of names to locations** — the default going forward:
///   ```yaml
///   protected: https://host/path/protected-catalog.yaml
///   local-lab: /data/catalogs/lab
///   ```
/// - **Legacy list of locations** — still accepted; entries promote
///   to stringified 0-based indexes as their names (`0`, `1`, …) in
///   file order:
///   ```yaml
///   - https://host/path/protected-catalog.yaml   # name: "0"
///   ```
///
/// Map keys win as names; anything name-less gets the stringified
/// index so every catalog is addressable.
pub fn parse_named_catalogs(content: &str) -> Result<Vec<NamedCatalogSource>, String> {
    let value: serde_yaml::Value = serde_yaml::from_str(content)
        .map_err(|e| format!("parse catalogs.yaml: {e}"))?;
    let mut out = Vec::new();
    match value {
        serde_yaml::Value::Sequence(seq) => {
            for (i, item) in seq.into_iter().enumerate() {
                let Some(loc) = item.as_str() else { continue };
                out.push(NamedCatalogSource {
                    name: i.to_string(),
                    location: loc.to_string(),
                });
            }
        }
        serde_yaml::Value::Mapping(map) => {
            for (i, (k, v)) in map.into_iter().enumerate() {
                let Some(loc) = v.as_str() else { continue };
                let name = k.as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| i.to_string());
                out.push(NamedCatalogSource { name, location: loc.to_string() });
            }
        }
        serde_yaml::Value::Null => {}
        other => {
            return Err(format!(
                "catalogs.yaml must be a list of locations or a map of name: location, got {other:?}"));
        }
    }
    Ok(out)
}

/// Load `catalogs.yaml` from a config directory.
///
/// The file is a map of name → location, or a legacy list of
/// locations. A relative local location is resolved against
/// `config_dir`, not the process's working directory, so a
/// `catalogs.yaml` means the same thing wherever it is read from.
fn load_config(
    config_dir: &str,
    diags: &mut Vec<super::CatalogDiagnostic>,
) -> Result<Vec<LocatedCatalog>, String> {
    let dir = Path::new(config_dir);
    let catalogs_yaml = dir.join("catalogs.yaml");

    if !catalogs_yaml.is_file() {
        return Err(format!(
            "no catalogs.yaml found in {}",
            dir.display()
        ));
    }

    let content = std::fs::read_to_string(&catalogs_yaml)
        .map_err(|e| format!("failed to read {}: {}", catalogs_yaml.display(), e))?;

    let entries = parse_named_catalogs(&content)
        .map_err(|e| format!("failed to parse {}: {e}", catalogs_yaml.display()))?;

    // Resolve each entry relative to the config directory. An entry
    // naming a directory with its own catalogs.yaml yields that file's
    // sources under that file's names.
    let configured_in = catalogs_yaml.to_string_lossy().into_owned();
    let mut located = Vec::new();
    for entry in entries {
        let expanded = relative_to(dir, &expand_tilde(&entry.location));
        for found in resolve_catalog_path(&expanded, diags) {
            located.push(match found {
                Found::Configured(nested) => nested,
                Found::Bare(location) => {
                    LocatedCatalog::as_written(entry.name.clone(), location, Some(configured_in.clone()))
                }
            });
        }
    }

    Ok(located)
}

/// `location` as written in a `catalogs.yaml` in `dir`: a relative
/// local path is joined onto `dir`; URLs and rooted paths are kept. A
/// rooted path (`/x`, or `\x` and `C:\x` on Windows) is not relative to
/// the file, though on Windows only a drive path is absolute.
fn relative_to(dir: &Path, location: &str) -> String {
    if crate::location::has_scheme(location) || Path::new(location).has_root() {
        return location.to_string();
    }
    dir.join(location).to_string_lossy().into_owned()
}

/// What a catalog path resolves to.
enum Found {
    /// A catalog a nested `catalogs.yaml` names, with that file's name
    /// for it.
    Configured(LocatedCatalog),
    /// A location that is itself a catalog; the caller names it.
    Bare(String),
}

/// Resolve a catalog path to concrete URLs or file paths: catalogs a
/// nested `catalogs.yaml` names, under its names, or the path itself.
///
/// Smart detection:
/// - Remote URLs (`http(s)://`, `s3://`) are kept as-is — the
///   resolver routes them through [`crate::transport`] which handles
///   the S3→virtual-hosted-HTTPS rewrite at fetch time.
/// - Directories containing `catalogs.yaml`: load recursively
/// - Directories containing `catalog.json`: use the directory URL
/// - Plain files: use directly
fn resolve_catalog_path(
    path: &str,
    diags: &mut Vec<super::CatalogDiagnostic>,
) -> Vec<Found> {
    // Any URL the shared transport speaks (currently `http://`,
    // `https://`, `s3://`) passes through verbatim — the catalog
    // fetcher normalises on the way to the wire.
    if crate::transport::is_remote_url(path) {
        return vec![Found::Bare(path.to_string())];
    }

    let p = Path::new(path);

    if p.is_dir() {
        // Directory with catalogs.yaml → load recursively
        let catalogs_yaml = p.join("catalogs.yaml");
        if catalogs_yaml.is_file()
            && let Ok(sources) = load_config(path, diags) {
            return sources.into_iter().map(Found::Configured).collect();
        }

        // Directory with catalog.json or catalog.yaml → use it
        let catalog_json = p.join("catalog.json");
        if catalog_json.is_file() {
            return vec![Found::Bare(path.to_string())];
        }
        let catalog_yaml = p.join("catalog.yaml");
        if catalog_yaml.is_file() {
            return vec![Found::Bare(path.to_string())];
        }

        // Fallback: `knn_entries.yaml` (legacy simplified format).
        // Treated as a complete catalog by the resolver — see
        // `super::knn_entries`.
        let knn_entries = p.join("knn_entries.yaml");
        if knn_entries.is_file() {
            return vec![Found::Bare(path.to_string())];
        }

        // Directory without any catalog file
        diags.push(super::CatalogDiagnostic::warning(format!(
            "directory {path} has no catalogs.yaml / catalog.json / catalog.yaml / knn_entries.yaml"
        )));
        return vec![];
    }

    if p.is_file() {
        return vec![Found::Bare(path.to_string())];
    }

    // Path doesn't exist — might be a remote URL without scheme
    vec![Found::Bare(path.to_string())]
}

/// Expand `~` prefix to the user's home directory.
pub fn expand_tilde(path: &str) -> String {
    if (path.starts_with("~/") || path == "~")
        && let Some(home) = home_dir() {
        return format!("{}{}", home, &path[1..]);
    }
    path.to_string()
}

/// The client configuration directory that holds `catalogs.yaml`.
///
/// `$VECTORDATA_HOME` takes precedence over `~/.config/vectordata`,
/// mirroring [`crate::settings::settings_path`] and the credentials store.
/// Routing every catalog read and write through here means that one env
/// var isolates *all* client config — catalogs, settings, and credentials —
/// under a single root, so tests and tutorials never touch the real config.
///
/// This is the thin env-reading wrapper; the resolution logic lives in the
/// pure `config_dir_from` so it can be tested without mutating the
/// process-wide environment.
pub fn config_dir() -> String {
    config_dir_from(std::env::var_os("VECTORDATA_HOME"))
}

/// Pure resolver behind [`config_dir`]: the given `$VECTORDATA_HOME` value if
/// set, else the tilde-expanded `~/.config/vectordata` default.
fn config_dir_from(vectordata_home: Option<std::ffi::OsString>) -> String {
    match vectordata_home {
        Some(root) => root.to_string_lossy().into_owned(),
        None => expand_tilde(DEFAULT_CONFIG_DIR),
    }
}

/// Get the user's home directory.
fn home_dir() -> Option<String> {
    std::env::var("HOME").ok()
}

/// Returns true when `location` ends in `.yaml`, `.yml`, or `.json`
/// — the strong signal that it names a catalog file directly rather
/// than a directory to probe. Matches `DataSetLoaderSimpleMFD`'s
/// `*.{yaml,yml}` discovery rule (jvector parity).
pub fn looks_like_catalog_file(location: &str) -> bool {
    let lower = location.to_ascii_lowercase();
    let trimmed = lower.trim_end_matches('/');
    trimmed.ends_with(".yaml") || trimmed.ends_with(".yml") || trimmed.ends_with(".json")
}

#[cfg(test)]
mod tests {
    /// `(name, location)` of each located catalog in `s` that has
    /// `requirement`, in order.
    fn listed(s: &CatalogSources, requirement: Requirement) -> Vec<(String, String)> {
        s.iter()
            .filter(|(_, r)| *r == requirement)
            .map(|(c, _)| (c.name().to_string(), c.origin().to_string()))
            .collect()
    }

    /// catalogs.yaml accepts both shapes: a map of name → location
    /// (the default going forward) and the legacy list, whose entries
    /// promote to stringified 0-based indexes as names.
    #[test]
    fn named_catalogs_map_and_list_forms() {
        let map_form = "\n# main\nprotected: https://h/p/protected-catalog.yaml\nlab: /data/lab\n";
        let named = super::parse_named_catalogs(map_form).unwrap();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].name, "protected");
        assert_eq!(named[0].location, "https://h/p/protected-catalog.yaml");
        assert_eq!(named[1].name, "lab");

        let list_form = "- https://h/a/\n- https://h/b/\n";
        let named = super::parse_named_catalogs(list_form).unwrap();
        assert_eq!(named[0].name, "0");
        assert_eq!(named[1].name, "1");

        assert!(super::parse_named_catalogs("just a string").is_err());
        assert!(super::parse_named_catalogs("").unwrap().is_empty());
    }

    use super::*;

    #[test]
    fn catalog_index_lookup_rules() {
        let configured = vec!["https://a/".to_string(), "https://b/".to_string()];
        assert_eq!(lookup_catalog_index("https://a/", &configured), None);
        assert_eq!(lookup_catalog_index("0", &configured), None);
        assert_eq!(lookup_catalog_index("1", &configured), Some(Ok("https://a/".to_string())));
        assert_eq!(lookup_catalog_index("3", &configured), Some(Err(3)));
    }

    #[test]
    fn test_expand_tilde() {
        let expanded = expand_tilde("~/foo/bar");
        assert!(!expanded.starts_with('~'));
        assert!(expanded.ends_with("/foo/bar"));
    }

    #[test]
    fn test_expand_tilde_no_tilde() {
        assert_eq!(expand_tilde("/absolute/path"), "/absolute/path");
        assert_eq!(expand_tilde("relative/path"), "relative/path");
    }

    #[test]
    fn config_dir_prefers_vectordata_home() {
        // The regression guard: a set $VECTORDATA_HOME wins over ~/.config,
        // so catalog reads/writes stay inside an isolated client home.
        let resolved = config_dir_from(Some(std::ffi::OsString::from("/tmp/iso-home")));
        assert_eq!(resolved, "/tmp/iso-home");
    }

    #[test]
    fn config_dir_falls_back_to_default_when_home_unset() {
        let resolved = config_dir_from(None);
        assert!(resolved.ends_with(".config/vectordata"), "got {resolved}");
        assert!(!resolved.starts_with('~'), "must be tilde-expanded, got {resolved}");
    }

    #[test]
    fn configure_optional_loads_from_the_given_dir_only() {
        // An explicit dir with no catalogs.yaml yields nothing — proving the
        // loader reads the dir it is handed and never a real config elsewhere.
        let empty = tempfile::tempdir().unwrap();
        let s = CatalogSources::new().configure_optional(&empty.path().to_string_lossy());
        assert!(s.is_empty());

        // With a catalogs.yaml in that dir, its entries are loaded.
        std::fs::write(
            empty.path().join("catalogs.yaml"),
            "- https://example.invalid/catalog/\n",
        )
        .unwrap();
        let s = CatalogSources::new().configure_optional(&empty.path().to_string_lossy());
        assert!(
            listed(&s, Requirement::Optional).iter().any(|(_, l)| l.contains("example.invalid")),
            "expected the dir's catalogs.yaml entry, got {s:?}",
        );
    }

    /// **Configuration problems are diagnostics on the sources**, never
    /// stderr: a required config directory without `catalogs.yaml`, a
    /// location that is a directory with no catalog file in it (named
    /// directly or listed in `catalogs.yaml`), and a malformed
    /// `catalogs.yaml` — which the default, optional configuration
    /// reports too. Only an absent optional `catalogs.yaml` says
    /// nothing: having none is the ordinary case.
    #[test]
    fn configuration_problems_are_diagnostics() {
        use super::super::Severity;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_string_lossy().to_string();

        let s = CatalogSources::new().configure(&path);
        assert_eq!(s.diagnostics().len(), 1, "{:?}", s.diagnostics());
        assert_eq!(s.diagnostics()[0].severity, Severity::Warning);
        assert!(s.diagnostics()[0].message.contains("no catalogs.yaml found"), "{}", s.diagnostics()[0]);
        assert!(CatalogSources::new().configure_optional(&path).diagnostics().is_empty());

        let bare = dir.path().join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        let s = CatalogSources::new().add_catalogs(&[bare.to_string_lossy().to_string()]);
        assert!(s.is_empty());
        assert_eq!(s.diagnostics().len(), 1, "{:?}", s.diagnostics());
        assert!(s.diagnostics()[0].message.contains("has no catalogs.yaml / catalog.json"), "{}", s.diagnostics()[0]);

        std::fs::write(dir.path().join("catalogs.yaml"), format!("lab: {}\n", bare.display())).unwrap();
        let s = CatalogSources::new().configure_optional(&path);
        assert!(s.is_empty());
        assert_eq!(s.diagnostics().len(), 1, "a listed bare directory: {:?}", s.diagnostics());

        std::fs::write(dir.path().join("catalogs.yaml"), "lab: [unclosed\n").unwrap();
        for s in [CatalogSources::new().configure(&path), CatalogSources::new().configure_optional(&path)] {
            assert_eq!(s.diagnostics().len(), 1, "a malformed catalogs.yaml: {:?}", s.diagnostics());
            assert!(s.diagnostics()[0].message.contains("failed to parse"), "{}", s.diagnostics()[0]);
            assert!(s.is_empty());
        }
    }

    #[test]
    fn test_resolve_http_url() {
        let locations = resolve_catalog_path("https://example.com/catalogs", &mut Vec::new());
        assert!(matches!(&locations[..], [Found::Bare(l)] if l == "https://example.com/catalogs"));
    }

    #[test]
    fn test_sources_empty() {
        let sources = CatalogSources::new();
        assert!(sources.is_empty());
    }

    #[test]
    fn test_load_config_from_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = "- /some/catalog\n- https://example.com/data\n";
        std::fs::write(tmp.path().join("catalogs.yaml"), yaml).unwrap();

        let sources = load_config(tmp.path().to_str().unwrap(), &mut Vec::new()).unwrap();
        // /some/catalog doesn't exist so it still gets returned as-is
        assert!(sources.iter().any(|s| s.location() == "/some/catalog"));
        assert!(sources.iter().any(|s| s.location() == "https://example.com/data"));
        // Legacy list entries promote to stringified 0-based indexes.
        assert!(sources.iter().any(|s| s.name() == "0"));
        assert!(sources.iter().any(|s| s.name() == "1"));
        // Each is bound to the catalogs.yaml that names it.
        let file = tmp.path().join("catalogs.yaml").to_string_lossy().into_owned();
        assert!(sources.iter().all(|s| matches!(
            s.origin(),
            super::super::CatalogOrigin::Located { configured_in: Some(c), .. } if c == file
        )));
    }

    #[test]
    fn test_load_config_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let result = load_config(tmp.path().to_str().unwrap(), &mut Vec::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_dir_with_catalog_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("catalog.json"), "[]").unwrap();

        let locations = resolve_catalog_path(tmp.path().to_str().unwrap(), &mut Vec::new());
        assert!(matches!(&locations[..], [Found::Bare(l)] if l == tmp.path().to_str().unwrap()));
    }

    /// A relative entry in catalogs.yaml names a file beside that
    /// catalogs.yaml, whatever the process's working directory is.
    #[test]
    fn a_relative_entry_resolves_against_its_catalogs_yaml() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("catalogs.yaml"), "shipped: shipped-catalog.yaml\n").unwrap();
        std::fs::write(tmp.path().join("shipped-catalog.yaml"), "[]").unwrap();

        let sources = CatalogSources::new().configure(tmp.path().to_str().unwrap());
        let expected = tmp.path().join("shipped-catalog.yaml");
        assert_eq!(
            listed(&sources, Requirement::Required),
            vec![("shipped".to_string(), expected.to_string_lossy().into_owned())],
        );
    }

    /// A directory added ad hoc keeps the names its own catalogs.yaml
    /// gives, through any depth of nesting, rather than being numbered.
    #[test]
    fn a_directory_added_by_path_keeps_its_catalogs_yaml_names() {
        let tmp = tempfile::tempdir().unwrap();
        let inner = tmp.path().join("inner");
        std::fs::create_dir(&inner).unwrap();
        std::fs::write(inner.join("catalogs.yaml"), "deep: deep-catalog.yaml\n").unwrap();
        std::fs::write(inner.join("deep-catalog.yaml"), "[]").unwrap();
        std::fs::write(
            tmp.path().join("catalogs.yaml"),
            "near: https://example.com/near/\nouter: inner\n",
        ).unwrap();

        let sources = CatalogSources::new()
            .add_catalogs(&[tmp.path().to_string_lossy().to_string()]);
        let deep = inner.join("deep-catalog.yaml");
        assert_eq!(listed(&sources, Requirement::Required), vec![
            ("near".to_string(), "https://example.com/near/".to_string()),
            ("deep".to_string(), deep.to_str().unwrap().to_string()),
        ]);
    }

    /// **A reload re-reads the configured catalogs and keeps the rest**:
    /// after an edit to `catalogs.yaml`, the explorer's set has the new
    /// configured catalogs where the old ones stood, and the program's
    /// catalog is still there (SRD RC-21).
    #[test]
    fn a_reload_rereads_the_configuration_and_keeps_the_programs_catalogs() {
        let tmp = tempfile::tempdir().unwrap();
        for d in ["one", "two"] {
            std::fs::create_dir_all(tmp.path().join(d)).unwrap();
            std::fs::write(tmp.path().join(d).join("catalog.json"), "[]").unwrap();
        }
        let config = tmp.path().to_string_lossy().into_owned();
        std::fs::write(tmp.path().join("catalogs.yaml"), "one: one\n").unwrap();
        let program = super::super::CatalogBuilder::named("mine").build();
        let sources = CatalogSources::new().with(program).configure(&config);
        let names = |s: &CatalogSources| s.iter().map(|(c, _)| c.name().to_string()).collect::<Vec<_>>();
        assert_eq!(names(&sources), ["mine", "one"]);

        std::fs::write(tmp.path().join("catalogs.yaml"), "one: one\ntwo: two\n").unwrap();
        let reloaded = sources.reload_configured();
        assert_eq!(names(&reloaded), ["mine", "one", "two"]);
        assert_eq!(names(&sources), ["mine", "one"], "the old set is unchanged");
    }

    /// **A catalog added under a name keeps it**, so what verifies it
    /// and what it reports say the name the user chose, not a number.
    #[test]
    fn a_catalog_added_under_a_name_keeps_it() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("catalog.json"), "[]").unwrap();
        let sources = CatalogSources::new().add_named_catalog("lab", &tmp.path().to_string_lossy());
        assert_eq!(listed(&sources, Requirement::Required)[0].0, "lab");
    }

    /// **A catalog `disabled_catalogs` names is held but not
    /// assembled.** The setting is read from the `settings.yaml` beside
    /// the `catalogs.yaml`; it disables a program catalog of a listed
    /// name too, never a location given directly; `named` finds a
    /// disabled catalog, assembled; `including_disabled` has them all,
    /// and a reload reads the setting again.
    #[test]
    fn a_disabled_catalog_is_held_but_not_assembled() {
        use super::CatalogSources;
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("catalogs.yaml"), "c1: /nowhere/c1\nc2: /nowhere/c2\n").unwrap();
        std::fs::write(dir.join("settings.yaml"), "# mine\ndisabled_catalogs: c2, bench\n").unwrap();
        let config = dir.to_string_lossy().into_owned();
        let names = |s: &CatalogSources| s.iter().map(|(c, _)| c.name().to_string()).collect::<Vec<_>>();
        let program = CatalogSources::new()
            .with(crate::catalog::CatalogBuilder::named("bench").build())
            .with(crate::catalog::CatalogBuilder::named("mine").build());

        let set = CatalogSources::new().configure(&config).extend(program.clone());
        assert_eq!(names(&set), ["c1", "mine"]);
        assert_eq!(names(&set.including_disabled()), ["c1", "c2", "bench", "mine"]);
        assert_eq!(names(&set.named("c2").unwrap().unwrap()), ["c2"], "named, it is read");
        assert_eq!(names(&set.named("bench").unwrap().unwrap()), ["bench"]);
        assert!(!set.is_empty());

        // Order does not matter: the program's catalogs first.
        assert_eq!(names(&program.clone().extend(CatalogSources::new().configure(&config))), ["mine", "c1"]);

        // A location given directly is never disabled, whatever its name.
        let given = CatalogSources::new().configure(&config).add_named_catalog("c2", "/nowhere/else");
        assert_eq!(names(&given), ["c1", "c2"]);
        assert_eq!(given.iter().nth(1).unwrap().0.origin().to_string(), "/nowhere/else");

        // `--catalog c2` enables it where it stands.
        assert_eq!(names(&set.clone().enabling("c2")), ["c1", "c2", "mine"]);

        std::fs::write(dir.join("settings.yaml"), "disabled_catalogs: c1, c2\n").unwrap();
        let reloaded = set.reload_configured();
        assert_eq!(names(&reloaded), ["bench", "mine"], "the setting is read again");
        let all_off = CatalogSources::new().configure(&config);
        assert!(all_off.is_empty(), "every catalog disabled: nothing assembled");
        assert_eq!(names(&all_off.including_disabled()), ["c1", "c2"]);
    }

    #[test]
    fn test_configure_add_catalogs() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("catalog.json"), "[]").unwrap();

        let sources = CatalogSources::new()
            .add_catalogs(&[tmp.path().to_string_lossy().to_string()]);
        assert_eq!(listed(&sources, Requirement::Required).len(), 1);
    }
}
