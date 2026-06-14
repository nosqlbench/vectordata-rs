// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dataset filtering predicates for `<binary> datasets list`.
//!
//! Each filter option maps to a predicate applied against [`CatalogEntry`]
//! fields. Filters compose conjunctively — all specified filters must match.
//!
//! Dynamic autocompletion of filter values is wired in
//! [`crate::datasets::dyncomp`]; this module supplies the pure
//! domain helpers it needs ([`parse_active_filters`], the
//! `distinct_*` extractors).

use regex::RegexBuilder;

use crate::dataset::source::parse_number_with_suffix;
use crate::dataset::CatalogEntry;

/// Detect if a string looks like a glob pattern rather than a regex.
///
/// Globs use `*` and `?` as wildcards without escaping. Regexes use
/// these differently (quantifiers). A string is treated as a glob if it
/// contains unescaped `*` or `?` but no regex-specific syntax like `(`, `|`, `+`.
fn looks_like_glob(s: &str) -> bool {
    let has_glob_chars = s.contains('*') || s.contains('?');
    let has_regex_chars = s.contains('(') || s.contains('|') || s.contains('+')
        || s.contains('^') || s.contains('$') || s.contains('{');
    has_glob_chars && !has_regex_chars
}

/// Convert a glob pattern to a regex pattern.
///
/// `*` → `.*`, `?` → `.`, everything else is escaped.
/// Prints a note to stderr so the user knows the conversion happened.
fn glob_to_regex_pattern(glob: &str) -> String {
    let mut regex = String::from("(?i)^");
    for ch in glob.chars() {
        match ch {
            '*' => regex.push_str(".*"),
            '?' => regex.push('.'),
            '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '\\' | '^' | '$' | '|' => {
                regex.push('\\');
                regex.push(ch);
            }
            _ => regex.push(ch),
        }
    }
    regex.push('$');
    regex
}

/// Normalize a matching pattern: if it looks like a glob, convert to regex
/// and notify the user. Returns the pattern ready for regex matching.
pub fn normalize_match_pattern(pattern: &str, option_name: &str) -> String {
    if looks_like_glob(pattern) {
        let converted = glob_to_regex_pattern(pattern);
        eprintln!("note: {} '{}' looks like a glob, using regex: {}",
            option_name, pattern, converted);
        converted
    } else {
        pattern.to_string()
    }
}

/// Collected filter predicates parsed from CLI arguments.
///
/// The `--matching-*` options accept regexes (or globs that are auto-converted).
/// Plain strings without regex/glob syntax are matched as case-insensitive substrings.
#[derive(Debug, Default)]
pub struct DatasetFilter {
    pub name: Option<String>,
    pub facet: Vec<String>,
    pub metric: Option<String>,
    pub desc: Option<String>,
    pub min_count: Option<u64>,
    pub max_count: Option<u64>,
    pub count: Option<u64>,
    pub min_dim: Option<u32>,
    pub max_dim: Option<u32>,
    pub dim: Option<u32>,
    pub vtype: Option<String>,
    pub min_data: Option<u64>,
    pub max_data: Option<u64>,
    /// Exact data size as `(bytes, granularity)`: the granularity is
    /// the multiplier of the unit the user spelled (`118MB` →
    /// `(118_000_000, 1_000_000)`), and matching tolerates half a
    /// unit either way. Data sizes are estimates/probes, so byte-for-
    /// byte equality would make every displayed candidate a miss;
    /// unit-granularity matching makes a picked candidate match the
    /// dataset it was derived from. Bare digits carry granularity 1 —
    /// exact equality.
    pub data: Option<(u64, u64)>,
}

impl DatasetFilter {
    /// Returns `true` if no filters are set.
    pub fn is_empty(&self) -> bool {
        self.name.is_none()
            && self.facet.is_empty()
            && self.metric.is_none()
            && self.desc.is_none()
            && self.min_count.is_none()
            && self.max_count.is_none()
            && self.count.is_none()
            && self.min_dim.is_none()
            && self.max_dim.is_none()
            && self.dim.is_none()
            && self.vtype.is_none()
            && self.min_data.is_none()
            && self.max_data.is_none()
            && self.data.is_none()
    }

    /// True when any active predicate may need a facet probe to
    /// decide — i.e. a count, dimension, or data-size filter is set.
    /// Completion uses this to skip warming the probe cache entirely
    /// for pure-offline filters (name/metric/vtype/facet/profile),
    /// which never touch a facet file.
    pub fn needs_probe(&self) -> bool {
        self.min_count.is_some()
            || self.max_count.is_some()
            || self.count.is_some()
            || self.min_dim.is_some()
            || self.max_dim.is_some()
            || self.dim.is_some()
            || self.min_data.is_some()
            || self.max_data.is_some()
            || self.data.is_some()
    }

    /// Test whether a catalog entry passes all filter predicates,
    /// probing the base-vector header / facet sizes live when the
    /// catalog metadata can't answer a dimension/count/data predicate.
    /// This is the command-run path (blocking reads are acceptable).
    pub fn matches(&self, entry: &CatalogEntry) -> bool {
        self.matches_with(entry, ProbeMode::Live)
    }

    /// [`Self::matches`] with no probing: dimension/count/data
    /// predicates see only metadata- and name-derived values, and
    /// entries whose values are only knowable by probing fail those
    /// predicates.
    pub fn matches_offline(&self, entry: &CatalogEntry) -> bool {
        self.matches_with(entry, ProbeMode::Off)
    }

    /// Test all predicates under an explicit [`ProbeMode`]. Completion
    /// passes [`ProbeMode::Cached`] with a pre-warmed
    /// [`FacetProbeCache`] so its narrowing is consistent with what a
    /// real run ([`ProbeMode::Live`]) would match — without blocking
    /// on the network. The cache holds whatever a bounded warm-up
    /// managed to resolve; a facet missing from it reads as "unknown"
    /// (same as offline), never a live fetch.
    pub fn matches_with(&self, entry: &CatalogEntry, mode: ProbeMode) -> bool {
        if let Some(ref name) = self.name
            && !smart_match(name, &entry.name) {
                return false;
            }

        if !self.facet.is_empty() {
            let all_views = collect_all_view_names(entry);
            for required in &self.facet {
                let canonical = resolve_facet_name(required);
                if !all_views.iter().any(|v| v.eq_ignore_ascii_case(&canonical)) {
                    return false;
                }
            }
        }

        if let Some(ref metric) = self.metric {
            // Synonym-aware on both sides: `--with-metric angular`
            // matches an entry whose name says `-cosine`, and vice
            // versa — both canonicalize to COSINE.
            let want = canonicalize_metric(metric);
            match infer_metric(entry) {
                Some(have) if have.eq_ignore_ascii_case(&want) => {}
                _ => return false,
            }
        }

        if let Some(ref desc) = self.desc
            && !matches_description_smart(entry, desc) {
                return false;
            }

        // Size filters: base_count across all profiles, else (when
        // `probe`) base-file bytes / bytes-per-record from the dim
        // header — the same `4 + dim × elem` math ping and the
        // picker's size probes use.
        if self.min_count.is_some() || self.max_count.is_some() || self.count.is_some() {
            let max_base_count =
                max_base_count(entry).or_else(|| probed_base_records(entry, mode));
            if let Some(min) = self.min_count {
                match max_base_count {
                    Some(c) if c >= min => {}
                    _ => return false,
                }
            }
            if let Some(max) = self.max_count {
                match max_base_count {
                    Some(c) if c <= max => {}
                    _ => return false,
                }
            }
            if let Some(exact) = self.count {
                match max_base_count {
                    Some(c) if c == exact => {}
                    _ => return false,
                }
            }
        }

        // Dimension filters: attributes/tags, then name conventions,
        // then (when `probe`) the base-vectors 4-byte dim header.
        if self.min_dim.is_some() || self.max_dim.is_some() || self.dim.is_some() {
            let dim = resolve_dimension(entry, mode);
            if let Some(min) = self.min_dim {
                match dim {
                    Some(d) if d >= min => {}
                    _ => return false,
                }
            }
            if let Some(max) = self.max_dim {
                match dim {
                    Some(d) if d <= max => {}
                    _ => return false,
                }
            }
            if let Some(exact) = self.dim {
                match dim {
                    Some(d) if d == exact => {}
                    _ => return false,
                }
            }
        }

        // Vector type: infer from base_vectors file extension
        if let Some(ref vtype) = self.vtype {
            let detected = infer_vtype(entry);
            match detected {
                Some(ref t) if t.eq_ignore_ascii_case(vtype) => {}
                _ => return false,
            }
        }

        // Data size filters: estimate from base_count × element_size × facet_count
        if self.min_data.is_some() || self.max_data.is_some() || self.data.is_some() {
            let est = total_data_bytes(entry, mode);
            if let Some(min) = self.min_data {
                match est {
                    Some(b) if b >= min => {}
                    _ => return false,
                }
            }
            if let Some(max) = self.max_data {
                match est {
                    Some(b) if b <= max => {}
                    _ => return false,
                }
            }
            if let Some((target, unit)) = self.data {
                match est {
                    Some(b) if b.abs_diff(target) <= unit / 2 => {}
                    _ => return false,
                }
            }
        }

        true
    }
}

/// Controls which profiles are shown in the output.
///
/// `--profile` limits to an exact name (case-insensitive).
/// `--profile-regex` limits to names matching a wildcard pattern.
/// When neither is set, all profiles are shown.
#[derive(Debug, Default)]
pub struct ProfileView {
    pub pattern: Option<String>,
}

impl ProfileView {
    pub fn new(pattern: Option<String>) -> Self {
        Self { pattern }
    }

    /// Returns `true` if any profile limiting is in effect.
    pub fn is_active(&self) -> bool {
        self.pattern.is_some()
    }

    /// Returns the profile names from `entry` that match the pattern.
    /// If no pattern is set, returns all profiles.
    pub fn matching_profiles<'a>(&self, entry: &'a CatalogEntry) -> Vec<&'a str> {
        let all = entry.profile_names();
        match &self.pattern {
            None => all,
            Some(pat) => all.into_iter()
                .filter(|name| smart_match(pat, name))
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Predicate helpers
// ---------------------------------------------------------------------------

/// Collect all view (facet) names from all profiles in an entry.
fn collect_all_view_names(entry: &CatalogEntry) -> Vec<String> {
    let mut names = Vec::new();
    for (_, profile) in &entry.layout.profiles.profiles {
        for key in profile.views.keys() {
            if !names.iter().any(|n: &String| n.eq_ignore_ascii_case(key)) {
                names.push(key.clone());
            }
        }
    }
    names
}

/// Resolve a user-provided facet name to its canonical form. Filter
/// vocabulary accepts capital facet codes (`B`, `Q`, `G`, …) in
/// addition to keys and aliases; the code table is the facet spec's
/// ([`StandardFacet::from_code`]), not duplicated here.
fn resolve_facet_name(name: &str) -> String {
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next())
        && c.is_ascii_uppercase()
        && let Some(facet) = crate::dataset::facet::StandardFacet::from_code(c)
    {
        return facet.key().to_string();
    }
    crate::dataset::facet::resolve_standard_key(name)
        .unwrap_or_else(|| name.to_string())
}

/// Get the maximum base_count across all profiles, falling back to the
/// default profile's base_vectors window size.
pub fn max_base_count(entry: &CatalogEntry) -> Option<u64> {
    let mut max: Option<u64> = None;
    for (_, profile) in &entry.layout.profiles.profiles {
        if let Some(bc) = profile.base_count {
            max = Some(max.map_or(bc, |m: u64| m.max(bc)));
        }
        if let Some(view) = profile.views.get("base_vectors") {
            let w = view.effective_window();
            if !w.is_empty() {
                for interval in &w.0 {
                    let count = interval.max_excl.saturating_sub(interval.min_incl);
                    max = Some(max.map_or(count, |m: u64| m.max(count)));
                }
            }
        }
    }
    max
}

/// Canonicalize a metric token to the names the rest of the
/// toolchain speaks (`COSINE` / `L2` / `DOT_PRODUCT` / `L1` /
/// `HAMMING` / `JACCARD`). Unrecognized tokens pass through
/// uppercased so declared-but-unknown metrics still display and
/// match as themselves.
pub fn canonicalize_metric(token: &str) -> String {
    metric_synonym(token)
        .map(str::to_string)
        .unwrap_or_else(|| token.to_uppercase())
}

/// The synonym table behind [`canonicalize_metric`]. `None` for
/// tokens that don't name a metric at all — which is what lets
/// [`infer_metric`] scan name tokens without false positives.
fn metric_synonym(token: &str) -> Option<&'static str> {
    match token.to_ascii_lowercase().as_str() {
        "cosine" | "angular" => Some("COSINE"),
        "euclidean" | "l2" => Some("L2"),
        "dot" | "dot_product" | "dotproduct" | "ip" | "inner_product" | "mips" => {
            Some("DOT_PRODUCT")
        }
        "l1" | "manhattan" | "taxicab" => Some("L1"),
        "hamming" => Some("HAMMING"),
        "jaccard" => Some("JACCARD"),
        _ => None,
    }
}

/// Infer the distance metric of an entry, canonicalized via
/// [`canonicalize_metric`]. Sources, in priority order:
///
/// 1. `attributes.distance_function` (the canonical catalog field);
/// 2. attribute tags named `metric`, `distance_metric`, or
///    `distance_function`;
/// 3. the dataset-name convention used by the ann-benchmarks corpus
///    (`glove-25-angular`, `sift-128-euclidean`, `lastfm-64-dot`).
pub fn infer_metric(entry: &CatalogEntry) -> Option<String> {
    if let Some(ref attrs) = entry.layout.attributes {
        if let Some(ref df) = attrs.distance_function {
            return Some(canonicalize_metric(df));
        }
        for (k, v) in &attrs.tags {
            if k.eq_ignore_ascii_case("metric")
                || k.eq_ignore_ascii_case("distance_metric")
                || k.eq_ignore_ascii_case("distance_function")
            {
                return Some(canonicalize_metric(v));
            }
        }
    }
    entry
        .name
        .split(['-', '_'])
        .find_map(metric_synonym)
        .map(str::to_string)
}

/// A probed facet file: its byte length, and — for xvec files — the
/// dimension from the 4-byte little-endian header. Either field is
/// `None` when the probe couldn't determine it. Serializable so the
/// completion layer can memoize a [`FacetProbeCache`] to disk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FacetProbe {
    pub bytes: Option<u64>,
    pub dim: Option<u32>,
}

/// Pre-resolved facet probes keyed by facet file path/URL. Built by a
/// bounded warm-up in the completion layer and consulted via
/// [`ProbeMode::Cached`] so filter matching there is probe-consistent
/// with a real run without blocking.
pub type FacetProbeCache = std::collections::BTreeMap<String, FacetProbe>;

/// How a filter predicate resolves a value the catalog metadata can't
/// supply (base-vector dimension, record count, total bytes).
#[derive(Clone, Copy)]
pub enum ProbeMode<'a> {
    /// Metadata + name inference only; never touch a file.
    Off,
    /// Read the file live (local stat/read or authed HEAD/range) —
    /// blocking. The command-run path.
    Live,
    /// Read probe values from this pre-warmed cache only; a cache miss
    /// reads as "unknown", never a live fetch. The completion path.
    Cached(&'a FacetProbeCache),
}

/// Byte length of a facet file under `mode`: `None` when off or a
/// cache miss; a live stat/HEAD under [`ProbeMode::Live`].
fn facet_bytes(path: &str, mode: ProbeMode) -> Option<u64> {
    match mode {
        ProbeMode::Off => None,
        ProbeMode::Live => facet_len(path),
        ProbeMode::Cached(cache) => cache.get(path).and_then(|p| p.bytes),
    }
}

/// Base-vector dimension from a facet's 4-byte header under `mode`:
/// `None` when off or a cache miss; a live 4-byte read under
/// [`ProbeMode::Live`].
fn facet_dim(path: &str, mode: ProbeMode) -> Option<u32> {
    match mode {
        ProbeMode::Off => None,
        ProbeMode::Live => probe_facet_live(path).dim,
        ProbeMode::Cached(cache) => cache.get(path).and_then(|p| p.dim),
    }
}

/// Live single-facet probe: byte length (local stat / authed HEAD)
/// plus, for an xvec-extension file, the validated dim from its
/// 4-byte little-endian header (local read / authed 4-byte range).
/// This is the unit the completion warm-up runs per facet so that the
/// resulting [`FacetProbeCache`] answers every probe predicate.
pub fn probe_facet_live(path: &str) -> FacetProbe {
    let bytes = facet_len(path);
    let dim = probe_facet_dim_header(path);
    FacetProbe { bytes, dim }
}

/// Read and validate the 4-byte dim header of an xvec file. `None`
/// for non-xvec extensions (npy/hdf5/parquet carry no such header) or
/// any read failure.
fn probe_facet_dim_header(path: &str) -> Option<u32> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())?
        .to_lowercase();
    if !ext.ends_with("vec") && !ext.ends_with("vecs") {
        return None;
    }
    let header: [u8; 4] = if crate::transport::is_remote_url(path) {
        use crate::transport::ChunkedTransport;
        let normalized = crate::transport::normalize_remote_url(path);
        let parsed = url::Url::parse(normalized.as_ref()).ok()?;
        let transport = crate::transport::HttpTransport::new(parsed);
        if !transport.supports_range() {
            return None;
        }
        transport.fetch_range(0, 4).ok()?.as_slice().try_into().ok()?
    } else {
        use std::io::Read;
        let mut file = std::fs::File::open(path).ok()?;
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf).ok()?;
        buf
    };
    let dim = i32::from_le_bytes(header);
    (dim > 0 && dim <= 1_000_000).then_some(dim as u32)
}

/// Infer dimensionality from attributes/tags or the dataset name.
/// Offline-only — see [`resolve_dimension`] for the probing variant.
pub fn infer_dimension(entry: &CatalogEntry) -> Option<u32> {
    if let Some(ref attrs) = entry.layout.attributes {
        for (k, v) in &attrs.tags {
            if (k.eq_ignore_ascii_case("dimension")
                || k.eq_ignore_ascii_case("dim")
                || k.eq_ignore_ascii_case("dimensions"))
                && let Ok(d) = v.parse::<u32>() {
                    return Some(d);
                }
        }
    }
    extract_dim_from_name(&entry.name)
}

/// Dimensionality with a last-resort header probe: [`infer_dimension`]
/// first; failing that, the base-vector facet's 4-byte dim header via
/// `mode` (a live read under [`ProbeMode::Live`], the warm cache under
/// [`ProbeMode::Cached`], nothing under [`ProbeMode::Off`]).
pub fn resolve_dimension(entry: &CatalogEntry, mode: ProbeMode) -> Option<u32> {
    infer_dimension(entry).or_else(|| {
        let path = base_vectors_path(entry)?;
        facet_dim(&path, mode)
    })
}

/// Extract dimension from a dataset name. Two conventions:
/// the pipeline's `_dNNN` token (`emb-base-v2_d768_b10000`) and the
/// ann-benchmarks bare-number token (`glove-25-angular`,
/// `openai-v3-large-3072-100k`). Count-ish tokens (`100k`, `1M`)
/// carry a suffix and never match the bare-number rule.
fn extract_dim_from_name(name: &str) -> Option<u32> {
    for part in name.split('_') {
        if let Some(num_str) = part.strip_prefix('d')
            && !num_str.is_empty() && num_str.chars().all(|c| c.is_ascii_digit())
                && let Ok(d) = num_str.parse::<u32>()
                    && d > 0 && d < 100_000 {
                        return Some(d);
                    }
    }
    for part in name.split(['-', '_']) {
        // Leading zeros mark version-ish tokens (`emb_002`), not dims.
        if !part.is_empty()
            && !part.starts_with('0')
            && part.chars().all(|c| c.is_ascii_digit())
            && let Ok(d) = part.parse::<u32>()
            && (2..100_000).contains(&d)
        {
            return Some(d);
        }
    }
    None
}

/// The absolute URL/path of the entry's base-vectors facet (first
/// profile that declares one), canonical name first, legacy `base`
/// alias second. Resolved through [`CatalogEntry::resolve_facet_url`]
/// so a `dataset.yaml`-shaped catalog's relative source becomes the
/// real URL the probe reads — not a bare relative string mistaken for
/// a local file.
pub(crate) fn base_vectors_path(entry: &CatalogEntry) -> Option<String> {
    for (_, profile) in &entry.layout.profiles.profiles {
        if let Some(view) = profile
            .views
            .get("base_vectors")
            .or_else(|| profile.views.get("base"))
        {
            return entry.resolve_facet_url(view.path());
        }
    }
    None
}

/// Infer vector data type from base_vectors file extension.
pub fn infer_vtype(entry: &CatalogEntry) -> Option<String> {
    if entry.dataset_type.eq_ignore_ascii_case("hdf5") {
        return Some("hdf5".to_string());
    }
    for (_, profile) in &entry.layout.profiles.profiles {
        if let Some(view) = profile.views.get("base_vectors") {
            return vtype_from_extension(view.path());
        }
        if let Some(view) = profile.views.get("base") {
            return vtype_from_extension(view.path());
        }
    }
    None
}

/// Map a file extension to a vector type name. Local table —
/// avoids depending on `veks-core` from the `vectordata` library
/// (which would be a circular dep). Kept in sync with
/// `veks-core::formats::VecFormat::from_extension` by extension.
fn vtype_from_extension(path: &str) -> Option<String> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())?
        .to_lowercase();
    let name = match ext.as_str() {
        "fvec" | "fvecs" => "float32",
        "mvec" | "mvecs" => "float16",
        "bvec" | "bvecs" => "uint8",
        "i8vec" | "i8vecs" => "int8",
        "ivec" | "ivecs" => "int32",
        "dvec" | "dvecs" => "float64",
        "svec" | "svecs" => "int16",
        "u16vec" | "u16vecs" => "uint16",
        "u32vec" | "u32vecs" => "uint32",
        "i64vec" | "i64vecs" => "int64",
        "u64vec" | "u64vecs" => "uint64",
        "npy" => "numpy",
        "h5" | "hdf5" => "hdf5",
        "parquet" => "parquet",
        _ => return None,
    };
    Some(name.to_string())
}

/// Total data bytes of an entry — the real on-disk/remote footprint.
///
/// Prefers the summed byte sizes of the entry's distinct facet files
/// (via `mode`'s probe), which is what `--with-data` means: how much
/// data the dataset actually weighs. Only when nothing can be probed
/// (`ProbeMode::Off`, or every probe failed) does it fall back to the
/// `base_count × dim × element_size` estimate — and that estimate
/// covers BASE VECTORS ONLY and uses the (possibly windowed)
/// `base_count`, so it badly under-reports a dataset with large
/// query / ground-truth / metadata facets. The fallback exists so an
/// offline caller still gets a rough number, never as the primary
/// answer when real sizes are available.
fn total_data_bytes(entry: &CatalogEntry, mode: ProbeMode) -> Option<u64> {
    probed_data_bytes(entry, mode).or_else(|| {
        let dim = resolve_dimension(entry, mode).unwrap_or(0) as u64;
        if dim == 0 {
            return None;
        }
        let elem_bytes = infer_vtype(entry)
            .as_deref()
            .and_then(vtype_elem_bytes)
            .unwrap_or(4);
        let base_count = max_base_count(entry)?;
        Some(base_count * dim * elem_bytes)
    })
}

/// Per-element byte width for an xvec-family vector type name.
/// `None` for container formats (numpy/hdf5/parquet) whose record
/// layout isn't dim-header-derivable.
fn vtype_elem_bytes(vtype: &str) -> Option<u64> {
    match vtype {
        "uint8" | "int8" => Some(1),
        "float16" | "int16" | "uint16" => Some(2),
        "float32" | "int32" | "uint32" => Some(4),
        "float64" | "int64" | "uint64" => Some(8),
        _ => None,
    }
}

/// Byte length of one facet file: a local stat, or an authed HEAD
/// via the unified transport for remote URLs. Live read — predicate
/// callers route through [`facet_bytes`] to honor a [`ProbeMode`];
/// the completion warm-up calls this directly for bytes-only facets.
pub(crate) fn facet_len(path: &str) -> Option<u64> {
    if crate::transport::is_remote_url(path) {
        use crate::transport::ChunkedTransport;
        let normalized = crate::transport::normalize_remote_url(path);
        let parsed = url::Url::parse(normalized.as_ref()).ok()?;
        let transport = crate::transport::HttpTransport::new(parsed);
        let len = transport.content_length().ok()?;
        (len > 0).then_some(len)
    } else {
        std::fs::metadata(path).ok().map(|m| m.len()).filter(|l| *l > 0)
    }
}

/// Record count of the base vectors derived from file bytes and the
/// dim header: `bytes / (4 + dim × elem)`. Only sound for uniform
/// xvec facets, which is what the element-size gate enforces. Probe
/// values come from `mode`.
fn probed_base_records(entry: &CatalogEntry, mode: ProbeMode) -> Option<u64> {
    let path = base_vectors_path(entry)?;
    let elem = infer_vtype(entry).as_deref().and_then(vtype_elem_bytes)?;
    let dim = resolve_dimension(entry, mode)? as u64;
    let total = facet_bytes(&path, mode)?;
    let bytes_per_record = 4 + dim * elem;
    Some(total / bytes_per_record)
}

/// The distinct absolute facet URLs/paths of an entry across all
/// profiles, resolved through [`CatalogEntry::resolve_facet_url`].
/// Distinct so profiles sharing a base file (or a window into it,
/// which resolves to the same stripped path) don't double-count.
pub(crate) fn entry_facet_paths(entry: &CatalogEntry) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for (_, profile) in &entry.layout.profiles.profiles {
        for view in profile.views.values() {
            if let Some(path) = entry.resolve_facet_url(view.path())
                && seen.insert(path.clone())
            {
                out.push(path);
            }
        }
    }
    out
}

/// Sum of the entry's distinct facet file sizes under `mode` — the
/// ground-truth data weight when catalog metadata carries no counts.
fn probed_data_bytes(entry: &CatalogEntry, mode: ProbeMode) -> Option<u64> {
    let mut total = 0u64;
    let mut any = false;
    for path in entry_facet_paths(entry) {
        if let Some(len) = facet_bytes(&path, mode) {
            total += len;
            any = true;
        }
    }
    any.then_some(total)
}

/// Strip surrounding single or double quotes from a string.
///
/// During shell completion, `COMP_WORDS` preserves the raw quotes typed by
/// the user (e.g. `'.*foo.*'`). We need to strip them so that
/// `parse_active_filters` sees the unquoted value.
fn strip_shell_quotes(s: &str) -> String {
    if (s.starts_with('\'') && s.ends_with('\''))
        || (s.starts_with('"') && s.ends_with('"'))
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Match `text` against a regular expression `pattern` (case-insensitive).
///
/// The pattern is treated as a standard regex. If the pattern is invalid,
/// it is treated as a literal substring match.
fn simple_match(pattern: &str, text: &str) -> bool {
    match RegexBuilder::new(pattern).case_insensitive(true).build() {
        Ok(re) => re.is_match(text),
        Err(_) => text.to_lowercase().contains(&pattern.to_lowercase()),
    }
}

/// Smart match: if the pattern has regex syntax, use regex matching.
/// Otherwise, use case-insensitive substring matching.
/// Glob patterns have already been converted to regex by `normalize_match_pattern`.
fn smart_match(pattern: &str, text: &str) -> bool {
    // If it looks like a regex (has anchors, quantifiers, groups, etc.), use regex
    let is_regex = pattern.contains('(') || pattern.contains('|') || pattern.contains('+')
        || pattern.contains('^') || pattern.contains('$') || pattern.contains('{')
        || pattern.contains('.') && pattern.contains('*');
    if is_regex {
        simple_match(pattern, text)
    } else {
        // Plain substring, case-insensitive
        text.to_lowercase().contains(&pattern.to_lowercase())
    }
}

/// Smart match against description-like fields (name, notes, model, tags).
fn matches_description_smart(entry: &CatalogEntry, pattern: &str) -> bool {
    let mut texts = vec![entry.name.clone()];
    if let Some(ref attrs) = entry.layout.attributes {
        if let Some(ref notes) = attrs.notes {
            texts.push(notes.clone());
        }
        if let Some(ref model) = attrs.model {
            texts.push(model.clone());
        }
        for v in attrs.tags.values() {
            texts.push(v.clone());
        }
    }
    texts.iter().any(|t| smart_match(pattern, t))
}

/// Parse a size value that supports K/M/B suffixes.
pub fn parse_size(s: &str) -> Result<u64, String> {
    parse_number_with_suffix(s)
}

/// Parse a byte size value with K/M/G/T suffixes.
pub fn parse_bytes(s: &str) -> Result<u64, String> {
    parse_number_with_suffix(s)
}

/// Parse a byte size value, also returning the granularity of the
/// unit the user spelled (`118MB` → `(118_000_000, 1_000_000)`).
/// Backs the tolerance semantics of the exact `--with-data` filter.
pub fn parse_bytes_with_unit(s: &str) -> Result<(u64, u64), String> {
    crate::dataset::source::parse_number_with_suffix_unit(s)
}

// ---------------------------------------------------------------------------
// Completion support — pure helpers over context words + entries
// ---------------------------------------------------------------------------
//
// The dynamic completers themselves are registered in
// [`crate::datasets::dyncomp`]; what belongs here is the filter
// domain knowledge they need: parsing the filters already typed on
// the line, and extracting the distinct filterable values present in
// a set of entries. Everything is pure over explicit inputs — the
// engine-provided context words, NOT process argv (during a
// completion callback argv is just `[binary, $COMP_LINE,
// $COMP_POINT]`, so scanning `std::env::args()` finds no flags).

/// Parse already-specified filter and profile options from the words
/// typed so far (the completion engine's context slice). Both
/// `--key value` and `--key=value` forms are honored; valueless or
/// malformed options are ignored. Each completer filters its
/// candidate values through the result so suggestions stay
/// consistent with the narrowing already on the line.
pub fn parse_active_filters(words: &[&str]) -> (DatasetFilter, ProfileView) {
    let mut filter = DatasetFilter::default();
    let mut profile: Option<String> = None;

    let mut i = 0;
    while i < words.len() {
        let (key, inline_val) = match words[i].find('=') {
            Some(eq) => (&words[i][..eq], Some(words[i][eq + 1..].to_string())),
            None => (words[i], None),
        };
        let value = inline_val
            .clone()
            .or_else(|| words.get(i + 1).map(|w| w.to_string()))
            .filter(|v| !v.is_empty())
            .map(|v| strip_shell_quotes(&v))
            .filter(|v| !v.is_empty());
        let advance = if inline_val.is_some() { 0 } else { 1 };

        if let Some(v) = value {
            let mut consumed = true;
            match key {
                "--dataset" | "-d" | "--matching-name" => filter.name = Some(v),
                "--with-facet" => filter.facet.push(v),
                "--with-metric" => filter.metric = Some(v),
                "--matching-desc" => filter.desc = Some(v),
                "--with-min-count" => filter.min_count = parse_size(&v).ok(),
                "--with-max-count" => filter.max_count = parse_size(&v).ok(),
                "--with-count" => filter.count = parse_size(&v).ok(),
                "--with-min-dim" => filter.min_dim = v.parse().ok(),
                "--with-max-dim" => filter.max_dim = v.parse().ok(),
                "--with-dim" => filter.dim = v.parse().ok(),
                "--with-vtype" => filter.vtype = Some(v),
                "--with-min-data" => filter.min_data = parse_bytes(&v).ok(),
                "--with-max-data" => filter.max_data = parse_bytes(&v).ok(),
                "--with-data" => filter.data = parse_bytes_with_unit(&v).ok(),
                "--matching-profile" => profile = Some(v),
                _ => consumed = false,
            }
            if consumed {
                i += advance;
            }
        }
        i += 1;
    }

    (filter, ProfileView::new(profile))
}

/// Distinct canonical metrics present across `entries`, sorted.
pub fn distinct_metrics(entries: &[&CatalogEntry]) -> Vec<String> {
    let set: std::collections::BTreeSet<String> =
        entries.iter().filter_map(|e| infer_metric(e)).collect();
    set.into_iter().collect()
}

/// Distinct vector data types present across `entries`, sorted.
pub fn distinct_vtypes(entries: &[&CatalogEntry]) -> Vec<String> {
    let set: std::collections::BTreeSet<String> =
        entries.iter().filter_map(|e| infer_vtype(e)).collect();
    set.into_iter().collect()
}

/// Distinct facet (view) names present across `entries`, sorted.
pub fn distinct_facets(entries: &[&CatalogEntry]) -> Vec<String> {
    let set: std::collections::BTreeSet<String> = entries
        .iter()
        .flat_map(|e| collect_all_view_names(e))
        .collect();
    set.into_iter().collect()
}

/// Distinct dimensionalities present across `entries` under `mode`,
/// ascending. Candidates for the `--with-dim` family.
pub fn distinct_dims(entries: &[&CatalogEntry], mode: ProbeMode) -> Vec<String> {
    let set: std::collections::BTreeSet<u32> =
        entries.iter().filter_map(|e| resolve_dimension(e, mode)).collect();
    set.into_iter().map(|d| d.to_string()).collect()
}

/// Distinct record counts present across `entries` under `mode`, in
/// compact suffix form (`100k`, `1m`), ascending. Candidates for the
/// `--with-count` family.
pub fn distinct_base_counts(entries: &[&CatalogEntry], mode: ProbeMode) -> Vec<String> {
    let set: std::collections::BTreeSet<u64> = entries
        .iter()
        .filter_map(|e| max_base_count(e).or_else(|| probed_base_records(e, mode)))
        .collect();
    set.into_iter().map(format_count_suffix).collect()
}

/// Distinct data sizes present across `entries` under `mode`, in raw
/// bytes, ascending. Feeds the `--with-data` family's candidates (the
/// completion layer formats them as byte sizes).
pub fn distinct_data_sizes(entries: &[&CatalogEntry], mode: ProbeMode) -> Vec<u64> {
    let set: std::collections::BTreeSet<u64> = entries
        .iter()
        .filter_map(|e| total_data_bytes(e, mode))
        .collect();
    set.into_iter().collect()
}

/// Render a byte count for the data axis: the largest decimal byte
/// unit (`TB`/`GB`/`MB`/`KB`) whose rounded quotient is at least 10,
/// so values keep 2–4 significant digits and read unmistakably as
/// bytes (`118MB`, `12GB`) rather than counts (`100k`). Rounding
/// error is at most half the displayed unit — exactly the tolerance
/// the `--with-data` filter applies, so a displayed candidate always
/// matches the dataset it came from.
pub fn format_bytes_approx(n: u64) -> String {
    for (mult, suffix) in [
        (1_000_000_000_000u64, "TB"),
        (1_000_000_000, "GB"),
        (1_000_000, "MB"),
        (1_000, "KB"),
    ] {
        let rounded = (n + mult / 2) / mult;
        if rounded >= 10 {
            return format!("{rounded}{suffix}");
        }
    }
    n.to_string()
}

/// Render `n` in the most compact form [`parse_number_with_suffix`]
/// reads back exactly: the largest decimal suffix (`t`/`g`/`m`/`k`)
/// that divides it evenly, else plain digits. Lossless by
/// construction — candidates must filter precisely as displayed.
fn format_count_suffix(n: u64) -> String {
    for (mult, suffix) in [
        (1_000_000_000_000u64, "t"),
        (1_000_000_000, "g"),
        (1_000_000, "m"),
        (1_000, "k"),
    ] {
        if n > 0 && n.is_multiple_of(mult) {
            return format!("{}{}", n / mult, suffix);
        }
    }
    n.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{CatalogLayout, DSProfile, DSProfileGroup};
    use indexmap::IndexMap;

    fn entry_with_views(name: &str, views: &[&str]) -> CatalogEntry {
        let mut view_map = IndexMap::new();
        for v in views {
            view_map.insert(
                v.to_string(),
                crate::dataset::DSView {
                    source: crate::dataset::DSSource {
                        path: format!("{}.fvec", v),
                        namespace: None,
                        window: crate::dataset::source::DSWindow::default(),
                    },
                    window: None,
                },
            );
        }
        let mut profiles = IndexMap::new();
        profiles.insert(
            "default".to_string(),
            DSProfile {
                maxk: Some(100),
                base_count: None,
                partition: false,
                views: view_map,
            },
        );
        CatalogEntry {
            name: name.to_string(),
            path: format!("{}/dataset.yaml", name),
            dataset_type: "dataset.yaml".to_string(),
            catalog_file: None,
            catalog_name: None,
            layout: CatalogLayout {
                attributes: None,
                profiles: DSProfileGroup::from_profiles(profiles),
            },
        }
    }

    fn entry_with_attrs(name: &str, metric: &str) -> CatalogEntry {
        let mut e = entry_with_views(name, &["base_vectors", "query_vectors"]);
        e.layout.attributes = Some(crate::dataset::DatasetAttributes {
            distance_function: Some(metric.to_string()),
            ..Default::default()
        });
        e
    }

    #[test]
    fn test_filter_name_substring() {
        let f = DatasetFilter {
            name: Some("vecs".to_string()),
            ..Default::default()
        };
        let entry = entry_with_views("vecs-128", &["base_vectors"]);
        assert!(f.matches(&entry));

        let entry2 = entry_with_views("glove-100", &["base_vectors"]);
        assert!(!f.matches(&entry2));
    }

    #[test]
    fn test_filter_name_regex() {
        let f = DatasetFilter {
            name: Some("vecs.*".to_string()),
            ..Default::default()
        };
        let entry = entry_with_views("vecs-128", &["base_vectors"]);
        assert!(f.matches(&entry));
    }

    #[test]
    fn test_filter_facet() {
        let f = DatasetFilter {
            facet: vec!["base_vectors".to_string()],
            ..Default::default()
        };
        let entry = entry_with_views("test", &["base_vectors", "query_vectors"]);
        assert!(f.matches(&entry));

        let f2 = DatasetFilter {
            facet: vec!["metadata_content".to_string()],
            ..Default::default()
        };
        assert!(!f2.matches(&entry));
    }

    #[test]
    fn test_filter_facet_alias() {
        let f = DatasetFilter {
            facet: vec!["base".to_string()],
            ..Default::default()
        };
        let entry = entry_with_views("test", &["base_vectors"]);
        assert!(f.matches(&entry));
    }

    #[test]
    fn test_filter_metric() {
        let f = DatasetFilter {
            metric: Some("COSINE".to_string()),
            ..Default::default()
        };
        let entry = entry_with_attrs("test", "COSINE");
        assert!(f.matches(&entry));

        let entry2 = entry_with_attrs("test2", "L2");
        assert!(!f.matches(&entry2));
    }

    #[test]
    fn test_filter_metric_case_insensitive() {
        let f = DatasetFilter {
            metric: Some("cosine".to_string()),
            ..Default::default()
        };
        let entry = entry_with_attrs("test", "COSINE");
        assert!(f.matches(&entry));
    }

    #[test]
    fn test_infer_dimension_from_name() {
        // `_dNNN` pipeline convention wins over bare numbers.
        assert_eq!(extract_dim_from_name("emb_002_d1536_b10000_q10000_mk100"), Some(1536));
        assert_eq!(extract_dim_from_name("emb-base-v2_d768_b10000"), Some(768));
        // ann-benchmarks bare-number convention.
        assert_eq!(extract_dim_from_name("vecs-128"), Some(128));
        assert_eq!(extract_dim_from_name("glove-25-angular"), Some(25));
        assert_eq!(extract_dim_from_name("sift-128-euclidean"), Some(128));
        assert_eq!(extract_dim_from_name("openai-v3-large-3072-100k"), Some(3072));
        // Count suffixes, version tokens, and dimensionless names stay unknown.
        assert_eq!(extract_dim_from_name("ada002-100k"), None);
        assert_eq!(extract_dim_from_name("colbert-1M"), None);
        assert_eq!(extract_dim_from_name("e5-base-v2-100k"), None);
        assert_eq!(extract_dim_from_name("emb_002_b10000"), None);
        assert_eq!(extract_dim_from_name("no-dimension-here"), None);
    }

    #[test]
    fn metric_canonicalization_and_synonyms() {
        assert_eq!(canonicalize_metric("angular"), "COSINE");
        assert_eq!(canonicalize_metric("Cosine"), "COSINE");
        assert_eq!(canonicalize_metric("euclidean"), "L2");
        assert_eq!(canonicalize_metric("L2"), "L2");
        assert_eq!(canonicalize_metric("dot"), "DOT_PRODUCT");
        assert_eq!(canonicalize_metric("ip"), "DOT_PRODUCT");
        // Declared-but-unknown metrics pass through uppercased.
        assert_eq!(canonicalize_metric("chebyshev"), "CHEBYSHEV");
    }

    #[test]
    fn infer_metric_from_name_convention() {
        let e = entry_with_views("glove-25-angular", &["base_vectors"]);
        assert_eq!(infer_metric(&e), Some("COSINE".to_string()));
        let e = entry_with_views("sift-128-euclidean", &["base_vectors"]);
        assert_eq!(infer_metric(&e), Some("L2".to_string()));
        let e = entry_with_views("lastfm-64-dot", &["base_vectors"]);
        assert_eq!(infer_metric(&e), Some("DOT_PRODUCT".to_string()));
        let e = entry_with_views("ada002-100k", &["base_vectors"]);
        assert_eq!(infer_metric(&e), None);
    }

    #[test]
    fn metric_filter_matches_across_synonyms() {
        // `--with-metric angular` matches a declared `cosine` and a
        // name-inferred `-angular` alike.
        let f = DatasetFilter { metric: Some("angular".to_string()), ..Default::default() };
        assert!(f.matches(&entry_with_attrs("test", "cosine")));
        assert!(f.matches(&entry_with_views("glove-25-angular", &["base_vectors"])));
        assert!(!f.matches(&entry_with_views("sift-128-euclidean", &["base_vectors"])));
    }

    #[test]
    fn parse_active_filters_reads_context_words() {
        let ctx = [
            "--with-metric", "angular",
            "--with-vtype=float32",
            "--with-min-dim", "64",
            "--matching-profile", "def*",
            "--dataset", "glove",
        ];
        let (f, pv) = parse_active_filters(&ctx);
        assert_eq!(f.metric.as_deref(), Some("angular"));
        assert_eq!(f.vtype.as_deref(), Some("float32"));
        assert_eq!(f.min_dim, Some(64));
        assert_eq!(f.name.as_deref(), Some("glove"));
        assert!(pv.is_active());
    }

    #[test]
    fn distinct_value_extractors() {
        let a = entry_with_views("glove-25-angular", &["base_vectors", "query_vectors"]);
        let b = entry_with_views("sift-128-euclidean", &["base_vectors", "neighbor_indices"]);
        let entries: Vec<&CatalogEntry> = vec![&a, &b];
        assert_eq!(distinct_metrics(&entries), vec!["COSINE".to_string(), "L2".to_string()]);
        assert_eq!(distinct_dims(&entries, ProbeMode::Off), vec!["25".to_string(), "128".to_string()]);
        let facets = distinct_facets(&entries);
        assert!(facets.contains(&"base_vectors".to_string()));
        assert!(facets.contains(&"neighbor_indices".to_string()));
    }

    #[test]
    fn facet_filter_accepts_capital_codes() {
        let entry = entry_with_views("ds", &["base_vectors", "neighbor_indices"]);
        for code in ["B", "G"] {
            let f = DatasetFilter { facet: vec![code.to_string()], ..Default::default() };
            assert!(f.matches(&entry), "code {code} should match its facet");
        }
        let f = DatasetFilter { facet: vec!["Q".to_string()], ..Default::default() };
        assert!(!f.matches(&entry), "code Q has no matching facet here");
        // Lowercase single letters are NOT codes — they fall through
        // to name resolution and match nothing standard.
        let f = DatasetFilter { facet: vec!["b".to_string()], ..Default::default() };
        assert!(!f.matches(&entry));
    }

    #[test]
    fn count_suffix_formatting_is_lossless() {
        assert_eq!(format_count_suffix(100_000), "100k");
        assert_eq!(format_count_suffix(1_000_000), "1m");
        assert_eq!(format_count_suffix(2_500_000), "2500k");
        assert_eq!(format_count_suffix(1_000_000_000), "1g");
        assert_eq!(format_count_suffix(123), "123");
        for n in [100_000u64, 2_500_000, 1_000_000_000, 123] {
            assert_eq!(parse_size(&format_count_suffix(n)), Ok(n), "round-trip {n}");
        }
    }

    #[test]
    fn distinct_size_and_data_extractors() {
        let mut a = entry_with_views("glove-25-angular", &["base_vectors"]);
        if let Some(p) = a.layout.profiles.profiles.get_mut("default") {
            p.base_count = Some(100_000);
        }
        let entries: Vec<&CatalogEntry> = vec![&a];
        assert_eq!(distinct_base_counts(&entries, ProbeMode::Off), vec!["100k".to_string()]);
        // Offline (no probe): the base-only estimate is the fallback —
        // 100_000 records × dim 25 × 4 bytes (fvec) = 10_000_000.
        assert_eq!(distinct_data_sizes(&entries, ProbeMode::Off), vec![10_000_000]);
        assert_eq!(format_bytes_approx(10_000_000), "10MB");
    }

    #[test]
    fn data_size_prefers_real_facet_bytes_over_base_only_estimate() {
        // Regression: `--with-data` must report the dataset's true
        // footprint (summed facet bytes), not the base_count × dim ×
        // elem estimate which covers base vectors only and badly
        // under-reports a dataset with large query / GT / metadata
        // facets (e.g. a 43MB estimate for a multi-GB dataset).
        let mut e = entry_with_views("ds", &["base_vectors", "neighbor_indices"]);
        if let Some(p) = e.layout.profiles.profiles.get_mut("default") {
            p.base_count = Some(1_000); // estimate would be tiny
        }
        // Cache real (large) facet sizes keyed by resolved URL.
        let paths = entry_facet_paths(&e);
        let cache: FacetProbeCache = paths
            .iter()
            .map(|p| (p.clone(), FacetProbe { bytes: Some(100_000_000), dim: Some(128) }))
            .collect();

        // Two facets × 100MB = 200MB physical — NOT the 1000×128×4
        // (512KB) base-only estimate.
        assert_eq!(
            distinct_data_sizes(&[&e], ProbeMode::Cached(&cache)),
            vec![200_000_000]
        );

        // And the over-/under-report no longer produces false matches:
        // a 1MB ceiling must NOT match this 200MB dataset.
        let f = DatasetFilter {
            max_data: Some(1_000_000),
            ..Default::default()
        };
        assert!(!f.matches_with(&e, ProbeMode::Cached(&cache)));
    }

    #[test]
    fn bytes_approx_reads_as_bytes_not_counts() {
        // Always a 2-byte (KB/MB/GB/TB) unit, never a bare k/m/g, so
        // the data axis is never mistaken for the count axis.
        assert_eq!(format_bytes_approx(118_400_000), "118MB");
        assert_eq!(format_bytes_approx(12_000_000_000), "12GB");
        assert_eq!(format_bytes_approx(10_000), "10KB");
        // Below 10 of the smallest unit, fall back to raw bytes.
        assert_eq!(format_bytes_approx(9_000), "9000");
    }

    #[test]
    fn exact_data_filter_tolerates_unit_granularity() {
        // A dataset estimated at ~118.4 MB must match the candidate
        // displayed as `118MB` (target 118_000_000, unit 1_000_000:
        // tolerance ±500_000).
        let mut a = entry_with_views("emb_d2960_b10000", &["base_vectors"]);
        if let Some(p) = a.layout.profiles.profiles.get_mut("default") {
            p.base_count = Some(10_000);
        }
        // 10_000 × 2960 × 4 = 118_400_000 bytes.
        assert_eq!(distinct_data_sizes(&[&a], ProbeMode::Off), vec![118_400_000]);
        let displayed = format_bytes_approx(118_400_000);
        assert_eq!(displayed, "118MB");
        let f = DatasetFilter {
            data: Some(parse_bytes_with_unit(&displayed).unwrap()),
            ..Default::default()
        };
        assert!(f.matches(&a), "picked candidate must match its own dataset");
        // A different MB value outside tolerance does not match.
        let f2 = DatasetFilter {
            data: Some(parse_bytes_with_unit("120MB").unwrap()),
            ..Default::default()
        };
        assert!(!f2.matches(&a));
    }

    #[test]
    fn offline_matching_never_probes_unknown_dims() {
        // Unknown dim + dim filter: offline matching must simply
        // fail the entry (no file/network reads to find out).
        let e = entry_with_views("ada002-100k", &["base_vectors"]);
        let f = DatasetFilter { min_dim: Some(1), ..Default::default() };
        assert!(!f.matches_offline(&e));
    }

    #[test]
    fn needs_probe_only_for_count_dim_data_filters() {
        // The gate completion uses to skip warming the probe cache:
        // pure-offline filters never trigger facet I/O.
        assert!(!DatasetFilter::default().needs_probe());
        assert!(!DatasetFilter { metric: Some("L2".into()), ..Default::default() }.needs_probe());
        assert!(!DatasetFilter { vtype: Some("float32".into()), ..Default::default() }.needs_probe());
        assert!(!DatasetFilter { name: Some("glove".into()), ..Default::default() }.needs_probe());
        assert!(!DatasetFilter { facet: vec!["B".into()], ..Default::default() }.needs_probe());
        // Count / dim / data predicates may need a probe → warm.
        assert!(DatasetFilter { count: Some(1), ..Default::default() }.needs_probe());
        assert!(DatasetFilter { min_count: Some(1), ..Default::default() }.needs_probe());
        assert!(DatasetFilter { dim: Some(1), ..Default::default() }.needs_probe());
        assert!(DatasetFilter { max_dim: Some(1), ..Default::default() }.needs_probe());
        assert!(DatasetFilter { data: Some((1, 1)), ..Default::default() }.needs_probe());
        assert!(DatasetFilter { min_data: Some(1), ..Default::default() }.needs_probe());
    }

    #[test]
    fn cached_probe_matching_is_consistent_with_a_live_run() {
        // The reported bug: a count that only resolves via probe
        // (no `base_count` metadata, no dim in the name) was matched
        // by the live run but dropped by completion's offline
        // narrowing — so the next filter completed to nothing.
        //
        // Here `ds` has dim 128, float32 (4B), base file of
        // 83818 × (4 + 128×4) bytes ⇒ exactly 83818 records.
        let e = entry_with_views("ds", &["base_vectors"]);
        // Key the cache by the resolved facet URL the probe path uses.
        let base = base_vectors_path(&e).expect("base path");
        let bytes = 83_818u64 * (4 + 128 * 4);
        let cache: FacetProbeCache = [(
            base,
            FacetProbe { bytes: Some(bytes), dim: Some(128) },
        )]
        .into_iter()
        .collect();

        let f = DatasetFilter { count: Some(83_818), ..Default::default() };

        // Offline narrowing drops it (the bug).
        assert!(!f.matches_offline(&e));
        // Cached narrowing (warm cache) matches — consistent with a run.
        assert!(f.matches_with(&e, ProbeMode::Cached(&cache)));

        // And the count appears as a candidate under the same cache,
        // so the user could have completed `--with-count 83818`.
        assert_eq!(
            distinct_base_counts(&[&e], ProbeMode::Cached(&cache)),
            vec!["83818".to_string()]
        );
        // A cache MISS reads as unknown — never a phantom match.
        let empty = FacetProbeCache::new();
        assert!(!f.matches_with(&e, ProbeMode::Cached(&empty)));
    }

    #[test]
    fn test_filter_dim() {
        let f = DatasetFilter {
            dim: Some(1536),
            ..Default::default()
        };
        let entry = entry_with_views("emb_002_d1536_b10000_q10000_mk100", &["base_vectors"]);
        assert!(f.matches(&entry));

        let entry2 = entry_with_views("emb_002_d768_b10000", &["base_vectors"]);
        assert!(!f.matches(&entry2));
    }

    #[test]
    fn test_filter_vtype() {
        let f = DatasetFilter {
            vtype: Some("float32".to_string()),
            ..Default::default()
        };
        let entry = entry_with_views("test", &["base_vectors"]);
        assert!(f.matches(&entry));
    }

    #[test]
    fn test_profile_view_exact() {
        let pv = ProfileView::new(Some("default".to_string()));
        let entry = entry_with_views("test", &["base_vectors"]);
        assert_eq!(pv.matching_profiles(&entry), vec!["default"]);

        let pv2 = ProfileView::new(Some("10m".to_string()));
        assert!(pv2.matching_profiles(&entry).is_empty());
    }

    #[test]
    fn test_profile_view_regex() {
        let mut profiles = IndexMap::new();
        profiles.insert("default".to_string(), DSProfile {
            maxk: None, base_count: None, partition: false, views: IndexMap::new(),
        });
        profiles.insert("10m".to_string(), DSProfile {
            maxk: None, base_count: Some(10_000_000), partition: false, views: IndexMap::new(),
        });
        profiles.insert("100m".to_string(), DSProfile {
            maxk: None, base_count: Some(100_000_000), partition: false, views: IndexMap::new(),
        });
        let entry = CatalogEntry {
            name: "test".to_string(),
            path: "test/dataset.yaml".to_string(),
            dataset_type: "dataset.yaml".to_string(),
            catalog_file: None,
            catalog_name: None,
            layout: CatalogLayout {
                attributes: None,
                profiles: DSProfileGroup::from_profiles(profiles),
            },
        };

        let pv = ProfileView::new(Some(".*m".to_string()));
        let matched = pv.matching_profiles(&entry);
        assert!(matched.contains(&"10m"));
        assert!(matched.contains(&"100m"));
        assert!(!matched.contains(&"default"));
    }

    #[test]
    fn test_profile_view_none_returns_all() {
        let pv = ProfileView::new(None);
        let entry = entry_with_views("test", &["base_vectors"]);
        assert_eq!(pv.matching_profiles(&entry), vec!["default"]);
    }

    #[test]
    fn test_filter_size() {
        let mut profiles = IndexMap::new();
        profiles.insert(
            "default".to_string(),
            DSProfile {
                maxk: None,
                base_count: Some(1_000_000),
                partition: false,
                views: IndexMap::new(),
            },
        );
        let entry = CatalogEntry {
            name: "test".to_string(),
            path: "test/dataset.yaml".to_string(),
            dataset_type: "dataset.yaml".to_string(),
            catalog_file: None,
            catalog_name: None,
            layout: CatalogLayout {
                attributes: None,
                profiles: DSProfileGroup::from_profiles(profiles),
            },
        };

        let f = DatasetFilter {
            min_count: Some(500_000),
            ..Default::default()
        };
        assert!(f.matches(&entry));

        let f2 = DatasetFilter {
            min_count: Some(2_000_000),
            ..Default::default()
        };
        assert!(!f2.matches(&entry));
    }

    #[test]
    fn test_simple_match_regex() {
        assert!(simple_match("vecs.*", "vecs-128"));
        assert!(simple_match(".*emb.*", "my-emb-128"));
        assert!(!simple_match("^emb.*", "vec-100"));
        assert!(simple_match("128$", "vec-128"));
        assert!(!simple_match("128$", "vec-256"));
        assert!(simple_match(".*set.*", "my_dataset"));
    }

    #[test]
    fn test_simple_match_exact() {
        assert!(simple_match("vecs-128", "vecs-128"));
        assert!(simple_match("VECS-128", "vecs-128"));
        assert!(!simple_match("^vecs-128$", "vecs-256"));
    }
}
