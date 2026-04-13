// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Dataset filtering predicates for `veks datasets list`.
//!
//! Each filter option maps to a predicate applied against [`CatalogEntry`]
//! fields. Filters compose conjunctively — all specified filters must match.
//!
//! Dynamic autocompletion is powered by [`catalog_completions`], which loads
//! the real catalog from the user's default config and extracts the actual
//! values present across all visible entries.

use std::sync::OnceLock;

use regex::RegexBuilder;

use vectordata::dataset::{CatalogEntry};
use vectordata::dataset::source::parse_number_with_suffix;

use crate::catalog::resolver::Catalog;
use crate::catalog::sources::CatalogSources;

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
    pub min_size: Option<u64>,
    pub max_size: Option<u64>,
    pub size: Option<u64>,
    pub min_dim: Option<u32>,
    pub max_dim: Option<u32>,
    pub dim: Option<u32>,
    pub vtype: Option<String>,
    pub data_min: Option<u64>,
    pub data_max: Option<u64>,
}

impl DatasetFilter {
    /// Returns `true` if no filters are set.
    pub fn is_empty(&self) -> bool {
        self.name.is_none()
            && self.facet.is_empty()
            && self.metric.is_none()
            && self.desc.is_none()
            && self.min_size.is_none()
            && self.max_size.is_none()
            && self.size.is_none()
            && self.min_dim.is_none()
            && self.max_dim.is_none()
            && self.dim.is_none()
            && self.vtype.is_none()
            && self.data_min.is_none()
            && self.data_max.is_none()
    }

    /// Test whether a catalog entry passes all filter predicates.
    pub fn matches(&self, entry: &CatalogEntry) -> bool {
        if let Some(ref name) = self.name {
            if !smart_match(name, &entry.name) {
                return false;
            }
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
            let df = entry
                .layout
                .attributes
                .as_ref()
                .and_then(|a| a.distance_function.as_deref())
                .unwrap_or("");
            if !df.eq_ignore_ascii_case(metric) {
                return false;
            }
        }

        if let Some(ref desc) = self.desc {
            if !matches_description_smart(entry, desc) {
                return false;
            }
        }

        // Size filters: check base_count across all profiles
        if self.min_size.is_some() || self.max_size.is_some() || self.size.is_some() {
            let max_base_count = max_base_count(entry);
            if let Some(min) = self.min_size {
                match max_base_count {
                    Some(c) if c >= min => {}
                    _ => return false,
                }
            }
            if let Some(max) = self.max_size {
                match max_base_count {
                    Some(c) if c <= max => {}
                    _ => return false,
                }
            }
            if let Some(exact) = self.size {
                match max_base_count {
                    Some(c) if c == exact => {}
                    _ => return false,
                }
            }
        }

        // Dimension filters: infer from name pattern _dNNN_ or attributes/tags
        if self.min_dim.is_some() || self.max_dim.is_some() || self.dim.is_some() {
            let dim = infer_dimension(entry);
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
        if self.data_min.is_some() || self.data_max.is_some() {
            let est = estimate_data_bytes(entry);
            if let Some(min) = self.data_min {
                match est {
                    Some(b) if b >= min => {}
                    _ => return false,
                }
            }
            if let Some(max) = self.data_max {
                match est {
                    Some(b) if b <= max => {}
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

/// Resolve a user-provided facet name to its canonical form.
fn resolve_facet_name(name: &str) -> String {
    vectordata::dataset::facet::resolve_standard_key(name)
        .unwrap_or_else(|| name.to_string())
}

/// Check if a description-like field contains the search term.
///
/// Searches in: attributes.notes, attributes.model, name, tags values.
fn matches_description(entry: &CatalogEntry, search: &str) -> bool {
    let lower = search.to_lowercase();
    if let Some(ref attrs) = entry.layout.attributes {
        if let Some(ref notes) = attrs.notes {
            if notes.to_lowercase().contains(&lower) {
                return true;
            }
        }
        if let Some(ref model) = attrs.model {
            if model.to_lowercase().contains(&lower) {
                return true;
            }
        }
        for v in attrs.tags.values() {
            if v.to_lowercase().contains(&lower) {
                return true;
            }
        }
    }
    entry.name.to_lowercase().contains(&lower)
}

/// Regex match against description-like fields.
fn matches_description_regex(entry: &CatalogEntry, pattern: &str) -> bool {
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
    texts.iter().any(|t| simple_match(pattern, t))
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

/// Infer dimensionality from the dataset name or attributes/tags.
pub fn infer_dimension(entry: &CatalogEntry) -> Option<u32> {
    if let Some(ref attrs) = entry.layout.attributes {
        for (k, v) in &attrs.tags {
            if k.eq_ignore_ascii_case("dimension")
                || k.eq_ignore_ascii_case("dim")
                || k.eq_ignore_ascii_case("dimensions")
            {
                if let Ok(d) = v.parse::<u32>() {
                    return Some(d);
                }
            }
        }
    }
    extract_dim_from_name(&entry.name)
}

/// Extract dimension from a dataset name using the `_dNNN` convention.
fn extract_dim_from_name(name: &str) -> Option<u32> {
    for part in name.split('_') {
        if let Some(num_str) = part.strip_prefix('d') {
            if !num_str.is_empty() && num_str.chars().all(|c| c.is_ascii_digit()) {
                if let Ok(d) = num_str.parse::<u32>() {
                    if d > 0 && d < 100_000 {
                        return Some(d);
                    }
                }
            }
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

/// Map a file extension to a vector type name.
fn vtype_from_extension(path: &str) -> Option<String> {
    use veks_core::formats::VecFormat;
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())?;
    let fmt = VecFormat::from_extension(ext)?;
    let name = match fmt {
        VecFormat::Fvec => "float32",
        VecFormat::Mvec => "float16",
        VecFormat::Bvec => "uint8",
        VecFormat::I8vec => "int8",
        VecFormat::Ivec => "int32",
        VecFormat::Dvec => "float64",
        VecFormat::Svec => "int16",
        VecFormat::U16vec => "uint16",
        VecFormat::U32vec => "uint32",
        VecFormat::I64vec => "int64",
        VecFormat::U64vec => "uint64",
        VecFormat::Npy => "numpy",
        VecFormat::Hdf5 => "hdf5",
        VecFormat::Parquet => "parquet",
        VecFormat::Slab => return None,
        // Scalar formats have no vector type name
        _ => return None,
    };
    Some(name.to_string())
}

/// Rough estimate of total data bytes from base_count × dimension × element_size.
fn estimate_data_bytes(entry: &CatalogEntry) -> Option<u64> {
    let dim = infer_dimension(entry).unwrap_or(0) as u64;
    if dim == 0 {
        return None;
    }
    let elem_bytes: u64 = match infer_vtype(entry).as_deref() {
        Some("float32") | Some("int32") => 4,
        Some("float16") => 2,
        Some("uint8") => 1,
        _ => 4,
    };
    let base_count = max_base_count(entry)?;
    Some(base_count * dim * elem_bytes)
}

/// Strip surrounding single or double quotes from a string.
///
/// During shell completion, `COMP_WORDS` preserves the raw quotes typed by
/// the user (e.g. `'.*ibm.*'`). We need to strip them so that
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

// ---------------------------------------------------------------------------
// Dynamic autocompletion — loads the real catalog
// ---------------------------------------------------------------------------

/// Lazily cached catalog entries for autocompletion.
///
/// Loads from the user's default `~/.config/vectordata/catalogs.yaml` once,
/// then reuses the result for all completer calls within the same completion
/// invocation.
static COMPLETION_CATALOG: OnceLock<Vec<CatalogEntry>> = OnceLock::new();

/// Load catalog entries for completion, caching on first call.
pub(crate) fn completion_entries() -> &'static [CatalogEntry] {
    COMPLETION_CATALOG.get_or_init(|| {
        let sources = CatalogSources::new().configure_default();
        if sources.is_empty() {
            return Vec::new();
        }
        let catalog = Catalog::of(&sources);
        catalog.datasets().to_vec()
    })
}

/// Returns catalog entries filtered by any `--with-*`, `--profile`, and
/// `--profile-regex` args already present on the command line, so each
/// completer only suggests values consistent with filters specified so far.
fn filtered_completion_entries() -> Vec<&'static CatalogEntry> {
    let all = completion_entries();
    let (filter, pv) = parse_active_filters();
    all.iter()
        .filter(|e| filter.matches(e))
        .filter(|e| !pv.is_active() || !pv.matching_profiles(e).is_empty())
        .collect()
}

/// Parse already-specified `--with-*`, `--profile`, and `--profile-regex`
/// args from the current process args. During shell completion, the binary
/// is re-invoked with the partial command line, so `std::env::args()`
/// contains the tokens typed so far.
fn parse_active_filters() -> (DatasetFilter, ProfileView) {
    let args: Vec<String> = std::env::args().collect();
    let mut filter = DatasetFilter::default();
    let mut profile: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        // Handle --key=value syntax
        let (key, inline_val) = if let Some(eq_pos) = args[i].find('=') {
            (&args[i][..eq_pos], Some(args[i][eq_pos + 1..].to_string()))
        } else {
            (args[i].as_str(), None)
        };

        let next_val = |i: usize, inline: &Option<String>, args: &[String]| -> Option<String> {
            inline.clone().or_else(|| args.get(i + 1).cloned())
                .filter(|v| !v.is_empty())
                .map(|v| strip_shell_quotes(&v))
                .filter(|v| !v.is_empty())
        };

        let advance = if inline_val.is_some() { 0 } else { 1 };

        match key {
            "--matching-name" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.name = Some(v);
                    i += advance;
                }
            }
            "--with-facet" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.facet.push(v);
                    i += advance;
                }
            }
            "--with-metric" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.metric = Some(v);
                    i += advance;
                }
            }
            "--matching-desc" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.desc = Some(v);
                    i += advance;
                }
            }
            "--with-min-size" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.min_size = parse_size(&v).ok();
                    i += advance;
                }
            }
            "--with-max-size" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.max_size = parse_size(&v).ok();
                    i += advance;
                }
            }
            "--with-size" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.size = parse_size(&v).ok();
                    i += advance;
                }
            }
            "--with-min-dim" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.min_dim = v.parse().ok();
                    i += advance;
                }
            }
            "--with-max-dim" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.max_dim = v.parse().ok();
                    i += advance;
                }
            }
            "--with-dim" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.dim = v.parse().ok();
                    i += advance;
                }
            }
            "--with-vtype" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.vtype = Some(v);
                    i += advance;
                }
            }
            "--with-data-min" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.data_min = parse_bytes(&v).ok();
                    i += advance;
                }
            }
            "--with-data-max" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    filter.data_max = parse_bytes(&v).ok();
                    i += advance;
                }
            }
            "--matching-profile" => {
                if let Some(v) = next_val(i, &inline_val, &args) {
                    profile = Some(v);
                    i += advance;
                }
            }
            _ => {}
        }
        i += 1;
    }

    (filter, ProfileView::new(profile))
}

/// Returns `true` if `--select` is already present on the command line,
/// meaning no further `--with-*` narrowing should be offered.
fn select_already_specified() -> bool {
    std::env::args().any(|a| a == "--select" || a.starts_with("--select="))
}

/// Returns `true` if `--select` has been given a non-empty value.
/// When the user is currently completing `--select`'s value (i.e. `--select <TAB>`),
/// the value is empty or missing, so this returns `false`.
fn select_has_value() -> bool {
    let args: Vec<String> = std::env::args().collect();
    for (i, arg) in args.iter().enumerate() {
        if arg.starts_with("--select=") {
            let val = &arg["--select=".len()..];
            return !val.is_empty();
        }
        if arg == "--select" {
            if let Some(next) = args.get(i + 1) {
                return !next.is_empty() && !next.starts_with("--");
            }
            return false;
        }
    }
    false
}

/// Returns the number of dataset:profile matches for the current filters
/// and profile view. Used to decide whether to suppress all further
/// completions (exactly 1 match with `--select`).
fn filtered_match_count() -> usize {
    let entries = filtered_completion_entries();
    let (_, pv) = parse_active_filters();
    let mut count = 0;
    for e in &entries {
        let profiles = pv.matching_profiles(e);
        if profiles.is_empty() && !pv.is_active() {
            count += 1;
        } else {
            count += profiles.len();
        }
    }
    count
}

// Per-argument value completers were removed — completion is now handled
// by the veks-completion crate via the dyncomp command tree.

// Retained below: hidden_list_args() for clap arg visibility control.
//
pub fn hidden_list_args() -> Vec<&'static str> {
    let (filter, pv) = parse_active_filters();
    let selected = select_already_specified();
    let mut hidden = Vec::new();

    // Hide already-specified single-value filter args
    if filter.name.is_some() { hidden.push("matching_name"); }
    if filter.metric.is_some() { hidden.push("metric"); }
    if filter.desc.is_some() { hidden.push("matching_desc"); }
    if filter.min_size.is_some() { hidden.push("min_size"); }
    if filter.max_size.is_some() { hidden.push("max_size"); }
    if filter.size.is_some() { hidden.push("size"); }
    if filter.min_dim.is_some() { hidden.push("min_dim"); }
    if filter.max_dim.is_some() { hidden.push("max_dim"); }
    if filter.dim.is_some() { hidden.push("dim"); }
    if filter.vtype.is_some() { hidden.push("vtype"); }
    if filter.data_min.is_some() { hidden.push("data_min"); }
    if filter.data_max.is_some() { hidden.push("data_max"); }

    // Hide --matching-profile when already specified or --select is active
    if pv.is_active() || selected {
        hidden.push("matching_profile");
    }

    // When --select is active, hide all remaining filter args
    if selected {
        for id in &["matching_name", "metric", "facet", "matching_desc",
                     "dim", "min_dim", "max_dim", "vtype", "size", "min_size",
                     "max_size", "data_min", "data_max"] {
            if !hidden.contains(id) {
                hidden.push(id);
            }
        }
    }

    // Hide --select only when it already has a non-empty value
    let select_valued = select_has_value();
    if select_valued {
        hidden.push("select");
    }

    // When --select has a value, hide output formatting options (they don't apply)
    if select_valued {
        hidden.push("output_format");
        hidden.push("verbose");
    }

    // When --select has a value and resolves to a single match, suppress everything —
    // the command is complete, no further options make sense.
    if select_valued && filtered_match_count() <= 1 {
        // Signal to hide --help via the special sentinel
        hidden.push("__disable_help");
    }

    // Once any --with-* filter is in use, the catalog source is committed —
    // hide --configdir, --catalog, --at to avoid confusing reloads.
    if !filter.is_empty() || pv.is_active() {
        hidden.push("configdir");
        hidden.push("catalog");
        hidden.push("at");
    }

    hidden
}

#[cfg(test)]
mod tests {
    use super::*;
    use vectordata::dataset::{CatalogLayout, DSProfile, DSProfileGroup};
    use indexmap::IndexMap;

    fn entry_with_views(name: &str, views: &[&str]) -> CatalogEntry {
        let mut view_map = IndexMap::new();
        for v in views {
            view_map.insert(
                v.to_string(),
                vectordata::dataset::DSView {
                    source: vectordata::dataset::DSSource {
                        path: format!("{}.fvec", v),
                        namespace: None,
                        window: vectordata::dataset::source::DSWindow::default(),
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
            layout: CatalogLayout {
                attributes: None,
                profiles: DSProfileGroup::from_profiles(profiles),
            },
        }
    }

    fn entry_with_attrs(name: &str, metric: &str) -> CatalogEntry {
        let mut e = entry_with_views(name, &["base_vectors", "query_vectors"]);
        e.layout.attributes = Some(vectordata::dataset::DatasetAttributes {
            distance_function: Some(metric.to_string()),
            ..Default::default()
        });
        e
    }

    #[test]
    fn test_filter_name_substring() {
        let f = DatasetFilter {
            name: Some("sift".to_string()),
            ..Default::default()
        };
        let entry = entry_with_views("sift-128", &["base_vectors"]);
        assert!(f.matches(&entry));

        let entry2 = entry_with_views("glove-100", &["base_vectors"]);
        assert!(!f.matches(&entry2));
    }

    #[test]
    fn test_filter_name_regex() {
        let f = DatasetFilter {
            name: Some("sift.*".to_string()),
            ..Default::default()
        };
        let entry = entry_with_views("sift-128", &["base_vectors"]);
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
        assert_eq!(extract_dim_from_name("ada_002_d1536_b10000_q10000_mk100"), Some(1536));
        assert_eq!(extract_dim_from_name("sift-128"), None);
        assert_eq!(extract_dim_from_name("E5-base-v2_d768_b10000"), Some(768));
        assert_eq!(extract_dim_from_name("no-dimension-here"), None);
    }

    #[test]
    fn test_filter_dim() {
        let f = DatasetFilter {
            dim: Some(1536),
            ..Default::default()
        };
        let entry = entry_with_views("ada_002_d1536_b10000_q10000_mk100", &["base_vectors"]);
        assert!(f.matches(&entry));

        let entry2 = entry_with_views("ada_002_d768_b10000", &["base_vectors"]);
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
            layout: CatalogLayout {
                attributes: None,
                profiles: DSProfileGroup::from_profiles(profiles),
            },
        };

        let f = DatasetFilter {
            min_size: Some(500_000),
            ..Default::default()
        };
        assert!(f.matches(&entry));

        let f2 = DatasetFilter {
            min_size: Some(2_000_000),
            ..Default::default()
        };
        assert!(!f2.matches(&entry));
    }

    #[test]
    fn test_simple_match_regex() {
        assert!(simple_match("sift.*", "sift-128"));
        assert!(simple_match(".*sift.*", "my-sift-128"));
        assert!(!simple_match("^sift.*", "glove-100"));
        assert!(simple_match("128$", "sift-128"));
        assert!(!simple_match("128$", "sift-256"));
        assert!(simple_match(".*lai.*", "cohere_laion"));
    }

    #[test]
    fn test_simple_match_exact() {
        assert!(simple_match("sift-128", "sift-128"));
        assert!(simple_match("SIFT-128", "sift-128"));
        assert!(!simple_match("^sift-128$", "sift-256"));
    }
}
