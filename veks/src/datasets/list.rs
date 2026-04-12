// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks datasets list` — list datasets from configured or specified catalogs.
//!
//! Uses the full catalog resolution chain:
//! 1. If `--at` is specified, use those locations directly (overrides everything).
//! 2. Otherwise, load catalogs from `--configdir` (default `~/.config/vectordata/catalogs.yaml`),
//!    then add any extra `--catalog` locations.
//! 3. Fetch `catalog.json` from each location (HTTP or local file).
//! 4. Display the aggregated dataset entries.

use crate::catalog::resolver::Catalog;
use crate::catalog::sources::CatalogSources;
use super::filter::{DatasetFilter, ProfileView};

use vectordata::dataset::CatalogEntry;

/// Summary used for structured (json/yaml) output.
#[derive(Debug, serde::Serialize)]
struct DatasetProfileSummary {
    dataset: String,
    profile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attributes: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    views: Option<serde_json::Value>,
}

pub fn run(
    configdir: &str,
    extra_catalogs: &[String],
    at: &[String],
    format: &str,
    verbose: bool,
    group_by: Option<&str>,
    filter: &DatasetFilter,
    profile_view: &ProfileView,
    select: Option<&str>,
) {
    let sources = build_sources(configdir, extra_catalogs, at);

    if sources.is_empty() {
        eprintln!("No catalog sources configured.");
        eprintln!("Add a catalog with:");
        eprintln!("  veks datasets config add-catalog <URL-or-path>");
        eprintln!();
        eprintln!("Or use --catalog/--at for one-off access.");
        std::process::exit(1);
    }

    let catalog = Catalog::of(&sources);

    if catalog.is_empty() {
        eprintln!("No datasets found in any configured catalog.");
        if select.is_some() {
            std::process::exit(1);
        }
        return;
    }

    // Apply filters
    let filtered: Vec<&CatalogEntry> = catalog
        .datasets()
        .iter()
        .filter(|e| filter.matches(e))
        .collect();

    if filtered.is_empty() {
        if filter.is_empty() {
            eprintln!("No datasets found in any configured catalog.");
        } else {
            eprintln!("No datasets match the specified filters.");
            eprintln!(
                "({} dataset(s) available before filtering)",
                catalog.datasets().len()
            );
        }
        if select.is_some() {
            std::process::exit(1);
        }
        return;
    }

    if select.is_some() {
        output_select(&filtered, profile_view, select.unwrap());
        return;
    }

    match format {
        "json" => output_json(&filtered, verbose, profile_view),
        "yaml" => output_yaml(&filtered, verbose, profile_view),
        "csv" => output_csv(&filtered, verbose, profile_view),
        _ => output_text(&filtered, verbose, group_by, profile_view),
    }
}

/// Handle `--select`: succeed with exactly one dataset:profile, fail if
/// ambiguous or empty.
///
/// The `select_value` is the argument to `--select`. If it matches exactly
/// one dataset:profile pair, that pair is printed. If the value doesn't
/// uniquely identify a pair, the remaining matches are shown.
fn output_select(entries: &[&CatalogEntry], pv: &ProfileView, select_value: &str) {
    // Collect all matching dataset:profile pairs
    let mut matches: Vec<String> = Vec::new();
    for entry in entries {
        let profiles = pv.matching_profiles(entry);
        if profiles.is_empty() && !pv.is_active() {
            matches.push(entry.name.clone());
        } else {
            for p in profiles {
                matches.push(format!("{}:{}", entry.name, p));
            }
        }
    }

    // If a value was provided, filter to exact match
    if !select_value.is_empty() {
        let exact: Vec<&String> = matches.iter()
            .filter(|m| m.eq_ignore_ascii_case(select_value))
            .collect();
        if exact.len() == 1 {
            println!("{}", exact[0]);
            return;
        }
        // Try prefix match if no exact match
        let prefix_matches: Vec<&String> = matches.iter()
            .filter(|m| m.to_lowercase().starts_with(&select_value.to_lowercase()))
            .collect();
        if prefix_matches.len() == 1 {
            println!("{}", prefix_matches[0]);
            return;
        }
        if prefix_matches.is_empty() {
            eprintln!("--select '{}' does not match any dataset.", select_value);
            if !matches.is_empty() {
                eprintln!("Available matches:");
                for m in &matches {
                    eprintln!("  {}", m);
                }
            }
            std::process::exit(1);
        }
        // Multiple prefix matches — ambiguous
        eprintln!("--select '{}' is ambiguous; {} matches remain:", select_value, prefix_matches.len());
        for m in &prefix_matches {
            eprintln!("  {}", m);
        }
        std::process::exit(1);
    }

    // No value given — check if filters narrowed to exactly one
    match matches.len() {
        0 => {
            eprintln!("No datasets match the specified filters.");
            std::process::exit(1);
        }
        1 => {
            println!("{}", matches[0]);
        }
        _ => {
            eprintln!("--select is ambiguous; {} matches remain:", matches.len());
            for m in &matches {
                eprintln!("  {}", m);
            }
            std::process::exit(1);
        }
    }
}

/// Build `CatalogSources` from CLI arguments, mirroring the Java
/// `CMD_datasets_list.call()` logic.
fn build_sources(configdir: &str, extra_catalogs: &[String], at: &[String]) -> CatalogSources {
    let mut sources = CatalogSources::new();

    if !at.is_empty() {
        // --at overrides everything: use those locations directly
        sources = sources.add_catalogs(at);
    } else {
        // Load from configdir (catalogs.yaml), then add extras
        sources = sources.configure(configdir);
        if !extra_catalogs.is_empty() {
            sources = sources.add_catalogs(extra_catalogs);
        }
    }

    sources
}

/// Detect terminal column width, defaulting to 80 if not on a terminal.
fn terminal_width() -> usize {
    crossterm::terminal::size()
        .map(|(w, _)| w as usize)
        .unwrap_or(80)
}

/// Text output: grouped by dataset name, profiles shown inline.
///
/// When connected to a terminal, uses the full terminal width to display
/// datasets. Each dataset is shown once, with its profiles listed
/// compactly on the same line:
///
/// ```text
/// sift-128             [default, 1m, 2m, 4m, 10m, 20m, 50m, 100m]  L2
/// glove-100            [default]
/// laion400m-img-search [default, 10m, 50m, 100m, 200m, 400m]       Cosine
/// ```
fn output_text(entries: &[&CatalogEntry], verbose: bool, group_by: Option<&str>, pv: &ProfileView) {
    if let Some(key) = group_by {
        output_text_grouped(entries, verbose, key, pv);
        return;
    }

    let width = terminal_width();

    // Find the longest dataset name for column alignment
    let max_name_len = entries.iter()
        .map(|e| e.name.len())
        .max()
        .unwrap_or(0)
        .min(width / 3); // don't let names take more than 1/3 of the line

    let name_col = max_name_len + 2; // padding

    // Header
    println!("{:<width$}  {:<width2$}  {}",
        "DATASET", "PROFILES", "METRIC",
        width = name_col.saturating_sub(2),
        width2 = 30);

    for entry in entries {
        let profile_names = pv.matching_profiles(entry);
        if profile_names.is_empty() {
            if !pv.is_active() {
                println!("{}", entry.name);
            }
            continue;
        }

        // Format profile list compactly
        let profiles_str = if profile_names.len() == 1 && profile_names[0] == "default" {
            "[default]".to_string()
        } else {
            format!("[{}]", profile_names.join(", "))
        };

        // Attribute summary (distance function, dimension if available)
        let attr_summary = entry.layout.attributes.as_ref()
            .and_then(|a| a.distance_function.as_deref())
            .unwrap_or("");

        // Truncate profiles if they'd overflow the line
        let name_part = &entry.name;
        let available = width.saturating_sub(name_col + attr_summary.len() + 2);
        let profiles_display = if profiles_str.len() > available && available > 10 {
            // Truncate with count
            format!("[{} profiles]", profile_names.len())
        } else {
            profiles_str
        };

        // Build the line, ensuring it fits within terminal width.
        // Reserve 1 column to prevent terminal wrapping on the last column.
        let usable = width.saturating_sub(1);
        let name_padded = format!("{:<width$}", name_part, width = name_col);

        if attr_summary.is_empty() {
            let line = format!("{}{}", name_padded, profiles_display);
            println!("{}", &line[..line.len().min(usable)]);
        } else {
            let fixed_len = name_padded.len() + profiles_display.len() + attr_summary.len();
            let gap = usable.saturating_sub(fixed_len);
            let spacer = " ".repeat(gap.max(2));
            let line = format!("{}{}{}{}", name_padded, profiles_display, spacer, attr_summary);
            println!("{}", &line[..line.len().min(usable)]);
        }

        if verbose {
            // Indented detail lines
            println!("  url: {}", entry.path);
            if let Some(ref attrs) = entry.layout.attributes {
                if let Some(ref model) = attrs.model {
                    println!("  model: {}", model);
                }
                if let Some(ref vendor) = attrs.vendor {
                    println!("  vendor: {}", vendor);
                }
                if let Some(ref license) = attrs.license {
                    println!("  license: {}", license);
                }
                for (k, v) in &attrs.tags {
                    println!("  tag.{}: {}", k, v);
                }
            }
            // Show views for default profile
            if let Some(profile) = entry.layout.profiles.profiles.get("default") {
                let views: Vec<&str> = profile.view_names();
                if !views.is_empty() {
                    println!("  views: {}", views.join(", "));
                }
            }
        }
    }
}

/// Text output grouped by a key: `source`, `profile`, or `metric`.
fn output_text_grouped(entries: &[&CatalogEntry], verbose: bool, key: &str, pv: &ProfileView) {
    use std::collections::BTreeMap;

    let width = terminal_width();

    // Build (group_key → Vec<(dataset_name, profiles)>) mapping
    let mut groups: BTreeMap<String, Vec<(&str, Vec<&str>)>> = BTreeMap::new();

    for entry in entries {
        let profile_names = pv.matching_profiles(entry);
        if profile_names.is_empty() && pv.is_active() {
            continue;
        }

        let group_key = match key {
            "source" => entry.path.clone(),
            "metric" => entry.layout.attributes.as_ref()
                .and_then(|a| a.distance_function.clone())
                .unwrap_or_else(|| "(no distance_function in attributes)".into()),
            "profile" => {
                // Group by profile: each profile becomes a group key
                for pname in &profile_names {
                    groups.entry(pname.to_string())
                        .or_default()
                        .push((&entry.name, vec![*pname]));
                }
                continue;
            }
            other => {
                eprintln!("Error: unknown --group-by key '{}'. Use: source, profile, metric", other);
                std::process::exit(1);
            }
        };

        groups.entry(group_key)
            .or_default()
            .push((&entry.name, profile_names));
    }

    // Find longest dataset name across all groups for alignment
    let max_name_len = groups.values()
        .flat_map(|v| v.iter().map(|(name, _)| name.len()))
        .max()
        .unwrap_or(0)
        .min(width / 3);
    let name_col = max_name_len + 2;

    let usable = width.saturating_sub(1);

    for (group_key, datasets) in &groups {
        println!("{}:", group_key);
        for (name, profiles) in datasets {
            let profiles_str = if profiles.len() == 1 && profiles[0] == "default" {
                "[default]".to_string()
            } else {
                format!("[{}]", profiles.join(", "))
            };

            let name_padded = format!("  {:<width$}", name, width = name_col);
            let line = format!("{}{}", name_padded, profiles_str);
            println!("{}", &line[..line.len().min(usable)]);

            if verbose {
                // Find the entry to show details
                if let Some(entry) = entries.iter().find(|e| e.name == **name) {
                    println!("    url: {}", entry.path);
                    if let Some(ref attrs) = entry.layout.attributes {
                        if let Some(ref df) = attrs.distance_function {
                            println!("    metric: {}", df);
                        }
                    }
                }
            }
        }
        println!();
    }
}

/// CSV output.
fn output_csv(entries: &[&CatalogEntry], verbose: bool, pv: &ProfileView) {
    if verbose {
        println!("dataset,profile,distance_function,model,vendor,license,views");
    } else {
        println!("dataset,profile");
    }

    for entry in entries {
        for profile_name in pv.matching_profiles(entry) {
            if verbose {
                let attrs = entry.layout.attributes.as_ref();
                let df = attrs.and_then(|a| a.distance_function.as_deref()).unwrap_or("");
                let model = attrs.and_then(|a| a.model.as_deref()).unwrap_or("");
                let vendor = attrs.and_then(|a| a.vendor.as_deref()).unwrap_or("");
                let license = attrs.and_then(|a| a.license.as_deref()).unwrap_or("");

                let views = entry.layout.profiles.profiles.get(profile_name)
                    .map(|p| p.view_names().join(";"))
                    .unwrap_or_default();

                println!(
                    "{},{},{},{},{},{},{}",
                    escape_csv(&entry.name),
                    escape_csv(profile_name),
                    escape_csv(df),
                    escape_csv(model),
                    escape_csv(vendor),
                    escape_csv(license),
                    escape_csv(&views),
                );
            } else {
                println!("{},{}", escape_csv(&entry.name), escape_csv(profile_name));
            }
        }
    }
}

/// JSON output.
fn output_json(entries: &[&CatalogEntry], verbose: bool, pv: &ProfileView) {
    let summaries = build_structured(entries, verbose, pv);
    let json = serde_json::to_string_pretty(&summaries).unwrap_or_default();
    println!("{}", json);
}

/// YAML output.
fn output_yaml(entries: &[&CatalogEntry], verbose: bool, pv: &ProfileView) {
    let summaries = build_structured(entries, verbose, pv);
    let yaml = serde_yaml::to_string(&summaries).unwrap_or_default();
    print!("{}", yaml);
}

/// Build structured output for JSON/YAML.
fn build_structured(entries: &[&CatalogEntry], verbose: bool, pv: &ProfileView) -> Vec<DatasetProfileSummary> {
    let mut out = Vec::new();

    for entry in entries {
        for profile_name in pv.matching_profiles(entry) {
            let mut summary = DatasetProfileSummary {
                dataset: entry.name.clone(),
                profile: profile_name.to_string(),
                url: None,
                attributes: None,
                views: None,
            };

            if verbose {
                summary.url = Some(entry.path.clone());
                if let Some(ref attrs) = entry.layout.attributes {
                    summary.attributes = serde_json::to_value(attrs).ok();
                }
                if let Some(profile) = entry.layout.profiles.profiles.get(profile_name) {
                    let view_names = profile.view_names();
                    if !view_names.is_empty() {
                        summary.views = Some(serde_json::json!(view_names));
                    }
                }
            }

            out.push(summary);
        }
    }

    out
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sources_from_dir(dir: &std::path::Path) -> CatalogSources {
        CatalogSources::new().add_catalogs(&[dir.to_string_lossy().to_string()])
    }

    #[test]
    fn test_build_sources_at_overrides() {
        let sources = build_sources(
            "~/.config/vectordata",
            &["extra".to_string()],
            &["https://example.com/catalog".to_string()],
        );
        // --at should take precedence, required should contain the at URL
        assert!(sources.required().iter().any(|s| s.contains("example.com")));
    }

    #[test]
    fn test_build_sources_configdir_plus_catalog() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = "- https://example.com/data\n";
        std::fs::write(tmp.path().join("catalogs.yaml"), yaml).unwrap();

        let sources = build_sources(
            tmp.path().to_str().unwrap(),
            &["/some/extra/path".to_string()],
            &[],
        );
        // Should have both the configured and extra catalog
        assert!(!sources.is_empty());
    }

    #[test]
    fn test_local_catalog_text_output() {
        let tmp = tempfile::tempdir().unwrap();
        let json = serde_json::json!([
            {
                "name": "sift-128",
                "path": "sift-128/dataset.yaml",
                "dataset_type": "dataset.yaml",
                "layout": {
                    "attributes": {"distance_function": "L2"},
                    "profiles": {
                        "default": {"base_vectors": "base.fvec"},
                        "10m": {"base_vectors": "base.fvec", "base_count": 10000000}
                    }
                }
            },
            {
                "name": "glove-100",
                "path": "glove-100/dataset.yaml",
                "dataset_type": "dataset.yaml",
                "layout": {
                    "profiles": {
                        "default": {"base_vectors": "base.fvec"}
                    }
                }
            }
        ]);
        std::fs::write(
            tmp.path().join("catalog.json"),
            serde_json::to_string_pretty(&json).unwrap(),
        )
        .unwrap();

        let sources = make_sources_from_dir(tmp.path());
        let catalog = Catalog::of(&sources);
        assert_eq!(catalog.datasets().len(), 2);

        // Verify profile_names works
        let sift = catalog.find_exact("sift-128").unwrap();
        let profiles = sift.profile_names();
        assert!(profiles.contains(&"default"));
    }
}
