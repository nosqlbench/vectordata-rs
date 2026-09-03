// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `veks prepare cleanup-profiles`: remove sized profiles nothing references.
//!
//! A stratum (`strata:` in `dataset.yaml`) generates sized profiles; the
//! runner materialises them as ordinary entries under `profiles:` with
//! their own directories of answer keys. Removing a stratum removes the
//! generator, not what it generated: the entries and directories stay,
//! and the pipeline keeps planning steps for them. This command finds
//! two kinds of leftovers and, unless `--dry-run`, removes them:
//!
//! - **unreferenced entries** — sized profiles under `profiles:` (a
//!   `base_count`, not `default`, not a partition) whose name appears in
//!   no stratum's `series`;
//! - **unreferenced directories** — `profiles/<name>/` directories for
//!   which `profiles:` has no entry at all (`base` and `default` aside).
//!
//! The cache is never touched: predicate-key segments, provenance
//! sidecars and the progress log stay. Records in the progress log for
//! the removed profiles' steps are harmless — the runner consults
//! records only for steps the definition has. The definition is backed
//! up before it is rewritten, and the progress log's mtime is refreshed
//! the way `stratify` does, so completed steps are not thought stale
//! for the definition being newer.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use vectordata::dataset::DatasetConfig;

/// One sized profile with nothing referencing it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Orphan {
    pub name: String,
    /// `profiles/<name>`, when the directory exists.
    pub dir: Option<PathBuf>,
    /// Bytes under the directory.
    pub bytes: u64,
}

/// What the command would remove.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Plan {
    /// Entries under `profiles:` no stratum names.
    pub entries: Vec<Orphan>,
    /// Directories under `profiles/` no entry names.
    pub dirs: Vec<Orphan>,
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.dirs.is_empty()
    }

    pub fn bytes(&self) -> u64 {
        self.entries.iter().chain(self.dirs.iter()).map(|o| o.bytes).sum()
    }
}

/// The names every stratum still generates.
fn referenced_names(config: &DatasetConfig) -> HashSet<String> {
    config
        .strata
        .iter()
        .flat_map(|(_, stratum)| stratum.series.iter().cloned())
        .collect()
}

/// Plan the removal for `config` as it stands, against the directories
/// under `dataset_dir/profiles`.
pub fn plan(config: &DatasetConfig, dataset_dir: &Path) -> Plan {
    let referenced = referenced_names(config);
    let profiles_dir = dataset_dir.join("profiles");
    let mut plan = Plan::default();
    for (name, profile) in &config.profiles.profiles {
        if name == "default" || profile.partition || profile.base_count.is_none() {
            continue;
        }
        if referenced.contains(name) {
            continue;
        }
        let dir = profiles_dir.join(name);
        let (dir, bytes) = if dir.is_dir() {
            let b = dir_size(&dir);
            (Some(dir), b)
        } else {
            (None, 0)
        };
        plan.entries.push(Orphan {
            name: name.clone(),
            dir,
            bytes,
        });
    }
    if let Ok(entries) = std::fs::read_dir(&profiles_dir) {
        let mut names: Vec<String> = entries
            .flatten()
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n != "base" && n != "default" && !config.profiles.profiles.contains_key(n))
            .collect();
        names.sort();
        for name in names {
            let dir = profiles_dir.join(&name);
            let bytes = dir_size(&dir);
            plan.dirs.push(Orphan {
                name,
                dir: Some(dir),
                bytes,
            });
        }
    }
    plan
}

/// Apply `plan`: drop the entries from `config`, write it back with a
/// backup, refresh the progress log's mtime, and remove the
/// directories. Returns the bytes removed.
pub fn apply(config: &mut DatasetConfig, dataset_path: &Path, plan: &Plan) -> Result<u64, String> {
    let dataset_dir = dataset_path.parent().unwrap_or(Path::new("."));
    if !plan.entries.is_empty() {
        for o in &plan.entries {
            config.profiles.profiles.swap_remove(&o.name);
        }
        let backup = crate::check::fix::create_backup(dataset_path)?;
        println!(
            "  Backed up {} → {}",
            crate::check::rel_display(dataset_path),
            crate::check::rel_display(&backup)
        );
        let yaml = serde_yaml::to_string(config).map_err(|e| format!("failed to serialize config: {}", e))?;
        let tmp = dataset_path.with_extension("yaml.tmp");
        std::fs::write(&tmp, &yaml).map_err(|e| format!("failed to write {}: {}", tmp.display(), e))?;
        std::fs::rename(&tmp, dataset_path).map_err(|e| format!("failed to rename {}: {}", tmp.display(), e))?;
        // The definition is newer now, but no remaining step changed.
        let progress = dataset_dir.join(".cache/.upstream.progress.yaml");
        if progress.exists() {
            let _ = filetime::set_file_mtime(&progress, filetime::FileTime::now());
        }
    }
    let mut removed = 0u64;
    for o in plan.entries.iter().chain(plan.dirs.iter()) {
        let Some(dir) = &o.dir else { continue };
        debug_assert!(dir.starts_with(dataset_dir.join("profiles")));
        match std::fs::remove_dir_all(dir) {
            Ok(()) => removed += o.bytes,
            Err(e) => eprintln!("  warning: failed to remove {}: {}", dir.display(), e),
        }
    }
    Ok(removed)
}

pub fn run(path: &Path, dry_run: bool) {
    let dataset_path = if path.join("dataset.yaml").exists() {
        path.join("dataset.yaml")
    } else if path.file_name().map(|n| n == "dataset.yaml").unwrap_or(false) {
        path.to_path_buf()
    } else {
        eprintln!("Error: no dataset.yaml found at {}", path.display());
        std::process::exit(1);
    };
    let dataset_dir = dataset_path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut config = DatasetConfig::load(&dataset_path).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    let plan = plan(&config, &dataset_dir);
    let strata: Vec<&str> = config.strata.iter().map(|(n, _)| n).collect();
    if plan.is_empty() {
        println!(
            "Every sized profile is named by a stratum ({}) and every profile directory has an entry — nothing to clean.",
            if strata.is_empty() { "none declared".to_string() } else { strata.join(", ") }
        );
        return;
    }
    if !plan.entries.is_empty() {
        println!(
            "Sized profiles under `profiles:` that no stratum ({}) names:",
            if strata.is_empty() { "none declared".to_string() } else { strata.join(", ") }
        );
        for o in &plan.entries {
            match &o.dir {
                Some(d) => println!("  {:>8}  {:>10}  {}", o.name, format_size(o.bytes), crate::check::rel_display(d)),
                None => println!("  {:>8}  {:>10}  (no directory)", o.name, "-"),
            }
        }
    }
    if !plan.dirs.is_empty() {
        println!("Profile directories with no entry under `profiles:`:");
        for o in &plan.dirs {
            println!("  {:>8}  {:>10}  {}", o.name, format_size(o.bytes), crate::check::rel_display(o.dir.as_ref().unwrap()));
        }
    }
    println!(
        "\n{} entr{} to drop from the definition, {} director{} to remove ({}); the cache is left alone.",
        plan.entries.len(),
        if plan.entries.len() == 1 { "y" } else { "ies" },
        plan.entries.iter().filter(|o| o.dir.is_some()).count() + plan.dirs.len(),
        if plan.entries.iter().filter(|o| o.dir.is_some()).count() + plan.dirs.len() == 1 { "y" } else { "ies" },
        format_size(plan.bytes()),
    );
    if dry_run {
        println!("\nDry run — nothing changed. Run without --dry-run to apply.");
        return;
    }
    match apply(&mut config, &dataset_path, &plan) {
        Ok(removed) => println!(
            "\nRemoved {} profile entr{} and {} of profile directories.",
            plan.entries.len(),
            if plan.entries.len() == 1 { "y" } else { "ies" },
            format_size(removed)
        ),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(m) = std::fs::metadata(&p) {
                total += m.len();
            }
        }
    }
    total
}

fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{:.1} {}", size, UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A definition with two strata and one profile a removed stratum
    /// left behind, plus a directory nothing names.
    fn dataset(dir: &Path) -> PathBuf {
        let yaml = dir.join("dataset.yaml");
        std::fs::write(
            &yaml,
            "format_version: 2\nname: t\nupstream:\n  steps: []\nstrata:\n  decade:\n    spec: decade\n    series: ['100k', '200k']\nprofiles:\n  default:\n    base_vectors: profiles/base/base_vectors.fvecs\n  100k:\n    maxk: 5\n    base_count: 100000\n    base_vectors: profiles/base/base_vectors.fvecs[0..100000]\n  200k:\n    maxk: 5\n    base_count: 200000\n    base_vectors: profiles/base/base_vectors.fvecs[0..200000]\n  150k:\n    maxk: 5\n    base_count: 150000\n    base_vectors: profiles/base/base_vectors.fvecs[0..150000]\n",
        )
        .unwrap();
        for p in ["base", "default", "100k", "200k", "150k", "stray"] {
            std::fs::create_dir_all(dir.join("profiles").join(p)).unwrap();
        }
        std::fs::write(dir.join("profiles/150k/neighbor_indices.ivecs"), [0u8; 4000]).unwrap();
        std::fs::write(dir.join("profiles/stray/neighbor_indices.ivecs"), [0u8; 2000]).unwrap();
        std::fs::create_dir_all(dir.join(".cache/provenance/profiles/150k")).unwrap();
        std::fs::write(dir.join(".cache/keys.predkeys.slab"), [0u8; 10]).unwrap();
        std::fs::write(dir.join(".cache/.upstream.progress.yaml"), "schema_version: 6\nsteps: {}\n").unwrap();
        yaml
    }

    #[test]
    fn plans_the_entry_no_stratum_names_and_the_directory_no_entry_names() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = dataset(tmp.path());
        let config = DatasetConfig::load(&yaml).unwrap();
        let plan = plan(&config, tmp.path());
        assert_eq!(plan.entries.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(), vec!["150k"]);
        assert_eq!(plan.entries[0].bytes, 4000);
        assert_eq!(plan.dirs.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(), vec!["stray"]);
        assert_eq!(plan.bytes(), 6000);
    }

    /// Applying drops only the orphaned entry, removes only the
    /// orphaned directories, keeps the backup, and leaves the cache alone.
    #[test]
    fn apply_removes_orphans_and_nothing_else() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = dataset(tmp.path());
        let mut config = DatasetConfig::load(&yaml).unwrap();
        let plan = plan(&config, tmp.path());
        let removed = apply(&mut config, &yaml, &plan).unwrap();
        assert_eq!(removed, 6000);
        let after = DatasetConfig::load(&yaml).unwrap();
        let mut names: Vec<&String> = after.profiles.profiles.keys().collect();
        names.sort();
        assert_eq!(names, vec!["100k", "200k", "default"]);
        assert_eq!(after.strata.len(), 1, "strata untouched");
        assert!(!tmp.path().join("profiles/150k").exists());
        assert!(!tmp.path().join("profiles/stray").exists());
        for kept in ["profiles/base", "profiles/default", "profiles/100k", "profiles/200k",
                     ".cache/keys.predkeys.slab", ".cache/provenance/profiles/150k", ".cache/.upstream.progress.yaml"] {
            assert!(tmp.path().join(kept).exists(), "{kept} must remain");
        }
        assert!(plan_is_clean(&after, tmp.path()));
    }

    fn plan_is_clean(config: &DatasetConfig, dir: &Path) -> bool {
        plan(config, dir).is_empty()
    }

    /// With every sized profile named, nothing is planned — including
    /// when there are no strata at all, since then nothing generated them.
    #[test]
    fn a_definition_whose_strata_name_every_profile_plans_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let yaml = tmp.path().join("dataset.yaml");
        std::fs::write(&yaml, "format_version: 2\nname: t\nupstream:\n  steps: []\nstrata:\n  decade:\n    spec: decade\n    series: ['100k']\nprofiles:\n  default:\n    base_vectors: b.fvecs\n  100k:\n    base_count: 100000\n    base_vectors: b.fvecs[0..100000]\n").unwrap();
        let config = DatasetConfig::load(&yaml).unwrap();
        assert!(plan(&config, tmp.path()).is_empty());
    }
}
