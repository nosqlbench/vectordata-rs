// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Cache housekeeping — removes files under `.cache/` that nothing in
//! the current pipeline can use.
//!
//! A cache file is **live** when the definition, as it stands, can
//! still consume it. The sources of that knowledge are the pipeline's
//! own, never a list of names kept here:
//!
//! 1. **The manifest projection.** Every input, intermediate and output
//!    under the cache that any step names, and every option value that
//!    mentions the cache at all — so a step the registry does not know,
//!    or one whose variables cannot yet be resolved, still protects
//!    what it names.
//! 2. **Recorded outputs.** What the progress log says a step now in
//!    the definition wrote. Records of steps the definition no longer
//!    has protect nothing.
//! 3. **Cache claims.** What a command keeps under the cache beyond
//!    its manifest — a KNN engine's segment results for a `(base,
//!    query)` pair, the predicate-key segments of a facet, an extract's
//!    resume partitions — declared by the command through
//!    [`CommandOp::project_cache_claims`], as a name prefix or a
//!    directory.
//! 4. **Twins.** A live file's provenance sidecar and its gzip form.
//! 5. **Infrastructure.** The progress log and its migration backup,
//!    the run and governor logs, `meta.json`, and the `provenance/`
//!    tree of dataset-artifact sidecars.
//!
//! Everything else is an orphan: an intermediate of a step that was
//! removed or re-keyed, a segment of an engine cache version no longer
//! written, a partition directory of an output that no longer exists.
//! A directory that holds something live is walked, not removed, so
//! its dead siblings go and the live entries stay.
//!
//! [`CommandOp::project_cache_claims`]: veks_pipeline::pipeline::command::CommandOp::project_cache_claims

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use vectordata::dataset::DatasetConfig;
use veks_pipeline::pipeline::command::CacheClaim;
use veks_pipeline::pipeline::manifest;
use veks_pipeline::pipeline::progress::ProgressLog;
use veks_pipeline::pipeline::registry::CommandRegistry;

/// Name of the progress log under the cache directory.
const PROGRESS_LOG: &str = ".upstream.progress.yaml";

/// Files the runner itself keeps under the cache.
const PRESERVED_NAMES: &[&str] = &["run.log", ".governor.log", "meta.json"];

/// Name prefixes the runner itself keeps under the cache: the progress
/// log and the backups a schema migration leaves beside it.
const PRESERVED_PREFIXES: &[&str] = &[".upstream.progress"];

/// Directories the runner itself keeps under the cache, whole.
const PRESERVED_DIRS: &[&str] = &["provenance"];

/// One entry under the cache directory that nothing can use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Orphan {
    /// Path relative to the cache directory.
    pub rel: String,
    /// Bytes it holds (the subtree, for a directory).
    pub bytes: u64,
    /// Whether it is a directory.
    pub is_dir: bool,
}

/// What a cache-gc pass would remove, and how much it found live.
#[derive(Debug, Default, Clone)]
pub struct Plan {
    /// Entries found live, at every level walked.
    pub live: usize,
    /// Entries to remove, largest first.
    pub orphans: Vec<Orphan>,
}

impl Plan {
    /// Bytes the orphans hold.
    pub fn bytes(&self) -> u64 {
        self.orphans.iter().map(|o| o.bytes).sum()
    }
}

/// The liveness rule, assembled from the pipeline's own knowledge.
#[derive(Debug, Default)]
struct Liveness {
    /// Cache-relative paths named by a step, recorded as an output of a
    /// defined step, or preserved outright.
    names: HashSet<String>,
    /// Top-level name prefixes claimed by a command's cache.
    prefixes: Vec<String>,
    /// Cache-relative directories claimed whole.
    dirs: Vec<String>,
}

impl Liveness {
    /// Whether `rel` — a cache-relative path — is live, directly or as
    /// the sidecar or gzip twin of something live.
    fn is_live(&self, rel: &str) -> bool {
        if self.names.contains(rel) {
            return true;
        }
        if !rel.contains('/') && self.prefixes.iter().any(|p| rel.starts_with(p.as_str())) {
            return true;
        }
        if self.dirs.iter().any(|d| rel == d || rel.strip_prefix(d.as_str()).is_some_and(|r| r.starts_with('/'))) {
            return true;
        }
        if let Some(base) = rel.strip_suffix(".provenance.json") {
            return self.is_live(base);
        }
        if let Some(base) = rel.strip_suffix(".gz") {
            return self.is_live(base);
        }
        false
    }

    /// Whether the directory `rel` holds a live name or claimed
    /// directory somewhere beneath it, so it must be walked rather than
    /// removed.
    fn holds_live(&self, rel: &str) -> bool {
        let under = format!("{}/", rel);
        self.dirs.iter().any(|d| d.starts_with(&under))
            || self.names.iter().any(|n| n.starts_with(&under))
    }
}

/// Reduce a path as the pipeline writes it — `${cache}/x`, `.cache/x`,
/// `/abs/ws/.cache/x`, with or without a `[window]` — to its
/// cache-relative form, or `None` when it does not lie under the cache.
fn cache_relative(value: &str, cache_dir: &Path) -> Option<String> {
    let v = value.trim();
    let v = v.split('[').next().unwrap_or(v).trim();
    let rel: String = if let Some(r) = v.strip_prefix("${cache}/") {
        r.to_string()
    } else if let Some(r) = v.strip_prefix(".cache/").or_else(|| v.strip_prefix("./.cache/")) {
        r.to_string()
    } else if let Ok(r) = Path::new(v).strip_prefix(cache_dir) {
        r.to_string_lossy().into_owned()
    } else if let Some(i) = v.find("/.cache/") {
        // An absolute path under some other `.cache` — a workspace that
        // has since moved — still names the entry; keeping it is the
        // safe reading.
        v[i + "/.cache/".len()..].to_string()
    } else {
        return None;
    };
    let rel = rel.trim_start_matches("./").trim_end_matches('/').to_string();
    if rel.is_empty() || rel == "." {
        None
    } else {
        Some(rel)
    }
}

/// Decide what under `.cache/` nothing in `config` can use.
pub fn plan(dataset_path: &Path, config: &DatasetConfig) -> Result<Plan, String> {
    let workspace = dataset_path.parent().unwrap_or(Path::new("."));
    let cache_dir = workspace.join(".cache");
    let registry = CommandRegistry::with_builtins();
    let wm = manifest::project_workspace(dataset_path, config, &registry)?;

    let mut live = Liveness::default();
    for name in PRESERVED_NAMES {
        live.names.insert((*name).to_string());
    }
    for prefix in PRESERVED_PREFIXES {
        live.prefixes.push((*prefix).to_string());
    }
    for dir in PRESERVED_DIRS {
        live.dirs.push((*dir).to_string());
    }
    for p in wm.retained_cache_paths() {
        if let Some(rel) = cache_relative(&p, &cache_dir) {
            live.names.insert(rel);
        }
    }
    let progress_path = cache_dir.join(PROGRESS_LOG);
    if progress_path.exists() {
        let (log, _) = ProgressLog::load(&progress_path)?;
        for (id, record) in &log.steps {
            if !wm.step_ids.contains(id) {
                continue;
            }
            for output in &record.outputs {
                if let Some(rel) = cache_relative(&output.path, &cache_dir) {
                    live.names.insert(rel);
                }
            }
        }
    }
    for claim in &wm.cache_claims {
        match claim {
            CacheClaim::Prefix(p) => live.prefixes.push(p.clone()),
            CacheClaim::Dir(d) => live.dirs.push(d.trim_matches('/').to_string()),
        }
    }

    let mut plan = Plan::default();
    walk(&cache_dir, "", &live, &mut plan);
    plan.orphans.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.rel.cmp(&b.rel)));
    Ok(plan)
}

fn walk(dir: &Path, rel_prefix: &str, live: &Liveness, plan: &mut Plan) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel = if rel_prefix.is_empty() { name } else { format!("{}/{}", rel_prefix, name) };
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        if live.is_live(&rel) {
            plan.live += 1;
            continue;
        }
        if is_dir && live.holds_live(&rel) {
            walk(&entry.path(), &rel, live, plan);
            continue;
        }
        let bytes = if is_dir {
            dir_size(&entry.path())
        } else {
            entry.metadata().map(|m| m.len()).unwrap_or(0)
        };
        plan.orphans.push(Orphan { rel, bytes, is_dir });
    }
}

/// Remove the plan's orphans from under `cache_dir`. Returns the bytes
/// freed; an entry that cannot be removed is reported and the rest
/// still go.
pub fn apply(cache_dir: &Path, plan: &Plan) -> Result<u64, String> {
    let mut freed = 0u64;
    let mut failed: Vec<String> = Vec::new();
    for o in &plan.orphans {
        let path: PathBuf = cache_dir.join(&o.rel);
        let result = if o.is_dir {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match result {
            Ok(()) => freed += o.bytes,
            Err(e) => failed.push(format!("{}: {}", o.rel, e)),
        }
    }
    if failed.is_empty() {
        Ok(freed)
    } else {
        Err(format!("could not remove {} entr{}:\n  {}", failed.len(), if failed.len() == 1 { "y" } else { "ies" }, failed.join("\n  ")))
    }
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
    let cache_dir = dataset_dir.join(".cache");
    if !cache_dir.exists() {
        println!("No .cache directory found — nothing to clean.");
        return;
    }
    let config = DatasetConfig::load(&dataset_path).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    let plan = plan(&dataset_path, &config).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    if plan.orphans.is_empty() {
        println!("Nothing under .cache/ is orphaned ({} live entries).", plan.live);
        return;
    }
    println!(
        "Orphaned under .cache/ ({} live, {} orphaned, {}):",
        plan.live,
        plan.orphans.len(),
        format_size(plan.bytes())
    );
    for o in &plan.orphans {
        let kind = if o.is_dir { "dir " } else { "file" };
        println!("  {} {:>10}  {}", kind, format_size(o.bytes), o.rel);
    }
    if dry_run {
        println!("\nDry run — nothing removed. Run without --dry-run to remove.");
        return;
    }
    match apply(&cache_dir, &plan) {
        Ok(freed) => println!("\nRemoved {} orphaned entries ({}).", plan.orphans.len(), format_size(freed)),
        Err(e) => {
            eprintln!("\nError: {}", e);
            std::process::exit(1);
        }
    }
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.filter_map(|e| e.ok()) {
            if let Ok(meta) = entry.metadata() {
                if meta.is_dir() {
                    total += dir_size(&entry.path());
                } else {
                    total += meta.len();
                }
            }
        }
    }
    total
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use veks_pipeline::pipeline::command::Status;
    use veks_pipeline::pipeline::progress::{OutputRecord, StepRecord};

    fn tmp_dir() -> tempfile::TempDir {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("target/tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    fn plant(path: &Path, bytes: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, vec![7u8; bytes]).unwrap();
    }

    fn record(path: &str) -> StepRecord {
        StepRecord {
            status: Status::Ok,
            message: String::new(),
            completed_at: chrono::Utc::now(),
            elapsed_secs: 0.0,
            outputs: vec![OutputRecord { path: path.to_string(), size: 1, mtime: None }],
            resolved_options: Default::default(),
            error: None,
            resource_summary: None,
            provenance: None,
        }
    }

    const DATASET: &str = "format_version: 2
name: t
upstream:
  defaults:
    query_count: '2'
  steps:
  - id: extract-margin
    run: transform extract
    source: ${cache}/margin.slab
    ivec-file: ${cache}/shuffle.ivecs
    output: profiles/base/topic_margin.mvecs
  - id: knn
    run: compute knn-stdarch
    base: profiles/base/base_vectors.fvecs[0..10)
    query: profiles/base/query_vectors.fvecs
    output: profiles/base/gt.ivecs
    k: '10'
  - id: evaluate
    run: compute evaluate-predicates
    source: profiles/base/metadata.slab
    predicates: profiles/base/predicates.slab
    output: profiles/base/results.slab
  - id: mystery
    run: no such command
    input: ${cache}/named_by_unknown.bin
profiles:
  default:
    base_vectors: profiles/base/base_vectors.fvecs
";

    /// Every source of liveness protects what it should, and nothing
    /// else survives: a name a step gives, an output a defined step
    /// recorded, an engine's segments for the step's pair, a facet's
    /// key segments, an extract's resume directory, the twins of each,
    /// and the runner's own files. A name only a removed step recorded,
    /// a segment of another pair, and plain junk go.
    #[test]
    fn plan_keeps_what_the_definition_can_use_and_drops_the_rest() {
        let dir = tmp_dir();
        let ws = dir.path();
        let dataset_path = ws.join("dataset.yaml");
        std::fs::write(&dataset_path, DATASET).unwrap();
        for f in ["base_vectors.fvecs", "query_vectors.fvecs", "metadata.slab", "predicates.slab"] {
            plant(&ws.join("profiles/base").join(f), 64);
        }
        let cache = ws.join(".cache");

        let config = DatasetConfig::load(&dataset_path).unwrap();
        let wm = manifest::project_workspace(&dataset_path, &config, &CommandRegistry::with_builtins()).unwrap();
        let prefixes: Vec<&str> = wm
            .cache_claims
            .iter()
            .filter_map(|c| match c {
                CacheClaim::Prefix(p) => Some(p.as_str()),
                _ => None,
            })
            .collect();
        let dirs: Vec<&str> = wm
            .cache_claims
            .iter()
            .filter_map(|c| match c {
                CacheClaim::Dir(d) => Some(d.as_str()),
                _ => None,
            })
            .collect();
        let knn_prefix = prefixes.iter().find(|p| p.starts_with("knn-stdarch.v3.")).expect("engine claim");
        let keys_prefix = prefixes.iter().find(|p| p.starts_with("keys.")).expect("facet segment claim");
        let extract_dir = dirs.iter().find(|d| d.starts_with("slab-extract/")).expect("extract resume claim");

        let kept = [
            "shuffle.ivecs".to_string(),
            "margin.slab".to_string(),
            "named_by_unknown.bin".to_string(),
            "topic_assign.u16vecs".to_string(),
            "topic_assign.u16vecs.provenance.json".to_string(),
            "topic_assign.u16vecs.gz".to_string(),
            format!("{}range_000000000000_000000000010.k10.l2.results.bin", knn_prefix),
            format!("{}range_000000000000_000000000010.k10.l2.results.bin.provenance.json", knn_prefix),
            format!("{}seg_0000000000_0000000010.predkeys.slab", keys_prefix),
            format!("{}/0000000000_0000000005.slab", extract_dir),
            "provenance/profiles/base/gt.ivecs.provenance.json".to_string(),
            ".upstream.progress.v5.yaml".to_string(),
            "run.log".to_string(),
            ".governor.log".to_string(),
            "meta.json".to_string(),
        ];
        let dropped = [
            "stale.bin".to_string(),
            "stale.bin.provenance.json".to_string(),
            "knn-stdarch.v3.other.query.1_1.range_000000000000_000000000010.k10.l2.results.bin".to_string(),
            "knn-blas.v2.old.range_000000000000_000000000010.k10.l2.results.bin".to_string(),
            "keys.deadbeef.seg_0000000000_0000000010.predkeys.slab".to_string(),
            "slab-extract/gone-0123/0000000000_0000000005.slab".to_string(),
            "junk.bin".to_string(),
            "dedup_runs/old.log".to_string(),
        ];
        for f in kept.iter().chain(dropped.iter()) {
            plant(&cache.join(f), 8);
        }
        let (mut log, _) = ProgressLog::load(&cache.join(PROGRESS_LOG)).unwrap();
        log.record_step("knn", record(".cache/topic_assign.u16vecs"));
        log.record_step("gone", record(".cache/stale.bin"));
        log.save().unwrap();

        let plan = plan(&dataset_path, &config).unwrap();
        let orphaned: HashSet<String> = plan.orphans.iter().map(|o| o.rel.clone()).collect();
        let expected: HashSet<String> = [
            "stale.bin",
            "stale.bin.provenance.json",
            "knn-stdarch.v3.other.query.1_1.range_000000000000_000000000010.k10.l2.results.bin",
            "knn-blas.v2.old.range_000000000000_000000000010.k10.l2.results.bin",
            "keys.deadbeef.seg_0000000000_0000000010.predkeys.slab",
            "slab-extract/gone-0123",
            "junk.bin",
            "dedup_runs",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(orphaned, expected, "orphans: {:?}", plan.orphans);
        assert!(plan.orphans.iter().find(|o| o.rel == "slab-extract/gone-0123").unwrap().is_dir);
        assert!(plan.orphans.windows(2).all(|w| w[0].bytes >= w[1].bytes), "largest first");

        let freed = apply(&cache, &plan).unwrap();
        assert_eq!(freed, plan.bytes());
        for f in &kept {
            assert!(cache.join(f).exists(), "{} should survive", f);
        }
        assert!(cache.join(PROGRESS_LOG).exists());
        for f in &dropped {
            assert!(!cache.join(f).exists(), "{} should be gone", f);
        }
        assert!(cache.join("slab-extract").is_dir(), "the claimed directory's parent is walked, not removed");

        let again = super::plan(&dataset_path, &config).unwrap();
        assert!(again.orphans.is_empty(), "a second pass finds nothing: {:?}", again.orphans);
    }

    #[test]
    fn cache_relative_reduces_every_spelling() {
        let cache = Path::new("/ws/.cache");
        for (v, want) in [
            ("${cache}/a.bin", Some("a.bin")),
            (".cache/a.bin", Some("a.bin")),
            ("./.cache/a.bin", Some("a.bin")),
            ("/ws/.cache/a.bin", Some("a.bin")),
            ("/ws/.cache/slab-extract/x/part", Some("slab-extract/x/part")),
            ("${cache}/a.bin[0,10)", Some("a.bin")),
            ("../other/.cache/a.bin", Some("a.bin")),
            ("profiles/base/a.bin", None),
            ("/elsewhere/.cache/a.bin", Some("a.bin")),
            ("${cache}/", None),
            ("/other/a.bin", None),
        ] {
            assert_eq!(cache_relative(v, cache).as_deref(), want, "{}", v);
        }
    }

    #[test]
    fn twins_and_claims_follow_their_subject() {
        let mut live = Liveness::default();
        live.names.insert("a.bin".into());
        live.names.insert("nested/b.bin".into());
        live.prefixes.push("knn-x.v3.p.".into());
        live.dirs.push("slab-extract/out-1".into());
        assert!(live.is_live("a.bin"));
        assert!(live.is_live("a.bin.provenance.json"));
        assert!(live.is_live("a.bin.gz"));
        assert!(live.is_live("a.bin.gz.provenance.json"));
        assert!(!live.is_live("a.bin.bak"));
        assert!(live.is_live("knn-x.v3.p.range_1.k10.l2.results.bin"));
        assert!(!live.is_live("knn-x.v2.p.range_1.k10.l2.results.bin"));
        assert!(!live.is_live("sub/knn-x.v3.p.range_1.k10.l2.results.bin"), "prefix claims are top-level");
        assert!(live.is_live("slab-extract/out-1"));
        assert!(live.is_live("slab-extract/out-1/part"));
        assert!(!live.is_live("slab-extract/out-10"));
        assert!(!live.is_live("slab-extract"));
        assert!(live.holds_live("slab-extract"));
        assert!(live.holds_live("nested"));
        assert!(!live.holds_live("other"));
    }
}
