// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Client-driven, resumable, content-addressed off-system backup of a
//! `vecd` store, and its inverse.
//!
//! `backup` walks the introspection API for everything the caller can
//! read — visible namespaces (`/-/whoami`), their version history
//! (`/-/versions`), each version's manifest, and the content — and mirrors
//! it into a **copy-on-write, content-addressed** tree:
//!
//! ```text
//! <dest>/blobs/<content-key>           # write-once content blobs (dedup)
//! <dest>/ns/<namespace>/versions.json  # the append-only version list
//! <dest>/ns/<namespace>/@<tag>/manifest.json   # one per committed version
//! ```
//!
//! It is **resumable**: a content blob already present is never re-fetched
//! (content-key presence = done), and with `--incremental` a version whose
//! `manifest.json` is already written (the completion marker, written
//! *last*) is skipped — so a re-run finishes only the missing work.
//!
//! `restore` republishes a mirror's latest version of each namespace into a
//! target `vecd` via the ordinary push path. (Full version-history replay
//! is a future extension; latest-state restore covers the common case.)

use std::path::{Path, PathBuf};

use crate::endpoint;

/// What a backup run moved.
#[derive(Debug, Default, Clone)]
pub struct BackupStats {
    pub namespaces: usize,
    pub versions: usize,
    pub versions_skipped: usize,
    pub blobs_fetched: usize,
    pub blobs_skipped: usize,
    pub bytes: u64,
}

/// What a restore run pushed.
#[derive(Debug, Default, Clone)]
pub struct RestoreStats {
    pub namespaces: usize,
    pub objects: usize,
}

/// Strip an optional `file://` scheme.
fn local_dir(dest: &str) -> PathBuf {
    PathBuf::from(dest.strip_prefix("file://").unwrap_or(dest))
}

/// Mirror everything the caller can read at `url` into `dest`.
pub fn run_backup(
    url: &str,
    dest: &str,
    incremental: bool,
    token: Option<&str>,
) -> Result<BackupStats, String> {
    let root = local_dir(dest);
    let blobs = root.join("blobs");
    std::fs::create_dir_all(&blobs).map_err(|e| e.to_string())?;

    // Enumerate readable namespaces from the caller's own access view.
    let view = endpoint::whoami(url, token)?;
    let readable: Vec<String> = view
        .get("namespaces")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|n| {
                    n.get("actions")
                        .and_then(|a| a.as_array())
                        .map(|a| a.iter().any(|x| x.as_str() == Some("read")))
                        .unwrap_or(false)
                })
                .filter_map(|n| n.get("path").and_then(|p| p.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let mut stats = BackupStats::default();
    for ns in &readable {
        let versions = endpoint::versions(url, ns, token)?;
        let list = versions.get("versions").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        // Only committed (downloadable) versions are mirrored.
        let committed: Vec<&serde_json::Value> =
            list.iter().filter(|v| v.get("state").and_then(|s| s.as_str()) == Some("committed")).collect();
        if committed.is_empty() {
            continue;
        }
        stats.namespaces += 1;

        let ns_dir = root.join("ns").join(ns);
        std::fs::create_dir_all(&ns_dir).map_err(|e| e.to_string())?;
        std::fs::write(
            ns_dir.join("versions.json"),
            serde_json::to_vec_pretty(&serde_json::json!({ "namespace": ns, "versions": list }))
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        for v in committed {
            let tag = v.get("tag").and_then(|t| t.as_str()).unwrap_or("");
            let vdir = ns_dir.join(format!("@{tag}"));
            let manifest_path = vdir.join("manifest.json");
            if incremental && manifest_path.exists() {
                stats.versions_skipped += 1;
                continue;
            }
            std::fs::create_dir_all(&vdir).map_err(|e| e.to_string())?;

            let manifest = endpoint::manifest(url, ns, tag, token)?;
            let objects = manifest.get("objects").and_then(|o| o.as_array()).cloned().unwrap_or_default();
            for obj in &objects {
                let key = obj.get("key").and_then(|k| k.as_str()).unwrap_or("");
                let ck = obj.get("content_key").and_then(|c| c.as_str()).unwrap_or("");
                if key.is_empty() || ck.is_empty() {
                    continue;
                }
                let blob_path = blobs.join(ck);
                // content-key presence = done (dedup + resume).
                if blob_path.exists() {
                    stats.blobs_skipped += 1;
                    continue;
                }
                let bytes = endpoint::get_object(url, ns, tag, key, token)?;
                atomic_write(&blob_path, &bytes).map_err(|e| e.to_string())?;
                stats.blobs_fetched += 1;
                stats.bytes += bytes.len() as u64;
            }
            // The manifest is written LAST — its presence marks the version
            // complete, which is what makes an interrupted backup resumable.
            atomic_write(&manifest_path, &serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            stats.versions += 1;
        }
    }
    Ok(stats)
}

/// Republish each mirrored namespace's latest version into `url`.
pub fn run_restore(src: &str, url: &str, token: Option<&str>) -> Result<RestoreStats, String> {
    let root = local_dir(src);
    let blobs = root.join("blobs");
    let ns_root = root.join("ns");

    let mut stats = RestoreStats::default();
    for versions_json in find_versions_files(&ns_root) {
        let text = std::fs::read_to_string(&versions_json).map_err(|e| e.to_string())?;
        let doc: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let ns = doc.get("namespace").and_then(|n| n.as_str()).ok_or("mirror versions.json missing namespace")?;
        // Latest committed version = highest seq.
        let latest = doc
            .get("versions")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                arr.iter()
                    .filter(|v| v.get("state").and_then(|s| s.as_str()) == Some("committed"))
                    .max_by_key(|v| v.get("seq").and_then(|s| s.as_i64()).unwrap_or(0))
            })
            .cloned();
        let Some(latest) = latest else { continue };
        let tag = latest.get("tag").and_then(|t| t.as_str()).unwrap_or("");

        let manifest_path = versions_json.parent().unwrap().join(format!("@{tag}")).join("manifest.json");
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;

        // Materialize the version's files into a temp dir and push it.
        let staging = tempdir_under(&root)?;
        let objects = manifest.get("objects").and_then(|o| o.as_array()).cloned().unwrap_or_default();
        for obj in &objects {
            let key = obj.get("key").and_then(|k| k.as_str()).unwrap_or("");
            let ck = obj.get("content_key").and_then(|c| c.as_str()).unwrap_or("");
            // Skip the provenance artifacts — push regenerates them.
            if key.is_empty()
                || key == "pushlog.jsonl"
                || key == ".publish_url"
                || key == "SHA256SUMS"
                || key.ends_with("/SHA256SUMS")
            {
                continue;
            }
            let bytes = std::fs::read(blobs.join(ck)).map_err(|e| format!("missing blob {ck}: {e}"))?;
            let target = staging.join(key.replace('/', std::path::MAIN_SEPARATOR_STR));
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(&target, &bytes).map_err(|e| e.to_string())?;
            stats.objects += 1;
        }

        push_dir(&staging, &format!("{}/{ns}/", url.trim_end_matches('/')), token)?;
        stats.namespaces += 1;
        let _ = std::fs::remove_dir_all(&staging);
    }
    Ok(stats)
}

/// Push a reconstructed dataset directory to a target namespace URL.
fn push_dir(dir: &Path, to: &str, token: Option<&str>) -> Result<(), String> {
    use crate::push::transport::TransportOptions;
    use crate::push::{execute, ChecksumPolicy, Options};
    let opts = Options {
        path: dir.to_path_buf(),
        to: Some(to.to_string()),
        message: Some("vectordata restore".to_string()),
        raw: false,
        checksums: ChecksumPolicy::Auto,
        dry_run: false,
        no_check: true,
        assume_yes: true,
        delete: false,
        abort_incomplete: false,
        concurrency: 4,
        files: None,
        transport: TransportOptions { token: token.map(String::from), profile: None, endpoint_url: None },
        cmd: "vectordata restore".into(),
        actor: "vectordata restore".into(),
    };
    execute(&opts).map(|_| ()).map_err(|e| format!("restoring {to}: {e:?}"))
}

/// Recursively find every `versions.json` under the mirror's `ns/` root.
fn find_versions_files(ns_root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.file_name().and_then(|n| n.to_str()) == Some("versions.json") {
                out.push(p);
            }
        }
    }
    walk(ns_root, &mut out);
    out
}

/// Atomic temp+rename write.
fn atomic_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)
}

fn tempdir_under(root: &Path) -> Result<PathBuf, String> {
    let dir = root.join(format!(".restore-staging-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}
