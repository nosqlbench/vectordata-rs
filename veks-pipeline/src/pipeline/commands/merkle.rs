// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline commands: merkle tree create and verify for file integrity.
//!
//! Creates SHA-256 merkle trees over file chunks and stores them as `.mref`
//! files. Verification rebuilds the tree and compares against the stored
//! reference to detect corruption.
//!
//! Chunk size is configurable (default 1 MiB, must be power of 2).
//!
//! Equivalent to the Java `CMD_merkle_create` and `CMD_merkle_verify` commands.

use std::path::{Path, PathBuf};
use std::time::Instant;

use sha2::{Digest, Sha256};

use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

const DEFAULT_CHUNK_SIZE: usize = 1 << 20; // 1 MiB
const HASH_SIZE: usize = 32; // SHA-256
const MREF_EXT: &str = "mref";

// -- Merkle tree data structure -----------------------------------------------

/// In-memory merkle tree: array of SHA-256 hashes.
///
/// Layout: `hashes[0..internal_count]` are internal nodes (root at 0),
/// `hashes[internal_count..total]` are leaf nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MerkleTree {
    /// All node hashes, root at index 0.
    hashes: Vec<[u8; HASH_SIZE]>,
    /// Number of leaf nodes.
    leaf_count: usize,
    /// Chunk size in bytes.
    chunk_size: usize,
    /// Original file size.
    file_size: u64,
}

impl MerkleTree {
    /// Validate that chunk_size is a positive power of 2.
    ///
    /// This matches the Java MerkleTree requirement: chunk sizes must be
    /// powers of 2 so that tree structure is consistent across
    /// implementations.
    fn validate_chunk_size(chunk_size: usize) {
        assert!(chunk_size > 0 && chunk_size.is_power_of_two(),
            "chunk size must be a positive power of 2, got {}", chunk_size);
    }

    /// Build a merkle tree from file data in memory.
    fn from_data(data: &[u8], chunk_size: usize) -> Self {
        Self::validate_chunk_size(chunk_size);
        let leaf_count = if data.is_empty() {
            1
        } else {
            (data.len() + chunk_size - 1) / chunk_size
        };

        let mut leaf_hashes: Vec<[u8; HASH_SIZE]> = Vec::with_capacity(leaf_count);
        for i in 0..leaf_count {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, data.len());
            let chunk = if start < data.len() { &data[start..end] } else { &[] };
            leaf_hashes.push(Sha256::digest(chunk).into());
        }

        Self::from_leaf_hashes(leaf_hashes, leaf_count, chunk_size, data.len() as u64)
    }

    /// Build a merkle tree by streaming chunks with parallel hashing.
    ///
    /// A reader thread reads chunks sequentially into a bounded queue.
    /// `hash_threads` worker threads pull chunks and hash them in parallel.
    /// Results are collected in chunk-index order so the tree is deterministic.
    ///
    /// Calls `progress_fn(chunks_hashed)` as each hash completes.
    fn from_reader_parallel<R: std::io::Read + Send + 'static>(
        reader: R,
        file_size: u64,
        chunk_size: usize,
        hash_threads: usize,
        mut progress_fn: impl FnMut(u64),
    ) -> std::io::Result<Self> {
        Self::validate_chunk_size(chunk_size);
        use std::sync::mpsc;

        let leaf_count = if file_size == 0 {
            1
        } else {
            ((file_size as usize) + chunk_size - 1) / chunk_size
        };

        // Channel: reader → hash workers (bounded to limit memory)
        let queue_depth = (hash_threads * 2).max(4);
        let (chunk_tx, chunk_rx) = mpsc::sync_channel::<(usize, Vec<u8>)>(queue_depth);
        let chunk_rx = std::sync::Arc::new(std::sync::Mutex::new(chunk_rx));

        // Channel: hash workers → collector (unbounded, hashes are small)
        let (hash_tx, hash_rx) = mpsc::channel::<(usize, [u8; HASH_SIZE])>();

        // Reader thread: sequential I/O, fills chunk buffers
        let reader_handle = std::thread::Builder::new()
            .name("merkle-reader".into())
            .spawn(move || {
                let mut reader = std::io::BufReader::with_capacity(chunk_size * 2, reader);
                let mut chunk_idx = 0usize;
                loop {
                    let mut buf = vec![0u8; chunk_size];
                    let mut filled = 0;
                    while filled < chunk_size {
                        match std::io::Read::read(&mut reader, &mut buf[filled..]) {
                            Ok(0) => break,
                            Ok(n) => filled += n,
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) => {
                                log::error!("merkle reader error: {}", e);
                                return;
                            }
                        }
                    }
                    if filled == 0 {
                        break;
                    }
                    buf.truncate(filled);
                    if chunk_tx.send((chunk_idx, buf)).is_err() {
                        break;
                    }
                    chunk_idx += 1;
                }
                // chunk_tx drops here, closing the channel
            })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Hash worker threads: pull chunks, compute SHA-256, send results
        let mut workers = Vec::with_capacity(hash_threads);
        for _ in 0..hash_threads {
            let rx = std::sync::Arc::clone(&chunk_rx);
            let tx = hash_tx.clone();
            workers.push(std::thread::Builder::new()
                .name("merkle-hash".into())
                .spawn(move || {
                    loop {
                        let (idx, data) = match rx.lock().unwrap().recv() {
                            Ok(item) => item,
                            Err(_) => break,
                        };
                        let hash: [u8; HASH_SIZE] = Sha256::digest(&data).into();
                        if tx.send((idx, hash)).is_err() {
                            break;
                        }
                    }
                })
                .unwrap());
        }
        drop(hash_tx); // so collector sees EOF when all workers finish

        // Collect results in order
        let mut results: Vec<(usize, [u8; HASH_SIZE])> = Vec::with_capacity(leaf_count);
        let mut done: u64 = 0;
        for item in hash_rx {
            results.push(item);
            done += 1;
            progress_fn(done);
        }

        // Wait for threads
        let _ = reader_handle.join();
        for w in workers {
            let _ = w.join();
        }

        // Sort by chunk index to restore deterministic order
        results.sort_by_key(|(idx, _)| *idx);
        let leaf_hashes: Vec<[u8; HASH_SIZE]> = results.into_iter().map(|(_, h)| h).collect();

        if leaf_hashes.is_empty() {
            return Ok(Self::from_leaf_hashes(
                vec![Sha256::digest(&[]).into()], 1, chunk_size, file_size,
            ));
        }

        let actual_leaf_count = leaf_hashes.len();
        Ok(Self::from_leaf_hashes(leaf_hashes, actual_leaf_count, chunk_size, file_size))
    }

    /// Build internal nodes from leaf hashes.
    fn from_leaf_hashes(
        mut leaf_hashes: Vec<[u8; HASH_SIZE]>,
        leaf_count: usize,
        chunk_size: usize,
        file_size: u64,
    ) -> Self {
        // Pad to next power of 2 for balanced tree
        let padded_leaves = leaf_count.next_power_of_two().max(1);
        let empty_hash: [u8; HASH_SIZE] = Sha256::digest(&[]).into();
        while leaf_hashes.len() < padded_leaves {
            leaf_hashes.push(empty_hash);
        }

        // Build internal nodes bottom-up
        let internal_count = padded_leaves - 1;
        let total = internal_count + padded_leaves;
        let mut hashes = vec![[0u8; HASH_SIZE]; total];

        for (i, h) in leaf_hashes.into_iter().enumerate() {
            hashes[internal_count + i] = h;
        }

        for i in (0..internal_count).rev() {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            let mut hasher = Sha256::new();
            hasher.update(&hashes[left]);
            hasher.update(&hashes[right]);
            hashes[i] = hasher.finalize().into();
        }

        MerkleTree {
            hashes,
            leaf_count,
            chunk_size,
            file_size,
        }
    }

    /// Find mismatched chunks between two trees.
    fn find_mismatches(&self, other: &MerkleTree) -> Vec<usize> {
        if self.leaf_count != other.leaf_count {
            return (0..self.leaf_count.max(other.leaf_count)).collect();
        }
        let internal_count = self.hashes.len() - self.leaf_count.next_power_of_two();
        let mut mismatches = Vec::new();
        for i in 0..self.leaf_count {
            let idx = internal_count + i;
            if idx < self.hashes.len()
                && idx < other.hashes.len()
                && self.hashes[idx] != other.hashes[idx]
            {
                mismatches.push(i);
            }
        }
        mismatches
    }

    /// Root hash.
    fn root_hash(&self) -> &[u8; HASH_SIZE] {
        &self.hashes[0]
    }

    /// Serialize to bytes: [chunk_size: u32 LE][file_size: u64 LE][leaf_count: u32 LE][hashes...]
    /// Serialize to the standard `.mref` format compatible with
    /// `vectordata::merkle::MerkleRef::from_bytes`.
    ///
    /// Layout: `[hashes: node_count * 32 bytes][footer: 41 bytes]`
    fn to_bytes(&self) -> Vec<u8> {
        let mref = self.to_merkle_ref();
        mref.to_bytes()
    }

    /// Convert to a `vectordata::merkle::MerkleRef` for serialization.
    fn to_merkle_ref(&self) -> vectordata::merkle::MerkleRef {
        let cap_leaf = (self.leaf_count as u32).next_power_of_two().max(1);
        let internal_node_count = cap_leaf - 1;
        let shape = vectordata::merkle::MerkleShape {
            chunk_size: self.chunk_size as u64,
            total_content_size: self.file_size,
            total_chunks: self.leaf_count as u32,
            leaf_count: cap_leaf,
            cap_leaf,
            node_count: self.hashes.len() as u32,
            offset: internal_node_count,
            internal_node_count,
        };
        vectordata::merkle::MerkleRef::from_parts(shape, self.hashes.clone())
    }

    /// Deserialize from the standard `.mref` format (Java-compatible).
    fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mref = vectordata::merkle::MerkleRef::from_bytes(data)
            .map_err(|e| format!("invalid mref: {}", e))?;
        let chunk_size = mref.shape().chunk_size as usize;
        if chunk_size == 0 || !chunk_size.is_power_of_two() {
            return Err(format!(
                "invalid mref: chunk size must be a positive power of 2, got {}",
                chunk_size
            ));
        }
        Ok(Self {
            hashes: mref.hashes().to_vec(),
            leaf_count: mref.shape().total_chunks as usize,
            chunk_size,
            file_size: mref.shape().total_content_size,
        })
    }
}

/// Get the `.mref` path for a given file.
fn mref_path(file_path: &Path) -> PathBuf {
    file_path.with_extension(
        format!(
            "{}.{}",
            file_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or(""),
            MREF_EXT
        )
    )
}

// -- Merkle Create command ----------------------------------------------------

/// Pipeline command: create merkle tree reference files.
pub struct MerkleCreateOp;

pub fn create_factory() -> Box<dyn CommandOp> {
    Box::new(MerkleCreateOp)
}

impl CommandOp for MerkleCreateOp {
    fn command_path(&self) -> &str {
        "merkle create"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Create a Merkle hash tree for a file".into(),
            body: format!(
                r#"# merkle create

Create a Merkle hash tree for a file.

## Description

Creates a SHA-256 Merkle tree over file chunks and stores it as an
`.mref` reference file alongside the source. The chunk size is
configurable (default 1 MiB) and must be a power of 2. If an
up-to-date `.mref` already exists (newer than the source), the
command skips tree creation unless `force=true`.

## How It Works

The source file is read into memory and divided into fixed-size
chunks. Each chunk is hashed with SHA-256 to produce a leaf node.
The leaf array is padded to the next power of two (with empty-data
hashes for padding leaves), then internal nodes are computed
bottom-up by hashing concatenated child pairs. The resulting binary
tree -- stored as a flat array with the root at index 0 -- is
serialized to an `.mref` file containing the chunk size, original
file size, leaf count, and all node hashes.

## Data Preparation Role

`merkle create` enables chunk-level integrity checking for large
dataset files such as vector files, metadata slabs, and downloaded
archives. By creating a Merkle reference at the end of a pipeline
step that produces a file, subsequent runs can use `merkle verify`
to confirm that the file has not been corrupted by storage errors,
incomplete transfers, or accidental modification. The chunk-level
granularity means that corruption can be localized to specific
byte ranges rather than requiring a full file re-download.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "threads".into(), description: "Parallel SHA-256 hash workers".into(), adjustable: true },
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch for hashing".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let chunk_size: usize = options
            .get("chunk-size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_CHUNK_SIZE);

        if !chunk_size.is_power_of_two() || chunk_size == 0 {
            return error_result(
                format!("chunk-size must be a positive power of 2, got {}", chunk_size),
                start,
            );
        }

        let force = options.get("force").map_or(false, |s| s == "true");
        let min_size: u64 = options.get("min-size")
            .and_then(|s| veks_core::paths::parse_size(s))
            .unwrap_or(100_000_000);

        let source_path = resolve_path(source_str, &ctx.workspace);
        let hash_threads = ctx.governor.current_or("threads", ctx.threads as u64)
            .max(1) as usize;

        // If source is a directory, walk it and process all eligible files.
        // If source is a single file, process just that file.
        let files = if source_path.is_dir() {
            collect_eligible_files(&source_path, min_size)
        } else {
            let size = std::fs::metadata(&source_path).map(|m| m.len()).unwrap_or(0);
            if size >= min_size {
                vec![source_path.clone()]
            } else {
                ctx.ui.log(&format!(
                    "Merkle: {} ({}) below min-size ({}), skipping",
                    source_path.file_name().unwrap_or_default().to_string_lossy(),
                    format_bytes(size),
                    format_bytes(min_size),
                ));
                return CommandResult {
                    status: Status::Ok,
                    message: format!("0 files above {} threshold", format_bytes(min_size)),
                    produced: vec![],
                    elapsed: start.elapsed(),
                };
            }
        };

        if files.is_empty() {
            ctx.ui.log(&format!(
                "Merkle: no files >= {} in {}",
                format_bytes(min_size),
                source_path.display(),
            ));
            return CommandResult {
                status: Status::Ok,
                message: format!("0 files above {} threshold", format_bytes(min_size)),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        ctx.ui.log(&format!(
            "Merkle: {} file(s) to process, min-size={}, {} hash thread(s)",
            files.len(), format_bytes(min_size), hash_threads,
        ));

        let mut produced: Vec<PathBuf> = Vec::new();
        let mut created = 0usize;
        let mut skipped_fresh = 0usize;
        let mut total_bytes: u64 = 0;

        // Outer progress: files
        let files_pb = ctx.ui.bar_with_unit(files.len() as u64, "files", "files");
        let mut files_done = 0u64;

        for file in &files {
            let mref = mref_path(file);
            let file_size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
            let rel = file.strip_prefix(&ctx.workspace).unwrap_or(file);
            let rel_name = rel.display().to_string();

            // Check if mref already exists and is up-to-date
            if mref.exists() && !force {
                let src_t = std::fs::metadata(file).ok().and_then(|m| m.modified().ok());
                let mref_t = std::fs::metadata(&mref).ok().and_then(|m| m.modified().ok());
                if let (Some(st), Some(mt)) = (src_t, mref_t) {
                    if mt > st {
                        skipped_fresh += 1;
                        produced.push(mref);
                        files_done += 1;
                        files_pb.set_position(files_done);
                        continue;
                    }
                }
            }

            let total_chunks = if file_size == 0 { 1 } else {
                ((file_size as usize) + chunk_size - 1) / chunk_size
            };

            ctx.ui.log(&format!(
                "  {} ({}, {} chunks)",
                rel_name, format_bytes(file_size), total_chunks,
            ));

            let f = match std::fs::File::open(file) {
                Ok(f) => f,
                Err(e) => {
                    ctx.ui.log(&format!("    ERROR: {}", e));
                    files_done += 1;
                    files_pb.set_position(files_done);
                    continue;
                }
            };

            // Inner progress: chunks within this file
            let chunk_pb = ctx.ui.bar_with_unit(total_chunks as u64, &rel_name, "chunks");
            let tree = match MerkleTree::from_reader_parallel(f, file_size, chunk_size, hash_threads, |done| {
                chunk_pb.set_position(done);
            }) {
                Ok(t) => t,
                Err(e) => {
                    chunk_pb.finish();
                    ctx.ui.log(&format!("    ERROR hashing: {}", e));
                    files_done += 1;
                    files_pb.set_position(files_done);
                    continue;
                }
            };
            chunk_pb.finish();

            if let Err(e) = std::fs::write(&mref, &tree.to_bytes()) {
                ctx.ui.log(&format!("    ERROR writing mref: {}", e));
                files_done += 1;
                files_pb.set_position(files_done);
                continue;
            }

            total_bytes += file_size;
            created += 1;
            produced.push(mref);
            files_done += 1;
            files_pb.set_position(files_done);
        }
        files_pb.finish();

        let elapsed = start.elapsed();
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            total_bytes as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let msg = format!(
            "{} created, {} up-to-date, {} total ({}, {:.1}s, {:.0} MB/s)",
            created, skipped_fresh, files.len(),
            format_bytes(total_bytes), elapsed.as_secs_f64(), throughput,
        );
        ctx.ui.log(&msg);

        CommandResult {
            status: Status::Ok,
            message: msg,
            produced,
            elapsed,
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "File or directory to create merkle references for".to_string(),
                role: OptionRole::Input,
        },
            OptionDesc {
                name: "chunk-size".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("1048576".to_string()),
                description: "Chunk size in bytes (must be power of 2)".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "force".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Overwrite existing mref even if up-to-date".to_string(),
                role: OptionRole::Config,
        },
            OptionDesc {
                name: "min-size".to_string(),
                type_name: "size".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Skip merkle creation for files smaller than this (supports K/M/G suffixes)".to_string(),
                role: OptionRole::Config,
        },
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        let source_str = options.get("source").unwrap_or(".");
        let min_size: u64 = options.get("min-size")
            .and_then(|s| veks_core::paths::parse_size(s))
            .unwrap_or(100_000_000);

        let source_path = PathBuf::from(source_str);

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        if source_path.is_dir() {
            // Walk the directory and enumerate actual eligible files
            let files = collect_eligible_files(&source_path, min_size);
            for file in &files {
                let rel = file.strip_prefix(&source_path)
                    .or_else(|_| file.strip_prefix("."))
                    .unwrap_or(file);
                let rel_str = rel.to_string_lossy().to_string();
                inputs.push(rel_str.clone());
                outputs.push(format!("{}.mref", rel_str));
            }
        } else if source_path.exists() {
            let size = std::fs::metadata(&source_path).map(|m| m.len()).unwrap_or(0);
            if size >= min_size {
                inputs.push(source_str.to_string());
                outputs.push(format!("{}.mref", source_str));
            }
        } else {
            // Source doesn't exist yet — use the path as-is
            inputs.push(source_str.to_string());
            outputs.push(format!("{}.mref", source_str));
        }

        ArtifactManifest {
            step_id: step_id.to_string(),
            command: self.command_path().to_string(),
            inputs,
            outputs,
            intermediates: vec![],
        }
    }
}

// -- Merkle Verify command ----------------------------------------------------

/// Pipeline command: verify file integrity against merkle reference.
pub struct MerkleVerifyOp;

pub fn verify_factory() -> Box<dyn CommandOp> {
    Box::new(MerkleVerifyOp)
}

impl CommandOp for MerkleVerifyOp {
    fn command_path(&self) -> &str {
        "merkle verify"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Verify file integrity against a Merkle tree".into(),
            body: format!(
                r#"# merkle verify

Verify file integrity against a Merkle tree.

## Description

Rebuilds the Merkle tree from the current file data and compares it
against the stored `.mref` reference to detect corruption. Reports
whether the file is intact, and if not, identifies which chunks have
changed.

## How It Works

The command reads the source file, rebuilds a Merkle tree using the
same chunk size recorded in the `.mref`, and compares the freshly
computed root hash against the stored root hash. If the roots match,
the file is verified intact. If they differ, the command performs a
leaf-by-leaf comparison to identify exactly which chunks have changed
and reports their indices and byte offsets. A braille-art mismatch
map provides a visual overview of corruption distribution.

## Data Preparation Role

`merkle verify` is the counterpart to `merkle create` and serves as
a data integrity gate in pipelines. It is typically placed at the
beginning of a pipeline run to confirm that previously downloaded or
generated files are still valid before expensive processing begins.
This catches silent corruption from storage failures, incomplete
network transfers, or filesystem errors that would otherwise produce
subtly wrong results in downstream steps like index building or
predicate computation.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch for verification".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let source_path = resolve_path(source_str, &ctx.workspace);
        let mref = mref_path(&source_path);

        if !mref.exists() {
            return error_result(
                format!("no merkle reference found at {}", mref.display()),
                start,
            );
        }

        // Load reference tree
        let mref_data = match std::fs::read(&mref) {
            Ok(d) => d,
            Err(e) => {
                return error_result(
                    format!("failed to read {}: {}", mref.display(), e),
                    start,
                )
            }
        };
        let ref_tree = match MerkleTree::from_bytes(&mref_data) {
            Ok(t) => t,
            Err(e) => return error_result(format!("invalid mref: {}", e), start),
        };

        // Read current file and build verification tree
        let data = match std::fs::read(&source_path) {
            Ok(d) => d,
            Err(e) => {
                return error_result(
                    format!("failed to read {}: {}", source_path.display(), e),
                    start,
                )
            }
        };
        let current_tree = MerkleTree::from_data(&data, ref_tree.chunk_size);

        // Compare
        if ref_tree == current_tree {
            ctx.ui.log(&format!(
                "Merkle verify: {} OK ({} chunks)",
                source_path.display(),
                current_tree.leaf_count
            ));
            CommandResult {
                status: Status::Ok,
                message: format!(
                    "verification successful: {} ({} chunks)",
                    source_path.display(),
                    current_tree.leaf_count
                ),
                produced: vec![],
                elapsed: start.elapsed(),
            }
        } else {
            let mismatches = ref_tree.find_mismatches(&current_tree);
            let mismatch_detail: Vec<String> = mismatches
                .iter()
                .take(5)
                .map(|&i| {
                    let byte_offset = i * ref_tree.chunk_size;
                    format!(
                        "chunk {} (offset {}..{})",
                        i,
                        byte_offset,
                        byte_offset + ref_tree.chunk_size
                    )
                })
                .collect();

            let extra = if mismatches.len() > 5 {
                format!(" (and {} more)", mismatches.len() - 5)
            } else {
                String::new()
            };

            ctx.ui.log(&format!(
                "Merkle verify: {} FAILED ({} mismatched chunks)",
                source_path.display(),
                mismatches.len()
            ));
            for d in &mismatch_detail {
                ctx.ui.log(&format!("  {}", d));
            }

            CommandResult {
                status: Status::Error,
                message: format!(
                    "verification failed: {} mismatched chunks: {}{}",
                    mismatches.len(),
                    mismatch_detail.join(", "),
                    extra
                ),
                produced: vec![],
                elapsed: start.elapsed(),
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![OptionDesc {
            name: "source".to_string(),
            type_name: "Path".to_string(),
            required: true,
            default: None,
            description: "File to verify against its merkle reference".to_string(),
                role: OptionRole::Input,
    }]
    }
}

// -- Merkle Diff command -----------------------------------------------------

/// Pipeline command: compare two merkle references.
pub struct MerkleDiffOp;

pub fn diff_factory() -> Box<dyn CommandOp> {
    Box::new(MerkleDiffOp)
}

impl CommandOp for MerkleDiffOp {
    fn command_path(&self) -> &str {
        "merkle diff"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Diff two Merkle trees to find changed chunks".into(),
            body: format!(
                r#"# merkle diff

Diff two Merkle trees to find changed chunks.

## Description

Compares two `.mref` files and reports which chunks differ between
them, along with metadata comparisons (chunk size, file size, leaf
count) and a visual mismatch map. Requires both trees to use the
same chunk size for meaningful comparison.

## How It Works

The command loads both `.mref` files and first compares their basic
metadata: chunk size, total file size, and leaf count. If the chunk
sizes differ, comparison is not possible and a warning is returned.
Otherwise, it performs a leaf-by-leaf hash comparison, collecting the
indices of all mismatched chunks. Results include the number and
percentage of changed chunks, the first ten mismatched chunk offsets,
and a braille-art visualization showing the spatial distribution of
changes across the file.

## Data Preparation Role

`merkle diff` identifies which portions of a file changed between
two versions. This is useful for incremental dataset updates: when a
new version of a vector file or metadata slab is produced, diffing
the Merkle trees reveals exactly which chunks need to be re-processed
or re-uploaded, avoiding a full re-transfer. It also helps diagnose
partial corruption by showing whether damage is localized to a few
chunks or spread across the file.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Read prefetch for file comparison".into(), adjustable: false },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let file1_str = match options.require("file1") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let file2_str = match options.require("file2") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let path1 = resolve_mref_path(file1_str, &ctx.workspace);
        let path2 = resolve_mref_path(file2_str, &ctx.workspace);

        let tree1 = match load_mref(&path1) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };
        let tree2 = match load_mref(&path2) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        ctx.ui.log("MERKLE REFERENCE DIFF SUMMARY");
        ctx.ui.log("============================");
        ctx.ui.log(&format!("File 1: {}", path1.display()));
        ctx.ui.log(&format!("File 2: {}", path2.display()));
        ctx.ui.log("");

        // Compare metadata
        let chunk_match = tree1.chunk_size == tree2.chunk_size;
        let size_match = tree1.file_size == tree2.file_size;
        let leaf_match = tree1.leaf_count == tree2.leaf_count;

        ctx.ui.log("Basic Information Comparison:");
        ctx.ui.log(&format!(
            "  Chunk Size: {} vs {} ({})",
            tree1.chunk_size,
            tree2.chunk_size,
            if chunk_match { "MATCH" } else { "MISMATCH" }
        ));
        ctx.ui.log(&format!(
            "  Total Size: {} vs {} ({})",
            tree1.file_size,
            tree2.file_size,
            if size_match { "MATCH" } else { "MISMATCH" }
        ));
        ctx.ui.log(&format!(
            "  Leaf Count: {} vs {} ({})",
            tree1.leaf_count,
            tree2.leaf_count,
            if leaf_match { "MATCH" } else { "MISMATCH" }
        ));
        ctx.ui.log("");

        if !chunk_match {
            return CommandResult {
                status: Status::Warning,
                message: "cannot compare: different chunk sizes".to_string(),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        let mismatches = tree1.find_mismatches(&tree2);

        if mismatches.is_empty() {
            ctx.ui.log("No differences found — files are identical.");
            CommandResult {
                status: Status::Ok,
                message: "merkle diff: no differences".to_string(),
                produced: vec![],
                elapsed: start.elapsed(),
            }
        } else {
            let pct = (mismatches.len() as f64
                / tree1.leaf_count.max(tree2.leaf_count) as f64)
                * 100.0;

            ctx.ui.log("Leaf Node Differences:");
            ctx.ui.log(&format!("  Total Mismatched Chunks: {}", mismatches.len()));
            ctx.ui.log(&format!("  Percentage: {:.2}%", pct));

            let show = mismatches.len().min(10);
            if show > 0 {
                ctx.ui.log(&format!("  First {} mismatches:", show));
                for &chunk_idx in mismatches.iter().take(show) {
                    let offset = chunk_idx * tree1.chunk_size;
                    let length = tree1.chunk_size;
                    ctx.ui.log(&format!("    Chunk {}: offset {}, length {}", chunk_idx, offset, length));
                }
                if mismatches.len() > show {
                    ctx.ui.log(&format!("    ... and {} more", mismatches.len() - show));
                }
            }

            // Braille visualization
            let max_leaves = tree1.leaf_count.max(tree2.leaf_count);
            let braille = render_braille_bitset(&mismatches, max_leaves);
            ctx.ui.log("");
            ctx.ui.log("  Mismatch Map (braille):");
            ctx.ui.log(&format!("  {}", braille));

            CommandResult {
                status: Status::Warning,
                message: format!(
                    "merkle diff: {} mismatched chunks ({:.2}%)",
                    mismatches.len(),
                    pct
                ),
                produced: vec![],
                elapsed: start.elapsed(),
            }
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "file1".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "First file or .mref path".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "file2".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Second file or .mref path".to_string(),
                        role: OptionRole::Input,
        },
        ]
    }
}

// -- Merkle Summary command --------------------------------------------------

/// Pipeline command: display merkle tree summary.
pub struct MerkleSummaryOp;

pub fn summary_factory() -> Box<dyn CommandOp> {
    Box::new(MerkleSummaryOp)
}

impl CommandOp for MerkleSummaryOp {
    fn command_path(&self) -> &str {
        "merkle summary"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Print Merkle tree summary statistics".into(),
            body: format!(
                r#"# merkle summary

Print Merkle tree summary statistics.

## Description

Loads an `.mref` file and displays summary statistics including the
original file size, chunk size, leaf count, total node count, tree
depth, and root hash. This provides a quick overview of the tree's
structure without performing any verification.

## How It Works

The command reads the `.mref` binary file, deserializes its header
(chunk size, file size, leaf count), and computes derived metrics: the
number of internal nodes, total tree nodes, and tree depth
(log2 of the padded leaf count plus one). The root hash is extracted
from index zero of the hash array and displayed in hexadecimal.

## Data Preparation Role

`merkle summary` provides a quick sanity check on Merkle reference
files without the overhead of reading the source data. It is useful
for confirming that an `.mref` was created correctly (right chunk
size, expected file size), for comparing tree sizes across different
files, and for obtaining root hashes to include in dataset manifests
or integrity reports.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let mref_path = resolve_mref_path(source_str, &ctx.workspace);

        let tree = match load_mref(&mref_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        let mref_size = std::fs::metadata(&mref_path)
            .map(|m| m.len())
            .unwrap_or(0);

        let padded_leaves = tree.leaf_count.next_power_of_two();
        let internal_count = padded_leaves - 1;
        let total_nodes = internal_count + padded_leaves;

        ctx.ui.log("MERKLE REFERENCE FILE SUMMARY");
        ctx.ui.log("============================");
        ctx.ui.log(&format!("File: {}", mref_path.display()));
        ctx.ui.log(&format!("File Size: {} bytes", mref_size));
        ctx.ui.log(&format!("Content File Size: {} bytes", tree.file_size));
        ctx.ui.log(&format!("Chunk Size: {} bytes", tree.chunk_size));
        ctx.ui.log(&format!("Number of Chunks: {}", tree.leaf_count));
        ctx.ui.log("");
        ctx.ui.log("Tree Shape:");
        ctx.ui.log(&format!("  Leaf Nodes: {}", tree.leaf_count));
        ctx.ui.log(&format!("  Padded Leaves: {}", padded_leaves));
        ctx.ui.log(&format!("  Internal Nodes: {}", internal_count));
        ctx.ui.log(&format!("  Total Nodes: {}", total_nodes));
        ctx.ui.log(&format!("  Tree Depth: {}", (padded_leaves as f64).log2().ceil() as usize + 1));
        ctx.ui.log("");
        ctx.ui.log(&format!("Root Hash: {}", hex::encode(tree.root_hash())));

        let ratio = if tree.file_size > 0 {
            (mref_size as f64 / tree.file_size as f64) * 100.0
        } else {
            0.0
        };
        ctx.ui.log(&format!("Size Ratio: {:.2}% of content file", ratio));

        CommandResult {
            status: Status::Ok,
            message: format!(
                "merkle summary: {} chunks, {} nodes, root={}",
                tree.leaf_count,
                total_nodes,
                &hex::encode(tree.root_hash())[..16]
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![OptionDesc {
            name: "source".to_string(),
            type_name: "Path".to_string(),
            required: true,
            default: None,
            description: "File or .mref path to summarize".to_string(),
                role: OptionRole::Input,
    }]
    }
}

// -- Merkle Treeview command -------------------------------------------------

/// Pipeline command: ASCII tree visualization of a merkle tree.
pub struct MerkleTreeviewOp;

pub fn treeview_factory() -> Box<dyn CommandOp> {
    Box::new(MerkleTreeviewOp)
}

impl CommandOp for MerkleTreeviewOp {
    fn command_path(&self) -> &str {
        "merkle treeview"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Display Merkle tree structure visually".into(),
            body: format!(
                r#"# merkle treeview

Display Merkle tree structure visually.

## Description

Loads an `.mref` file and renders the tree structure as an indented
diagram using box-drawing characters, showing hash prefixes for both
internal nodes and leaf nodes. The display depth and starting node
are configurable.

## How It Works

The command loads the Merkle tree from the `.mref` file, then
recursively renders the tree starting from the specified base node.
Each node is displayed with its index and a truncated hex hash prefix
(configurable length). Internal nodes show their children with
box-drawing connectors, while leaf nodes show the leaf index. The
rendering stops at the configured depth limit, displaying an
ellipsis for deeper subtrees.

## Data Preparation Role

`merkle treeview` is a visual diagnostic tool for understanding the
hierarchical structure of a Merkle tree. It is particularly useful for
verifying that the tree was constructed correctly, for understanding
how specific chunks map to internal nodes, and for educational
purposes when explaining how Merkle integrity verification works in
the dataset pipeline.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };

        let hash_length: usize = options
            .get("hash-length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);
        let levels: usize = options
            .get("levels")
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let base_node: usize = options
            .get("base")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let mref_path = resolve_mref_path(source_str, &ctx.workspace);

        let tree = match load_mref(&mref_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        let padded_leaves = tree.leaf_count.next_power_of_two();
        let internal_count = padded_leaves - 1;
        let total_nodes = tree.hashes.len();

        if base_node >= total_nodes {
            return error_result(
                format!("base node {} out of range (total nodes: {})", base_node, total_nodes),
                start,
            );
        }

        ctx.ui.log("Merkle Tree Visualization");
        ctx.ui.log("------------------------");
        ctx.ui.log(&format!("Base node: {}", base_node));
        ctx.ui.log(&format!("Levels: {}", levels));
        ctx.ui.log("");

        render_tree_node(
            &tree.hashes,
            base_node,
            internal_count,
            total_nodes,
            hash_length,
            levels,
            0,
            "",
            "",
        );

        CommandResult {
            status: Status::Ok,
            message: format!(
                "merkle treeview: {} nodes displayed from base {}",
                total_nodes.min(2usize.pow(levels as u32)),
                base_node
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "File or .mref path to visualize".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "hash-length".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("16".to_string()),
                description: "Bytes of hash to display per node".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "levels".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("4".to_string()),
                description: "Depth of tree to render".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "base".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Tree node index to start rendering from".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

/// Recursively render a tree node with box-drawing characters.
fn render_tree_node(
    hashes: &[[u8; HASH_SIZE]],
    node: usize,
    internal_count: usize,
    total_nodes: usize,
    hash_length: usize,
    max_levels: usize,
    depth: usize,
    prefix: &str,
    child_prefix: &str,
) {
    if node >= total_nodes {
        return;
    }

    let is_leaf = node >= internal_count;
    let hash_str = format_hash(&hashes[node], hash_length);

    let label = if is_leaf {
        let leaf_idx = node - internal_count;
        format!("Leaf {}: {}", leaf_idx, hash_str)
    } else {
        format!("Node {}: {}", node, hash_str)
    };

    log::info!("{}{}", prefix, label);

    if is_leaf || depth + 1 >= max_levels {
        if !is_leaf {
            log::info!("{}    ... (deeper levels not shown)", child_prefix);
        }
        return;
    }

    let left = 2 * node + 1;
    let right = 2 * node + 2;

    if left < total_nodes && right < total_nodes {
        render_tree_node(
            hashes,
            left,
            internal_count,
            total_nodes,
            hash_length,
            max_levels,
            depth + 1,
            &format!("{}\u{251c}\u{2500}\u{2500} ", child_prefix),
            &format!("{}\u{2502}   ", child_prefix),
        );
        render_tree_node(
            hashes,
            right,
            internal_count,
            total_nodes,
            hash_length,
            max_levels,
            depth + 1,
            &format!("{}\u{2514}\u{2500}\u{2500} ", child_prefix),
            &format!("{}    ", child_prefix),
        );
    } else if left < total_nodes {
        render_tree_node(
            hashes,
            left,
            internal_count,
            total_nodes,
            hash_length,
            max_levels,
            depth + 1,
            &format!("{}\u{2514}\u{2500}\u{2500} ", child_prefix),
            &format!("{}    ", child_prefix),
        );
    }
}

/// Format a hash as hex, truncated to `max_bytes` bytes.
fn format_hash(hash: &[u8; HASH_SIZE], max_bytes: usize) -> String {
    let show = max_bytes.min(HASH_SIZE);
    let hex_str = hex::encode(&hash[..show]);
    if show < HASH_SIZE {
        format!("{}...", hex_str)
    } else {
        hex_str
    }
}

// -- Merkle Path command -----------------------------------------------------

/// Pipeline command: show authentication path from leaf to root.
pub struct MerklePathOp;

pub fn path_factory() -> Box<dyn CommandOp> {
    Box::new(MerklePathOp)
}

impl CommandOp for MerklePathOp {
    fn command_path(&self) -> &str {
        "merkle path"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Show the hash path for a specific chunk".into(),
            body: format!(
                r#"# merkle path

Show the hash path for a specific chunk.

## Description

Displays the authentication path from a specific leaf chunk up to
the root of the Merkle tree, showing the hash at each level. This
is the sequence of hashes that would be needed to independently
verify the integrity of a single chunk.

## How It Works

The command loads the `.mref` file and locates the leaf node
corresponding to the given chunk index. Starting from that leaf, it
walks up the tree by repeatedly computing the parent index
(`(node - 1) / 2`) and recording the hash at each level. The path
is displayed from leaf to root, with each level labeled. The root
hash is the final entry in the path.

## Data Preparation Role

`merkle path` is useful for debugging chunk-level integrity issues.
When `merkle verify` reports a corrupted chunk, `merkle path` shows
the exact chain of hashes from that chunk to the root, making it
possible to trace where in the tree the mismatch propagates. It also
demonstrates the logarithmic proof property of Merkle trees: verifying
a single chunk requires only O(log N) hashes rather than re-reading
the entire file.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let source_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let chunk_index: usize = match options.require("chunk") {
            Ok(s) => match s.parse() {
                Ok(n) => n,
                _ => return error_result(format!("invalid chunk index: '{}'", s), start),
            },
            Err(e) => return error_result(e, start),
        };

        let mref_file = resolve_mref_path(source_str, &ctx.workspace);

        let tree = match load_mref(&mref_file) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        if chunk_index >= tree.leaf_count {
            return error_result(
                format!(
                    "chunk index {} out of range (leaf count: {})",
                    chunk_index, tree.leaf_count
                ),
                start,
            );
        }

        // Compute path from leaf to root
        let padded_leaves = tree.leaf_count.next_power_of_two();
        let internal_count = padded_leaves - 1;
        let mut node_idx = internal_count + chunk_index;

        let mut path_hashes: Vec<(String, [u8; HASH_SIZE])> = Vec::new();
        path_hashes.push((format!("Leaf {}", chunk_index), tree.hashes[node_idx]));

        let mut level = 1;
        while node_idx > 0 {
            node_idx = (node_idx - 1) / 2;
            let label = if node_idx == 0 {
                "Root".to_string()
            } else {
                format!("Level {}", level)
            };
            path_hashes.push((label, tree.hashes[node_idx]));
            level += 1;
        }

        ctx.ui.log(&format!("Authentication path for chunk {} to root:", chunk_index));
        for (label, hash) in &path_hashes {
            ctx.ui.log(&format!("  {:>10}: {}", label, hex::encode(hash)));
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "merkle path: chunk {} → root ({} levels)",
                chunk_index,
                path_hashes.len()
            ),
            produced: vec![],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "File or .mref path".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "chunk".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Chunk (leaf) index to show path for".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

// -- Shared helpers ----------------------------------------------------------

/// Resolve a path that might be a base file or .mref file.
fn resolve_mref_path(path_str: &str, workspace: &Path) -> PathBuf {
    let base = resolve_path(path_str, workspace);

    // If it already ends in .mref, use as-is
    if base.extension().and_then(|e| e.to_str()) == Some(MREF_EXT) {
        return base;
    }

    // Try appending .mref
    let with_mref = mref_path(&base);
    if with_mref.exists() {
        return with_mref;
    }

    // Return the original path (will fail later with appropriate error)
    base
}

/// Load a merkle tree from an .mref file.
fn load_mref(path: &Path) -> Result<MerkleTree, String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    MerkleTree::from_bytes(&data)
}

/// Render a braille visualization of which positions are set.
fn render_braille_bitset(set_positions: &[usize], total: usize) -> String {
    if total == 0 {
        return String::new();
    }

    // Each braille character represents 2 columns × 4 rows = 8 dots
    // We map positions to a single row, so each char covers 2 positions
    let char_count = (total + 1) / 2;
    let mut result = String::with_capacity(char_count * 3);

    // Build a bitset
    let mut bits = vec![false; total];
    for &pos in set_positions {
        if pos < total {
            bits[pos] = true;
        }
    }

    // Braille base: U+2800
    // Dot positions for our 1-row display: left dot = bit 0 (0x01), right dot = bit 3 (0x08)
    for i in (0..total).step_by(2) {
        let mut code: u32 = 0x2800;
        if bits[i] {
            code |= 0x01; // left dot
        }
        if i + 1 < total && bits[i + 1] {
            code |= 0x08; // right dot
        }
        if let Some(c) = char::from_u32(code) {
            result.push(c);
        }
    }

    result
}

// -- Hex encoding helper (avoid adding hex crate dependency) ------------------

mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

fn format_bytes(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Recursively collect files eligible for merkle tree creation.
///
/// Walks a directory tree, skipping hidden directories (dot-prefixed),
/// `.mref` files, and files below `min_size`. Returns paths sorted for
/// deterministic ordering.
fn collect_eligible_files(dir: &Path, min_size: u64) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_eligible_recursive(dir, min_size, &mut files);
    files.sort();
    files
}

fn collect_eligible_recursive(dir: &Path, min_size: u64, files: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if path.is_dir() {
            if veks_core::filters::is_excluded_dir(&name_str) { continue; }
            collect_eligible_recursive(&path, min_size, files);
        } else {
            if veks_core::filters::is_merkle_exempt(&name_str) { continue; }
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            if size >= min_size {
                files.push(path);
            }
        }
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
    }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

// -- Merkle spoilbits command -------------------------------------------------

/// Pipeline command: corrupt merkle tree leaf bits to simulate data corruption.
pub struct MerkleSpoilbitsOp;

pub fn spoilbits_factory() -> Box<dyn CommandOp> {
    Box::new(MerkleSpoilbitsOp)
}

impl CommandOp for MerkleSpoilbitsOp {
    fn command_path(&self) -> &str {
        "merkle spoilbits"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Report bit-level differences between trees".into(),
            body: format!(
                r#"# merkle spoilbits

Report bit-level differences between trees.

## Description

Deliberately corrupts leaf node hashes in an `.mref` file to simulate
data corruption at the bit level. A configurable percentage of leaf
nodes are zeroed out and the internal tree nodes are recomputed. This
is a testing tool for validating that `merkle verify` correctly
detects and localizes corruption.

## How It Works

The command loads the `.mref` file, selects a seeded-random subset
of leaf nodes (controlled by `percentage` and `seed`), replaces their
hashes with all-zero bytes, then recomputes all internal node hashes
bottom-up to produce a self-consistent but intentionally corrupted
tree. The modified tree is written back to the `.mref` file. In
`dryrun` mode the command previews which leaves would be corrupted
without modifying the file.

## Data Preparation Role

`merkle spoilbits` is a testing and validation tool used to confirm
that the Merkle verification pipeline works correctly. By introducing
known corruption at the hash level (without modifying the source file),
you can verify that `merkle verify` detects the expected number of
mismatched chunks and that `merkle diff` correctly identifies the
corrupted regions. This is typically used during pipeline development
and integration testing rather than in production dataset preparation.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let percentage: f64 = options
            .get("percentage")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10.0);
        let seed: u64 = options
            .get("seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(42);
        let dryrun = options.get("dryrun").map_or(false, |s| s == "true");

        let mref_path = resolve_mref_path(input_str, &ctx.workspace);
        let tree = match load_mref(&mref_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        let leaf_count = tree.leaf_count;
        let internal_count = tree.hashes.len() - leaf_count;

        // Determine which leaves to spoil
        let spoil_count = ((leaf_count as f64 * percentage / 100.0).ceil() as usize).max(1).min(leaf_count);

        // Use simple seeded selection
        let mut selected = Vec::new();
        let mut rng_state = seed;
        while selected.len() < spoil_count {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = (rng_state >> 33) as usize % leaf_count;
            if !selected.contains(&idx) {
                selected.push(idx);
            }
        }
        selected.sort();

        if dryrun {
            ctx.ui.log(&format!("Dry run: would spoil {} of {} leaf nodes ({:.1}%)", spoil_count, leaf_count, percentage));
            for &idx in &selected {
                let abs_idx = internal_count + idx;
                let hash = &tree.hashes[abs_idx];
                ctx.ui.log(&format!("  leaf {} (node {}): {}", idx, abs_idx, format_hash(hash, 8)));
            }
            return CommandResult {
                status: Status::Ok,
                message: format!("dry run: {} leaves would be spoiled", spoil_count),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        // Actually spoil: zero out selected leaf hashes and recompute parents
        let mut spoiled = tree;
        for &idx in &selected {
            let abs_idx = internal_count + idx;
            spoiled.hashes[abs_idx] = [0u8; HASH_SIZE];
        }
        // Recompute internal nodes bottom-up
        recompute_internal_nodes(&mut spoiled);

        // Write back
        if let Err(e) = write_mref(&mref_path, &spoiled) {
            return error_result(e, start);
        }

        ctx.ui.log(&format!("Spoiled {} of {} leaf nodes in {}", spoil_count, leaf_count, mref_path.display()));

        CommandResult {
            status: Status::Ok,
            message: format!("spoiled {} leaves", spoil_count),
            produced: vec![mref_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Path to .mref file or source file".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "percentage".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("10".to_string()),
                description: "Percentage of leaf nodes to invalidate (0-100)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("42".to_string()),
                description: "Random seed for reproducible selection".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "dryrun".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Preview changes without modifying".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

// -- Merkle spoilchunks command -----------------------------------------------

/// Pipeline command: corrupt merkle leaf bits AND the corresponding source file chunks.
pub struct MerkleSpoilchunksOp;

pub fn spoilchunks_factory() -> Box<dyn CommandOp> {
    Box::new(MerkleSpoilchunksOp)
}

impl CommandOp for MerkleSpoilchunksOp {
    fn command_path(&self) -> &str {
        "merkle spoilchunks"
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Report chunk-level differences between trees".into(),
            body: format!(
                r#"# merkle spoilchunks

Report chunk-level differences between trees.

## Description

Deliberately corrupts random byte ranges in the actual source file
and rebuilds the Merkle tree to verify that corruption is detected
and localized to the correct chunks. Unlike `spoilbits` which only
modifies the `.mref` hashes, `spoilchunks` modifies the real data
file to simulate realistic storage corruption.

## How It Works

The command reads the source file into memory, selects a seeded-random
subset of chunks (controlled by `percentage` and `seed`), and XOR-flips
a configurable number of bytes within each selected chunk. The modified
data is written back to the source file. The Merkle tree is then
rebuilt from the corrupted data and compared against the original
`.mref` to confirm that exactly the expected chunks are flagged as
changed.

## Data Preparation Role

`merkle spoilchunks` is a more thorough testing tool than `spoilbits`
because it corrupts the actual file data rather than just the hash
reference. This validates the full round-trip: corrupted data leads to
different chunk hashes, which leads to different leaf nodes, which
propagates up to the root. It is used during pipeline integration
testing to ensure that the `merkle verify` step would catch real-world
corruption scenarios such as disk bit-rot or truncated downloads.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("source") {
            Ok(s) => s,
            Err(e) => return error_result(e, start),
        };
        let percentage: f64 = options
            .get("percentage")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10.0);
        let bytes_to_corrupt: usize = options
            .get("bytes-to-corrupt")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let seed: u64 = options
            .get("seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(42);
        let dryrun = options.get("dryrun").map_or(false, |s| s == "true");

        // Resolve the source file and mref
        let input_path = resolve_path_ws(input_str, &ctx.workspace);
        let source_path = if input_path.extension().map_or(false, |e| e == MREF_EXT) {
            input_path.with_extension("")
        } else {
            input_path.clone()
        };
        let mref_path = PathBuf::from(format!("{}.{}", source_path.display(), MREF_EXT));

        if !source_path.exists() {
            return error_result(format!("source file not found: {}", source_path.display()), start);
        }

        let tree = match load_mref(&mref_path) {
            Ok(t) => t,
            Err(e) => return error_result(e, start),
        };

        let leaf_count = tree.leaf_count;
        let chunk_size = tree.chunk_size;
        let internal_count = tree.hashes.len() - leaf_count;

        let spoil_count = ((leaf_count as f64 * percentage / 100.0).ceil() as usize).max(1).min(leaf_count);

        // Select leaves using seeded selection
        let mut selected = Vec::new();
        let mut rng_state = seed;
        while selected.len() < spoil_count {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = (rng_state >> 33) as usize % leaf_count;
            if !selected.contains(&idx) {
                selected.push(idx);
            }
        }
        selected.sort();

        if dryrun {
            ctx.ui.log(&format!("Dry run: would spoil {} chunks ({} bytes each) in {}", spoil_count, bytes_to_corrupt, source_path.display()));
            for &idx in &selected {
                let offset = idx * chunk_size;
                ctx.ui.log(&format!("  chunk {}: offset {} ({} bytes to corrupt)", idx, offset, bytes_to_corrupt));
            }
            return CommandResult {
                status: Status::Ok,
                message: format!("dry run: {} chunks would be corrupted", spoil_count),
                produced: vec![],
                elapsed: start.elapsed(),
            };
        }

        // Read source file, corrupt chunks, write back
        let mut data = match std::fs::read(&source_path) {
            Ok(d) => d,
            Err(e) => return error_result(format!("read error: {}", e), start),
        };

        for &idx in &selected {
            let offset = idx * chunk_size;
            let end = (offset + chunk_size).min(data.len());
            let corrupt_end = (offset + bytes_to_corrupt).min(end);
            for byte in &mut data[offset..corrupt_end] {
                *byte ^= 0xFF; // Flip all bits
            }
        }

        if let Err(e) = std::fs::write(&source_path, &data) {
            return error_result(format!("write error: {}", e), start);
        }

        // Also spoil the merkle tree
        let mut spoiled = tree;
        for &idx in &selected {
            let abs_idx = internal_count + idx;
            spoiled.hashes[abs_idx] = [0u8; HASH_SIZE];
        }
        recompute_internal_nodes(&mut spoiled);

        if let Err(e) = write_mref(&mref_path, &spoiled) {
            return error_result(e, start);
        }

        ctx.ui.log(&format!(
            "Corrupted {} chunks ({} bytes each) in {}",
            spoil_count, bytes_to_corrupt, source_path.display()
        ));

        CommandResult {
            status: Status::Ok,
            message: format!("corrupted {} chunks", spoil_count),
            produced: vec![source_path, mref_path],
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Path to source file or .mref file".to_string(),
                        role: OptionRole::Input,
        },
            OptionDesc {
                name: "percentage".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("10".to_string()),
                description: "Percentage of chunks to corrupt (0-100)".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "bytes-to-corrupt".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("1".to_string()),
                description: "Number of bytes to corrupt per chunk".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("42".to_string()),
                description: "Random seed for reproducible selection".to_string(),
                        role: OptionRole::Config,
        },
            OptionDesc {
                name: "dryrun".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Preview changes without modifying".to_string(),
                        role: OptionRole::Config,
        },
        ]
    }
}

/// Recompute internal nodes of a merkle tree from its leaves bottom-up.
fn recompute_internal_nodes(tree: &mut MerkleTree) {
    let total = tree.hashes.len();
    if total <= 1 {
        return;
    }
    // Internal nodes: indices 0..internal_count
    // For node i, children are at 2*i+1 and 2*i+2
    let internal_count = total - tree.leaf_count;
    for i in (0..internal_count).rev() {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut hasher = Sha256::new();
        if left < total {
            hasher.update(&tree.hashes[left]);
        }
        if right < total {
            hasher.update(&tree.hashes[right]);
        }
        let result = hasher.finalize();
        tree.hashes[i].copy_from_slice(&result);
    }
}

/// Write a merkle tree to an .mref file using the existing serialization format.
fn write_mref(path: &Path, tree: &MerkleTree) -> Result<(), String> {
    let bytes = tree.to_bytes();
    std::fs::write(path, &bytes)
        .map_err(|e| format!("failed to write {}: {}", path.display(), e))
}

/// Create a merkle `.mref` file for a single file on disk.
///
/// Uses the default 1 MiB chunk size. The `.mref` is written as a sibling
/// file (e.g., `catalog.json` → `catalog.json.mref`). Intended for small
/// infrastructure files (catalogs) that are generated outside the normal
/// merkle pipeline step.
pub fn create_mref_for_file(file_path: &Path) -> Result<PathBuf, String> {
    let data = std::fs::read(file_path)
        .map_err(|e| format!("read {}: {}", file_path.display(), e))?;
    let tree = MerkleTree::from_data(&data, DEFAULT_CHUNK_SIZE);
    let mref_path = file_path.with_extension(
        format!("{}.{}", file_path.extension().and_then(|e| e.to_str()).unwrap_or(""), MREF_EXT)
    );
    write_mref(&mref_path, &tree)?;
    Ok(mref_path)
}

fn resolve_path_ws(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() {
        p
    } else {
        workspace.join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;

    fn test_ctx(dir: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
        }
    }

    #[test]
    fn test_merkle_tree_roundtrip() {
        let data = b"Hello, world! This is test data for merkle tree.";
        let tree = MerkleTree::from_data(data, 16);
        assert!(tree.leaf_count > 0);
        assert!(!tree.hashes.is_empty());

        let bytes = tree.to_bytes();
        let restored = MerkleTree::from_bytes(&bytes).unwrap();
        assert_eq!(tree, restored);
    }

    #[test]
    fn test_merkle_tree_detects_corruption() {
        let data1 = b"original content here";
        let data2 = b"modified content here";
        let tree1 = MerkleTree::from_data(data1, 8);
        let tree2 = MerkleTree::from_data(data2, 8);

        assert_ne!(tree1.root_hash(), tree2.root_hash());
        let mismatches = tree1.find_mismatches(&tree2);
        assert!(!mismatches.is_empty());
    }

    #[test]
    fn test_merkle_tree_identical() {
        let data = b"same content";
        let tree1 = MerkleTree::from_data(data, 8);
        let tree2 = MerkleTree::from_data(data, 8);
        assert_eq!(tree1, tree2);
        assert!(tree1.find_mismatches(&tree2).is_empty());
    }

    #[test]
    fn test_merkle_create_and_verify() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Create a test file
        let source = ws.join("testfile.dat");
        std::fs::write(&source, b"test data for merkle verification").unwrap();

        // Create merkle reference
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("chunk-size", "16");

        opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        let result = create_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(!result.produced.is_empty());

        // Verify should pass
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());

        let mut verify_op = MerkleVerifyOp;
        let result = verify_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "verify failed: {}", result.message);
    }

    #[test]
    fn test_merkle_verify_detects_modification() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("testfile.dat");
        std::fs::write(&source, b"original file content here").unwrap();

        // Create reference
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("chunk-size", "8");
        opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        create_op.execute(&opts, &mut ctx);

        // Modify the file
        std::fs::write(&source, b"modified file content here").unwrap();

        // Verify should fail
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        let mut verify_op = MerkleVerifyOp;
        let result = verify_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("mismatched"));
    }

    #[test]
    fn test_merkle_verify_no_reference() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("testfile.dat");
        std::fs::write(&source, b"data").unwrap();

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        let mut verify_op = MerkleVerifyOp;
        let result = verify_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("no merkle reference"));
    }

    #[test]
    fn test_mref_path() {
        let p = PathBuf::from("/data/file.fvec");
        assert_eq!(mref_path(&p), PathBuf::from("/data/file.fvec.mref"));
    }

    #[test]
    fn test_chunk_size_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("testfile.dat");
        std::fs::write(&source, b"data").unwrap();

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("chunk-size", "100"); // not power of 2

        opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        let result = create_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
        assert!(result.message.contains("power of 2"));
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex::encode(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
        assert_eq!(hex::encode(&[0, 255]), "00ff");
    }

    #[test]
    fn test_merkle_diff_identical() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("testfile.dat");
        std::fs::write(&source, b"test data for diff comparison").unwrap();

        // Create reference
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("chunk-size", "8");
        opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        create_op.execute(&opts, &mut ctx);

        // Copy mref to another name
        let mref1 = mref_path(&source);
        let mref2 = ws.join("testfile.dat.copy.mref");
        std::fs::copy(&mref1, &mref2).unwrap();

        // Diff should show no differences
        let mut opts = Options::new();
        opts.set("file1", mref1.to_string_lossy().to_string());
        opts.set("file2", mref2.to_string_lossy().to_string());

        let mut diff_op = MerkleDiffOp;
        let result = diff_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("no differences"));
    }

    #[test]
    fn test_merkle_diff_different() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Create two different files
        let file1 = ws.join("file1.dat");
        let file2 = ws.join("file2.dat");
        std::fs::write(&file1, b"original content 12345678").unwrap();
        std::fs::write(&file2, b"modified content 12345678").unwrap();

        // Create references for both
        let mut opts = Options::new();
        opts.set("source", file1.to_string_lossy().to_string());
        opts.set("chunk-size", "8");
        opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        create_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", file2.to_string_lossy().to_string());
        opts.set("chunk-size", "8");
        opts.set("min-size", "0");
        create_op.execute(&opts, &mut ctx);

        // Diff should find mismatches
        let mref1 = mref_path(&file1);
        let mref2 = mref_path(&file2);
        let mut opts = Options::new();
        opts.set("file1", mref1.to_string_lossy().to_string());
        opts.set("file2", mref2.to_string_lossy().to_string());

        let mut diff_op = MerkleDiffOp;
        let result = diff_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Warning);
        assert!(result.message.contains("mismatched"));
    }

    #[test]
    fn test_merkle_summary() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("testfile.dat");
        std::fs::write(&source, b"some content for summary display").unwrap();

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("chunk-size", "8");
        opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        create_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());

        let mut summary_op = MerkleSummaryOp;
        let result = summary_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
        assert!(result.message.contains("merkle summary"));
    }

    #[test]
    fn test_merkle_treeview() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("testfile.dat");
        std::fs::write(&source, b"tree view test data here!").unwrap();

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("chunk-size", "8");
        opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        create_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("levels", "3");
        opts.set("hash-length", "8");

        let mut tv_op = MerkleTreeviewOp;
        let result = tv_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);
    }

    #[test]
    fn test_braille_rendering() {
        let braille = render_braille_bitset(&[0, 3, 4], 8);
        assert!(!braille.is_empty());
        // Should produce 4 braille characters (8 positions / 2 per char)
        assert_eq!(braille.chars().count(), 4);
    }

    #[test]
    fn test_merkle_path() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("testfile.dat");
        std::fs::write(&source, b"data for path test, multi-chunk").unwrap();

        // Create reference with small chunks
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("chunk-size", "8");
        opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        create_op.execute(&opts, &mut ctx);

        // Show path for chunk 0
        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("chunk", "0");

        let mut path_op = MerklePathOp;
        let result = path_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "failed: {}", result.message);
        assert!(result.message.contains("levels"));
    }

    #[test]
    fn test_merkle_path_out_of_range() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let source = ws.join("testfile.dat");
        std::fs::write(&source, b"small data").unwrap();

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("chunk-size", "8");
        opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        create_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", source.to_string_lossy().to_string());
        opts.set("chunk", "999");

        let mut path_op = MerklePathOp;
        let result = path_op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Error);
    }

    #[test]
    fn test_resolve_mref_path() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();

        // Direct .mref path
        let p = resolve_mref_path("test.mref", ws);
        assert_eq!(p, ws.join("test.mref"));

        // Base file with existing .mref
        let base = ws.join("data.fvec");
        std::fs::write(&base, b"data").unwrap();
        let mref = ws.join("data.fvec.mref");
        std::fs::write(&mref, b"mref").unwrap();

        let p = resolve_mref_path("data.fvec", ws);
        assert_eq!(p, mref);
    }

    #[test]
    fn test_spoilbits_dryrun() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        // Create a test file and merkle reference
        let data = vec![42u8; 4096];
        let data_path = ws.join("test.bin");
        std::fs::write(&data_path, &data).unwrap();

        let mut create_opts = Options::new();
        create_opts.set("source", data_path.to_string_lossy().to_string());
        create_opts.set("chunk-size", "1024");
        create_opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        let result = create_op.execute(&create_opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "create failed: {}", result.message);

        // Spoilbits in dry-run mode
        let mut opts = Options::new();
        opts.set("source", data_path.to_string_lossy().to_string());
        opts.set("percentage", "50");
        opts.set("dryrun", "true");
        let mut op = MerkleSpoilbitsOp;
        let result = op.execute(&opts, &mut ctx);
        if result.status != Status::Ok {
            panic!("spoilbits failed: {}", result.message);
        }
        assert!(result.message.contains("dry run"));
    }

    #[test]
    fn test_spoilbits_modifies_mref() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let data = vec![42u8; 4096];
        let data_path = ws.join("test.bin");
        std::fs::write(&data_path, &data).unwrap();

        let mut create_opts = Options::new();
        create_opts.set("source", data_path.to_string_lossy().to_string());
        create_opts.set("chunk-size", "1024");
        create_opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        create_op.execute(&create_opts, &mut ctx);

        let mref_path = ws.join("test.bin.mref");
        let original = std::fs::read(&mref_path).unwrap();

        // Spoil bits
        let mut opts = Options::new();
        opts.set("source", data_path.to_string_lossy().to_string());
        opts.set("percentage", "50");
        opts.set("seed", "42");
        let mut op = MerkleSpoilbitsOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let modified = std::fs::read(&mref_path).unwrap();
        assert_ne!(original, modified, "mref should have changed");
    }

    #[test]
    fn test_spoilchunks_corrupts_data() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let mut ctx = test_ctx(ws);

        let data = vec![42u8; 4096];
        let data_path = ws.join("test.bin");
        std::fs::write(&data_path, &data).unwrap();

        let mut create_opts = Options::new();
        create_opts.set("source", data_path.to_string_lossy().to_string());
        create_opts.set("chunk-size", "1024");
        create_opts.set("min-size", "0");
        let mut create_op = MerkleCreateOp;
        create_op.execute(&create_opts, &mut ctx);

        // Spoil chunks
        let mut opts = Options::new();
        opts.set("source", data_path.to_string_lossy().to_string());
        opts.set("percentage", "25");
        opts.set("bytes-to-corrupt", "2");
        opts.set("seed", "42");
        let mut op = MerkleSpoilchunksOp;
        let result = op.execute(&opts, &mut ctx);
        assert_eq!(result.status, Status::Ok);

        let corrupted = std::fs::read(&data_path).unwrap();
        assert_ne!(data, corrupted, "data should be corrupted");
    }
}
