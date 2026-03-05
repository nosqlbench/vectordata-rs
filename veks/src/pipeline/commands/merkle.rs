// Copyright (c) DataStax, Inc.
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
    CommandOp, CommandResult, OptionDesc, Options, Status, StreamContext,
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
    /// Build a merkle tree from file data.
    fn from_data(data: &[u8], chunk_size: usize) -> Self {
        let file_size = data.len() as u64;
        let leaf_count = if data.is_empty() {
            1
        } else {
            (data.len() + chunk_size - 1) / chunk_size
        };

        // Compute leaf hashes
        let mut leaf_hashes: Vec<[u8; HASH_SIZE]> = Vec::with_capacity(leaf_count);
        for i in 0..leaf_count {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, data.len());
            let chunk = if start < data.len() {
                &data[start..end]
            } else {
                &[]
            };
            let hash = Sha256::digest(chunk);
            leaf_hashes.push(hash.into());
        }

        // Pad to next power of 2 for balanced tree
        let padded_leaves = leaf_count.next_power_of_two();
        let empty_hash: [u8; HASH_SIZE] = Sha256::digest(&[]).into();
        while leaf_hashes.len() < padded_leaves {
            leaf_hashes.push(empty_hash);
        }

        // Build internal nodes bottom-up
        let internal_count = padded_leaves - 1;
        let total = internal_count + padded_leaves;
        let mut hashes = vec![[0u8; HASH_SIZE]; total];

        // Place leaves
        for (i, h) in leaf_hashes.into_iter().enumerate() {
            hashes[internal_count + i] = h;
        }

        // Compute internal nodes (right to left, bottom to top)
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
    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.chunk_size as u32).to_le_bytes());
        buf.extend_from_slice(&self.file_size.to_le_bytes());
        buf.extend_from_slice(&(self.leaf_count as u32).to_le_bytes());
        for h in &self.hashes {
            buf.extend_from_slice(h);
        }
        buf
    }

    /// Deserialize from bytes.
    fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 16 {
            return Err("mref file too small".to_string());
        }
        let chunk_size =
            u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let file_size = u64::from_le_bytes([
            data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11],
        ]);
        let leaf_count =
            u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

        let hash_data = &data[16..];
        let hash_count = hash_data.len() / HASH_SIZE;
        if hash_data.len() % HASH_SIZE != 0 {
            return Err("mref hash data not aligned to 32 bytes".to_string());
        }

        let mut hashes = Vec::with_capacity(hash_count);
        for i in 0..hash_count {
            let mut h = [0u8; HASH_SIZE];
            h.copy_from_slice(&hash_data[i * HASH_SIZE..(i + 1) * HASH_SIZE]);
            hashes.push(h);
        }

        Ok(MerkleTree {
            hashes,
            leaf_count,
            chunk_size,
            file_size,
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

        let source_path = resolve_path(source_str, &ctx.workspace);
        let mref = mref_path(&source_path);

        // Check if mref already exists and is up-to-date
        if mref.exists() && !force {
            let src_modified = std::fs::metadata(&source_path)
                .and_then(|m| m.modified())
                .ok();
            let mref_modified = std::fs::metadata(&mref)
                .and_then(|m| m.modified())
                .ok();
            if let (Some(src_t), Some(mref_t)) = (src_modified, mref_modified) {
                if mref_t > src_t {
                    return CommandResult {
                        status: Status::Ok,
                        message: format!(
                            "merkle reference up-to-date: {}",
                            mref.display()
                        ),
                        produced: vec![mref],
                        elapsed: start.elapsed(),
                    };
                }
            }
        }

        // Read source file
        let data = match std::fs::read(&source_path) {
            Ok(d) => d,
            Err(e) => {
                return error_result(
                    format!("failed to read {}: {}", source_path.display(), e),
                    start,
                )
            }
        };

        let tree = MerkleTree::from_data(&data, chunk_size);

        eprintln!(
            "Merkle create: {} ({} bytes, {} chunks, root={})",
            source_path.display(),
            data.len(),
            tree.leaf_count,
            hex::encode(tree.root_hash())
        );

        // Write mref
        let mref_bytes = tree.to_bytes();
        if let Err(e) = std::fs::write(&mref, &mref_bytes) {
            return error_result(
                format!("failed to write {}: {}", mref.display(), e),
                start,
            );
        }

        CommandResult {
            status: Status::Ok,
            message: format!(
                "created merkle reference {} ({} chunks, {} nodes)",
                mref.display(),
                tree.leaf_count,
                tree.hashes.len()
            ),
            produced: vec![mref],
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
                description: "File to create merkle reference for".to_string(),
            },
            OptionDesc {
                name: "chunk-size".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("1048576".to_string()),
                description: "Chunk size in bytes (must be power of 2)".to_string(),
            },
            OptionDesc {
                name: "force".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Overwrite existing mref even if up-to-date".to_string(),
            },
        ]
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
            eprintln!(
                "Merkle verify: {} OK ({} chunks)",
                source_path.display(),
                current_tree.leaf_count
            );
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

            eprintln!(
                "Merkle verify: {} FAILED ({} mismatched chunks)",
                source_path.display(),
                mismatches.len()
            );
            for d in &mismatch_detail {
                eprintln!("  {}", d);
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

        eprintln!("MERKLE REFERENCE DIFF SUMMARY");
        eprintln!("============================");
        eprintln!("File 1: {}", path1.display());
        eprintln!("File 2: {}", path2.display());
        eprintln!();

        // Compare metadata
        let chunk_match = tree1.chunk_size == tree2.chunk_size;
        let size_match = tree1.file_size == tree2.file_size;
        let leaf_match = tree1.leaf_count == tree2.leaf_count;

        eprintln!("Basic Information Comparison:");
        eprintln!(
            "  Chunk Size: {} vs {} ({})",
            tree1.chunk_size,
            tree2.chunk_size,
            if chunk_match { "MATCH" } else { "MISMATCH" }
        );
        eprintln!(
            "  Total Size: {} vs {} ({})",
            tree1.file_size,
            tree2.file_size,
            if size_match { "MATCH" } else { "MISMATCH" }
        );
        eprintln!(
            "  Leaf Count: {} vs {} ({})",
            tree1.leaf_count,
            tree2.leaf_count,
            if leaf_match { "MATCH" } else { "MISMATCH" }
        );
        eprintln!();

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
            eprintln!("No differences found — files are identical.");
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

            eprintln!("Leaf Node Differences:");
            eprintln!("  Total Mismatched Chunks: {}", mismatches.len());
            eprintln!("  Percentage: {:.2}%", pct);

            let show = mismatches.len().min(10);
            if show > 0 {
                eprintln!("  First {} mismatches:", show);
                for &chunk_idx in mismatches.iter().take(show) {
                    let offset = chunk_idx * tree1.chunk_size;
                    let length = tree1.chunk_size;
                    eprintln!("    Chunk {}: offset {}, length {}", chunk_idx, offset, length);
                }
                if mismatches.len() > show {
                    eprintln!("    ... and {} more", mismatches.len() - show);
                }
            }

            // Braille visualization
            let max_leaves = tree1.leaf_count.max(tree2.leaf_count);
            let braille = render_braille_bitset(&mismatches, max_leaves);
            eprintln!();
            eprintln!("  Mismatch Map (braille):");
            eprintln!("  {}", braille);

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
            },
            OptionDesc {
                name: "file2".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Second file or .mref path".to_string(),
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

        eprintln!("MERKLE REFERENCE FILE SUMMARY");
        eprintln!("============================");
        eprintln!("File: {}", mref_path.display());
        eprintln!("File Size: {} bytes", mref_size);
        eprintln!("Content File Size: {} bytes", tree.file_size);
        eprintln!("Chunk Size: {} bytes", tree.chunk_size);
        eprintln!("Number of Chunks: {}", tree.leaf_count);
        eprintln!();
        eprintln!("Tree Shape:");
        eprintln!("  Leaf Nodes: {}", tree.leaf_count);
        eprintln!("  Padded Leaves: {}", padded_leaves);
        eprintln!("  Internal Nodes: {}", internal_count);
        eprintln!("  Total Nodes: {}", total_nodes);
        eprintln!("  Tree Depth: {}", (padded_leaves as f64).log2().ceil() as usize + 1);
        eprintln!();
        eprintln!("Root Hash: {}", hex::encode(tree.root_hash()));

        let ratio = if tree.file_size > 0 {
            (mref_size as f64 / tree.file_size as f64) * 100.0
        } else {
            0.0
        };
        eprintln!("Size Ratio: {:.2}% of content file", ratio);

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

        eprintln!("Merkle Tree Visualization");
        eprintln!("------------------------");
        eprintln!("Base node: {}", base_node);
        eprintln!("Levels: {}", levels);
        eprintln!();

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
            },
            OptionDesc {
                name: "hash-length".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("16".to_string()),
                description: "Bytes of hash to display per node".to_string(),
            },
            OptionDesc {
                name: "levels".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("4".to_string()),
                description: "Depth of tree to render".to_string(),
            },
            OptionDesc {
                name: "base".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("0".to_string()),
                description: "Tree node index to start rendering from".to_string(),
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

    eprintln!("{}{}", prefix, label);

    if is_leaf || depth + 1 >= max_levels {
        if !is_leaf {
            eprintln!("{}    ... (deeper levels not shown)", child_prefix);
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

        eprintln!("Authentication path for chunk {} to root:", chunk_index);
        for (label, hash) in &path_hashes {
            eprintln!("  {:>10}: {}", label, hex::encode(hash));
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
            },
            OptionDesc {
                name: "chunk".to_string(),
                type_name: "int".to_string(),
                required: true,
                default: None,
                description: "Chunk (leaf) index to show path for".to_string(),
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

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
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
            eprintln!("Dry run: would spoil {} of {} leaf nodes ({:.1}%)", spoil_count, leaf_count, percentage);
            for &idx in &selected {
                let abs_idx = internal_count + idx;
                let hash = &tree.hashes[abs_idx];
                eprintln!("  leaf {} (node {}): {}", idx, abs_idx, format_hash(hash, 8));
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

        eprintln!("Spoiled {} of {} leaf nodes in {}", spoil_count, leaf_count, mref_path.display());

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
                name: "input".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Path to .mref file or source file".to_string(),
            },
            OptionDesc {
                name: "percentage".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("10".to_string()),
                description: "Percentage of leaf nodes to invalidate (0-100)".to_string(),
            },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("42".to_string()),
                description: "Random seed for reproducible selection".to_string(),
            },
            OptionDesc {
                name: "dryrun".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Preview changes without modifying".to_string(),
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

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let input_str = match options.require("input") {
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
            eprintln!("Dry run: would spoil {} chunks ({} bytes each) in {}", spoil_count, bytes_to_corrupt, source_path.display());
            for &idx in &selected {
                let offset = idx * chunk_size;
                eprintln!("  chunk {}: offset {} ({} bytes to corrupt)", idx, offset, bytes_to_corrupt);
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

        eprintln!(
            "Corrupted {} chunks ({} bytes each) in {}",
            spoil_count, bytes_to_corrupt, source_path.display()
        );

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
                name: "input".to_string(),
                type_name: "Path".to_string(),
                required: true,
                default: None,
                description: "Path to source file or .mref file".to_string(),
            },
            OptionDesc {
                name: "percentage".to_string(),
                type_name: "float".to_string(),
                required: false,
                default: Some("10".to_string()),
                description: "Percentage of chunks to corrupt (0-100)".to_string(),
            },
            OptionDesc {
                name: "bytes-to-corrupt".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("1".to_string()),
                description: "Number of bytes to corrupt per chunk".to_string(),
            },
            OptionDesc {
                name: "seed".to_string(),
                type_name: "int".to_string(),
                required: false,
                default: Some("42".to_string()),
                description: "Random seed for reproducible selection".to_string(),
            },
            OptionDesc {
                name: "dryrun".to_string(),
                type_name: "bool".to_string(),
                required: false,
                default: Some("false".to_string()),
                description: "Preview changes without modifying".to_string(),
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
            workspace: dir.to_path_buf(),
            scratch: dir.join(".scratch"),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
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
        let mut create_op = MerkleCreateOp;
        create_op.execute(&opts, &mut ctx);

        let mut opts = Options::new();
        opts.set("source", file2.to_string_lossy().to_string());
        opts.set("chunk-size", "8");
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
        let mut create_op = MerkleCreateOp;
        let result = create_op.execute(&create_opts, &mut ctx);
        assert_eq!(result.status, Status::Ok, "create failed: {}", result.message);

        // Spoilbits in dry-run mode
        let mut opts = Options::new();
        opts.set("input", data_path.to_string_lossy().to_string());
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
        let mut create_op = MerkleCreateOp;
        create_op.execute(&create_opts, &mut ctx);

        let mref_path = ws.join("test.bin.mref");
        let original = std::fs::read(&mref_path).unwrap();

        // Spoil bits
        let mut opts = Options::new();
        opts.set("input", data_path.to_string_lossy().to_string());
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
        let mut create_op = MerkleCreateOp;
        create_op.execute(&create_opts, &mut ctx);

        // Spoil chunks
        let mut opts = Options::new();
        opts.set("input", data_path.to_string_lossy().to_string());
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
