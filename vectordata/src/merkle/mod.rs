// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Merkle tree integrity verification for chunked data.
//!
//! This module implements wire-compatible merkle tree structures matching the
//! Java companion project (`nbdatatools`). The binary formats for `.mref`
//! (reference tree) and `.mrkl` (verification state) files are byte-identical
//! between Rust and Java implementations.
//!
//! # Wire Format
//!
//! Both file types share a 41-byte big-endian footer and SHA-256 hashes stored
//! contiguously as `node_count * 32` bytes. The `.mrkl` format additionally
//! contains a validity bitset between the hashes and footer, encoded as a
//! little-endian `u64` array matching `java.util.BitSet` serialization.

mod mref;
mod mstate;

pub use mref::MerkleRef;
pub use mstate::MerkleState;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Cursor, Read, Write};

/// Footer size in bytes — fixed at 41 for wire compatibility.
/// Footer size for the original format (without bitSetSize field).
pub const FOOTER_SIZE: usize = 41;
/// Footer size for the v2 format (with bitSetSize field).
pub const FOOTER_SIZE_V2: usize = 45;

/// SHA-256 hash size.
pub const HASH_SIZE: usize = 32;

/// Computed tree geometry derived from footer fields.
///
/// All fields match the Java `MerkleTreeLayout` semantics exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MerkleShape {
    /// Bytes per chunk (except possibly the last chunk).
    pub chunk_size: u64,
    /// Total content bytes protected by the tree.
    pub total_content_size: u64,
    /// Number of leaf chunks (actual data chunks).
    pub total_chunks: u32,
    /// Number of leaf nodes in tree (>= total_chunks, padded to fill level).
    pub leaf_count: u32,
    /// Capacity at leaf level (power of 2).
    pub cap_leaf: u32,
    /// Total nodes (internal + leaf).
    pub node_count: u32,
    /// Index where leaf nodes begin in the hash array.
    pub offset: u32,
    /// Number of internal nodes.
    pub internal_node_count: u32,
}

impl MerkleShape {
    /// Compute tree shape from content parameters.
    ///
    /// This matches the Java `MerkleTreeLayout.forContent()` logic:
    /// - `cap_leaf` is the smallest power of 2 >= `total_chunks`
    /// - `leaf_count` = `cap_leaf` (tree is always a complete binary tree at leaf level)
    /// - `internal_node_count` = `cap_leaf - 1` (complete binary tree internals)
    /// - `node_count` = `internal_node_count + leaf_count`
    /// - `offset` = `internal_node_count` (leaves start right after internals)
    pub fn for_content(chunk_size: u64, total_content_size: u64) -> Self {
        let total_chunks = if total_content_size == 0 {
            0
        } else {
            ((total_content_size + chunk_size - 1) / chunk_size) as u32
        };

        let cap_leaf = if total_chunks <= 1 {
            1
        } else {
            (total_chunks as u64).next_power_of_two() as u32
        };

        let leaf_count = cap_leaf;
        let internal_node_count = cap_leaf - 1;
        let node_count = internal_node_count + leaf_count;
        let offset = internal_node_count;

        MerkleShape {
            chunk_size,
            total_content_size,
            total_chunks,
            leaf_count,
            cap_leaf,
            node_count,
            offset,
            internal_node_count,
        }
    }

    /// Byte offset of a chunk's data within the content.
    pub fn chunk_start(&self, chunk_index: u32) -> u64 {
        chunk_index as u64 * self.chunk_size
    }

    /// Byte length of a specific chunk (last chunk may be shorter).
    pub fn chunk_len(&self, chunk_index: u32) -> u64 {
        let start = self.chunk_start(chunk_index);
        let remaining = self.total_content_size.saturating_sub(start);
        remaining.min(self.chunk_size)
    }

    /// Convert a chunk index to its node index in the hash array.
    pub fn leaf_node_index(&self, chunk_index: u32) -> u32 {
        self.offset + chunk_index
    }

    /// Index of the left child of an internal node.
    pub fn left_child(&self, node: u32) -> u32 {
        2 * node + 1
    }

    /// Index of the right child of an internal node.
    pub fn right_child(&self, node: u32) -> u32 {
        2 * node + 2
    }

    /// Index of the parent of a node (0 is root, has no parent).
    pub fn parent(&self, node: u32) -> u32 {
        (node - 1) / 2
    }

    /// Read a footer from a byte slice (supports both v1 41-byte and v2 45-byte formats).
    ///
    /// Returns `(MerkleShape, bitset_size)` where `bitset_size` is the number of
    /// bytes between the hash data and the footer (0 for v1, >0 for v2).
    pub fn read_footer(data: &[u8]) -> io::Result<Self> {
        if data.len() < FOOTER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("footer too short: {} bytes", data.len()),
            ));
        }

        let mut cursor = Cursor::new(data);
        let chunk_size = cursor.read_u64::<BigEndian>()?;
        let total_content_size = cursor.read_u64::<BigEndian>()?;
        let total_chunks = cursor.read_u32::<BigEndian>()?;
        let leaf_count = cursor.read_u32::<BigEndian>()?;
        let cap_leaf = cursor.read_u32::<BigEndian>()?;
        let node_count = cursor.read_u32::<BigEndian>()?;
        let offset = cursor.read_u32::<BigEndian>()?;
        let internal_node_count = cursor.read_u32::<BigEndian>()?;

        // V2 format has an additional bitSetSize field (4 bytes) before the length marker
        let _bitset_size = if data.len() >= FOOTER_SIZE_V2 {
            cursor.read_u32::<BigEndian>()?
        } else {
            0u32
        };

        let footer_length = cursor.read_u8()?;

        if footer_length != FOOTER_SIZE as u8 && footer_length != FOOTER_SIZE_V2 as u8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid footer length marker: expected {} or {}, got {}",
                    FOOTER_SIZE, FOOTER_SIZE_V2, footer_length
                ),
            ));
        }

        Ok(MerkleShape {
            chunk_size,
            total_content_size,
            total_chunks,
            leaf_count,
            cap_leaf,
            node_count,
            offset,
            internal_node_count,
        })
    }

    /// Write the 41-byte footer.
    pub fn write_footer<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_u64::<BigEndian>(self.chunk_size)?;
        w.write_u64::<BigEndian>(self.total_content_size)?;
        w.write_u32::<BigEndian>(self.total_chunks)?;
        w.write_u32::<BigEndian>(self.leaf_count)?;
        w.write_u32::<BigEndian>(self.cap_leaf)?;
        w.write_u32::<BigEndian>(self.node_count)?;
        w.write_u32::<BigEndian>(self.offset)?;
        w.write_u32::<BigEndian>(self.internal_node_count)?;
        w.write_u8(FOOTER_SIZE as u8)?;
        Ok(())
    }
}

/// Read `node_count` hashes (each 32 bytes) from a reader.
pub(crate) fn read_hashes<R: Read>(
    reader: &mut R,
    node_count: u32,
) -> io::Result<Vec<[u8; 32]>> {
    let mut hashes = Vec::with_capacity(node_count as usize);
    for _ in 0..node_count {
        let mut hash = [0u8; 32];
        reader.read_exact(&mut hash)?;
        hashes.push(hash);
    }
    Ok(hashes)
}

/// Write hashes to a writer.
pub(crate) fn write_hashes<W: Write>(
    writer: &mut W,
    hashes: &[[u8; 32]],
) -> io::Result<()> {
    for hash in hashes {
        writer.write_all(hash)?;
    }
    Ok(())
}

/// Compute SHA-256 of a byte slice.
pub(crate) fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Compute SHA-256 of the concatenation of two hashes (for internal nodes).
pub(crate) fn sha256_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_for_content_basic() {
        let shape = MerkleShape::for_content(1024, 4096);
        assert_eq!(shape.total_chunks, 4);
        assert_eq!(shape.cap_leaf, 4);
        assert_eq!(shape.leaf_count, 4);
        assert_eq!(shape.internal_node_count, 3);
        assert_eq!(shape.node_count, 7);
        assert_eq!(shape.offset, 3);
    }

    #[test]
    fn test_shape_for_content_non_power_of_two() {
        let shape = MerkleShape::for_content(1024, 3000);
        assert_eq!(shape.total_chunks, 3);
        assert_eq!(shape.cap_leaf, 4); // next power of 2
        assert_eq!(shape.leaf_count, 4);
        assert_eq!(shape.node_count, 7);
    }

    #[test]
    fn test_shape_for_content_single_chunk() {
        let shape = MerkleShape::for_content(4096, 100);
        assert_eq!(shape.total_chunks, 1);
        assert_eq!(shape.cap_leaf, 1);
        assert_eq!(shape.leaf_count, 1);
        assert_eq!(shape.internal_node_count, 0);
        assert_eq!(shape.node_count, 1);
        assert_eq!(shape.offset, 0);
    }

    #[test]
    fn test_shape_for_content_empty() {
        let shape = MerkleShape::for_content(1024, 0);
        assert_eq!(shape.total_chunks, 0);
    }

    #[test]
    fn test_chunk_len_last_chunk_shorter() {
        let shape = MerkleShape::for_content(1024, 3000);
        assert_eq!(shape.chunk_len(0), 1024);
        assert_eq!(shape.chunk_len(1), 1024);
        assert_eq!(shape.chunk_len(2), 952); // 3000 - 2048
    }

    #[test]
    fn test_footer_round_trip() {
        let shape = MerkleShape::for_content(65536, 1_000_000);

        let mut buf = Vec::new();
        shape.write_footer(&mut buf).unwrap();
        assert_eq!(buf.len(), FOOTER_SIZE);

        let parsed = MerkleShape::read_footer(&buf).unwrap();
        assert_eq!(shape, parsed);
    }

    #[test]
    fn test_footer_big_endian() {
        let shape = MerkleShape {
            chunk_size: 0x0000_0000_0001_0000, // 65536
            total_content_size: 0x0000_0000_000F_4240, // 1_000_000
            total_chunks: 16,
            leaf_count: 16,
            cap_leaf: 16,
            node_count: 31,
            offset: 15,
            internal_node_count: 15,
        };

        let mut buf = Vec::new();
        shape.write_footer(&mut buf).unwrap();

        // Verify big-endian encoding of chunk_size (first 8 bytes)
        assert_eq!(&buf[0..8], &[0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00]);
        // Last byte is footer length
        assert_eq!(buf[40], 41);
    }

    #[test]
    fn test_sha256_known_value() {
        // SHA-256 of empty input
        let hash = sha256(b"");
        assert_eq!(
            hex::encode(hash),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_leaf_node_index() {
        let shape = MerkleShape::for_content(1024, 4096);
        assert_eq!(shape.leaf_node_index(0), 3);
        assert_eq!(shape.leaf_node_index(1), 4);
        assert_eq!(shape.leaf_node_index(2), 5);
        assert_eq!(shape.leaf_node_index(3), 6);
    }

    #[test]
    fn test_parent_child_relationships() {
        let shape = MerkleShape::for_content(1024, 4096);
        // Root is 0, children are 1 and 2
        assert_eq!(shape.left_child(0), 1);
        assert_eq!(shape.right_child(0), 2);
        // Node 1's children are 3 and 4 (leaves)
        assert_eq!(shape.left_child(1), 3);
        assert_eq!(shape.right_child(1), 4);
        // Parent of leaf 3 is node 1
        assert_eq!(shape.parent(3), 1);
        assert_eq!(shape.parent(4), 1);
        assert_eq!(shape.parent(1), 0);
    }
}
