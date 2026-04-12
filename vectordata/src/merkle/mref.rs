// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Read-only merkle reference tree loaded from `.mref` files.

use std::fs;
use std::io::{self, Cursor};
use std::path::Path;

use super::{FOOTER_SIZE, FOOTER_SIZE_V2, HASH_SIZE, MerkleShape, read_hashes, sha256, write_hashes};

/// Read-only reference merkle tree. Loaded from `.mref` files produced by
/// either the Java or Rust tooling.
///
/// The `.mref` file layout is:
/// ```text
/// [hash_data: nodeCount * 32 bytes][footer: 41 bytes]
/// ```
#[derive(Debug, Clone)]
pub struct MerkleRef {
    shape: MerkleShape,
    hashes: Vec<[u8; 32]>,
}

impl MerkleRef {
    /// Construct a `MerkleRef` directly from pre-computed shape and hashes.
    ///
    /// Used by `MerkleState::to_ref()` to extract a reference from the
    /// dual-mode `.mrkl` file without needing a separate `.mref` file.
    pub fn from_parts(shape: MerkleShape, hashes: Vec<[u8; 32]>) -> Self {
        MerkleRef { shape, hashes }
    }

    /// Load a `.mref` file from disk.
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Parse a `.mref` from a byte buffer.
    ///
    /// Layout: `[hashes: node_count * 32 bytes][footer: 41 or 45 bytes]`
    ///
    /// The last byte is a footer-length marker (41 for v1, 45 for v2).
    /// This format is byte-compatible with the Java `MerkleTree` serialization.
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < FOOTER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("mref too short: {} bytes (minimum {})", data.len(), FOOTER_SIZE),
            ));
        }

        let actual_footer_size = data[data.len() - 1] as usize;
        if actual_footer_size != FOOTER_SIZE && actual_footer_size != super::FOOTER_SIZE_V2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid footer length marker: expected {} or {}, got {} — \
                     this file may need to be regenerated with the current merkle format",
                    FOOTER_SIZE, super::FOOTER_SIZE_V2, actual_footer_size
                ),
            ));
        }
        if data.len() < actual_footer_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("mref too short for footer: {} bytes, footer says {}", data.len(), actual_footer_size),
            ));
        }

        let footer_start = data.len() - actual_footer_size;
        let shape = MerkleShape::read_footer(&data[footer_start..])?;

        let expected_hash_bytes = shape.node_count as usize * HASH_SIZE;

        if footer_start < expected_hash_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "hash data too short: expected {} bytes ({} nodes), got {} before footer",
                    expected_hash_bytes, shape.node_count, footer_start
                ),
            ));
        }

        let mut cursor = Cursor::new(&data[..expected_hash_bytes]);
        let hashes = read_hashes(&mut cursor, shape.node_count)?;

        Ok(MerkleRef { shape, hashes })
    }

    /// Build a reference tree from raw content by computing all hashes.
    ///
    /// This is used to generate `.mref` files from content data.
    pub fn from_content(content: &[u8], chunk_size: u64) -> Self {
        let shape = MerkleShape::for_content(chunk_size, content.len() as u64);
        let mut hashes = vec![[0u8; 32]; shape.node_count as usize];

        // Compute leaf hashes
        for i in 0..shape.total_chunks {
            let start = shape.chunk_start(i) as usize;
            let len = shape.chunk_len(i) as usize;
            let chunk_data = &content[start..start + len];
            hashes[shape.leaf_node_index(i) as usize] = sha256(chunk_data);
        }

        // Unused leaf slots (padding) get hash of empty = all zeros left as-is.
        // The Java implementation hashes them as SHA-256(empty), so match that.
        let empty_hash = sha256(b"");
        for i in shape.total_chunks..shape.leaf_count {
            hashes[shape.leaf_node_index(i) as usize] = empty_hash;
        }

        // Compute internal node hashes bottom-up
        if shape.internal_node_count > 0 {
            for i in (0..shape.internal_node_count).rev() {
                let left = shape.left_child(i) as usize;
                let right = shape.right_child(i) as usize;
                hashes[i as usize] = super::sha256_pair(&hashes[left], &hashes[right]);
            }
        }

        MerkleRef { shape, hashes }
    }

    /// Write this reference tree as a `.mref` file.
    pub fn write<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        write_hashes(w, &self.hashes)?;
        self.shape.write_footer(w)?;
        Ok(())
    }

    /// Save to a file path.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut buf = Vec::with_capacity(self.hashes.len() * HASH_SIZE + FOOTER_SIZE_V2);
        self.write(&mut buf)?;
        fs::write(path, &buf)
    }

    /// Serialize to bytes in the standard `.mref` format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.hashes.len() * HASH_SIZE + FOOTER_SIZE_V2);
        self.write(&mut buf).expect("write to Vec cannot fail");
        buf
    }

    /// Tree geometry.
    pub fn shape(&self) -> &MerkleShape {
        &self.shape
    }

    /// All node hashes.
    pub fn hashes(&self) -> &[[u8; 32]] {
        &self.hashes
    }

    /// Hash for a specific node.
    pub fn node_hash(&self, node_index: u32) -> &[u8; 32] {
        &self.hashes[node_index as usize]
    }

    /// Hash for a leaf chunk.
    pub fn leaf_hash(&self, chunk_index: u32) -> &[u8; 32] {
        &self.hashes[self.shape.leaf_node_index(chunk_index) as usize]
    }

    /// Root hash (node 0).
    pub fn root_hash(&self) -> &[u8; 32] {
        &self.hashes[0]
    }

    /// Verify a chunk's data against its expected hash.
    pub fn verify_chunk(&self, chunk_index: u32, data: &[u8]) -> bool {
        let computed = sha256(data);
        computed == *self.leaf_hash(chunk_index)
    }

    /// Return the merkle proof path from a leaf to the root.
    ///
    /// Returns the sibling hashes needed to verify the path.
    pub fn proof_path(&self, chunk_index: u32) -> Vec<[u8; 32]> {
        let mut path = Vec::new();
        let mut node = self.shape.leaf_node_index(chunk_index);
        while node > 0 {
            let parent = self.shape.parent(node);
            let sibling = if node == self.shape.left_child(parent) {
                self.shape.right_child(parent)
            } else {
                self.shape.left_child(parent)
            };
            path.push(self.hashes[sibling as usize]);
            node = parent;
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_content_single_chunk() {
        let data = b"hello world";
        let mref = MerkleRef::from_content(data, 1024);

        assert_eq!(mref.shape().total_chunks, 1);
        assert_eq!(mref.shape().node_count, 1);
        assert!(mref.verify_chunk(0, data));
        assert!(!mref.verify_chunk(0, b"wrong data"));
    }

    #[test]
    fn test_from_content_multiple_chunks() {
        let data = vec![0u8; 4096];
        let mref = MerkleRef::from_content(&data, 1024);

        assert_eq!(mref.shape().total_chunks, 4);
        assert_eq!(mref.shape().node_count, 7);

        // Verify each chunk
        for i in 0..4 {
            let start = (i * 1024) as usize;
            assert!(mref.verify_chunk(i, &data[start..start + 1024]));
        }
    }

    #[test]
    fn test_save_load_round_trip() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mref = MerkleRef::from_content(data, 16);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.mref");
        mref.save(&path).unwrap();

        let loaded = MerkleRef::load(&path).unwrap();
        assert_eq!(mref.shape(), loaded.shape());
        assert_eq!(mref.root_hash(), loaded.root_hash());

        // Verify chunks still work after reload
        assert!(loaded.verify_chunk(0, &data[0..16]));
        assert!(loaded.verify_chunk(1, &data[16..32]));
    }

    #[test]
    fn test_proof_path() {
        let data = vec![0u8; 4096];
        let mref = MerkleRef::from_content(&data, 1024);

        let path = mref.proof_path(0);
        // 4 leaves → depth 2 → proof has 2 siblings
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_corrupted_chunk_detected() {
        let mut data = vec![0u8; 4096];
        let mref = MerkleRef::from_content(&data, 1024);

        // Corrupt one byte
        data[100] = 0xFF;
        assert!(!mref.verify_chunk(0, &data[0..1024]));
        // Other chunks still valid
        assert!(mref.verify_chunk(1, &data[1024..2048]));
    }
}
