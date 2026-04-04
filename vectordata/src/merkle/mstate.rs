// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Mutable merkle verification state loaded from / persisted to `.mrkl` files.

use std::fs;
use std::io::{self, Cursor};
use std::path::Path;

use super::{FOOTER_SIZE, FOOTER_SIZE_V2, HASH_SIZE, MerkleRef, MerkleShape, read_hashes, sha256, write_hashes};

/// Mutable verification state for a merkle-protected file.
///
/// Tracks which chunks have been downloaded and verified. The validity bitset
/// uses Java `BitSet`-compatible encoding: a little-endian `u64` array where
/// bit N maps to `words[N / 64] & (1 << (N % 64))`.
///
/// The `.mrkl` file layout is:
/// ```text
/// [hash_data: nodeCount * 32 bytes][valid_bitset: bitSetSize bytes][footer: 41 bytes]
/// ```
#[derive(Debug)]
pub struct MerkleState {
    shape: MerkleShape,
    hashes: Vec<[u8; 32]>,
    /// Validity bits — one per leaf node. Encoded as little-endian u64 words
    /// matching `java.util.BitSet`.
    valid_words: Vec<u64>,
}

impl MerkleState {
    /// Initialize a new state from a reference tree (all chunks invalid).
    pub fn from_ref(mref: &MerkleRef) -> Self {
        let shape = mref.shape().clone();
        let mut hashes = Vec::with_capacity(shape.node_count as usize);
        for i in 0..shape.node_count {
            hashes.push(*mref.node_hash(i));
        }

        let word_count = Self::word_count_for_leaves(shape.leaf_count);

        MerkleState {
            shape,
            hashes,
            valid_words: vec![0u64; word_count],
        }
    }

    /// Number of u64 words needed for a given leaf count.
    fn word_count_for_leaves(leaf_count: u32) -> usize {
        ((leaf_count as usize) + 63) / 64
    }

    /// Bitset size in bytes (for serialization).
    fn bitset_byte_size(&self) -> usize {
        self.valid_words.len() * 8
    }

    /// Load an existing `.mrkl` state file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let data = fs::read(path)?;
        Self::from_bytes(&data)
    }

    /// Parse a `.mrkl` from a byte buffer.
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < FOOTER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too short for footer",
            ));
        }

        // Detect footer version from last byte
        let actual_footer_size = data[data.len() - 1] as usize;
        let footer_size = if actual_footer_size == FOOTER_SIZE_V2 {
            FOOTER_SIZE_V2
        } else {
            FOOTER_SIZE
        };
        let footer_start = data.len() - footer_size;
        let shape = MerkleShape::read_footer(&data[footer_start..])?;

        let hash_bytes = shape.node_count as usize * HASH_SIZE;
        let word_count = Self::word_count_for_leaves(shape.leaf_count);
        let bitset_bytes = word_count * 8;
        let expected_size = hash_bytes + bitset_bytes + footer_size;

        if data.len() != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "file size mismatch: expected {} bytes, got {}",
                    expected_size,
                    data.len()
                ),
            ));
        }

        let mut cursor = Cursor::new(&data[..hash_bytes]);
        let hashes = read_hashes(&mut cursor, shape.node_count)?;

        // Read bitset as little-endian u64 words (Java BitSet format)
        let mut valid_words = Vec::with_capacity(word_count);
        let bitset_data = &data[hash_bytes..hash_bytes + bitset_bytes];
        for i in 0..word_count {
            let offset = i * 8;
            let word = u64::from_le_bytes([
                bitset_data[offset],
                bitset_data[offset + 1],
                bitset_data[offset + 2],
                bitset_data[offset + 3],
                bitset_data[offset + 4],
                bitset_data[offset + 5],
                bitset_data[offset + 6],
                bitset_data[offset + 7],
            ]);
            valid_words.push(word);
        }

        Ok(MerkleState {
            shape,
            hashes,
            valid_words,
        })
    }

    /// Save state to a `.mrkl` file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut buf = Vec::with_capacity(
            self.hashes.len() * HASH_SIZE + self.bitset_byte_size() + FOOTER_SIZE_V2,
        );
        self.write(&mut buf)?;
        fs::write(path, &buf)
    }

    /// Write state to a writer.
    pub fn write<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        write_hashes(w, &self.hashes)?;

        // Write bitset as little-endian u64 words
        for &word in &self.valid_words {
            w.write_all(&word.to_le_bytes())?;
        }

        self.shape.write_footer_with_bitset(w, self.bitset_byte_size() as u32)?;
        Ok(())
    }

    /// Tree geometry.
    pub fn shape(&self) -> &MerkleShape {
        &self.shape
    }

    /// Check if a chunk has been verified.
    pub fn is_valid(&self, chunk_index: u32) -> bool {
        let word_idx = chunk_index as usize / 64;
        let bit_idx = chunk_index as usize % 64;
        if word_idx >= self.valid_words.len() {
            return false;
        }
        (self.valid_words[word_idx] & (1u64 << bit_idx)) != 0
    }

    /// Mark a chunk as verified (monotonic — bits only transition 0 → 1).
    pub fn mark_valid(&mut self, chunk_index: u32) {
        let word_idx = chunk_index as usize / 64;
        let bit_idx = chunk_index as usize % 64;
        if word_idx < self.valid_words.len() {
            self.valid_words[word_idx] |= 1u64 << bit_idx;
        }
    }

    /// Number of verified chunks.
    pub fn valid_count(&self) -> u32 {
        let mut count = 0u32;
        for i in 0..self.shape.total_chunks {
            if self.is_valid(i) {
                count += 1;
            }
        }
        count
    }

    /// Are all chunks verified?
    pub fn is_complete(&self) -> bool {
        self.valid_count() == self.shape.total_chunks
    }

    /// Verify chunk data and mark as valid if the hash matches.
    ///
    /// Returns `true` if the data is valid (hash matches the reference).
    pub fn verify_and_mark(&mut self, chunk_index: u32, data: &[u8]) -> bool {
        let computed = sha256(data);
        let node_idx = self.shape.leaf_node_index(chunk_index) as usize;
        if computed == self.hashes[node_idx] {
            self.mark_valid(chunk_index);
            true
        } else {
            false
        }
    }

    /// Construct a `MerkleRef` from the hashes embedded in this state.
    ///
    /// This enables the dual-mode pattern: a single `.mrkl` file serves as
    /// both the reference tree (hashes) and the mutable verification state
    /// (validity bitset), matching the Java `MerkleDataImpl` semantics.
    pub fn to_ref(&self) -> MerkleRef {
        MerkleRef::from_parts(self.shape.clone(), self.hashes.clone())
    }

    /// Indices of chunks that have not yet been verified.
    pub fn missing_chunks(&self) -> Vec<u32> {
        (0..self.shape.total_chunks)
            .filter(|&i| !self.is_valid(i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_ref() -> (Vec<u8>, MerkleRef) {
        let data = vec![0u8; 4096];
        let mref = MerkleRef::from_content(&data, 1024);
        (data, mref)
    }

    #[test]
    fn test_initial_state_all_invalid() {
        let (_, mref) = make_test_ref();
        let state = MerkleState::from_ref(&mref);

        assert_eq!(state.valid_count(), 0);
        assert!(!state.is_complete());
        for i in 0..4 {
            assert!(!state.is_valid(i));
        }
    }

    #[test]
    fn test_mark_valid() {
        let (_, mref) = make_test_ref();
        let mut state = MerkleState::from_ref(&mref);

        state.mark_valid(0);
        assert!(state.is_valid(0));
        assert!(!state.is_valid(1));
        assert_eq!(state.valid_count(), 1);

        // Marking again is idempotent
        state.mark_valid(0);
        assert_eq!(state.valid_count(), 1);
    }

    #[test]
    fn test_verify_and_mark() {
        let (data, mref) = make_test_ref();
        let mut state = MerkleState::from_ref(&mref);

        // Good data
        assert!(state.verify_and_mark(0, &data[0..1024]));
        assert!(state.is_valid(0));

        // Bad data
        let mut bad = data[1024..2048].to_vec();
        bad[0] = 0xFF;
        assert!(!state.verify_and_mark(1, &bad));
        assert!(!state.is_valid(1));
    }

    #[test]
    fn test_is_complete() {
        let (data, mref) = make_test_ref();
        let mut state = MerkleState::from_ref(&mref);

        for i in 0..4u32 {
            let start = (i * 1024) as usize;
            assert!(state.verify_and_mark(i, &data[start..start + 1024]));
        }
        assert!(state.is_complete());
    }

    #[test]
    fn test_missing_chunks() {
        let (data, mref) = make_test_ref();
        let mut state = MerkleState::from_ref(&mref);

        state.verify_and_mark(0, &data[0..1024]);
        state.verify_and_mark(2, &data[2048..3072]);

        let missing = state.missing_chunks();
        assert_eq!(missing, vec![1, 3]);
    }

    #[test]
    fn test_save_load_round_trip() {
        let (data, mref) = make_test_ref();
        let mut state = MerkleState::from_ref(&mref);

        // Verify some chunks
        state.verify_and_mark(0, &data[0..1024]);
        state.verify_and_mark(2, &data[2048..3072]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.mrkl");
        state.save(&path).unwrap();

        let loaded = MerkleState::load(&path).unwrap();
        assert_eq!(loaded.shape(), state.shape());
        assert!(loaded.is_valid(0));
        assert!(!loaded.is_valid(1));
        assert!(loaded.is_valid(2));
        assert!(!loaded.is_valid(3));
        assert_eq!(loaded.valid_count(), 2);
    }

    #[test]
    fn test_bitset_java_compatible() {
        // Java BitSet stores bits in little-endian u64 words.
        // Bit 0 is the LSB of word 0. Bit 63 is the MSB of word 0.
        // Bit 64 is the LSB of word 1.
        let (_, mref) = make_test_ref();
        let mut state = MerkleState::from_ref(&mref);

        state.mark_valid(0); // bit 0 → word[0] |= 1
        state.mark_valid(2); // bit 2 → word[0] |= 4

        assert_eq!(state.valid_words[0], 0b101); // bits 0 and 2

        // Serialize and check raw bytes
        let mut buf = Vec::new();
        state.write(&mut buf).unwrap();

        // Bitset starts after hashes (7 nodes * 32 = 224 bytes)
        let bitset_start = 7 * 32;
        let bitset_bytes = &buf[bitset_start..bitset_start + 8];
        // Little-endian u64 with value 5
        assert_eq!(bitset_bytes, &[5, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_resume_from_checkpoint() {
        let (data, mref) = make_test_ref();
        let mut state = MerkleState::from_ref(&mref);

        // Simulate partial download
        state.verify_and_mark(0, &data[0..1024]);
        state.verify_and_mark(1, &data[1024..2048]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("resume.mrkl");
        state.save(&path).unwrap();

        // "Resume" — load state, continue verifying
        let mut resumed = MerkleState::load(&path).unwrap();
        assert_eq!(resumed.valid_count(), 2);
        assert_eq!(resumed.missing_chunks(), vec![2, 3]);

        resumed.verify_and_mark(2, &data[2048..3072]);
        resumed.verify_and_mark(3, &data[3072..4096]);
        assert!(resumed.is_complete());
    }

    #[test]
    fn test_many_chunks_bitset() {
        // Test with > 64 chunks to exercise multi-word bitset
        let data = vec![0u8; 100 * 64]; // 100 chunks of 64 bytes
        let mref = MerkleRef::from_content(&data, 64);
        let mut state = MerkleState::from_ref(&mref);

        assert_eq!(state.shape().total_chunks, 100);

        // Mark chunk 65 (second word)
        state.mark_valid(65);
        assert!(state.is_valid(65));
        assert!(!state.is_valid(64));
        assert!(!state.is_valid(66));

        // Verify round-trip preserves multi-word bitset
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("many.mrkl");
        state.save(&path).unwrap();

        let loaded = MerkleState::load(&path).unwrap();
        assert!(loaded.is_valid(65));
        assert!(!loaded.is_valid(64));
    }
}
