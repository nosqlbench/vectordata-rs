// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Mutable merkle verification state loaded from / persisted to `.mrkl` files.

use std::fs;
use std::io::{self, Cursor};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{FOOTER_SIZE, FOOTER_SIZE_V2, HASH_SIZE, MerkleRef, MerkleShape, read_hashes, sha256, write_hashes};

/// Mutable verification state for a merkle-protected file.
///
/// Tracks which chunks have been downloaded and verified. The validity bitset
/// uses Java `BitSet`-compatible encoding: a little-endian `u64` array where
/// bit N maps to `words[N / 64] & (1 << (N % 64))`.
///
/// `valid_words` is stored as `Vec<AtomicU64>` so [`mark_valid`] is
/// lock-free — workers in `CachedChannel::parallel_fetch_verify_write`
/// can update the bitmap concurrently without taking a shared mutex.
/// All reads use `Relaxed`/`Acquire` ordering as appropriate.
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
    /// matching `java.util.BitSet`. Workers update via atomic `fetch_or`;
    /// readers snapshot via `load(Relaxed)`.
    valid_words: Vec<AtomicU64>,
}

impl Clone for MerkleState {
    fn clone(&self) -> Self {
        let valid_words = self.valid_words.iter()
            .map(|w| AtomicU64::new(w.load(Ordering::Relaxed)))
            .collect();
        MerkleState {
            shape: self.shape.clone(),
            hashes: self.hashes.clone(),
            valid_words,
        }
    }
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

        let mut valid_words = Vec::with_capacity(word_count);
        for _ in 0..word_count { valid_words.push(AtomicU64::new(0)); }
        MerkleState {
            shape,
            hashes,
            valid_words,
        }
    }

    /// Number of u64 words needed for a given leaf count.
    fn word_count_for_leaves(leaf_count: u32) -> usize {
        (leaf_count as usize).div_ceil(64)
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
            valid_words.push(AtomicU64::new(word));
        }

        Ok(MerkleState {
            shape,
            hashes,
            valid_words,
        })
    }

    /// Persist the validity bitset into an existing `.mrkl`, merged
    /// with the bits already on disk, without rewriting the tree.
    ///
    /// The hashes ahead of the bitset never change after the file is
    /// created, so a checkpoint writes only the bitset (one bit per
    /// chunk): 49 KB instead of 32 MB on a 410 GB shard. Bits are
    /// monotone (missing → verified) and several channels may share
    /// one state file, so the words on disk are OR-ed into memory
    /// before the union is written back; an exclusive `flock` makes
    /// that read-merge-write atomic against other checkpointers.
    /// Falls back to a full [`MerkleState::save`] when the file is
    /// absent or does not have this tree's layout.
    ///
    /// This runs after every fetch call on a cached channel, so its
    /// cost must not scale with the tree; the cache tests guard that.
    pub fn checkpoint(&self, path: &Path) -> io::Result<()> {
        use std::io::{Read, Seek, SeekFrom, Write};
        let hash_bytes = self.hashes.len() * HASH_SIZE;
        let bitset_bytes = self.bitset_byte_size();
        let mut file = match fs::OpenOptions::new().read(true).write(true).open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return self.save(path),
            Err(e) => return Err(e),
        };
        let len = file.metadata()?.len() as usize;
        let layout_matches = [FOOTER_SIZE, FOOTER_SIZE_V2]
            .into_iter()
            .any(|footer| len == hash_bytes + bitset_bytes + footer);
        if !layout_matches {
            return self.save(path);
        }
        let _lock = FileLock::exclusive(&file)?;
        let mut on_disk = vec![0u8; bitset_bytes];
        file.seek(SeekFrom::Start(hash_bytes as u64))?;
        file.read_exact(&mut on_disk)?;
        let mut merged = Vec::with_capacity(bitset_bytes);
        for (i, word) in self.valid_words.iter().enumerate() {
            let mut w = [0u8; 8];
            w.copy_from_slice(&on_disk[i * 8..i * 8 + 8]);
            let union = word.fetch_or(u64::from_le_bytes(w), Ordering::AcqRel)
                | u64::from_le_bytes(w);
            merged.extend_from_slice(&union.to_le_bytes());
        }
        file.seek(SeekFrom::Start(hash_bytes as u64))?;
        file.write_all(&merged)?;
        Ok(())
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

        // Write bitset as little-endian u64 words. Each atomic is
        // loaded with `Relaxed` — concurrent `mark_valid` calls
        // may flip bits 0 → 1 during the iteration, but the
        // serialization is well-defined for any consistent snapshot
        // of each word, and the `.mrkl` we write is monotone
        // (bits only go from missing to verified across runs).
        for word in &self.valid_words {
            let v = word.load(Ordering::Relaxed);
            w.write_all(&v.to_le_bytes())?;
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
        (self.valid_words[word_idx].load(Ordering::Acquire) & (1u64 << bit_idx)) != 0
    }

    /// Mark a chunk as verified (monotonic — bits only transition 0 → 1).
    ///
    /// Lock-free: workers in `CachedChannel::parallel_fetch_verify_write`
    /// call this concurrently without external synchronization. The
    /// `fetch_or` is `AcqRel`-ordered so concurrent `is_valid` /
    /// `valid_count` readers see the bit set as soon as the worker
    /// returns from this call. Out-of-range indices are silently
    /// ignored (matching the previous mutable-self contract).
    pub fn mark_valid(&self, chunk_index: u32) {
        let word_idx = chunk_index as usize / 64;
        let bit_idx = chunk_index as usize % 64;
        if word_idx < self.valid_words.len() {
            self.valid_words[word_idx]
                .fetch_or(1u64 << bit_idx, Ordering::AcqRel);
        }
    }

    /// Number of verified chunks. Sums `count_ones` across the
    /// atomic bitmap rather than walking chunk-by-chunk via
    /// `is_valid`, so it's O(words) instead of O(chunks).
    pub fn valid_count(&self) -> u32 {
        count_verified(self.shape.total_chunks, self.valid_words.len(), |i| {
            self.valid_words[i].load(Ordering::Relaxed)
        })
    }

    /// Are all chunks verified?
    pub fn is_complete(&self) -> bool {
        self.valid_count() == self.shape.total_chunks
    }

    /// Whether the `.mrkl` at `path` records every chunk as verified,
    /// answered without loading the tree.
    ///
    /// Reads the footer and the validity bitset only, never the
    /// `node_count * 32` bytes of hashes ahead of them. On a
    /// terabyte-scale shard the file is tens of megabytes and the
    /// bitset tens of kilobytes; a reader polling for another
    /// instance's completion must not pay for the tree on each poll.
    /// Gives the same answer [`MerkleState::is_complete`] would on the
    /// loaded state, and the same errors [`MerkleState::load`] would on
    /// a malformed file.
    pub fn complete_on_disk(path: &Path) -> io::Result<bool> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len < FOOTER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too short for footer",
            ));
        }
        let tail_len = len.min(FOOTER_SIZE_V2);
        let mut tail = vec![0u8; tail_len];
        file.seek(SeekFrom::Start((len - tail_len) as u64))?;
        file.read_exact(&mut tail)?;
        let footer_size = if tail[tail_len - 1] as usize == FOOTER_SIZE_V2 {
            FOOTER_SIZE_V2
        } else {
            FOOTER_SIZE
        };
        let shape = MerkleShape::read_footer(&tail[tail_len - footer_size..])?;

        let hash_bytes = shape.node_count as usize * HASH_SIZE;
        let word_count = Self::word_count_for_leaves(shape.leaf_count);
        let bitset_bytes = word_count * 8;
        let expected_size = hash_bytes + bitset_bytes + footer_size;
        if len != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("file size mismatch: expected {} bytes, got {}", expected_size, len),
            ));
        }

        let mut bitset = vec![0u8; bitset_bytes];
        file.seek(SeekFrom::Start(hash_bytes as u64))?;
        file.read_exact(&mut bitset)?;
        let word = |i: usize| {
            let mut w = [0u8; 8];
            w.copy_from_slice(&bitset[i * 8..i * 8 + 8]);
            u64::from_le_bytes(w)
        };
        Ok(count_verified(shape.total_chunks, word_count, word) == shape.total_chunks)
    }

    /// Verify chunk data and mark as valid if the hash matches.
    ///
    /// Returns `true` if the data is valid (hash matches the reference).
    pub fn verify_and_mark(&self, chunk_index: u32, data: &[u8]) -> bool {
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

/// An exclusive advisory lock on an open file, released on drop.
/// Serialises the read-merge-write of [`MerkleState::checkpoint`]
/// between processes sharing one state file. Advisory locks are
/// not available everywhere; where `flock` is missing the lock is a
/// no-op and concurrent checkpointers may each persist only their
/// own bits, which costs a re-download, never correctness.
struct FileLock {
    #[cfg(unix)]
    fd: std::os::unix::io::RawFd,
}

impl FileLock {
    /// Take the lock on `file`, which must stay open for as long as
    /// the returned guard lives.
    fn exclusive(file: &fs::File) -> io::Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = file.as_raw_fd();
            // SAFETY: flock on a valid, open descriptor; LOCK_EX blocks
            // until the lock is held.
            if unsafe { libc::flock(fd, libc::LOCK_EX) } != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(FileLock { fd })
        }
        #[cfg(not(unix))]
        {
            let _ = file;
            Ok(FileLock {})
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            // SAFETY: releasing a lock this guard took on a descriptor
            // its creator keeps open for the guard's lifetime.
            unsafe { libc::flock(self.fd, libc::LOCK_UN) };
        }
    }
}

/// Number of verified chunks among the first `total` bits of a
/// validity bitset of `words` little-endian `u64` words, read through
/// `word`. Sums `count_ones` per word rather than testing chunk by
/// chunk, so it is O(words) not O(chunks); bits past `total` in the
/// last word are masked off. Shared by the in-memory state and the
/// on-disk probe so both count the same way.
fn count_verified(total: u32, words: usize, word: impl Fn(usize) -> u64) -> u32 {
    if total == 0 {
        return 0;
    }
    let full_words = (total as usize) / 64;
    let tail_bits = (total as usize) % 64;
    let mut count: u32 = 0;
    for i in 0..full_words.min(words) {
        count = count.saturating_add(word(i).count_ones());
    }
    if tail_bits > 0 && full_words < words {
        let mask: u64 = (1u64 << tail_bits) - 1;
        count = count.saturating_add((word(full_words) & mask).count_ones());
    }
    count.min(total)
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
    fn checkpoint_merges_sibling_bits_and_writes_only_the_bitset() {
        // 4096 chunks: 262 KB of hashes, a 512-byte bitset.
        let data = vec![3u8; 4096 * 1024];
        let mref = MerkleRef::from_content(&data, 1024);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.mrkl");
        let a = MerkleState::from_ref(&mref);
        a.save(&path).unwrap();
        let b = MerkleState::from_ref(&mref);

        a.mark_valid(0);
        a.checkpoint(&path).unwrap();
        b.mark_valid(2);
        b.checkpoint(&path).unwrap();

        // Disk holds the union; `b` learned `a`'s bit while merging.
        let on_disk = MerkleState::load(&path).unwrap();
        assert!(on_disk.is_valid(0) && on_disk.is_valid(2) && !on_disk.is_valid(1));
        assert_eq!(on_disk.valid_count(), 2);
        assert!(b.is_valid(0));
        // The tree ahead of the bitset is intact.
        assert_eq!(on_disk.hashes, a.hashes);

        #[cfg(target_os = "linux")]
        {
            let wchar = || -> u64 {
                std::fs::read_to_string("/proc/thread-self/io").unwrap()
                    .lines().find_map(|l| l.strip_prefix("wchar:"))
                    .and_then(|v| v.trim().parse().ok()).unwrap()
            };
            let before = wchar();
            a.mark_valid(7);
            a.checkpoint(&path).unwrap();
            let written = wchar() - before;
            assert!(written <= 4096, "checkpoint wrote {written} bytes; it must write the bitset, not the tree");
        }

        // Without a file, or with a foreign layout, it is a full save.
        std::fs::remove_file(&path).unwrap();
        a.checkpoint(&path).unwrap();
        assert!(MerkleState::load(&path).unwrap().is_valid(7));
        std::fs::write(&path, b"not a state file").unwrap();
        a.checkpoint(&path).unwrap();
        assert!(MerkleState::load(&path).unwrap().is_valid(7));
    }

    #[test]
    fn complete_on_disk_matches_the_loaded_state() {
        // Ragged tail: 101 chunks over 100 KiB + 5 bytes.
        let data = vec![7u8; 100 * 1024 + 5];
        let mref = MerkleRef::from_content(&data, 1024);
        let state = MerkleState::from_ref(&mref);
        let total = state.shape().total_chunks;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.mrkl");

        state.save(&path).unwrap();
        assert!(!MerkleState::complete_on_disk(&path).unwrap());

        for i in 0..total - 1 {
            state.mark_valid(i);
        }
        state.save(&path).unwrap();
        assert!(!MerkleState::complete_on_disk(&path).unwrap());
        assert!(!MerkleState::load(&path).unwrap().is_complete());

        state.mark_valid(total - 1);
        state.save(&path).unwrap();
        assert!(MerkleState::complete_on_disk(&path).unwrap());
        assert!(MerkleState::load(&path).unwrap().is_complete());

        // A malformed file fails the probe the way it fails `load`.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 7]).unwrap();
        assert!(MerkleState::complete_on_disk(&path).is_err());
        assert!(MerkleState::load(&path).is_err());
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
        let state = MerkleState::from_ref(&mref);

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
        let state = MerkleState::from_ref(&mref);

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
        let state = MerkleState::from_ref(&mref);

        for i in 0..4u32 {
            let start = (i * 1024) as usize;
            assert!(state.verify_and_mark(i, &data[start..start + 1024]));
        }
        assert!(state.is_complete());
    }

    #[test]
    fn test_missing_chunks() {
        let (data, mref) = make_test_ref();
        let state = MerkleState::from_ref(&mref);

        state.verify_and_mark(0, &data[0..1024]);
        state.verify_and_mark(2, &data[2048..3072]);

        let missing = state.missing_chunks();
        assert_eq!(missing, vec![1, 3]);
    }

    #[test]
    fn test_save_load_round_trip() {
        let (data, mref) = make_test_ref();
        let state = MerkleState::from_ref(&mref);

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
        let state = MerkleState::from_ref(&mref);

        state.mark_valid(0); // bit 0 → word[0] |= 1
        state.mark_valid(2); // bit 2 → word[0] |= 4

        assert_eq!(state.valid_words[0].load(Ordering::Relaxed), 0b101); // bits 0 and 2

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
        let state = MerkleState::from_ref(&mref);

        // Simulate partial download
        state.verify_and_mark(0, &data[0..1024]);
        state.verify_and_mark(1, &data[1024..2048]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("resume.mrkl");
        state.save(&path).unwrap();

        // "Resume" — load state, continue verifying
        let resumed = MerkleState::load(&path).unwrap();
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
        let state = MerkleState::from_ref(&mref);

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

    #[test]
    fn test_corrupt_mrkl_truncated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt_trunc.mrkl");

        // Write a file that is too short to contain even a footer
        fs::write(&path, [0u8; 10]).unwrap();
        let result = MerkleState::load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_corrupt_mrkl_random_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt_rand.mrkl");

        // Write random-ish bytes that are footer-sized but invalid
        let garbage: Vec<u8> = (0..200).map(|i| (i * 37 + 13) as u8).collect();
        fs::write(&path, &garbage).unwrap();
        let result = MerkleState::load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_corrupt_mrkl_valid_footer_wrong_size() {
        // Create a valid state, save it, then truncate the file
        let data = vec![0u8; 4096];
        let mref = MerkleRef::from_content(&data, 1024);
        let state = MerkleState::from_ref(&mref);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt_trunc2.mrkl");
        state.save(&path).unwrap();

        // Read the file, chop off some bytes from the middle
        let mut file_data = fs::read(&path).unwrap();
        file_data.truncate(file_data.len() / 2);
        fs::write(&path, &file_data).unwrap();

        let result = MerkleState::load(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_chunk_mark_complete() {
        let data = vec![0xFFu8; 100];
        let mref = MerkleRef::from_content(&data, 4096);
        let state = MerkleState::from_ref(&mref);

        assert_eq!(state.shape().total_chunks, 1);
        assert!(!state.is_complete());
        assert_eq!(state.valid_count(), 0);

        state.mark_valid(0);
        assert!(state.is_valid(0));
        assert!(state.is_complete());
        assert_eq!(state.valid_count(), 1);
    }

    #[test]
    fn test_1000_chunks_all_valid() {
        let chunk_size = 64usize;
        let data = vec![0u8; chunk_size * 1000];
        let mref = MerkleRef::from_content(&data, chunk_size as u64);
        let state = MerkleState::from_ref(&mref);

        assert_eq!(state.shape().total_chunks, 1000);
        assert!(!state.is_complete());

        for i in 0..1000u32 {
            state.mark_valid(i);
        }

        assert!(state.is_complete());
        assert_eq!(state.valid_count(), 1000);
    }

    #[test]
    fn test_round_trip_large_bitset() {
        // 1000 chunks exercises multi-word bitset (16 u64 words)
        let chunk_size = 64usize;
        let data = vec![0u8; chunk_size * 1000];
        let mref = MerkleRef::from_content(&data, chunk_size as u64);
        let state = MerkleState::from_ref(&mref);

        // Mark a scattered pattern of valid chunks
        for i in (0..1000u32).step_by(3) {
            state.mark_valid(i);
        }

        let expected_count = state.valid_count();
        assert!(expected_count > 300);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large_bitset.mrkl");
        state.save(&path).unwrap();

        let loaded = MerkleState::load(&path).unwrap();
        assert_eq!(loaded.valid_count(), expected_count);

        // Verify exact bitset equality
        for i in 0..1000u32 {
            assert_eq!(
                loaded.is_valid(i),
                state.is_valid(i),
                "mismatch at chunk {}",
                i
            );
        }
    }
}
