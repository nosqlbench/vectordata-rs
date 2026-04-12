<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 09 — Vectordata Access Layer

This document defines the requirements and implementation plan for the Rust
`vectordata` crate — the client-side access layer for hosted vector datasets.
The Rust implementation must be **user-facing compatible** with the Java
companion project (`nbdatatools` / `datatools-vectordata`) while being
**architecturally idiomatic** to Rust.

## 9.1 Scope

The `vectordata` crate provides:

- Transparent access to local and remote vector datasets
- Merkle-backed integrity verification of downloaded data
- Differential on-demand download with local caching
- Profile resolution with windowed views
- Facet discovery (manifest) including custom facets
- Wire-format compatibility with the Java companion project

## 9.2 Companion Project Reference

Java companion: `nbdatatools` at commit `56fa8a267335b8c0762a72c8bc12`.

Key Java modules:

| Java module | Rust equivalent | Status |
|-------------|-----------------|--------|
| `datatools-vectordata` (access layer) | `vectordata` crate | Done |
| `datatools-io-transport` (HTTP/file transport) | `vectordata::transport` | Done |
| `datatools-vectordata/merklev2` (integrity) | `vectordata::merkle` | Done |
| `datatools-vectordata/layoutv2` (profiles) | `vectordata::dataset` module | Done |
| `datatools-vectordata/spec` (facet descriptors) | `vectordata::dataset` + `vectordata` | Done |

## 9.3 Gap Analysis

### Current State (Rust)

| Feature | Status | Notes |
|---------|--------|-------|
| `dataset.yaml` parsing | **Done** | `vectordata::dataset` module: profiles, views, windows, aliases, custom facets |
| Mmap vector readers (fvec, ivec, mvec) | **Done** | `MmapVectorReader<T>` with zero-copy slices, madvise |
| HTTP Range vector readers | **Done** | fvec, ivec, mvec; connection pooling via reqwest |
| `TestDataGroup::load()` | **Done** | Local path and HTTP URL |
| `TestDataView` trait | **Done** | Typed accessors, facet manifest, generic facet(), distance_function() |
| Integration tests over HTTP | **Done** | `testserver` utility merged into `vectordata/tests/support/`; cached_channel, http_access, facet_access, transport tests |
| Custom facet support in schema | **Done** | `vectordata::dataset` module preserves non-standard keys |

### Missing (Required for Parity)

| Feature | Priority | Complexity | Section |
|---------|----------|------------|---------|
| Merkle tree core (wire-compatible) | **Done** | Medium | 9.5 |
| Merkle state tracking (.mrkl) | **Done** | Medium | 9.5 |
| Chunked transport with retry | **Done** | Medium | 9.6 |
| Cache-backed file channel | **Done** | High | 9.7 |
| Prebuffer (eager download) | **Done** | Low | 9.7 |
| Facet manifest + generic accessor | **Done** | Low | 9.8 |
| FacetDescriptor | **Done** | Low | 9.8 |
| Download progress tracking | **Done** | Low | 9.6 |
| Dataset metadata (distance_function, etc.) | **Done** | Low | 9.9 |
| mvec HTTP reader | **Done** | Low | 9.9 |
| Token/template expansion in source URLs | **P2** | Low | 9.9 |
| Chunk scheduling strategies | **P2** | Medium | 9.6 |

## 9.4 Wire Compatibility Requirements

The following formats **must** be byte-identical between Rust and Java
implementations. A `.mref` file produced by Java must be readable by Rust,
and vice versa. A `.mrkl` file written by Rust must be resumable by Java.

### Merkle reference file (.mref)

Binary layout:

```
[hash_data: nodeCount * 32 bytes][footer: 41 bytes]
```

Each hash is SHA-256 (32 bytes). Hashes are stored for all tree nodes:
internal nodes at indices `[0, offset)`, leaf nodes at `[offset, nodeCount)`.

### Merkle state file (.mrkl)

Binary layout:

```
[hash_data: nodeCount * 32 bytes][valid_bitset: bitSetSize bytes][footer: 41 bytes]
```

The `valid_bitset` tracks which chunks have been verified. Bits only
transition from 0 to 1 (never revert). BitSet encoding must match
`java.util.BitSet` serialization: little-endian long array where bit N
maps to `words[N / 64] & (1L << (N % 64))`.

### Merkle footer (41 bytes, big-endian)

```
Offset  Size  Field               Description
 0       8    chunk_size          Bytes per chunk (except possibly last)
 8       8    total_content_size  Total content bytes protected by the tree
16       4    total_chunks        Number of leaf chunks
20       4    leaf_count          Number of leaf nodes in tree (>= total_chunks)
24       4    cap_leaf            Capacity at leaf level (power of 2)
28       4    node_count          Total nodes (internal + leaf)
32       4    offset              Index where leaf nodes begin
36       4    internal_node_count Number of internal nodes
40       1    footer_length       Always 41
```

All multi-byte integers are big-endian (matching Java's `DataOutputStream`).

### Chunk hashing

Each chunk's hash is computed as `SHA-256(content_bytes)` over the raw
content bytes for that chunk. The last chunk may be shorter than
`chunk_size`. Internal node hashes are `SHA-256(left_child_hash ||
right_child_hash)`.

## 9.5 Merkle Core

### Design

Rust-idiomatic implementation in `vectordata::merkle` (or a dedicated
`merkle` crate if the module grows substantial).

```rust
/// Read-only reference merkle tree. Loaded from .mref files.
pub struct MerkleRef {
    shape: MerkleShape,
    hashes: Vec<[u8; 32]>,  // indexed by node position
}

/// Mutable verification state. Loaded from / persisted to .mrkl files.
pub struct MerkleState {
    shape: MerkleShape,
    hashes: Vec<[u8; 32]>,
    valid: BitVec,           // chunk validity tracking
}

/// Computed tree geometry — derived from footer fields.
pub struct MerkleShape {
    chunk_size: u64,
    total_content_size: u64,
    total_chunks: u32,
    leaf_count: u32,
    cap_leaf: u32,
    node_count: u32,
    offset: u32,
    internal_node_count: u32,
}
```

### MerkleRef operations

| Method | Description |
|--------|-------------|
| `load(path) -> Result<Self>` | Parse .mref file with footer validation |
| `leaf_hash(chunk_index) -> &[u8; 32]` | Hash for a leaf chunk |
| `path_to_root(leaf_index) -> Vec<&[u8; 32]>` | Merkle proof path |
| `verify_chunk(index, data) -> bool` | SHA-256 data, compare to leaf hash |
| `shape() -> &MerkleShape` | Tree geometry |

### MerkleState operations

| Method | Description |
|--------|-------------|
| `from_ref(mref) -> Self` | Initialize state from reference (all invalid) |
| `load(path) -> Result<Self>` | Load existing state |
| `save(&self, path) -> Result<()>` | Persist to .mrkl |
| `is_valid(chunk_index) -> bool` | Check if chunk is verified |
| `mark_valid(chunk_index)` | Set chunk as verified (monotonic) |
| `valid_count() -> u32` | Number of verified chunks |
| `is_complete() -> bool` | All chunks verified |
| `save_if_valid(index, data, sink) -> bool` | Verify + mark + write in one step |

### Implementation notes

- Use `sha2` crate (already in workspace via veks)
- `BitVec` from `bitvec` crate, or hand-roll a `Vec<u64>` matching Java's
  BitSet layout for wire compatibility
- Footer parsing/writing: `byteorder` crate with BigEndian
- All file I/O uses explicit seeking (no buffered streams) to match Java's
  absolute-position semantics

## 9.6 Chunked Transport

### Transport trait

```rust
/// Byte-range data fetcher — abstracts HTTP vs local file access.
pub trait ChunkedTransport: Send + Sync {
    /// Fetch bytes in range [start, start+len) from the resource.
    fn fetch_range(&self, start: u64, len: u64) -> Result<Vec<u8>>;

    /// Total size of the resource in bytes.
    fn content_length(&self) -> Result<u64>;

    /// Whether the server supports Range requests.
    fn supports_range(&self) -> bool;
}
```

### HTTP transport

```rust
pub struct HttpTransport {
    client: reqwest::blocking::Client,  // shared, connection-pooled
    url: Url,
    content_length: OnceCell<u64>,
    supports_range: OnceCell<bool>,
}
```

- Connection pooling via shared `Client` (reqwest already does this)
- Size detection: HEAD request, fallback to Range probe
- Range support detection: check Accept-Ranges header or 206 response

### Retry policy

```rust
pub struct RetryPolicy {
    max_retries: u32,       // default: 10
    base_delay_ms: u64,     // default: 1000
    max_delay_ms: u64,      // default: 30000
    jitter_fraction: f64,   // default: 0.10
}
```

Exponential backoff: `delay = min(base * 2^attempt, max) * (1 + random(0, jitter))`.

### Parallel downloads

- Use `rayon` or `std::thread` pool for blocking downloads
- Configurable concurrency (default: 8 concurrent fetches)
- Shared failure flag: one chunk failure short-circuits remaining tasks
- `DownloadProgress` struct with atomic counters:

```rust
pub struct DownloadProgress {
    total_bytes: u64,
    downloaded_bytes: AtomicU64,
    total_chunks: u32,
    completed_chunks: AtomicU32,
    failed: AtomicBool,
}
```

### Chunk scheduling

Initial implementation: default scheduler (download requested chunks only).
Adaptive/aggressive strategies deferred to P2.

## 9.7 Cache-Backed File Channel

The central abstraction that ties merkle verification to transparent access.

### Design

```rust
/// A file channel that transparently downloads, verifies, and caches
/// data from a remote source using merkle tree integrity checking.
pub struct CachedChannel {
    /// Local cache file (random access read/write)
    cache: File,
    /// Merkle verification state
    state: MerkleState,
    /// Remote data source
    transport: Box<dyn ChunkedTransport>,
    /// Reference tree for hash verification
    reference: MerkleRef,
    /// Download progress
    progress: Arc<DownloadProgress>,
}
```

### Read path

```
read(offset, len) →
  1. Determine which chunks overlap [offset, offset+len)
  2. For each chunk:
     a. If state.is_valid(chunk) → read from cache file
     b. Else:
        i.   transport.fetch_range(chunk_start, chunk_size)
        ii.  reference.verify_chunk(chunk_index, data)
        iii. If valid: write to cache, state.mark_valid(chunk_index)
        iv.  If invalid: error (data corruption)
  3. Assemble requested byte range from chunk data
  4. Return bytes
```

### Dataset Metadata Caching

When `RemoteDatasetView::open` is called, `dataset.yaml` is fetched
and cached to the local cache directory **before** any vector data is
downloaded. This ensures the metadata is available locally for tools
that inspect cache directories.

Freshness is maintained using HTTP `Last-Modified` timestamps:

1. On first access: GET `dataset.yaml`, write to cache, set local
   mtime to the remote `Last-Modified` value.
2. On subsequent access: HEAD request to check remote `Last-Modified`.
   If the remote timestamp is newer than local mtime, re-download.
   Otherwise skip (no data transferred).

This approach avoids re-downloading on every open while detecting
remote changes (e.g., after re-publishing with new profiles).

### Initialization

```
CachedChannel::open(remote_url, cache_dir) →
  1. Fetch .mref from remote (e.g., base_vectors.fvec.mref)
  2. Check for existing .mrkl in cache_dir
     a. If .mrkl exists + cache file exists → resume (load state)
     b. If neither exists → cold start (create state from ref, create cache file)
     c. Otherwise → error (inconsistent state)
  3. Return CachedChannel
```

### Prebuffer

```rust
impl CachedChannel {
    /// Eagerly download and verify all unverified chunks.
    pub fn prebuffer(&self) -> Result<()>;

    /// Eagerly download with progress callback.
    pub fn prebuffer_with_progress<F: Fn(&DownloadProgress)>(
        &self, callback: F
    ) -> Result<()>;
}
```

### State persistence

- `MerkleState::save()` called after each successful chunk verification
- Crash recovery: on restart, load `.mrkl` and continue from last checkpoint
- `.mrkl` file is the source of truth for what's been verified

## 9.8 Facet Manifest and Generic Access

### FacetDescriptor

```rust
/// Describes a single facet declared in a dataset profile.
pub struct FacetDescriptor {
    /// Facet name as declared in dataset.yaml
    pub name: String,
    /// Source file path
    pub source_path: Option<String>,
    /// Source format type (e.g., "fvec", "slab", "json")
    pub source_type: Option<String>,
    /// Matching StandardFacet if this is a standard facet
    pub standard_kind: Option<StandardFacet>,
}

impl FacetDescriptor {
    /// Returns true if this is a recognized standard facet.
    pub fn is_standard(&self) -> bool {
        self.standard_kind.is_some()
    }
}
```

### TestDataView extensions

```rust
pub trait TestDataView: Send + Sync {
    // --- existing typed accessors ---
    fn base_vectors(&self) -> Result<Arc<dyn VectorReader<f32>>>;
    fn query_vectors(&self) -> Result<Arc<dyn VectorReader<f32>>>;
    fn neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>>;
    fn neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>>;
    fn filtered_neighbor_indices(&self) -> Result<Arc<dyn VectorReader<i32>>>;
    fn filtered_neighbor_distances(&self) -> Result<Arc<dyn VectorReader<f32>>>;

    // --- facet discovery (new) ---

    /// Returns descriptors for all facets in the profile, without
    /// materializing data. Includes both standard and custom facets.
    fn facet_manifest(&self) -> HashMap<String, FacetDescriptor>;

    /// Materializes and returns the reader for any named facet.
    /// For standard facets, delegates to the typed accessor.
    /// For custom facets, this is the only access path.
    fn facet(&self, name: &str) -> Result<Arc<dyn VectorReader<f32>>>;

    // --- metadata accessors (new) ---

    fn metadata_content(&self) -> Option<&FacetConfig>;
    fn metadata_predicates(&self) -> Option<&FacetConfig>;
    fn predicate_results(&self) -> Option<&FacetConfig>;
    fn metadata_layout(&self) -> Option<&FacetConfig>;

    // --- dataset metadata (new) ---

    fn distance_function(&self) -> Option<String>;

    // --- download management (new) ---

    /// Eagerly download all facets to local cache.
    fn prebuffer(&self) -> Result<()>;
}
```

## 9.9 Additional Features

### mvec HTTP reader

Add `HttpVectorReader<half::f16>::open_mvec(url)` following the same
pattern as the existing fvec/ivec readers. Entry size = `4 + dim * 2`.

### Dataset attributes

`TestDataGroup` already exposes `attribute(name)`. The view layer should
surface common attributes (distance_function, dimension) as typed accessors
derived from the `attributes` map in `dataset.yaml`.

### Token expansion

Source paths in `DSView` may contain `${token}` references. The Java
project supports tokens for dataset name, version, distance function, model,
and vendor. The Rust implementation should support the same expansion using
values from the `attributes` map.

Priority: P2 — only needed when datasets use templated source URLs.

## 9.10 Implementation Phases

### Phase 1: Merkle core (wire-compatible)

**Status: Complete.** Implemented in `vectordata::merkle` — `MerkleRef`, `MerkleState`, `MerkleShape` with full .mref/.mrkl wire compatibility.

**Goal:** Read `.mref` files, write/read `.mrkl` state files, verify chunks.

**Deliverables:**
- `MerkleShape`, `MerkleRef`, `MerkleState` structs
- Footer parsing/writing (41-byte big-endian format)
- SHA-256 chunk verification
- BitSet with Java-compatible wire format
- Unit tests + cross-validation against Java-produced `.mref` files

**Crate location:** `vectordata::merkle` module

### Phase 2: Chunked transport with retry

**Status: Complete.** Implemented in `vectordata::transport` — `HttpTransport`, `RetryPolicy`, `DownloadProgress`. Integration tests use the `testserver` utility in `vectordata/tests/support/`.

**Goal:** Fetch byte ranges from HTTP with retry, connection pooling, progress.

**Deliverables:**
- `ChunkedTransport` trait
- `HttpTransport` implementation
- `RetryPolicy` with exponential backoff + jitter
- `DownloadProgress` with atomic counters
- Tests using `testserver` utility in `vectordata/tests/support/`

**Crate location:** `vectordata::transport` module

### Phase 3: Cache-backed file channel

**Status: Complete.** Implemented in `vectordata::cache` — `CachedChannel` with on-demand fetch, prebuffer, crash recovery via .mrkl checkpoint. Full integration test suite in `cached_channel.rs`.

**Goal:** Transparent read-through caching with merkle verification.

**Deliverables:**
- `CachedChannel` struct
- Read path: check state → fetch if needed → verify → cache → return
- `prebuffer()` for eager download
- State persistence and crash recovery
- Integration tests: start server, download via CachedChannel, kill and
  resume, verify integrity

**Crate location:** `vectordata::cache` module

### Phase 4: Facet manifest + view enrichment

**Status: Complete.** `FacetDescriptor`, `facet_manifest()`, `facet()`, `distance_function()` on `TestDataView`. mvec HTTP reader via `HttpVectorReader<half::f16>::open_mvec()`. Integration tests in `facet_access.rs`.

**Goal:** Generic facet discovery and access, dataset metadata.

**Deliverables:**
- `FacetDescriptor` struct
- `facet_manifest()` and `facet()` on `TestDataView`
- `distance_function()` accessor
- mvec HTTP reader
- Update `GenericTestDataView` to use `CachedChannel` for remote access

**Crate location:** `vectordata::view` + `vectordata::io`

## 9.11 Testing Strategy

### Unit tests

- Merkle footer round-trip (write → read → compare)
- Merkle shape computation from parameters
- Chunk verification (correct hash passes, corrupted data fails)
- BitSet wire format matches Java encoding

### Cross-validation tests

- Load `.mref` files produced by Java tooling, verify all fields parse correctly
- Produce `.mrkl` files in Rust, load in Java (manual or scripted)
- Byte-for-byte comparison of footer encoding

### Integration tests (using `testserver` utility)

- Full download lifecycle: cold start → partial download → crash → resume → complete
- Prebuffer: download all chunks, verify all marked valid
- Corrupted chunk detection: server serves wrong data, verify error
- Progress tracking: verify counters advance correctly
- Concurrent reads triggering parallel downloads

### Compatibility tests

- Round-trip: Java writes .mref → Rust reads → Rust writes .mrkl → Java resumes
- Identical hash computation: same content produces same hashes in both
