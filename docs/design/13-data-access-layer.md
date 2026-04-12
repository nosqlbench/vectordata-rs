<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 13 — Data Access Layer

This document defines the vectordata access layer — the unified API for
reading vector data from both local files and remote catalogs with
on-demand caching and merkle-verified integrity. This is the foundation
that all user-facing tools (visualize commands, benchmark runners, data
consumers) use to access data transparently regardless of location.

---

## 13.1 Design Principles

**Location transparency.** A consumer opens a dataset by specifier
(`dataset:profile` or local path) and receives a typed view interface.
The consumer does not know or care whether the data is local, cached,
or streamed from a remote URL. The access API is identical in all cases.

**Lazy on-demand fetching.** Remote data is NOT bulk-downloaded before
use. Individual chunks are fetched on first access and cached locally.
Only the chunks that are actually read are downloaded. This enables
interactive exploration of multi-terabyte remote datasets without
waiting for full downloads.

**Merkle-verified integrity.** Every chunk downloaded from a remote
source is verified against a Merkle hash tree before being written to
the local cache. This guarantees data integrity across unreliable
networks and detects bit-rot in cached files. The merkle reference
(`.mref` file) is the authoritative source of truth.

**Incremental and resumable.** Partial downloads are valid. If a
download is interrupted, the cached chunks remain valid. Subsequent
accesses download only the missing chunks. The merkle state (`.mrkl`
file) tracks which chunks have been verified.

**Dual-path optimization.** Two access paths are available:
- **Channel-backed**: For remote/partially-cached data, reads through
  a file channel with on-demand chunk fetching.
- **Mmap-backed**: For fully-local or fully-cached data, zero-copy
  memory-mapped access for maximum throughput.
After prebuffering completes, the view is promoted from channel-backed
to mmap-backed automatically.

---

## 13.2 Access API

### 13.2.1 DatasetLoader

The primary entry point for all data access:

```rust
/// Load a dataset from a specifier.
///
/// Accepts:
/// - A local path: `/path/to/dataset/` or `/path/to/dataset.yaml`
/// - A URL: `https://example.com/dataset/`
/// - A catalog specifier: `dataset-name:profile`
///
/// Returns a ProfileSelector that provides typed views.
pub fn load(spec: &str) -> Result<ProfileSelector, Error>;
```

**Resolution order:**
1. Check if `spec` is a URL (http/https) → remote loader
2. Check if `spec` is a local path → filesystem loader
3. Parse as `dataset:profile` → catalog resolution → remote loader
4. Parse as `dataset` → catalog resolution with default profile

### 13.2.2 ProfileSelector

```rust
/// Access to named profiles within a dataset.
pub trait ProfileSelector {
    /// Get a typed view for a named profile.
    fn profile(&self, name: &str) -> Result<TestDataView, Error>;
    /// List available profile names.
    fn profile_names(&self) -> Vec<String>;
}
```

### 13.2.3 TestDataView

The primary consumer interface:

```rust
/// Composite view into a dataset profile's facets.
pub trait TestDataView {
    fn base_vectors(&self) -> Option<Box<dyn VectorView<f32>>>;
    fn query_vectors(&self) -> Option<Box<dyn VectorView<f32>>>;
    fn neighbor_indices(&self) -> Option<Box<dyn VectorView<i32>>>;
    fn neighbor_distances(&self) -> Option<Box<dyn VectorView<f32>>>;
    fn metadata_content(&self) -> Option<Box<dyn SlabView>>;
    fn filtered_neighbor_indices(&self) -> Option<Box<dyn VectorView<i32>>>;
    fn filtered_neighbor_distances(&self) -> Option<Box<dyn VectorView<f32>>>;

    /// Prebuffer all facets for this profile (async).
    fn prebuffer(&self) -> impl Future<Output = Result<(), Error>>;
}
```

### 13.2.4 VectorView

Typed, indexed access to vector data:

```rust
/// Indexed access to vectors of type T.
pub trait VectorView<T> {
    /// Number of vectors accessible (may be windowed).
    fn count(&self) -> usize;
    /// Dimensionality of each vector.
    fn dim(&self) -> usize;
    /// Read a single vector by index.
    fn get(&self, index: usize) -> Result<&[T], Error>;
    /// Read a range of vectors.
    fn get_range(&self, start: usize, end: usize) -> Result<Vec<&[T]>, Error>;
    /// Prebuffer a range (async download + verify).
    fn prebuffer(&self, start: usize, end: usize) -> impl Future<Output = Result<(), Error>>;
}
```

---

## 13.3 Transport Layer

### 13.3.1 Chunked Resource Transport

Remote files are downloaded in fixed-size chunks (default: 1 MB) with
optional parallelism (default: 4 concurrent downloads).

**Protocol:**
- **HTTP Range requests**: `Range: bytes=offset-offset+length`
- **Size discovery**: HEAD request to determine total file size
- **Parallel downloads**: Multiple chunks fetched concurrently via
  a thread pool, written to the cache file at absolute offsets
- **Progress tracking**: Bytes downloaded counter, per-chunk completion

**Retry semantics:**
- Configurable retry count per chunk
- Exponential backoff between retries
- Failed chunks logged but do not abort the entire download

### 13.3.2 Merkle-Authenticated File Channel (MAFileChannel)

The core abstraction for cached remote file access:

```
MAFileChannel
├── Cache file (raw content, sparse)
├── Merkle state (.mrkl) — tracks which chunks are verified
├── Merkle reference (.mref) — authoritative hashes from remote
└── ChunkedResourceTransportService — downloads chunks
```

**Initialization (3 cases):**

1. **Fresh**: No cache, no merkle state → download `.mref`, create
   `.mrkl` from it, create empty cache file
2. **Resuming**: Both cache and state exist → load as-is, resume
3. **Invalid state**: Any other combination → error

**Read semantics:**
1. Check merkle state for the chunk(s) containing the requested range
2. If valid: read from cache file directly
3. If invalid: download chunk, hash it, verify against reference,
   write to cache, mark as valid in state
4. Return the requested bytes

**Write-through caching:**
- Downloads are written directly to the cache file at the correct offset
- Merkle state is updated atomically (single bit set per chunk)
- Multiple concurrent reads of the same chunk deduplicate to one download

### 13.3.3 Chunk Scheduling

The scheduler determines which chunks to fetch when a read request
arrives. Strategies:

- **Default**: Fetch only the exact chunks needed
- **Aggressive**: Pre-fetch nearby chunks anticipating sequential access
- **Conservative**: Single chunk at a time
- **Adaptive**: Monitors access patterns, switches between strategies

---

## 13.4 Cache Layer

### 13.4.1 Cache Directory Structure

```
{cache_dir}/                          (from settings.yaml)
├── {dataset-name}/                   (one per dataset)
│   ├── {source-file}.fvec            (content cache)
│   ├── {source-file}.fvec.mrkl       (merkle state)
│   ├── {source-file}.fvec.mref       (merkle reference, temporary)
│   ├── {another-file}.ivec
│   └── {another-file}.ivec.mrkl
```

### 13.4.2 Cache Key Derivation

- **Remote files**: Dataset name (from catalog entry) + source path
  from the profile view. The URL is NOT used as the cache key —
  the dataset name provides a stable, human-readable directory.
- **Merkle files**: Same path as content with `.mrkl` extension.

### 13.4.3 Cache Invalidation

- **Hash-based**: Chunks are valid only if they hash correctly against
  the merkle reference. There is no time-based expiration.
- **Reference refresh**: If the remote `.mref` changes (new dataset
  version), the old `.mrkl` becomes invalid and chunks are re-verified.
- **Manual deletion**: Users can delete cache files; they will be
  re-downloaded on next access.

### 13.4.4 Promotion

After all chunks for a file are verified (via prebuffer or accumulated
reads), the access path is promoted from channel-backed to mmap-backed:

1. Channel-backed: `pread()` syscalls through the file channel
2. Mmap-backed: Zero-copy via `mmap()`, no syscalls per read

Promotion is automatic and transparent to the consumer.

---

## 13.5 Configuration

### 13.5.1 Settings File

Location: `~/.config/vectordata/settings.yaml`

```yaml
cache_dir: /mnt/testdata/vectordata-cache
protect_settings: true
```

### 13.5.2 Cache Directory Resolution

When `cache_dir` is not explicitly configured, the system can
auto-resolve using mount point selection:

| Strategy | Behavior |
|---|---|
| `default` | `~/.cache/vectordata` |
| `auto:largest-non-root` | Writable mount with most free space, excluding `/` |
| `auto:largest-any` | Writable mount with most free space, including `/` |
| Explicit path | Validated for existence, writability, absolute path |

The resolved path is persisted back to `settings.yaml` so subsequent
runs don't repeat the resolution.

### 13.5.3 Mount Point Selection

Mount point selection for auto-resolution:

1. Enumerate all mounted filesystems
2. Filter out pseudo-filesystems (tmpfs, proc, sysfs, cgroup, overlay)
3. Filter out system directories (/proc, /sys, /dev, /run, /boot)
4. Check POSIX write permissions for the current user (uid/gid/groups)
5. Sort by available space (largest first)
6. Create `vectordata-cache/` subdirectory on selected mount

---

## 13.6 Windowed Access

Profile views can specify windows (record ranges) on source files:

```yaml
profiles:
  10m:
    base_count: 10000000
    base_vectors: "profiles/base/base_vectors.mvec[0..10000000)"
```

### 13.6.1 Window Specification Syntax

```
[0..100)        # Half-open: indices 0 through 99
[0..100]        # Closed: indices 0 through 100
0..100          # Shorthand for half-open
10M             # Single size: [0..10000000)
```

### 13.6.2 Window Semantics

- Windows are resolved at view creation time
- `count()` returns the window size, not the file size
- `get(0)` returns the first vector in the window (not the file)
- Byte offsets are computed from the window start + record size
- For remote files, only chunks within the window are downloaded
- Windows do not affect the merkle tree — the full-file merkle is used,
  but only the relevant chunks are verified

---

## 13.7 Integration Requirements

### 13.7.1 Visualize Commands

All `veks visualize` commands MUST use the data access layer instead
of directly opening files:

```rust
// WRONG: direct file access
let reader = MmapVectorReader::<f32>::open_fvec(&path);

// RIGHT: data access layer
let loader = DatasetLoader::new();
let view = loader.load("dataset:profile")?;
let base = view.base_vectors().unwrap();
let vec = base.get(0)?;
```

This ensures that `veks visualize pca cohere_msmarco:90m` works
transparently with on-demand chunk fetching.

### 13.7.2 Pipeline Commands

Pipeline commands that read input files SHOULD use the data access
layer when the source is a catalog specifier. For local files
(the common pipeline case), direct mmap access remains valid.

### 13.7.3 Prebuffer Command

The `veks datasets prebuffer` command uses the same data access layer
but explicitly calls `prebuffer()` to download all chunks upfront.
After prebuffering, subsequent accesses use the mmap fast path.

---

## 13.8 Implementation Status

| Component | Status |
|---|---|
| Catalog resolution | Implemented (CatalogSources, Catalog) |
| Settings/config | Implemented (settings.yaml, cache_dir) |
| Mount point selection | Implemented (list-mounts) |
| MmapVectorReader (local) | Implemented (f32/f16/f64/i32/i16/u8) |
| Merkle tree core | Implemented (MerkleRef, MerkleState, .mref/.mrkl) |
| HTTP transport | Implemented (reqwest, Range requests, parallel chunks) |
| CachedChannel (MAFileChannel) | Implemented (merkle-verified chunk cache) |
| DatasetLoader | Implemented (local + catalog specifier resolution) |
| TypedVectorView trait | Implemented (local + remote, windowed) |
| On-demand chunk fetching | Implemented (lazy per-chunk download + verify) |
| Channel → mmap promotion | Implemented (automatic on prebuffer complete) |
| Windowed remote access | Implemented (window applied at view level) |
| Dataset attributes + variables | Implemented (synced to dataset.yaml) |

---

## 13.9 Wire Formats and Interoperability

This section specifies the on-disk and over-the-wire formats precisely
enough for other languages to implement a compatible client.

### 13.9.1 xvec Binary Format

All vector files (`.fvec`, `.mvec`, `.ivec`, `.dvec`, etc.) use the
"xvec" format: a flat sequence of records where each record is:

```
[dim: i32 LE][v0: T][v1: T]...[v_{dim-1}: T]
```

- `dim` is a 4-byte little-endian signed integer, repeated identically
  for every record in the file.
- Each element `v_i` is `sizeof(T)` bytes in little-endian order.
- The **entry size** is `4 + dim * sizeof(T)` bytes.
- The **record count** is `file_size / entry_size`.

| Extension | Element type | sizeof(T) |
|-----------|-------------|-----------|
| `.fvec`   | float32     | 4         |
| `.mvec`   | float16     | 2         |
| `.dvec`   | float64     | 8         |
| `.ivec`   | int32       | 4         |
| `.svec`   | int16       | 2         |
| `.bvec`   | uint8       | 1         |

To read vector `i`: seek to `i * entry_size`, read 4 bytes for dim
(may be skipped after first read), then read `dim * sizeof(T)` bytes.

### 13.9.2 Merkle Tree Format (.mref)

The `.mref` file contains a complete merkle hash tree over the content
file, structured as a heap-indexed binary tree:

```
[node_hashes: node_count * 32 bytes][footer: 41 bytes]
```

**Hash array layout** (heap-indexed):
- Index 0: root hash
- Indices 1..internal_node_count-1: internal nodes
- Indices internal_node_count..node_count-1: leaf nodes
- Left child of node `i`: `2*i + 1`
- Right child of node `i`: `2*i + 2`
- Parent of node `i`: `(i - 1) / 2`

**Leaf hashes**: `SHA-256(chunk_data)` where each chunk is
`chunk_size` bytes (last chunk may be shorter).

**Internal node hashes**: `SHA-256(left_child_hash || right_child_hash)`

**Footer (41 bytes, big-endian)**:

| Offset | Size | Type   | Field                |
|--------|------|--------|----------------------|
| 0      | 8    | u64 BE | chunk_size           |
| 8      | 8    | u64 BE | total_content_size   |
| 16     | 4    | u32 BE | total_chunks         |
| 20     | 4    | u32 BE | leaf_count (= cap_leaf) |
| 24     | 4    | u32 BE | cap_leaf (power of 2) |
| 28     | 4    | u32 BE | node_count           |
| 32     | 4    | u32 BE | offset (= internal_node_count) |
| 36     | 4    | u32 BE | internal_node_count  |
| 40     | 1    | u8     | footer_length (41)   |

**V2 footer (45 bytes)**: Adds `bitset_size: u32 BE` at offset 40,
footer_length at offset 44 = 45.

This format is byte-compatible with the Java `MerkleTree` serialization.

### 13.9.3 Merkle State Format (.mrkl)

The `.mrkl` file is a superset of `.mref` — the same hash array and
footer, plus a validity bitset between the hashes and footer:

```
[node_hashes: node_count * 32 bytes][bitset: ceil(leaf_count/8) bytes][footer]
```

Each bit in the bitset corresponds to a leaf chunk. Bit `i` = 1 means
chunk `i` has been downloaded and verified against its hash. Zero bits
indicate chunks that need to be fetched.

When an `.mref` is first downloaded for a new dataset, it is converted
to `.mrkl` by appending a zero bitset. As chunks are verified, bits
are flipped to 1.

### 13.9.4 dataset.yaml Schema

```yaml
name: string                          # required
description: string                   # optional

attributes:                           # optional but recommended
  distance_function: string           # COSINE, L2, DOT_PRODUCT, L1
  is_normalized: bool                 # vectors are L2-normalized
  is_zero_vector_free: bool           # required for publishable datasets
  is_duplicate_vector_free: bool      # required for publishable datasets
  model: string                       # embedding model name
  license: string                     # e.g., Apache-2.0
  vendor: string                      # dataset provider
  notes: string                       # freeform
  tags: {key: value, ...}             # freeform key-value

variables:                            # pipeline-produced, dynamic
  base_count: '340838098'             # after dedup + query extraction
  vector_count: '407314954'           # raw source vectors
  duplicate_count: '66466856'         # duplicates removed
  zero_count: '0'                     # zero vectors found
  clean_count: '340848098'            # after dedup, before query split
  # ... additional verified_count:* entries

upstream:                             # pipeline definition
  defaults:
    query_count: 10000
    seed: 42
  steps:
    - id: step-name
      run: "group command"
      options: {key: value, ...}
      after: [dependency-ids]
      per_profile: bool               # expand per sized profile
      phase: 0                        # execution phase

profiles:
  default:
    maxk: 100
    base_vectors: profiles/base/base_vectors.mvec
    query_vectors: profiles/base/query_vectors.mvec
    neighbor_indices: profiles/default/neighbor_indices.ivec
    neighbor_distances: profiles/default/neighbor_distances.fvec
  sized:
    ranges: ["mul:1mi/2"]             # compact spec, expanded at runtime
    facets:
      base_vectors: "profiles/base/base_vectors.mvec:${range}"
      neighbor_indices: "profiles/${profile}/neighbor_indices.ivec"
```

### 13.9.5 Catalog Format (catalog.json)

```json
[
  {
    "name": "dataset-name",
    "path": "relative/path/to/dataset.yaml",
    "dataset_type": "dataset.yaml",
    "layout": {
      "attributes": { ... },          // same as dataset.yaml attributes
      "profiles": {
        "default": {
          "maxk": 100,
          "base_vectors": "profiles/base/base_vectors.mvec",
          ...
        },
        "1mi": {
          "base_count": 1048576,
          "base_vectors": {
            "source": "profiles/base/base_vectors.mvec",
            "window": [{"min_incl": 0, "max_excl": 1048576}]
          },
          ...
        }
      }
    }
  }
]
```

**URL resolution**: Given a catalog at `https://host/path/catalog.json`
and an entry with `"path": "subdir/dataset.yaml"`, the dataset URL is
`https://host/path/subdir/dataset.yaml`. Facet URLs are relative to
the dataset directory: `https://host/path/subdir/profiles/base/file.mvec`.

### 13.9.6 Catalog Discovery

Catalogs are configured in `~/.config/vectordata/catalogs.yaml`:

```yaml
- https://example.com/datasets/
- /local/path/to/datasets/
```

Each entry is a base URL. The client appends `catalog.json` to
discover datasets. Multiple catalogs are merged — entries from all
catalogs are visible, with name collisions resolved by first-match.

### 13.9.7 Settings (settings.yaml)

Location: `~/.config/vectordata/settings.yaml`

```yaml
cache_dir: /path/to/cache      # local cache for remote datasets
protect_settings: true          # prevent auto-modification
```

### 13.9.8 Standard Facet Keys

| Key | Description | Format |
|-----|-------------|--------|
| `base_vectors` | Corpus vectors for search | fvec/mvec |
| `query_vectors` | Query vectors for evaluation | fvec/mvec |
| `neighbor_indices` | Ground-truth KNN neighbor ordinals | ivec |
| `neighbor_distances` | Ground-truth KNN distances | fvec |
| `filtered_neighbor_indices` | Filtered KNN neighbor ordinals | ivec |
| `filtered_neighbor_distances` | Filtered KNN distances | fvec |
| `metadata_content` | Metadata records | slab |
| `metadata_predicates` | Predicate trees for filtering | slab |
| `metadata_indices` | Predicate evaluation results | slab |

### 13.9.9 Profile Naming Convention

Sized profile names use SI/IEC suffixes with compound notation:

| Suffix | Multiplier |
|--------|-----------|
| `k`    | 1,000 |
| `m`    | 1,000,000 |
| `g`    | 1,000,000,000 |
| `t`    | 1,000,000,000,000 |
| `ki`   | 1,024 |
| `mi`   | 1,048,576 |
| `gi`   | 1,073,741,824 |
| `ti`   | 1,099,511,627,776 |

Compound: `1g24m` = 1,024,000,000. Binary doubling series use IEC
suffixes: `1mi, 2mi, 4mi, ..., 512mi, 1gi, 2gi, ...`.

Profiles are sorted by `base_count` ascending (derived from name if
not explicit), with `default` always first.
