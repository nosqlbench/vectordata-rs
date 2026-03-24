<!-- Copyright (c) DataStax, Inc. -->
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

The following components exist in the Rust port:

| Component | Status |
|---|---|
| Catalog resolution | Implemented (CatalogSources, Catalog) |
| Settings/config | Implemented (settings.yaml, cache_dir) |
| Mount point selection | Implemented (list-mounts) |
| MmapVectorReader (local) | Implemented |
| Merkle tree core | Implemented (MerkleRef, MerkleState, .mref/.mrkl) |
| HTTP transport | Partial (reqwest-based, no chunked/range) |
| MAFileChannel | **Not implemented** |
| DatasetLoader | **Not implemented** |
| VectorView trait | **Not implemented** |
| On-demand chunk fetching | **Not implemented** |
| Chunk scheduling | **Not implemented** |
| Channel → mmap promotion | **Not implemented** |
| Windowed remote access | **Not implemented** |

The critical missing piece is the MAFileChannel equivalent — the
merkle-authenticated, lazily-caching file channel that sits between
the transport layer and the vector reader. Once this is implemented,
all other components plug in naturally.
