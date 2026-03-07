# Access Layer (Vectordata)

The **Access Layer** (provided by the `vectordata` crate) is a Rust library designed for efficient, transparent, and secure access to vector datasets. It bridges the gap between raw data storage (local files or remote URLs) and high-level application logic.

## Transparent Data Access

The Access Layer provides a unified interface for reading vector data regardless of its physical location:
- **Local Access**: Efficiently memory-mapped (`mmap`) access to local files.
- **Remote Access**: On-demand retrieval of data ranges from remote URLs via HTTP Range requests.

The Access Layer handles the underlying transfer details, connection pooling, and retry logic, allowing the application to interact with a remote dataset as if it were a local file.

## Merkle-Backed Integrity

To ensure the integrity of the data being downloaded, the Access Layer uses **Merkle-tree-based verification**. This allows the client to verify each chunk of data as it is received, rather than waiting for the entire file.

### Key Components
- **Merkle Reference File (`.mref`)**: A precomputed Merkle tree containing the SHA-256 hashes for all data chunks.
- **Merkle State File (`.mrkl`)**: A local state file that tracks which chunks have been successfully verified and stored.
- **Chunked Verification**: Data is downloaded in fixed-size chunks (e.g., 64 KB). Each chunk is hashed and compared against the corresponding leaf in the Merkle tree.

## Cached Channels

The central abstraction that ties verification to transparent access is the **Cached Channel**.
- **Differential Cache**: Only the specific data chunks requested by the application are downloaded and stored.
- **Read Path**:
  1. Check if the requested chunk is already in the local cache and verified.
  2. If not, fetch the chunk from the remote source.
  3. Verify the chunk against the Merkle reference.
  4. Save the verified chunk to the local cache and update the Merkle state.
  5. Return the requested bytes.
- **Prebuffer Support**: Eagerly download and verify all unverified chunks in the background to ensure zero-latency access during later operations.
- **Local Fallback After Prebuffer**: When `prebuffer()` completes and all chunks are verified, the backend automatically switches to a direct local-file implementation. This eliminates per-read overhead from range validity checks, Merkle state lookups, and lock contention, giving the same performance as a purely local file once the download is complete.

## Facet Discovery and Views

The Access Layer leverages the dataset specifications (`dataset.yaml`) to provide a structured view of the data:
- **Facet Manifest**: Dynamically discovers all available facets in a dataset profile.
- **Typed Accessors**: High-level methods for standard facets (e.g., `base_vectors()`, `query_vectors()`, `neighbor_indices()`).
- **Profile Resolution**: Automatically resolves the correct file paths and formats based on the selected dataset profile.

By combining efficient I/O, robust integrity checks, and a flexible data model, the Access Layer provides a powerful foundation for building vector-based applications.
