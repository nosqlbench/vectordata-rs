# Architecture Overview

`vectordata-rs` is a high-performance Rust workspace designed for the full lifecycle of vector dataset preparation, analysis, and access, particularly for Approximate Nearest Neighbor (ANN) benchmarking.

## System Components

The workspace is organized into three crates, each with a distinct responsibility:

### 1. `vectordata` (The Access Layer)
A library providing uniform access to vector datasets, including the dataset specification.
- **Dataset Specification**: The `vectordata::dataset` module defines the canonical `dataset.yaml` schema, including `DatasetConfig`, `DSProfile`, profile inheritance, aliases, views, and pipeline step definitions.
- **Unified I/O**: The `VectorReader<T>` trait provides a common interface for reading vectors by ordinal across different storage backends.
- **Format Codecs**: The `vectordata::formats` module implements the binary and text codecs for metadata (MNode), predicates (PNode), and the unified ANode wrapper with vernacular rendering.
- **Remote Access**: Handles transparent downloading, Merkle-backed integrity verification, and differential caching for remote datasets.
- **View Resolution**: Resolves `DSView` definitions into active vector readers for specific facets.

### 2. `slabtastic` (The Storage Engine)
A specialized storage library for non-uniform, randomly-accessible data.
- **Slab Format**: A page-aligned record container that stores opaque byte payloads (like metadata or predicates).
- **Ordinal Addressing**: Every record is assigned a sequential ordinal, allowing for O(1) expected lookup time.
- **High Scale**: Designed to handle files up to 2^63 bytes with efficient forward and backward traversal.

### 3. `veks` (The Toolbox)
The umbrella CLI and pipeline engine that ties everything together.
- **Command-Stream Pipelines**: A DAG-based execution engine that runs complex data processing steps (import, convert, analyze, etc.) defined in the `vectordata::dataset` schema.
- **Resource Governance**: A dynamic system that monitors system pressure and adjusts memory/thread allocations in real-time to prevent OOMs.

## Core Design Principles

### Performance at Scale
The system is built to handle multi-terabyte datasets (e.g., LAION-400M) without saturating system resources. This is achieved through:
- **Zero-copy I/O**: Extensive use of memory mapping (`mmap`) and slice-based access.
- **SIMD Acceleration**: Hand-rolled and library-provided SIMD distance functions for brute-force KNN computation.
- **Resource Awareness**: Every large-scale operation is governed by a central `ResourceGovernor`.

### Decoupling of Storage and Logic
The storage layer (`slabtastic`) knows nothing about the data it stores. The interpretation of bytes into meaningful records (like MNode or PNode) happens in a separate codec layer. This allows the system to evolve its data models without changing the underlying storage format.

### Transparent Remote Access
By integrating Merkle trees and range-based HTTP requests into the access layer, the system allows tools to operate on remote datasets as if they were local, downloading and verifying only the specific chunks of data required for the current operation.

## Technology Stack

- **Language**: Rust (Edition 2024)
- **CLI**: `clap` v4 for robust command-line parsing.
- **I/O**: `memmap2` for efficient local file access; `reqwest` for HTTP transport.
- **Concurrency**: `std::thread::scope` and `rayon` for safe, high-performance parallelism.
- **SIMD Distance**: `simsimd` for high-performance distance computations.
- **Observability**: `indicatif` for real-time progress bars and terminal feedback.
