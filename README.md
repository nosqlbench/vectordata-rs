# ⚡️ vectordata-rs

**High-performance Rust toolkit for large-scale vector datasets.**

`vectordata-rs` is an industrial-strength workspace for loading, converting, analyzing, and storing ANN (Approximate Nearest Neighbor) benchmark data at scale. Built for speed, resource efficiency, and transparent remote access.

---

## 🛠 The Workspace

### [**vectordata**](./vectordata/) — Transparent Access Layer
A library for efficient dataset access via `dataset.yaml`.
- **Zero-copy I/O**: Memory-mapped local access for maximum throughput.
- **Smart Remote**: On-demand, Merkle-verified retrieval via HTTP Range requests.
- **Unified API**: One interface for local files and remote URLs.

### [**veks**](./veks/) — The Umbrella CLI
The primary entry point for all vector operations.
- 🔍 **analyze** — Inspect, profile, and visualize vector distributions.
- 🔄 **convert** — Seamlessly transform between `.fvec`, `.ivec`, `.mvec`, and more.
- 📥 **import** — Canonicalize data into optimized internal formats.
- 🚀 **pipeline** — Execute complex DAG-ordered data processing streams.
- 📦 **bulkdl** — Parallel, config-driven dataset downloader.

### [**slabtastic**](./slabtastic/) — Random-Access Storage
The storage engine for non-uniform data (metadata, predicates, etc.).
- **Page-Aligned**: Optimized for efficient random and sequential I/O.
- **High Scale**: Supports append-friendly files up to 2^63 bytes.
- **Ordinal Addressing**: O(1) expected lookup for variable-length records.

---

## 📖 Documentation

Everything you need to know, organized by the **Diátaxis** framework:

- 🎓 **[Tutorials](./docs/tutorials/README.md)** — Step-by-step lessons to get you started.
- 💡 **[How-to Guides](./docs/howto/README.md)** — Goal-oriented solutions for specific tasks.
- 🧠 **[Explanation](./docs/explanation/README.md)** — Deep dives into architecture and design.
- 📚 **[Reference](./docs/reference/README.md)** — Technical specs, APIs, and wire formats.
- 🏗 **[SRD](./docs/design/README.md)** — Formal System Requirements and design documents.

---

## ⚡️ Quick Start

```bash
# Build the entire workspace
cargo build --release

# Run a sample analysis
./target/release/veks analyze describe path/to/data.fvec

# Run all tests
cargo test
```

## ⚖️ License

Apache-2.0
