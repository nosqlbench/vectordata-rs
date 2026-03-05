# vectordata-rs

A Rust workspace for working with vector datasets — loading, converting, analyzing, and storing ANN benchmark data.

## Crates

### vectordata

A library for efficiently accessing vector datasets described by `dataset.yaml` configurations. Provides a unified interface for reading vector data from local files (via memory mapping) or remote URLs (via HTTP Range requests).

- Parses `dataset.yaml` to understand dataset structure, profiles, and views
- Supports `.fvec`, `.ivec`, and other standard ANN benchmark formats
- Memory-mapped local access and on-demand remote retrieval

```toml
[dependencies]
vectordata = { path = "vectordata" }
```

### veks

An umbrella CLI (`veks`) for vector data operations:

- **analyze** — inspect, profile, compare, and visualize vector data files
- **bulkdl** — bulk file downloader driven by YAML config with token expansion
- **convert** — convert vector data between formats
- **import** — import data into preferred internal format by facet type
- **run / pipeline** — execute command-stream pipelines defined in `dataset.yaml`

### slabtastic

A library and CLI (`slab`) for the slabtastic file format — a streamable, random-accessible, appendable data layout for non-uniform data by ordinal. Used as the storage format for metadata facets in vector datasets.

- Pages with forward and backward traversal
- Supports files up to 2^63 bytes
- Conventional `.slab` extension

```toml
[dependencies]
slabtastic = { path = "slabtastic" }
```

## Building

```sh
cargo build            # build all crates
cargo build -p veks    # build only the veks CLI
cargo test             # run all tests
```

## License

Apache-2.0
