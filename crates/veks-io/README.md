# veks-io

Standalone Rust library for reading and writing vector data formats used in
approximate nearest neighbor (ANN) benchmarks and embedding datasets.

## Supported Formats

| Format | Extension | Element Type | Read | Write | Mmap |
|--------|-----------|-------------|------|-------|------|
| fvec   | `.fvecs`   | float32     | ✓    | ✓     | ✓    |
| ivec   | `.ivecs`   | int32       | ✓    | ✓     | ✓    |
| bvec   | `.bvecs`   | uint8       | ✓    | ✓     | ✓    |
| dvec   | `.dvecs`   | float64     | ✓    | ✓     | ✓    |
| mvec   | `.mvecs`   | float16     | ✓    | ✓     | ✓    |
| svec   | `.svecs`   | int16       | ✓    | ✓     | ✓    |
| npy    | `.npy`    | float       | ✓*   | —     | —    |
| parquet| `.parquet` | float      | ✓*   | —     | —    |
| slab   | `.slab`   | binary      | ✓*   | ✓*    | —    |

\* Requires optional feature flag (`npy`, `parquet`, or `slab`).

## Installation

```toml
[dependencies]
veks-io = "0.13"
```

With optional format support:

```toml
[dependencies]
veks-io = { version = "0.13", features = ["npy", "parquet"] }
```

## Quick Start

```rust
use veks_io::{open, create, probe};

// Probe metadata without reading all data
let meta = probe("dataset.fvecs")?;
println!("dim={}, records={:?}", meta.dimension, meta.record_count);

// Stream-read all records
let mut reader = open("dataset.fvecs")?;
// Ok(None) is the end; an Err is a record that cannot be read
while let Some(record) = reader.next_record()? {
    // record: Vec<u8> — raw little-endian element bytes
}

// Write vectors
let mut writer = create("output.fvecs", 128)?;  // dimension=128
writer.write_record(0, &data);
writer.finish()?;
```

## Zero-Copy Random Access

For random access on xvec files, use `XvecReader` at the file's own
type (`ScalarReader` for a header-free column, `IndexedVvecReader` for
variable-length records). Each reads a local file through a memory
mapping by default, and any other `ByteSource` you supply:

```rust
use veks_io::{VectorReader, XvecReader};

let reader = XvecReader::<f32>::open_path("base.fvecs".as_ref())?;
let vec: Option<&[f32]> = VectorReader::get_slice(&reader, 42);  // zero-copy, no allocation
```

These are the readers `vectordata` uses for every dataset it opens,
over its own source for remote and cached files.

## Variable-Length Vectors

For files where records have non-uniform dimensions:

```rust
let mut reader = veks_io::open_varlen("mixed.fvecs")?;
while let Some(record) = reader.next_record()? {
    println!("dim={}, bytes={}", record.dimension, record.data.len());
}
```

## Feature Flags

| Feature   | Dependencies           | Purpose                          |
|-----------|------------------------|----------------------------------|
| `npy`     | ndarray, ndarray-npy   | NumPy `.npy` directory reading   |
| `parquet` | arrow, parquet         | Apache Parquet format support    |
| `slab`    | slabtastic             | Page-oriented binary container   |

By default, only the xvec formats are enabled (zero heavy dependencies — just
byteorder, half, memmap2, libc, log).

## Examples

See the `examples/` directory:

- `read_write.rs` — Write, probe, and stream-read an fvec file
- `mmap_access.rs` — Zero-copy random access via memory-mapped I/O
- `varlen.rs` — Variable-length vector I/O with non-uniform dimensions

```bash
cargo run --example read_write -p veks-io
cargo run --example mmap_access -p veks-io
cargo run --example varlen -p veks-io
```

## License

Apache-2.0
