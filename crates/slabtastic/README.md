# Slabtastic

A streamable, readable, writeable, randomly accessible file format for
non-uniform data indexed by ordinal.

Slab files (`.slab`) store variable-length records addressable by integer
ordinal. The format supports files up to 2^63 bytes, memory-mapped random
access with zero syscalls per read, streaming writes, and background I/O
with progress tracking.

## Format overview

A slab file is a sequence of **pages** followed by a trailing **pages page**
(index). Each data page holds a contiguous range of records:

```
[header: 8 bytes][record data...][offset table: (n+1)×4 bytes][footer: 16 bytes]
```

- **Header**: 4-byte magic (`SLAB`) + 4-byte page size
- **Footer**: start ordinal, record count, page type, namespace index
- **Pages page**: sorted `(start_ordinal, file_offset)` entries enabling
  O(log n) lookup — or O(1) with interpolation search for uniform distributions

## Quick start

### Writing

```rust
use slabtastic::{SlabWriter, WriterConfig};

let config = WriterConfig::default();
let mut writer = SlabWriter::new("output.slab", config)?;
writer.add_record(b"first record")?;
writer.add_record(b"second record")?;
writer.finish()?;
```

### Reading

```rust
use slabtastic::SlabReader;

let reader = SlabReader::open("output.slab")?;
let data = reader.get(0)?;           // fetch by ordinal
let data_ref = reader.get_ref(1)?;   // zero-copy reference

// iterate all records in batches
for batch in reader.batch_iter(4096) {
    for (ordinal, bytes) in &batch {
        // process record
    }
}
```

### Async / background I/O

```rust
use slabtastic::{SlabWriter, WriterConfig};

let records = vec![vec![1u8; 100]; 1000];
let task = SlabWriter::write_from_iter_async(
    "output.slab",
    WriterConfig::default(),
    records.iter().map(|r| r.as_slice()),
    |count| println!("wrote {} records", count),
)?;

// task.progress() returns a SlabProgress with completion fraction
task.join()?;
```

## Writer configuration

`WriterConfig` controls page sizing:

| Field                  | Default    | Description                              |
|------------------------|------------|------------------------------------------|
| `min_page_size`        | 512 bytes  | Minimum page size                        |
| `preferred_page_size`  | 4 MiB      | Flush threshold — pages flush at this size |
| `max_page_size`        | u32::MAX   | Hard ceiling per page                    |
| `page_alignment`       | false      | Pad pages to `min_page_size` multiples   |

## Reader performance

- **Interpolation search** over the pages page — O(1) expected for uniform
  ordinal distributions (~1–2 probes vs ~12 for binary search)
- **Zero syscalls per get** — all access through `mmap`
- **Two memory loads** — offset pair lookup from the page's offset table
- **One memcpy** — record bytes to output buffer
- **Batch iteration** — page-at-a-time streaming for sequential access

## Namespaces

A single slab file can hold multiple logical streams via **namespaces**.
Each namespace is an independent sequence of ordinals. The trailing
namespaces page maps namespace names to indices used in page footers.

## CLI tool

The `slab` binary provides file inspection and manipulation:

```
slab analyze   — display file structure and statistics
slab check     — validate file integrity
slab get       — retrieve records by ordinal (hex, raw, base64)
slab rewrite   — reorder and repack with new page settings
slab append    — extend an existing file
slab export    — write records to external formats
slab import    — read records from external sources
slab explain   — detailed structural explanation
```

## License

Apache-2.0
