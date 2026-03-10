# Getting Started

This tutorial walks through writing a slab file, reading records back, and
using the CLI to inspect the result.

## Prerequisites

- Rust toolchain (edition 2024)
- `slabtastic` crate added to your `Cargo.toml`

```toml
[dependencies]
slabtastic = "0.8"
```

## Step 1: Write a slab file

Create a new file, add some records, and call `finish()` to flush the pages
page (the index):

```rust
use slabtastic::{SlabWriter, WriterConfig};

fn main() -> slabtastic::Result<()> {
    let config = WriterConfig::default();
    let mut writer = SlabWriter::new("demo.slab", config)?;

    writer.add_record(b"hello")?;
    writer.add_record(b"world")?;
    writer.add_record(b"from slabtastic")?;

    writer.finish()?;
    println!("Wrote 3 records to demo.slab");
    Ok(())
}
```

Key points:

- `WriterConfig::default()` uses 512-byte minimum, 4 MiB preferred, and
  no page alignment.
- Records are accumulated in memory and flushed to disk as complete pages
  when the preferred page size is reached.
- You **must** call `finish()` — this flushes any remaining records and
  writes the trailing pages page that makes the file valid.

## Step 2: Read records by ordinal

Open the file and fetch individual records by their zero-based ordinal:

```rust
use slabtastic::SlabReader;

fn main() -> slabtastic::Result<()> {
    let reader = SlabReader::open("demo.slab")?;

    // Zero-copy: get_ref returns a &[u8] slice into the mmap
    let first = reader.get_ref(0)?;
    println!("ordinal 0: {}", String::from_utf8_lossy(first));

    // Copying: get returns an owned Vec<u8>
    let second = reader.get(1)?;
    println!("ordinal 1: {}", String::from_utf8_lossy(&second));

    // Check if an ordinal exists
    assert!(reader.contains(2));
    assert!(!reader.contains(99));

    Ok(())
}
```

## Step 3: Iterate all records

Use `iter()` to read every record in ordinal order:

```rust
use slabtastic::SlabReader;

fn main() -> slabtastic::Result<()> {
    let reader = SlabReader::open("demo.slab")?;
    let all = reader.iter()?;

    for (ordinal, data) in &all {
        println!("ordinal {ordinal}: {}", String::from_utf8_lossy(data));
    }
    println!("Total: {} records", all.len());
    Ok(())
}
```

## Step 4: Inspect with the CLI

The `slab` binary provides file maintenance commands:

```bash
# Show file structure and statistics
slab analyze demo.slab

# Retrieve specific records
slab get demo.slab 0 1 2

# Check file integrity
slab check demo.slab
```

## Next steps

- [Streaming I/O](streaming-io.md) — batched reads, sink reads, and
  async operations with progress polling
- [Append Data](../how-to/append-data.md) — add records to an existing file
- [Wire Format Specification](../reference/wire-format.md) — understand the
  on-disk layout
