# Streaming I/O

This tutorial covers the streaming read and write APIs: batched iteration,
sink-based reading, and async operations with progress polling.

## Batched iteration

When processing large files, loading all records into memory at once
(via `iter()`) may not be practical. Use `batch_iter()` to consume records
in configurable-size batches:

```rust
use slabtastic::SlabReader;

fn main() -> slabtastic::Result<()> {
    let reader = SlabReader::open("large.slab")?;
    let mut iter = reader.batch_iter(1024); // up to 1024 records per batch

    loop {
        let batch = iter.next_batch()?;
        if batch.is_empty() {
            break; // exhausted — 0 records means no more data
        }
        for (ordinal, data) in &batch {
            // process each record
            println!("ordinal {ordinal}: {} bytes", data.len());
        }
    }
    Ok(())
}
```

Key points:

- `batch_iter()` **consumes** the reader (takes ownership).
- Each `next_batch()` call returns up to `batch_size` records as
  `(ordinal, data)` pairs.
- An **empty vector** signals exhaustion — per the design doc: "if the
  reader returns 0 then the requestor should assume there are no more."
- A batch may contain fewer than `batch_size` records at page boundaries.

## Sink-based reading

Write all records directly to any `std::io::Write` sink without
intermediate allocation:

```rust
use slabtastic::SlabReader;
use std::fs::File;

fn main() -> slabtastic::Result<()> {
    let reader = SlabReader::open("data.slab")?;
    let mut output = File::create("all_records.bin")?;
    let count = reader.read_all_to_sink(&mut output)?;
    println!("Wrote {count} records to all_records.bin");
    Ok(())
}
```

Records are written in ordinal order with no framing — raw bytes
concatenated end-to-end.

## Async read with progress polling

For background reading with progress feedback, use `read_to_sink_async`:

```rust
use slabtastic::SlabReader;
use std::path::PathBuf;

fn main() -> slabtastic::Result<()> {
    let sink: Vec<u8> = Vec::new();
    let task = SlabReader::read_to_sink_async(
        PathBuf::from("data.slab"),
        sink,
        |count| println!("Done! Read {count} records"),
    );

    // Poll progress while the background thread works
    while !task.is_done() {
        let progress = task.progress();
        println!(
            "{}/{} ({:.1}%)",
            progress.completed(),
            progress.total(),
            progress.fraction() * 100.0
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let count = task.wait()?;
    println!("Final count: {count}");
    Ok(())
}
```

## Async write from iterator

Write records from any iterator on a background thread:

```rust
use slabtastic::{SlabWriter, WriterConfig};
use std::path::PathBuf;

fn main() -> slabtastic::Result<()> {
    let records = (0..100_000).map(|i| format!("record-{i}").into_bytes());

    let task = SlabWriter::write_from_iter_async(
        PathBuf::from("output.slab"),
        WriterConfig::default(),
        records,
        |count| println!("Finished writing {count} records"),
    );

    // Poll progress
    while !task.is_done() {
        let p = task.progress();
        println!("Written: {}", p.completed());
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    let total = task.wait()?;
    println!("Total written: {total}");
    Ok(())
}
```

Key points:

- The iterator is consumed on the background thread.
- `progress().total()` is set once the iterator is exhausted (it may be 0
  while the iterator is still running, since total is not known in advance).
- `progress().completed()` increments after each record is written.
- The `on_complete` callback runs on the background thread.
- `wait()` blocks until the thread finishes and returns the result.

## Multi-batch concurrent read

When you need to look up several independent sets of ordinals at once,
`multi_batch_get()` runs them concurrently and returns results in
submission order:

```rust
use slabtastic::SlabReader;

fn main() -> slabtastic::Result<()> {
    let reader = SlabReader::open("data.slab")?;

    let users: Vec<i64> = vec![0, 1, 2];
    let metadata: Vec<i64> = vec![100, 101];

    let results = reader.multi_batch_get(&[&users, &metadata]);

    // results[0] corresponds to `users`, results[1] to `metadata`
    for (ordinal, data) in &results[0].records {
        if let Some(bytes) = data {
            println!("user ordinal {ordinal}: {} bytes", bytes.len());
        }
    }
    Ok(())
}
```

Key points:

- Each batch runs on its own scoped thread (no `Arc` needed — threads
  borrow `&self` directly).
- Single-batch requests skip thread spawning and run inline.
- Missing ordinals return `None` instead of failing the batch.
- Use `is_empty()` to check if a batch found nothing, and
  `found_count()` / `missing_count()` for per-batch statistics.

## Next steps

- [Bulk Read and Write](../how-to/bulk-read-write.md) — synchronous bulk APIs
- [Background Tasks with Progress](../how-to/async-progress.md) — more
  patterns for `SlabTask`
