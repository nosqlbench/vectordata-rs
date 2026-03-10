# How to Bulk Read and Write

## Bulk write

Use `add_records()` to write multiple records in a single call:

```rust
use slabtastic::{SlabWriter, WriterConfig};

fn bulk_write(path: &str) -> slabtastic::Result<()> {
    let data: Vec<Vec<u8>> = (0..1000)
        .map(|i| format!("item-{i}").into_bytes())
        .collect();
    let refs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();

    let mut writer = SlabWriter::new(path, WriterConfig::default())?;
    writer.add_records(&refs)?;
    writer.finish()?;
    Ok(())
}
```

`add_records` is semantically equivalent to calling `add_record` in a
loop — pages are flushed automatically as the preferred page size is
reached.

## Batched read

Use `batch_iter()` to read records in configurable batches without loading
the entire file into memory:

```rust
use slabtastic::SlabReader;

fn batched_read(path: &str) -> slabtastic::Result<()> {
    let reader = SlabReader::open(path)?;
    let mut iter = reader.batch_iter(256);

    loop {
        let batch = iter.next_batch()?;
        if batch.is_empty() {
            break;
        }
        println!("Got {} records", batch.len());
    }
    Ok(())
}
```

## Sink read

Write all records to a sink (e.g. a file or network socket) without
intermediate buffering:

```rust
use slabtastic::SlabReader;

fn sink_read(path: &str) -> slabtastic::Result<()> {
    let reader = SlabReader::open(path)?;
    let mut sink = Vec::new();
    let count = reader.read_all_to_sink(&mut sink)?;
    println!("Read {count} records ({} bytes)", sink.len());
    Ok(())
}
```

Records are written in ordinal order as raw bytes with no framing.

## Multi-batch concurrent read

Use `multi_batch_get()` to submit multiple independent batch read requests
for concurrent execution. Each batch is a list of ordinals; results are
returned in submission order with partial success for missing ordinals:

```rust
use slabtastic::SlabReader;

fn multi_batch(path: &str) -> slabtastic::Result<()> {
    let reader = SlabReader::open(path)?;

    let batch_a: Vec<i64> = vec![0, 5, 10];
    let batch_b: Vec<i64> = vec![1, 999, 3]; // 999 may not exist

    let results = reader.multi_batch_get(&[&batch_a, &batch_b]);

    // Results match submission order
    for (i, result) in results.iter().enumerate() {
        println!("Batch {i}: {} found, {} missing",
            result.found_count(), result.missing_count());

        for (ordinal, data) in &result.records {
            match data {
                Some(bytes) => println!("  ordinal {ordinal}: {} bytes", bytes.len()),
                None => println!("  ordinal {ordinal}: not found"),
            }
        }
    }
    Ok(())
}
```

Key points:

- All batches execute concurrently using scoped threads (one thread per
  batch for 2+ batches; single-batch requests skip thread spawning).
- Results are returned in the same order as the input batches.
- Each batch's records are in the same order as the requested ordinals.
- Missing ordinals produce `None` rather than failing the entire batch.
- Use `is_empty()` to check if a batch found no records at all.
