// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Example: variable-length vector I/O.
//!
//! Demonstrates reading and writing vectors with non-uniform dimensions
//! using the `VarlenSource` / `VarlenSink` traits.

fn main() -> Result<(), String> {
    let path = std::env::temp_dir().join("veks_io_varlen_example.fvec");

    // ── Write vectors with different dimensions ──────────────────────
    let mut writer = veks_io::create_varlen(&path)?;
    // Each record can have a different dimension
    writer.write_record(2, &[1.0f32, 2.0].iter()
        .flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    writer.write_record(5, &[10.0f32, 20.0, 30.0, 40.0, 50.0].iter()
        .flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    writer.write_record(3, &[100.0f32, 200.0, 300.0].iter()
        .flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    writer.finish()?;
    println!("Wrote 3 variable-length records to {}", path.display());

    // ── Read back ────────────────────────────────────────────────────
    let mut reader = veks_io::open_varlen(&path)?;
    println!("\nRecords:");
    while let Some(record) = reader.next_record() {
        let floats: Vec<f32> = record.data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        println!("  dim={}: {:?}", record.dimension, floats);
    }

    // Clean up
    let _ = std::fs::remove_file(&path);
    Ok(())
}
