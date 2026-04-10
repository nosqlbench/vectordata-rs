// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Example: write, probe, and read back an fvec file.
//!
//! Demonstrates the core veks-io API: `create()`, `probe()`, `open()`.

fn main() -> Result<(), String> {
    let path = std::env::temp_dir().join("veks_io_example.fvec");

    // ── Write 5 vectors of dimension 3 ───────────────────────────────
    let mut writer = veks_io::create(&path, 3)?;
    for i in 0..5u32 {
        let data: Vec<u8> = [i as f32, (i * 10) as f32, (i * 100) as f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        writer.write_record(i as i64, &data);
    }
    writer.finish()?;
    println!("Wrote 5 vectors to {}", path.display());

    // ── Probe metadata ───────────────────────────────────────────────
    let meta = veks_io::probe(&path)?;
    println!(
        "Probed: dimension={}, element_size={}, records={:?}",
        meta.dimension, meta.element_size, meta.record_count,
    );

    // ── Stream-read all records ──────────────────────────────────────
    let mut reader = veks_io::open(&path)?;
    println!("\nRecords:");
    let mut ordinal = 0;
    while let Some(record) = reader.next_record() {
        let floats: Vec<f32> = record
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        println!("  [{}] {:?}", ordinal, floats);
        ordinal += 1;
    }

    // Clean up
    let _ = std::fs::remove_file(&path);
    Ok(())
}
