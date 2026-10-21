// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Example: zero-copy random access via memory-mapped I/O.
//!
//! Demonstrates `XvecReader<f32>` for zero-copy, random-access reads on
//! fvec files, and `read_into` for a run of records at once. The reader
//! opens a file at the type its extension names and no other.

use veks_io::{VectorReader, XvecReader};

fn main() -> Result<(), String> {
    let path = std::env::temp_dir().join("veks_io_mmap_example.fvec");

    // Write 1000 vectors of dimension 128
    let dim = 128u32;
    let count = 1000u32;
    let mut writer = veks_io::create(&path, dim)?;
    for i in 0..count {
        let data: Vec<u8> = (0..dim)
            .flat_map(|d| ((i * dim + d) as f32).to_le_bytes())
            .collect();
        writer.write_record(i as i64, &data);
    }
    writer.finish()?;
    println!("Wrote {} vectors (dim={}) to {}", count, dim, path.display());

    // Open for zero-copy mmap access
    let reader = XvecReader::<f32>::open_path(&path)
        .map_err(|e| format!("mmap open: {}", e))?;
    println!("Mmap: count={}, dim={}", reader.count(), reader.dim());

    // Random access — no copying, no allocation. `None` only for an
    // index past the end, or a record that is not on an f32 boundary
    // (never, for a 4-byte element behind a 4-byte header).
    let vec_0: &[f32] = VectorReader::get_slice(&reader, 0).expect("record 0");
    let vec_999: &[f32] = VectorReader::get_slice(&reader, 999).expect("record 999");
    println!("\nvec[0]   first 4 elements: {:?}", &vec_0[..4]);
    println!("vec[999] first 4 elements: {:?}", &vec_999[..4]);

    // A run of records into one buffer: the bulk path.
    let mut block = vec![0f32; dim as usize * 10];
    let got = reader.read_into(990, &mut block).map_err(|e| e.to_string())?;
    println!("read_into(990): {} records, last starts {:?}", got, &block[dim as usize * 9..dim as usize * 9 + 2]);

    // Compute L2 distance between two vectors
    let a = VectorReader::get_slice(&reader, 0).expect("record 0");
    let b = VectorReader::get_slice(&reader, 1).expect("record 1");
    let l2: f32 = a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("\nL2 distance between vec[0] and vec[1]: {:.4}", l2);

    // Clean up
    let _ = std::fs::remove_file(&path);
    Ok(())
}
