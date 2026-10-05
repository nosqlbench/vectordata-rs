// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Example: zero-copy random access via memory-mapped I/O.
//!
//! Demonstrates `MmapReader<f32>` for zero-copy, random-access reads on
//! fvec files. This is the fastest way to access individual vectors.

use veks_io::xvec::mmap::MmapReader;

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
    let reader = MmapReader::<f32>::open_fvec(&path)
        .map_err(|e| format!("mmap open: {}", e))?;
    println!("Mmap: count={}, dim={}", reader.count(), reader.dim());

    // Random access — no copying, no allocation
    let vec_0: &[f32] = reader.get_slice(0);
    let vec_999: &[f32] = reader.get_slice(999);
    println!("\nvec[0]   first 4 elements: {:?}", &vec_0[..4]);
    println!("vec[999] first 4 elements: {:?}", &vec_999[..4]);

    // Compute L2 distance between two vectors
    let a = reader.get_slice(0);
    let b = reader.get_slice(1);
    let l2: f32 = a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    println!("\nL2 distance between vec[0] and vec[1]: {:.4}", l2);

    // Clean up
    let _ = std::fs::remove_file(&path);
    Ok(())
}
