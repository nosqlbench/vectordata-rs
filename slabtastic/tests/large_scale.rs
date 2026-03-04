// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Large-scale throughput tests for slabtastic.
//!
//! These tests are **ignored by default** because they write multi-GB
//! files and can take significant time and disk space.
//!
//! ## Running
//!
//! ```sh
//! # Run a specific scale:
//! cargo test --release -p slabtastic --test large_scale test_write_read_100m -- --ignored
//!
//! # Run all large-scale tests:
//! cargo test --release -p slabtastic --test large_scale -- --ignored
//! ```
//!
//! ## What each test does
//!
//! Each test creates a temporary slabtastic file, writes `N` records of
//! 5–25 bytes each (deterministic, based on ordinal), then:
//!
//! 1. **Write phase** — times `SlabWriter::new` → N × `add_record` →
//!    `finish` and reports records/s and file size in GB.
//! 2. **Sequential read phase** — times `SlabReader::open` → `iter()`
//!    over all records and reports records/s.
//! 3. **Spot-check** — verifies five records at known positions (first,
//!    quartiles, last) against the expected deterministic content.
//! 4. **Random-access phase** — opens a fresh reader and performs 10 000
//!    pseudo-random `get()` calls, verifying each against expected
//!    content.
//!
//! ## Approximate resource requirements
//!
//! | Scale | Records | Approx file size | Approx time (release) |
//! |-------|--------:|:----------------:|:---------------------:|
//! | 1M    |     1 M |         ~15 MB   |            < 1 s      |
//! | 10M   |    10 M |        ~150 MB   |            ~ 5 s      |
//! | 100M  |   100 M |        ~1.5 GB   |           ~ 60 s      |
//! | 1B    |     1 B |        ~15 GB    |          ~10 min      |

use std::time::Instant;

use slabtastic::{SlabReader, SlabWriter, WriterConfig};
use tempfile::NamedTempFile;

/// Generate a deterministic record of 5–25 bytes based on ordinal.
///
/// Length cycles through 5..=25 (mod 21) so page packing sees non-uniform
/// record sizes. Content is the ordinal zero-padded to the target length.
fn make_record(ordinal: u64) -> Vec<u8> {
    let len = 5 + (ordinal % 21) as usize;
    let base = format!("{ordinal:0>width$}", width = len);
    base.into_bytes()[..len].to_vec()
}

/// Write `count` records, then read them back sequentially and via random
/// access, printing timing results to stdout.
///
/// See module-level documentation for the phases and verification steps.
fn write_and_verify(label: &str, count: u64) {
    let tmp = NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    // -- Write phase ---------------------------------------------------
    let start = Instant::now();
    let config = WriterConfig::default();
    let mut writer = SlabWriter::new(&path, config).unwrap();
    for i in 0..count {
        let rec = make_record(i);
        writer.add_record(&rec).unwrap();
    }
    writer.finish().unwrap();
    let write_elapsed = start.elapsed();

    let file_size = std::fs::metadata(&path).unwrap().len();
    let write_rate = count as f64 / write_elapsed.as_secs_f64();
    println!(
        "[{label}] Wrote {count} records in {:.2}s ({:.0} rec/s, {:.2} GB)",
        write_elapsed.as_secs_f64(),
        write_rate,
        file_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // -- Sequential read phase -----------------------------------------
    let start = Instant::now();
    let reader = SlabReader::open(&path).unwrap();
    let all = reader.iter().unwrap();
    let read_elapsed = start.elapsed();

    assert_eq!(
        all.len() as u64, count,
        "expected {count} records, got {}",
        all.len()
    );

    let read_rate = count as f64 / read_elapsed.as_secs_f64();
    println!(
        "[{label}] Read {count} records in {:.2}s ({:.0} rec/s)",
        read_elapsed.as_secs_f64(),
        read_rate,
    );

    // -- Spot-check at quartile positions ------------------------------
    let spot_checks = [0u64, count / 4, count / 2, 3 * count / 4, count - 1];
    for &i in &spot_checks {
        let expected = make_record(i);
        let (ord, data) = &all[i as usize];
        assert_eq!(*ord, i as i64);
        assert_eq!(*data, expected, "mismatch at ordinal {i}");
    }

    // -- Random-access read phase (10k samples) ------------------------
    let sample_count = 10_000u64.min(count);
    let start = Instant::now();
    let reader = SlabReader::open(&path).unwrap();
    for i in 0..sample_count {
        let ordinal = ((i * 7919) % count) as i64;
        let data = reader.get(ordinal).unwrap();
        let expected = make_record(ordinal as u64);
        assert_eq!(data, expected);
    }
    let random_elapsed = start.elapsed();
    let random_rate = sample_count as f64 / random_elapsed.as_secs_f64();
    println!(
        "[{label}] Random-read {sample_count} records in {:.2}s ({:.0} rec/s)",
        random_elapsed.as_secs_f64(),
        random_rate,
    );

    println!("[{label}] Page count: {}", reader.page_count());
}

/// 1 million records (~15 MB file). The smallest large-scale test;
/// useful for quick smoke-testing of release builds.
#[test]
#[ignore]
fn test_write_read_1m() {
    write_and_verify("1M", 1_000_000);
}

/// 10 million records (~150 MB file). Exercises multi-page indexing at
/// moderate scale.
#[test]
#[ignore]
fn test_write_read_10m() {
    write_and_verify("10M", 10_000_000);
}

/// 100 million records (~1.5 GB file). Stresses page-flush cadence and
/// the pages-page index with thousands of entries.
#[test]
#[ignore]
fn test_write_read_100m() {
    write_and_verify("100M", 100_000_000);
}

/// 1 billion records (~15 GB file). Full-scale stress test. Requires
/// sufficient disk space and several minutes in release mode. Validates
/// that the format handles datasets well beyond typical in-memory sizes.
#[test]
#[ignore]
fn test_write_read_1b() {
    write_and_verify("1B", 1_000_000_000);
}
