// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Profiling harness for hvec→fvec conversion.
//!
//! Replicates the convert pipeline's actual allocation and threading
//! patterns to diagnose progressive slowdowns. Prints per-interval
//! throughput stats and can pause for perf snapshots.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

/// Read buffer size (4 MiB)
const READ_BUF: usize = 4 << 20;
/// Write buffer size (4 MiB)
const WRITE_BUF: usize = 4 << 20;
/// Writeback interval (64 MiB)
const WB_INTERVAL: u64 = 64 << 20;
/// Records per reporting interval
const REPORT_INTERVAL: u64 = 1_000_000;
/// Read-ahead channel buffer (matches real pipeline)
const CHANNEL_BUF: usize = 4096;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: profile_convert <input.hvec> <output.fvec> [max_records] [mode]");
        eprintln!("  mode: reuse | alloc | channel (default: channel)");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);
    let max_records: Option<u64> = args.get(3).and_then(|s| s.parse().ok());
    let mode = args.get(4).map(|s| s.as_str()).unwrap_or("channel");

    // Open input
    let in_file = File::open(&input_path).expect("open input");
    let in_size = in_file.metadata().map(|m| m.len()).unwrap_or(0);
    advise_sequential(&in_file);
    let mut reader = BufReader::with_capacity(READ_BUF, in_file);

    // Read dimension from first record
    let dim = reader.read_i32::<LittleEndian>().expect("read dim") as u32;
    let src_elem: usize = 2; // hvec = f16
    let dst_elem: usize = 4; // fvec = f32
    let src_record_bytes = dim as usize * src_elem;
    let dst_record_bytes = dim as usize * dst_elem;
    let record_wire_size = 4 + src_record_bytes as u64;
    let total_records = in_size / record_wire_size;

    eprintln!("Input: {} ({:.1} GB)", input_path.display(), in_size as f64 / (1 << 30) as f64);
    eprintln!("  dimension={}, src_elem={}, dst_elem={}, records={}", dim, src_elem, dst_elem, total_records);
    eprintln!("  mode={}", mode);
    eprintln!("  PID={}", std::process::id());

    // Re-open to reset position
    drop(reader);
    let in_file = File::open(&input_path).expect("reopen input");
    advise_sequential(&in_file);
    let reader = BufReader::with_capacity(READ_BUF, in_file);

    // Open output
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let out_file = File::create(&output_path).expect("create output");
    let writer = BufWriter::with_capacity(WRITE_BUF, out_file);

    match mode {
        "reuse" => run_reuse(reader, writer, dim, src_record_bytes, dst_record_bytes, record_wire_size, total_records, max_records),
        "alloc" => run_alloc(reader, writer, dim, src_record_bytes, dst_record_bytes, record_wire_size, total_records, max_records),
        "channel" => run_channel(reader, writer, dim, src_record_bytes, dst_record_bytes, record_wire_size, total_records, max_records),
        _ => { eprintln!("Unknown mode: {}", mode); std::process::exit(1); }
    }
}

/// Mode 1: reuse buffers (baseline, no per-record allocation)
fn run_reuse(
    mut reader: BufReader<File>, mut writer: BufWriter<File>,
    dim: u32, src_bytes: usize, dst_bytes: usize, wire_size: u64,
    total: u64, max: Option<u64>,
) {
    print_header();
    let start = Instant::now();
    let mut int_start = Instant::now();
    let mut count: u64 = 0;
    let mut int_count: u64 = 0;
    let mut bw: u64 = 0;
    let mut int_br: u64 = 0;
    let mut int_bw: u64 = 0;
    let mut last_wb: u64 = 0;
    let mut last_dn: u64 = 0;
    let mut last_rd_dn: u64 = 0;
    let mut read_buf = vec![0u8; src_bytes];
    let mut conv_buf = vec![0u8; dst_bytes];

    loop {
        let d = match reader.read_i32::<LittleEndian>() {
            Ok(d) => d,
            Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(_) => break,
        };
        assert_eq!(d as u32, dim);
        if reader.read_exact(&mut read_buf).is_err() { break; }

        f16_to_f32_into(&read_buf, &mut conv_buf);

        writer.write_i32::<LittleEndian>(dim as i32).expect("w");
        writer.write_all(&conv_buf).expect("w");

        let rec_out = 4 + dst_bytes as u64;
        bw += rec_out;
        int_br += wire_size;
        int_bw += rec_out;
        count += 1;
        int_count += 1;

        if bw - last_wb >= WB_INTERVAL {
            writer.flush().ok();
            write_release(&writer, &mut last_wb, &mut last_dn, bw);
        }
        let br = count * wire_size;
        if br - last_rd_dn >= WB_INTERVAL {
            read_release(&reader, last_rd_dn, br);
            last_rd_dn = br;
        }

        if int_count >= REPORT_INTERVAL {
            print_stats(count, total, int_count, &int_start, &start, int_br, int_bw);
            int_count = 0; int_br = 0; int_bw = 0;
            int_start = Instant::now();
        }
        if max.is_some_and(|m| count >= m) { break; }
    }
    writer.flush().expect("flush");
    print_done(count, &start);
}

/// Mode 2: allocate new Vec per record (matches RawXvecReader + convert_elements)
fn run_alloc(
    mut reader: BufReader<File>, mut writer: BufWriter<File>,
    dim: u32, src_bytes: usize, dst_bytes: usize, wire_size: u64,
    total: u64, max: Option<u64>,
) {
    print_header();
    let start = Instant::now();
    let mut int_start = Instant::now();
    let mut count: u64 = 0;
    let mut int_count: u64 = 0;
    let mut bw: u64 = 0;
    let mut int_br: u64 = 0;
    let mut int_bw: u64 = 0;
    let mut last_wb: u64 = 0;
    let mut last_dn: u64 = 0;
    let mut last_rd_dn: u64 = 0;

    loop {
        let d = match reader.read_i32::<LittleEndian>() {
            Ok(d) => d,
            Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(_) => break,
        };
        assert_eq!(d as u32, dim);

        // Allocate new Vec per record (matches next_record())
        let mut read_buf = vec![0u8; src_bytes];
        if reader.read_exact(&mut read_buf).is_err() { break; }

        // Allocate new Vec for conversion (matches convert_elements())
        let converted = f16_to_f32_alloc(&read_buf);

        writer.write_i32::<LittleEndian>(dim as i32).expect("w");
        writer.write_all(&converted).expect("w");

        let rec_out = 4 + dst_bytes as u64;
        bw += rec_out;
        int_br += wire_size;
        int_bw += rec_out;
        count += 1;
        int_count += 1;

        if bw - last_wb >= WB_INTERVAL {
            writer.flush().ok();
            write_release(&writer, &mut last_wb, &mut last_dn, bw);
        }
        let br = count * wire_size;
        if br - last_rd_dn >= WB_INTERVAL {
            read_release(&reader, last_rd_dn, br);
            last_rd_dn = br;
        }

        if int_count >= REPORT_INTERVAL {
            print_stats(count, total, int_count, &int_start, &start, int_br, int_bw);
            int_count = 0; int_br = 0; int_bw = 0;
            int_start = Instant::now();
        }
        if max.is_some_and(|m| count >= m) { break; }
    }
    writer.flush().expect("flush");
    print_done(count, &start);
}

/// Mode 3: channel with reader thread (matches real pipeline exactly)
fn run_channel(
    mut reader: BufReader<File>, mut writer: BufWriter<File>,
    dim: u32, src_bytes: usize, dst_bytes: usize, wire_size: u64,
    total: u64, max: Option<u64>,
) {
    let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(CHANNEL_BUF);

    let reader_handle = std::thread::Builder::new()
        .name("reader".into())
        .spawn(move || {
            let mut rd_bytes: u64 = 0;
            let mut last_rd_dn: u64 = 0;
            let mut limit = max;
            loop {
                let d = match reader.read_i32::<LittleEndian>() {
                    Ok(d) => d,
                    Err(ref e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(_) => break,
                };
                assert_eq!(d as u32, dim);

                let mut buf = vec![0u8; src_bytes];
                if reader.read_exact(&mut buf).is_err() { break; }
                rd_bytes += wire_size;

                if rd_bytes - last_rd_dn >= WB_INTERVAL {
                    read_release(&reader, last_rd_dn, rd_bytes);
                    last_rd_dn = rd_bytes;
                }

                if tx.send(buf).is_err() { break; }

                if let Some(ref mut rem) = limit {
                    *rem -= 1;
                    if *rem == 0 { break; }
                }
            }
        })
        .expect("spawn reader");

    print_header();
    let start = Instant::now();
    let mut int_start = Instant::now();
    let mut count: u64 = 0;
    let mut int_count: u64 = 0;
    let mut bw: u64 = 0;
    let mut int_bw: u64 = 0;
    let mut last_wb: u64 = 0;
    let mut last_dn: u64 = 0;

    for data in rx {
        // Allocate new Vec for conversion (matches convert_elements())
        let converted = f16_to_f32_alloc(&data);
        drop(data); // explicit drop like real code

        writer.write_i32::<LittleEndian>(dim as i32).expect("w");
        writer.write_all(&converted).expect("w");

        let rec_out = 4 + dst_bytes as u64;
        bw += rec_out;
        int_bw += rec_out;
        count += 1;
        int_count += 1;

        if bw - last_wb >= WB_INTERVAL {
            writer.flush().ok();
            write_release(&writer, &mut last_wb, &mut last_dn, bw);
        }

        if int_count >= REPORT_INTERVAL {
            print_stats(count, total, int_count, &int_start, &start, 0, int_bw);
            int_count = 0; int_bw = 0;
            int_start = Instant::now();
        }
    }

    reader_handle.join().expect("reader join");
    writer.flush().expect("flush");
    print_done(count, &start);
}

// ---- conversion ----

fn f16_to_f32_into(src: &[u8], dst: &mut [u8]) {
    let n = src.len() / 2;
    for i in 0..n {
        let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        let val = half::f16::from_bits(bits).to_f32();
        let off = i * 4;
        dst[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }
}

fn f16_to_f32_alloc(src: &[u8]) -> Vec<u8> {
    let n = src.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        let val = half::f16::from_bits(bits).to_f32();
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

// ---- reporting ----

fn print_header() {
    eprintln!("\n{:>12} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "records", "pct", "int_rec/s", "cum_rec/s", "int_ms", "wr_MB/s");
}

fn print_stats(count: u64, total: u64, int_count: u64, int_start: &Instant, start: &Instant, _int_br: u64, int_bw: u64) {
    let elapsed = int_start.elapsed();
    let secs = elapsed.as_secs_f64();
    let total_secs = start.elapsed().as_secs_f64();
    let pct = if total > 0 { count as f64 / total as f64 * 100.0 } else { 0.0 };
    let int_rps = int_count as f64 / secs;
    let cum_rps = count as f64 / total_secs;
    let wr_mbps = int_bw as f64 / secs / (1 << 20) as f64;
    eprintln!("{:>12} {:>9.1}% {:>12.0} {:>12.0} {:>10.0} {:>10.1}",
        count, pct, int_rps, cum_rps, elapsed.as_millis(), wr_mbps);
}

fn print_done(count: u64, start: &Instant) {
    let total = start.elapsed();
    eprintln!("\nDone: {} records in {:.1}s ({:.0} rec/s)",
        count, total.as_secs_f64(), count as f64 / total.as_secs_f64());
}

// ---- OS helpers ----

#[cfg(target_os = "linux")]
fn advise_sequential(file: &File) {
    use std::os::unix::io::AsRawFd;
    unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_SEQUENTIAL); }
}
#[cfg(not(target_os = "linux"))]
fn advise_sequential(_: &File) {}

#[cfg(target_os = "linux")]
fn read_release(reader: &BufReader<File>, offset: u64, end: u64) {
    use std::os::unix::io::AsRawFd;
    let fd = reader.get_ref().as_raw_fd();
    unsafe {
        libc::posix_fadvise(fd, offset as i64, (end - offset) as i64, libc::POSIX_FADV_DONTNEED);
    }
}
#[cfg(not(target_os = "linux"))]
fn read_release(_: &BufReader<File>, _: u64, _: u64) {}

#[cfg(target_os = "linux")]
fn write_release(writer: &BufWriter<File>, last_wb: &mut u64, last_dn: &mut u64, bytes_written: u64) {
    use std::os::unix::io::AsRawFd;
    let fd = writer.get_ref().as_raw_fd();
    unsafe {
        if *last_dn < *last_wb {
            libc::sync_file_range(fd, *last_dn as i64, (*last_wb - *last_dn) as i64,
                libc::SYNC_FILE_RANGE_WAIT_BEFORE);
            libc::posix_fadvise(fd, *last_dn as i64, (*last_wb - *last_dn) as i64,
                libc::POSIX_FADV_DONTNEED);
            *last_dn = *last_wb;
        }
        if bytes_written > *last_wb {
            libc::sync_file_range(fd, *last_wb as i64, (bytes_written - *last_wb) as i64,
                libc::SYNC_FILE_RANGE_WRITE);
            *last_wb = bytes_written;
        }
    }
}
#[cfg(not(target_os = "linux"))]
fn write_release(_: &BufWriter<File>, _: &mut u64, _: &mut u64, _: u64) {}
