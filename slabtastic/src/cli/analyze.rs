// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `slab analyze` subcommand — display file structure and statistics.
//!
//! Displays page layout, record size statistics, page utilization, and
//! ordinal monotonicity analysis. When `--samples` or `--sample-percent`
//! is given, only a subset of pages are read from disk and the per-page
//! table is suppressed. Ordinal structure is derived from the page index
//! without additional I/O.

use std::io::Write;
use std::time::Instant;

use crate::SlabReader;

/// Simple xorshift64 pseudo-random number generator.
///
/// Deterministic and reproducible given the same seed. Used internally
/// by the alias-method sampling so that `slab analyze` results are
/// repeatable across runs.
struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new PRNG seeded with the given value.
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Return the next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a pseudo-random `f64` in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Return a pseudo-random `usize` in `[0, bound)`.
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Weighted sampling table using Vose's alias method.
///
/// Provides O(n) construction from a set of weights and O(1) per-sample
/// weighted random selection. Used to fairly sample records across pages
/// so that pages with many records do not dominate the statistics.
struct AliasTable {
    /// Probability threshold for each column (scaled to `n`).
    prob: Vec<f64>,
    /// Alias index for each column.
    alias: Vec<usize>,
    /// Number of entries (same as weights length).
    n: usize,
}

impl AliasTable {
    /// Build an alias table from the given weights.
    ///
    /// All weights must be non-negative and at least one must be positive.
    /// Panics if `weights` is empty or all-zero.
    fn new(weights: &[usize]) -> Self {
        let n = weights.len();
        assert!(n > 0, "AliasTable requires at least one weight");
        let total: f64 = weights.iter().sum::<usize>() as f64;
        assert!(total > 0.0, "AliasTable requires at least one positive weight");

        // Scaled probabilities: each entry should ideally be 1.0
        let mut prob: Vec<f64> = weights.iter().map(|&w| w as f64 * n as f64 / total).collect();
        let mut alias = vec![0usize; n];

        let mut small: Vec<usize> = Vec::new();
        let mut large: Vec<usize> = Vec::new();
        for (i, &p) in prob.iter().enumerate() {
            if p < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        while let (Some(s), Some(&l)) = (small.pop(), large.last()) {
            alias[s] = l;
            prob[l] -= 1.0 - prob[s];
            if prob[l] < 1.0 {
                large.pop();
                small.push(l);
            }
        }
        // Remaining entries get probability 1.0 (numerical cleanup)
        for &i in large.iter().chain(small.iter()) {
            prob[i] = 1.0;
        }

        Self { prob, alias, n }
    }

    /// Sample a single index according to the original weights.
    fn sample(&self, rng: &mut Rng) -> usize {
        let col = rng.next_usize(self.n);
        if rng.next_f64() < self.prob[col] {
            col
        } else {
            self.alias[col]
        }
    }
}

/// Run the `analyze` subcommand.
pub fn run(
    file: &str,
    samples: Option<usize>,
    sample_percent: Option<f64>,
    namespace: &Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = SlabReader::open_namespace(file, namespace.as_deref())?;
    let entries = reader.page_entries();
    let file_len = reader.file_len()?;
    let page_count = entries.len();

    let sampling_requested = samples.is_some() || sample_percent.is_some();

    println!("File: {file}");
    println!("File size: {file_len} bytes");
    println!("Page count: {page_count}");

    // Namespace information
    let ns_entries = SlabReader::list_namespaces(file)?;
    println!("Namespaces: {}", ns_entries.len());
    for ns in &ns_entries {
        let name_display = if ns.name.is_empty() {
            "\"\" (default)".to_string()
        } else {
            format!("\"{}\"", ns.name)
        };
        println!("  index {}: {}", ns.namespace_index, name_display);
    }

    // Ordinal structure from the page index (no page reads required).
    let mut is_monotonic = true;
    for w in entries.windows(2) {
        if w[1].start_ordinal <= w[0].start_ordinal {
            is_monotonic = false;
            break;
        }
    }

    // Determine which pages to read.
    let sample_page_count = if let Some(s) = samples {
        s.min(page_count)
    } else if let Some(pct) = sample_percent {
        ((page_count as f64 * pct / 100.0).ceil() as usize).clamp(1, page_count)
    } else {
        // Default: all pages
        page_count
    };

    let sampled_indices = sample_page_indices(page_count, sample_page_count);
    let reading_all = sampled_indices.len() == page_count;

    if !sampling_requested {
        println!();
        println!(
            "{:<6} {:>5} {:>14} {:>10} {:>10} {:>10}",
            "Page", "NS", "Start Ordinal", "Records", "Size", "Offset"
        );
        println!("{}", "-".repeat(61));
    }

    // Accumulators for sampled pages
    let mut sampled_total_records: u64 = 0;
    let mut page_sizes: Vec<u32> = Vec::new();
    let mut page_used_bytes: Vec<usize> = Vec::new();
    let mut min_ordinal: Option<i64> = None;
    let mut max_ordinal: Option<i64> = None;
    let mut has_gaps = false;
    let mut prev_end_ordinal: Option<i64> = None;

    // Content detection accumulators
    let mut sampled_bytes_non_ascii = false;
    let mut sampled_bytes_has_null = false;
    let mut sampled_bytes_has_newlines = false;

    // Per-page record sizes for alias-method sampling (only used when sampling)
    let mut per_page_record_sizes: Vec<Vec<usize>> = Vec::new();
    // All record sizes (only used when reading all pages)
    let mut record_sizes: Vec<usize> = Vec::new();

    let total_to_scan = sampled_indices.len();
    let mut pages_scanned = 0usize;
    let mut last_progress_time = Instant::now();

    let mut sampled_idx_pos = 0;
    for (i, entry) in entries.iter().enumerate() {
        if sampled_idx_pos >= sampled_indices.len() || sampled_indices[sampled_idx_pos] != i {
            // Not in the sample set — use index-only info for ordinal tracking
            // when reading all pages (otherwise skip entirely).
            if reading_all {
                // Ordinal bounds from index
                update_ordinal_bounds(entry.start_ordinal, 0, &mut min_ordinal, &mut max_ordinal);
            }
            continue;
        }
        sampled_idx_pos += 1;
        pages_scanned += 1;

        // Progress reporting: update every 500ms or on final page
        if sampling_requested {
            let now = Instant::now();
            if now.duration_since(last_progress_time).as_millis() >= 500
                || pages_scanned == total_to_scan
            {
                let pct = pages_scanned as f64 / total_to_scan as f64 * 100.0;
                eprint!(
                    "\r\x1b[2KSampling pages: {pages_scanned}/{total_to_scan} ({pct:.0}%)"
                );
                let _ = std::io::stderr().flush();
                last_progress_time = now;
            }
        }

        let page = reader.read_data_page(entry)?;
        let record_count = page.record_count();
        let start_ord = page.start_ordinal();
        let page_size = page.footer.page_size;
        let ns_index = page.footer.namespace_index;

        sampled_total_records += record_count as u64;
        page_sizes.push(page_size);

        // Compute used bytes (header + records + offsets + footer)
        let mut data_bytes: usize = 0;
        let mut page_rec_sizes: Vec<usize> = Vec::with_capacity(record_count);
        let mut content_check_count = 0usize;
        for r in 0..record_count {
            let rec = page.get_record(r).unwrap();
            let rec_len = rec.len();
            data_bytes += rec_len;
            page_rec_sizes.push(rec_len);

            // Content detection: check first 100 records for content hints
            if content_check_count < 100 {
                for &b in rec {
                    if !b.is_ascii() {
                        sampled_bytes_non_ascii = true;
                    }
                    if b == 0 {
                        sampled_bytes_has_null = true;
                    }
                    if b == b'\n' {
                        sampled_bytes_has_newlines = true;
                    }
                }
                content_check_count += 1;
            }
        }
        let overhead = 8 + (record_count + 1) * 4 + 16;
        page_used_bytes.push(data_bytes + overhead);

        if reading_all {
            // Full scan: collect all record sizes directly
            record_sizes.extend_from_slice(&page_rec_sizes);
        } else {
            // Sampling: store per-page for alias-method selection later
            per_page_record_sizes.push(page_rec_sizes);
        }

        // Ordinal tracking
        let end_ord = if record_count > 0 {
            start_ord + record_count as i64 - 1
        } else {
            start_ord
        };
        update_ordinal_bounds(start_ord, record_count, &mut min_ordinal, &mut max_ordinal);

        if reading_all {
            if let Some(prev_end) = prev_end_ordinal {
                if start_ord > prev_end + 1 {
                    has_gaps = true;
                }
            }
            if record_count > 0 {
                prev_end_ordinal = Some(end_ord);
            }
        }

        if !sampling_requested {
            println!(
                "{:<6} {:>5} {:>14} {:>10} {:>10} {:>10}",
                i, ns_index, start_ord, record_count, page_size, entry.file_offset
            );
        }
    }

    // When sampling, use the alias method to fairly select records across pages
    if !reading_all && !per_page_record_sizes.is_empty() {
        let page_weights: Vec<usize> = per_page_record_sizes.iter().map(|v| v.len()).collect();
        let total_records: usize = page_weights.iter().sum();
        if total_records > 0 {
            let alias_table = AliasTable::new(&page_weights);
            let sample_count = total_records.min(10_000);
            let mut rng = Rng::new(42);
            for _ in 0..sample_count {
                let page_idx = alias_table.sample(&mut rng);
                let rec_idx = rng.next_usize(per_page_record_sizes[page_idx].len());
                record_sizes.push(per_page_record_sizes[page_idx][rec_idx]);
            }
        }
    }

    // Clear progress line before summary output
    if sampling_requested {
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }

    // Record / ordinal summary
    println!();
    if reading_all {
        println!("Total records: {sampled_total_records}");
    } else {
        let estimated_total =
            (sampled_total_records as f64 / sampled_indices.len() as f64 * page_count as f64)
                as u64;
        println!(
            "Records in sample: {sampled_total_records} ({} pages of {page_count})",
            sampled_indices.len()
        );
        println!("Estimated total records: ~{estimated_total}");
    }
    if let (Some(min), Some(max)) = (min_ordinal, max_ordinal) {
        println!("Ordinal range: {min}..={max}");
    }

    // Ordinal monotonicity (derived from index)
    println!();
    if reading_all {
        if is_monotonic && !has_gaps {
            println!("Ordinal structure: strictly monotonic, no gaps");
        } else if is_monotonic && has_gaps {
            println!("Ordinal structure: monotonic with sparse gaps");
        } else {
            println!("Ordinal structure: NOT monotonic");
        }
    } else if is_monotonic {
        println!("Ordinal structure: monotonic (gap detection requires full scan)");
    } else {
        println!("Ordinal structure: NOT monotonic");
    }

    // Record size statistics
    if !record_sizes.is_empty() {
        println!();
        if reading_all {
            println!("Record size statistics ({} records):", record_sizes.len());
        } else {
            println!(
                "Record size statistics (alias-sampled {} records from {} pages):",
                record_sizes.len(),
                sampled_indices.len()
            );
        }
        print_stats(&record_sizes, "bytes");
    }

    // Page size statistics
    if !page_sizes.is_empty() {
        let ps: Vec<usize> = page_sizes.iter().map(|&s| s as usize).collect();
        println!();
        if reading_all {
            println!("Page size statistics ({} pages):", ps.len());
        } else {
            println!(
                "Page size statistics (sampled {} of {} pages):",
                ps.len(),
                page_count
            );
        }
        print_stats(&ps, "bytes");
    }

    // Page utilization
    if !page_used_bytes.is_empty() && !page_sizes.is_empty() {
        println!();
        println!("Page utilization:");
        let mut utils: Vec<f64> = page_used_bytes
            .iter()
            .zip(page_sizes.iter())
            .map(|(&used, &total)| used as f64 / total as f64 * 100.0)
            .collect();
        utils.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_util = utils.first().unwrap();
        let max_util = utils.last().unwrap();
        let avg_util: f64 = utils.iter().sum::<f64>() / utils.len() as f64;
        println!("  min: {min_util:.1}%  avg: {avg_util:.1}%  max: {max_util:.1}%");
    }

    // Content type detection
    if sampled_total_records > 0 {
        let content_type = if sampled_bytes_non_ascii {
            if sampled_bytes_has_null {
                "binary (null-terminated / cstrings)"
            } else {
                "binary"
            }
        } else if sampled_bytes_has_newlines {
            "text (newline-delimited)"
        } else {
            "text"
        };
        println!();
        println!("Detected content type: {content_type}");
    }

    Ok(())
}

/// Update ordinal min/max bounds.
fn update_ordinal_bounds(
    start_ord: i64,
    record_count: usize,
    min_ordinal: &mut Option<i64>,
    max_ordinal: &mut Option<i64>,
) {
    match *min_ordinal {
        None => *min_ordinal = Some(start_ord),
        Some(m) if start_ord < m => *min_ordinal = Some(start_ord),
        _ => {}
    }
    let end_ord = if record_count > 0 {
        start_ord + record_count as i64 - 1
    } else {
        start_ord
    };
    match *max_ordinal {
        None => *max_ordinal = Some(end_ord),
        Some(m) if end_ord > m => *max_ordinal = Some(end_ord),
        _ => {}
    }
}

/// Return `n` evenly-spaced indices from `0..total`.
fn sample_page_indices(total: usize, n: usize) -> Vec<usize> {
    if n >= total {
        return (0..total).collect();
    }
    if n == 0 {
        return Vec::new();
    }
    let step = total as f64 / n as f64;
    (0..n).map(|i| (i as f64 * step) as usize).collect()
}

/// Print min/avg/max and a simple histogram for a set of values.
fn print_stats(values: &[usize], unit: &str) {
    if values.is_empty() {
        return;
    }
    let mut sorted = values.to_vec();
    sorted.sort();
    let min = sorted[0];
    let max = *sorted.last().unwrap();
    let sum: usize = sorted.iter().sum();
    let avg = sum as f64 / sorted.len() as f64;

    println!("  min: {min} {unit}  avg: {avg:.1} {unit}  max: {max} {unit}");

    // Simple 5-bucket histogram
    if max > min {
        let bucket_width = ((max - min) as f64 / 5.0).ceil() as usize;
        if bucket_width > 0 {
            let mut buckets = [0usize; 5];
            for &v in &sorted {
                let idx = ((v - min) / bucket_width).min(4);
                buckets[idx] += 1;
            }
            println!("  histogram:");
            for (i, &count) in buckets.iter().enumerate() {
                let lo = min + i * bucket_width;
                let hi = if i == 4 { max } else { lo + bucket_width - 1 };
                let bar_len = (count * 40 / sorted.len()).max(if count > 0 { 1 } else { 0 });
                let bar = "#".repeat(bar_len);
                println!("    {lo:>8}..={hi:<8} [{count:>6}] {bar}");
            }
        }
    }
}
