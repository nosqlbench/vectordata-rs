// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! `slab check` subcommand — validate slabtastic file structure.
//!
//! Performs three validation passes:
//!
//! 1. **Index-driven** — iterates page entries from the pages page,
//!    reads each data page, and validates: magic bytes, page type,
//!    version, footer length, header/footer page_size agreement,
//!    page_size minimum (512), record count consistency, offset array
//!    bounds, ordinal monotonicity, and index/page ordinal agreement.
//!
//! 2. **Forward traversal** — walks the file from offset 0 using
//!    header page_size fields, validating each page structurally
//!    without relying on the index. Checks that the last page is a
//!    Pages type and that the traversal exactly covers the file length.
//!
//! 3. **Cross-check** — verifies every index entry offset appears in
//!    the forward traversal, and that all forward-traversal data pages
//!    are accounted for in the index.

use crate::constants::{FOOTER_V1_SIZE, HEADER_SIZE, MIN_PAGE_SIZE};
use crate::{PageType, SlabReader};

/// Run the `check` subcommand.
pub fn run(file: &str, namespace: &Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut errors: Vec<String> = Vec::new();

    let reader = SlabReader::open_namespace(file, namespace.as_deref())?;
    let entries = reader.page_entries();
    let file_len = reader.file_len()?;

    // -----------------------------------------------------------------------
    // Pass 1: index-driven validation
    // -----------------------------------------------------------------------
    println!("Pass 1: index-driven validation ({} page entries)", entries.len());
    let mut prev_end_ordinal: Option<i64> = None;

    for (i, entry) in entries.iter().enumerate() {
        // Validate the index entry offset is within the file
        if (entry.file_offset as u64) >= file_len {
            errors.push(format!(
                "page {i}: index entry offset {} is beyond file length {file_len}",
                entry.file_offset
            ));
            continue;
        }

        match reader.read_data_page(entry) {
            Ok(page) => {
                let footer = &page.footer;

                // Magic — already validated by Page::deserialize, but if we
                // got here the magic was correct.

                // Page type must be Data for entries referenced from the pages page
                if footer.page_type != PageType::Data {
                    errors.push(format!(
                        "page {i}: expected Data page type, got {:?}",
                        footer.page_type
                    ));
                }

                // Namespace index must be valid (1–127)
                if footer.namespace_index == 0 || footer.namespace_index >= 128 {
                    errors.push(format!(
                        "page {i}: invalid namespace index {}",
                        footer.namespace_index
                    ));
                }

                // Footer length must be >= 16 and a multiple of 16
                if footer.footer_length < FOOTER_V1_SIZE as u16 {
                    errors.push(format!(
                        "page {i}: footer_length {} < minimum {FOOTER_V1_SIZE}",
                        footer.footer_length
                    ));
                }
                if footer.footer_length % 16 != 0 {
                    errors.push(format!(
                        "page {i}: footer_length {} not a multiple of 16",
                        footer.footer_length
                    ));
                }

                // Page size minimum
                if footer.page_size < MIN_PAGE_SIZE {
                    errors.push(format!(
                        "page {i}: page_size {} below minimum {MIN_PAGE_SIZE}",
                        footer.page_size
                    ));
                }

                // Structural size: header + offsets + footer must fit in page_size
                let record_count = page.record_count();
                let min_structural_size =
                    HEADER_SIZE + ((record_count + 1) * 4) + FOOTER_V1_SIZE;
                if (footer.page_size as usize) < min_structural_size {
                    errors.push(format!(
                        "page {i}: page_size {} too small for {record_count} records \
                         (need at least {min_structural_size} bytes for header+offsets+footer)",
                        footer.page_size
                    ));
                }

                // Footer record_count must match actual records deserialized
                if footer.record_count as usize != record_count {
                    errors.push(format!(
                        "page {i}: footer record_count {} != deserialized record count {record_count}",
                        footer.record_count
                    ));
                }

                // Validate offset array bounds: every record must be within
                // the page data region [HEADER_SIZE .. page_size - offsets - footer]
                for r in 0..record_count {
                    if let Some(data) = page.get_record(r) {
                        // Record data pointer is already validated by deserialize,
                        // but check for zero-length sanity (always valid, just note)
                        let _ = data;
                    } else {
                        errors.push(format!(
                            "page {i}: record {r} out of bounds in offset array"
                        ));
                    }
                }

                // Index entry start_ordinal must agree with page footer
                if entry.start_ordinal != page.start_ordinal() {
                    errors.push(format!(
                        "page {i}: index start_ordinal {} != page footer start_ordinal {}",
                        entry.start_ordinal,
                        page.start_ordinal()
                    ));
                }

                // Ordinal monotonicity across pages
                let start = page.start_ordinal();
                if let Some(prev_end) = prev_end_ordinal
                    && start <= prev_end {
                        errors.push(format!(
                            "page {i}: start ordinal {start} overlaps or is not after \
                             previous page end ordinal {prev_end}"
                        ));
                    }
                if record_count > 0 {
                    prev_end_ordinal = Some(start + record_count as i64 - 1);
                }
            }
            Err(e) => {
                errors.push(format!("page {i}: failed to read: {e}"));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Pass 2: forward traversal
    // -----------------------------------------------------------------------
    println!("Pass 2: forward traversal");
    let mut offset: u64 = 0;
    let mut forward_pages: Vec<(u64, PageType, u32)> = Vec::new();

    while offset < file_len {
        // Check that at least a header is readable
        if offset + HEADER_SIZE as u64 > file_len {
            errors.push(format!(
                "forward: truncated header at offset {offset} \
                 ({} bytes remaining, need {HEADER_SIZE})",
                file_len - offset
            ));
            break;
        }

        match reader.read_page_at_offset(offset) {
            Ok(page) => {
                let footer = &page.footer;
                let page_size = footer.page_size;

                // Zero page_size would cause infinite loop
                if page_size == 0 {
                    errors.push(format!("forward: zero page_size at offset {offset}"));
                    break;
                }

                // Page size minimum
                if page_size < MIN_PAGE_SIZE {
                    errors.push(format!(
                        "forward: page at offset {offset} has page_size {page_size} \
                         below minimum {MIN_PAGE_SIZE}"
                    ));
                }

                // Magic — validated by deserialize, but check namespace_index/footer_length
                if footer.namespace_index == 0 || footer.namespace_index >= 128 {
                    errors.push(format!(
                        "forward: page at offset {offset} has invalid namespace index {}",
                        footer.namespace_index
                    ));
                }

                if footer.footer_length < FOOTER_V1_SIZE as u16 {
                    errors.push(format!(
                        "forward: page at offset {offset} has footer_length {} < {FOOTER_V1_SIZE}",
                        footer.footer_length
                    ));
                }

                // Page must not extend beyond file
                if offset + page_size as u64 > file_len {
                    errors.push(format!(
                        "forward: page at offset {offset} with size {page_size} \
                         extends beyond file length {file_len}"
                    ));
                    break;
                }

                forward_pages.push((offset, footer.page_type, page_size));
                offset += page_size as u64;
            }
            Err(e) => {
                errors.push(format!("forward: failed to read page at offset {offset}: {e}"));
                break;
            }
        }
    }

    // Verify forward traversal consumed the exact file length
    if offset != file_len && errors.iter().all(|e| !e.starts_with("forward:")) {
        errors.push(format!(
            "forward: traversal ended at offset {offset} but file length is {file_len} \
             ({} bytes unaccounted for)",
            file_len - offset
        ));
    }

    // Last page must be Pages or Namespaces type
    if let Some((_off, ptype, _size)) = forward_pages.last() {
        if *ptype != PageType::Pages && *ptype != PageType::Namespaces {
            errors.push(format!(
                "forward: last page is {:?}, expected Pages or Namespaces",
                ptype
            ));
        }
    } else if forward_pages.is_empty()
        && !errors.iter().any(|e| e.starts_with("forward:"))
    {
        errors.push("forward: no pages found in file".to_string());
    }

    // Only one terminal index page allowed (Pages or Namespaces, at the end)
    let pages_type_count = forward_pages
        .iter()
        .filter(|(_, pt, _)| *pt == PageType::Pages || *pt == PageType::Namespaces)
        .count();
    if pages_type_count > 1 {
        errors.push(format!(
            "forward: found {pages_type_count} Pages/Namespaces-type pages \
             (expected exactly 1 at end)"
        ));
    }

    // No Invalid-type pages allowed
    for (off, pt, _) in &forward_pages {
        if *pt == PageType::Invalid {
            errors.push(format!(
                "forward: page at offset {off} has Invalid page type"
            ));
        }
    }

    // -----------------------------------------------------------------------
    // Pass 3: cross-check index vs. forward traversal
    // -----------------------------------------------------------------------
    println!("Pass 3: cross-checking index against forward traversal");
    let forward_offsets: std::collections::HashSet<u64> =
        forward_pages.iter().map(|(off, _, _)| *off).collect();

    // Every index entry must appear in forward traversal
    for (i, entry) in entries.iter().enumerate() {
        if !forward_offsets.contains(&(entry.file_offset as u64)) {
            errors.push(format!(
                "cross-check: index entry {i} at offset {} not found in forward traversal",
                entry.file_offset
            ));
        }
    }

    // Every Data page in forward traversal must be in the index
    let index_offsets: std::collections::HashSet<u64> =
        entries.iter().map(|e| e.file_offset as u64).collect();
    for (off, pt, _) in &forward_pages {
        if *pt == PageType::Data && !index_offsets.contains(off) {
            errors.push(format!(
                "cross-check: data page at offset {off} found in forward traversal \
                 but not referenced in pages page index (orphaned page)"
            ));
        }
    }

    // Index entry count must match the number of Data pages in forward traversal
    let forward_data_count = forward_pages
        .iter()
        .filter(|(_, pt, _)| *pt == PageType::Data)
        .count();
    if entries.len() != forward_data_count {
        errors.push(format!(
            "cross-check: index has {} entries but forward traversal found \
             {forward_data_count} data pages",
            entries.len()
        ));
    }

    // -----------------------------------------------------------------------
    // Report results
    // -----------------------------------------------------------------------
    println!();
    let data_page_count = forward_pages
        .iter()
        .filter(|(_, pt, _)| *pt == PageType::Data)
        .count();
    println!(
        "Summary: {data_page_count} data page(s), {} forward page(s), \
         file size {file_len} bytes",
        forward_pages.len()
    );

    if errors.is_empty() {
        println!("OK: no errors found");
        Ok(())
    } else {
        println!("ERRORS: {} issue(s) found:", errors.len());
        for e in &errors {
            println!("  - {e}");
        }
        Err(format!("{} error(s) found", errors.len()).into())
    }
}
