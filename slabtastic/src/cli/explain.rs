// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! `slab explain` subcommand — render page layouts as block diagrams.
//!
//! Displays a visual block diagram of each page in a slabtastic file,
//! showing the header, records, offset array, and footer fields. Pages
//! can be filtered by page index, namespace, or ordinal range.

use std::io::Write;

use crate::constants::{FOOTER_V1_SIZE, HEADER_SIZE, PageType};
use crate::page::Page;
use crate::pages_page::PageEntry;
use crate::SlabReader;

use super::ordinal_range;

/// Run the `explain` subcommand, writing output to stdout.
pub fn run(
    file: &str,
    pages: &Option<Vec<usize>>,
    namespace: &Option<String>,
    ordinals: &Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    explain_to(&mut std::io::stdout(), file, pages, namespace, ordinals)
}

/// Render page layout diagrams for the given file to the provided writer.
///
/// When no filters are specified, all pages (data pages and the terminal
/// page) are diagrammed. Filters narrow the output:
///
/// - `pages` — show only the specified page indices (0-based, referring
///   to data pages in the pages-page index)
/// - `namespace` — show only pages matching the given namespace name
///   (currently matches by namespace index since namespace names require
///   the namespaces page)
/// - `ordinals` — show only pages whose ordinal range overlaps the
///   specified range
pub fn explain_to<W: Write>(
    w: &mut W,
    file: &str,
    pages_filter: &Option<Vec<usize>>,
    namespace: &Option<String>,
    ordinals_filter: &Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = SlabReader::open(file)?;
    let entries = reader.page_entries();
    let _file_len = reader.file_len()?;

    // Resolve namespace index filter if provided
    let ns_index_filter: Option<u8> = match namespace {
        Some(name) => {
            let ns_entries = SlabReader::list_namespaces(file)?;
            let found = ns_entries.iter().find(|e| &e.name == name);
            match found {
                Some(entry) => Some(entry.namespace_index),
                None => {
                    let available: Vec<String> = ns_entries
                        .iter()
                        .map(|e| {
                            if e.name.is_empty() {
                                format!("  index {}: (default)", e.namespace_index)
                            } else {
                                format!("  index {}: '{}'", e.namespace_index, e.name)
                            }
                        })
                        .collect();
                    return Err(format!(
                        "namespace '{}' not found. Available namespaces:\n{}",
                        name,
                        available.join("\n")
                    )
                    .into());
                }
            }
        }
        None => None,
    };

    // Parse ordinal range filter if provided
    let ord_range = match ordinals_filter {
        Some(spec) => Some(ordinal_range::parse_ordinal_range(spec)?),
        None => None,
    };

    let no_filter = pages_filter.is_none() && namespace.is_none() && ordinals_filter.is_none();

    // Render data pages
    for (i, entry) in entries.iter().enumerate() {
        // Apply page index filter
        if let Some(indices) = pages_filter
            && !indices.contains(&i) {
                continue;
            }

        let page = reader.read_data_page(entry)?;

        // Apply namespace index filter
        if let Some(ns_idx) = ns_index_filter
            && page.footer.namespace_index != ns_idx {
                continue;
            }

        // Apply ordinal range filter
        if let Some((range_start, range_end)) = ord_range {
            let page_start = page.start_ordinal();
            let page_end = page_start + page.record_count() as i64;
            // Check overlap: page range [page_start, page_end) vs filter [range_start, range_end)
            if page_end <= range_start || page_start >= range_end {
                continue;
            }
        }

        render_page_diagram(w, i, &page, entry.file_offset)?;
    }

    // Render terminal page (pages page) when no filter or showing all
    if no_filter {
        // The pages page starts after the last data page
        let pages_page_offset = if entries.is_empty() {
            0u64
        } else {
            // The pages page is at file_len - pages_page_size
            // Read from the end of file
            let last_entry = entries.last().unwrap();
            let last_page = reader.read_data_page(last_entry)?;
            (last_entry.file_offset as u64) + last_page.footer.page_size as u64
        };

        let pages_page = reader.read_page_at_offset(pages_page_offset)?;
        render_pages_page_diagram(w, &pages_page, pages_page_offset, &entries)?;
    }

    Ok(())
}

/// Width of the diagram box interior (characters between the vertical bars).
const BOX_WIDTH: usize = 50;

/// Write a horizontal rule with the given left, fill, and right characters.
fn hline<W: Write>(w: &mut W, left: char, fill: char, right: char) -> std::io::Result<()> {
    write!(w, "{left}")?;
    for _ in 0..BOX_WIDTH {
        write!(w, "{fill}")?;
    }
    writeln!(w, "{right}")
}

/// Write a line of text inside the box, left-aligned and padded.
fn boxline<W: Write>(w: &mut W, text: &str) -> std::io::Result<()> {
    let display_len = text.chars().count();
    if display_len >= BOX_WIDTH {
        writeln!(w, "\u{2502} {text} \u{2502}")
    } else {
        let padding = BOX_WIDTH - display_len;
        write!(w, "\u{2502} {text}")?;
        for _ in 0..padding - 1 {
            write!(w, " ")?;
        }
        writeln!(w, "\u{2502}")
    }
}

/// Render a block diagram for a single data page.
fn render_page_diagram<W: Write>(
    w: &mut W,
    page_index: usize,
    page: &Page,
    file_offset: i64,
) -> std::io::Result<()> {
    let footer = &page.footer;
    let record_count = page.record_count();
    let page_size = footer.page_size;

    // Compute section sizes
    let record_data_bytes: usize = page.records.iter().map(|r| r.len()).sum();
    let offset_entries = record_count + 1;
    let offset_bytes = offset_entries * 4;

    let page_type_str = match footer.page_type {
        PageType::Data => "Data",
        PageType::Pages => "Pages",
        PageType::Namespaces => "Namespaces",
        PageType::Invalid => "Invalid",
    };

    writeln!(
        w,
        "Page {page_index} ({page_type_str}) at offset {file_offset}, {page_size} bytes"
    )?;
    hline(w, '\u{250C}', '\u{2500}', '\u{2510}')?;

    // Header section
    boxline(w, "Header")?;
    boxline(w, "  magic: SLAB")?;
    boxline(w, &format!("  page_size: {page_size}"))?;
    hline(w, '\u{251C}', '\u{2500}', '\u{2524}')?;

    // Records section
    boxline(
        w,
        &format!(
            "Records ({record_count} records, {record_data_bytes} bytes)"
        ),
    )?;
    // Show individual records (up to a reasonable limit)
    let max_show = 20;
    for i in 0..record_count.min(max_show) {
        let rec = page.get_record(i).unwrap();
        let ordinal = footer.start_ordinal + i as i64;
        boxline(w, &format!("  [{i}] ordinal {ordinal}: {} bytes", rec.len()))?;
    }
    if record_count > max_show {
        boxline(
            w,
            &format!("  ... ({} more records)", record_count - max_show),
        )?;
    }
    hline(w, '\u{251C}', '\u{2500}', '\u{2524}')?;

    // Offsets section
    boxline(
        w,
        &format!("Offsets ({offset_entries} entries, {offset_bytes} bytes)"),
    )?;
    // Show offset values in groups of 4
    let mut offset = HEADER_SIZE as u32;
    let mut offsets: Vec<u32> = Vec::with_capacity(offset_entries);
    for rec in &page.records {
        offsets.push(offset);
        offset += rec.len() as u32;
    }
    offsets.push(offset); // sentinel

    let max_offset_show = 16;
    let show_count = offsets.len().min(max_offset_show);
    for chunk_start in (0..show_count).step_by(4) {
        let chunk_end = (chunk_start + 4).min(show_count);
        let entries_str: Vec<String> = (chunk_start..chunk_end)
            .map(|i| format!("[{i}]={}", offsets[i]))
            .collect();
        boxline(w, &format!("  {}", entries_str.join("  ")))?;
    }
    if offsets.len() > max_offset_show {
        boxline(
            w,
            &format!("  ... ({} more entries)", offsets.len() - max_offset_show),
        )?;
    }
    hline(w, '\u{251C}', '\u{2500}', '\u{2524}')?;

    // Footer section
    boxline(w, &format!("Footer ({FOOTER_V1_SIZE} bytes)"))?;
    boxline(w, &format!("  start_ordinal: {}", footer.start_ordinal))?;
    boxline(w, &format!("  record_count: {}", footer.record_count))?;
    boxline(w, &format!("  page_size: {page_size}"))?;
    boxline(w, &format!("  page_type: {page_type_str}"))?;
    boxline(w, &format!("  namespace_index: {}", footer.namespace_index))?;
    boxline(w, &format!("  footer_length: {}", footer.footer_length))?;
    hline(w, '\u{2514}', '\u{2500}', '\u{2518}')?;

    writeln!(w)?;
    Ok(())
}

/// Render a block diagram for the pages page (terminal index page).
fn render_pages_page_diagram<W: Write>(
    w: &mut W,
    page: &Page,
    file_offset: u64,
    entries: &[PageEntry],
) -> std::io::Result<()> {
    let footer = &page.footer;
    let record_count = page.record_count();
    let page_size = footer.page_size;
    let record_data_bytes: usize = page.records.iter().map(|r| r.len()).sum();
    let offset_entries = record_count + 1;
    let offset_bytes = offset_entries * 4;

    writeln!(
        w,
        "Pages Page at offset {file_offset}, {page_size} bytes"
    )?;
    hline(w, '\u{250C}', '\u{2500}', '\u{2510}')?;

    // Header
    boxline(w, "Header")?;
    boxline(w, "  magic: SLAB")?;
    boxline(w, &format!("  page_size: {page_size}"))?;
    hline(w, '\u{251C}', '\u{2500}', '\u{2524}')?;

    // Records (page index entries)
    boxline(
        w,
        &format!(
            "Records ({record_count} entries, {record_data_bytes} bytes)"
        ),
    )?;
    let max_show = 20;
    for (i, entry) in entries.iter().enumerate().take(max_show) {
        boxline(
            w,
            &format!(
                "  [{i}] start_ordinal={}, offset={}",
                entry.start_ordinal, entry.file_offset
            ),
        )?;
    }
    if entries.len() > max_show {
        boxline(
            w,
            &format!("  ... ({} more entries)", entries.len() - max_show),
        )?;
    }
    hline(w, '\u{251C}', '\u{2500}', '\u{2524}')?;

    // Offsets
    boxline(
        w,
        &format!("Offsets ({offset_entries} entries, {offset_bytes} bytes)"),
    )?;
    hline(w, '\u{251C}', '\u{2500}', '\u{2524}')?;

    // Footer
    boxline(w, &format!("Footer ({FOOTER_V1_SIZE} bytes)"))?;
    boxline(w, &format!("  start_ordinal: {}", footer.start_ordinal))?;
    boxline(w, &format!("  record_count: {}", footer.record_count))?;
    boxline(w, &format!("  page_size: {page_size}"))?;
    boxline(w, "  page_type: Pages")?;
    boxline(w, &format!("  namespace_index: {}", footer.namespace_index))?;
    boxline(w, &format!("  footer_length: {}", footer.footer_length))?;
    hline(w, '\u{2514}', '\u{2500}', '\u{2518}')?;

    writeln!(w)?;
    Ok(())
}
