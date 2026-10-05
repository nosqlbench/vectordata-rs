// Copyright 2026 Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `slab namespaces` subcommand — list all namespaces in a slab file.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::constants::{FOOTER_V1_SIZE, HEADER_SIZE};
use crate::footer::Footer;
use crate::namespaces_page::NamespacesPage;
use crate::{PageType, SlabError};

/// Run the `namespaces` subcommand.
///
/// Lists all namespaces in the given slab file. If the file ends with a
/// pages page (single namespace), reports just the default namespace.
pub fn run(file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = File::open(file)?;
    let file_len = f.seek(SeekFrom::End(0))?;

    if file_len < (HEADER_SIZE + FOOTER_V1_SIZE) as u64 {
        return Err(SlabError::TruncatedPage {
            expected: HEADER_SIZE + FOOTER_V1_SIZE,
            actual: file_len as usize,
        }
        .into());
    }

    // Read the last footer
    f.seek(SeekFrom::End(-(FOOTER_V1_SIZE as i64)))?;
    let mut footer_buf = [0u8; FOOTER_V1_SIZE];
    f.read_exact(&mut footer_buf)?;
    let footer = Footer::read_from(&footer_buf)?;

    println!("File: {file}");
    println!();

    if footer.page_type == PageType::Pages {
        // Single-namespace file
        println!("Namespaces: 1 (default only)");
        println!();
        println!(
            "{:<6} {:<10} {:<20} {:>10}",
            "Index", "Name", "Description", "Pages Offset"
        );
        println!("{}", "-".repeat(50));
        let pp_offset = file_len - footer.page_size as u64;
        println!(
            "{:<6} {:<10} {:<20} {:>10}",
            1, "\"\"", "default namespace", pp_offset
        );
    } else if footer.page_type == PageType::Namespaces {
        // Read the namespaces page
        let ns_page_offset = file_len - footer.page_size as u64;
        f.seek(SeekFrom::Start(ns_page_offset))?;
        let mut ns_buf = vec![0u8; footer.page_size as usize];
        f.read_exact(&mut ns_buf)?;

        let ns_page = NamespacesPage::deserialize(&ns_buf)?;
        let entries = ns_page.entries()?;

        println!("Namespaces: {}", entries.len());
        println!();
        println!(
            "{:<6} {:<20} {:>14}",
            "Index", "Name", "Pages Offset"
        );
        println!("{}", "-".repeat(44));

        for entry in &entries {
            let name_display = if entry.name.is_empty() {
                "\"\" (default)".to_string()
            } else {
                format!("\"{}\"", entry.name)
            };
            println!(
                "{:<6} {:<20} {:>14}",
                entry.namespace_index, name_display, entry.pages_page_offset
            );
        }
    } else {
        return Err(format!(
            "file does not end with a Pages or Namespaces page (found {:?})",
            footer.page_type
        )
        .into());
    }

    Ok(())
}
