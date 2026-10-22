// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! **What a per-read path calls across the crate boundary inlines**
//! (`docs/design/srd-crate-boundaries.md` PR-2). vectordata's storage
//! reads a cache copy through these methods on every record read; the
//! release profile has no LTO, so a non-generic method here is inlined
//! into vectordata only when it says so. Moving the cache files out of
//! vectordata must not turn a pointer add into a call.

use std::path::Path;

/// The per-read methods of each cache file and source.
const HOT: &[(&str, &[&str])] = &[
    ("files.rs", &["mapped", "read_bytes", "mmap_base", "mmap_slice", "total_size", "is_complete", "is_local"]),
    ("verified.rs", &["content_size", "chunk_size", "total_chunks"]),
    ("chunk_store.rs", &["total_size"]),
];

/// Whether every `pub fn name` in `text` carries `#[inline]`, and how
/// many there are.
fn inline_everywhere(text: &str, name: &str) -> (usize, Vec<usize>) {
    let lines: Vec<&str> = text.lines().collect();
    let mut seen = 0;
    let mut missing = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        let Some(rest) = t.strip_prefix("pub fn ") else { continue };
        if !rest.starts_with(&format!("{name}(")) {
            continue;
        }
        seen += 1;
        let inline = lines[..i]
            .iter()
            .rev()
            .take_while(|l| {
                let l = l.trim_start();
                l.starts_with("///") || l.starts_with("#[")
            })
            .any(|l| l.trim_start().starts_with("#[inline"));
        if !inline {
            missing.push(i + 1);
        }
    }
    (seen, missing)
}

#[test]
fn per_read_methods_are_inlinable_across_crates() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut problems = Vec::new();
    for (file, names) in HOT {
        let text = std::fs::read_to_string(src.join(file)).unwrap();
        for name in *names {
            let (seen, missing) = inline_everywhere(&text, name);
            if seen == 0 {
                problems.push(format!("{file}: no `pub fn {name}` (update this list)"));
            }
            for line in missing {
                problems.push(format!("{file}:{line}: `pub fn {name}` lacks #[inline]"));
            }
        }
    }
    assert!(problems.is_empty(), "per-read methods without #[inline] (PR-2):\n{}", problems.join("\n"));
}
