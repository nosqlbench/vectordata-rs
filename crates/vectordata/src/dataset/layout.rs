// Copyright 2020-2025 The original authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The metadata **layout** (field schema) namespace.
//!
//! Per the facet design (see
//! `docs/design/metadata-facets-and-layout-namespace.md`), the metadata
//! field schema has two homes, both addressed by a view locator:
//!
//! - **Authoritative, slicing-proof:** a standalone `metadata_layout.slab`
//!   whose *default* namespace holds the schema (the whole file is the
//!   layout). Locator: `metadata_layout.slab` (no `#`).
//! - **Convenience copy:** a `layout` namespace *inside* the metadata
//!   content slab. Locator: `metadata_content.slab#layout`. (This copy may
//!   be dropped by slab slicing/extract, which is why the standalone file
//!   is authoritative.)
//!
//! [`read_layout_bytes`] honors whichever namespace the locator names — the
//! default namespace for a bare path, or the named namespace after `#`.
//!
//! The schema is stored as **opaque bytes** (an `anode` structural
//! fingerprint, today). `vectordata` exposes those raw bytes; each caller
//! decides whether to decode them with an `anode` implementation. The bytes
//! are written identically wherever the schema appears, so **compatibility
//! between two metadata artifacts reduces to a byte-for-byte comparison of
//! their `layout` namespaces** — the design driver behind storing it this
//! way.

use std::path::Path;

/// The slab namespace, inside a metadata slab, that carries the field
/// schema (the "layout"). Mirrors the producer-side constant in
/// `veks-pipeline`'s `gen_metadata`.
pub const LAYOUT_NAMESPACE: &str = "layout";

/// Split a view locator into `(path, namespace)`. A locator of the form
/// `path#namespace` addresses that namespace; a bare `path` addresses the
/// default namespace (`None`).
fn split_locator(locator: &str) -> (&str, Option<&str>) {
    match locator.split_once('#') {
        Some((path, ns)) if !ns.is_empty() => (path, Some(ns)),
        _ => (locator, None),
    }
}

/// Read the opaque schema bytes from a metadata-layout slab locator.
///
/// `locator` is a `path` (reads the default namespace — the standalone
/// `metadata_layout.slab` case) or a `path#namespace` locator (reads that
/// namespace — the embedded `metadata_content.slab#layout` case). The
/// schema is a single record at ordinal 0 of the addressed namespace.
///
/// Returns `Ok(Some(bytes))` when the namespace exists and carries a
/// record, `Ok(None)` when the slab has no such namespace (e.g. an index
/// with no schema of its own, or a content slab whose convenience copy was
/// dropped), and `Err` only for genuine I/O / format failures on a file
/// that *should* be a slab.
pub fn read_layout_bytes(locator: &str) -> Result<Option<Vec<u8>>, String> {
    let (path, ns) = split_locator(locator);
    read_namespace_record0(Path::new(path), ns)
}

/// Open `namespace` (default when `None`) in the slab at `path` and return
/// its ordinal-0 record.
///
/// A missing namespace is reported as `Ok(None)` (not an error): the layout
/// is optional, so its absence is a normal state, not a failure. The slab
/// is opened first to separate "file isn't a readable slab" (a real error)
/// from "slab lacks this namespace" (`None`).
fn read_namespace_record0(path: &Path, namespace: Option<&str>) -> Result<Option<Vec<u8>>, String> {
    // Validate the file is a readable slab at all — surfaces real I/O and
    // format errors rather than masking them as a missing namespace.
    slabtastic::SlabReader::open(path)
        .map_err(|e| format!("open slab '{}': {}", path.display(), e))?;

    // Probe the addressed namespace. `open_namespace` errors when a named
    // namespace is absent; on this already-validated slab that means the
    // layout simply was not written — a normal `None`.
    let reader = match slabtastic::SlabReader::open_namespace(path, namespace) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    match reader.get(0) {
        Ok(bytes) => Ok(Some(bytes)),
        // Namespace present but empty: treat as no schema.
        Err(_) => Ok(None),
    }
}

/// Whether two metadata artifacts carry **byte-identical** layout schemas.
///
/// This is the content↔results (or dataset↔dataset) compatibility test:
/// the two are compatible iff both expose a `layout` namespace and the
/// schema bytes match exactly. If *either* side has no layout namespace,
/// compatibility is **unknown** and this returns `Ok(None)` — the caller
/// decides whether a missing schema is acceptable (it is not a hard
/// mismatch, since legacy artifacts may simply predate the namespace).
///
/// `Ok(Some(true))` = both present and identical; `Ok(Some(false))` = both
/// present and differ.
pub fn layouts_compatible(a_locator: &str, b_locator: &str) -> Result<Option<bool>, String> {
    let a = read_layout_bytes(a_locator)?;
    let b = read_layout_bytes(b_locator)?;
    match (a, b) {
        (Some(a), Some(b)) => Ok(Some(a == b)),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a standalone layout slab: the schema is a single record in the
    /// *default* namespace (the whole file is the layout).
    fn write_standalone_layout(path: &Path, schema: &[u8]) {
        let cfg = slabtastic::WriterConfig::default();
        let mut w = slabtastic::SlabWriter::new(path, cfg).unwrap();
        w.add_record(schema).unwrap();
        w.finish().unwrap();
    }

    /// Write a content slab: `content` records in the default namespace,
    /// plus an optional `layout`-namespace convenience copy of `schema`.
    fn write_content_slab(path: &Path, content: &[&[u8]], schema: Option<&[u8]>) {
        let cfg = slabtastic::WriterConfig::default();
        let mut w = slabtastic::SlabWriter::new(path, cfg).unwrap();
        for rec in content {
            w.add_record(rec).unwrap();
        }
        if let Some(s) = schema {
            w.start_namespace(LAYOUT_NAMESPACE).unwrap();
            w.add_record(s).unwrap();
        }
        w.finish().unwrap();
    }

    #[test]
    fn reads_standalone_layout_via_bare_locator() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("metadata_layout.slab");
        write_standalone_layout(&p, b"SCHEMA-V1");
        // Bare path → default namespace → the schema.
        assert_eq!(read_layout_bytes(p.to_str().unwrap()).unwrap().as_deref(), Some(&b"SCHEMA-V1"[..]));
    }

    #[test]
    fn reads_embedded_layout_via_hash_locator() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("metadata_content.slab");
        write_content_slab(&p, &[b"row0", b"row1"], Some(b"SCHEMA-V1"));
        // `#layout` locator → the embedded convenience copy.
        let hashed = format!("{}#{}", p.to_str().unwrap(), LAYOUT_NAMESPACE);
        assert_eq!(read_layout_bytes(&hashed).unwrap().as_deref(), Some(&b"SCHEMA-V1"[..]));
    }

    #[test]
    fn missing_layout_namespace_is_none_not_error() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("index.slab");
        write_content_slab(&p, &[b"a", b"b"], None);
        // No `layout` namespace → None (not an error).
        let hashed = format!("{}#{}", p.to_str().unwrap(), LAYOUT_NAMESPACE);
        assert_eq!(read_layout_bytes(&hashed).unwrap(), None);
    }

    #[test]
    fn nonexistent_file_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("nope.slab");
        assert!(read_layout_bytes(p.to_str().unwrap()).is_err());
    }

    #[test]
    fn compatible_iff_layout_bytes_match() {
        let tmp = tempfile::tempdir().unwrap();
        // Standalone authoritative layout vs the content slab's embedded copy.
        let standalone = tmp.path().join("metadata_layout.slab");
        let content = tmp.path().join("metadata_content.slab");
        let other = tmp.path().join("other_layout.slab");
        write_standalone_layout(&standalone, b"SCHEMA-V1");
        write_content_slab(&content, &[b"c0"], Some(b"SCHEMA-V1"));
        write_standalone_layout(&other, b"SCHEMA-V2");

        let s = standalone.to_str().unwrap();
        let c = format!("{}#{}", content.to_str().unwrap(), LAYOUT_NAMESPACE);
        let o = other.to_str().unwrap();
        // Standalone ↔ embedded copy, identical schema → compatible.
        assert_eq!(layouts_compatible(s, &c).unwrap(), Some(true));
        // Different schema → incompatible.
        assert_eq!(layouts_compatible(s, o).unwrap(), Some(false));
    }

    #[test]
    fn compatibility_unknown_when_a_layout_is_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let standalone = tmp.path().join("metadata_layout.slab");
        let index = tmp.path().join("index.slab");
        write_standalone_layout(&standalone, b"SCHEMA-V1");
        write_content_slab(&index, &[b"i0"], None);
        let idx = format!("{}#{}", index.to_str().unwrap(), LAYOUT_NAMESPACE);
        assert_eq!(
            layouts_compatible(standalone.to_str().unwrap(), &idx).unwrap(),
            None,
        );
    }
}
