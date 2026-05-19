// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Schema descriptor for metadata slabs (the `:schema` namespace).
//!
//! Every metadata slab produced by the `convert` pipeline command
//! carries a single-record sidecar in a dedicated `schema` namespace
//! describing the fields it contains. The sidecar is a compact JSON
//! object — see [`MetadataSchema`] — and lets non-veks consumers
//! introspect a slab's structure without parsing a single MNode
//! record. veks-internal consumers ([`crate::datasets::derive`],
//! `analyze survey`) use it for planning and provenance.
//!
//! The descriptor is intentionally small and stable. Field types are
//! the wire-level MNode tag names (`"int"`, `"text"`, `"float32"`,
//! …) returned by [`veks_anode::mnode::TypeTag::name`], so any
//! tool that knows the MNode tag table can interpret them.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Slab namespace name used for schema sidecars. Co-located with the
/// content namespace (the default, `""`) inside the same `.slab`
/// file.
pub const SCHEMA_NAMESPACE: &str = "schema";

/// A single field in the imported metadata schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaField {
    /// Field name as it appears in MNode records.
    pub name: String,
    /// Wire-level MNode type tag name (`"int"`, `"text"`, `"float32"`,
    /// …). Matches `TypeTag::name()`.
    #[serde(rename = "type")]
    pub type_name: String,
    /// Whether the source column can contain nulls.
    pub nullable: bool,
}

/// Schema descriptor for a metadata slab.
///
/// Written into the slab's `:schema` namespace as a single JSON
/// record at import time. Stable wire format — bump
/// [`MetadataSchema::VERSION`] when changing the shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetadataSchema {
    /// Descriptor format version. See [`MetadataSchema::VERSION`].
    pub version: u32,
    /// Provenance string identifying the source the slab was imported
    /// from. Convention: `"<kind>:<path-or-spec>"` — for example
    /// `"parquet:/data/laion400b/*.parquet"`.
    pub source: String,
    /// Wall-clock time of the import, as Unix epoch seconds.
    pub imported_at_epoch_secs: u64,
    /// Number of records in the content namespace, when known at
    /// import time. `None` if the source row count wasn't available.
    pub record_count: Option<u64>,
    /// Ordered list of fields. Order matches the source schema.
    pub fields: Vec<SchemaField>,
}

impl MetadataSchema {
    /// Current descriptor format version. Bumped on any incompatible
    /// schema shape change.
    pub const VERSION: u32 = 1;

    /// Build a descriptor stamped with the current wall-clock time.
    pub fn new(source: impl Into<String>, fields: Vec<SchemaField>) -> Self {
        Self {
            version: Self::VERSION,
            source: source.into(),
            imported_at_epoch_secs: now_epoch_secs(),
            record_count: None,
            fields,
        }
    }

    /// Attach an authoritative record count.
    pub fn with_record_count(mut self, n: u64) -> Self {
        self.record_count = Some(n);
        self
    }

    /// Serialize to compact JSON bytes suitable for storage as a
    /// single slab record.
    pub fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("MetadataSchema must serialize")
    }

    /// Parse a descriptor from JSON bytes.
    pub fn from_json_bytes(buf: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(buf)
            .map_err(|e| format!("parse metadata schema: {e}"))
    }
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_minimal() {
        let s = MetadataSchema::new(
            "parquet:/tmp/x.parquet",
            vec![
                SchemaField {
                    name: "id".into(),
                    type_name: "int".into(),
                    nullable: false,
                },
                SchemaField {
                    name: "text".into(),
                    type_name: "text".into(),
                    nullable: true,
                },
            ],
        );
        let bytes = s.to_json_bytes();
        let parsed = MetadataSchema::from_json_bytes(&bytes).unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn record_count_roundtrip() {
        let s = MetadataSchema::new("parquet:/tmp/x", vec![]).with_record_count(42);
        let bytes = s.to_json_bytes();
        let parsed = MetadataSchema::from_json_bytes(&bytes).unwrap();
        assert_eq!(parsed.record_count, Some(42));
    }

    #[test]
    fn rejects_garbage() {
        let r = MetadataSchema::from_json_bytes(b"not json");
        assert!(r.is_err());
    }
}
