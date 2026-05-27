// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Schema descriptors for slab files (the `:schema` namespace).
//!
//! Every typed slab produced by the toolchain carries a single-
//! record sidecar in a dedicated `schema` namespace describing
//! what's in the content namespace. The sidecar is a compact JSON
//! object that lets non-veks consumers introspect a slab's
//! structure without parsing a single content record.
//!
//! Two descriptor flavors share the namespace:
//!
//! - **[`MetadataSchema`]** (`kind: "metadata"`) — emitted by
//!   `convert` for metadata slabs (MNode content). Lists field
//!   names, types, and nullability.
//! - **[`PredicateSchema`]** (`kind: "predicate"`) — emitted by
//!   `generate predicates` for predicate slabs (PNode content).
//!   Records the template in PNode `Display` vernacular plus the
//!   generation parameters that produced the records.
//!
//! Consumers read the namespace and dispatch on the top-level
//! `kind` field. Older slabs without `kind` are assumed to be
//! `"metadata"` for backwards compatibility.
//!
//! The descriptors are intentionally small and stable. Bump
//! their `VERSION` constant on any incompatible shape change.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Slab namespace name used for schema sidecars. Co-located with the
/// content namespace (the default, `""`) inside the same `.slab`
/// file.
pub const SCHEMA_NAMESPACE: &str = "schema";

/// Slab namespace name used for embedded survey reports. A
/// predicate slab produced by `generate predicates` carries the
/// full `SurveyReport` JSON it was generated against, so any
/// downstream consumer can introspect the metadata distributions
/// without needing the original survey JSON. Single record per
/// slab; payload is the raw `SurveyReport` JSON bytes.
pub const SURVEY_NAMESPACE: &str = "survey";

/// `kind` discriminator value for metadata-slab schemas.
pub const SCHEMA_KIND_METADATA: &str = "metadata";

/// `kind` discriminator value for predicate-slab schemas.
pub const SCHEMA_KIND_PREDICATE: &str = "predicate";

fn default_metadata_kind() -> String { SCHEMA_KIND_METADATA.into() }

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
    /// Top-level discriminator — always `"metadata"`. Marked
    /// `#[serde(default = ...)]` so pre-discriminator schemas
    /// (written before the `kind` field existed) still parse as
    /// metadata when read back.
    #[serde(default = "default_metadata_kind")]
    pub kind: String,
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
            kind: SCHEMA_KIND_METADATA.into(),
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

// ─────────────────────────────────────────────────────────────────────────────
// PredicateSchema — descriptor for predicate slabs
// ─────────────────────────────────────────────────────────────────────────────

/// Schema descriptor for a predicate slab.
///
/// Written into the slab's `:schema` namespace as a single JSON
/// record at predicate-generation time. The content namespace of
/// the same slab carries `PNode::to_bytes_named()` records — see
/// the `wire_format` field for the exact identifier.
///
/// Consumers read the schema namespace and dispatch on `kind` to
/// know what they're looking at. The `template` field carries the
/// originating `PNode::Display` vernacular string (with `?`
/// placeholders where comparand values were drawn from a survey)
/// so any tool that can parse the Display grammar — see
/// [`veks_anode::pnode::from_display`] — can recover the
/// structural shape of every record without scanning the content
/// namespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateSchema {
    /// Descriptor format version. See [`PredicateSchema::VERSION`].
    pub version: u32,
    /// Top-level discriminator — always `"predicate"`.
    pub kind: String,
    /// Wire format of records in the content namespace. Stable
    /// identifier — currently always `"pnode:named"` (the named-
    /// field-reference binary form `PNode::to_bytes_named`
    /// produces). Reserved for a future indexed-mode encoding.
    pub wire_format: String,
    /// Generation template in `PNode::Display` vernacular with
    /// `?` placeholders for survey-filled comparands. Round-trips
    /// through `veks_anode::pnode::from_display::from_display` for
    /// the concrete-value form, and through the templating
    /// parser in `gen_predicates_proto::parse_template` for the
    /// `?`-bearing form.
    pub template: String,
    /// Selectivity spec used at generation time. Either a scalar
    /// (`"0.10"`) or an interval (`"0.05..0.20"`). Mirrors the
    /// proto YAML's `selectivity:` field for replay parity.
    pub selectivity: String,
    /// RNG seed pinned for reproducible re-runs.
    pub seed: u64,
    /// Number of predicates in the content namespace.
    pub count: u64,
    /// Wall-clock time of generation, as Unix epoch seconds.
    pub created_at_epoch_secs: u64,
}

impl PredicateSchema {
    /// Current descriptor format version. Bumped on any
    /// incompatible shape change.
    pub const VERSION: u32 = 1;

    /// Build a descriptor stamped with the current wall-clock time.
    /// `template` should be the PNode-Display string with `?`
    /// placeholders that drove generation.
    pub fn new(template: impl Into<String>, selectivity: impl Into<String>, seed: u64, count: u64) -> Self {
        Self {
            version: Self::VERSION,
            kind: SCHEMA_KIND_PREDICATE.into(),
            wire_format: "pnode:named".into(),
            template: template.into(),
            selectivity: selectivity.into(),
            seed,
            count,
            created_at_epoch_secs: now_epoch_secs(),
        }
    }

    /// Serialize to compact JSON bytes suitable for storage as a
    /// single slab record.
    pub fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("PredicateSchema must serialize")
    }

    /// Parse a descriptor from JSON bytes.
    pub fn from_json_bytes(buf: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(buf)
            .map_err(|e| format!("parse predicate schema: {e}"))
    }
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

    /// `kind` defaults to `"metadata"` when absent, so descriptors
    /// written before the discriminator existed still parse back
    /// correctly.
    #[test]
    fn metadata_kind_defaults_on_legacy_record() {
        let legacy = br#"{"version":1,"source":"parquet:/a","imported_at_epoch_secs":1,"record_count":null,"fields":[]}"#;
        let s = MetadataSchema::from_json_bytes(legacy).unwrap();
        assert_eq!(s.kind, SCHEMA_KIND_METADATA);
    }

    #[test]
    fn predicate_schema_round_trip() {
        let p = PredicateSchema::new(
            "(age >= ? AND name MATCHES ?)",
            "0.05..0.20",
            42,
            500,
        );
        let bytes = p.to_json_bytes();
        let parsed = PredicateSchema::from_json_bytes(&bytes).unwrap();
        assert_eq!(parsed, p);
        assert_eq!(parsed.kind, SCHEMA_KIND_PREDICATE);
        assert_eq!(parsed.wire_format, "pnode:named");
    }

    /// The two descriptor shapes live in the same namespace; a
    /// consumer must read the top-level `kind` to know which to
    /// expect. Confirm both shapes carry the discriminator and
    /// neither can be silently confused with the other.
    #[test]
    fn metadata_and_predicate_discriminate_by_kind() {
        let m = MetadataSchema::new("parquet:/x", vec![]);
        let p = PredicateSchema::new("age = ?", "0.1", 1, 1);
        let m_json = m.to_json_bytes();
        let p_json = p.to_json_bytes();
        assert!(std::str::from_utf8(&m_json).unwrap().contains("\"kind\":\"metadata\""));
        assert!(std::str::from_utf8(&p_json).unwrap().contains("\"kind\":\"predicate\""));
    }
}
