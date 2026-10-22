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
//! Three descriptor flavors share the namespace:
//!
//! - **[`MetadataSchema`]** (`kind: "metadata"`) — emitted by
//!   `convert` for metadata slabs (MNode content). Lists field
//!   names, types, and nullability.
//! - **[`PredicateSchema`]** (`kind: "predicate"`) — emitted by
//!   `generate predicates` for predicate slabs (PNode content).
//!   Records the template in PNode `Display` vernacular plus the
//!   generation parameters that produced the records.
//! - **[`VectorSchema`]** (`kind: "vector"`) — emitted by veks-io's
//!   slab writer for a slab of vectors. States the element type of
//!   every record, which nothing else in a slab says.
//!
//! Consumers read the namespace and dispatch on the top-level
//! `kind` field. Older slabs without `kind` are assumed to be
//! `"metadata"` for backwards compatibility. A slab with no
//! `vector` descriptor declares no element type: its records are
//! bytes.
//!
//! The descriptors are intentionally small and stable. Bump
//! their `VERSION` constant on any incompatible shape change.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use veks_datatypes::ElementType;

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

/// Slab namespace of a predicate facet holding, per predicate ordinal,
/// what the predicate *is*: its family, the selectivity it was
/// measured at, and for topical predicates the topic level and query
/// placement. Written by `generate predicates --strategy stratified`.
/// Absent from every predicate set written before it, and absence
/// means one unlabelled family rather than none — the `forms`
/// precedent.
pub const FAMILIES_NAMESPACE: &str = "families";

/// Slab namespace of a predicate facet holding, per predicate ordinal,
/// why the predicate was *selected*: the cell it filled, the pool it
/// was drawn from, the census source and the expected match count.
/// Kept apart from `families` so a consumer can read what a predicate
/// is without also reading the experimental design.
pub const GENERATION_NAMESPACE: &str = "generation";

/// `kind` discriminator value for metadata-slab schemas.
pub const SCHEMA_KIND_METADATA: &str = "metadata";

/// `kind` discriminator value for predicate-slab schemas.
pub const SCHEMA_KIND_PREDICATE: &str = "predicate";

/// `kind` discriminator value for vector-slab schemas.
pub const SCHEMA_KIND_VECTOR: &str = "vector";

fn default_metadata_kind() -> String { SCHEMA_KIND_METADATA.into() }

/// The `kind` of the descriptor in a `:schema` sidecar record: the
/// top-level `kind` field, or [`SCHEMA_KIND_METADATA`] for a
/// descriptor written before the discriminator existed. An error when
/// the record is not a JSON object.
///
/// ```
/// use veks_apicore::metadata_schema::{sidecar_kind, VectorSchema};
/// use veks_datatypes::ElementType;
/// let vector = VectorSchema::new(ElementType::U8).to_json_bytes().unwrap();
/// assert_eq!(sidecar_kind(&vector).unwrap(), "vector");
/// assert_eq!(sidecar_kind(br#"{"version":1,"fields":[]}"#).unwrap(), "metadata");
/// assert!(sidecar_kind(b"not json").is_err());
/// ```
pub fn sidecar_kind(buf: &[u8]) -> Result<String, String> {
    let value: serde_json::Value =
        serde_json::from_slice(buf).map_err(|e| format!("parse slab schema: {e}"))?;
    let object = value.as_object().ok_or("parse slab schema: not a JSON object")?;
    Ok(object
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap_or(SCHEMA_KIND_METADATA)
        .to_string())
}

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
    /// `"parquet:/data/embeddings/*.parquet"`.
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
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|e| format!("serialize the metadata schema: {e}"))
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
/// `veks_anode::pnode::from_display` — can recover the
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
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|e| format!("serialize the predicate schema: {e}"))
    }

    /// Parse a descriptor from JSON bytes.
    pub fn from_json_bytes(buf: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(buf)
            .map_err(|e| format!("parse predicate schema: {e}"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VectorSchema — descriptor for vector slabs
// ─────────────────────────────────────────────────────────────────────────────

/// Schema descriptor for a slab of vectors.
///
/// Written into the slab's `:schema` namespace as a single JSON
/// record by the slab writer, so a reader can report the records'
/// real element type and width. A slab without one — every slab
/// written before this descriptor existed, and every metadata or
/// predicate slab — declares no element type, and its records read
/// as bytes.
///
/// ```
/// use veks_apicore::metadata_schema::{VectorSchema, SCHEMA_KIND_VECTOR};
/// use veks_datatypes::ElementType;
///
/// let schema = VectorSchema::new(ElementType::BF16);
/// let bytes = schema.to_json_bytes().unwrap();
/// assert_eq!(std::str::from_utf8(&bytes).unwrap(), r#"{"version":1,"kind":"vector","element_type":"bf16"}"#);
/// let parsed = VectorSchema::from_json_bytes(&bytes).unwrap();
/// assert_eq!((parsed.kind.as_str(), parsed.element_type), (SCHEMA_KIND_VECTOR, ElementType::BF16));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorSchema {
    /// Descriptor format version. See [`VectorSchema::VERSION`].
    pub version: u32,
    /// Top-level discriminator — always `"vector"`.
    pub kind: String,
    /// The element type of every record, spelled by
    /// [`ElementType::name`] (`"u8"`, `"f32"`, `"bf16"`).
    #[serde(with = "element_type_name")]
    pub element_type: ElementType,
}

impl VectorSchema {
    /// Current descriptor format version. Bumped on any
    /// incompatible shape change.
    pub const VERSION: u32 = 1;

    /// A descriptor declaring `element_type`.
    pub fn new(element_type: ElementType) -> Self {
        Self { version: Self::VERSION, kind: SCHEMA_KIND_VECTOR.into(), element_type }
    }

    /// Serialize to compact JSON bytes suitable for storage as a
    /// single slab record.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|e| format!("serialize the vector schema: {e}"))
    }

    /// Parse a descriptor from JSON bytes. An element type that is
    /// not one of [`ElementType::ALL`] is an error, not a guess.
    pub fn from_json_bytes(buf: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(buf)
            .map_err(|e| format!("parse vector schema: {e}"))
    }
}

/// [`ElementType`] as its [`name`](ElementType::name) in JSON.
mod element_type_name {
    use serde::{Deserialize, Deserializer, Serializer};
    use veks_datatypes::ElementType;

    pub fn serialize<S: Serializer>(ty: &ElementType, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(ty.name())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ElementType, D::Error> {
        let name = String::deserialize(d)?;
        ElementType::from_name(&name)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown element type '{name}'")))
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
        let bytes = s.to_json_bytes().unwrap();
        let parsed = MetadataSchema::from_json_bytes(&bytes).unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn record_count_roundtrip() {
        let s = MetadataSchema::new("parquet:/tmp/x", vec![]).with_record_count(42);
        let bytes = s.to_json_bytes().unwrap();
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
        let bytes = p.to_json_bytes().unwrap();
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
        let m_json = m.to_json_bytes().unwrap();
        let p_json = p.to_json_bytes().unwrap();
        assert!(std::str::from_utf8(&m_json).unwrap().contains("\"kind\":\"metadata\""));
        assert!(std::str::from_utf8(&p_json).unwrap().contains("\"kind\":\"predicate\""));
    }

    /// Every element type round-trips through a vector descriptor
    /// by its name.
    #[test]
    fn vector_schema_round_trips_every_element_type() {
        for &ty in ElementType::ALL {
            let v = VectorSchema::new(ty);
            let bytes = v.to_json_bytes().unwrap();
            assert!(std::str::from_utf8(&bytes).unwrap().contains(&format!("\"element_type\":\"{}\"", ty.name())), "{ty}");
            assert_eq!(VectorSchema::from_json_bytes(&bytes).unwrap(), v, "{ty}");
        }
    }

    /// A vector descriptor naming no known element type, or naming
    /// none, does not parse: the declaration is there and unreadable,
    /// and reading it as some type would be a guess.
    #[test]
    fn vector_schema_rejects_an_unknown_or_missing_element_type() {
        assert!(VectorSchema::from_json_bytes(br#"{"version":1,"kind":"vector","element_type":"f128"}"#).is_err());
        assert!(VectorSchema::from_json_bytes(br#"{"version":1,"kind":"vector"}"#).is_err());
    }
}
