// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Attribute types and typed column storage for predicated datasets.
//!
//! Defines `FieldType` variants (Int, Long, Enum, EnumSet) and
//! `AttributeColumn` for efficient per-vector metadata storage.

use serde::{Deserialize, Serialize};

/// Supported attribute field types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    /// 32-bit signed integer.
    Int,
    /// 64-bit signed integer.
    Long,
    /// Single-valued enumeration (stored as u32 index).
    Enum,
    /// Multi-valued enumeration (stored as a bitmask).
    EnumSet,
}

impl FieldType {
    /// Parse a field type from a string.
    pub fn from_str(s: &str) -> Option<FieldType> {
        match s.to_lowercase().as_str() {
            "int" => Some(FieldType::Int),
            "long" => Some(FieldType::Long),
            "enum" => Some(FieldType::Enum),
            "enum_set" | "enumset" => Some(FieldType::EnumSet),
            _ => None,
        }
    }
}

/// Descriptor for a single attribute field in the schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDescriptor {
    /// Field name.
    pub name: String,
    /// Field data type.
    pub field_type: FieldType,
    /// Cardinality: number of distinct values (for Int/Long: range bound,
    /// for Enum/EnumSet: number of enum labels).
    pub cardinality: u32,
    /// Enum label names (only for Enum and EnumSet types).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub enum_values: Vec<String>,
}

/// Schema for all attribute fields in a predicated dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeSchema {
    /// Ordered list of field descriptors.
    pub fields: Vec<FieldDescriptor>,
}

impl AttributeSchema {
    /// Serialize the schema to YAML.
    pub fn to_yaml(&self) -> Result<String, String> {
        serde_yaml::to_string(self).map_err(|e| format!("YAML serialization failed: {}", e))
    }

    /// Deserialize the schema from YAML.
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        serde_yaml::from_str(yaml).map_err(|e| format!("YAML parse failed: {}", e))
    }
}

/// Typed column storage for attribute values.
///
/// Each variant stores a `Vec` with one entry per base vector.
#[derive(Debug, Clone)]
pub enum AttributeColumn {
    /// 32-bit integer values.
    Int(Vec<i32>),
    /// 64-bit integer values.
    Long(Vec<i64>),
    /// Enum index values (0-based).
    Enum(Vec<u32>),
    /// EnumSet bitmask values. Each entry is a `Vec<u8>` bitmask
    /// where bit `i` indicates membership of enum value `i`.
    EnumSet(Vec<Vec<u8>>),
}

impl AttributeColumn {
    /// Number of entries in this column.
    pub fn len(&self) -> usize {
        match self {
            AttributeColumn::Int(v) => v.len(),
            AttributeColumn::Long(v) => v.len(),
            AttributeColumn::Enum(v) => v.len(),
            AttributeColumn::EnumSet(v) => v.len(),
        }
    }

    /// Get the i32 value at index `idx` (for Int columns).
    pub fn get_int(&self, idx: usize) -> Option<i32> {
        if let AttributeColumn::Int(v) = self {
            v.get(idx).copied()
        } else {
            None
        }
    }

    /// Get the i64 value at index `idx` (for Long columns).
    pub fn get_long(&self, idx: usize) -> Option<i64> {
        if let AttributeColumn::Long(v) = self {
            v.get(idx).copied()
        } else {
            None
        }
    }

    /// Get the enum index at index `idx`.
    pub fn get_enum(&self, idx: usize) -> Option<u32> {
        if let AttributeColumn::Enum(v) = self {
            v.get(idx).copied()
        } else {
            None
        }
    }

    /// Get the enum set bitmask at index `idx`.
    pub fn get_enum_set(&self, idx: usize) -> Option<&[u8]> {
        if let AttributeColumn::EnumSet(v) = self {
            v.get(idx).map(|b| b.as_slice())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_type_parse() {
        assert_eq!(FieldType::from_str("int"), Some(FieldType::Int));
        assert_eq!(FieldType::from_str("Long"), Some(FieldType::Long));
        assert_eq!(FieldType::from_str("ENUM"), Some(FieldType::Enum));
        assert_eq!(FieldType::from_str("enum_set"), Some(FieldType::EnumSet));
        assert_eq!(FieldType::from_str("enumset"), Some(FieldType::EnumSet));
        assert_eq!(FieldType::from_str("unknown"), None);
    }

    #[test]
    fn test_attribute_schema_yaml_roundtrip() {
        let schema = AttributeSchema {
            fields: vec![
                FieldDescriptor {
                    name: "age".to_string(),
                    field_type: FieldType::Int,
                    cardinality: 100,
                    enum_values: vec![],
                },
                FieldDescriptor {
                    name: "color".to_string(),
                    field_type: FieldType::Enum,
                    cardinality: 5,
                    enum_values: vec![
                        "red".to_string(),
                        "green".to_string(),
                        "blue".to_string(),
                        "yellow".to_string(),
                        "purple".to_string(),
                    ],
                },
            ],
        };

        let yaml = schema.to_yaml().unwrap();
        let parsed = AttributeSchema::from_yaml(&yaml).unwrap();
        assert_eq!(parsed.fields.len(), 2);
        assert_eq!(parsed.fields[0].name, "age");
        assert_eq!(parsed.fields[1].field_type, FieldType::Enum);
        assert_eq!(parsed.fields[1].enum_values.len(), 5);
    }

    #[test]
    fn test_attribute_column_accessors() {
        let col = AttributeColumn::Int(vec![10, 20, 30]);
        assert_eq!(col.len(), 3);
        assert_eq!(col.get_int(1), Some(20));
        assert_eq!(col.get_long(0), None);

        let col = AttributeColumn::Enum(vec![0, 2, 1]);
        assert_eq!(col.get_enum(2), Some(1));

        let col = AttributeColumn::EnumSet(vec![vec![0b101], vec![0b010]]);
        assert_eq!(col.get_enum_set(0), Some([0b101].as_slice()));
    }
}
