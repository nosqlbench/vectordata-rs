// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Compiled MNode writer for Arrow `RecordBatch` rows.
//!
//! Analyzes an Arrow [`Schema`] once to produce a [`CompiledMnodeWriter`] — a
//! flat vector of [`FieldOp`]s that knows pre-encoded field name bytes, the
//! MNode type tag, and which Arrow column/array type to read from. At write
//! time it iterates the ops and writes MNode wire format directly to a
//! `Vec<u8>` buffer, avoiding per-row HashMap and allocation overhead.

use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Int16Type, Int32Type, Int64Type, Int8Type, Schema};

use super::mnode::TypeTag;

/// Determines how to extract a value from an Arrow column and write its MNode
/// value bytes.
#[derive(Debug)]
enum ValueWriter {
    AlwaysNull,
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Utf8,
    LargeUtf8,
    Binary,
    LargeBinary,
    FixedSizeBinary,
    TimestampMillis,
    TimestampSecs,
    TimestampMicros,
    TimestampNanos,
    Date32,
    Date64,
    DictInt8(Box<ValueWriter>),
    DictInt16(Box<ValueWriter>),
    DictInt32(Box<ValueWriter>),
    DictInt64(Box<ValueWriter>),
}

/// A single field operation: pre-encoded name prefix, type tag, column index,
/// nullable flag, and the value writer that extracts and serializes the value.
#[derive(Debug)]
struct FieldOp {
    /// Pre-encoded `[name_len: u16 LE][name_utf8: N bytes]`
    name_prefix: Vec<u8>,
    /// MNode type tag byte for non-null values
    type_tag: u8,
    /// Column index in the RecordBatch
    col_index: usize,
    /// Whether the column can contain nulls
    nullable: bool,
    /// How to read and write the value
    writer: ValueWriter,
}

/// A compiled plan for converting Arrow `RecordBatch` rows to MNode wire
/// format.
///
/// Created once via [`compile`](CompiledMnodeWriter::compile), then reused for
/// every row in the batch (and any batch with the same schema).
#[derive(Debug)]
pub struct CompiledMnodeWriter {
    /// Pre-encoded `[field_count: u16 LE]`
    field_count_header: [u8; 2],
    /// One op per schema field
    ops: Vec<FieldOp>,
}

impl CompiledMnodeWriter {
    /// Analyze an Arrow schema and produce a compiled writer.
    ///
    /// Returns `Err` if any field has an unsupported Arrow type (Struct, List,
    /// Map, Union, etc.).
    pub fn compile(schema: &Schema) -> Result<Self, String> {
        let field_count = schema.fields().len();
        let field_count_header = (field_count as u16).to_le_bytes();
        let mut ops = Vec::with_capacity(field_count);

        for (i, field) in schema.fields().iter().enumerate() {
            let name_bytes = field.name().as_bytes();
            let mut name_prefix = Vec::with_capacity(2 + name_bytes.len());
            name_prefix.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            name_prefix.extend_from_slice(name_bytes);

            let (type_tag, writer) = resolve_writer(field.data_type())?;

            ops.push(FieldOp {
                name_prefix,
                type_tag,
                col_index: i,
                nullable: field.is_nullable(),
                writer,
            });
        }

        Ok(CompiledMnodeWriter {
            field_count_header,
            ops,
        })
    }

    /// Write one row as an unframed MNode payload into `buf`.
    ///
    /// Prepends the `DIALECT_MNODE` leader byte before the field data.
    /// The caller is responsible for clearing `buf` beforehand if needed.
    pub fn write_row(&self, batch: &RecordBatch, row: usize, buf: &mut Vec<u8>) {
        buf.push(super::mnode::DIALECT_MNODE);
        buf.extend_from_slice(&self.field_count_header);
        for op in &self.ops {
            buf.extend_from_slice(&op.name_prefix);
            let col = batch.column(op.col_index);
            match &op.writer {
                ValueWriter::AlwaysNull => {
                    buf.push(TypeTag::Null as u8);
                }
                _ if op.nullable && col.is_null(row) => {
                    buf.push(TypeTag::Null as u8);
                }
                _ => {
                    buf.push(op.type_tag);
                    write_value_bytes(&op.writer, col, row, buf);
                }
            }
        }
    }

    /// Write one row as a framed MNode payload (u32 LE length prefix +
    /// payload) into `buf`.
    #[cfg(test)]
    pub fn write_row_framed(&self, batch: &RecordBatch, row: usize, buf: &mut Vec<u8>) {
        let len_pos = buf.len();
        buf.extend_from_slice(&[0u8; 4]); // placeholder
        self.write_row(batch, row, buf);
        let payload_len = (buf.len() - len_pos - 4) as u32;
        buf[len_pos..len_pos + 4].copy_from_slice(&payload_len.to_le_bytes());
    }
}

/// Resolve the MNode type tag and `ValueWriter` for an Arrow `DataType`.
fn resolve_writer(dt: &DataType) -> Result<(u8, ValueWriter), String> {
    match dt {
        DataType::Null => Ok((TypeTag::Null as u8, ValueWriter::AlwaysNull)),
        DataType::Boolean => Ok((TypeTag::Bool as u8, ValueWriter::Bool)),
        DataType::Int8 => Ok((TypeTag::Short as u8, ValueWriter::Int8)),
        DataType::Int16 => Ok((TypeTag::Short as u8, ValueWriter::Int16)),
        DataType::Int32 => Ok((TypeTag::Int32 as u8, ValueWriter::Int32)),
        DataType::Int64 => Ok((TypeTag::Int as u8, ValueWriter::Int64)),
        DataType::UInt8 => Ok((TypeTag::Int32 as u8, ValueWriter::UInt8)),
        DataType::UInt16 => Ok((TypeTag::Int32 as u8, ValueWriter::UInt16)),
        DataType::UInt32 => Ok((TypeTag::Int as u8, ValueWriter::UInt32)),
        DataType::UInt64 => Ok((TypeTag::Int as u8, ValueWriter::UInt64)),
        DataType::Float16 => Ok((TypeTag::Half as u8, ValueWriter::Float16)),
        DataType::Float32 => Ok((TypeTag::Float32 as u8, ValueWriter::Float32)),
        DataType::Float64 => Ok((TypeTag::Float as u8, ValueWriter::Float64)),
        DataType::Utf8 => Ok((TypeTag::Text as u8, ValueWriter::Utf8)),
        DataType::LargeUtf8 => Ok((TypeTag::Text as u8, ValueWriter::LargeUtf8)),
        DataType::Binary => Ok((TypeTag::Bytes as u8, ValueWriter::Binary)),
        DataType::LargeBinary => Ok((TypeTag::Bytes as u8, ValueWriter::LargeBinary)),
        DataType::FixedSizeBinary(_) => Ok((TypeTag::Bytes as u8, ValueWriter::FixedSizeBinary)),
        DataType::Timestamp(unit, _) => {
            use arrow::datatypes::TimeUnit;
            match unit {
                TimeUnit::Second => Ok((TypeTag::Millis as u8, ValueWriter::TimestampSecs)),
                TimeUnit::Millisecond => Ok((TypeTag::Millis as u8, ValueWriter::TimestampMillis)),
                TimeUnit::Microsecond => Ok((TypeTag::Millis as u8, ValueWriter::TimestampMicros)),
                TimeUnit::Nanosecond => Ok((TypeTag::Millis as u8, ValueWriter::TimestampNanos)),
            }
        }
        DataType::Date32 => Ok((TypeTag::Int32 as u8, ValueWriter::Date32)),
        DataType::Date64 => Ok((TypeTag::Millis as u8, ValueWriter::Date64)),
        DataType::Dictionary(key_type, value_type) => {
            let (tag, inner) = resolve_writer(value_type)?;
            let boxed = Box::new(inner);
            match key_type.as_ref() {
                DataType::Int8 => Ok((tag, ValueWriter::DictInt8(boxed))),
                DataType::Int16 => Ok((tag, ValueWriter::DictInt16(boxed))),
                DataType::Int32 => Ok((tag, ValueWriter::DictInt32(boxed))),
                DataType::Int64 => Ok((tag, ValueWriter::DictInt64(boxed))),
                other => Err(format!("unsupported dictionary key type: {:?}", other)),
            }
        }
        other => Err(format!("unsupported Arrow type for MNode: {:?}", other)),
    }
}

/// Write the value bytes for a single cell (no tag byte — caller writes that).
fn write_value_bytes(writer: &ValueWriter, col: &Arc<dyn Array>, row: usize, buf: &mut Vec<u8>) {
    match writer {
        ValueWriter::AlwaysNull => {}
        ValueWriter::Bool => {
            let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            buf.push(if arr.value(row) { 1 } else { 0 });
        }
        ValueWriter::Int8 => {
            let arr = col.as_any().downcast_ref::<Int8Array>().unwrap();
            buf.extend_from_slice(&(arr.value(row) as i16).to_le_bytes());
        }
        ValueWriter::Int16 => {
            let arr = col.as_any().downcast_ref::<Int16Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        ValueWriter::Int32 => {
            let arr = col.as_any().downcast_ref::<Int32Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        ValueWriter::Int64 => {
            let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        ValueWriter::UInt8 => {
            let arr = col.as_any().downcast_ref::<UInt8Array>().unwrap();
            buf.extend_from_slice(&(arr.value(row) as i32).to_le_bytes());
        }
        ValueWriter::UInt16 => {
            let arr = col.as_any().downcast_ref::<UInt16Array>().unwrap();
            buf.extend_from_slice(&(arr.value(row) as i32).to_le_bytes());
        }
        ValueWriter::UInt32 => {
            let arr = col.as_any().downcast_ref::<UInt32Array>().unwrap();
            buf.extend_from_slice(&(arr.value(row) as i64).to_le_bytes());
        }
        ValueWriter::UInt64 => {
            let arr = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        ValueWriter::Float16 => {
            let arr = col.as_any().downcast_ref::<Float16Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_bits().to_le_bytes());
        }
        ValueWriter::Float32 => {
            let arr = col.as_any().downcast_ref::<Float32Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        ValueWriter::Float64 => {
            let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        ValueWriter::Utf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
            let s = arr.value(row);
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        ValueWriter::LargeUtf8 => {
            let arr = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
            let s = arr.value(row);
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        ValueWriter::Binary => {
            let arr = col.as_any().downcast_ref::<BinaryArray>().unwrap();
            let b = arr.value(row);
            buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
            buf.extend_from_slice(b);
        }
        ValueWriter::LargeBinary => {
            let arr = col.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            let b = arr.value(row);
            buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
            buf.extend_from_slice(b);
        }
        ValueWriter::FixedSizeBinary => {
            let arr = col.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
            let b = arr.value(row);
            buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
            buf.extend_from_slice(b);
        }
        ValueWriter::TimestampMillis => {
            let arr = col.as_any().downcast_ref::<TimestampMillisecondArray>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        ValueWriter::TimestampSecs => {
            let arr = col.as_any().downcast_ref::<TimestampSecondArray>().unwrap();
            buf.extend_from_slice(&(arr.value(row) * 1000).to_le_bytes());
        }
        ValueWriter::TimestampMicros => {
            let arr = col.as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
            buf.extend_from_slice(&(arr.value(row) / 1000).to_le_bytes());
        }
        ValueWriter::TimestampNanos => {
            let arr = col.as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
            buf.extend_from_slice(&(arr.value(row) / 1_000_000).to_le_bytes());
        }
        ValueWriter::Date32 => {
            let arr = col.as_any().downcast_ref::<Date32Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        ValueWriter::Date64 => {
            let arr = col.as_any().downcast_ref::<Date64Array>().unwrap();
            buf.extend_from_slice(&arr.value(row).to_le_bytes());
        }
        ValueWriter::DictInt8(inner) => {
            let dict = col.as_any().downcast_ref::<DictionaryArray<Int8Type>>().unwrap();
            let key = dict.key(row).unwrap() as usize;
            let values = dict.values();
            write_value_bytes(inner, values, key, buf);
        }
        ValueWriter::DictInt16(inner) => {
            let dict = col.as_any().downcast_ref::<DictionaryArray<Int16Type>>().unwrap();
            let key = dict.key(row).unwrap() as usize;
            let values = dict.values();
            write_value_bytes(inner, values, key, buf);
        }
        ValueWriter::DictInt32(inner) => {
            let dict = col.as_any().downcast_ref::<DictionaryArray<Int32Type>>().unwrap();
            let key = dict.key(row).unwrap() as usize;
            let values = dict.values();
            write_value_bytes(inner, values, key, buf);
        }
        ValueWriter::DictInt64(inner) => {
            let dict = col.as_any().downcast_ref::<DictionaryArray<Int64Type>>().unwrap();
            let key = dict.key(row).unwrap() as usize;
            let values = dict.values();
            write_value_bytes(inner, values, key, buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::mnode::{MNode, MValue};
    use arrow::datatypes::Field;
    use half::f16;
    use std::sync::Arc;

    /// Helper: compile schema, write row, decode via MNode::from_bytes
    fn roundtrip_row(batch: &RecordBatch, row: usize) -> MNode {
        let writer = CompiledMnodeWriter::compile(batch.schema().as_ref()).unwrap();
        let mut buf = Vec::new();
        writer.write_row(batch, row, &mut buf);
        MNode::from_bytes(&buf).unwrap()
    }

    #[test]
    fn test_fixed_size_types_roundtrip() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("bool_col", DataType::Boolean, false),
            Field::new("i8_col", DataType::Int8, false),
            Field::new("i16_col", DataType::Int16, false),
            Field::new("i32_col", DataType::Int32, false),
            Field::new("i64_col", DataType::Int64, false),
            Field::new("u8_col", DataType::UInt8, false),
            Field::new("u16_col", DataType::UInt16, false),
            Field::new("u32_col", DataType::UInt32, false),
            Field::new("u64_col", DataType::UInt64, false),
            Field::new("f16_col", DataType::Float16, false),
            Field::new("f32_col", DataType::Float32, false),
            Field::new("f64_col", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(BooleanArray::from(vec![true])),
            Arc::new(Int8Array::from(vec![42i8])),
            Arc::new(Int16Array::from(vec![1000i16])),
            Arc::new(Int32Array::from(vec![100_000i32])),
            Arc::new(Int64Array::from(vec![9_000_000_000i64])),
            Arc::new(UInt8Array::from(vec![200u8])),
            Arc::new(UInt16Array::from(vec![60_000u16])),
            Arc::new(UInt32Array::from(vec![3_000_000_000u32])),
            Arc::new(UInt64Array::from(vec![12_345_678_901u64])),
            Arc::new(Float16Array::from(vec![f16::from_f32(1.5)])),
            Arc::new(Float32Array::from(vec![3.14f32])),
            Arc::new(Float64Array::from(vec![2.718281828f64])),
        ]).unwrap();

        let node = roundtrip_row(&batch, 0);
        assert_eq!(node.fields["bool_col"], MValue::Bool(true));
        assert_eq!(node.fields["i8_col"], MValue::Short(42));
        assert_eq!(node.fields["i16_col"], MValue::Short(1000));
        assert_eq!(node.fields["i32_col"], MValue::Int32(100_000));
        assert_eq!(node.fields["i64_col"], MValue::Int(9_000_000_000));
        assert_eq!(node.fields["u8_col"], MValue::Int32(200));
        assert_eq!(node.fields["u16_col"], MValue::Int32(60_000));
        assert_eq!(node.fields["u32_col"], MValue::Int(3_000_000_000));
        assert_eq!(node.fields["u64_col"], MValue::Int(12_345_678_901));
        assert_eq!(node.fields["f16_col"], MValue::Half(f16::from_f32(1.5).to_bits()));
        assert_eq!(node.fields["f32_col"], MValue::Float32(3.14));
        assert_eq!(node.fields["f64_col"], MValue::Float(2.718281828));
    }

    #[test]
    fn test_string_binary_roundtrip() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("utf8", DataType::Utf8, false),
            Field::new("large_utf8", DataType::LargeUtf8, false),
            Field::new("binary", DataType::Binary, false),
            Field::new("large_binary", DataType::LargeBinary, false),
            Field::new("fixed_binary", DataType::FixedSizeBinary(3), false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(StringArray::from(vec!["hello"])),
            Arc::new(LargeStringArray::from(vec!["world"])),
            Arc::new(BinaryArray::from(vec![&b"\xde\xad"[..]])),
            Arc::new(LargeBinaryArray::from(vec![&b"\xbe\xef"[..]])),
            Arc::new(FixedSizeBinaryArray::try_from_iter(
                vec![vec![1u8, 2, 3]].into_iter(),
            ).unwrap()),
        ]).unwrap();

        let node = roundtrip_row(&batch, 0);
        assert_eq!(node.fields["utf8"], MValue::Text("hello".into()));
        assert_eq!(node.fields["large_utf8"], MValue::Text("world".into()));
        assert_eq!(node.fields["binary"], MValue::Bytes(vec![0xde, 0xad]));
        assert_eq!(node.fields["large_binary"], MValue::Bytes(vec![0xbe, 0xef]));
        assert_eq!(node.fields["fixed_binary"], MValue::Bytes(vec![1, 2, 3]));
    }

    #[test]
    fn test_temporal_types() {
        use arrow::datatypes::TimeUnit;

        let schema = Arc::new(Schema::new(vec![
            Field::new("ts_s", DataType::Timestamp(TimeUnit::Second, None), false),
            Field::new("ts_ms", DataType::Timestamp(TimeUnit::Millisecond, None), false),
            Field::new("ts_us", DataType::Timestamp(TimeUnit::Microsecond, None), false),
            Field::new("ts_ns", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
            Field::new("date32", DataType::Date32, false),
            Field::new("date64", DataType::Date64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(TimestampSecondArray::from(vec![1_700_000i64])),
            Arc::new(TimestampMillisecondArray::from(vec![1_700_000_000i64])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_700_000_000_000i64])),
            Arc::new(TimestampNanosecondArray::from(vec![1_700_000_000_000_000i64])),
            Arc::new(Date32Array::from(vec![19000])),
            Arc::new(Date64Array::from(vec![1_700_000_000_000i64])),
        ]).unwrap();

        let node = roundtrip_row(&batch, 0);
        // Second → millis: ×1000
        assert_eq!(node.fields["ts_s"], MValue::Millis(1_700_000_000));
        // Millisecond → passthrough
        assert_eq!(node.fields["ts_ms"], MValue::Millis(1_700_000_000));
        // Microsecond → millis: ÷1000
        assert_eq!(node.fields["ts_us"], MValue::Millis(1_700_000_000));
        // Nanosecond → millis: ÷1_000_000
        assert_eq!(node.fields["ts_ns"], MValue::Millis(1_700_000_000));
        // Date32 as raw int32
        assert_eq!(node.fields["date32"], MValue::Int32(19000));
        // Date64 as millis
        assert_eq!(node.fields["date64"], MValue::Millis(1_700_000_000_000));
    }

    #[test]
    fn test_nullable_with_mixed_nulls() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Int64Array::from(vec![Some(100), None, Some(300)])),
        ]).unwrap();

        let node0 = roundtrip_row(&batch, 0);
        assert_eq!(node0.fields["val"], MValue::Int(100));

        let node1 = roundtrip_row(&batch, 1);
        assert_eq!(node1.fields["val"], MValue::Null);

        let node2 = roundtrip_row(&batch, 2);
        assert_eq!(node2.fields["val"], MValue::Int(300));
    }

    #[test]
    fn test_always_null_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("nothing", DataType::Null, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(NullArray::new(3)),
        ]).unwrap();

        for row in 0..3 {
            let node = roundtrip_row(&batch, row);
            assert_eq!(node.fields["nothing"], MValue::Null);
        }
    }

    #[test]
    fn test_dictionary_column() {
        let values = StringArray::from(vec!["alpha", "beta", "gamma"]);
        let keys = Int32Array::from(vec![0, 2, 1]);
        let dict = DictionaryArray::try_new(keys, Arc::new(values)).unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("label", dict.data_type().clone(), false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(dict)]).unwrap();

        let node0 = roundtrip_row(&batch, 0);
        assert_eq!(node0.fields["label"], MValue::Text("alpha".into()));

        let node1 = roundtrip_row(&batch, 1);
        assert_eq!(node1.fields["label"], MValue::Text("gamma".into()));

        let node2 = roundtrip_row(&batch, 2);
        assert_eq!(node2.fields["label"], MValue::Text("beta".into()));
    }

    #[test]
    fn test_framed_output() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(Int32Array::from(vec![42])),
        ]).unwrap();

        let writer = CompiledMnodeWriter::compile(batch.schema().as_ref()).unwrap();
        let mut buf = Vec::new();
        writer.write_row_framed(&batch, 0, &mut buf);

        // First 4 bytes are the LE length prefix
        let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert_eq!(len as usize, buf.len() - 4);

        // The payload should decode correctly
        let node = MNode::from_bytes(&buf[4..]).unwrap();
        assert_eq!(node.fields["x"], MValue::Int32(42));
    }

    #[test]
    fn test_empty_schema() {
        let schema = Arc::new(Schema::empty());
        let batch = RecordBatch::new_empty(schema);

        let writer = CompiledMnodeWriter::compile(batch.schema().as_ref()).unwrap();
        let mut buf = Vec::new();
        writer.write_row(&batch, 0, &mut buf);

        assert_eq!(buf, vec![crate::formats::mnode::DIALECT_MNODE, 0x00, 0x00]);
    }

    #[test]
    fn test_unsupported_type() {
        let schema = Schema::new(vec![
            Field::new(
                "nested",
                DataType::Struct(
                    vec![Field::new("a", DataType::Int32, false)].into(),
                ),
                false,
            ),
        ]);
        let result = CompiledMnodeWriter::compile(&schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsupported"));
    }
}
