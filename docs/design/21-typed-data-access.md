# 21 — Typed Data Access

The `vectordata` crate provides typed access to vector and scalar data files
with compile-time type safety and runtime width validation. This document
describes the access patterns, type negotiation rules, and integration with
the dataset profile system.

---

## 21.1 Supported File Formats

### Scalar formats (flat packed, no header)

Each file is a contiguous array of fixed-size elements. Ordinal N is at
byte offset `N × element_size`. No dimension header.

| Extension | Element type | Size | Example |
|-----------|-------------|------|---------|
| `.u8`     | uint8       | 1 B  | Metadata field values 0..255 |
| `.i8`     | int8        | 1 B  | Signed byte values -128..127 |
| `.u16`    | uint16      | 2 B  | Wider unsigned integers |
| `.i16`    | int16       | 2 B  | Signed 16-bit values |
| `.u32`    | uint32      | 4 B  | Unsigned 32-bit |
| `.i32`    | int32       | 4 B  | Signed 32-bit |
| `.u64`    | uint64      | 8 B  | Unsigned 64-bit |
| `.i64`    | int64       | 8 B  | Signed 64-bit |

### Vector formats (xvec, `[dim:i32, data...]` per record)

Each record has a 4-byte little-endian dimension header followed by
`dim` elements. Multiple records are concatenated.

| Extension | Element type | Legacy alias |
|-----------|-------------|-------------|
| `.fvec`   | float32     | —           |
| `.dvec`   | float64     | —           |
| `.mvec`   | float16     | —           |
| `.bvec`   | uint8       | `.u8vec`    |
| `.ivec`   | int32       | `.i32vec`   |
| `.svec`   | int16       | `.i16vec`   |
| `.i8vec`  | int8        | —           |
| `.u16vec` | uint16      | —           |
| `.u32vec` | uint32      | —           |
| `.i64vec` | int64       | —           |
| `.u64vec` | uint64      | —           |

---

## 21.2 TypedReader — Standalone File Access

The simplest way to read typed data is via `TypedReader<T>`, which opens
a file and validates that the requested type T is compatible with the
native element type.

### Opening with the native type (zero-copy)

```rust
use vectordata::typed_access::TypedReader;

let reader = TypedReader::<u8>::open("metadata.u8")?;
assert_eq!(reader.count(), 1_000_000);
assert_eq!(reader.dim(), 1);           // scalar = always 1

// Zero-copy access — no conversion, no allocation
let val: u8 = reader.get_native(42);
let slice: &[u8] = reader.get_native_slice(42);
```

### Opening with a wider type (widening, always succeeds)

```rust
// File is .u8 (1 byte) but client wants i32 (4 bytes)
let reader = TypedReader::<i32>::open("metadata.u8")?;
let val: i32 = reader.get_value(42)?;  // 0..255 → i32, always fits
```

Widening conversions never fail — every value in the narrower type
fits in the wider type.

### Same-width cross-sign (checked per value)

```rust
// File is .u8 but client wants i8 (same width, different signedness)
let reader = TypedReader::<i8>::open("metadata.u8")?;
let val = reader.get_value(0)?;   // Ok(42) if value ≤ 127
let err = reader.get_value(3);    // Err(ValueOverflow) if value = 200
```

Cross-sign access is allowed at open time but checked per value. A u8
value of 200 cannot be represented as i8, so the access returns an error
for that specific ordinal.

### Narrowing (rejected at open time)

```rust
// File is .i32 (4 bytes) — cannot open as u8 (1 byte)
let err = TypedReader::<u8>::open("data.i32");
assert!(err.is_err()); // Err(Narrowing { native: I32, target: "u8" })
```

Narrowing is rejected immediately — no data is read, no conversion
is attempted.

### Vector records (xvec)

```rust
let reader = TypedReader::<i32>::open("indices.ivec")?;
assert_eq!(reader.dim(), 100);    // 100 neighbors per query
assert_eq!(reader.count(), 10000); // 10000 queries

let record: Vec<i32> = reader.get_record(42)?;
assert_eq!(record.len(), 100);
```

---

## 21.3 Type Negotiation Rules

The width rule is: `size_of::<T>() >= native_element_size`. The access
layer does NOT enforce signedness at open time — only at value access time.

### Conversion table

| Native → Target | Same type | Same width, cross-sign | Wider | Narrower |
|-----------------|-----------|----------------------|-------|----------|
| **Open**        | OK        | OK                   | OK    | **Error** |
| **Access**      | zero-copy | checked per-value    | convert per-value | — |
| **Failure mode** | —        | ValueOverflow on specific ordinal | — | Narrowing at open |

### Concrete examples

```
u8(127) → i8  = Ok(127)      # fits in [-128, 127]
u8(200) → i8  = Err           # 200 > 127
i8(-1)  → u8  = Err           # negative
u8(255) → i32 = Ok(255)       # widening, always fits
i32(42) → u8  = Err at open   # narrowing rejected
u16(0)  → i16 = Ok(0)         # cross-sign, fits
u16(40000) → i16 = Err        # 40000 > 32767
```

---

## 21.4 ElementType — Interrogation Before Opening

Before opening a file, clients can query the native element type:

```rust
use vectordata::typed_access::ElementType;

let etype = ElementType::from_path("metadata.u8")?;
assert_eq!(etype, ElementType::U8);
assert_eq!(etype.byte_width(), 1);
assert_eq!(etype.name(), "u8");

// Also works for vector formats
let etype = ElementType::from_path("data.ivec")?;
assert_eq!(etype, ElementType::I32);

// Extension detection
let etype = ElementType::from_extension("u16vec");
assert_eq!(etype, Some(ElementType::U16));
```

---

## 21.5 Dataset Profile Access

For datasets loaded via `TestDataGroup`, facets are accessed through
profiles. The typed access layer integrates with the profile system:

### Interrogate facet type through profile

```rust
use vectordata::{TestDataGroup, TestDataView};
use vectordata::typed_access::ElementType;

let group = TestDataGroup::load("./sift1m/")?;
let view = group.profile("default").unwrap();

// Query the native type of any facet
let meta_type = view.facet_element_type("metadata_content")?;
println!("metadata is: {}", meta_type);  // "u8"

let base_type = view.facet_element_type("base_vectors")?;
println!("base vectors: {}", base_type); // "f32"

let indices_type = view.facet_element_type("predicate_results")?;
println!("results: {}", indices_type);   // "i32"
```

### Open facets with typed readers

Use `generic_view()` to get the concrete view type, which provides
`open_facet_typed::<T>()`:

```rust
let group = TestDataGroup::load("./sift1m/")?;
let view = group.generic_view("default").unwrap();

// Open metadata as native u8
let meta = view.open_facet_typed::<u8>("metadata_content")?;
for i in 0..meta.count().min(5) {
    println!("meta[{}] = {}", i, meta.get_native(i));
}

// Open predicates as wider i32
let pred = view.open_facet_typed::<i32>("metadata_predicates")?;
for i in 0..pred.count().min(5) {
    println!("pred[{}] = {}", i, pred.get_value(i)?);
}
```

### Runtime dispatch with native arms

When the client doesn't know the type at compile time but wants to
use the most efficient access path:

```rust
let group = TestDataGroup::load("./dataset/")?;
let view = group.generic_view("default").unwrap();

let etype = view.facet_element_type("metadata_content")?;
let total: i64 = match etype {
    ElementType::U8 => {
        let r = view.open_facet_typed::<u8>("metadata_content")?;
        (0..r.count()).map(|i| r.get_native(i) as i64).sum()
    }
    ElementType::I32 => {
        let r = view.open_facet_typed::<i32>("metadata_content")?;
        (0..r.count()).map(|i| r.get_native(i) as i64).sum()
    }
    ElementType::U16 => {
        let r = view.open_facet_typed::<u16>("metadata_content")?;
        (0..r.count()).map(|i| r.get_value(i).unwrap() as i64).sum()
    }
    other => {
        // Fall back to widest integer type
        let r = view.open_facet_typed::<i64>("metadata_content")?;
        (0..r.count()).map(|i| r.get_value(i).unwrap()).sum()
    }
};
println!("sum of all metadata values: {}", total);
```

Each arm uses the most efficient access for its type — `get_native()`
for zero-copy when the type matches, `get_value()` for conversions.

### Dispatch macro (standalone files)

For standalone file access without a dataset, the `dispatch_typed!`
macro handles the match boilerplate:

```rust
use vectordata::dispatch_typed;

let path = "metadata.u8";
dispatch_typed!(path, reader => {
    println!("type: {}", reader.native_type());
    println!("count: {}", reader.count());
    for i in 0..reader.count().min(5) {
        println!("[{}] = {:?}", i, reader.get_record(i)?);
    }
});
```

The macro expands to a match over all integer element types, opening
the file with its native type and calling the body.

---

## 21.6 Mixed-Format Datasets

A single dataset can have facets in different formats:

```yaml
# dataset.yaml
name: mixed-example
profiles:
  default:
    base_vectors: profiles/default/base.fvec      # f32 vectors
    query_vectors: profiles/default/query.fvec     # f32 vectors
    neighbor_indices: ground_truth.ivec            # i32 indices
    metadata_content: profiles/default/meta.u8     # u8 scalar
    metadata_predicates: profiles/default/pred.u8  # u8 scalar
    predicate_results: metadata_indices.ivec       # i32 ordinal lists
```

The client accesses each facet with the appropriate type:

```rust
let view = group.generic_view("default").unwrap();

// Standard vector facets — existing API (always f32/i32)
let base = view.base_vectors()?;       // VectorReader<f32>
let gt = view.neighbor_indices()?;     // VectorReader<i32>

// Metadata facets — typed access
let meta = view.open_facet_typed::<u8>("metadata_content")?;
let pred = view.open_facet_typed::<u8>("metadata_predicates")?;
let results = view.open_facet_typed::<i32>("predicate_results")?;

// Verify: each predicate matches the expected metadata records
for qi in 0..pred.count().min(10) {
    let pred_val = pred.get_native(qi);
    let matching_ordinals = results.get_record(qi)?;

    // Check: every matching ordinal has the predicate value
    for &ord in &matching_ordinals {
        let meta_val = meta.get_native(ord as usize);
        assert_eq!(meta_val, pred_val,
            "predicate {} expects {}, but meta[{}] = {}",
            qi, pred_val, ord, meta_val);
    }
}
```

---

## 21.7 Error Handling

### Error types

| Error | When | Contains |
|-------|------|----------|
| `Narrowing` | Open time | Native type, target type name |
| `ValueOverflow` | Access time | Ordinal, value, target type name |
| `Io` | Open/access | Error message string |

### Handling patterns

```rust
use vectordata::typed_access::TypedAccessError;

match reader.get_value(ordinal) {
    Ok(val) => println!("value: {}", val),
    Err(TypedAccessError::ValueOverflow { ordinal, value, target }) => {
        eprintln!("record {} has value {} which doesn't fit in {}",
            ordinal, value, target);
    }
    Err(e) => eprintln!("error: {}", e),
}
```

---

## 21.8 Performance Characteristics

| Access mode | Allocation | Conversion | Use when |
|------------|-----------|-----------|----------|
| `get_native()` | None (zero-copy) | None | Type matches exactly |
| `get_native_slice()` | None (zero-copy) | None | Need slice of native elements |
| `get_value()` | None (single value) | Per-value checked | Type is wider or cross-sign |
| `get_record()` | Vec<T> allocation | Per-element checked | Need full record as Vec |

For bulk processing with native types, `get_native()` and
`get_native_slice()` are the fastest paths — they return references
directly into the memory-mapped file with no copies.

For cross-type access, `get_value()` reads the native bytes and
converts to T with a range check. The conversion cost is a single
comparison + cast per element.
