# 22 — Vector File Extension Scheme

This section defines the canonical file extension scheme for vector and
scalar data files. The scheme is fully backwards-compatible with legacy
extensions (`.fvec`, `.ivec`, `.bvec`, `.svec`, `.mvec`, `.dvec`) while
providing a systematic naming convention for all element types and
record structures.

---

## 22.1 Extension Grammar

```
extension = <datatype> [ <structure> ] [ "s" ]

datatype  = "u8" | "i8" | "u16" | "i16" | "u32" | "i32"
          | "u64" | "i64" | "f16" | "f32" | "f64"
          | legacy_alias

structure = "vec"   — uniform vector (all records have the same dimension)
          | "vvec"  — variable-length vector (per-record dimension header)

trailing "s" is an optional plural form (e.g., ".fvecs", ".i32vvecs")
```

When `<structure>` is absent, the file is **scalar**: a flat-packed
array of fixed-size elements with no per-record header. Ordinal N is
at byte offset `N × element_size`.

### Examples

| Extension | Datatype | Structure | Description |
|-----------|----------|-----------|-------------|
| `.f32vec` | f32 | uniform | Float vectors, fixed dimension |
| `.i32vec` | i32 | uniform | Integer vectors, fixed dimension |
| `.i32vvec` | i32 | variable | Integer vectors, variable dimension per record |
| `.u8` | u8 | scalar | Flat-packed unsigned bytes |
| `.i64` | i64 | scalar | Flat-packed signed 64-bit integers |
| `.f16vec` | f16 | uniform | Half-precision float vectors |
| `.u8vvec` | u8 | variable | Variable-length byte vectors |

---

## 22.2 Record Layout

### Scalar (no `vec`/`vvec` suffix)

```
[ element₀ | element₁ | element₂ | ... | elementₙ₋₁ ]
```

No header. Element N is at byte offset `N × sizeof(element)`.
Record count = `file_size / sizeof(element)`.

### Uniform vector (`vec` suffix)

```
record = [ dim:i32 | element₀ | element₁ | ... | elementdim₋₁ ]
file   = record₀ record₁ record₂ ... recordₙ₋₁
```

Each record has a 4-byte little-endian `i32` dimension header followed
by `dim` elements. **All records share the same dimension.** The stride
(bytes per record) is `4 + dim × sizeof(element)`, and
`file_size % stride == 0` must hold.

Record count = `file_size / stride`.

Random access to record N: seek to `N × stride`.

### Variable-length vector (`vvec` suffix)

```
record = [ dim:i32 | element₀ | element₁ | ... | elementdim₋₁ ]
file   = record₀ record₁ record₂ ... recordₙ₋₁
```

Same per-record layout as uniform vectors, but **each record may have a
different dimension**. There is no fixed stride. The file is a
concatenation of variable-size records.

Random access requires an **offset index** — a companion file that maps
ordinal → byte offset. See §22.5.

---

## 22.3 Legacy Aliases

The following one-letter legacy extensions are fully supported. Each
maps to a canonical `<datatype>vec` form:

| Legacy | Canonical | Element | Size |
|--------|-----------|---------|------|
| `.fvec` | `.f32vec` | float32 | 4 B |
| `.dvec` | `.f64vec` | float64 | 8 B |
| `.mvec` | `.f16vec` | float16 | 2 B |
| `.bvec` | `.u8vec` | uint8 | 1 B |
| `.svec` | `.i16vec` | int16 | 2 B |
| `.ivec` | `.i32vec` | int32 | 4 B |

Legacy variable-length forms use the same one-letter prefix with
double-v:

| Legacy | Canonical | Description |
|--------|-----------|-------------|
| `.fvvec` | `.f32vvec` | Variable-length f32 vectors |
| `.ivvec` | `.i32vvec` | Variable-length i32 vectors |
| `.bvvec` | `.u8vvec` | Variable-length u8 vectors |
| `.svvec` | `.i16vvec` | Variable-length i16 vectors |
| `.dvvec` | `.f64vvec` | Variable-length f64 vectors |
| `.mvvec` | `.f16vvec` | Variable-length f16 vectors |

Plural forms (`.fvecs`, `.ivvecs`, `.i32vvecs`, etc.) are accepted
everywhere an extension is recognized.

---

## 22.4 Detection Rules

Format detection from a file extension proceeds as follows:

1. Strip optional trailing `s` (plural form).
2. Check for `vvec` suffix → variable-length vector.
3. Check for `vec` suffix → uniform vector.
4. Remaining string is the datatype.
5. If no `vec`/`vvec` suffix, the file is scalar.

The datatype is resolved through the legacy alias table (§22.3) or
parsed directly as a type name (`u8`, `i32`, `f32`, etc.).

### Validation at open time

| File type | Validation |
|-----------|------------|
| Scalar | `file_size % element_size == 0` |
| Uniform vector | `file_size % stride == 0` where `stride = 4 + dim × element_size` |
| Variable vector | No stride check; requires offset index for random access |

**Opening a variable-length file with a uniform reader is an error.**
`MmapVectorReader::open_ivec()` returns `Err(VariableLengthRecords)`
when the file does not pass the stride check. Callers must use
`IndexedXvecReader` for variable-length files.

---

## 22.5 Offset Index Files

Variable-length vector files require an offset index for random access.
The index is a **scalar file** containing one offset per record, stored
as a sibling file:

```
IDXFOR__<original_filename>.<offset_type>
```

Where `<offset_type>` is:

| Type | Max file size | Bytes per offset |
|------|--------------|-----------------|
| `.i32` | 2 GB | 4 |
| `.i64` | unlimited | 8 |

### Example

```
profiles/default/metadata_indices.ivvec        (3.2 GB data)
profiles/default/IDXFOR__metadata_indices.ivvec.i64   (80 KB index)
```

The index file is a flat-packed array of byte offsets. To read record N:

1. Read `offset = index[N]` from the index file.
2. Seek to `offset` in the data file.
3. Read the 4-byte dimension header.
4. Read `dim × element_size` bytes of data.

### Index lifecycle

- **Created automatically** on first random access to a variable-length
  file (`IndexedXvecReader::open()`).
- **Cached**: reused on subsequent opens if the index file's mtime is
  newer than the data file's mtime.
- **Regenerated**: if the data file is newer than the index, the index
  is rebuilt.
- **Best-effort write**: index creation failures are non-fatal; the
  index is a cache, not a requirement. The reader falls back to
  walking the file if the index cannot be written.

### Index files and publishing

Index files (`IDXFOR__*`) are **not published**. They are local caches
that are regenerated on demand. The underscore-prefix convention
(`IDXFOR__`) ensures they are excluded from publishing by the standard
file filters.

---

## 22.6 Pipeline Usage

### Ground truth and KNN results (uniform)

Ground truth indices (G facet) and filtered KNN indices (F facet) are
always uniform — every query has exactly `k` neighbors:

```yaml
neighbor_indices: profiles/base/neighbor_indices.ivec       # i32, uniform, dim=k
filtered_neighbor_indices: profiles/default/filtered_neighbor_indices.ivec
```

### Predicate result ordinals (variable)

Predicate evaluation results (R facet) have one record per predicate,
where each record lists the base vector ordinals that match. Different
predicates match different numbers of vectors:

```yaml
metadata_indices: profiles/default/metadata_indices.ivvec   # i32, variable
```

### Metadata and predicates (scalar)

Per-vector metadata values and per-query predicate values are scalar:

```yaml
metadata_content: profiles/base/metadata_content.u8         # scalar u8
metadata_predicates: profiles/base/predicates.u8            # scalar u8
```

---

## 22.7 Complete Extension Table

| Extension | Element | Structure | Element size |
|-----------|---------|-----------|-------------|
| `.u8` | uint8 | scalar | 1 |
| `.i8` | int8 | scalar | 1 |
| `.u16` | uint16 | scalar | 2 |
| `.i16` | int16 | scalar | 2 |
| `.u32` | uint32 | scalar | 4 |
| `.i32` | int32 | scalar | 4 |
| `.u64` | uint64 | scalar | 8 |
| `.i64` | int64 | scalar | 8 |
| `.u8vec` / `.bvec` | uint8 | uniform | 1 |
| `.i8vec` | int8 | uniform | 1 |
| `.u16vec` | uint16 | uniform | 2 |
| `.i16vec` / `.svec` | int16 | uniform | 2 |
| `.u32vec` | uint32 | uniform | 4 |
| `.i32vec` / `.ivec` | int32 | uniform | 4 |
| `.u64vec` | uint64 | uniform | 8 |
| `.i64vec` | int64 | uniform | 8 |
| `.f16vec` / `.mvec` | float16 | uniform | 2 |
| `.f32vec` / `.fvec` | float32 | uniform | 4 |
| `.f64vec` / `.dvec` | float64 | uniform | 8 |
| `.u8vvec` / `.bvvec` | uint8 | variable | 1 |
| `.i8vvec` | int8 | variable | 1 |
| `.u16vvec` | uint16 | variable | 2 |
| `.i16vvec` / `.svvec` | int16 | variable | 2 |
| `.u32vvec` | uint32 | variable | 4 |
| `.i32vvec` / `.ivvec` | int32 | variable | 4 |
| `.u64vvec` | uint64 | variable | 8 |
| `.i64vvec` | int64 | variable | 8 |
| `.f16vvec` / `.mvvec` | float16 | variable | 2 |
| `.f32vvec` / `.fvvec` | float32 | variable | 4 |
| `.f64vvec` / `.dvvec` | float64 | variable | 8 |
