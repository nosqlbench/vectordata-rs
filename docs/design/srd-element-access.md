# SRD — Element types and typed facet access

**Status:** proposed 2026-10-19. Decisions in §14 were taken on
2026-10-19. Not implemented.
**Scope:** the element model shared by every crate in the workspace (a
new `veks-datatypes` crate), the `vectordata` readers and the facet
access API on a dataset view, the `veks-io` readers, the file-format
spellings, three new standard facets for pre-quantized corpora, the
`veks check` verifications that go with them, the retirement of the
duplicate element and facet enums in `veks-pipeline` and `veks-core`,
and two catalog defects.
**Builds on** `srd-multifile-facet-shards.md` (series and windows are
reused, not re-specified), `srd-api-discoverability.md` (the view is
the one access path), `metadata-facets-and-layout-namespace.md` (the
facet spec is the single authority).

**Origin.** An index-building program (`fastervector`) builds at three
precisions — u8 codes, bf16, f32 — and reads only f32 and i32 from
disk, because that is what the dataset view serves. It asked for a
typed access API of the same shape, with the type a parameter; for
bf16 as an element type; for zero-copy bulk reads of a native file;
for the access rules to be stated once and enforced at open; for a
convention that lets a pre-quantized corpus travel with its
quantization parameters as one dataset; and for `veks-io`, which it
uses for local reads, to follow the same rules. The requester is one
consumer and does not shape the interface; the requirement is general.

---

## 1. Problem

Line numbers are as of 2026-10-19, at 3.0.4. Paths are under `crates/`.

**Three element models, none complete.**
- `vectordata::typed_access::ElementType` (`typed_access.rs:41`): eleven
  variants, no bf16, matched exhaustively in three places.
- `veks-pipeline/src/pipeline/element_type.rs`: a second enum with the
  same eleven variants, 248 match arms over 26 files. Its
  `dispatch_reader!` opens `.i8vec` as `XvecReader<u8>`, `.u16vec` as
  `XvecReader<i16>`, `.u32vec` as `XvecReader<i32>` and the 64-bit
  integer files as `XvecReader<f64>` — reinterpretations by width that
  are only possible because the reader does not check (next point).
- `veks-io` has no element trait: `MmapReader<T>` is `T: Copy` with a
  constructor per type that passes the width by hand (`xvec/mmap.rs:117`).

Twelve files carry their own extension tables (`"mvecs"` grep). The
pipeline's copy admits `.f32`/`.f64` scalar files; vectordata's refuses
them. They have drifted.

**Two element traits in vectordata.** `io::VvecElement` (`io.rs:92`,
bytes → value) and `typed_access::TypedElement` (`typed_access.rs:219`,
integers only, conversions through `i128`). One idea, two bounds.

**The raw reader reinterprets.** `XvecReader::<T>::open` (`io.rs:521`)
checks only that the extension is *known* (`validate_element_for_source`,
`io.rs:364`); `open_path` (`io.rs:529`) checks nothing. A `.dvec` opened
as f32 yields a reader whose count and values are garbage; it errors only
when the file size happens not to divide by the wrong stride, and then as
`VariableLengthRecords`, which is not what happened. `docs/sysref/
02-api.md:341` documents the check that does not exist. `IndexedVvecReader`
does check (`io.rs:807`).

**Typed scalar access returns zeros for floats.** `TypedReader::
read_native_value` (`typed_access.rs:275`) returns `0` for every float
element, silently, and the open-time check (`T::width() <
native.byte_width()`) lets an F32 file open as i32, u32, i64 or u64.

**The view is pinned.** `TestDataView` (`view.rs:528`) serves
`base_vectors`, `query_vectors`, `facet` and the distances as
`VectorReader<f32>`, the indices as `VectorReader<i32>`. Every reader is
built by the private `open_uniform<T>` (`view.rs:3008`), instantiated at
exactly those two types. A u8 or bf16 base has no way out of a view.

**No whole-range read.** `VectorReader::get_slice` is per record;
`TypedReader::get_native_slice` likewise, and only for u8. A header-free
column cannot be borrowed as one `&[T]`.

**No header-free float format.** `is_scalar_ext` (`io.rs:356`) and
`FacetFormat::ScalarPacked` (`dataset/facet.rs:76`) are integers only.

**veks-io validates less than vectordata.** `MmapReader::open_impl`
(`xvec/mmap.rs:56`) casts a negative dimension to `usize` unchecked,
computes `4 + dim * elem_size` unchecked, and its `get_slice` casts an
offset that is 4 mod 8 to `&[f64]` — undefined behaviour. The streaming
`xvec/reader.rs::open` has no remainder check. `MmapReader` has no
production consumer in this workspace (one dead re-export in veks-core).

**Two facet enums.** `veks-core/src/formats/facet.rs::Facet` duplicates
`vectordata::dataset::StandardFacet` and has drifted: it lacks the
`metadata_indices` alias, and `preferred_format` is keyed by width, so
i8 and u16 collapse onto u8 and i16.

**Nothing checks cardinality.** `veks check` (PL-3, `check/profiles.rs`)
checks F/E rows against the query count. Nothing checks G, and nothing
checks that a ground-truth index is below the base count.

**Catalog defects** (reported against 3.0.4, verified):
1. `parse_catalog_content` (`catalog/source.rs:224`) routes every map to
   the `knn_entries` parser, so a catalog source pointed at a
   `dataset.yaml` yields datasets literally named `profiles` and
   `attributes`. `DatasetHandle::of_content` (`group.rs:69`) already has
   the rule (`contains_key("profiles")`); the catalog path does not use it.
2. `open_spec` (`catalog/resolver.rs:231`) takes a spec that exists on
   disk as a path, selector and all. That is the stated rule ("a path
   that exists is a path, whatever punctuation it holds"), applied in
   three places, and what makes `C:\data\ds` resolve. Not a defect.
3. `of_knn_entries` (`group.rs:89`) opens a directory whose
   `knn_entries.yaml` lists several datasets, none named after the
   directory, as its first entry, silently. Documented and tested as
   intended; it contradicts the ruling that ambiguity is never resolved
   by order (`Error::AmbiguousDataset` for same-named datasets across
   catalogs).

## 2. Principles

- **One element model.** One enum, one trait, one extension table, one
  conversion table, in one crate every other crate depends on.
- **Native is zero-copy; everything else is a conversion, and a
  conversion is either lossless or refused.** A reader never
  reinterprets bytes as a type the file does not hold.
- **The rule is stated once and enforced wherever bytes are opened.**
  The raw readers refuse anything but native. The view applies the
  conversion table. Both consult the same function.
- **Same shape, type added.** The generic API has the names of the
  pinned one with `_as` and a type parameter. The pinned one stays,
  deprecated, defined as the generic one at f32/i32.
- **Facets, not formats.** Quantization parameters are standard facets
  in the existing facet model, with the scheme in the dataset's
  attributes. No new container.

## 3. Element model — `veks-datatypes`

A new leaf crate, `crates/veks-datatypes`, depending on `half` only. It
is published and versioned with the others.

**EA-1.** `ElementType` is `#[non_exhaustive]` and has the variants
`U8 I8 U16 I16 U32 I32 U64 I64 F16 BF16 F32 F64`. `ElementType::ALL`
lists them. `byte_width`, `name`, `Display`, `from_name`, and
`kind() -> ElementKind` with `Unsigned | Signed | Float(FloatFormat)`,
`FloatFormat::{Ieee, Brain}`.

**EA-2.** `half::f16` and `half::bf16` are the f16 and bf16 element
types, re-exported as `veks_datatypes::{f16, bf16}` so no downstream
needs its own `half` pin. `bf16` is `#[repr(transparent)]` over `u16`;
a consumer with its own transparent bf16 wrapper reinterprets a
`&[bf16]` soundly. It is never a `u16` to this crate: the type carries
the kind.

**EA-3.** One extension table. `Container::{Scalar, Vec, Vvec}`;
`ElementType::extension(self, Container) -> &'static str` is the
canonical spelling (`f32vecs`, `bf16vecs`, `u8vvecs`, `f32`);
`aliases(self, Container)` lists the legacy spellings (`fvecs`, `fvec`,
`mvecs`, `bvecs`, `svecs`, `dvecs`, `ivecs`, singular forms);
`parse_extension(ext) -> Option<(ElementType, Container)>` accepts
canonical and legacy spellings, case-insensitively. New spellings:
`bf16vec(s)`, `bf16vvec(s)`, and the scalar `.bf16`, `.f16`, `.f32`,
`.f64`. `mvec(s)` remains IEEE f16. `ivec(s)` parses as `(I32, Vec)`;
vectordata's vvec reader keeps accepting it as it does today, because
old datasets used it for variable-length files.

**EA-4.** The `Element` trait, the one bound for every reader:
```rust
pub trait Element: Copy + Send + Sync + 'static {
    const TYPE: ElementType;
    const SIZE: usize;
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn write_le_bytes(self, out: &mut [u8]);
    fn to_value(self) -> Value;                 // conversion path only
    fn from_value(v: Value) -> Option<Self>;    // None: does not fit
    fn as_f64(self) -> f64;
}
pub enum Value { Int(i128), Float(f64) }
```
Implemented for the twelve types. `to_value`/`from_value` are never on
the native path.

**EA-5.** The conversion table, stated once:
```rust
pub enum ElementConversion { Native, Widening, CrossSign, Refused }
impl ElementType { pub fn conversion_to(self, target: ElementType) -> ElementConversion; }
```
| file → requested | result |
|---|---|
| same type | `Native` — zero-copy |
| int → wider int, source unsigned or target signed (u8→i16, u8→u32, i16→i64) | `Widening` — never fails |
| int → int, same width, other sign (u8↔i8, i32↔u32) | `CrossSign` — every value checked |
| int → wider unsigned from signed (i8→u16) | `CrossSign` |
| int → narrower int | `Refused` |
| int → float, integer bits ≤ significand bits (u8/i8/u16/i16→f32; ≤32-bit→f64; u8/i8→f16; u8/i8→bf16) | `Widening` |
| int → float otherwise (i32→f32, i64→f64) | `Refused` — not lossless |
| f16→f32, f16→f64, bf16→f32, bf16→f64, f32→f64 | `Widening` |
| f16↔bf16 | `Refused` — neither direction is lossless, and never a reinterpretation |
| float → narrower float, float → int | `Refused` |

`Refused` carries no conversion; the caller reads natively and converts
itself. `pub fn convert<N: Element, T: Element>(n: N) -> Result<T,
Overflow>` is the per-value conversion the `Widening` and `CrossSign`
readers use.

**EA-6.** `dispatch_element!($ty, $T => $body)` expands `$body` once per
variant with `$T` bound to the element type, so a runtime `ElementType`
reaches a monomorphic reader without a hand-written twelve-arm match at
every site.

## 4. Readers (`vectordata::io`)

**EA-7.** `VvecElement` and `TypedElement` are replaced by `Element`.
Both remain as deprecated blanket traits (`pub trait VvecElement:
Element {}` with `impl<T: Element> VvecElement for T {}`), so existing
bounds compile with a warning. `TypedElement`'s own methods (`width`,
`to_i128`) are gone.

**EA-8.** `VectorReader<T>` gains three defaulted methods:
- `fn as_slice(&self) -> Option<&[T]>` — every element of every record,
  contiguous, borrowed from the mapping (`count() * dim()` elements).
  `Some` only for a header-free column in one resident mapped file, or a
  window over one. `None` for xvec files (headers interleave), series,
  HTTP storage, and converted readers.
- `fn conversion(&self) -> ElementConversion` — `Native` unless the
  reader converts.
- `fn native_element_type(&self) -> Option<ElementType>` — what the file
  holds.

**EA-9.** `ScalarReader<T: Element>`: a header-free column. `dim() ==
1`, `count = size / T::SIZE`, a remainder is `InvalidFormat`. `get`,
`get_slice`, bulk `read_into` (one `copy_nonoverlapping`), `as_slice`
after the same alignment check `XvecReader::get_slice` makes. Series of
scalar shards are read through the generalised `ShardedReader<T, R>`
(§5), so `Shards::locate` and the seam-crossing `read_into` are written
once.

**EA-10.** `ConvertingReader<N: Element, T: Element>` implements
`VectorReader<T>` over an `Arc<dyn VectorReader<N>>` for `Widening` and
`CrossSign`. `get_slice` and `as_slice` are `None`; `get` and
`read_into` convert through a bounded scratch of `N`; the first value
that does not fit is `IoError::ValueOverflow { index, value, target }`
at its facet ordinal. It is never constructed for `Native`, so the
native path costs nothing.

**EA-11.** Raw readers are native-only. `XvecReader::open`, `open_path`,
`IndexedVvecReader::open`, `open_path`, and `ScalarReader::open`,
`open_path` call one helper, `io::require_native::<T>(source)`, which
parses the extension and requires `conversion_to(T::TYPE) == Native`.
Anything else is `IoError::ElementMismatch { file, requested, reason }`.
`from_storage` (crate-private, no extension) stays unchecked; the view
checks before calling it. `IoError` becomes `#[non_exhaustive]`.

**EA-12.** `AnyVectorReader`: an enum with one variant per element type
holding `Arc<dyn VectorReader<T>>`. `element_type()`; `into_typed::<T>()
-> Result<Arc<dyn VectorReader<T>>, IoError>` (a `dyn Any` downcast,
no unsafe); crate-private `converted_to::<T>(mode)`. It is the carrier
for both erasures (a runtime target type to a static `T`; a static `T`
over a runtime native type), and the public "open at whatever type the
file has" result.

**EA-13.** `io::open_vec_as::<T>(source)` opens a raw file at its own
type and applies the conversion table; `open_vec` stays native-only.
The explorer opens bases through `open_vec_as::<f32>` and drops its
private two-variant enum, so f16, bf16 and u8 bases become browsable.

**EA-14.** `TypedReader`, `TypedAccessError`, `open_facet_typed` and
`dispatch_typed!` are deprecated with a migration note (`get_value(i)`
→ `get(i)?[0]` or `as_slice()?[i]`; `get_native_slice(i)` →
`get_slice(i)`; `is_native()` → `conversion() == Native`). One guard
closes the float hole: a float native type is refused at open instead
of reading zeros.

## 5. The view

**EA-15.** `TestDataView` gains, beside `open_facet_storage`, one
dyn-safe method with a default body that returns an error (external
implementors keep compiling) and one provided method:
```rust
fn open_facet_as_type(&self, name: &str, target: ElementType) -> Result<AnyVectorReader>;
fn open_facet_native(&self, name: &str) -> Result<AnyVectorReader>;   // at the file's own type
```

**EA-16.** `TestDataViewExt`, a blanket extension over `V: TestDataView
+ ?Sized`, so it works on `Arc<dyn TestDataView>` and on the concrete
view:
```rust
fn facet_as<T: Element>(&self, name: &str) -> Result<Arc<dyn VectorReader<T>>>;
fn base_vectors_as<T: Element>(&self) -> Result<Arc<dyn VectorReader<T>>>;
fn query_vectors_as<T: Element>(&self) -> ...;
fn neighbor_indices_as<T: Element>(&self) -> ...;
fn neighbor_distances_as<T: Element>(&self) -> ...;
fn prefiltered_neighbor_indices_as / prefiltered_neighbor_distances_as
fn postfiltered_neighbor_indices_as / postfiltered_neighbor_distances_as
fn metadata_results_as<T: Element>(&self) -> Result<Arc<dyn VvecReader<T>>>;
fn quantized_base<C: Element>(&self) -> Result<QuantizedBase<C>>;        // §7
```
`facet_as` is `open_facet_as_type(name, T::TYPE)?.into_typed()`.

**EA-17.** Inside the view, `open_uniform<T>` is: require the Elements
shape → the facet's native type (`facet_element_type`) →
`conversion_to(T::TYPE)` (`Refused` → `ElementMismatch`) → open at the
native type (today's `open_uniform` body, which keeps the series and
window logic, written once; it picks `ScalarReader` for a scalar
extension) → `into_typed` or `converted_to`. `open_variable<T>` requires
`Native`.

**EA-18.** The pinned methods (`base_vectors`, `query_vectors`,
`neighbor_indices`, `neighbor_distances`, the pre/post-filtered pairs,
`facet`, `metadata_results`) stay on the trait, `#[deprecated]`, and are
exactly the generic ones at f32 or i32. Consequence: `facet()` on a u8
facet widens, and on an `.ivec` facet is refused, where it reinterpreted
bits before.

**EA-19.** The ordinal and windowing semantics of `srd-multifile-facet-
shards.md` (SH-62, SH-64, SH-67) hold for every `T` and for scalar
series. The existing scalar-series tests keep passing through the
generic path.

## 6. File formats and the facet spec

**EA-20.** `FacetFormat::FloatXvec` accepts the bf16 spellings. A new
`FacetFormat::FloatScalar` covers `f32 f64 f16 bf16`; `ScalarPacked`
remains the integer scalar format. Both lists are derived from
`ElementType::ALL × Container`, not written out.

**EA-21.** Every extension table in the workspace (`typed_access.rs`,
`io.rs`, `datasets/filter.rs`, `explore/shared.rs`, `dataset/facet.rs`,
`veks-io/src/format.rs`, `veks-core/src/formats/mod.rs`,
`veks/src/prepare/{import,wizard}.rs`, `veks-pipeline …/element_type.rs`,
`compute_dedup.rs`, `gen_vectors.rs`) is replaced by a call into
`ElementType`.

**EA-22.** `veks-io::VecFormat` gains `Bf16vec`, `ScalarF32`,
`ScalarF64`, `ScalarF16`, `ScalarBf16`; `element_type() ->
Option<ElementType>` and `of(Container, ElementType) -> Option<VecFormat>`;
`from_extension` and `preferred_extension` delegate to `ElementType`.
The scalar and xvec writers accept the new formats.

## 7. Pre-quantized corpora

A quantized vector is not self-describing. The requester's corpus
quantizer is affine over each row's own range, so every vector has a
scale (f32), a zero point (i32), and the sum of its codes (i32), and the
dequantized distance needs all three, as three separate arrays. Other
systems use per-dimension or per-block scales. The convention must say
which.

**EA-23.** Three standard facets, each one header-free packed column,
index-aligned with `base_vectors`:

| key | canonical file | format | role |
|---|---|---|---|
| `quant_scale` | `quant_scale.f32` | `FloatScalar` | Windowed (follows `base_vectors`) |
| `quant_zero_point` | `quant_zero_point.i32` | `ScalarPacked` | Windowed |
| `quant_sum` | `quant_sum.i32` | `ScalarPacked` | Windowed |

No facet code letter (as `metadata_layout`). They sit after
`base_vectors` in `PREFERRED_ORDER`. Sized and windowed profiles carry
them with `base_vectors` (`facet_role` → `Windowed`).

**EA-24.** The scheme is a dataset or profile attribute, typed:
```yaml
attributes:
  quantization:
    scheme: affine            # affine | symmetric
    granularity: per-vector   # per-vector | per-dimension | per-block
    block: 64                 # per-block only
    codes: u8                 # the element type of base_vectors
    range: [0, 255]
```
`dataset::Quantization` (serde) with `codes: ElementType`; `TestDataView::
quantization() -> Option<Quantization>`. Per-dimension and per-block
layouts of the three columns are declared but not specified here (§10).

**EA-25.** `QuantizedBase<C>` resolves codes and parameters as one
thing:
```rust
pub struct QuantizedBase<C: Element> {
    pub codes: Arc<dyn VectorReader<C>>,
    pub scale: Arc<dyn VectorReader<f32>>,
    pub zero_point: Arc<dyn VectorReader<i32>>,
    pub sum: Arc<dyn VectorReader<i32>>,
    pub scheme: Quantization,
}
```
`quantized_base::<C>()` opens the four, requires `C::TYPE ==
scheme.codes`, and requires each column's `count()` to equal the codes'
`count()`, else `Error::FacetCardinality { facet, expected, found }`.
On a local single-file dataset each column's `as_slice()` is `Some`.

**EA-26.** `veks check`: the three facets are present together or not
at all; each column's count equals the base count; the `quantization`
attribute is present iff the facets are; `codes` names the base facet's
element type. Conformance already refuses a wrong extension.

**EA-27.** Producing quantized corpora is not in this SRD (§10). The
writers can write the columns (EA-22).

## 8. veks-io parity

**EA-28.** `MmapReader<T: Element>::open(path)` is generic. Validation:
the dimension is read as `i32` and must be in `1..=i32::MAX`;
`entry_bytes` is computed with checked arithmetic; the remainder check
stays; the extension must be `Native` for `T` (EA-11's rule, same
function). `get_slice` returns `Option<&[T]>`, `None` when the record
is misaligned for `T`; `get` reads unaligned; `read_into` has
vectordata's contract. The per-type constructors become deprecated
wrappers. The streaming `xvec/reader.rs::open` and `probe` get the
remainder and dimension checks.

**EA-29.** `ScalarMmapReader<T: Element>::open(path)` is generic and
gains `as_slice`; the per-type constructors are deprecated.

**EA-30.** veks-io stays native-only. The conversion table is shared
(it is in `veks-datatypes`); converting readers are vectordata's.

## 9. Consolidation and verification

**EA-31.** `veks-pipeline`'s `ElementType` is deleted; the path
re-exports `veks_datatypes::ElementType`. The pipeline-policy methods
(`is_float`, `supports_simd_distance`, `machine_epsilon`,
`normalization_threshold`) move to an `ElementTypeExt` trait in the
same file. `ToF64` is replaced by `Element::as_f64`. `dispatch_reader!`
opens every type as itself.

**EA-32.** `veks-core`'s `Facet` is deleted; its consumers use
`vectordata::dataset::StandardFacet`. `preferred_format` becomes
`veks_core::formats::preferred_vec_format(StandardFacet, ElementType)
-> VecFormat`, keyed by type. The dead `IoMmapReader` re-export goes.

**EA-33.** `veks check` verifies ground truth: `neighbor_indices` rows
equal the query count; `neighbor_distances` rows equal the indices'
rows; every index is below the base count, as a full scan with a
progress line. The same for the pre- and post-filtered pairs. File
names are never inferred from jvector's `_base_`/`_query_`/`_gt_`
infixes; `knn_entries.yaml` names its files.

**EA-34.** Catalog defect 1: `parse_catalog_content` treats a map with a
`profiles` key as one `dataset.yaml` and builds its single entry, by the
rule `DatasetHandle::of_content` already applies. Defect 3:
`of_knn_entries` with several entries and none named after the
directory is `Error::AmbiguousDataset` listing them; one entry opens;
none is the empty dataset as today. Defect 2 is the documented rule and
is unchanged.

## 10. Non-goals

- A transport-free `vectordata` feature (dropping `reqwest`). The
  requester rates it low once veks-io shares the model.
- Inference of facets from jvector's file-name infixes.
- Quantizing a corpus in the pipeline, and the on-disk layout of
  per-dimension and per-block parameters. The scheme attribute names
  them so a later SRD can specify them without renaming anything.
- Query-side quantization parameters. Engines quantize queries at run
  time.
- bf16 SIMD kernels. SK-10 of `srd-native-simd-kernels.md` stands: a
  bf16 corpus reaches the KNN engines through `Widening` to f32, at the
  stated cost.
- A header-free matrix format for vectors. The per-record header makes
  an xvec file contiguous one row at a time; `read_into` is the bulk
  primitive and that copy is inherent to the format.
- A Java mirror. The Java library's parity work tracks this SRD
  separately.

## 11. Acceptance tests

Guards that fail on 3.0.4, then pass:
- `XvecReader::<f32>::open_path("x.ivec")` and `XvecReader::<i32>::
  open("x.fvec")` are `Err` (today `Ok`).
- A `.dvec` of 5 records × 4 opened as f32 is `ElementMismatch` (today a
  reader with `count() == 9`).
- `IndexedVvecReader::<i32>::open_path("x.u32vvec")` is `Err`.
- `TypedReader::<i32>::open("x.f32")` is `Err` (today zeros).
- `facet_as::<f32>` on an `.mvec` facet of 3 × 2 has `count() == 3`, the
  exact widened values, `conversion() == Widening`, `get_slice() ==
  None`, `native_element_type() == Some(F16)` (today count 2, garbage).
- `facet()` on an `.ivec` facet is `Err` (today reinterpreted).
- veks-io: a negative dimension is rejected; a truncated file is
  rejected; `get_slice` on a misaligned f64 record is `None`;
  `MmapReader::<f32>::open("x.bvecs")` is refused.
- `TestDataGroup::load` on a two-entry `knn_entries` directory with no
  matching name is `AmbiguousDataset`.
- A catalog source pointed at a `dataset.yaml` lists one dataset with
  the document's name.
- `dispatch_reader!` opens an `.i8vecs` as i8.

Behaviour of the new surface:
- The conversion table: every pair in `ALL × ALL` has the result in
  EA-5; `Native` on the diagonal; `CrossSign` is symmetric at equal
  width.
- u8 codes: `facet_as::<u8>` has `as_slice()` of `count()` elements and
  `get_slice(i)` per element; `facet_as::<i8>` over a byte 200 reports
  `ValueOverflow { index }` from `get` and from `read_into`;
  `facet_as::<u16>` and `::<f32>` widen exactly.
- bf16: a `.bf16vecs` round-trips through the writer and reader;
  `facet_as::<bf16>` is native with `get_slice` `Some`;
  `facet_as::<f32>` equals `bf16::to_f32` per element; `facet_as::<f16>`
  is refused.
- A scalar series (the sharded-facets fixture) through `facet_as::<u32>`:
  `count() == 175`, `read_into` across the seam, `as_slice() == None`; a
  sliced series maps facet ordinal 10 to its value.
- A window `codes.u8[2..6)`: `as_slice() == Some(&codes[2..6])`.
- `open_facet_native` on a u8 facet is `AnyVectorReader::U8`;
  `into_typed::<i32>()` on it is `ElementMismatch`.
- `base_vectors()` equals `base_vectors_as::<f32>()` on a windowed and on
  a sharded profile.
- A function over `&Arc<dyn TestDataView>` calls `facet_as::<f32>` with
  only `TestDataViewExt` imported.
- A profile with codes and three columns opens as one `QuantizedBase`
  with `Some` slices equal to the written columns; a short `quant_sum`
  is `FacetCardinality` at open and a `veks check` failure; conformance
  refuses `quant_scale.i32`.
- `veks check` reports an index ≥ base count and a G/Q row mismatch.
- Five rustdoc examples compile and run under `cargo test --doc` with no
  network, each naming its conversion: a native f32 window through
  `read_into`; u8 codes zero-copy; bf16 native then widened; i32 ground
  truth checked against `base_count`; the refusal and its error.
- The existing suites (vectordata, veks-pipeline, veks, veks-io,
  veks-core) pass, with `#![allow(deprecated)]` on the typed-reader
  tests.

## 12. Semver

Breaking: `ElementType` gains `BF16` and becomes `#[non_exhaustive]`;
`IoError` gains variants and becomes `#[non_exhaustive]`; `FacetFormat`
and `StandardFacet` gain variants; `VvecElement`/`TypedElement` are
replaced (the blanket shims keep bounds compiling); the raw readers and
`facet()` refuse what they reinterpreted; a multi-entry `knn_entries`
directory errors; veks-io's constructors become generic; veks-core's
`Facet` and veks-pipeline's `ToF64` are removed. Non-breaking: the
defaulted `VectorReader` and `TestDataView` methods, `TestDataViewExt`,
`AnyVectorReader`, `ScalarReader`, `ConvertingReader`, the deprecations.
The crates are versioned in lockstep; the bump is the maintainer's call.

Upgrading, old → new:

| 3.0 | next |
|---|---|
| `view.base_vectors()` (f32) | `view.base_vectors_as::<f32>()` — the old call stays, deprecated |
| `view.facet(name)` (f32) | `view.facet_as::<T>(name)` |
| `open_facet_typed::<T>(&*view, name)` / `TypedReader` | `view.facet_as::<T>(name)`; `get_value(i)` → `get(i)?[0]` or `as_slice()?[i]` |
| `T: VvecElement` / `T: TypedElement` | `T: Element` |
| `XvecReader::<f32>::open("x.mvec")` (reinterpreted) | `open_vec_as::<f32>("x.mvec")` (widened) |
| `veks_core::formats::Facet` | `vectordata::dataset::StandardFacet` |
| `veks_pipeline::pipeline::ElementType` | the same path, now `veks_datatypes::ElementType` |
| `veks_io::MmapReader::open_fvec(p)` | `MmapReader::<f32>::open(p)` |

## 13. Plan

Each step is one themed commit. Every guard is run once against the
code before the step and shown to fail, then the step lands and it
passes. Line numbers are as of 2026-10-19. After each step: `cargo t`
for the crates touched, clippy clean on the changed lines, `cargo doc
--no-deps` with warnings denied for any crate whose public items
changed.

### 13.1 Step 1 — `veks-datatypes`

**Files.** New `crates/veks-datatypes/{Cargo.toml, src/lib.rs,
src/element_type.rs, src/element.rs, src/conversion.rs}`. Root
`Cargo.toml`: `members` and `default-members`. Version in lockstep.
License header on each file; the first doc line of `lib.rs` names it a
library API.

**Builds.** EA-1 to EA-6 as written in §3:
- `ElementType` (`ALL`, `byte_width`, `name`, `from_name`, `kind`,
  `Display`), `ElementKind`, `FloatFormat`, `Container`.
- `extension`, `aliases`, `parse_extension` — one table, written as a
  `const` array of `(ElementType, Container, canonical, &[aliases])`
  rows so the three functions read the same data.
- `Element` for u8, i8, u16, i16, u32, i32, u64, i64, `f16`, `bf16`,
  f32, f64, through one macro; `Value`; `convert`; `Overflow`.
- `conversion_to`; `ElementConversion`; `dispatch_element!`.
- `pub use half::{bf16, f16}`.

**Tests** (in-crate): every pair of `ALL × ALL` against the EA-5 table,
written as a literal 12×12 matrix so a change to the rule is a visible
diff; `parse_extension(extension(t, c)) == Some((t, c))` for every pair
and for every alias; case-insensitivity; `convert` exact for every
representable value of each `Widening` pair sampled at the edges;
`CrossSign` overflow at the first value out of range; `bf16` and `f16`
byte round trips; `from_value` refuses NaN and out-of-range for
integers.

**Done when** the crate builds with no dependencies but `half`, docs
build with warnings denied, and the matrix test is green.

### 13.2 Step 2 — veks-io on the model

**Files.** `crates/veks-io/Cargo.toml` (depend on `veks-datatypes`),
`src/xvec/mmap.rs`, `src/xvec/reader.rs`, `src/scalar/mmap.rs`,
`src/scalar/reader.rs`, `src/scalar/writer.rs`, `src/xvec/writer.rs`,
`src/format.rs`, `src/lib.rs`.

**Builds.**
- `MmapReader<T: Element>::open(path) -> Result<Self, MmapError>`;
  `open_impl` validates per EA-28 (`dim` in `1..=i32::MAX`, checked
  `entry_bytes`, remainder, `require_native`); `get_slice(index) ->
  Option<&[T]>`; `get(index) -> Vec<T>` by unaligned reads;
  `read_into(start, out: &mut [T]) -> Result<usize, MmapError>`.
  `open_fvec` … `open_svec` deprecated, each `Self::open(path)`. The
  `bvec` doc corrected.
- `ScalarMmapReader<T: Element>::open(path)`; `as_slice() ->
  Option<&[T]>`; per-type constructors deprecated.
- `xvec/reader.rs::open` and `probe`: dimension range and remainder
  checks; a dimension that is not positive is an error, not `as u32`.
- `VecFormat`: `Bf16vec`, `ScalarF32`, `ScalarF64`, `ScalarF16`,
  `ScalarBf16`; `element_type()`, `of(Container, ElementType)`;
  `from_extension`, `preferred_extension`, `canonical_extension`,
  `detect` and `element_size` delegate to `ElementType`. The scalar and
  xvec writers accept the new formats.

**Guards** (fail on old): a file whose header is `-4` is rejected
(old: `dim` becomes a huge `usize`); a file one byte short of a record
is rejected by the streaming reader (old: the tail is dropped); an f64
xvec's `get_slice(0)` is `None` (old: a misaligned `&[f64]`);
`MmapReader::<f32>::open("x.bvecs")` is `Err` (old: `Ok`). Behaviour:
a `.bf16vecs` written by the writer reads back exactly; a `.f32` column
written by the scalar writer reads back through `as_slice`.

**Done when** veks-io's suite and veks-core's suite pass (veks-core's
scalar reader wrapper is the only in-workspace consumer).

### 13.3 Step 3 — vectordata's element model and the width check

**Files.** `crates/vectordata/Cargo.toml`, `src/typed_access.rs`,
`src/io.rs`, `src/dataset/facet.rs`, `src/datasets/filter.rs`,
`src/explore/shared.rs`, `src/lib.rs`; `crates/veks-pipeline/src/
pipeline/element_type.rs` (`dispatch_reader!` only) and `commands/
compute_knn_distances.rs`.

**Builds.**
- `typed_access::ElementType`, `ElementConversion`, `Element` are
  `pub use veks_datatypes::…`; the root re-exports `Element` and
  `ElementConversion` beside `ElementType`. `ElementType::from_path`,
  `from_url`, `from_extension`, `is_scalar_format`, `can_open_as`
  become thin wrappers (`can_open_as` deprecated in favour of
  `conversion_to`).
- `io::VvecElement` and `typed_access::TypedElement` are the deprecated
  blanket traits of EA-7. The hand-written `impl VvecElement for …`
  blocks (`io.rs:99-164`) and `impl_typed_element!` go.
- `io::infer_elem_size`, `is_scalar_ext`, `is_vvec_ext`, `ext_of` wrap
  `parse_extension`; `is_vvec_ext` keeps `ivec(s)` true.
- `io::require_native::<T>(source) -> Result<ElementType, IoError>`
  replaces `validate_element_for_source`; called from `XvecReader::
  open`, `open_path`, `IndexedVvecReader::open_path`, `validate_source`.
  `IoError` is `#[non_exhaustive]` and gains `ElementMismatch { file,
  requested, reason: &'static str }` and `ValueOverflow { index, value:
  i128, target }`.
- `TypedReader::open` / `open_auto` / `from_storage`: a float native
  type is refused (`TypedAccessError::Narrowing`-class, with a message
  that names `facet_as`).
- `FacetFormat::FloatScalar`; `FloatXvec` bf16 spellings; both
  `extensions()` lists derived from `ElementType::ALL`.
- The explorer's `AnyVectorReader` keeps working through this step
  (it opens f32 and f16 natively); it is replaced in step 4.
- `dispatch_reader!` opens every type as itself through
  `dispatch_element!`; `compute_knn_distances.rs` bounds on `Element`.

**Guards** (fail on old): `XvecReader::<f32>::open_path("x.ivec")` is
`Err`; `XvecReader::<i32>::open("x.fvec")` is `Err`; a `.dvec` of 5 × 4
opened as `XvecReader<f32>` is `ElementMismatch { file: F64, requested:
F32 }` (old: `count() == 9`); `IndexedVvecReader::<i32>::open_path
("x.u32vvec")` is `Err`; `TypedReader::<i32>::open("x.f32")` is `Err`;
`open_vec::<i32>("x.f32")` is `Err`; `ElementType::from_extension
("bf16vecs") == Some(BF16)`; `FacetFormat::from_extension("f32") ==
Some(FloatScalar)`; `dispatch_reader!` on an `.i8vecs` yields an i8
reader (old: u8).

**Done when** the vectordata, veks-pipeline and veks suites pass with
the existing tests unchanged except for `#![allow(deprecated)]` where
they bound on the old traits.

### 13.4 Step 4 — the generic API

**Files.** `src/io.rs`, `src/view.rs`, `src/explore/shared.rs`,
`src/group.rs` (callers), `src/lib.rs` (re-exports, the Common-tasks
table, the typed-metadata doctest), `AGENTS.md`,
`docs/access-datasets-from-rust.md`; new `tests/facet_as.rs`.

**Builds.**
- `VectorReader<T>`: `as_slice`, `conversion`, `native_element_type`
  (EA-8), all defaulted.
- `ScalarReader<T: Element>` (EA-9): `open`, `open_path`, `pub(crate)
  from_storage(storage)`; `VectorReader<T>` with `dim() == 1`.
- `ShardedXvecReader<T>` (`view.rs:2651`) becomes `pub(crate) struct
  ShardedReader<T, R: VectorReader<T>>` with an `open_file: fn(Arc<
  Storage>) -> Result<R, IoError>` field; `reader(i)` calls it; `open`'s
  dimension check, `locate` and `read_into` are unchanged. Type aliases
  keep the old name inside the crate.
- `WindowedVectorReader` delegates `as_slice` (sub-slice over `[start *
  dim .. end * dim]`), `conversion`, `native_element_type`.
- `ConvertingReader<N, T>` (EA-10): `pub fn new(inner: Arc<dyn
  VectorReader<N>>) -> Result<Self, IoError>` (Native and Refused are
  errors here); scratch of `max(1, 64 KiB / (dim * N::SIZE))` records.
- `AnyVectorReader` (EA-12) with `for_each_element!` generating the
  twelve-arm matches; `into_typed` by `Box<dyn Any + Send + Sync>`
  downcast of the inner `Arc`.
- `TestDataView::open_facet_as_type` (default `Err(Error::Other(..))`)
  and `open_facet_native` (EA-15). `GenericTestDataView` overrides the
  first: `match target { … => T::into_any(self.open_uniform::<T>(..)?) }`.
- In `GenericTestDataView`: `open_native<N>` (today's `open_uniform`
  body, plus the scalar branch choosing `ScalarReader::from_storage`
  when `is_scalar_ext`), `open_any_native(facet, name, native)`, and the
  new `open_uniform<T>` of EA-17. `open_variable<T>` requires `Native`.
- `TestDataViewExt` (EA-16) with the blanket impl; re-exported at the
  root. The pinned methods get `#[deprecated(note = "use
  TestDataViewExt::<name>_as")]`; in-crate callers move to `*_as`
  (`explore/shared.rs:309`, `group.rs:659-759`, the `lib.rs` and
  `catalog/resolver.rs` doctests).
- `io::open_vec_as::<T>(source) -> Result<Arc<dyn VectorReader<T>>,
  IoError>` (EA-13); the explorer opens through it and its enum goes.
- `open_facet_typed` (both), `TypedReader`, `TypedElement`,
  `TypedAccessError`, `dispatch_typed!` deprecated with the EA-14 note.
- Five doc examples on `TestDataViewExt::base_vectors_as`, `facet_as`
  (two), `neighbor_indices_as`, and `IoError::ElementMismatch`, each a
  plain ``` block: `tempfile::tempdir()`, hidden lines write the bytes
  and a `dataset.yaml`, `TestDataGroup::load(dir)?.profile("default")`.

**Guards** (fail on old): an `.mvec` facet of 3 × 2 through `facet_as::
<f32>` has `count() == 3` and the exact widened values (old: `count()
== 2`); `facet()` on an `.ivec` facet is `Err` (old: reinterpreted).
Behaviour (new surface): `facet_as::<u8>` on a `.u8` facet —
`conversion() == Native`, `as_slice().len() == count()`, `get_slice(i)`
per element; `facet_as::<i8>` over a byte 200 — `ValueOverflow {
index }` from `get` and from `read_into`; `facet_as::<u16>` and
`::<f32>` widen exactly with `get_slice() == None`; `facet_as::<u8>` on
an f32 facet — `ElementMismatch { file: F32, requested: U8, .. }`; the
scalar series fixture of `tests/sharded_facets.rs:407` through
`facet_as::<u32>` — `count() == 175`, `read_into` across the seam
equals the ordinals, `as_slice() == None`; the sliced series (`:445`)
maps facet ordinal 10 to 990; a window `codes.u8[2..6)` — `as_slice()
== Some(&codes[2..6])` and `count() == 4`; `open_facet_native` on the
u8 facet is `AnyVectorReader::U8` and `into_typed::<i32>` on it is
`ElementMismatch`; `neighbor_indices_as::<i64>` equals
`neighbor_indices()`; `base_vectors()` equals `base_vectors_as::<f32>()`
on a windowed and on a sharded profile; a function over `&Arc<dyn
TestDataView>` calls `facet_as::<f32>` importing only `TestDataViewExt`;
a `.bf16vecs` facet — `facet_as::<bf16>` native with `get_slice` `Some`,
`facet_as::<f32>` equals `to_f32` per element, `facet_as::<f16>`
refused; the explorer opens an `.mvec` and a `.bf16vecs` base.

**Done when** `cargo test --doc -p vectordata` runs the five examples,
`tests/facet_as.rs` and the existing suites pass, and
`tests/discoverability.rs` is green with the new modules.

### 13.5 Step 5 — quantized corpora

**Files.** `src/dataset/facet.rs`, `src/dataset/profile.rs`
(`facet_role`, `derive_sized_profile`), `src/dataset/config.rs`
(`DatasetAttributes`), new `src/dataset/quantization.rs`, `src/view.rs`
(`quantization`, `quantized_base`), `src/lib.rs` (`Error::
FacetCardinality`), `src/dataset/conformance.rs`;
`crates/veks/src/check/profiles.rs`.

**Builds.** EA-23 to EA-26: the three `StandardFacet` variants (`key`,
`code() == None`, `formats`, `basenames`, `PREFERRED_ORDER` after
`BaseVectors`, `from_key`/`from_alias`), `facet_role` → `Windowed`;
`Quantization` (serde, `scheme`, `granularity`, `block`, `codes:
ElementType`, `range: [i64; 2]`) read from `attributes.quantization` at
dataset or profile level, profile winning; `TestDataView::quantization()`
(provided, reads attributes); `QuantizedBase<C>` and
`TestDataViewExt::quantized_base::<C>()`; `Error::FacetCardinality {
facet, expected, found }`; `veks check`'s three checks beside PL-3.

**Guards.** A profile with `base_vectors.u8vecs` and the three columns
opens as one `QuantizedBase<u8>` whose columns' `as_slice()` are `Some`
and equal the written values; `quantized_base::<i8>` on it is
`ElementMismatch`; a `quant_sum.i32` one record short is
`FacetCardinality` at open and a `veks check` failure; `veks check`
fails on columns without the attribute and on the attribute without
columns; conformance refuses `quant_scale.i32`; a sized profile derived
from it windows the three columns with `base_vectors`; the dataset.yaml
renderer writes the facets in `PREFERRED_ORDER` and parses back equal.

**Done when** the vectordata and veks suites pass and `veks check` on a
scratch dataset under `target/test-tmp` reports the new checks.

### 13.6 Step 6 — catalog defects

**Files.** `src/catalog/source.rs` (`parse_catalog_content`,
`LocatedCatalog` and `CatalogBuilder::content` docs), `src/group.rs`
(`of_knn_entries`), `tests/catalog_knn_entries_fallback.rs`.

**Builds.** EA-34. Defect 1: in the `Object` arm, `map.contains_key
("profiles")` → parse as a dataset description (`DatasetConfig` from
content, the same call `of_dataset_yaml` makes), name from `name:` else
the document's directory, one embedded entry at `base`. Defect 3:
`position` found → that entry; `entries.len() == 1` → it; `> 1` →
`Error::AmbiguousDataset` naming the directory and the entries; `0` →
the empty dataset as today. The doc at `group.rs:84` and the test at
`group.rs:714` change with it.

**Guards** (fail on old): `CatalogBuilder::named("x").content(<a
dataset.yaml with name: ds>).build()` lists one entry named `ds` (old:
`profiles` and `attributes`); `--at <path to dataset.yaml>` the same
through `LocatedCatalog`; `TestDataGroup::load(dir)` over a two-entry
`knn_entries.yaml` with no matching name is `AmbiguousDataset` naming
both (old: opens the first); a one-entry file still opens.

### 13.7 Step 7 — pipeline and veks-core consolidation

**Files.** `crates/veks-pipeline/src/pipeline/element_type.rs`, the 26
files with `ElementType::` arms, `crates/veks-core/src/formats/
{facet.rs, mod.rs, reader/mod.rs}`, the 7 consumers of `Facet`,
`crates/veks/src/prepare/{import.rs, wizard.rs}`.

**Builds.** EA-31, EA-32. The pipeline `ElementType` file keeps its
path and becomes `pub use veks_datatypes::ElementType;` plus `pub trait
ElementTypeExt` (`is_float`, `supports_simd_distance`,
`machine_epsilon`, `normalization_threshold`) implemented for it;
`ToF64` deleted, callers use `Element::as_f64`; exhaustive matches gain
`BF16` beside `F16` where a float belongs (SIMD distance: `false` for
bf16 — SK-10). veks-core: `Facet` deleted, `preferred_vec_format
(StandardFacet, ElementType) -> VecFormat` and `upstream_extension
(StandardFacet)` as free functions in `formats/mod.rs`,
`xvec_for_element_size` and `IoMmapReader` deleted; the extension table
in `formats/mod.rs` → `VecFormat`. veks import and wizard tables →
`ElementType`.

**Guards.** The existing suites (the arms are mechanical);
`preferred_vec_format(BaseVectors, I8) == VecFormat::I8vec` and `(…,
U16) == U16vec` (old: `Bvec`, `Svec` by width); `Facet::from_alias
("metadata_indices")` resolves through `StandardFacet` (old: `None` in
veks-core).

### 13.8 Step 8 — ground truth in `veks check`

**Files.** `crates/veks/src/check/profiles.rs`, its tests.

**Builds.** EA-33: for G and for each pre/post-filtered pair, rows ==
query count, distances rows == indices rows, every index `< base
count` by a full `read_into` scan reporting progress through the
check's reporter at the existing cadence; a `-1` padding value (sparse
E) is allowed only where the facet is a post-filtered one.

**Guards.** A G file with one index equal to the base count fails with
the ordinal named; a G with one row more than Q fails; a D whose rows
differ from G fails; a sparse E with `-1` passes; the report shows the
progress line.

### 13.9 Step 9 — docs

**Files.** `docs/sysref/01-data-model.md` (§1.2 bf16 row and the float
scalar column; §1.5 the quant facets; §1.7 the `quantization`
attribute), `docs/sysref/02-api.md` (§2.3 extensions; §2.4 `open_vec`
native-only and `open_vec_as`; §2.5 `VectorReader` additions, `Element`;
§2.6 `*_as` on views; §2.7 rewritten around `facet_as` with the EA-5
table; §2.8 the facet table with the three new rows; §2.9 the path
rule note under selectors; §2.15 element interrogation through
`open_facet_native`), `CHANGELOG.md` under `## Unreleased` (a
"Breaking" subsection with the §12 table; Fixes for the width check,
the typed-reader float zero, the two catalog defects, the veks-io
validation), this SRD's §15 records and `**Status:**`.

**Done when** the sysref tables agree with `ElementType::ALL` and
`StandardFacet::PREFERRED_ORDER`, and the requester is told what
shipped and the defect-2 ruling — after the maintainer releases.

## 14. Settled questions

Decided 2026-10-19:
- **Where the model lives:** a new crate, `veks-datatypes`, not
  vectordata and not veks-io. (Alternatives: vectordata hosts it and
  veks-io keeps a parallel definition; veks-io hosts it and vectordata
  depends on veks-io.)
- **Quantization parameters:** three standard scalar facets plus a typed
  attribute block. (Alternatives: one sidecar container of our own; the
  attribute only, with author-named facets.)
- **Multi-entry `knn_entries` directory with no name match:**
  `AmbiguousDataset`. (Alternatives: first entry with a warning; leave
  as is.)
- **In scope beyond the request:** retire the pipeline's `ElementType`,
  retire veks-core's `Facet`, ground-truth checks in `veks check`. Out:
  the transport-free feature.
- **Lossless or refused.** `i32 → f32` is refused, not widened: the
  requester asked that narrowing be refused at open, and a conversion
  that silently loses precision is a narrowing by another name.

## 15. Implementation

### 15.1 Step 1: `veks-datatypes` (2026-10-19)

**Built:** `crates/veks-datatypes`, depending on `half` only, in the
workspace's members and default members at the lockstep version.
- `ElementType` (EA-1), `#[non_exhaustive]`, with `ALL`, `byte_width`,
  `kind`, `is_integer`, `is_float`, `name`, `from_name`, `Display`;
  `ElementKind`, `FloatFormat`, `Container`.
- One extension table (EA-3), a `const` of 36 rows generated from
  twelve lines, behind `extension`, `aliases`, `parse_extension` and
  `from_extension`. A test proves no spelling is claimed twice and
  every spelling round-trips.
- `Element` (EA-4) for the twelve types, `Value`, `convert`,
  `Overflow` (EA-5), `ElementConversion` with `is_allowed` and
  `converts`, and `conversion_to` as a `const fn`. The table test is
  the literal 12 × 12 matrix.
- `dispatch_element!` (EA-6) binds a type alias per arm, so the body
  names the type as `T`; a wildcard arm keeps it usable from crates
  that see the enum as non-exhaustive.
- `pub use half::{bf16, f16}` (EA-2).

**Decided during the step: canonical and legacy spellings are both
answered.** `extension()` is the typed spelling the data-model doc
calls canonical (`f32vecs`), but every writer in the workspace emits
the historical plural (`fvecs`, `ivecs`, `bvecs`, `svecs`, `mvecs`,
`dvecs`), and the published datasets are named that way. Changing what
writers emit would rename files on regeneration and move every
dataset's content key. So `legacy_extension(container)` returns that
spelling where one exists, and step 2's `VecFormat::preferred_extension`
reads it first. The two are separate questions — what a file may be
called, and what a new file is called — and the table answers both.

**Tests:** 20 unit, 2 doc. `cargo doc` clean with warnings denied.
