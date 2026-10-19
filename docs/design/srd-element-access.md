# SRD — Element types and typed facet access

**Status:** proposed 2026-10-19. Decisions in §15 were taken on
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
validation), this SRD's §16 records and `**Status:**`.

**Done when** the sysref tables agree with `ElementType::ALL` and
`StandardFacet::PREFERRED_ORDER`, and the requester is told what
shipped and the defect-2 ruling — after the maintainer releases.

## 14. Testing plan and coverage

### 14.1 Rules for every test in this SRD

- **Guard first.** A test that exists to pin a fix is run against the
  code before the step, its failure is pasted into the step's §16
  record (the assertion and the old value), and only then does the
  step land. A guard that passes on the old code is not a guard.
- **One fixture layer.** `crates/vectordata/tests/support/fixtures.rs`
  grows generic writers — `write_xvec::<T>(path, records, dim, f)`,
  `write_column::<T>(path, values)`, `write_vvec::<T>`, and
  `write_dataset_yaml(dir, profile_lines)` — and every new test
  builds its files through them. The five doc examples use the same
  byte layouts inline (hidden lines), since a doctest cannot reach
  `tests/support`. veks-io's tests keep their own small writers; its
  crate must not depend on vectordata's test tree.
- **Stateless, local, under `target/test-tmp`.** `tempfile::tempdir()`
  everywhere; no test needs the network; the HTTP cases reuse
  `tests/support/testserver.rs` and are limited to the two paths that
  differ remotely (`as_slice()` is `None` over HTTP; `ElementMismatch`
  is raised before any byte is fetched).
- **Names say the behaviour**, in the suite's sentence style
  (`a_dvec_opened_as_f32_is_refused`), not the function under test.
- **Flips are listed, not silent.** An existing test whose assertion
  this SRD reverses is named below with the reason; it is rewritten in
  the step that changes the rule, never deleted in passing.
- **Deprecated surfaces keep their tests.** `tests/typed_access.rs` and
  the `TypedReader` unit tests get `#![allow(deprecated)]` and run
  unchanged; they are the regression suite for the migration note.

### 14.2 Where the existing tests stand

Counts are `#[test]` functions as of 2026-10-19. "Direct references"
is an approximate grep over test code for the symbol; zero means no
test names it, not that no test reaches it.

| Area | File(s) | Tests | What they cover today | Gap against this SRD |
|---|---|---|---|---|
| Element types | `vectordata/src/typed_access.rs` | 9 | extension detection, u8/i32 native, u8→i32 widening, u8→i8 overflow, narrowing refused, bounds, empty | floats absent entirely; no test notices `read_native_value` returning 0 |
| Typed scalar access | `vectordata/tests/typed_access.rs` | 32 | every integer native read, 7 widenings, 7 cross-sign cases, 4 narrowings, ivec record access, dataset-level open and dispatch | integers only; `get_native_slice` referenced once; nothing opens a float through it |
| Raw readers | `vectordata/src/io.rs` | 14 | vvec sidecar sentinel rules, scalar extension set (**pins integers-only**: `the_scalar_extensions_are_exactly_the_integer_widths`), sharded reads across seams | `validate_element_for_source`, `VariableLengthRecords`, `prefetch_range`, `advise_sequential`: zero direct references; no test opens a file as the wrong width |
| View and series | `vectordata/src/view.rs` | 27 | window parsing, range coalescing, offset-index handles (including the width check on a reused handle), file-cap eviction | nothing on `open_uniform`'s type dispatch; `facet()` on a non-f32 facet untested |
| Sharded facets | `vectordata/tests/sharded_facets.rs` | 50 | series open/refuse, seam reads, windows, typed scalar series (`typed_access_reads_a_scalar_series`, `_sliced_series`), plans, derive, version gates | all at f32/i32/u32; the scalar-series tests go through `TypedReader` and must also pass through `facet_as` |
| Series consumers | `vectordata/tests/series_consumers.rs` | 13 | derive over series, typed reader over series, native read on series | same: `TypedReader` path only |
| Record facets | `vectordata/tests/record_facets.rs` | 23 | slab shape, `facet_shape`, the Elements/Records refusal both ways | unchanged by this SRD; the shape refusal must still fire through `open_facet_as_type` |
| Facet spec | `vectordata/src/dataset/facet.rs`, `conformance.rs` | 20 + 7 | key/alias/code round trips, `classify`, format validity, series conformance | no float scalar format; no quant facets; `facet_role` and `derive_sized_profile` have zero direct references |
| Explorer reader | `vectordata/src/explore/shared.rs` | 12 | clump/prefetch arithmetic, sorted mirror | nothing opens a file through `AnyVectorReader`; no f16 base test |
| Group / knn dir | `vectordata/src/group.rs` | 15 | file URI, built entries, **knn directory picks the entry named after it, else the first** (`a_knn_entries_directory_opens_the_entry_named_after_it`, second half — flips in step 6), defaults base_url, local-first | the multi-entry ambiguity has no test because it was not an error |
| Catalog content | `vectordata/src/catalog/source.rs`, `tests/catalog_knn_entries_fallback.rs` | 5 + 19 | built vs located equality, diagnostics, arbitrary yaml filenames, knn fallbacks | no test gives a `dataset.yaml` as catalog content; `parse_catalog_content` has zero direct references |
| veks-io readers | `veks-io/src/xvec/mmap.rs`, `scalar/mmap.rs`, `lib.rs` | 3 + 6 + 31 | f32/i32 mmap round trips, empty file, scalar round trips and size checks, streaming round trips for every xvec type, `truncated_file_stops_gracefully`, `corrupted_dim_mid_file`, `mmap_truncated_file` | `xvec/reader.rs` and `scalar/reader.rs` have no unit tests of their own; no negative-dimension test; no alignment test; `open_dvec`/`open_bvec` zero direct references; `preferred_extension` zero |
| veks-io formats | `veks-io/src/format.rs` | 21 | extension round trips, **scalar set pins integers-only** (`test_scalar_extensions`), detect, writability | no bf16; no float scalars |
| veks-core facets | `veks-core/src/formats/facet.rs` | 14 | key/alias/code round trips, **`test_preferred_formats_by_element_size` pins the width-keyed rule** (flips in step 7) | the `metadata_indices` alias gap has no test because it was not noticed |
| Pipeline element types | `veks-pipeline/src/pipeline/element_type.rs` | 9 | extension parsing (including `.f32` scalar), `is_float`, sizes, epsilon | `dispatch_reader!` has zero direct references, so its I8→u8 and U16→i16 opens were never observed |
| `veks check` | `veks/src/check/profiles.rs`, `tests/check_and_import.rs` | 7 + 87 | version, selectors, tags, predicate groups; e2e import/check | no ground-truth cardinality or bounds test; `integrity.rs` has no unit tests (covered through e2e) |

Three existing tests assert what this SRD reverses and are rewritten
in their step: `the_scalar_extensions_are_exactly_the_integer_widths`
(step 3: floats join), `test_scalar_extensions` in veks-io (step 2),
`test_preferred_formats_by_element_size` (step 7: keyed by type), and
the second half of `a_knn_entries_directory_opens_the_entry_named_
after_it` (step 6: ambiguity is an error).

### 14.3 Tests per step

Each entry: guards (fail on the code before the step), behaviour
(new surface), flips (existing tests rewritten), fixtures.

**Step 1 — `veks-datatypes`** (landed; 20 unit, 2 doc):
`the_conversion_table` (the literal matrix), `native_is_the_diagonal_
and_cross_sign_is_symmetric`, `widening_is_exact_at_the_edges`,
`cross_sign_checks_each_value`, `the_integer_to_float_boundary_is_the_
significand`, `f16_and_bf16_never_convert_to_each_other`; the
extension table round trips, aliases, uniqueness, unknowns; byte
round trips and little-endian layout for all twelve; `from_value`
range refusals; dispatch at every width.

**Step 2 — veks-io.**
- Guards: `a_negative_dimension_is_rejected` (header `-4`; old: `dim`
  becomes `usize::MAX - 3`), `a_dimension_that_overflows_the_record_is_
  rejected` (old: wrapping arithmetic), `a_truncated_file_is_rejected_by_
  the_streaming_reader` (old: tail dropped), `a_misaligned_f64_record_
  has_no_slice` (old: a misaligned `&[f64]`), `a_bvecs_file_does_not_open_
  as_f32` (old: `Ok`).
- Behaviour: `every_type_round_trips_through_mmap` (twelve types,
  `get`, `get_slice`, `read_into`), `a_scalar_column_is_one_slice`
  (`as_slice` equals the written column, every type), `bf16_vectors_
  round_trip_through_the_writer`, `f32_columns_round_trip_through_the_
  scalar_writer`, `the_deprecated_constructors_open_the_same_file`,
  `vec_format_names_every_element_type_and_layout` (`of` × `element_
  type` round trip over `ALL × Container`), `preferred_extension_keeps_
  the_historical_spelling` (`Fvec` → `fvecs`, `Bf16vec` → `bf16vecs`).
- Flips: `test_scalar_extensions` admits the float columns.
- Fixtures: a `write_xvec<T>`/`write_column<T>` pair local to veks-io's
  tests.

**Step 3 — vectordata model and width check.**
- Guards: `an_ivec_does_not_open_as_f32_by_path`, `an_fvec_does_not_open_
  as_i32`, `a_dvec_opened_as_f32_is_a_mismatch_not_a_short_file`
  (asserts `ElementMismatch { file: F64, requested: F32 }`; old:
  `count() == 9`), `a_u32vvec_does_not_open_as_i32`, `a_float_column_is_
  refused_by_the_typed_reader` (old: zeros), `open_vec_refuses_a_
  column`, `dispatch_reader_opens_an_i8_file_as_i8` (in veks-pipeline;
  old: `XvecReader<u8>`).
- Behaviour: `bf16_and_the_float_columns_are_recognised_extensions`,
  `a_float_column_has_its_own_facet_format` (`FacetFormat::from_
  extension("f32") == FloatScalar`; `"i32"` stays `ScalarPacked`),
  `the_facet_format_lists_derive_from_the_element_table` (every
  `ALL × Container` spelling lands in exactly one `FacetFormat`),
  `the_mismatch_error_names_both_types_and_the_reason`.
- Flips: `the_scalar_extensions_are_exactly_the_integer_widths` →
  `the_scalar_extensions_are_every_element_type`.
- Fixtures: `write_xvec::<T>`, `write_column::<T>`, `write_vvec::<T>`,
  `write_dataset_yaml` in `tests/support/fixtures.rs`; `write_fvec`
  becomes `write_xvec::<f32>`.

**Step 4 — the generic API** (new `tests/facet_as.rs`).
- Guards: `an_f16_facet_widens_to_f32_with_the_right_count` (old:
  `count() == 2` and garbage), `facet_refuses_an_index_facet` (old:
  reinterpreted bits).
- Behaviour, native: `a_u8_facet_reads_in_place` (`conversion() ==
  Native`, `native_element_type() == Some(U8)`, `as_slice().len() ==
  count()`, `get_slice(i)` per element), `a_bf16_facet_reads_in_place`
  (`get_slice` `Some`), `an_i32_index_facet_reads_in_place_as_i32`,
  `base_vectors_equals_base_vectors_as_f32_on_a_windowed_profile`, the
  same `_on_a_sharded_profile`.
- Behaviour, converting: `a_u8_facet_widens_to_u16_i32_and_f32`
  (`get_slice() == None`, values exact), `a_bf16_facet_widens_to_f32_
  exactly` (per-element `to_f32`), `a_u8_facet_as_i8_reports_the_first_
  overflow_by_ordinal` (`get` and `read_into` both), `an_i32_index_facet_
  as_u32_fails_on_a_negative_sentinel`, `neighbor_indices_as_i64_equals_
  neighbor_indices`.
- Behaviour, refused: `an_f32_facet_as_u8_is_a_mismatch`, `an_f16_facet_
  as_bf16_is_a_mismatch`, `into_typed_at_the_wrong_type_is_a_mismatch`.
- Behaviour, columns and series: `a_scalar_series_reads_through_facet_
  as` (the `sharded_facets.rs:407` fixture: `count() == 175`, `read_
  into` across the seam, `as_slice() == None`), `a_sliced_scalar_series_
  maps_ordinal_10` (`:445`), `a_window_over_a_column_is_a_sub_slice`
  (`codes.u8[2..6)`), `a_remote_column_has_no_slice` (testserver).
- Behaviour, dispatch: `open_facet_native_names_the_files_type`,
  `the_extension_trait_works_on_a_dyn_view` (compile-level: a fn over
  `&Arc<dyn TestDataView>` importing only `TestDataViewExt`), `a_record_
  facet_is_still_refused_as_elements` (shape check through the new
  path), `open_vec_as_widens_a_raw_file`, `the_explorer_opens_an_f16_and_
  a_bf16_base`.
- Doc tests (five, each a plain fenced block): `base_vectors_as`
  (native, window, `read_into`), `facet_as` (u8 codes zero-copy),
  `facet_as` (bf16 native then widened), `neighbor_indices_as` (bounds
  against `base_count`), `IoError::ElementMismatch` (the refusal).
- Flips: none; `tests/typed_access.rs` gets `#![allow(deprecated)]`.
- Also: `tests/discoverability.rs` must stay green — the new modules
  carry the `Library API.` first line and print nothing.

**Step 5 — quantized corpora.**
- Guards: none fail on old code (new surface); the cardinality checks
  are proven by construction below.
- Behaviour: `a_quantized_profile_opens_as_one_base` (`QuantizedBase<u8>`
  with `Some` slices equal to the written columns and `scheme.codes ==
  U8`), `quantized_base_at_the_wrong_code_type_is_a_mismatch`, `a_short_
  quant_sum_is_a_cardinality_error` (at open, naming the facet and both
  counts), `a_sized_profile_windows_the_columns_with_the_base`, `the_
  renderer_writes_the_quant_facets_after_the_base_and_parses_back`,
  `conformance_refuses_an_integer_scale` (`quant_scale.i32`), `the_
  quantization_attribute_round_trips_through_serde` (every granularity,
  `codes` as a name); in `veks check`: `columns_without_the_attribute_
  are_reported`, `the_attribute_without_columns_is_reported`, `a_column_
  count_that_disagrees_with_the_base_is_reported`, `a_consistent_
  quantized_dataset_passes`.
- Fixtures: `write_quantized_profile(dir, n, dim)` in
  `tests/support/fixtures.rs` building codes and the three columns.

**Step 6 — catalog defects.**
- Guards: `a_dataset_yaml_as_catalog_content_lists_one_dataset` (old:
  entries `profiles` and `attributes`), `a_dataset_yaml_at_a_catalog_
  path_lists_one_dataset` (through `--at`/`LocatedCatalog`; old: the
  same two), `a_knn_directory_with_two_entries_and_no_match_is_
  ambiguous` (old: opens the first).
- Behaviour: `a_knn_directory_with_one_entry_opens_it`, `the_ambiguity_
  names_the_directory_and_every_entry`.
- Flips: the second half of `a_knn_entries_directory_opens_the_entry_
  named_after_it` becomes the ambiguity guard; `test_data_group_picks_
  dataset_matching_directory_name` is unchanged.

**Step 7 — pipeline and veks-core consolidation.**
- Guards: `preferred_vec_format_is_keyed_by_type` (`I8` → `I8vec`,
  `U16` → `U16vec`; old: `Bvec`, `Svec`), `metadata_indices_resolves_in_
  veks_core` (old: `None`).
- Behaviour: the existing pipeline, veks and veks-core suites (the arm
  changes are mechanical); `the_pipeline_element_type_is_the_shared_one`
  (type identity: `veks_pipeline::pipeline::ElementType` is
  `veks_datatypes::ElementType`), `bf16_is_a_float_without_simd_distance`.
- Flips: `test_preferred_formats_by_element_size` → by type.

**Step 8 — ground truth in `veks check`.**
- Behaviour: `an_index_at_the_base_count_is_reported_with_its_ordinal`,
  `a_ground_truth_with_more_rows_than_queries_is_reported`, `distances_
  whose_rows_differ_from_the_indices_are_reported`, `a_sparse_post_
  filtered_truth_allows_minus_one`, `a_pre_filtered_truth_does_not`,
  `the_scan_reports_progress` (the reporter sees a progress line on a
  fixture large enough to cross the cadence, with the cadence lowered
  through the existing knob rather than a large file).

**Step 9 — docs.** `cargo test --doc` across the published crates with
warnings denied; the sysref tables are checked by two small tests:
`the_element_table_in_the_docs_matches_ALL` (parses the §1.2 rows and
compares with `ElementType::ALL × Container`) and `the_facet_table_in_
the_docs_matches_PREFERRED_ORDER`.

### 14.4 Requirement to test

| Requirement | Tests |
|---|---|
| EA-1, EA-2 | `every_type_has_a_row_per_container_and_round_trips`, `bf16_is_not_f16_and_not_u16`, `names_round_trip_and_widths_match_the_kind`, `little_endian_is_the_encoding` |
| EA-3 | `canonical_spellings_are_the_type_name_plus_the_layout`, `the_historical_letters_name_the_types_they_always_did`, `no_spelling_is_claimed_twice`, `unknown_and_ambiguous_spellings_are_none` |
| EA-4 | `bytes_round_trip_for_every_type`, `integers_refuse_what_does_not_fit`, `floats_take_integers_exactly_where_they_fit`, `values_and_f64_agree` |
| EA-5 | `the_conversion_table`, `native_is_the_diagonal_and_cross_sign_is_symmetric`, `widening_is_exact_at_the_edges`, `cross_sign_checks_each_value`, `the_integer_to_float_boundary_is_the_significand`, `f16_and_bf16_never_convert_to_each_other` |
| EA-6 | `dispatch_reaches_every_type_at_its_own_width`, the `dispatch_element!` doc test |
| EA-7 | the existing suites compiling under the blanket traits; `tests/typed_access.rs` under `allow(deprecated)` |
| EA-8 | `a_u8_facet_reads_in_place`, `a_u8_facet_widens_to_u16_i32_and_f32`, `a_window_over_a_column_is_a_sub_slice`, `a_remote_column_has_no_slice` |
| EA-9 | `a_scalar_column_is_one_slice` (veks-io), `a_scalar_series_reads_through_facet_as`, `a_sliced_scalar_series_maps_ordinal_10`, `open_vec_refuses_a_column` |
| EA-10 | `a_u8_facet_as_i8_reports_the_first_overflow_by_ordinal`, `an_i32_index_facet_as_u32_fails_on_a_negative_sentinel`, `a_bf16_facet_widens_to_f32_exactly` |
| EA-11 | `an_ivec_does_not_open_as_f32_by_path`, `an_fvec_does_not_open_as_i32`, `a_dvec_opened_as_f32_is_a_mismatch_not_a_short_file`, `a_u32vvec_does_not_open_as_i32`, `the_mismatch_error_names_both_types_and_the_reason` |
| EA-12 | `open_facet_native_names_the_files_type`, `into_typed_at_the_wrong_type_is_a_mismatch` |
| EA-13 | `open_vec_as_widens_a_raw_file`, `the_explorer_opens_an_f16_and_a_bf16_base` |
| EA-14 | `a_float_column_is_refused_by_the_typed_reader`; the deprecated suite unchanged |
| EA-15, EA-16 | `the_extension_trait_works_on_a_dyn_view`, `base_vectors_equals_base_vectors_as_f32_*`, `neighbor_indices_as_i64_equals_neighbor_indices` |
| EA-17 | `a_record_facet_is_still_refused_as_elements`, `an_f32_facet_as_u8_is_a_mismatch`, `an_f16_facet_as_bf16_is_a_mismatch` |
| EA-18 | `facet_refuses_an_index_facet`, `an_f16_facet_widens_to_f32_with_the_right_count` |
| EA-19 | the sharded and series suites at every type; `a_scalar_series_reads_through_facet_as` |
| EA-20 | `a_float_column_has_its_own_facet_format`, `the_facet_format_lists_derive_from_the_element_table` |
| EA-21 | `bf16_and_the_float_columns_are_recognised_extensions`; `the_element_table_in_the_docs_matches_ALL` |
| EA-22 | `vec_format_names_every_element_type_and_layout`, `preferred_extension_keeps_the_historical_spelling`, `bf16_vectors_round_trip_through_the_writer`, `f32_columns_round_trip_through_the_scalar_writer` |
| EA-23 | `the_renderer_writes_the_quant_facets_after_the_base_and_parses_back`, `a_sized_profile_windows_the_columns_with_the_base`, `conformance_refuses_an_integer_scale` |
| EA-24 | `the_quantization_attribute_round_trips_through_serde` |
| EA-25 | `a_quantized_profile_opens_as_one_base`, `quantized_base_at_the_wrong_code_type_is_a_mismatch`, `a_short_quant_sum_is_a_cardinality_error` |
| EA-26 | the four `veks check` quantization tests |
| EA-28 | the five veks-io guards, `every_type_round_trips_through_mmap`, `the_deprecated_constructors_open_the_same_file` |
| EA-29 | `a_scalar_column_is_one_slice` |
| EA-31 | `dispatch_reader_opens_an_i8_file_as_i8`, `the_pipeline_element_type_is_the_shared_one`, `bf16_is_a_float_without_simd_distance` |
| EA-32 | `preferred_vec_format_is_keyed_by_type`, `metadata_indices_resolves_in_veks_core` |
| EA-33 | the six step-8 tests |
| EA-34 | the three step-6 guards and two behaviour tests |

EA-27 and EA-30 are non-goals stated as requirements and have no test.

### 14.5 Measured coverage

`cargo-llvm-cov` 0.9.1 with the `llvm-tools` component was installed
on 2026-10-19 (the maintainer's call). The baseline and the per-step
comparison are:

```
RUSTFLAGS="-C debuginfo=1" cargo llvm-cov --workspace \
  --exclude veks-studies --exclude tools --exclude vectordata-rs \
  --lcov --output-path target/coverage/<step>.lcov
```

**`RUSTFLAGS` must be set.** `.cargo/config.toml` links every binary
with mold, and mold 1.0.3 drops the profile runtime's section symbols:
the instrumented binaries run, pass, and write no profile data (one
`.profraw` with zero functions for the whole workspace; every file
reported at 0%). Any `RUSTFLAGS` value replaces the config's
`rustflags`, which takes mold out of the link; `-C debuginfo=1` is a
harmless one. The flag is noted here rather than fixed in the config
because mold is the right linker for every other build.

Per-file numbers come from the lcov file (`SF`/`LF`/`LH`/`FNF`/`FNH`
records), since `cargo llvm-cov report` cannot take the `--exclude`
list.

**Baseline, 2026-10-19, after step 1** (the whole set: 75.6% of lines,
74.0% of functions, 323 test binaries). Line coverage, then function
coverage, from each file's own crate tests and the integration suites:

| File | Lines | Functions |
|---|---|---|
| `veks-datatypes/src/conversion.rs` | 97.9% (138/141) | 100% (14/14) |
| `veks-datatypes/src/element.rs` | 100% (107/107) | 100% (20/20) |
| `veks-datatypes/src/element_type.rs` | 97.9% (142/145) | 96.0% (24/25) |
| `vectordata/src/io.rs` | 82.9% (840/1013) | 81.6% (111/136) |
| `vectordata/src/typed_access.rs` | 80.2% (308/384) | 64.1% (41/64) |
| `vectordata/src/view.rs` | 84.7% (1992/2353) | 73.4% (246/335) |
| `vectordata/src/explore/shared.rs` | 42.4% (209/493) | 36.6% (30/82) |
| `vectordata/src/group.rs` | 93.2% (451/484) | 95.0% (57/60) |
| `vectordata/src/catalog/source.rs` | 92.1% (197/214) | 94.3% (33/35) |
| `vectordata/src/catalog/knn_entries.rs` | 94.9% (409/431) | 92.7% (51/55) |
| `vectordata/src/dataset/facet.rs` | 93.3% (306/328) | 95.7% (45/47) |
| `vectordata/src/dataset/conformance.rs` | 88.2% (202/229) | 100% (21/21) |
| `vectordata/src/dataset/profile.rs` | 95.5% (2321/2431) | 90.0% (216/240) |
| `vectordata/src/dataset/shards.rs` | 93.8% (1232/1313) | 82.4% (126/153) |
| `veks-io/src/format.rs` | 89.2% (298/334) | 89.7% (35/39) |
| `veks-io/src/lib.rs` | 94.6% (508/537) | 91.4% (53/58) |
| `veks-io/src/xvec/mmap.rs` | 72.6% (77/106) | 61.1% (11/18) |
| `veks-io/src/xvec/reader.rs` | 87.0% (67/77) | 61.5% (8/13) |
| `veks-io/src/xvec/writer.rs` | 90.2% (46/51) | 71.4% (5/7) |
| `veks-io/src/scalar/mmap.rs` | 90.1% (109/121) | 75.0% (15/20) |
| `veks-io/src/scalar/reader.rs` | 90.0% (54/60) | 66.7% (6/9) |
| `veks-io/src/scalar/writer.rs` | 85.2% (23/27) | 60.0% (3/5) |
| `veks-io/src/scalar/convert.rs` | 98.6% (352/357) | 100% (57/57) |
| `veks-core/src/formats/facet.rs` | 92.7% (178/192) | 88.0% (22/25) |
| `veks-core/src/formats/mod.rs` | 65.3% (143/219) | 91.3% (21/23) |
| `veks-pipeline/src/pipeline/element_type.rs` | 81.1% (137/169) | 46.3% (19/41) |
| `veks/src/check/profiles.rs` | 86.6% (356/411) | 87.5% (35/40) |
| `veks/src/check/integrity.rs` | 82.6% (76/92) | 50.0% (4/8) |

What the numbers say that §14.2 did not: the explorer's reader path
is the least exercised file this SRD touches (42%); the pipeline's
`ElementType` has fewer than half its functions called (the
`dispatch_reader!` arms); veks-io's xvec mmap reader is at 61% of
functions (the per-type constructors other than f32/i32 are never
called). Each step's record in §16 states the same files after it
lands.

Targets this SRD holds itself to, by file, line coverage from the
crate's own tests:
- `veks-datatypes`: every line of `conversion.rs` and
  `element_type.rs` (the tables are the product).
- `vectordata/src/io.rs`: every open path (`open`, `open_path`,
  `require_native`), `ScalarReader`, `ConvertingReader::read_into`
  including the overflow branch, `AnyVectorReader::into_typed` both
  outcomes.
- `vectordata/src/view.rs`: `open_uniform` on all four conversions,
  `open_any_native` on every type (through the twelve-type facet
  fixture), `open_facet_as_type`'s default body.
- `veks-io/src/xvec/mmap.rs` and `scalar/mmap.rs`: every validation
  branch.
- `veks/src/check/profiles.rs`: every new report line.

Until the tool is present, the proxy is §14.4: every requirement names
the tests that exercise it, and every guard's failure on the old code
is recorded in §16.

## 15. Settled questions

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

## 16. Implementation

### 16.1 Step 1: `veks-datatypes` (2026-10-19)

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

### 16.2 Step 2: veks-io on the model (2026-10-19)

**Guards, run against the code before the step** (`crates/veks-io/
tests/element_model.rs`, first written in the old API's spellings):
- `a_negative_dimension_is_rejected`: panicked at `xvec/mmap.rs:69:31`
  — `4 + dim * elem_size` overflowed after `-4` was cast to `usize`.
- `a_truncated_file_is_rejected_by_the_streaming_reader`: the 16-byte
  file opened `Ok` and the tail was dropped.
- `a_misaligned_f64_record_has_no_slice`: SIGABRT — "unsafe
  precondition(s) violated: slice::from_raw_parts requires the pointer
  to be aligned" at `xvec/mmap.rs:99`. The undefined behaviour was
  real, not theoretical: the debug build aborts on it.
- `a_bvecs_file_does_not_open_as_f32`: a 340-byte `.bvecs` opened `Ok`
  as f32 with `count() == 5`.

**Built:**
- veks-io depends on `veks-datatypes` and re-exports the model
  (`ElementType`, `Element`, `ElementConversion`, `Container`, `bf16`,
  `f16`) at its root. New `src/element.rs` holds the one
  `MmapError`, `require_native::<T>(path, Container)` (the check EA-11
  names, applied here to both mmap readers), an alignment test, and the
  little-endian decode used by every bulk read.
- `MmapReader<T: Element>::open` (EA-28): dimension must be
  `1..=i32::MAX`, record size by checked arithmetic, remainder
  refused, extension must name `T` in the `Vec` layout. `get_slice ->
  Option<&[T]>` (`None` past the end or misaligned), `get ->
  Option<Vec<T>>`, `read_into(start, &mut [T]) -> Result<usize>`,
  `Debug`. The six per-type constructors are deprecated wrappers.
- `ScalarMmapReader<T: Element>::open` (EA-29): same check in the
  `Scalar` layout; `get -> Option<T>`, `get_slice`, `as_slice` (the
  whole column), `read_into`, `Debug`. The eight per-type constructors
  are deprecated wrappers.
- The streaming `xvec::reader::open` and `probe` share one `shape`
  check: positive dimension, checked record size, whole-file
  remainder.
- `VecFormat` (EA-22): `Bf16vec`, `ScalarF16`, `ScalarBf16`,
  `ScalarF32`, `ScalarF64`; `ALL`; `element_type`, `container`, `of`;
  `is_xvec`, `is_scalar`, `is_integer`, `is_signed`, `element_size`,
  `from_extension`, `data_type_name` derived from the element model;
  `#[non_exhaustive]`. `preferred_extension` keeps every historical
  spelling (§16.1's decision) through `legacy_extension`. The writers
  take the new formats with no change of their own.
- `scalar::convert` rebuilt on `Element`: `convert_scalars` dispatches
  both types through `dispatch_element!` and converts with
  `veks_datatypes::convert`; `ScalarType` is a deprecated alias of
  `ElementType`; `ConvertError` is an enum (`Overflow`, `Refused`,
  `Length`). It still narrows integers when every value fits — a
  converter writes a new column, which is not what a reader does — and
  refuses float-to-integer, float narrowing, and lossy integer-to-float.

**Decided during the step:**
- **One `MmapError`.** The two readers had two enums with different
  variants (`Io`/`InvalidSize`/`EmptyFile` against
  `Open`/`Mmap`/`InvalidFormat`). There is one now, with
  `ElementMismatch` and `OutOfBounds` added; the scalar reader's old
  variants fold into `InvalidFormat` with the size in the message.
  Semver-breaking for a caller matching the scalar reader's errors.
- **The streaming reader refuses a short file at open**, as the mmap
  reader always did, instead of yielding the records before the cut.
  Two lib tests flipped: `truncated_file_stops_gracefully` is now
  `a_truncated_file_is_refused`, and `corrupted_dim_mid_file` keeps a
  whole-record file size so it still tests a bad mid-file header. A
  mid-file header that disagrees still ends iteration with `None`
  (`VecSource::next_record` has no error channel; unchanged).
- **A file without an extension, or with a variable-length one, is
  refused by the mmap readers.** They read a file at the type its
  extension names; a path that names none has nothing to check
  against. vectordata's readers over storage (step 3) decide for
  themselves.

**Flips:** `test_scalar_extensions` admits the float columns (§14.2).

**Tests:** 123 unit, 13 in `tests/element_model.rs` (the four guards
and nine behaviour tests, among them `every_type_round_trips_through_
mmap` and `a_scalar_column_is_one_slice` over all twelve types), 7
doc; veks-core's 131 pass unchanged; the workspace compiles with no
new warnings. `cargo doc` clean with warnings denied. The example
`examples/mmap_access.rs` moved to the generic open and shows
`read_into`.

**Coverage after the step** (the §14.5 command, whole workspace;
baseline → after, lines, then functions after):

| File | Baseline | After | Functions |
|---|---|---|---|
| `veks-io/src/element.rs` | — | 97.9% (46/47) | 100% (8/8) |
| `veks-io/src/format.rs` | 89.2% | 98.7% (385/390) | 98.0% (49/50) |
| `veks-io/src/xvec/mmap.rs` | 72.6% | 95.4% (166/174) | 95.8% (23/24) |
| `veks-io/src/xvec/reader.rs` | 87.0% | 90.0% (81/90) | 62.5% (10/16) |
| `veks-io/src/scalar/mmap.rs` | 90.1% | 98.0% (145/148) | 94.7% (18/19) |
| `veks-io/src/scalar/convert.rs` | 98.6% | 96.6% (199/206) | 98.4% (60/61) |
| `veks-io/src/lib.rs` | 94.6% | 94.6% (507/536) | 91.4% (53/58) |

The whole set is unchanged at 75.6%. What is left uncovered in the
readers is the record-size overflow branch, which a 64-bit host cannot
reach, and the `Open`/`Mmap` display arms.

### 16.3 Step 3: vectordata's element model and the width check (2026-10-19)

**Guards, run against the code before the step** (`crates/vectordata/
tests/element_model.rs`, first written in the old API's spellings, and
one in `veks-pipeline`'s `element_type.rs` tests):
- `an_ivec_does_not_open_as_f32_by_path` and
  `an_fvec_does_not_open_as_i32`: both opened `Ok`.
- `a_dvec_opened_as_f32_is_a_mismatch_not_a_short_file`: "opened an
  f64 file as f32 with count 9" — five records of four f64 read as
  nine records of garbage.
- `a_u32vvec_does_not_open_as_i32`: opened `Ok` (the width-only check
  passed).
- `a_float_file_is_refused_by_the_typed_reader`: `TypedReader::<i32>::
  open("base.fvec")` was `Ok`; every value read as 0. (§14.3 named this
  guard with a `.f32` column; the old code refused that for the wrong
  reason — an unknown extension — so the guard uses an `.fvec`, where
  the zeros were real.)
- `bf16_and_the_float_columns_are_recognised_extensions`: `None` for
  all three.
- `dispatch_reader_opens_an_i8_file_as_i8`: `XvecReader<u8>` where
  `XvecReader<i8>` is named.

**Built:**
- `vectordata` depends on `veks-datatypes`. `typed_access::ElementType`
  is the shared enum (EA-1: `BF16`, `#[non_exhaustive]`); `Element`,
  `ElementConversion`, `ElementKind`, `FloatFormat`, `Container`,
  `Value`, `bf16` and `f16` are re-exported there and at the root
  (`io` re-exports `Element` and `ElementType` too). The path and URL
  forms (`from_path`, `from_url`, `is_scalar_format`) and the deprecated
  `can_open_as` are an extension trait, `ElementTypeExt`, since they
  cannot be inherent on a foreign type.
- `io::VvecElement` and `typed_access::TypedElement` are deprecated
  blanket traits over `Element` (EA-7); every bound in `io.rs` and
  `view.rs` is `T: Element` and every `ELEM_SIZE` is `SIZE`. The eleven
  hand-written `VvecElement` impls and `impl_typed_element!` are gone.
- `io::infer_elem_size`, `is_vvec_ext` (still counting the bare
  `ivec(s)`) and `is_scalar_ext` answer from `ElementType::
  parse_extension`. The three extension tables in `typed_access.rs`,
  `io.rs` and `datasets/filter.rs` are gone; `filter.rs` names a file's
  type through a new `ElementType::long_name` (`float32`, `bfloat16`),
  which veks-io's `data_type_name` now uses as well.
- `io::require_native::<T>(source) -> (ElementType, Container)` (EA-11)
  replaces `validate_element_for_source`; `XvecReader::open` and
  `open_path` (which checked nothing) call it and also refuse a column;
  `IndexedVvecReader::open_path` and `validate_source` call it in place
  of the width comparison. `IoError` is `#[non_exhaustive]` and gains
  `ElementMismatch { file, requested, reason }` and `ValueOverflow {
  index, value, target }`. The `reason` says what to do: "open it
  through a dataset view to widen" for a pair the table allows, "no
  conversion between the two" otherwise.
- The view's `open_uniform` calls `require_native::<T>` before
  `from_storage`, so `base_vectors()` on an `.mvec` facet is now an
  `ElementMismatch` where it reinterpreted bits; step 4 makes it widen.
- `XvecReader`'s unchecked inherent `get_slice` is one generic impl
  over `Element` in place of the f32 one plus a macro for five more
  types. Generic code that wants the checked form names it
  (`VectorReader::get_slice(reader, i)`); the sharded reader did.
- `TypedReader` (EA-14's guard): `check_open` consults
  `conversion_to`, so a float file is refused as an integer, and —
  since the reader now decodes through `Element::to_value` — opens at
  its own float type or widened (`TypedReader::<f64>` over an `.fvec`
  reads the values). `is_native()` is by type, not width. `dispatch_
  typed!` reaches the float types and carries a wildcard arm.
- `FacetFormat` (EA-20): `FloatScalar`; `#[non_exhaustive]`; `ALL`;
  `of(Container, ElementType)`; every `extensions()` list is derived
  from `ElementType::ALL × Container` (each type's canonical spelling,
  then its aliases) and held in a `LazyLock`. `IntegerVarXvec` thereby
  admits every integer width, where it listed 32-bit only.
- `veks-pipeline`'s `dispatch_reader!` opens i8, u16, u32, u64 and i64
  files as themselves (EA-31's first half; the enum itself moves in
  step 7); `compute_knn_distances` bounds on `Element`.

**Decided during the step:**
- **`ivec(s)` stays a variable-length spelling for the vvec reader**
  while parsing as `(I32, Vec)` in the model — both readers accept it,
  as before, because old datasets used it both ways.
- **The raw readers refuse a column** (`open_vec::<f32>("x.f32")` is
  `InvalidFormat` naming the typed reader) rather than reading its
  first four bytes as a dimension.
- **`FacetFormat::extensions()` leads with the typed spelling**
  (`f16vecs` before `mvecs`): it lists what a file may be called;
  what a new file is called is veks-io's `preferred_extension`, which
  keeps the historical spelling (§16.1).

**Flips:** `the_scalar_extensions_are_exactly_the_integer_widths` →
`the_scalar_extensions_are_every_element_type` (§14.2).

**Tests:** vectordata 896 unit and every integration suite, with the
10 in `tests/element_model.rs` (six guards and four behaviour tests);
veks-io 123; veks-core 131; the veks-pipeline, veks and vecd suites
pass unchanged but for the dispatch guard. `cargo doc` clean with
warnings denied for the three crates; clippy clean on the changed
files.

**Coverage after the step** (the §14.5 command, whole workspace:
75.6% -> 75.6%; baseline → after, lines, then functions after):

| File | Baseline | After | Functions |
|---|---|---|---|
| `vectordata/src/io.rs` | 82.9% | 85.3% (824/966) | 85.6% (107/125) |
| `vectordata/src/typed_access.rs` | 80.2% | 81.1% (279/344) | 66.1% (39/59) |
| `vectordata/src/view.rs` | 84.7% | 84.7% (1992/2353) | 73.4% (246/335) |
| `vectordata/src/dataset/facet.rs` | 93.3% | 93.5% (319/341) | 96.4% (54/56) |
| `vectordata/src/datasets/filter.rs` | 83.5% | 84.6% (871/1029) | 86.2% (106/123) |
| `veks-pipeline/src/pipeline/element_type.rs` | 81.1% | 81.6% (146/179) | 48.8% (21/43) |
| `veks-datatypes/src/element_type.rs` | 97.9% | 100.0% (162/162) | 100.0% (26/26) |

### 16.4 Step 4: the generic facet API (2026-10-19)

**Guards, run against the code before the step:**
- `an_f16_facet_widens_to_f32_with_the_right_count`: at step 3's
  state, `base_vectors_as` did not exist and `base_vectors()` on an
  `.mvec` facet was `ElementMismatch` (step 3's own guard); on 3.0.4
  it was a reader with `count() == 2` of garbage. Now `count() == 3`,
  the exact widened values, `Widening`, `get_slice() == None`,
  `native_element_type() == Some(F16)`.
- `facet_refuses_an_index_facet`: already refused since step 3; on
  3.0.4 the bits were reinterpreted as f32. Kept as the pin.

**Built:**
- `VectorReader<T>` (EA-8): `as_slice`, `conversion`,
  `native_element_type`, all defaulted; `XvecReader`, the sharded and
  windowed readers report their type, the windowed reader sub-slices
  `as_slice`.
- `ScalarReader<T>` (EA-9): `open`, `open_path`, crate-private
  `from_storage`; `dim() == 1`; `as_slice` over the mapping after the
  alignment check; bulk `read_into` by one copy; `Debug`. The raw open
  refuses a vector file, as `XvecReader` refuses a column.
- `ShardedXvecReader<T>` became `ShardedReader<T, R>`, generic over
  the per-file reader with an `open` function pointer, so a sharded
  column reads through the same `locate` and seam-crossing `read_into`
  as a sharded vector facet (`ShardedScalarReader<T>`).
- `ConvertingReader<N, T>` (EA-10): `new` consults the table (a native
  or refused pair is an error); `get` and `read_into` convert through a
  64 KiB scratch of `N`; the first value that does not fit is
  `IoError::ValueOverflow` at its facet ordinal; `conversion()` says
  which mode.
- `AnyVectorReader` and `AnyVvecReader` (EA-12), twelve variants each,
  generated by one macro with the crate-private `AnyElement` trait that
  erases a typed reader; `element_type`, `count`, `dim`, `into_typed`
  (a `dyn Any` downcast, no unsafe), `as_type` (the table applied),
  `Debug`.
- `io::open_vec_native` and `io::open_vec_as::<T>` (EA-13); `open_vec`
  stays native-only. The explorer's two-variant enum is gone: its local
  reader is `open_vec_as::<f32>`, so f16, bf16, u8, i8 and i16 bases
  are browsable and an f64 or integer-32 base is refused by the table
  with the reason printed.
- `TestDataView` (EA-15): `open_facet_as_type` (default `Err`, so an
  external implementor keeps compiling), `open_facet_native`,
  `open_vvec_facet_native`. `GenericTestDataView` dispatches each
  through `dispatch_element!`.
- Inside the view (EA-17): the old `open_uniform` body is
  `open_native<N>`, choosing `ScalarReader` or `XvecReader` by the
  extension's layout for a single file and the sharded column or
  vector reader for a series, with the window logic unchanged;
  `open_any_native` erases; the new `open_uniform<T>` is shape check →
  native type → `as_type::<T>`. `open_variable<T>` refuses a series
  whose type is not `T`'s.
- `TestDataViewExt` (EA-16) with the blanket impl: `facet_as`,
  `base_vectors_as`, `query_vectors_as`, `neighbor_indices_as`,
  `neighbor_distances_as`, the pre/post-filtered four,
  `metadata_results_as`. Re-exported at the root with
  `AnyVectorReader`, `AnyVvecReader`, `ScalarReader`,
  `ConvertingReader`, `open_vec_as`, `open_vec_native`.
- The pinned methods and `facet()` are `#[deprecated]` (EA-18) and are
  `open_uniform::<f32|i32>` exactly; `facet()` keeps routing the
  legacy `filtered_*` names. `open_facet_typed` (both), `TypedReader`,
  `TypedAccessError` and `dispatch_typed!` are `#[deprecated]` with the
  migration note (EA-14). The in-crate callers moved to `*_as` (the
  three examples, the explorer, group.rs's tests); the twelve test files
  that cover the deprecated surface carry `#![allow(deprecated)]` and
  pass unchanged.
- Six runnable doc examples (`cargo test --doc`, 24 → 30): on the
  trait, u8 codes native and widened; on `base_vectors_as`, a native
  f32 window streamed through `read_into`; on `facet_as`, bf16 native
  then widened exactly, and a u8 column as one slice; on
  `neighbor_indices_as`, i32 ground truth checked against the base
  count; on `open_facet_as_type`, the refusal matched as a value. The
  Common-tasks table and `AGENTS.md` name `facet_as`.

**Decided during the step:**
- **`ElementType::conversion_to` is applied in exactly one place on
  the view path**, `open_uniform`, through `AnyVectorReader::as_type`;
  the raw readers stay native-only (step 3).
- **A variable-length facet is never converted** (`metadata_results_as`
  requires the file's own type): its reader decodes records from bytes
  at a fixed width, and a converting layer over that is step 4's scope
  creep for no requester.
- **The erased carrier is an enum, not `Box<dyn Any>`**, so a caller
  can match on the variant without a downcast; `into_typed` downcasts
  inside the enum for the generic path.
- **The explorer test of §14.3 is not written**: its reader is
  `pub(super)` with no test harness, and the explorer's own suites run
  the TUI. The remote case is covered (`a_remote_column_has_no_slice_
  until_fetched`: over the HTTP test server a column's `as_slice()` is
  `None` while its values read through the transport, and `Some` once
  a fetch brings it whole into the cache).

**Tests:** vectordata 896 unit, 20 in `tests/facet_as.rs` (the two
guards and eighteen behaviour tests: native, widening, cross-sign with
the overflow ordinal, refusal, scalar and sliced series, a window over
a column, the erased opens, the dyn-view call, `open_vec_as`, the
remote column, every `*_as` reader including `metadata_results_as`
and the vvec refusal, and the column and converting readers' own
opens, errors and `Debug`), the 10
of `tests/element_model.rs`, every other integration suite unchanged,
30 doc tests. `cargo doc` clean with warnings denied; clippy clean on
the changed files; the workspace compiles.

**Coverage after the step** (the §14.5 command, whole workspace:
75.6% → 75.7%; baseline → after, lines, then functions after). The
veks-pipeline, veks, veks-core and vecd suites: 1,583 passed, 0 failed.

| File | Baseline | After | Functions |
|---|---|---|---|
| `vectordata/src/io.rs` | 82.9% | 87.4% (1065/1219) | 89.0% (162/182) |
| `vectordata/src/view.rs` | 84.7% | 84.3% (2086/2474) | 73.9% (266/360) |
| `vectordata/src/typed_access.rs` | 80.2% | 81.1% (279/344) | 66.1% (39/59) |
| `vectordata/src/explore/shared.rs` | 42.4% | 44.3% (209/472) | 39.5% (30/76) |

### 16.5 Step 5: quantized corpora (2026-10-19)

**Guards:** none fail on the code before the step — the surface is
new — except what the step itself turned up (below).

**Built:**
- `StandardFacet::QuantScale`, `QuantZeroPoint`, `QuantSum` (EA-23):
  keys `quant_scale`, `quant_zero_point`, `quant_sum`; no code letter;
  `formats()` of `[FloatScalar]` for the scale and `[ScalarPacked]` for
  the two integer columns, so a consumer borrows each as one slice;
  placed after `BaseVectors` in `PREFERRED_ORDER`; `StandardFacet::
  QUANT` and `is_quant` name the three as a set. `ProfileConfig` carries
  them as fields beside the other standard facets; the view's lookup
  and manifest list them; `facet_role` makes them `Windowed`, so a
  sized profile windows them with `base_vectors`.
- `dataset::Quantization` (EA-24): `scheme` (`affine` | `symmetric`),
  `granularity` (`per-vector` | `per-dimension` | `per-block`), `block`
  for the last, `codes` as an element type name, `range: [lo, hi]`;
  serde round trip; `validate` refuses a non-integer code type, a
  reversed range, per-block without a block, and a block elsewhere.
  `TestDataView::quantization()` (provided, default `None`) reads the
  profile's attribute, else the dataset's.
- `QuantizedBase<C>` and `TestDataViewExt::quantized_base::<C>()`
  (EA-25): opens the codes at `C`, the three columns at their types,
  requires `C::TYPE == scheme.codes` (`ElementMismatch` otherwise) and
  each column's count to equal the codes' count, else the new
  `Error::FacetCardinality { facet, against, expected, found }`.
- `veks check` (EA-26): the three columns are present together or not
  at all; the attribute is present iff the columns are; the attribute
  parses; its `codes` is the base's element type; each column's count
  equals the base's, by `open_vec_native` on both. In the
  `profile-selectors` category, after PL-3.
- A fixture, `write_quantized_profile`, and a doc example on
  `QuantizedBase`.
- `crates/veks/tests/e2e_http_sized_profiles.rs` moved from the
  deprecated pinned readers to `*_as` (the one caller outside
  vectordata, found by its deprecation warnings).

**Decided during the step: `base_vectors` and `query_vectors` admit
integer vectors.** The facet spec allowed only `FloatXvec` for B and Q,
so a quantized u8 base — the input this step exists for — failed
conformance in `veks check`. The spec now lists `[FloatXvec,
IntegerXvec]` for both: a base or query facet is vectors of any element
type, and the conversion table decides how each reads as f32. Distances
stay float. The test that pinned float-only (`spec_answers_facet_
format_validity`) now pins the new rule.

**Tests:** `dataset::quantization` 2 unit tests; `quant_facets_are_
standard_columns`; `quantization_columns_are_typed` (conformance);
`a_sized_profile_windows_the_quant_columns_with_the_base`; 6 in
`tests/quantized.rs` (one base, the wrong code type, a short column,
no attribute, the profile's attribute winning, a malformed attribute);
4 in `veks check`. The vectordata suite and the doc tests (31) pass;
clippy clean on the changed files; docs with warnings denied; the
workspace compiles.

**Coverage after the step** (the §14.5 command, whole workspace:
75.6% → 75.7%; baseline → after, lines, then functions after). The
veks-pipeline, veks, veks-core and vecd suites: 1587 passed, 0 failed.

| File | Baseline | After | Functions |
|---|---|---|---|
| `vectordata/src/dataset/quantization.rs` | — | 95.3% (61/64) | 100.0% (10/10) |
| `vectordata/src/dataset/facet.rs` | 93.3% | 94.0% (358/381) | 96.7% (59/61) |
| `vectordata/src/dataset/conformance.rs` | 88.2% | 89.0% (219/246) | 100.0% (24/24) |
| `vectordata/src/dataset/profile.rs` | 95.5% | 95.5% (2337/2447) | 90.0% (217/241) |
| `vectordata/src/view.rs` | 84.7% | 84.4% (2143/2539) | 74.0% (271/366) |
| `veks/src/check/profiles.rs` | 86.6% | 88.9% (503/566) | 90.6% (58/64) |
