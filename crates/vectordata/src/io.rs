// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Vector I/O — uniform and variable-length record readers.
//!
//! The readers are veks-io's, the one implementation of each file
//! format in the workspace, over [`StorageSource`]: the bytes of a file
//! as vectordata reaches them.
//!
//! - [`XvecReader<T>`] — uniform-stride records (`fvec`, `ivec`, `mvec`,
//!   `dvec`, `bvec`, `svec`, …). All records have the same dimension.
//! - [`IndexedVvecReader<T>`] — variable-length records (`ivvec`,
//!   `fvvec`, `bvvec`, …). Each record may have a different dimension.
//! - [`ScalarReader<T>`] — header-free columns (`.u8`, `.i32`, `.f32`, …).
//!
//! Each is transport-agnostic: one `open(source)` call reads a local
//! file (mapped), a remote URL with a published `.mref` (merkle-cached,
//! mapped once complete), and a remote URL without one (fetched in
//! chunks on demand), and a multi-file series as one file.
//!
//! The traits [`VectorReader<T>`] and [`VvecReader<T>`] are the
//! consumer-facing abstractions; the concrete types are for callers
//! that want the storage-aware methods like `precache()`,
//! `prefetch_range()`, `release_range()`.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use crate::storage::Storage;

pub use veks_datatypes::{Container, Element, ElementConversion, ElementType};
pub use veks_io::reader::{
    AnyVectorReader, AnyVvecReader, ConvertingReader, DIM_UNDEFINED, IoError, VectorReader, VvecReader,
    remove_vvec_index,
};
pub use veks_io::source::ByteSource;
pub use veks_io::writer::{ScalarWriter, VvecWriter, XvecWriter};

pub(crate) use veks_io::reader::{
    AnyElement, ext_of, infer_elem_size, is_scalar_ext, is_vvec_ext, records_to_read, require_native,
};

/// The bound the readers put on their element type before the shared
/// model. Every [`Element`] is one; bound on [`Element`] instead.
#[deprecated(since = "4.0.0", note = "bound on `Element` (re-exported here and at the crate root)")]
pub trait VvecElement: Element {}
#[allow(deprecated)]
impl<T: Element> VvecElement for T {}

/// The bytes of one file as vectordata reaches them: a local file, a
/// remote one cached as it is read (merkle-verified when its publisher
/// gave a reference), or a multi-file series read as one file.
///
/// The [`ByteSource`] the readers here are veks-io's readers over. It
/// is opened by the readers' own `open`, never built directly.
#[derive(Clone)]
pub struct StorageSource(Arc<Storage>);

impl StorageSource {
    /// A source over storage already opened.
    pub(crate) fn new(storage: Arc<Storage>) -> Self {
        Self(storage)
    }
}

impl std::fmt::Debug for StorageSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl ByteSource for StorageSource {
    type Progress = crate::transport::DownloadProgress;

    fn open(source: &str) -> Result<Self, IoError> {
        Ok(Self(Storage::open(source)?))
    }

    fn open_path(path: &Path) -> Result<Self, IoError> {
        Ok(Self(Storage::open_path(path)?))
    }

    fn total_size(&self) -> u64 {
        self.0.total_size()
    }

    fn read_bytes(&self, offset: u64, len: u64) -> io::Result<Vec<u8>> {
        self.0.read_bytes(offset, len)
    }

    fn mmap_slice(&self, offset: u64, len: u64) -> Option<&[u8]> {
        self.0.mmap_slice(offset, len)
    }

    fn mmap_base(&self) -> Option<*const u8> {
        self.0.mmap_base()
    }

    fn parts(&self) -> Vec<veks_io::SourcePart> {
        self.0
            .parts()
            .into_iter()
            .map(|p| veks_io::SourcePart { base: p.mmap_base(), size: p.total_size() })
            .collect()
    }

    fn local_path(&self) -> Option<PathBuf> {
        self.0.local_path()
    }

    fn is_remote(&self) -> bool {
        self.0.is_remote()
    }

    fn is_complete(&self) -> bool {
        self.0.is_complete()
    }

    fn precache(&self) -> io::Result<()> {
        self.0.precache()
    }

    fn prebuffer_with_progress(&self, cb: &mut dyn FnMut(&Self::Progress)) -> io::Result<()> {
        self.0.prebuffer_with_progress(cb)
    }

    /// The sibling published beside `data_source`, with the read-side
    /// auth every data fetch carries; `None` offline, for a local file,
    /// and for anything but a success.
    fn fetch_sibling(&self, data_source: &str, name: &str) -> Option<Vec<u8>> {
        if crate::settings::offline() || !crate::transport::is_remote_url(data_source) {
            return None;
        }
        let url = url::Url::parse(data_source).ok()?.join(name).ok()?;
        let client = crate::transport::shared_client_for(url.as_str());
        let resp = crate::transport::apply_read_auth(client.get(url.clone()), Some(&url)).send().ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.bytes().ok().map(|b| b.to_vec())
    }

    fn advise_sequential(&self) {
        self.0.advise_sequential()
    }

    fn advise_random(&self) {
        self.0.advise_random()
    }

    fn prefetch_range_bytes(&self, start: u64, end: u64) {
        self.0.prefetch_range_bytes(start, end)
    }

    fn release_range_bytes(&self, start: u64, end: u64) {
        self.0.release_range_bytes(start, end)
    }

    fn prefetch_pages_bytes(&self, start: u64, end: u64, paged: Option<&AtomicU64>) {
        self.0.prefetch_pages_bytes(start, end, paged)
    }
}

/// veks-io's [`XvecReader`](veks_io::XvecReader) over vectordata's
/// storage: a uniform vector file, local or remote.
///
/// It reads a file at its own type only (`T` must be what the
/// extension names). A record borrows from the mapping when its bytes
/// are aligned for `T` — an 8-byte type at a 4-byte header is not,
/// every other record — and reads by copy otherwise:
///
/// ```
/// use vectordata::io::XvecReader;
/// use vectordata::VectorReader;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("dist.dvecs"))?;
/// # for v in [[0.25f64], [-1e300]] { f.write_all(&1i32.to_le_bytes())?; for x in v { f.write_all(&f64::to_le_bytes(x))?; } }
/// # drop(f);
/// let r = XvecReader::<f64>::open_path(&dir.path().join("dist.dvecs"))?;
/// assert_eq!((r.count(), r.dim()), (2, 1));
/// assert_eq!(VectorReader::get(&r, 1)?, vec![-1e300]);              // always: a decoded copy
/// assert!(VectorReader::get_slice(&r, 0).is_none(), "offset 4 is not 8-aligned");
/// assert_eq!(VectorReader::get_slice(&r, 1), Some(&[-1e300][..]));   // offset 16 is: zero-copy
/// # Ok(()) }
/// ```
///
/// For a remote dataset, prefer opening it by
/// [`Catalog`](crate::catalog::Catalog) or
/// [`TestDataGroup`](crate::TestDataGroup) and reading the facet from
/// its view, which fetches, plans and tracks it as one dataset.
/// Opening a URL here reads only the header; reads then fetch the
/// chunks they touch.
pub type XvecReader<T> = veks_io::XvecReader<T, StorageSource>;

/// veks-io's [`IndexedVvecReader`](veks_io::IndexedVvecReader) over
/// vectordata's storage: a variable-length file, local or remote.
///
/// Random access needs an offset index: a published `IDXFOR__` sidecar,
/// one kept from an earlier open, or a walk of a complete local copy. A
/// remote file with none of those is refused with
/// [`IoError::OffsetIndexUnavailable`] rather than downloaded whole
/// behind the call; fetch it first
/// ([`TestDataView::fetch`](crate::TestDataView::fetch)).
///
/// ```
/// use vectordata::io::IndexedVvecReader;
/// use vectordata::VvecReader;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("labels.u8vvecs"))?;
/// # for v in [vec![9u8, 8, 7], vec![], vec![255]] { f.write_all(&(v.len() as i32).to_le_bytes())?; f.write_all(&v)?; }
/// # drop(f);
/// let r = IndexedVvecReader::<u8>::open_path(&dir.path().join("labels.u8vvecs"))?;   // builds IDXFOR__labels.u8vvecs.i32 beside it
/// assert_eq!(r.count(), 3);
/// assert_eq!((r.dim_at(0)?, r.dim_at(1)?, r.dim_at(2)?), (3, 0, 1));
/// assert_eq!(VvecReader::get(&r, 2)?, vec![255u8]);
/// # Ok(()) }
/// ```
pub type IndexedVvecReader<T> = veks_io::IndexedVvecReader<T, StorageSource>;

/// veks-io's [`ScalarReader`](veks_io::ScalarReader) over vectordata's
/// storage: a header-free column (`.u8`, `.i32`, `.f32`, `.bf16`, …),
/// `dim() == 1`, the whole column borrowable as one slice.
///
/// ```
/// use vectordata::io::ScalarReader;
/// use vectordata::VectorReader;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("scale.f32"))?;
/// # for x in [0.5f32, 0.25, 2.0] { f.write_all(&x.to_le_bytes())?; }
/// # drop(f);
/// # let mut g = std::fs::File::create(dir.path().join("hash.u64"))?;
/// # for x in [u64::MAX, 1] { g.write_all(&x.to_le_bytes())?; }
/// # drop(g);
/// let scale = ScalarReader::<f32>::open_path(&dir.path().join("scale.f32"))?;
/// assert_eq!((scale.count(), scale.dim()), (3, 1));
/// assert_eq!(scale.as_slice(), Some(&[0.5f32, 0.25, 2.0][..]));     // the whole column, zero-copy
/// let hash = ScalarReader::<u64>::open_path(&dir.path().join("hash.u64"))?;
/// assert_eq!(hash.get(0)?, vec![u64::MAX]);
/// # Ok(()) }
/// ```
pub type ScalarReader<T> = veks_io::ScalarReader<T, StorageSource>;

/// veks-io's [`StreamReclaim`](veks_io::StreamReclaim) over an
/// [`XvecReader`]: a bounded-memory sequential scan.
pub type StreamReclaim<'a, T> = veks_io::StreamReclaim<'a, T, StorageSource>;

/// A uniform reader over storage already opened. The caller has
/// established that the bytes are a vector file of `T`.
pub(crate) fn xvec_from_storage<T: Element>(storage: Arc<Storage>) -> Result<XvecReader<T>, IoError> {
    XvecReader::<T>::from_source(StorageSource::new(storage))
}

/// A column reader over storage already opened.
pub(crate) fn scalar_from_storage<T: Element>(storage: Arc<Storage>) -> Result<ScalarReader<T>, IoError> {
    ScalarReader::<T>::from_source(StorageSource::new(storage))
}

/// A variable-length reader over storage already opened. `source`
/// names the file whose `IDXFOR__` sidecar holds the offsets: for one
/// file of a series, that file's own name (each carries its own index,
/// in its own ordinals).
pub(crate) fn vvec_from_storage<T: Element>(
    storage: Arc<Storage>,
    source: &str,
    elem_size: usize,
) -> Result<IndexedVvecReader<T>, IoError> {
    IndexedVvecReader::<T>::from_source_sized(StorageSource::new(storage), &sidecar_name(source), elem_size)
}

/// The record offsets of the variable-length file `source`, whose bytes
/// `storage` holds, without its element type: the lookup the vvec
/// reader makes, so a windowed prefetch and a later read agree.
pub(crate) fn load_offsets(source: &str, storage: &Arc<Storage>, elem_size: usize) -> Result<Vec<u64>, IoError> {
    veks_io::reader::load_offsets(&sidecar_name(source), &StorageSource::new(storage.clone()), elem_size)
}

/// The name a file's sidecars are found by: a remote URL in the scheme
/// it is fetched by (`s3://` becomes the HTTPS endpoint the storage
/// layer reads), anything else as given.
fn sidecar_name(source: &str) -> String {
    if crate::transport::is_remote_url(source) {
        crate::transport::normalize_remote_url(source).into_owned()
    } else {
        source.to_string()
    }
}

/// Open a vector or column file at the type its extension names, for a
/// caller that does not know the type ahead of time. A column opens as
/// a [`ScalarReader`] (`dim() == 1`), anything else as an
/// [`XvecReader`].
///
/// ```
/// use vectordata::io::{open_vec_native, AnyVectorReader};
/// use vectordata::ElementType;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("scores.i16vecs"))?;
/// # for v in [[-3i16, 7], [i16::MIN, i16::MAX]] { f.write_all(&2i32.to_le_bytes())?; for x in v { f.write_all(&i16::to_le_bytes(x))?; } }
/// # drop(f);
/// let any = open_vec_native(dir.path().join("scores.i16vecs").to_str().unwrap())?;
/// assert_eq!((any.element_type(), any.count(), any.dim()), (ElementType::I16, 2, 2));
/// match any {
///     AnyVectorReader::I16(r) => assert_eq!(r.get(1)?, vec![i16::MIN, i16::MAX]),   // zero-copy, its own type
///     other => panic!("an .i16vecs is i16, not {}", other.element_type()),
/// }
/// # Ok(()) }
/// ```
pub fn open_vec_native(source: &str) -> Result<AnyVectorReader, IoError> {
    veks_io::reader::open_vec_native_with::<StorageSource>(source)
}

/// Open a vector or column file as `T`: in place when `T` is the type
/// the extension names, converted by [`ElementType::conversion_to`]
/// where the table allows (an `.mvec` as f32, a `.bvecs` as i32),
/// refused otherwise. [`open_vec`] is the native-only form.
///
/// ```
/// use vectordata::io::{open_vec_as, IoError};
/// use vectordata::{ElementConversion, ElementType, VectorReader};
/// use vectordata::typed_access::f16;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("half.mvecs"))?;
/// # for v in [[f16::from_f32(0.5), f16::from_f32(-2.0)]] { f.write_all(&2i32.to_le_bytes())?; for x in v { f.write_all(&f16::to_le_bytes(x))?; } }
/// # drop(f);
/// # let mut f = std::fs::File::create(dir.path().join("base.fvecs"))?;
/// # for v in [[1.0f32]] { f.write_all(&1i32.to_le_bytes())?; for x in v { f.write_all(&f32::to_le_bytes(x))?; } }
/// # drop(f);
/// # let mut f = std::fs::File::create(dir.path().join("counts.u16vecs"))?;
/// # for v in [[1u16, 2], [3, 40000]] { f.write_all(&2i32.to_le_bytes())?; for x in v { f.write_all(&u16::to_le_bytes(x))?; } }
/// # drop(f);
/// # let p = |n: &str| dir.path().join(n).to_str().unwrap().to_string();
/// // Widening: an f16 file as f32 — exact, one conversion per element read.
/// let wide = open_vec_as::<f32>(&p("half.mvecs"))?;
/// assert_eq!((wide.conversion(), wide.native_element_type()), (ElementConversion::Widening, Some(ElementType::F16)));
/// assert_eq!(wide.get(0)?, vec![0.5, -2.0]);
///
/// // CrossSign: u16 counts as i16 — what fits reads, 40000 is refused at its ordinal.
/// let checked = open_vec_as::<i16>(&p("counts.u16vecs"))?;
/// assert_eq!(checked.get(0)?, vec![1i16, 2]);
/// assert!(matches!(checked.get(1), Err(IoError::ValueOverflow { index: 1, .. })));
///
/// // Refused: f32 as u8 is not lossless, and never a reinterpretation.
/// assert!(matches!(open_vec_as::<u8>(&p("base.fvecs")), Err(IoError::ElementMismatch { file: ElementType::F32, requested: ElementType::U8, .. })));
/// # Ok(()) }
/// ```
pub fn open_vec_as<T: Element>(source: &str) -> Result<Arc<dyn VectorReader<T>>, IoError> {
    veks_io::reader::open_vec_as_with::<StorageSource, T>(source)
}

/// Open a uniform vector file for typed random access, local or remote,
/// at the type its extension names only. For a remote dataset, prefer
/// [`Catalog`](crate::catalog::Catalog) or
/// [`TestDataGroup`](crate::TestDataGroup) and the view's readers.
///
/// ```
/// use vectordata::io::{open_vec, IoError};
/// use vectordata::ElementType;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("ords.u32vecs"))?;
/// # for v in [[1u32, 2, u32::MAX]] { f.write_all(&3i32.to_le_bytes())?; for x in v { f.write_all(&u32::to_le_bytes(x))?; } }
/// # drop(f);
/// # let path = dir.path().join("ords.u32vecs");
/// let reader = open_vec::<u32>(path.to_str().unwrap())?;              // zero-copy
/// assert_eq!(reader.get(0)?, vec![1, 2, u32::MAX]);
/// // Same width, other sign: refused here; open_vec_as checks it per value.
/// assert!(matches!(open_vec::<i32>(path.to_str().unwrap()), Err(IoError::ElementMismatch { file: ElementType::U32, requested: ElementType::I32, .. })));
/// # Ok(()) }
/// ```
pub fn open_vec<T: Element>(source: &str) -> Result<Box<dyn VectorReader<T>>, IoError> {
    veks_io::reader::open_vec_with::<StorageSource, T>(source)
}

/// Open a variable-length vector file for typed random access, local
/// or remote. A remote file with no published offset index is refused
/// rather than downloaded whole; see [`IndexedVvecReader`].
///
/// ```
/// use vectordata::io::open_vvec;
/// # use std::io::Write;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let dir = tempfile::tempdir()?;
/// # let mut f = std::fs::File::create(dir.path().join("matches.i64vvecs"))?;
/// # for v in [vec![7i64, -7], vec![i64::MAX]] { f.write_all(&(v.len() as i32).to_le_bytes())?; for x in v { f.write_all(&x.to_le_bytes())?; } }
/// # drop(f);
/// let r = open_vvec::<i64>(dir.path().join("matches.i64vvecs").to_str().unwrap())?;
/// assert_eq!((r.count(), r.dim_at(0)?, r.dim_at(1)?), (2, 2, 1));
/// assert_eq!(r.get(1)?, vec![i64::MAX]);
/// # Ok(()) }
/// ```
pub fn open_vvec<T: Element>(source: &str) -> Result<Box<dyn VvecReader<T>>, IoError> {
    IndexedVvecReader::<T>::validate_source(source)?;
    Ok(Box::new(vvec_from_storage::<T>(Storage::open(source)?, source, T::SIZE)?))
}

/// Open a variable-length vector file as raw `u8` records, for a caller
/// that decodes them itself. A remote file with no published offset
/// index is refused rather than downloaded whole.
pub fn open_vvec_untyped(source: &str) -> Result<Box<dyn VvecReader<u8>>, IoError> {
    let ext = ext_of(source);
    let elem_size = infer_elem_size(ext);
    if elem_size == 0 {
        return Err(IoError::InvalidFormat(format!("cannot infer element size from extension '.{ext}'")));
    }
    Ok(Box::new(vvec_from_storage::<u8>(Storage::open(source)?, source, elem_size)?))
}


/// How a facet's extension decides whether its first four bytes are a
/// dimension header or data.
#[cfg(test)]
mod scalar_classification {
    use super::*;

    /// The path and URL classifiers must answer from that one table.
    /// A local open and a remote open of the same facet that disagree
    /// read the same bytes two different ways, one of them wrong.
    #[test]
    fn path_and_url_classification_agree() {
        for ext in [
            "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "fvec", "ivvec", "f32", "f64",
        ] {
            let by_path = <crate::typed_access::ElementType as crate::typed_access::ElementTypeExt>::is_scalar_format(format!("/d/x.{ext}"));
            let by_ext = is_scalar_ext(ext);
            assert_eq!(by_path, by_ext, "path classification of .{ext}");
        }
    }
}


#[cfg(test)]
mod series_reader_tests {
    use super::*;

    fn tmpdir() -> tempfile::TempDir {
        let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp");
        std::fs::create_dir_all(&base).unwrap();
        tempfile::tempdir_in(&base).unwrap()
    }

    /// Write records `from..from+n` of a dim-`d` fvec into shard `i`.
    /// Record `r` is `[r, r+0.5, ...]`, so a value identifies its own
    /// ordinal.
    fn shard(dir: &std::path::Path, i: u32, d: usize, from: usize, n: usize) {
        let p = dir.join(crate::dataset::shards::shard_name("base", "fvecs", i));
        let mut bytes = Vec::new();
        for r in from..from + n {
            bytes.extend_from_slice(&(d as i32).to_le_bytes());
            for k in 0..d {
                bytes.extend_from_slice(&((r * 10 + k) as f32).to_le_bytes());
            }
        }
        std::fs::write(p, bytes).unwrap();
    }

    /// **A sharded facet reads as one vector space.** Count, dim, and
    /// every record come back as if the shards were one file — which
    /// is what lets a kernel scan a series without knowing it is one.
    #[test]
    fn a_sharded_facet_reads_as_one_vector_space() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 4, 0, 10);
        shard(tmp.path(), 1, 4, 10, 10);
        shard(tmp.path(), 2, 4, 20, 5);

        let r = XvecReader::<f32>::open_path(&tmp.path().join("base.fvecs")).unwrap();
        assert_eq!(VectorReader::count(&r), 25);
        assert_eq!(VectorReader::dim(&r), 4);

        for o in 0..25usize {
            let want: Vec<f32> = (0..4).map(|k| (o * 10 + k) as f32).collect();
            assert_eq!(r.get(o).unwrap(), want, "record {o} via get");
        }
    }

    /// **The zero-copy path crosses shards.** `get_slice` is the KNN
    /// inner loop; if it resolved every ordinal against the first
    /// shard's base it would read adjacent memory and return plausible
    /// garbage rather than failing.
    #[test]
    fn the_zero_copy_path_resolves_per_shard() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 3, 0, 8);
        shard(tmp.path(), 1, 3, 8, 8);
        shard(tmp.path(), 2, 3, 16, 4);

        let r = XvecReader::<f32>::open_path(&tmp.path().join("base.fvecs")).unwrap();
        assert_eq!(VectorReader::count(&r), 20);

        for o in 0..20usize {
            let want: Vec<f32> = (0..3).map(|k| (o * 10 + k) as f32).collect();
            assert_eq!(r.get_slice(o), want.as_slice(), "record {o} via get_slice");
        }
        // The two paths agree, which is the invariant the hot loop
        // relies on.
        for o in [0usize, 7, 8, 15, 16, 19] {
            assert_eq!(r.get_slice(o), r.get(o).unwrap().as_slice(), "record {o}");
        }
    }

    /// A seam is where an off-by-one hides: the last record of each
    /// shard and the first of the next are checked explicitly.
    #[test]
    fn records_either_side_of_a_seam_are_correct() {
        let tmp = tmpdir();
        shard(tmp.path(), 0, 2, 0, 6);
        shard(tmp.path(), 1, 2, 6, 6);
        let r = XvecReader::<f32>::open_path(&tmp.path().join("base.fvecs")).unwrap();

        assert_eq!(r.get_slice(5), [50.0f32, 51.0].as_slice(), "last of shard 0");
        assert_eq!(r.get_slice(6), [60.0f32, 61.0].as_slice(), "first of shard 1");
    }

    /// An unsharded file behaves exactly as it did — the series
    /// machinery is inert when there is one file.
    #[test]
    fn a_single_file_is_unaffected() {
        let tmp = tmpdir();
        let path = tmp.path().join("solo.fvecs");
        let mut bytes = Vec::new();
        for r in 0..5usize {
            bytes.extend_from_slice(&2i32.to_le_bytes());
            for k in 0..2 {
                bytes.extend_from_slice(&((r * 10 + k) as f32).to_le_bytes());
            }
        }
        std::fs::write(&path, bytes).unwrap();

        let r = XvecReader::<f32>::open_path(&path).unwrap();
        assert_eq!(VectorReader::count(&r), 5);
        assert_eq!(r.get_slice(4), [40.0f32, 41.0].as_slice());
    }

    /// A series of integer records reads the same way — the shard
    /// resolution is about layout, not element type.
    #[test]
    fn an_integer_series_reads_across_shards() {
        let tmp = tmpdir();
        for (i, from) in [(0u32, 0usize), (1, 4)] {
            let p = tmp.path().join(crate::dataset::shards::shard_name("n", "ivecs", i));
            let mut bytes = Vec::new();
            for r in from..from + 4 {
                bytes.extend_from_slice(&3i32.to_le_bytes());
                for k in 0..3 {
                    bytes.extend_from_slice(&((r * 10 + k) as i32).to_le_bytes());
                }
            }
            std::fs::write(p, bytes).unwrap();
        }
        let r = XvecReader::<i32>::open_path(&tmp.path().join("n.ivecs")).unwrap();
        assert_eq!(VectorReader::count(&r), 8);
        assert_eq!(r.get_slice(7), [70, 71, 72].as_slice());
        assert_eq!(r.get(4).unwrap(), vec![40, 41, 42]);
    }
}
