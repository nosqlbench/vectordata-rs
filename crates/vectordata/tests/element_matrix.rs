// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The type matrix (SRD element access, §13.10, §14.6): every reader
//! and every conversion pair, iterated over `ElementType::ALL` rather
//! than over the types an author had in hand. Each test here is one
//! scenario written once; the twelve types, and the 144 pairs of the
//! conversion table, are covered by construction.

mod support;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use support::fixtures::{write_column, write_dataset_yaml, write_mref, write_vvec, write_xvec};
use support::samples::{Samples, rows_of, same, widened};
use support::testserver::TestServer;
use veks_datatypes::dispatch_element;
use vectordata::io::{IndexedVvecReader, IoError, ScalarReader, XvecReader, open_vec, open_vec_as, open_vec_native, open_vvec};
use vectordata::typed_access::{Container, Element, ElementConversion, ElementType};
use vectordata::{Error, TestDataGroup, TestDataView, TestDataViewExt, VectorReader, VvecReader};

fn load(dir: &Path) -> Arc<dyn TestDataView> {
    TestDataGroup::load(dir.to_str().unwrap()).unwrap().profile("default").unwrap()
}

fn io_err(e: Error) -> IoError {
    match e {
        Error::VectorIo(io) => io,
        other => panic!("expected an io error, got {other}"),
    }
}

fn mismatch(e: &IoError) -> Option<(ElementType, ElementType)> {
    match e {
        IoError::ElementMismatch { file, requested, .. } => Some((*file, *requested)),
        _ => None,
    }
}

fn overflow(e: &IoError) -> Option<(usize, ElementType)> {
    match e {
        IoError::ValueOverflow { index, target, .. } => Some((*index, *target)),
        _ => None,
    }
}

/// `name.<ext>` under `dir`, in the canonical spelling of `ty` in `c`.
fn file(dir: &Path, name: &str, ty: ElementType, c: Container) -> PathBuf {
    dir.join(format!("{name}.{}", ty.extension(c)))
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// The first type the table widens `from` into, and the first it
/// checks per value into, if any.
fn widening_target(from: ElementType) -> Option<ElementType> {
    ElementType::ALL.iter().copied().find(|t| from.conversion_to(*t) == ElementConversion::Widening)
}

fn cross_sign_target(from: ElementType) -> Option<ElementType> {
    ElementType::ALL.iter().copied().find(|t| from.conversion_to(*t) == ElementConversion::CrossSign)
}

fn refused_target(from: ElementType) -> Option<ElementType> {
    ElementType::ALL.iter().copied().find(|t| from.conversion_to(*t) == ElementConversion::Refused)
}

/// Assert that `reader`, opened at `T` over a file of `N` whose records
/// are `rows_of(samples)`, behaves as the table says for the pair.
fn check_pair<N: Samples, T: Element>(opened: Result<Arc<dyn VectorReader<T>>, IoError>, label: &str) {
    let samples = N::samples();
    match N::TYPE.conversion_to(T::TYPE) {
        ElementConversion::Native => {
            let r = opened.unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(r.conversion(), ElementConversion::Native, "{label}");
            assert_eq!(r.native_element_type(), Some(N::TYPE), "{label}");
            assert_eq!(r.count(), samples.len(), "{label}");
            for (i, v) in samples.iter().enumerate() {
                let got = r.get(i).unwrap()[0];
                // The same bytes read back; N and T are the same type here.
                assert!(widened(*v, got), "{label}: record {i}");
            }
        }
        ElementConversion::Widening => {
            let r = opened.unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(r.conversion(), ElementConversion::Widening, "{label}");
            assert_eq!(r.native_element_type(), Some(N::TYPE), "{label}");
            assert_eq!(r.count(), samples.len(), "{label}");
            assert_eq!(r.get_slice(0).map(|s| s.len()), None, "{label}: nothing to borrow at a converted type");
            assert!(r.as_slice().is_none(), "{label}");
            let mut all = vec![T::from_le_bytes(&vec![0u8; T::SIZE]); samples.len()];
            assert_eq!(r.read_into(0, &mut all).unwrap(), samples.len(), "{label}");
            for (i, v) in samples.iter().enumerate() {
                let got = r.get(i).unwrap()[0];
                assert!(widened(*v, got), "{label}: record {i} holds {} and reads as {}", v.as_f64(), got.as_f64());
                assert!(same(got, all[i]), "{label}: read_into disagrees with get at {i}");
            }
        }
        ElementConversion::CrossSign => {
            let r = opened.unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(r.conversion(), ElementConversion::CrossSign, "{label}");
            let mut first_bad = None;
            for (i, v) in samples.iter().enumerate() {
                let fits = veks_datatypes::convert::<N, T>(*v).is_ok();
                match r.get(i) {
                    Ok(got) => {
                        assert!(fits, "{label}: record {i} ({}) does not fit {} and read anyway", v.as_f64(), T::TYPE);
                        assert!(widened(*v, got[0]), "{label}: record {i}");
                    }
                    Err(e) => {
                        assert!(!fits, "{label}: record {i} ({}) fits {} and was refused: {e}", v.as_f64(), T::TYPE);
                        assert_eq!(overflow(&e), Some((i, T::TYPE)), "{label}: {e}");
                        first_bad.get_or_insert(i);
                    }
                }
            }
            let first_bad = first_bad.unwrap_or_else(|| panic!("{label}: the samples of {} include an extreme no {} holds", N::TYPE, T::TYPE));
            let mut all = vec![T::from_le_bytes(&vec![0u8; T::SIZE]); samples.len()];
            let e = r.read_into(0, &mut all).unwrap_err();
            assert_eq!(overflow(&e), Some((first_bad, T::TYPE)), "{label}: a bulk read stops at the first value that does not fit");
            if first_bad > 0 {
                assert_eq!(r.read_into(0, &mut all[..first_bad]).unwrap(), first_bad, "{label}: a bulk read short of it succeeds");
            }
        }
        ElementConversion::Refused => {
            let e = match opened {
                Err(e) => e,
                Ok(r) => panic!("{label}: refused by the table, opened with count {}", r.count()),
            };
            assert_eq!(mismatch(&e), Some((N::TYPE, T::TYPE)), "{label}: {e}");
        }
    }
}

// ── Native, every type, every reader ───────────────────────────────────

/// **Every element type opens natively as a vector file, a column and
/// a vvec**, through the raw readers, the boxed opens and the erased
/// open, with every sample value read back as written — and the
/// legacy spelling opens where the type has one.
#[test]
fn every_type_opens_natively_as_a_vector_a_column_and_a_vvec() {
    for &ty in ElementType::ALL {
        dispatch_element!(ty, T => {
            let dir = tempfile::tempdir().unwrap();
            let values: Vec<T> = T::samples();
            let dim = values.len();

            // A vector file: record i, element d holds values[(i + d) % dim].
            let vec_path = file(dir.path(), "v", ty, Container::Vec);
            write_xvec::<T>(&vec_path, 3, dim, |i, d| values[(i + d) % dim]);
            let r = XvecReader::<T>::open_path(&vec_path).unwrap_or_else(|e| panic!("{ty}: {e}"));
            assert_eq!((r.count(), r.dim()), (3, dim), "{ty}");
            assert_eq!(r.native_element_type(), Some(ty));
            for i in 0..3 {
                let row = VectorReader::get(&r, i).unwrap();
                for d in 0..dim {
                    assert!(same(row[d], values[(i + d) % dim]), "{ty}: record {i} element {d}");
                }
                if let Some(slice) = VectorReader::get_slice(&r, i) {
                    assert!(slice.iter().zip(&row).all(|(a, b)| same(*a, *b)), "{ty}: the slice is the record");
                }
            }
            let boxed = open_vec::<T>(s(&vec_path)).unwrap();
            assert_eq!((boxed.count(), boxed.dim()), (3, dim), "{ty}");
            let any = open_vec_native(s(&vec_path)).unwrap();
            assert_eq!((any.element_type(), any.count(), any.dim()), (ty, 3, dim), "{ty}");
            let typed = any.into_typed::<T>().unwrap();
            assert!(same(typed.get(2).unwrap()[0], values[2 % dim]), "{ty}");
            if let Some(legacy) = ty.legacy_extension(Container::Vec) {
                let legacy_path = dir.path().join(format!("v.{legacy}"));
                std::fs::copy(&vec_path, &legacy_path).unwrap();
                let r = XvecReader::<T>::open_path(&legacy_path).unwrap_or_else(|e| panic!("{ty} as .{legacy}: {e}"));
                assert_eq!(r.count(), 3);
                assert_eq!(open_vec_native(s(&legacy_path)).unwrap().element_type(), ty, ".{legacy} is {ty}");
            }

            // A column of the samples.
            let col_path = file(dir.path(), "c", ty, Container::Scalar);
            write_column::<T>(&col_path, &values);
            let c = ScalarReader::<T>::open_path(&col_path).unwrap_or_else(|e| panic!("{ty}: {e}"));
            assert_eq!((c.count(), c.dim()), (dim, 1), "{ty}");
            let slice = c.as_slice().expect("a mapped column is one slice");
            assert!(slice.iter().zip(&values).all(|(a, b)| same(*a, *b)), "{ty}: the column is the values");
            assert!(same(c.get(2).unwrap()[0], values[2]), "{ty}");
            let any = open_vec_native(s(&col_path)).unwrap();
            assert_eq!((any.element_type(), any.count(), any.dim()), (ty, dim, 1), "{ty}");

            // A vvec of three records of differing length.
            let records: Vec<Vec<T>> = vec![values.clone(), vec![values[1]], values[..2].to_vec()];
            let vv_path = file(dir.path(), "r", ty, Container::Vvec);
            write_vvec::<T>(&vv_path, &records);
            let v = IndexedVvecReader::<T>::open_path(&vv_path).unwrap_or_else(|e| panic!("{ty}: {e}"));
            assert_eq!(v.count(), 3, "{ty}");
            for (i, rec) in records.iter().enumerate() {
                assert_eq!(VvecReader::dim_at(&v, i).unwrap(), rec.len(), "{ty}: record {i}");
                let got = VvecReader::get(&v, i).unwrap();
                assert!(got.iter().zip(rec).all(|(a, b)| same(*a, *b)), "{ty}: record {i}");
            }
            assert_eq!(open_vvec::<T>(s(&vv_path)).unwrap().count(), 3, "{ty}");
        });
    }
}

/// **Every element type reads natively through the view**: a base of
/// it, a column of it and (for the integers, which the facet spec
/// admits as variable-length) a vvec of it, each at its own type with
/// `Native` conversion, and the erased opens name the type.
#[test]
fn every_type_reads_natively_through_the_view() {
    for &ty in ElementType::ALL {
        dispatch_element!(ty, T => {
            let dir = tempfile::tempdir().unwrap();
            let values: Vec<T> = T::samples();
            let dim = values.len();
            write_xvec::<T>(&file(dir.path(), "v", ty, Container::Vec), 3, dim, |i, d| values[(i + d) % dim]);
            write_column::<T>(&file(dir.path(), "c", ty, Container::Scalar), &values);
            let mut facets = vec![
                ("base_vectors", format!("v.{}", ty.extension(Container::Vec))),
                ("metadata_content", format!("c.{}", ty.extension(Container::Scalar))),
            ];
            if ty.is_integer() {
                write_vvec::<T>(&file(dir.path(), "r", ty, Container::Vvec), &[values.clone(), vec![values[1]]]);
                facets.push(("metadata_results", format!("r.{}", ty.extension(Container::Vvec))));
            }
            let facets: Vec<(&str, &str)> = facets.iter().map(|(k, v)| (*k, v.as_str())).collect();
            write_dataset_yaml(dir.path(), "native", &facets);
            let view = load(dir.path());

            let base = view.base_vectors_as::<T>().unwrap_or_else(|e| panic!("{ty}: {e}"));
            assert_eq!((base.count(), base.dim()), (3, dim), "{ty}");
            assert_eq!(base.conversion(), ElementConversion::Native, "{ty}");
            assert_eq!(base.native_element_type(), Some(ty), "{ty}");
            let row = base.get(1).unwrap();
            assert!((0..dim).all(|d| same(row[d], values[(1 + d) % dim])), "{ty}: record 1");
            assert_eq!(view.open_facet_native("base_vectors").unwrap().element_type(), ty, "{ty}");
            assert_eq!(view.facet_element_type("base_vectors").unwrap(), ty, "{ty}");

            let column = view.facet_as::<T>("metadata_content").unwrap_or_else(|e| panic!("{ty}: {e}"));
            assert_eq!((column.count(), column.dim()), (dim, 1), "{ty}");
            let slice = column.as_slice().unwrap_or_else(|| panic!("{ty}: a local column is one slice"));
            assert!(slice.iter().zip(&values).all(|(a, b)| same(*a, *b)), "{ty}: the column is the values");

            if ty.is_integer() {
                let results = view.metadata_results_as::<T>().unwrap_or_else(|e| panic!("{ty}: {e}"));
                assert_eq!(results.count(), 2, "{ty}");
                assert!(results.get(0).unwrap().iter().zip(&values).all(|(a, b)| same(*a, *b)), "{ty}: record 0");
                assert_eq!(results.get(1).unwrap().len(), 1, "{ty}: record 1");
                assert_eq!(view.open_vvec_facet_native("metadata_results").unwrap().element_type(), ty, "{ty}");
            }
        });
    }
}

// ── Every pair of the table, on files ──────────────────────────────────

/// **Every (file type, requested type) pair opens as the conversion
/// table says**, through the raw opens: `Native` reads the bytes,
/// `Widening` reads every sample exactly (subnormals, infinities and
/// NaN included) with nothing to borrow, `CrossSign` reads what fits
/// and reports `ValueOverflow` at the ordinal of the first value that
/// does not, and `Refused` is `ElementMismatch` naming both types —
/// for a vector file, a column, and the raw typed constructors, which
/// admit only the native pair.
#[test]
fn every_pair_of_the_table_opens_as_the_table_says_on_raw_files() {
    for &from in ElementType::ALL {
        dispatch_element!(from, N => {
            let dir = tempfile::tempdir().unwrap();
            let samples: Vec<N> = N::samples();
            let vec_path = file(dir.path(), "v", from, Container::Vec);
            write_xvec::<N>(&vec_path, samples.len(), 1, |i, _| samples[i]);
            let col_path = file(dir.path(), "c", from, Container::Scalar);
            write_column::<N>(&col_path, &samples);
            let rows: Vec<Vec<N>> = rows_of(&samples);
            assert_eq!(rows.len(), samples.len());

            for &to in ElementType::ALL {
                dispatch_element!(to, T => {
                    let label = format!("{from} file as {to}");
                    check_pair::<N, T>(open_vec_as::<T>(s(&vec_path)), &format!("vector, {label}"));
                    check_pair::<N, T>(open_vec_as::<T>(s(&col_path)), &format!("column, {label}"));
                    check_pair::<N, T>(open_vec_native(s(&vec_path)).unwrap().as_type::<T>(), &format!("erased, {label}"));

                    // The typed constructors are native-only.
                    let native = from == to;
                    match XvecReader::<T>::open_path(&vec_path) {
                        Ok(r) => assert!(native, "XvecReader<{to}> opened a {from} file with count {}", r.count()),
                        Err(e) => {
                            assert!(!native, "XvecReader<{to}> refused its own type: {e}");
                            assert_eq!(mismatch(&e), Some((from, to)), "{label}: {e}");
                        }
                    }
                    match ScalarReader::<T>::open_path(&col_path) {
                        Ok(r) => assert!(native, "ScalarReader<{to}> opened a {from} column with count {}", r.count()),
                        Err(e) => {
                            assert!(!native, "ScalarReader<{to}> refused its own type: {e}");
                            assert_eq!(mismatch(&e), Some((from, to)), "{label}: {e}");
                        }
                    }
                    match open_vec_native(s(&vec_path)).unwrap().into_typed::<T>() {
                        Ok(_) => assert!(native, "into_typed::<{to}> took a {from} reader"),
                        Err(e) => assert_eq!(mismatch(&e), Some((from, to)), "{label}: {e}"),
                    }
                });
            }
        });
    }
}

/// **Every pair of the table holds through the view** — `facet_as`,
/// `base_vectors_as` and `open_facet_as_type` apply the same table to
/// a declared facet, on a base and on a column.
#[test]
fn every_pair_of_the_table_opens_as_the_table_says_through_the_view() {
    for &from in ElementType::ALL {
        dispatch_element!(from, N => {
            let dir = tempfile::tempdir().unwrap();
            let samples: Vec<N> = N::samples();
            write_xvec::<N>(&file(dir.path(), "v", from, Container::Vec), samples.len(), 1, |i, _| samples[i]);
            write_column::<N>(&file(dir.path(), "c", from, Container::Scalar), &samples);
            let v = format!("v.{}", from.extension(Container::Vec));
            let c = format!("c.{}", from.extension(Container::Scalar));
            write_dataset_yaml(dir.path(), "pairs", &[("base_vectors", &v), ("metadata_content", &c)]);
            let view = load(dir.path());

            for &to in ElementType::ALL {
                dispatch_element!(to, T => {
                    let label = format!("{from} facet as {to}");
                    check_pair::<N, T>(view.base_vectors_as::<T>().map_err(io_err), &format!("base, {label}"));
                    check_pair::<N, T>(view.facet_as::<T>("metadata_content").map_err(io_err), &format!("column, {label}"));
                    check_pair::<N, T>(
                        view.open_facet_as_type("base_vectors", to).map_err(io_err).and_then(|any| any.into_typed::<T>()),
                        &format!("by type, {label}"),
                    );
                });
            }
        });
    }
}

// ── Windows, series and HTTP, every type ───────────────────────────────

/// **A window over any type is the records it names**, for a vector
/// facet and a column, natively and through one widening and one
/// cross-sign conversion where the table has them (the overflow is
/// reported at the window-relative ordinal).
#[test]
fn every_type_reads_through_a_window() {
    for &ty in ElementType::ALL {
        dispatch_element!(ty, N => {
            let dir = tempfile::tempdir().unwrap();
            let samples: Vec<N> = N::samples();
            let n = samples.len();
            write_xvec::<N>(&file(dir.path(), "v", ty, Container::Vec), n, 1, |i, _| samples[i]);
            write_column::<N>(&file(dir.path(), "c", ty, Container::Scalar), &samples);
            let v = format!("v.{}[1..{n})", ty.extension(Container::Vec));
            let c = format!("c.{}[1..{n})", ty.extension(Container::Scalar));
            write_dataset_yaml(dir.path(), "window", &[("base_vectors", &v), ("metadata_content", &c)]);
            let view = load(dir.path());

            let base = view.base_vectors_as::<N>().unwrap_or_else(|e| panic!("{ty}: {e}"));
            assert_eq!(base.count(), n - 1, "{ty}");
            assert!((0..n - 1).all(|i| same(base.get(i).unwrap()[0], samples[i + 1])), "{ty}: the window starts at record 1");
            let column = view.facet_as::<N>("metadata_content").unwrap();
            let slice = column.as_slice().unwrap_or_else(|| panic!("{ty}: a window over a mapped column is a sub-slice"));
            assert!(slice.iter().zip(&samples[1..]).all(|(a, b)| same(*a, *b)), "{ty}");

            if let Some(to) = widening_target(ty) {
                dispatch_element!(to, T => {
                    let wide = view.facet_as::<T>("metadata_content").unwrap_or_else(|e| panic!("{ty} window as {to}: {e}"));
                    assert_eq!(wide.count(), n - 1);
                    assert!((0..n - 1).all(|i| widened(samples[i + 1], wide.get(i).unwrap()[0])), "{ty} window as {to}");
                });
            }
            if let Some(to) = cross_sign_target(ty) {
                dispatch_element!(to, T => {
                    let checked = view.base_vectors_as::<T>().unwrap_or_else(|e| panic!("{ty} window as {to}: {e}"));
                    for i in 0..n - 1 {
                        let fits = veks_datatypes::convert::<N, T>(samples[i + 1]).is_ok();
                        match checked.get(i) {
                            Ok(got) => assert!(fits && widened(samples[i + 1], got[0]), "{ty} window as {to}: record {i}"),
                            Err(e) => assert_eq!((fits, overflow(&e)), (false, Some((i, to))), "{ty} window as {to}: {e}"),
                        }
                    }
                });
            }
        });
    }
}

/// **A series of any type reads across its seams**, for a vector
/// facet and a column, natively and through one widening conversion
/// where the table has one; a series has no one mapping to borrow.
#[test]
fn every_type_reads_through_a_series() {
    for &ty in ElementType::ALL {
        dispatch_element!(ty, N => {
            let dir = tempfile::tempdir().unwrap();
            let samples: Vec<N> = N::samples();
            let n = samples.len();
            let stride = 3;
            let shards = n.div_ceil(stride);
            let (vext, cext) = (ty.extension(Container::Vec), ty.extension(Container::Scalar));
            for k in 0..shards {
                let part = &samples[k * stride..((k + 1) * stride).min(n)];
                write_xvec::<N>(&dir.path().join(format!("v__{k:04}.{vext}")), part.len(), 1, |i, _| part[i]);
                write_column::<N>(&dir.path().join(format!("c__{k:04}.{cext}")), part);
            }
            std::fs::write(
                dir.path().join("dataset.yaml"),
                format!(
                    "format_version: 2\nname: series\nprofiles:\n  default:\n    base_vectors:\n      source: v__NNNN.{vext}\n      \
                     shard_stride: {stride}\n      shard_count: {shards}\n      record_count: {n}\n    metadata_content:\n      \
                     source: c__NNNN.{cext}\n      shard_stride: {stride}\n      shard_count: {shards}\n      record_count: {n}\n"
                ),
            )
            .unwrap();
            let view = load(dir.path());

            for (name, reader) in [
                ("base", view.base_vectors_as::<N>().unwrap_or_else(|e| panic!("{ty}: {e}"))),
                ("column", view.facet_as::<N>("metadata_content").unwrap_or_else(|e| panic!("{ty}: {e}"))),
            ] {
                assert_eq!(reader.count(), n, "{ty} {name}");
                assert!(reader.as_slice().is_none(), "{ty} {name}: a series has no one mapping");
                assert!((0..n).all(|i| same(reader.get(i).unwrap()[0], samples[i])), "{ty} {name}: every ordinal");
                let mut across = vec![<N as Element>::from_le_bytes(&[0u8; N::SIZE]); n];
                assert_eq!(reader.read_into(0, &mut across).unwrap(), n, "{ty} {name}");
                assert!(across.iter().zip(&samples).all(|(a, b)| same(*a, *b)), "{ty} {name}: a bulk read crosses the seams");
                assert!(reader.get(n).is_err(), "{ty} {name}: past the end");
            }
            if let Some(to) = widening_target(ty) {
                dispatch_element!(to, T => {
                    let wide = view.facet_as::<T>("metadata_content").unwrap_or_else(|e| panic!("{ty} series as {to}: {e}"));
                    let mut across = vec![<T as Element>::from_le_bytes(&[0u8; T::SIZE]); n];
                    assert_eq!(wide.read_into(0, &mut across).unwrap(), n);
                    assert!(across.iter().zip(&samples).all(|(a, b)| widened(*b, *a)), "{ty} series as {to}");
                });
            }
        });
    }
}

static TEST_CACHE_DIR: std::sync::LazyLock<tempfile::TempDir> =
    std::sync::LazyLock::new(|| tempfile::tempdir().unwrap());

/// **Every type reads over HTTP**: natively through the transport
/// before any fetch (no slice to borrow), as one slice once fetched
/// whole, and through one widening, one cross-sign and one refused
/// conversion where the table has them — the refusal before a byte is
/// asked for.
#[test]
fn every_type_reads_over_http() {
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
    let dir = tempfile::tempdir().unwrap();
    let mut yaml = String::from("name: remote-matrix\nprofiles:\n");
    for &ty in ElementType::ALL {
        dispatch_element!(ty, N => {
            let samples: Vec<N> = N::samples();
            let v = file(dir.path(), &format!("v_{ty}"), ty, Container::Vec);
            let c = file(dir.path(), &format!("c_{ty}"), ty, Container::Scalar);
            write_xvec::<N>(&v, samples.len(), 1, |i, _| samples[i]);
            write_column::<N>(&c, &samples);
            write_mref(&v);
            write_mref(&c);
            yaml.push_str(&format!(
                "  p_{ty}:\n    base_vectors: {}\n    metadata_content: {}\n",
                v.file_name().unwrap().to_str().unwrap(),
                c.file_name().unwrap().to_str().unwrap()
            ));
        });
    }
    std::fs::write(dir.path().join("dataset.yaml"), yaml).unwrap();
    let server = TestServer::start(dir.path()).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();

    for &ty in ElementType::ALL {
        dispatch_element!(ty, N => {
            let samples: Vec<N> = N::samples();
            let n = samples.len();
            let view = group.profile(&format!("p_{ty}")).unwrap_or_else(|| panic!("profile p_{ty}"));

            let base = view.base_vectors_as::<N>().unwrap_or_else(|e| panic!("{ty} remote: {e}"));
            assert_eq!((base.count(), base.conversion()), (n, ElementConversion::Native), "{ty} remote");
            assert!((0..n).all(|i| same(base.get(i).unwrap()[0], samples[i])), "{ty} remote: every record through the transport");
            let column = view.facet_as::<N>("metadata_content").unwrap();
            assert!(column.as_slice().is_none(), "{ty} remote: nothing mapped before a fetch");
            assert!(same(column.get(n - 1).unwrap()[0], samples[n - 1]), "{ty} remote column");

            if let Some(to) = widening_target(ty) {
                dispatch_element!(to, T => {
                    let wide = view.base_vectors_as::<T>().unwrap_or_else(|e| panic!("{ty} remote as {to}: {e}"));
                    assert!((0..n).all(|i| widened(samples[i], wide.get(i).unwrap()[0])), "{ty} remote as {to}");
                });
            }
            if let Some(to) = cross_sign_target(ty) {
                dispatch_element!(to, T => {
                    let checked = view.base_vectors_as::<T>().unwrap_or_else(|e| panic!("{ty} remote as {to}: {e}"));
                    let bad = (0..n).find(|&i| veks_datatypes::convert::<N, T>(samples[i]).is_err()).unwrap();
                    assert_eq!(overflow(&checked.get(bad).unwrap_err()), Some((bad, to)), "{ty} remote as {to}");
                });
            }
            if let Some(to) = refused_target(ty) {
                let before = server.accepted_connections();
                let e = io_err(view.open_facet_as_type("base_vectors", to).unwrap_err());
                assert_eq!(mismatch(&e), Some((ty, to)), "{ty} remote as {to}: {e}");
                assert_eq!(server.accepted_connections(), before, "{ty} remote as {to}: refused before any byte is asked for");
            }

            let mut silent = vectordata::fetch::Silent;
            view.fetch(&vectordata::fetch::FetchRequest::facets(["metadata_content"]), &mut silent).unwrap();
            let fetched = view.facet_as::<N>("metadata_content").unwrap();
            let slice = fetched.as_slice().unwrap_or_else(|| panic!("{ty} remote: fetched whole, the column maps"));
            assert!(slice.iter().zip(&samples).all(|(a, b)| same(*a, *b)), "{ty} remote: the fetched column is the values");
        });
    }
}
