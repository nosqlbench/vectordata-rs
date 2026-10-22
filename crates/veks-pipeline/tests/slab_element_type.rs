// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `transform convert` and slabs of vectors
//! (`docs/design/srd-crate-boundaries.md` acceptance test 5): a slab it
//! writes declares its element type in the schema sidecar, a declared
//! slab converts at that type, and a slab that declares none — whose
//! records are bytes — lands only in another slab, never in a typed
//! file chosen by a guess.

mod support;

use std::path::{Path, PathBuf};

use veks_datatypes::{Element, ElementType, Value, dispatch_element};
use veks_io::{VecFormat, VecSource};
use veks_pipeline::pipeline::command::Status;

use support::{Run, options, read_as_f64, run_command, vec_path, write_vec_file};

/// Two records of four small integers, exact in every element type.
fn values() -> Vec<Vec<u8>> {
    vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]
}

fn convert(dir: &Path, source: &Path, output: &Path) -> Run {
    run_command(
        "transform convert",
        dir,
        &options(&[("source", &source.display().to_string()), ("output", &output.display().to_string())]),
    )
}

fn records(mut source: Box<dyn VecSource>) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    while let Some(r) = source.next_record().unwrap() {
        out.push(r);
    }
    out
}

/// A slab of `f32` vectors written the way every slab was before the
/// vector descriptor: content only, no sidecar.
fn undeclared_f32_slab(dir: &Path) -> PathBuf {
    let path = dir.join("legacy.slab");
    let mut w = slabtastic::SlabWriter::new(&path, slabtastic::WriterConfig::default()).unwrap();
    for v in values() {
        let bytes: Vec<u8> = v.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect();
        w.add_record(&bytes).unwrap();
    }
    w.finish().unwrap();
    path
}

/// **Converting a vector file of every element type into a slab
/// declares that type**: the slab reads back at the source's type,
/// width and dimension, with the source's bytes.
#[test]
fn convert_into_a_slab_declares_every_element_type() {
    let dir = tempfile::tempdir().unwrap();
    for &ty in ElementType::ALL {
        let source = vec_path(dir.path(), &format!("source_{ty}"), ty);
        dispatch_element!(ty, T => {
            let typed: Vec<Vec<T>> = values()
                .iter()
                .map(|v| v.iter().map(|&x| T::from_value(Value::Int(x as i128)).unwrap()).collect())
                .collect();
            write_vec_file(&source, &typed);
        });
        let output = dir.path().join(format!("out_{ty}.slab"));
        let run = convert(dir.path(), &source, &output);
        assert_eq!(run.result.status, Status::Ok, "{ty}: {}", run.result.message);

        let slab = veks_io::open_format(&output, VecFormat::Slab).unwrap();
        assert_eq!(
            (slab.dimension(), slab.element_size(), slab.element_type()),
            (4, ty.byte_width(), Some(ty)),
            "{ty}"
        );
        let want = records(veks_io::open(&source).unwrap());
        assert_eq!(records(slab), want, "{ty}: the slab holds the source's records");
    }
}

/// **A declared slab converts at its type**: `u8` codes in a slab widen
/// into `.fvecs` value for value, as a `.bvecs` source would.
#[test]
fn a_declared_slab_converts_at_its_type() {
    let dir = tempfile::tempdir().unwrap();
    let source = vec_path(dir.path(), "codes", ElementType::U8);
    write_vec_file(&source, &values());
    let slab = dir.path().join("codes.slab");
    assert_eq!(convert(dir.path(), &source, &slab).result.status, Status::Ok);

    let output = vec_path(dir.path(), "widened", ElementType::F32);
    let run = convert(dir.path(), &slab, &output);
    assert_eq!(run.result.status, Status::Ok, "{}", run.result.message);
    let want: Vec<Vec<f64>> = values().iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
    assert_eq!(read_as_f64(&output), want);
}

/// **A slab that declares no type is not converted into a typed file**:
/// its records are bytes, so reading them as `f32` (what they were
/// taken as before) or widening each byte (what bytes would give) are
/// both guesses. The refusal says why, and nothing is written.
#[test]
fn an_undeclared_slab_is_not_converted_into_a_typed_file() {
    let dir = tempfile::tempdir().unwrap();
    let slab = undeclared_f32_slab(dir.path());
    for ty in [ElementType::F32, ElementType::U8] {
        let output = vec_path(dir.path(), &format!("refused_{ty}"), ty);
        let run = convert(dir.path(), &slab, &output);
        assert_eq!(run.result.status, Status::Error, "{ty}: {}", run.result.message);
        assert!(run.result.message.contains("declares no element type"), "{ty}: {}", run.result.message);
        assert!(!output.exists(), "{ty}: nothing is written on a refusal");
    }
}

/// **An undeclared slab copies into a slab as bytes**, still
/// undeclared: the copy claims no type the source did not.
#[test]
fn an_undeclared_slab_copies_into_an_undeclared_slab() {
    let dir = tempfile::tempdir().unwrap();
    let slab = undeclared_f32_slab(dir.path());
    let output = dir.path().join("copy.slab");
    let run = convert(dir.path(), &slab, &output);
    assert_eq!(run.result.status, Status::Ok, "{}", run.result.message);

    let copy = veks_io::open_format(&output, VecFormat::Slab).unwrap();
    assert_eq!((copy.dimension(), copy.element_size(), copy.element_type()), (16, 1, None));
    assert_eq!(records(copy), records(veks_io::open_format(&slab, VecFormat::Slab).unwrap()));
}
