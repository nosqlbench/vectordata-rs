// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The element type matrix for the pipeline's commands (SRD §13.10,
//! §14.6): the analysis commands and `analyze describe` over the same
//! values in every element type, `generate vectors --type` and
//! `transform convert` over every type as a target, and `compute knn`
//! over a bf16 and a u8 base.
//!
//! Every test iterates `ElementType::ALL` through `dispatch_element!`
//! rather than naming types one by one. The values are small integers
//! (and, for `transform convert`, dyadic fractions), exact in every
//! type including `bf16`, so that every type's run must agree with the
//! `f32` run exactly, not within a tolerance.

mod support;

use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};
use std::thread::ThreadId;

use veks_datatypes::{Element, ElementType, Value, dispatch_element};
use veks_pipeline::pipeline::command::Status;

use support::samples::same_f64;
use support::{Run, hide_path, options, read_as_f64, run_command, vec_path, write_vec_file};

// ---------------------------------------------------------------------------
// Shared values and per-type files
// ---------------------------------------------------------------------------

/// Twelve records of four small non-negative integers, exact in every
/// element type (`bf16` holds integers up to 256 exactly); record 5 is
/// the zero vector for `analyze zeros`.
fn matrix_vectors() -> Vec<Vec<f32>> {
    let mut state: u32 = 7;
    let mut next = || {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
        ((state >> 16) % 21) as f32
    };
    let mut vectors: Vec<Vec<f32>> = (0..12).map(|_| (0..4).map(|_| next()).collect()).collect();
    vectors[5] = vec![0.0; 4];
    vectors
}

/// `matrix_vectors` sorted lexicographically, for the binary-search
/// branch of `analyze find`.
fn sorted_matrix_vectors() -> Vec<Vec<f32>> {
    let mut vectors = matrix_vectors();
    vectors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vectors
}

/// `v` as an element of `T`, asserting the value is exact there.
fn as_element<T: Element>(v: f32) -> T {
    let value = if T::TYPE.is_float() { Value::Float(v as f64) } else { Value::Int(v as i128) };
    let e = T::from_value(value).unwrap_or_else(|| panic!("{v} does not fit {}", T::TYPE));
    assert!(same_f64(e.as_f64(), v as f64), "{v} is not exact in {}", T::TYPE);
    e
}

/// Write `vectors` as a uniform vector file of `ty` under `stem`.
fn write_as(dir: &Path, stem: &str, ty: ElementType, vectors: &[Vec<f32>]) -> PathBuf {
    let path = vec_path(dir, stem, ty);
    dispatch_element!(ty, T => {
        let records: Vec<Vec<T>> = vectors
            .iter()
            .map(|v| v.iter().map(|&x| as_element::<T>(x)).collect())
            .collect();
        write_vec_file(&path, &records);
    });
    path
}

/// `vectors` written once per element type: `f32` first, as the
/// reference every other type is compared against, then the rest of
/// `ElementType::ALL`.
fn matrix_files(dir: &Path, stem: &str, vectors: &[Vec<f32>]) -> Vec<(ElementType, PathBuf)> {
    let mut order = vec![ElementType::F32];
    order.extend(ElementType::ALL.iter().copied().filter(|t| *t != ElementType::F32));
    order.into_iter().map(|ty| (ty, write_as(dir, stem, ty, vectors))).collect()
}

/// Every run after the first (the `f32` run) equals it.
fn assert_every_type_matches_f32<R: PartialEq + std::fmt::Debug>(runs: &[(ElementType, R)], what: &str) {
    let (reference_ty, reference) = &runs[0];
    assert_eq!(*reference_ty, ElementType::F32);
    for (ty, run) in &runs[1..] {
        assert_eq!(run, reference, "{what}: the {ty} run differs from the f32 run");
    }
}

/// A run's status, message and log lines with the input path hidden,
/// which is what one type's run must share with another's.
fn normalized(run: &Run, path: &Path) -> (Status, String, Vec<String>) {
    (
        run.result.status.clone(),
        hide_path(&run.result.message, path),
        run.logs.iter().map(|l| hide_path(l, path)).collect(),
    )
}

// ---------------------------------------------------------------------------
// `log` capture for the commands that report through `log::info!`
// ---------------------------------------------------------------------------

struct CaptureLogger {
    lines: Mutex<Vec<(ThreadId, String)>>,
}

impl log::Log for CaptureLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            self.lines.lock().unwrap().push((std::thread::current().id(), record.args().to_string()));
        }
    }
    fn flush(&self) {}
}

static CAPTURE: CaptureLogger = CaptureLogger { lines: Mutex::new(Vec::new()) };

/// Install the capturing logger once for this test binary.
fn capture_logs() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        log::set_logger(&CAPTURE).expect("the first logger of this binary");
        log::set_max_level(log::LevelFilter::Info);
    });
}

/// Take the lines logged from the calling thread since the last take.
fn take_logged_here() -> Vec<String> {
    let me = std::thread::current().id();
    let mut lines = CAPTURE.lines.lock().unwrap();
    let (mine, others): (Vec<_>, Vec<_>) = lines.drain(..).partition(|(id, _)| *id == me);
    *lines = others;
    mine.into_iter().map(|(_, l)| l).collect()
}

// ---------------------------------------------------------------------------
// The analysis commands over every type
// ---------------------------------------------------------------------------

/// `analyze zeros` finds the same near-zero record, with the same
/// message and the same ordinals file, whatever type the vectors are.
#[test]
fn analyze_zeros_finds_the_same_zero_record_in_every_type() {
    let dir = tempfile::tempdir().unwrap();
    let files = matrix_files(dir.path(), "zeros", &matrix_vectors());
    let runs: Vec<_> = files
        .iter()
        .map(|(ty, path)| {
            let output = dir.path().join(format!("zeros_{ty}.ivecs"));
            let run = run_command(
                "analyze zeros",
                dir.path(),
                &options(&[("source", &path.display().to_string()), ("output", &output.display().to_string())]),
            );
            assert_eq!(run.result.status, Status::Warning, "{ty}: {}", run.result.message);
            assert!(run.result.message.starts_with("1 near-zero vectors"), "{ty}: {}", run.result.message);
            let ordinals = std::fs::read(&output).unwrap();
            (*ty, (hide_path(&run.result.message, path), ordinals))
        })
        .collect();
    assert_every_type_matches_f32(&runs, "analyze zeros");
}

/// `analyze stats` reports the same global, per-dimension and
/// all-dimension statistics for every type.
#[test]
fn analyze_stats_reports_the_same_statistics_for_every_type() {
    let dir = tempfile::tempdir().unwrap();
    let files = matrix_files(dir.path(), "stats", &matrix_vectors());
    let modes: [&[(&str, &str)]; 3] = [&[], &[("dimension", "1")], &[("all-dimensions", "true")]];
    for mode in modes {
        let runs: Vec<_> = files
            .iter()
            .map(|(ty, path)| {
                let mut pairs = vec![("source", path.display().to_string())];
                pairs.extend(mode.iter().map(|(k, v)| (*k, v.to_string())));
                let pairs: Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (*k, v.as_str())).collect();
                let run = run_command("analyze stats", dir.path(), &options(&pairs));
                assert_eq!(run.result.status, Status::Ok, "{ty}: {}", run.result.message);
                let (status, message, logs) = normalized(&run, path);
                // The source line names the file's own type; everything
                // after it is the statistics.
                let logs: Vec<String> = logs.into_iter().filter(|l| !l.starts_with("  source:")).collect();
                assert!(logs.iter().any(|l| l.contains("Mean")), "{ty}: no statistics logged: {logs:?}");
                (*ty, (status, message, logs))
            })
            .collect();
        assert_every_type_matches_f32(&runs, &format!("analyze stats {mode:?}"));
    }
}

/// `analyze slice` prints the same records and components, in text and
/// csv form, for every type.
#[test]
fn analyze_slice_prints_the_same_slice_for_every_type() {
    capture_logs();
    let dir = tempfile::tempdir().unwrap();
    let files = matrix_files(dir.path(), "slice", &matrix_vectors());
    for format in ["text", "csv", "json"] {
        let runs: Vec<_> = files
            .iter()
            .map(|(ty, path)| {
                take_logged_here();
                let run = run_command(
                    "analyze slice",
                    dir.path(),
                    &options(&[
                        ("source", &path.display().to_string()),
                        ("ordinal-range", "[2,7]"),
                        ("component-range", "[1,3]"),
                        ("format", format),
                    ]),
                );
                let printed = take_logged_here();
                assert_eq!(run.result.status, Status::Ok, "{ty}: {}", run.result.message);
                // `[2,7]` is inclusive at both ends: six records; json
                // brackets them.
                assert_eq!(printed.len(), if format == "json" { 8 } else { 6 }, "{ty} {format}: {printed:?}");
                (*ty, (run.result.message.clone(), printed))
            })
            .collect();
        assert_every_type_matches_f32(&runs, &format!("analyze slice {format}"));
    }
}

/// `analyze select` prints the same vectors for every type, through
/// the integer branch for the integer types and the float branch for
/// the float types; only the type label in the json form differs.
#[test]
fn analyze_select_prints_the_same_vectors_for_every_type() {
    let dir = tempfile::tempdir().unwrap();
    let files = matrix_files(dir.path(), "select", &matrix_vectors());
    for format in ["text", "csv", "json"] {
        let runs: Vec<_> = files
            .iter()
            .map(|(ty, path)| {
                let run = run_command(
                    "analyze select",
                    dir.path(),
                    &options(&[("source", &path.display().to_string()), ("range", "0,3,5"), ("format", format)]),
                );
                assert_eq!(run.result.status, Status::Ok, "{ty}: {}", run.result.message);
                assert_eq!(run.result.message, format!("selected 3 {ty} vectors"));
                let label = format!("\"type\":\"{ty}[4]\"");
                let logs: Vec<String> = run.logs.iter().map(|l| l.replace(&label, "\"type\":\"<type>[4]\"")).collect();
                assert_eq!(logs.len(), 3, "{ty} {format}: {logs:?}");
                (*ty, logs)
            })
            .collect();
        assert_every_type_matches_f32(&runs, &format!("analyze select {format}"));
    }
}

/// `analyze display-histogram` draws the same histogram for every type.
#[test]
fn analyze_histogram_draws_the_same_histogram_for_every_type() {
    let dir = tempfile::tempdir().unwrap();
    let files = matrix_files(dir.path(), "histogram", &matrix_vectors());
    let modes: [&[(&str, &str)]; 2] = [&[("dimension", "0"), ("bins", "10")], &[("dimension", "2"), ("sample", "8")]];
    for mode in modes {
        let runs: Vec<_> = files
            .iter()
            .map(|(ty, path)| {
                let mut pairs = vec![("source", path.display().to_string())];
                pairs.extend(mode.iter().map(|(k, v)| (*k, v.to_string())));
                let pairs: Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (*k, v.as_str())).collect();
                let run = run_command("analyze display-histogram", dir.path(), &options(&pairs));
                assert_eq!(run.result.status, Status::Ok, "{ty}: {}", run.result.message);
                let out = normalized(&run, path);
                assert!(out.2.iter().any(|l| l.starts_with("Histogram:")), "{ty}: {:?}", out.2);
                (*ty, out)
            })
            .collect();
        assert_every_type_matches_f32(&runs, &format!("analyze display-histogram {mode:?}"));
    }
}

/// `analyze find` finds the same record by exhaustive scan (an
/// unsorted target) and by binary search (a sorted target) for every
/// type, and across types: an `f32` needle is found in a target of any
/// type.
#[test]
fn analyze_find_locates_the_same_record_in_every_type() {
    let dir = tempfile::tempdir().unwrap();
    let unsorted = matrix_files(dir.path(), "find", &matrix_vectors());
    let sorted = matrix_files(dir.path(), "find_sorted", &sorted_matrix_vectors());
    let f32_source = &unsorted[0].1;
    for (variant, targets) in [("exhaustive scan", &unsorted), ("binary search", &sorted)] {
        let runs: Vec<_> = targets
            .iter()
            .map(|(ty, target)| {
                let source = &unsorted.iter().find(|(t, _)| t == ty).unwrap().1;
                let find = |source: &Path| {
                    let run = run_command(
                        "analyze find",
                        dir.path(),
                        &options(&[
                            ("source", &source.display().to_string()),
                            ("target", &target.display().to_string()),
                            ("index", "7"),
                        ]),
                    );
                    assert_eq!(run.result.status, Status::Ok, "{ty} {variant}: {}", run.result.message);
                    assert!(run.result.message.starts_with("found"), "{ty} {variant}: {}", run.result.message);
                    normalized(&run, target)
                };
                let same_type = find(source);
                let f32_needle = find(f32_source);
                assert_eq!(same_type, f32_needle, "{ty} {variant}: an f32 needle finds the same record");
                (*ty, same_type)
            })
            .collect();
        assert_every_type_matches_f32(&runs, &format!("analyze find ({variant})"));
        let expected = if variant == "exhaustive scan" { "found exact match at index" } else { "found at index" };
        assert!(runs[0].1.1.starts_with(expected), "{variant} was taken: {}", runs[0].1.1);
    }
}

/// `analyze compare-files` finds identical distributions between the
/// `f32` file and the same values in every other type, and reports
/// what it reports for `f32` against itself.
#[test]
fn analyze_compare_finds_every_type_identical_to_f32() {
    let dir = tempfile::tempdir().unwrap();
    let files = matrix_files(dir.path(), "compare", &matrix_vectors());
    let original = &files[0].1;
    let runs: Vec<_> = files
        .iter()
        .map(|(ty, synthetic)| {
            let run = run_command(
                "analyze compare-files",
                dir.path(),
                &options(&[
                    ("original", &original.display().to_string()),
                    ("synthetic", &synthetic.display().to_string()),
                    ("verbose", "true"),
                ]),
            );
            assert_eq!(run.result.status, Status::Ok, "{ty}: {}", run.result.message);
            assert!(run.result.message.starts_with("K-S test: 4/4 dims passed"), "{ty}: {}", run.result.message);
            (*ty, (run.result.message.clone(), run.logs.clone()))
        })
        .collect();
    assert_every_type_matches_f32(&runs, "analyze compare-files");
}

/// `analyze describe` reports the same dimensions, record count and
/// uniform structure for every type, and the element width of the type.
#[test]
fn describe_reports_the_same_shape_and_the_types_width_for_every_type() {
    let dir = tempfile::tempdir().unwrap();
    for (ty, path) in matrix_files(dir.path(), "describe", &matrix_vectors()) {
        let run = run_command("analyze describe", dir.path(), &options(&[("source", &path.display().to_string())]));
        assert_eq!(run.result.status, Status::Ok, "{ty}: {}", run.result.message);
        let lines: Vec<&str> = run.result.message.lines().collect();
        for want in [
            "Dimensions: 4".to_string(),
            "Records:    12".to_string(),
            format!("Element:    {} bytes", ty.byte_width()),
            format!("Record size: {} bytes", 4 + 4 * ty.byte_width()),
            "Structure:   uniform (all records dim=4, 12 records)".to_string(),
        ] {
            assert!(lines.iter().any(|l| l.starts_with(&want)), "{ty}: no line {want:?} in {lines:#?}");
        }
    }
}

/// `analyze find-zeros` scans `f32` and `f16` files only, by design
/// (a pread streaming scanner per width); any other type is refused
/// with the file's type named, never misread.
#[test]
fn find_zeros_refuses_every_type_but_f32_and_f16_by_name() {
    let dir = tempfile::tempdir().unwrap();
    for (ty, path) in matrix_files(dir.path(), "find_zeros", &matrix_vectors()) {
        let run = run_command("analyze find-zeros", dir.path(), &options(&[("source", &path.display().to_string())]));
        match ty {
            ElementType::F32 | ElementType::F16 => {
                assert_ne!(run.result.status, Status::Error, "{ty}: {}", run.result.message);
            }
            _ => {
                assert_eq!(run.result.status, Status::Error, "{ty}: {}", run.result.message);
                assert!(run.result.message.contains(ty.name()), "{ty}: the refusal names the type: {}", run.result.message);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `generate vectors --type`
// ---------------------------------------------------------------------------

/// The element types `generate vectors` has a generator for. Every other
/// type is refused by name (a design limit, recorded here).
const GENERATE_ACCEPTS: [ElementType; 6] =
    [ElementType::U8, ElementType::I16, ElementType::I32, ElementType::F16, ElementType::F32, ElementType::F64];

/// `generate vectors --type <t>` writes a file of `t`, of the right size,
/// that opens as `t`, for every type it accepts, and refuses every other
/// type naming it.
#[test]
fn generate_vectors_writes_every_accepted_type_and_refuses_the_rest_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let (count, dim) = (6usize, 4usize);
    for &ty in ElementType::ALL {
        let output = vec_path(dir.path(), "generated", ty);
        let run = run_command(
            "generate vectors",
            dir.path(),
            &options(&[
                ("output", &output.display().to_string()),
                ("type", ty.name()),
                ("dimension", &dim.to_string()),
                ("count", &count.to_string()),
                ("seed", "7"),
            ]),
        );
        if GENERATE_ACCEPTS.contains(&ty) {
            assert_eq!(run.result.status, Status::Ok, "{ty}: {}", run.result.message);
            let size = std::fs::metadata(&output).unwrap().len();
            assert_eq!(size, (count * (4 + dim * ty.byte_width())) as u64, "{ty}: file size");
            let src = veks_pipeline::pipeline::element_type::F64Vectors::open(&output).unwrap();
            assert_eq!((src.element_type(), src.count(), src.dim()), (ty, count, dim), "{ty}");
            assert!((0..count).all(|i| src.get(i).iter().all(|v| v.is_finite())), "{ty}: finite values");
        } else {
            assert_eq!(run.result.status, Status::Error, "{ty}: {}", run.result.message);
            assert!(run.result.message.contains(ty.name()), "{ty}: the refusal names the type: {}", run.result.message);
            assert!(!output.exists(), "{ty}: nothing is written on a refusal");
        }
    }
}

// ---------------------------------------------------------------------------
// `transform convert`
// ---------------------------------------------------------------------------

/// Two records of dyadic values exact in `f16`, `bf16`, `f32` and
/// `f64` alike.
fn convert_vectors() -> Vec<Vec<f32>> {
    vec![vec![1.5, -2.25, 0.125, 100.0], vec![-0.5, 3.0, 0.0, 7.75]]
}

/// Convert `source` to `ty` under `stem`, returning the run and the
/// output path.
fn convert_to(dir: &Path, source: &Path, stem: &str, ty: ElementType) -> (Run, PathBuf) {
    let output = vec_path(dir, stem, ty);
    let run = run_command(
        "transform convert",
        dir,
        &options(&[("source", &source.display().to_string()), ("output", &output.display().to_string())]),
    );
    (run, output)
}

/// `transform convert` from an `f32` source to every float target
/// writes a file of that type holding the same values.
#[test]
fn convert_round_trips_an_f32_source_through_every_float_target() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = convert_vectors();
    let source = write_as(dir.path(), "source", ElementType::F32, &vectors);
    for &ty in ElementType::ALL.iter().filter(|t| t.is_float()) {
        let (run, output) = convert_to(dir.path(), &source, "converted", ty);
        assert_eq!(run.result.status, Status::Ok, "f32 -> {ty}: {}", run.result.message);
        let src = veks_pipeline::pipeline::element_type::F64Vectors::open(&output).unwrap();
        assert_eq!(src.element_type(), ty, "f32 -> {ty}");
        let got = read_as_f64(&output);
        let want: Vec<Vec<f64>> = vectors.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
        assert_eq!(got, want, "f32 -> {ty}: the values survive the conversion");
    }
}

/// `transform convert` from an `f32` source to any integer target is
/// refused, with both types named, rather than written as something
/// that is not the values.
#[test]
fn convert_refuses_an_f32_source_into_every_integer_target_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let source = write_as(dir.path(), "source", ElementType::F32, &convert_vectors());
    for &ty in ElementType::ALL.iter().filter(|t| t.is_integer()) {
        let (run, output) = convert_to(dir.path(), &source, "refused", ty);
        assert_eq!(run.result.status, Status::Error, "f32 -> {ty}: {}", run.result.message);
        assert!(
            run.result.message.contains("f32") && run.result.message.contains(ty.name()),
            "f32 -> {ty}: the refusal names both types: {}",
            run.result.message
        );
        assert!(!output.exists(), "f32 -> {ty}: nothing is written on a refusal");
    }
}

/// `transform convert` widens a `u8` source into every type the
/// conversion table widens `u8` to, which is every other type but `i8`,
/// and changes sign into `i8` with a per-value check that these small
/// values pass.
#[test]
fn convert_widens_a_u8_source_into_every_wider_type() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = matrix_vectors();
    let source = write_as(dir.path(), "source", ElementType::U8, &vectors);
    let want: Vec<Vec<f64>> = vectors.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect();
    for &ty in ElementType::ALL {
        let (run, output) = convert_to(dir.path(), &source, "widened", ty);
        match ElementType::U8.conversion_to(ty) {
            veks_datatypes::ElementConversion::Native
            | veks_datatypes::ElementConversion::Widening
            | veks_datatypes::ElementConversion::CrossSign => {
                assert_eq!(run.result.status, Status::Ok, "u8 -> {ty}: {}", run.result.message);
                let src = veks_pipeline::pipeline::element_type::F64Vectors::open(&output).unwrap();
                assert_eq!(src.element_type(), ty, "u8 -> {ty}");
                assert_eq!(read_as_f64(&output), want, "u8 -> {ty}: the values survive the widening");
            }
            _ => {
                assert_eq!(run.result.status, Status::Error, "u8 -> {ty}: {}", run.result.message);
                assert!(run.result.message.contains("u8") && run.result.message.contains(ty.name()), "u8 -> {ty}: {}", run.result.message);
            }
        }
    }
}

/// A sign change is checked value by value: a `u8` value that does not
/// fit in `i8` fails the conversion, naming the value and the target.
#[test]
fn convert_fails_a_sign_change_on_the_value_that_does_not_fit() {
    let dir = tempfile::tempdir().unwrap();
    let source = write_as(dir.path(), "source", ElementType::U8, &[vec![1.0, 2.0], vec![3.0, 200.0]]);
    let (run, _) = convert_to(dir.path(), &source, "overflow", ElementType::I8);
    assert_eq!(run.result.status, Status::Error, "u8 -> i8: {}", run.result.message);
    assert!(
        run.result.message.contains("record 1") && run.result.message.contains("value 200 does not fit in i8"),
        "u8 -> i8: {}",
        run.result.message
    );
}

// ---------------------------------------------------------------------------
// `compute knn` over a bf16 and a u8 base
// ---------------------------------------------------------------------------

/// Deterministic vectors of small integers in `0..16`, exact in `u8`
/// and `bf16`.
fn knn_vectors(seed: u32, n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
        ((state >> 16) % 16) as f32
    };
    (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
}

/// Run a KNN engine; the neighbour indices and distances per query, or
/// the command's error message.
fn run_knn(
    command: &str,
    base: &Path,
    query: &Path,
    out_dir: &Path,
    k: usize,
    metric: &str,
) -> Result<(Vec<Vec<i32>>, Vec<Vec<f32>>), String> {
    use vectordata::VectorReader;
    use vectordata::io::XvecReader;
    std::fs::create_dir_all(out_dir).unwrap();
    let indices = out_dir.join("indices.ivecs");
    let distances = out_dir.join("distances.fvecs");
    let run = run_command(
        command,
        out_dir,
        &options(&[
            ("base", &base.display().to_string()),
            ("query", &query.display().to_string()),
            ("indices", &indices.display().to_string()),
            ("distances", &distances.display().to_string()),
            ("neighbors", &k.to_string()),
            ("metric", metric),
        ]),
    );
    if run.result.status != Status::Ok {
        return Err(run.result.message);
    }
    let idx = XvecReader::<i32>::open_path(&indices).unwrap();
    let dst = XvecReader::<f32>::open_path(&distances).unwrap();
    Ok((
        (0..VectorReader::<i32>::count(&idx)).map(|i| idx.get_slice(i).to_vec()).collect(),
        (0..VectorReader::<f32>::count(&dst)).map(|i| dst.get_slice(i).to_vec()).collect(),
    ))
}

/// Each KNN engine answers the same indices, and distances within
/// 1e-5, over a base written as `bf16` and as `u8` as it does over the
/// same vectors written as `f32` (SRD §10: a non-f32 base computes
/// through widening).
fn knn_over_base_type_matches_f32(command: &str) {
    let dir = tempfile::tempdir().unwrap();
    let (n_base, n_query, dim, k) = (64, 6, 8, 5);
    let base = knn_vectors(42, n_base, dim);
    let query = knn_vectors(99, n_query, dim);
    let query_path = write_as(dir.path(), "query", ElementType::F32, &query);
    let f32_base = write_as(dir.path(), "base", ElementType::F32, &base);
    for metric in ["L2", "DOT_PRODUCT"] {
        let (want_idx, want_dist) =
            run_knn(command, &f32_base, &query_path, &dir.path().join(format!("f32_{metric}")), k, metric)
                .unwrap_or_else(|e| panic!("{command} {metric} over f32: {e}"));
        assert_eq!(want_idx.len(), n_query);
        for ty in [ElementType::BF16, ElementType::U8] {
            let typed_base = write_as(dir.path(), "base", ty, &base);
            let (idx, dist) =
                run_knn(command, &typed_base, &query_path, &dir.path().join(format!("{ty}_{metric}")), k, metric)
                    .unwrap_or_else(|e| panic!("{command} {metric} over a {ty} base: {e}"));
            assert_eq!(idx, want_idx, "{command} {metric}: indices over a {ty} base equal the f32 run's");
            for (q, (a, b)) in dist.iter().zip(&want_dist).enumerate() {
                for (j, (x, y)) in a.iter().zip(b).enumerate() {
                    assert!((x - y).abs() <= 1e-5, "{command} {metric} {ty}: query {q} distance {j}: {x} vs {y}");
                }
            }
        }
    }
}

/// `compute knn-metal` over a `bf16` and a `u8` base equals its `f32` run:
/// the base is scanned at its own width and widened per vector, the
/// f32 queries read as they are.
#[test]
fn knn_metal_over_a_bf16_and_a_u8_base_matches_f32() {
    knn_over_base_type_matches_f32("compute knn-metal");
}

/// `compute knn-stdarch` over a `bf16` and a `u8` base equals its `f32`
/// run: each pread chunk is widened into a packed f32 buffer as it lands.
#[test]
fn knn_stdarch_over_a_bf16_and_a_u8_base_matches_f32() {
    knn_over_base_type_matches_f32("compute knn-stdarch");
}

/// `compute knn-blas` over a `bf16` and a `u8` base equals its `f32` run:
/// the sgemm scan widens each record at unpack.
#[test]
fn knn_blas_over_a_bf16_and_a_u8_base_matches_f32() {
    knn_over_base_type_matches_f32("compute knn-blas");
}

/// **`transform extract` gathers records of every element type**, by an
/// index file and by a range, at the source's own type: the output
/// holds exactly the selected records, byte for byte, whatever the type
/// — the specialized f32, f16 and i32 extractors and the record gather
/// that serves the other nine.
#[test]
fn transform_extract_gathers_every_type_by_index_and_by_range() {
    let dir = tempfile::tempdir().unwrap();
    let vectors = matrix_vectors();
    let order = [7i32, 0, 11, 3, 3, 5];
    let index = dir.path().join("order.ivecs");
    write_vec_file(&index, &order.iter().map(|&o| vec![o]).collect::<Vec<_>>());
    let record = |bytes: &[u8], dim: usize, width: usize, i: usize| bytes[i * (4 + dim * width)..(i + 1) * (4 + dim * width)].to_vec();
    for (ty, source) in matrix_files(dir.path(), "source", &vectors) {
        let src = std::fs::read(&source).unwrap();
        let (dim, width) = (vectors[0].len(), ty.byte_width());

        let by_index = vec_path(dir.path(), &format!("by_index_{ty}"), ty);
        let run = run_command("transform extract", dir.path(), &options(&[
            ("source", &source.display().to_string()),
            ("ivec-file", &index.display().to_string()),
            ("output", &by_index.display().to_string()),
        ]));
        assert_eq!(run.result.status, Status::Ok, "{ty} by index: {}", run.result.message);
        let out = std::fs::read(&by_index).unwrap();
        let want: Vec<u8> = order.iter().flat_map(|&o| record(&src, dim, width, o as usize)).collect();
        assert_eq!(out, want, "{ty}: the records the index names, in its order");

        let by_range = vec_path(dir.path(), &format!("by_range_{ty}"), ty);
        let run = run_command("transform extract", dir.path(), &options(&[
            ("source", &source.display().to_string()),
            ("range", "[2,6)"),
            ("output", &by_range.display().to_string()),
        ]));
        assert_eq!(run.result.status, Status::Ok, "{ty} by range: {}", run.result.message);
        let want: Vec<u8> = (2..6).flat_map(|i| record(&src, dim, width, i)).collect();
        assert_eq!(std::fs::read(&by_range).unwrap(), want, "{ty}: records [2, 6)");
    }
}

/// **The record gather refuses `normalize`**, a float transform the f32
/// and f16 extractors carry, rather than writing records it was asked
/// to change unchanged.
#[test]
fn transform_extract_refuses_normalize_on_a_type_it_gathers_as_is() {
    let dir = tempfile::tempdir().unwrap();
    let source = write_as(dir.path(), "source", ElementType::BF16, &matrix_vectors());
    let out = vec_path(dir.path(), "out", ElementType::BF16);
    let run = run_command("transform extract", dir.path(), &options(&[
        ("source", &source.display().to_string()),
        ("normalize", "true"),
        ("output", &out.display().to_string()),
    ]));
    assert_eq!(run.result.status, Status::Error);
    assert!(run.result.message.contains("bf16") && run.result.message.contains("normalize"), "{}", run.result.message);
}

/// **`verify knn-consolidated` reads a base of every type that reads
/// as f32**: the ground truth computed over the f32 copy of the base
/// verifies against the same values held as bf16 and as u8 (and as
/// f32), every sampled query passing — the sgemm scan widens the base
/// at unpack. The ground truth comes from the f32 copy so the test
/// pins the verifier alone.
#[test]
fn verify_knn_consolidated_reads_a_bf16_and_a_u8_base() {
    let (n_base, n_query, dim, k) = (64, 6, 8, 5);
    let base = knn_vectors(42, n_base, dim);
    let query = knn_vectors(99, n_query, dim);
    for ty in [ElementType::F32, ElementType::BF16, ElementType::U8] {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path();
        let reference = write_as(ws, "reference", ElementType::F32, &base);
        let base_path = write_as(ws, "base", ty, &base);
        let query_path = write_as(ws, "query", ElementType::F32, &query);
        let out = ws.join("profiles/default");
        run_knn("compute knn-blas", &reference, &query_path, &out, k, "L2")
            .unwrap_or_else(|e| panic!("knn-blas over the f32 copy: {e}"));
        std::fs::write(
            ws.join("dataset.yaml"),
            format!(
                "name: typed\nprofiles:\n  default:\n    base_vectors: {}\n    query_vectors: {}\n    \
                 neighbor_indices: profiles/default/indices.ivecs\n    neighbor_distances: profiles/default/distances.fvecs\n",
                base_path.file_name().unwrap().to_string_lossy(),
                query_path.file_name().unwrap().to_string_lossy(),
            ),
        )
        .unwrap();
        let report = ws.join("report.json");
        let run = run_command("verify knn-consolidated", ws, &options(&[
            ("base", &base_path.display().to_string()),
            ("query", &query_path.display().to_string()),
            ("metric", "L2"),
            ("sample", &n_query.to_string()),
            ("output", &report.display().to_string()),
        ]));
        assert_eq!(run.result.status, Status::Ok, "verify over a {ty} base: {}", run.result.message);
        let report: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
        let profiles = report["profiles"].as_array().expect("a profiles array");
        assert!(!profiles.is_empty(), "{ty}: a profile was verified");
        for prof in profiles {
            assert_eq!(prof["fail"].as_u64(), Some(0), "{ty}: {prof}");
            assert!(prof["pass"].as_u64().unwrap_or(0) > 0, "{ty}: {prof}");
        }
    }
}

/// `compute knn-faiss` over a `bf16` and a `u8` base equals its `f32`
/// run: the base and queries are read as f32 through the conversion
/// table into the buffers FAISS takes.
#[cfg(feature = "faiss")]
#[test]
fn knn_faiss_over_a_bf16_and_a_u8_base_matches_f32() {
    knn_over_base_type_matches_f32("compute knn-faiss");
}

/// **`verify knn-groundtruth` reads a base of every type that widens to
/// f32**: the ground truth computed over the f32 copy of the base
/// verifies against the same values held as bf16 and as u8, every
/// sampled query passing, and an i32 base is refused naming its type.
#[test]
fn verify_knn_groundtruth_reads_a_bf16_and_a_u8_base() {
    let (n_base, n_query, dim, k) = (64, 6, 8, 5);
    let base = knn_vectors(42, n_base, dim);
    let query = knn_vectors(99, n_query, dim);
    for ty in [ElementType::F32, ElementType::BF16, ElementType::U8] {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path();
        let reference = write_as(ws, "reference", ElementType::F32, &base);
        let base_path = write_as(ws, "base", ty, &base);
        let query_path = write_as(ws, "query", ElementType::F32, &query);
        let out = ws.join("gt");
        run_knn("compute knn-metal", &reference, &query_path, &out, k, "L2")
            .unwrap_or_else(|e| panic!("knn over the f32 copy: {e}"));
        let report = ws.join("report.json");
        let run = run_command("verify knn-groundtruth", ws, &options(&[
            ("base", &base_path.display().to_string()),
            ("query", &query_path.display().to_string()),
            ("indices", &out.join("indices.ivecs").display().to_string()),
            ("neighbors", &k.to_string()),
            ("metric", "L2"),
            ("sample", &n_query.to_string()),
            ("output", &report.display().to_string()),
        ]));
        assert_eq!(run.result.status, Status::Ok, "verify over a {ty} base: {}", run.result.message);
        let report: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&report).unwrap()).unwrap();
        assert_eq!(report["fail_count"].as_u64(), Some(0), "{ty}: {report}");
        assert_eq!(report["sample_count"].as_u64(), Some(n_query as u64), "{ty}");
    }

    let dir = tempfile::tempdir().unwrap();
    let ws = dir.path();
    let base_path = write_as(ws, "base", ElementType::I32, &base);
    let query_path = write_as(ws, "query", ElementType::F32, &query);
    let indices = ws.join("gt.ivecs");
    write_vec_file(&indices, &(0..n_query).map(|_| (0..k as i32).collect::<Vec<i32>>()).collect::<Vec<_>>());
    let run = run_command("verify knn-groundtruth", ws, &options(&[
        ("base", &base_path.display().to_string()),
        ("query", &query_path.display().to_string()),
        ("indices", &indices.display().to_string()),
        ("neighbors", &k.to_string()),
        ("output", &ws.join("r.json").display().to_string()),
    ]));
    assert_eq!(run.result.status, Status::Error);
    assert!(run.result.message.contains("i32"), "{}", run.result.message);
}
