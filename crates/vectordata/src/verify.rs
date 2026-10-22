// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Verify that a profile's facets belong together, before
//! a vector of it is used.
//!
//! [`TestDataView::verify`] proves a
//! profile's base, queries and ground truth agree:
//!
//! | Check | What it proves |
//! |-------|----------------|
//! | V1 | every facet opens: a positive dimension and a whole number of records |
//! | V2 | the queries have the base's dimension |
//! | V3 | a ground truth has one row per query (equal, not at least) |
//! | V4 | a ground truth's rows are not empty |
//! | V5 | every ground-truth index names a base vector (`0..base count`); a filtered truth may also hold `-1` for a short row |
//! | V6 | a ground truth's distances have its rows and its width |
//! | V7 | a declared `base_count` is the base's record count |
//!
//! V5 is the one check that proves the truth was computed against this
//! base, and the only one that reads more than headers: one pass over
//! `queries × k` indices, reported through [`VerifyProgress`]. Opening a
//! view never runs these; a caller asks.
//!
//! (`docs/design/srd-loose-files-overlays-integrity.md` §5.1.)

pub use veks_apicore::verify::*;


use crate::view::{TestDataView, TestDataViewExt};
use crate::VectorReader;


/// Whether the base is unit length: [`NORMALIZATION_SAMPLE`] vectors
/// (every one, when there are fewer) a fixed stride apart across the
/// whole base, read as `f32`, each within [`UNIT_NORM_TOLERANCE`] of
/// norm 1. A base that does not widen to `f32` is the error its open
/// gives.
pub fn measure_normalization<V: TestDataView + ?Sized>(view: &V) -> crate::Result<Measured<bool>> {
    let base = view.facet_as::<f32>("base_vectors")?;
    let n = base.count();
    let sample = n.min(NORMALIZATION_SAMPLE);
    let stride = n.checked_div(sample).unwrap_or(1);
    let mut worst = 0.0f64;
    for i in 0..sample {
        let v = base.get(i * stride)?;
        let norm = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        worst = worst.max((norm - 1.0).abs());
    }
    Ok(Measured { value: sample > 0 && worst <= UNIT_NORM_TOLERANCE, sample, worst_deviation: worst })
}

/// The file `view`'s measurements are recorded in: the dataset's
/// directory of the cache.
fn measured_path<V: TestDataView + ?Sized>(view: &V) -> Option<std::path::PathBuf> {
    let name = view.cache_name()?;
    crate::settings::cache_dir().ok().map(|c| c.join(name).join(MEASURED_FILE))
}

/// The file name measurements are recorded in, in a dataset's cache
/// directory.
pub const MEASURED_FILE: &str = "measured.yaml";

/// One recorded measurement.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RecordedMeasure {
    value: bool,
    sample: usize,
    worst_deviation: f64,
}

/// Every measurement in a `measured.yaml`, by the content identity of
/// the bytes measured, then by attribute.
type MeasuredFile = std::collections::BTreeMap<String, std::collections::BTreeMap<String, RecordedMeasure>>;

/// The base's unit-length measurement: the one recorded for these bytes
/// when there is one, else a sample taken now
/// ([`measure_normalization`]) and recorded. Recorded by the base's
/// content identity as the cache knows it — a published `.mref` root,
/// or the root recorded once a fetch completed — so a measurement is
/// never re-taken for the same bytes nor applied to others. A local
/// base, whose identity is a read of the whole file, is measured each
/// time (1000 rows) and not recorded; neither is one partly fetched.
/// Returns the measurement and the file it is recorded in.
pub fn recorded_normalization<V: TestDataView + ?Sized>(view: &V) -> crate::Result<(Measured<bool>, Option<String>)> {
    let key = view.open_facet_storage("base_vectors").ok().and_then(|s| s.known_content_id()).map(|c| c.hex());
    let path = measured_path(view);
    let read = |p: &std::path::Path| -> MeasuredFile {
        std::fs::read_to_string(p).ok().and_then(|t| serde_yaml::from_str(&t).ok()).unwrap_or_default()
    };
    if let (Some(key), Some(path)) = (&key, &path)
        && let Some(r) = read(path).get(key).and_then(|m| m.get("is_normalized"))
    {
        return Ok((Measured { value: r.value, sample: r.sample, worst_deviation: r.worst_deviation }, Some(path.display().to_string())));
    }
    let m = measure_normalization(view)?;
    let (Some(key), Some(path)) = (key, path) else { return Ok((m, None)) };
    let mut all = read(&path);
    all.entry(key)
        .or_default()
        .insert("is_normalized".to_string(), RecordedMeasure { value: m.value, sample: m.sample, worst_deviation: m.worst_deviation });
    let text = format!(
        "# Measured by vectordata, by the content identity of the bytes measured. Machine-written.\n{}",
        serde_yaml::to_string(&all).map_err(|e| crate::Error::Other(e.to_string()))?
    );
    let recorded = path
        .parent()
        .map(std::fs::create_dir_all)
        .transpose()
        .and_then(|_| crate::atomic_file::replace_config(&path, text.as_bytes(), false))
        .is_ok();
    Ok((m, recorded.then(|| path.display().to_string())))
}

/// Whether `view`'s base is unit length, and on whose word
/// ([`Normalization`]). A declared value the sample contradicts is kept
/// and marked; when an overlay declared it, the contradiction is a
/// warning, once (OV-5), and [`TestDataView::verify`] reports it as
/// V8 either way.
pub fn normalization<V: TestDataView + ?Sized>(view: &V) -> crate::Result<Normalization> {
    let declared = view.dataset_attribute("is_normalized").and_then(|v| v.as_bool());
    let (measured, file) = recorded_normalization(view)?;
    Ok(match declared {
        Some(value) => {
            let source = view.dataset_attribute_source("is_normalized").unwrap_or(crate::overlay::AttributeSource::Dataset);
            let contradicted = value != measured.value;
            if contradicted && matches!(source, crate::overlay::AttributeSource::Overlay { .. }) {
                crate::overlay::warn_contradiction(
                    &view.cache_name().unwrap_or_else(|| "the dataset".to_string()),
                    "is_normalized",
                    &value.to_string(),
                    &format!("a sample of {} base vectors with a norm {:.3e} from 1", measured.sample, measured.worst_deviation),
                    &source,
                );
            }
            Normalization { value, source, measured, contradicted }
        }
        None => Normalization { value: measured.value, source: crate::overlay::AttributeSource::Measured { file }, measured, contradicted: false },
    })
}

/// The ground truths a profile may hold, with their distances, and
/// whether a short row may end in the `-1` sentinel.
const TRUTHS: [(&str, &str, bool); 3] = [
    ("neighbor_indices", "neighbor_distances", false),
    ("prefiltered_neighbor_indices", "prefiltered_neighbor_distances", true),
    ("postfiltered_neighbor_indices", "postfiltered_neighbor_distances", true),
];

/// Indices per scan block: 64 Ki.
const SCAN_BLOCK: usize = 1 << 16;

/// Run every check on `view`. The checks [`TestDataView::verify`](crate::TestDataView::verify)
/// and [`TestDataView::verify_report`](crate::TestDataView::verify_report) make.
pub(crate) fn verify_view<V: TestDataView + ?Sized>(
    view: &V,
    options: &VerifyOptions,
    progress: &mut dyn VerifyProgress,
) -> VerifyReport {
    let mut report = VerifyReport::default();
    let manifest = view.facet_manifest();
    let present = |name: &str| {
        if !manifest.contains_key(name) {
            return false;
        }
        if !options.skip_absent {
            return true;
        }
        // A series has no single path (`facet_source` is `None`) and is
        // checked; a local file that is not there is skipped.
        match view.facet_source(name) {
            Some(src) if !crate::transport::is_remote_url(&src) => std::path::Path::new(&src).exists(),
            _ => true,
        }
    };
    let file = |name: &str| view.facet_source(name);
    let fail = |report: &mut VerifyReport, check, facet: &str, found: String| {
        report.failures.push(VerifyFailure { check, facet: facet.to_string(), file: file(facet), found });
    };

    // V1: each facet that matters here opens, at its own type.
    let open = |report: &mut VerifyReport, name: &str| -> Option<(usize, usize)> {
        if !present(name) {
            return None;
        }
        if !report.checked.contains(&Check::V1Opens) {
            report.checked.push(Check::V1Opens);
        }
        match view.open_facet_native(name) {
            Ok(r) => Some((r.count(), r.dim())),
            Err(e) => {
                fail(report, Check::V1Opens, name, format!("does not open: {e}"));
                None
            }
        }
    };
    let base = open(&mut report, "base_vectors");
    let query = open(&mut report, "query_vectors");

    // V2: the queries are in the base's space.
    if let (Some((bn, bd)), Some((qn, qd))) = (base, query) {
        report.checked.push(Check::V2QueryDimension);
        if bn > 0 && qn > 0 && bd != qd {
            fail(&mut report, Check::V2QueryDimension, "query_vectors", format!("has dimension {qd}; the base has {bd}"));
        }
    }

    // V7: a declared base count is the base's: the profile's own, or,
    // for a profile that does not size the base, the dataset's (an
    // overlay's included), with its dimension.
    let attr_count = |key: &str| view.dataset_attribute(key).and_then(|v| v.as_u64());
    if let Some((bn, bd)) = base {
        let declared = view.base_count().map(|n| (n, "base_count declares")).or_else(|| {
            attr_count("base_count").map(|n| (n, "the dataset's base_count attribute states"))
        });
        if let Some((declared, by)) = declared {
            report.checked.push(Check::V7DeclaredCount);
            if bn as u64 != declared {
                fail(&mut report, Check::V7DeclaredCount, "base_vectors", format!("holds {bn} vectors; {by} {declared}"));
            }
        }
        if let Some(dim) = attr_count("dimension")
            && bn > 0
        {
            report.checked.push(Check::V7DeclaredCount);
            if bd as u64 != dim {
                fail(&mut report, Check::V7DeclaredCount, "base_vectors", format!("has dimension {bd}; the dataset's dimension attribute states {dim}"));
            }
        }
    }

    for (idx_key, dist_key, sparse) in TRUTHS {
        if !present(idx_key) {
            continue;
        }
        // Any integer type reads as i64; a u64 index above i64::MAX is
        // refused by the conversion at its ordinal, and reported.
        if !report.checked.contains(&Check::V1Opens) {
            report.checked.push(Check::V1Opens);
        }
        let ids = match view.facet_as::<i64>(idx_key) {
            Ok(r) => r,
            Err(e) => {
                fail(&mut report, Check::V1Opens, idx_key, format!("does not open: {e}"));
                continue;
            }
        };
        let (rows, k) = (ids.count(), ids.dim());

        // V3: one row per query.
        if let Some((q, _)) = query {
            report.checked.push(Check::V3TruthRows);
            if rows != q {
                fail(&mut report, Check::V3TruthRows, idx_key, format!("holds {rows} rows for {q} queries"));
            }
        }
        // V7: a stated maxk is the truth's width.
        if idx_key == "neighbor_indices"
            && rows > 0
            && let Some(maxk) = attr_count("maxk")
        {
            report.checked.push(Check::V7DeclaredCount);
            if k as u64 != maxk {
                fail(&mut report, Check::V7DeclaredCount, idx_key, format!("has {k} neighbors per row; the dataset's maxk attribute states {maxk}"));
            }
        }
        // V4: rows with something in them.
        report.checked.push(Check::V4TruthWidth);
        if rows > 0 && (k == 0 || k == crate::io::DIM_UNDEFINED) {
            fail(&mut report, Check::V4TruthWidth, idx_key, "has rows of no neighbors".to_string());
        }
        // V6: distances to go with them.
        if let Some((drows, dk)) = open(&mut report, dist_key) {
            report.checked.push(Check::V6Distances);
            if drows != rows {
                fail(&mut report, Check::V6Distances, dist_key, format!("holds {drows} rows for the {rows} rows of {idx_key}"));
            } else if rows > 0 && dk != k {
                fail(&mut report, Check::V6Distances, dist_key, format!("holds {dk} distances per row for the {k} neighbors of {idx_key}"));
            }
        }
        // V5: every index names a base vector.
        let Some((base_n, _)) = base else { continue };
        if rows == 0 || k == 0 || k == crate::io::DIM_UNDEFINED {
            continue;
        }
        report.checked.push(Check::V5TruthIndices);
        if let Some(found) = scan_indices(&*ids, base_n, sparse, &format!("scanning {idx_key}"), progress) {
            fail(&mut report, Check::V5TruthIndices, idx_key, found);
        }
    }

    // V8: a declared normalization is what the base shows — checked
    // when the base is here, so verifying never fetches a sample.
    if present("base_vectors")
        && view.dataset_attribute("is_normalized").and_then(|v| v.as_bool()).is_some()
        && view.open_facet_storage("base_vectors").is_ok_and(|s| s.is_complete())
        && let Ok(n) = normalization(view)
    {
        report.checked.push(Check::V8Normalized);
        if n.contradicted {
            fail(
                &mut report,
                Check::V8Normalized,
                "base_vectors",
                format!(
                    "is declared is_normalized: {} ({}) but a sample of {} vectors has a norm {:.3e} from 1",
                    n.value, n.source, n.measured.sample, n.measured.worst_deviation
                ),
            );
        }
    }

    // V9: what is cached is what landed.
    if options.cached_chunks {
        let mut names: Vec<&String> = manifest.keys().collect();
        names.sort();
        for name in names {
            if !present(name) {
                continue;
            }
            let Ok(storage) = view.open_facet_storage(name) else { continue };
            let label = format!("checking cached {name}");
            let mut hear = |_: &str, done: u64, total: u64| progress.report(&label, done, total);
            match storage.verify_cached(&mut hear) {
                Ok(None) => {}
                Ok(Some(bad)) => {
                    report.checked.push(Check::V9CachedChunks);
                    if !bad.is_empty() {
                        let chunks: Vec<String> = bad.iter().map(u32::to_string).collect();
                        fail(&mut report, Check::V9CachedChunks, name, format!("has cached chunks that no longer hash to what they were when they landed: {}", chunks.join(", ")));
                    }
                }
                Err(e) => fail(&mut report, Check::V9CachedChunks, name, format!("cannot be read to check its cache: {e}")),
            }
        }
    }
    report
}

/// Read every index of `ids` and return what is wrong with the first
/// that names no base vector of `base`, or the read that failed.
fn scan_indices(
    ids: &dyn VectorReader<i64>,
    base: usize,
    sparse: bool,
    label: &str,
    progress: &mut dyn VerifyProgress,
) -> Option<String> {
    let (rows, k) = (ids.count(), ids.dim());
    let per_block = (SCAN_BLOCK / k).max(1);
    let mut block = vec![0i64; per_block * k];
    let mut done = 0usize;
    progress.report(label, 0, rows as u64);
    while done < rows {
        let got = match ids.read_into(done, &mut block) {
            Ok(0) => break,
            // A value the conversion to i64 refuses, or a short read, is
            // a failure of the facet, not an end of it.
            Err(e) => return Some(format!("does not read: {e}")),
            Ok(n) => n,
        };
        let bad = block[..got * k].iter().position(|&id| !((0..base as i64).contains(&id) || (sparse && id == -1)));
        if let Some(at) = bad {
            progress.report(label, (done + at / k) as u64, rows as u64);
            return Some(format!(
                "row {} column {} holds {}, which names no base vector of {base} base vectors{}",
                done + at / k,
                at % k,
                block[at],
                if sparse { "" } else { "; only a filtered truth may hold the -1 sentinel" }
            ));
        }
        done += got;
        progress.report(label, done as u64, rows as u64);
    }
    None
}
