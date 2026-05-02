// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute neighbor distances from pre-existing
//! neighbor indices.
//!
//! Some source datasets ship ground-truth indices but not the
//! corresponding distances. When the pipeline aliases such a source
//! via the `Identity` slot, the default profile ends up with
//! `neighbor_indices.ivecs` but no `neighbor_distances.fvecs` — a
//! missing facet that downstream readers and prebufferers report as a
//! 403 / "facet not present" error.
//!
//! This command closes that gap by reading the existing indices, the
//! base, and the query vectors, computing each `(query, base[index])`
//! distance under the configured metric, and writing the row-major
//! `.fvecs` output. The output uses the same FAISS publication
//! convention (sign + sqrt-or-not) as `compute knn`, so verifiers can
//! compare values bit-for-bit.

use std::path::{Path, PathBuf};
use std::time::Instant;

use byteorder::{LittleEndian, WriteBytesExt};

use vectordata::VectorReader;
use vectordata::io::{VvecElement, XvecReader};

use crate::pipeline::atomic_write::{AtomicWriter, safe_create_file};
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};
use crate::pipeline::element_type::ElementType;
use crate::pipeline::simd_distance::{self, Metric};
use super::knn_segment::{CosineMode, resolve_cosine_mode_for};

pub struct ComputeKnnDistancesOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputeKnnDistancesOp)
}

fn error_result(msg: impl Into<String>, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message: msg.into(),
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn resolve_path(value: &str, workspace: &Path) -> PathBuf {
    let p = Path::new(value);
    if p.is_absolute() { p.to_path_buf() } else { workspace.join(p) }
}

fn parse_metric(s: &str) -> Result<Metric, String> {
    match s.to_uppercase().as_str() {
        "L2" | "EUCLIDEAN" => Ok(Metric::L2),
        "IP" | "DOT" | "DOT_PRODUCT" | "DOTPRODUCT" => Ok(Metric::DotProduct),
        "COSINE" | "COS" => Ok(Metric::Cosine),
        "L1" | "MANHATTAN" => Ok(Metric::L1),
        other => Err(format!(
            "unsupported metric '{}'; expected L2, IP/DOT, COSINE, or L1",
            other,
        )),
    }
}

/// Convert kernel-internal distance (smaller = better) to the on-disk
/// FAISS publication convention. Mirrors `compute_knn::publication_distance`
/// without taking on the dependency that file's monomorphic top-K
/// machinery would drag in.
#[inline]
fn publication_distance(d_kernel: f32, metric: Metric) -> f32 {
    if d_kernel.is_infinite() { return f32::INFINITY; }
    match metric {
        Metric::L2 => d_kernel,           // already L2sq, kernel == publication
        Metric::DotProduct => -d_kernel,  // kernel stores -dot, publication is +dot
        Metric::Cosine => 1.0 - d_kernel, // kernel stores 1-cos, publication is cos_sim
        Metric::L1 => d_kernel,           // L1 has no sign flip
    }
}

impl CommandOp for ComputeKnnDistancesOp {
    fn command_path(&self) -> &str {
        "compute knn-distances"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_COMPUTE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn build_version(&self) -> &'static str {
        concat!(env!("CARGO_PKG_VERSION"), "+", env!("VEKS_BUILD_HASH"), ".", env!("VEKS_BUILD_NUMBER"))
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Compute neighbor distances from existing neighbor indices".into(),
            body: format!(r#"# compute knn-distances

Materialize `neighbor_distances.fvecs` from a pre-existing
`neighbor_indices.ivecs` plus the base and query vectors.

## When to use

Some published datasets ship ground-truth neighbor indices but no
distances. When the dataset.yaml aliases such a source through the
`Identity` slot, the default profile ends up with indices
but no distances, and any downstream consumer that expects the full
`neighbor_distances` facet will fail.

This command fills the gap. For each query, it walks the recorded
indices, reads the corresponding base vectors, computes the configured
metric, and writes a row-major `.fvecs` output that matches the FAISS
publication convention used by `compute knn` — so verifiers can compare
bit-for-bit between sources that compute distances and sources that
recover them this way.

## Options

{}"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc { name: "readahead".into(), description: "Sequential read prefetch".into(), adjustable: false },
        ]
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "base".into(),
                type_name: "Path".into(),
                required: true,
                default: None,
                description: "Base vectors (.fvec or .mvec)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "query".into(),
                type_name: "Path".into(),
                required: true,
                default: None,
                description: "Query vectors (.fvec or .mvec)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "indices".into(),
                type_name: "Path".into(),
                required: true,
                default: None,
                description: "Existing neighbor indices (.ivec)".into(),
                role: OptionRole::Input,
            },
            OptionDesc {
                name: "output".into(),
                type_name: "Path".into(),
                required: true,
                default: None,
                description: "Output distances file (.fvec)".into(),
                role: OptionRole::Output,
            },
            OptionDesc {
                name: "metric".into(),
                type_name: "Metric".into(),
                required: false,
                default: Some("IP".into()),
                description: "L2, IP/DOT, COSINE, or L1".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "assume_normalized_like_faiss".into(),
                type_name: "bool".into(),
                required: false,
                default: None,
                description: "For COSINE metric: treat inputs as pre-normalized and evaluate cosine as inner product (FAISS / numpy / knn_utils convention). Exactly one of this and use_proper_cosine_metric must be set when metric=COSINE.".into(),
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "use_proper_cosine_metric".into(),
                type_name: "bool".into(),
                required: false,
                default: None,
                description: "For COSINE metric: compute cosine in-kernel from raw vectors via dot/(|q|×|b|). Use when inputs are not pre-normalized.".into(),
                role: OptionRole::Config,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        let base_str = match options.require("base") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let query_str = match options.require("query") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let indices_str = match options.require("indices") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let output_str = match options.require("output") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let metric = match parse_metric(options.get("metric").unwrap_or("IP")) {
            Ok(m) => m, Err(e) => return error_result(e, start),
        };
        // For COSINE the user must opt into one of two modes (mirrors
        // `compute knn-blas`). For other metrics this returns None.
        let cosine_mode = match resolve_cosine_mode_for(matches!(metric, Metric::Cosine), options) {
            Ok(m) => m, Err(e) => return error_result(e, start),
        };

        let base_path = resolve_path(base_str, &ctx.workspace);
        let query_path = resolve_path(query_str, &ctx.workspace);
        let indices_path = resolve_path(indices_str, &ctx.workspace);
        let output_path = resolve_path(output_str, &ctx.workspace);

        let base_etype = match ElementType::from_path(&base_path) {
            Ok(et @ (ElementType::F32 | ElementType::F16)) => et,
            Ok(et) => return error_result(
                format!("unsupported base element type {:?} for {}", et, base_path.display()),
                start,
            ),
            Err(e) => return error_result(format!("base element type: {}", e), start),
        };
        let query_etype = match ElementType::from_path(&query_path) {
            Ok(et @ (ElementType::F32 | ElementType::F16)) => et,
            Ok(et) => return error_result(
                format!("unsupported query element type {:?} for {}", et, query_path.display()),
                start,
            ),
            Err(e) => return error_result(format!("query element type: {}", e), start),
        };

        let indices_reader = match XvecReader::<i32>::open_path(&indices_path) {
            Ok(r) => r,
            Err(e) => return error_result(
                format!("open indices {}: {}", indices_path.display(), e), start,
            ),
        };
        let query_count = <XvecReader<i32> as VectorReader<i32>>::count(&indices_reader);
        let k = <XvecReader<i32> as VectorReader<i32>>::dim(&indices_reader);
        if query_count == 0 || k == 0 {
            return error_result(
                format!("indices file {} is empty (count={}, k={})",
                    indices_path.display(), query_count, k),
                start,
            );
        }

        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return error_result(format!("create output dir: {}", e), start);
                }
            }
        }

        // For COSINE+AssumeNormalized, we evaluate cosine as the inner
        // product (vectors are pre-normalized, so dot = cos_sim). The
        // distance kernel and publication conversion both run as if
        // the metric were DotProduct — matching `compute knn` /
        // `compute knn-blas` exactly so distances are bit-comparable.
        // For COSINE+ProperMetric (or anything else), the metric flows
        // through unchanged.
        let kernel_metric = match (metric, cosine_mode) {
            (Metric::Cosine, Some(CosineMode::AssumeNormalized)) => Metric::DotProduct,
            _ => metric,
        };

        ctx.ui.log(&format!(
            "compute knn-distances: {} queries × k={} ({:?} → publication f32, metric={:?}{})",
            query_count, k, base_etype, metric,
            match cosine_mode {
                Some(CosineMode::AssumeNormalized) => " [cosine: assume-normalized → dot]",
                Some(CosineMode::ProperMetric) => " [cosine: proper-metric]",
                None => "",
            },
        ));

        // Two arms — one per (base × query) element-type pair we
        // actually support (f32×f32, f16×f16, f32×f16, f16×f32). The
        // arms monomorphize the per-vector reads so the inner loop has
        // no dynamic dispatch.
        let result = match (base_etype, query_etype) {
            (ElementType::F32, ElementType::F32) => run::<f32, f32>(
                &base_path, &query_path, &indices_reader, &output_path,
                query_count, k, kernel_metric, ctx,
            ),
            (ElementType::F32, ElementType::F16) => run::<f32, half::f16>(
                &base_path, &query_path, &indices_reader, &output_path,
                query_count, k, kernel_metric, ctx,
            ),
            (ElementType::F16, ElementType::F32) => run::<half::f16, f32>(
                &base_path, &query_path, &indices_reader, &output_path,
                query_count, k, kernel_metric, ctx,
            ),
            (ElementType::F16, ElementType::F16) => run::<half::f16, half::f16>(
                &base_path, &query_path, &indices_reader, &output_path,
                query_count, k, kernel_metric, ctx,
            ),
            _ => unreachable!("element types pre-filtered"),
        };
        match result {
            Ok((rows_written, base_count)) => CommandResult {
                status: Status::Ok,
                message: format!(
                    "wrote {} × {} distances to {} (base_count={})",
                    rows_written, k, output_path.display(), base_count,
                ),
                produced: vec![output_path],
                elapsed: start.elapsed(),
            },
            Err(e) => error_result(e, start),
        }
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["base", "query", "indices"],
            &["output"],
        )
    }
}

/// Trait abstracting the per-element upcast to f32. Both supported
/// element types implement it natively in one machine instruction; the
/// trait exists so the inner loop can stay generic across (base, query)
/// combinations without runtime dispatch.
trait UpcastF32: Copy {
    fn to_f32(self) -> f32;
}
impl UpcastF32 for f32 {
    #[inline] fn to_f32(self) -> f32 { self }
}
impl UpcastF32 for half::f16 {
    #[inline] fn to_f32(self) -> f32 { half::f16::to_f32(self) }
}

/// Read every vector through the generic `VectorReader::get` API and
/// upcast into the supplied f32 buffer.
fn upcast_into<T: UpcastF32 + VvecElement>(
    reader: &XvecReader<T>, idx: usize, out: &mut [f32], dim: usize,
)
where XvecReader<T>: VectorReader<T>,
{
    let v = reader.get(idx).unwrap_or_else(|e| panic!("read[{}]: {}", idx, e));
    debug_assert_eq!(v.len(), dim);
    for (j, x) in v.iter().enumerate() {
        out[j] = (*x).to_f32();
    }
}

fn run<B: UpcastF32 + VvecElement, Q: UpcastF32 + VvecElement>(
    base_path: &Path,
    query_path: &Path,
    indices_reader: &XvecReader<i32>,
    output_path: &Path,
    query_count: usize,
    k: usize,
    metric: Metric,
    ctx: &mut StreamContext,
) -> Result<(usize, usize), String>
where
    XvecReader<B>: VectorReader<B>,
    XvecReader<Q>: VectorReader<Q>,
{
    let base_reader = XvecReader::<B>::open_path(base_path)
        .map_err(|e| format!("open base {}: {}", base_path.display(), e))?;
    let query_reader = XvecReader::<Q>::open_path(query_path)
        .map_err(|e| format!("open query {}: {}", query_path.display(), e))?;
    let base_count = <XvecReader<B> as VectorReader<B>>::count(&base_reader);
    let dim_base = <XvecReader<B> as VectorReader<B>>::dim(&base_reader);
    let dim_query = <XvecReader<Q> as VectorReader<Q>>::dim(&query_reader);
    let q_total = <XvecReader<Q> as VectorReader<Q>>::count(&query_reader);
    if dim_base != dim_query {
        return Err(format!(
            "dimension mismatch: base dim={}, query dim={}", dim_base, dim_query,
        ));
    }
    if q_total < query_count {
        return Err(format!(
            "query file has {} vectors but indices file has {} rows",
            q_total, query_count,
        ));
    }
    let dim = dim_base;
    base_reader.advise_sequential();

    let dist_fn = simd_distance::select_distance_fn(metric);

    // Output layout: one record per query, each record = (k as i32) +
    // k × f32. Use AtomicWriter so a partial write doesn't pollute the
    // workspace with a half-baked output.
    let f = safe_create_file(output_path)
        .map_err(|e| format!("create output: {}", e))?;
    drop(f); // AtomicWriter wants to own the path; the placeholder confirms parent dir is writable.
    let mut writer = AtomicWriter::new(output_path)
        .map_err(|e| format!("open atomic writer for {}: {}", output_path.display(), e))?;

    // Per-query scratch buffers. Sized to dim so reads upcast in place
    // with no per-row allocation.
    let mut q_buf: Vec<f32> = vec![0.0; dim];
    let mut b_buf: Vec<f32> = vec![0.0; dim];
    let mut row_dists: Vec<f32> = vec![0.0; k];

    let pb = ctx.ui.bar_with_unit(query_count as u64, "computing distances", "queries");
    let mut bad_index_msg: Option<String> = None;
    for qi in 0..query_count {
        upcast_into(&query_reader, qi, &mut q_buf, dim);
        let row = indices_reader.get_slice(qi);
        debug_assert_eq!(row.len(), k);
        for j in 0..k {
            let bi = row[j];
            if bi < 0 {
                row_dists[j] = f32::INFINITY;
                continue;
            }
            let bi_us = bi as usize;
            if bi_us >= base_count {
                bad_index_msg = Some(format!(
                    "indices[{}][{}] = {} is out of range (base_count={})",
                    qi, j, bi, base_count,
                ));
                break;
            }
            upcast_into(&base_reader, bi_us, &mut b_buf, dim);
            let d_kernel = dist_fn(&q_buf, &b_buf);
            row_dists[j] = publication_distance(d_kernel, metric);
        }
        if let Some(ref m) = bad_index_msg {
            return Err(m.clone());
        }
        // Write `(k as i32) + k × f32` little-endian.
        writer.write_i32::<LittleEndian>(k as i32)
            .map_err(|e| format!("write dim header: {}", e))?;
        for &d in &row_dists {
            writer.write_f32::<LittleEndian>(d)
                .map_err(|e| format!("write distance: {}", e))?;
        }
        if qi % 1024 == 0 { pb.set_position(qi as u64); }
    }
    pb.finish();
    writer.finish()
        .map_err(|e| format!("finalize output: {}", e))?;

    Ok((query_count, base_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::command::StreamContext;
    use crate::pipeline::progress::ProgressLog;
    use indexmap::IndexMap;
    use std::io::Write as _;

    fn make_ctx(workspace: &Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: workspace.to_path_buf(),
            cache: workspace.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    fn write_fvec(path: &Path, vectors: &[Vec<f32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for v in vectors {
            f.write_all(&(v.len() as i32).to_le_bytes()).unwrap();
            for &x in v { f.write_all(&x.to_le_bytes()).unwrap(); }
        }
    }
    fn write_ivec(path: &Path, rows: &[Vec<i32>]) {
        let mut f = std::fs::File::create(path).unwrap();
        for r in rows {
            f.write_all(&(r.len() as i32).to_le_bytes()).unwrap();
            for &x in r { f.write_all(&x.to_le_bytes()).unwrap(); }
        }
    }
    fn read_fvec(path: &Path) -> Vec<Vec<f32>> {
        use std::io::Read;
        let mut f = std::fs::File::open(path).unwrap();
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        let mut out = Vec::new();
        let mut p = 0usize;
        while p < buf.len() {
            let dim = i32::from_le_bytes([buf[p], buf[p+1], buf[p+2], buf[p+3]]) as usize;
            p += 4;
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                v.push(f32::from_le_bytes([buf[p], buf[p+1], buf[p+2], buf[p+3]]));
                p += 4;
            }
            out.push(v);
        }
        out
    }

    #[test]
    fn ip_distances_match_handcomputed() {
        let tmp = tempfile::tempdir().unwrap();
        let base_p = tmp.path().join("base.fvec");
        let query_p = tmp.path().join("query.fvec");
        let indices_p = tmp.path().join("indices.ivec");
        let output_p = tmp.path().join("distances.fvec");

        let base = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];
        let query = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        // Indices: query 0 → bases [0, 3]; query 1 → bases [1, 3].
        let indices = vec![vec![0i32, 3], vec![1, 3]];
        write_fvec(&base_p, &base);
        write_fvec(&query_p, &query);
        write_ivec(&indices_p, &indices);

        let mut ctx = make_ctx(tmp.path());
        let mut opts = Options::new();
        opts.set("base", base_p.to_str().unwrap());
        opts.set("query", query_p.to_str().unwrap());
        opts.set("indices", indices_p.to_str().unwrap());
        opts.set("output", output_p.to_str().unwrap());
        opts.set("metric", "IP");
        let res = ComputeKnnDistancesOp.execute(&opts, &mut ctx);
        assert_eq!(res.status, Status::Ok, "{}", res.message);

        let got = read_fvec(&output_p);
        assert_eq!(got.len(), 2);
        // Publication for IP = +dot. q0 · b0 = 1.0, q0 · b3 = 0.5.
        assert_eq!(got[0], vec![1.0, 0.5]);
        // q1 · b1 = 1.0, q1 · b3 = 0.5.
        assert_eq!(got[1], vec![1.0, 0.5]);
    }

    #[test]
    fn l2_distances_are_squared_l2() {
        let tmp = tempfile::tempdir().unwrap();
        let base_p = tmp.path().join("base.fvec");
        let query_p = tmp.path().join("query.fvec");
        let indices_p = tmp.path().join("indices.ivec");
        let output_p = tmp.path().join("distances.fvec");

        let base = vec![
            vec![0.0, 0.0],
            vec![3.0, 4.0], // L2sq from origin = 25
        ];
        let query = vec![vec![0.0, 0.0]];
        let indices = vec![vec![0i32, 1]];
        write_fvec(&base_p, &base);
        write_fvec(&query_p, &query);
        write_ivec(&indices_p, &indices);

        let mut ctx = make_ctx(tmp.path());
        let mut opts = Options::new();
        opts.set("base", base_p.to_str().unwrap());
        opts.set("query", query_p.to_str().unwrap());
        opts.set("indices", indices_p.to_str().unwrap());
        opts.set("output", output_p.to_str().unwrap());
        opts.set("metric", "L2");
        let res = ComputeKnnDistancesOp.execute(&opts, &mut ctx);
        assert_eq!(res.status, Status::Ok, "{}", res.message);

        let got = read_fvec(&output_p);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0][0], 0.0);
        assert!((got[0][1] - 25.0).abs() < 1e-5);
    }

    #[test]
    fn dim_mismatch_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let base_p = tmp.path().join("base.fvec");
        let query_p = tmp.path().join("query.fvec");
        let indices_p = tmp.path().join("indices.ivec");
        let output_p = tmp.path().join("distances.fvec");
        write_fvec(&base_p, &[vec![1.0, 2.0, 3.0]]);
        write_fvec(&query_p, &[vec![1.0, 2.0]]); // dim 2, mismatched
        write_ivec(&indices_p, &[vec![0i32]]);

        let mut ctx = make_ctx(tmp.path());
        let mut opts = Options::new();
        opts.set("base", base_p.to_str().unwrap());
        opts.set("query", query_p.to_str().unwrap());
        opts.set("indices", indices_p.to_str().unwrap());
        opts.set("output", output_p.to_str().unwrap());
        let res = ComputeKnnDistancesOp.execute(&opts, &mut ctx);
        assert_eq!(res.status, Status::Error);
        assert!(res.message.contains("dimension mismatch"), "{}", res.message);
    }

    #[test]
    fn cosine_assume_normalized_matches_ip_publication() {
        // Pre-normalized inputs; AssumeNormalized cosine should
        // produce the same +dot publication as IP would.
        let tmp = tempfile::tempdir().unwrap();
        let base_p = tmp.path().join("base.fvec");
        let query_p = tmp.path().join("query.fvec");
        let indices_p = tmp.path().join("indices.ivec");
        let output_p_ip = tmp.path().join("d_ip.fvec");
        let output_p_cos = tmp.path().join("d_cos.fvec");

        let s = 1.0_f32 / (2.0_f32).sqrt();
        let base = vec![vec![1.0, 0.0], vec![s, s]];
        let query = vec![vec![1.0, 0.0]];
        let indices = vec![vec![0i32, 1]];
        write_fvec(&base_p, &base);
        write_fvec(&query_p, &query);
        write_ivec(&indices_p, &indices);

        let mut ctx = make_ctx(tmp.path());
        let mut opts_ip = Options::new();
        opts_ip.set("base", base_p.to_str().unwrap());
        opts_ip.set("query", query_p.to_str().unwrap());
        opts_ip.set("indices", indices_p.to_str().unwrap());
        opts_ip.set("output", output_p_ip.to_str().unwrap());
        opts_ip.set("metric", "IP");
        let r = ComputeKnnDistancesOp.execute(&opts_ip, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);

        let mut opts_cos = Options::new();
        opts_cos.set("base", base_p.to_str().unwrap());
        opts_cos.set("query", query_p.to_str().unwrap());
        opts_cos.set("indices", indices_p.to_str().unwrap());
        opts_cos.set("output", output_p_cos.to_str().unwrap());
        opts_cos.set("metric", "COSINE");
        opts_cos.set("assume_normalized_like_faiss", "true");
        let r = ComputeKnnDistancesOp.execute(&opts_cos, &mut ctx);
        assert_eq!(r.status, Status::Ok, "{}", r.message);

        let ip = read_fvec(&output_p_ip);
        let cos = read_fvec(&output_p_cos);
        assert_eq!(ip, cos, "AssumeNormalized cosine must match IP publication");
    }

    #[test]
    fn cosine_without_mode_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let base_p = tmp.path().join("base.fvec");
        let query_p = tmp.path().join("query.fvec");
        let indices_p = tmp.path().join("indices.ivec");
        let output_p = tmp.path().join("d.fvec");
        write_fvec(&base_p, &[vec![1.0, 0.0]]);
        write_fvec(&query_p, &[vec![1.0, 0.0]]);
        write_ivec(&indices_p, &[vec![0i32]]);

        let mut ctx = make_ctx(tmp.path());
        let mut opts = Options::new();
        opts.set("base", base_p.to_str().unwrap());
        opts.set("query", query_p.to_str().unwrap());
        opts.set("indices", indices_p.to_str().unwrap());
        opts.set("output", output_p.to_str().unwrap());
        opts.set("metric", "COSINE");
        // Neither flag set — should error per the same contract as
        // compute knn-blas.
        let r = ComputeKnnDistancesOp.execute(&opts, &mut ctx);
        assert_eq!(r.status, Status::Error);
        assert!(r.message.to_lowercase().contains("cosine"), "{}", r.message);
    }

    #[test]
    fn out_of_range_index_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let base_p = tmp.path().join("base.fvec");
        let query_p = tmp.path().join("query.fvec");
        let indices_p = tmp.path().join("indices.ivec");
        let output_p = tmp.path().join("distances.fvec");
        write_fvec(&base_p, &[vec![1.0, 0.0]]);
        write_fvec(&query_p, &[vec![1.0, 0.0]]);
        write_ivec(&indices_p, &[vec![5i32]]); // index 5 doesn't exist

        let mut ctx = make_ctx(tmp.path());
        let mut opts = Options::new();
        opts.set("base", base_p.to_str().unwrap());
        opts.set("query", query_p.to_str().unwrap());
        opts.set("indices", indices_p.to_str().unwrap());
        opts.set("output", output_p.to_str().unwrap());
        let res = ComputeKnnDistancesOp.execute(&opts, &mut ctx);
        assert_eq!(res.status, Status::Error);
        assert!(res.message.contains("out of range"), "{}", res.message);
    }
}
