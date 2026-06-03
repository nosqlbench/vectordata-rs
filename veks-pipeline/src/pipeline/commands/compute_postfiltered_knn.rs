// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: compute the post-filter KNN ground-truth (E facet).
//!
//! `F = G ∩ R` — for each query, keep only those neighbors from the
//! unfiltered top-K (G) whose ordinal also passes the predicate (R).
//! This is ACORN's **post-filtering** (§3.2) restricted to the
//! no-scope-expansion case: the search scope is fixed at the unfiltered
//! top-K, and predicate failures are dropped without rescue.
//!
//! The result is **sparse**: `|F| ∈ [0, K]`, depending on how well the
//! query's unfiltered nearest neighbors satisfy its predicate.
//! Restrictive or query-uncorrelated predicates produce small F sets;
//! permissive or query-correlated predicates produce near-full sets.
//!
//! Cheap to compute: the producer reads `G` (neighbor_indices), `D`
//! (neighbor_distances, optional), and `R` (metadata_indices) only —
//! no base/query rereads, no distance recomputation. Cost is O(K) per
//! query for the intersection test plus an O(|R|) membership-set
//! construction.
//!
//! See `docs/design/prefilter-postfilter-facets.md` for context, and
//! `compute prefiltered-knn` for the perfect-recall pre-filter sibling
//! (F facet).

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vectordata::io::XvecReader;
use vectordata::VectorReader;

use crate::pipeline::atomic_write::AtomicWriter;
use crate::pipeline::command::{
    ArtifactManifest, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};
use super::compute_prefiltered_knn::PredicateIndices;

/// Pipeline command: derive post-filter KNN ground truth.
pub struct ComputePostfilteredKnnOp;

/// Create a boxed `ComputePostfilteredKnnOp` command.
pub fn factory() -> Box<dyn CommandOp> {
    Box::new(ComputePostfilteredKnnOp)
}

impl CommandOp for ComputePostfilteredKnnOp {
    fn command_path(&self) -> &str {
        "compute postfiltered-knn"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_COMPUTE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag { &crate::pipeline::command::LVL_PRIMARY }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Derive post-filter KNN ground truth (E facet) as G ∩ R".into(),
            body: format!(r#"# compute postfiltered-knn

Derive the **E facet** — post-filter KNN ground truth — as the
intersection of the unfiltered top-K (G) with the predicate-passing
set (R).

## Description

ACORN's *post-filtering* (§3.2) performs ANN-search over the full base
vector set first, then expands the search scope to find K
predicate-passing survivors. This command produces the **no-scope-
expansion** specialisation: the search scope is exactly the unfiltered
top-K, and predicate failures are dropped without rescue. The output
matches what a naive post-filter ANN engine would return for queries
where its search scope equals the ground-truth top-K.

For each query:

1. Read its unfiltered top-K neighbors `G_q` and (optionally) their
   distances `D_q`.
2. Read the predicate-matching ordinal set `R_q`.
3. Emit only those ordinals in `G_q` that also appear in `R_q`,
   preserving G's distance ordering.
4. Pad the remainder with sentinel values (-1 indices, +∞ distances)
   to length K.

Result cardinality `|F_q| ∈ [0, K]`. Sparsity is *expected*; it is the
correct ground truth for evaluating ANN engines that do not perform
search-scope expansion to compensate for predicate failures.

## Inputs

- `--ground-truth` (G facet): unfiltered top-K indices, `.ivec` from
  `compute knn` or its variants.
- `--ground-truth-distances` (D facet, optional): unfiltered top-K
  distances paired with G. When supplied, survivor distances are
  copied verbatim into the F output. When omitted, the output omits
  distances.
- `--metadata-indices` (R facet): per-query predicate-matching base
  ordinals, slab or `ivvec`, from `compute evaluate-predicates`.

## Outputs

- `--indices`: `postfiltered_neighbor_indices.ivec`.
- `--distances` (optional): `postfiltered_neighbor_distances.fvec`.

## Distance sign convention

Distances are passed through verbatim from the D input, preserving its
sign convention (FAISS publication convention for KNN outputs from
`compute knn*`). No conversion is applied.

## Options

{}

## Notes

- The producer is cheap (O(K) per query). It does *not* re-open base
  or query vectors and does *not* recompute distances.
- Queries with empty `R_q` produce an all-sentinel F row.
- See `compute prefiltered-knn` for the perfect-recall sibling (F facet).
"#, render_options_table(&options)),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![
            ResourceDesc {
                name: "mem".into(),
                description: "Per-query predicate-set membership tests (O(|R_q|) hash sets)".into(),
                adjustable: false,
            },
        ]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();

        // Resolve the G (ground-truth) input through the facet layer: when
        // the pipeline doesn't pass an explicit `--ground-truth`, this
        // resolves the profile's `neighbor_indices` facet from
        // `dataset.yaml` — the correct per-profile path, with the locator's
        // `.ivecs`/`.ivec` tolerance for extant datasets. (The previous
        // hardcoded literal was both in the wrong directory and the wrong
        // extension.)
        let gt_str = match crate::pipeline::dataset_lookup::resolve_path_option(
            ctx, options, "ground-truth", "neighbor_indices",
        ) {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let keys_str = match options.require("metadata-indices") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };
        let indices_str = match options.require("indices") {
            Ok(s) => s, Err(e) => return error_result(e, start),
        };

        let gt_path = resolve_path(&gt_str, &ctx.workspace);
        let keys_path = resolve_path(&keys_str, &ctx.workspace);
        let indices_path = resolve_path(&indices_str, &ctx.workspace);

        let distances_path = options.get("distances")
            .map(|s| resolve_path(s, &ctx.workspace));
        // D (ground-truth-distances) is needed only to emit E's distances.
        // Resolve it the same way as G when a distances output is wanted
        // and no explicit input was supplied.
        let gt_distances_path = if let Some(s) = options.get("ground-truth-distances") {
            Some(resolve_path(s, &ctx.workspace))
        } else if distances_path.is_some() {
            match crate::pipeline::dataset_lookup::resolve_path_option(
                ctx, options, "ground-truth-distances", "neighbor_distances",
            ) {
                Ok(s) => Some(resolve_path(&s, &ctx.workspace)),
                Err(e) => return error_result(e, start),
            }
        } else {
            None
        };

        // Distances output without distances input is nonsensical — we
        // have nothing to write. Reject up front rather than silently
        // producing all-sentinel distance rows.
        if distances_path.is_some() && gt_distances_path.is_none() {
            return error_result(
                "--distances output requested but --ground-truth-distances input not supplied; \
                 cannot derive F distances without D".into(),
                start,
            );
        }

        // Create parent dirs for outputs.
        for path in std::iter::once(&indices_path).chain(distances_path.iter()) {
            if let Some(parent) = path.parent()
                && !parent.exists()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                return error_result(format!("create output dir: {}", e), start);
            }
        }

        let gt_reader = match XvecReader::<i32>::open_path(&gt_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open ground-truth {}: {}", gt_path.display(), e), start),
        };
        let gt_dist_reader = match gt_distances_path.as_ref() {
            Some(p) => match XvecReader::<f32>::open_path(p) {
                Ok(r) => Some(r),
                Err(e) => return error_result(format!("open ground-truth-distances {}: {}", p.display(), e), start),
            },
            None => None,
        };
        let keys_reader = match PredicateIndices::open(&keys_path) {
            Ok(r) => r,
            Err(e) => return error_result(format!("open metadata-indices {}: {}", keys_path.display(), e), start),
        };

        let query_count = <XvecReader<i32> as VectorReader<i32>>::count(&gt_reader);
        let keys_count = keys_reader.count();
        if query_count != keys_count {
            // We tolerate a mismatch with a warning rather than failing —
            // it's plausible to ship a metadata-indices file covering a
            // superset of queries, but a strict typo is also worth
            // surfacing.
            ctx.ui.log(&format!(
                "  WARNING: G has {} queries, R has {} predicates; using min={}",
                query_count, keys_count, query_count.min(keys_count),
            ));
        }
        let n = query_count.min(keys_count);

        // The k for F equals the k of G; we just filter-and-pad to that
        // width. Read one row to discover it.
        if n == 0 {
            return error_result("ground-truth file has zero queries".into(), start);
        }
        let k = gt_reader.get_slice(0).len();
        if k == 0 {
            return error_result("ground-truth k=0; cannot derive F".into(), start);
        }

        // Validate D shape matches G when both are present.
        if let Some(ref dr) = gt_dist_reader {
            let dn = <XvecReader<f32> as VectorReader<f32>>::count(dr);
            if dn < n {
                return error_result(
                    format!("ground-truth-distances has fewer queries ({}) than ground-truth ({})", dn, n),
                    start,
                );
            }
            let dk = dr.get_slice(0).len();
            if dk != k {
                return error_result(
                    format!("ground-truth-distances k={} differs from ground-truth k={}", dk, k),
                    start,
                );
            }
        }

        let mut idx_writer = match AtomicWriter::with_capacity(1 << 20, &indices_path) {
            Ok(w) => w,
            Err(e) => return error_result(format!("create {}: {}", indices_path.display(), e), start),
        };
        let mut dist_writer = match distances_path.as_ref() {
            Some(p) => match AtomicWriter::with_capacity(1 << 20, p) {
                Ok(w) => Some(w),
                Err(e) => return error_result(format!("create {}: {}", p.display(), e), start),
            },
            None => None,
        };

        let pb = ctx.ui.bar(n as u64, "intersecting G ∩ R");
        let dim_le = (k as i32).to_le_bytes();

        let mut produced = vec![indices_path.clone()];
        if let Some(p) = distances_path.as_ref() {
            produced.push(p.clone());
        }

        let mut survivors_total: u64 = 0;
        let mut sentinels_total: u64 = 0;

        for q in 0..n {
            let g_row = gt_reader.get_slice(q);
            let d_row: Option<&[f32]> = gt_dist_reader.as_ref().map(|r| r.get_slice(q));
            let r_ords = match keys_reader.get_ordinals(q) {
                Ok(v) => v,
                Err(e) => return error_result(format!("read R record {}: {}", q, e), start),
            };
            // Hash-set membership: O(K) tests * O(1) hash.
            let r_set: HashSet<i32> = r_ords.into_iter().collect();

            if let Err(e) = idx_writer.write_all(&dim_le) {
                return error_result(format!("write indices header: {}", e), start);
            }
            if let Some(ref mut dw) = dist_writer
                && let Err(e) = dw.write_all(&dim_le)
            {
                return error_result(format!("write distances header: {}", e), start);
            }

            // Walk G in rank order; emit survivors first, then sentinel
            // pad to k. Preserving rank order keeps the F output sorted
            // by distance (since G is rank-sorted), which downstream
            // consumers rely on.
            let mut emitted = 0usize;
            for (i, &ord) in g_row.iter().enumerate() {
                if ord < 0 { continue; } // G sentinel
                if !r_set.contains(&ord) { continue; }
                if let Err(e) = idx_writer.write_all(&ord.to_le_bytes()) {
                    return error_result(format!("write F index q={}: {}", q, e), start);
                }
                if let Some(ref mut dw) = dist_writer
                    && let Some(drow) = d_row
                {
                    let dist = drow[i];
                    if let Err(e) = dw.write_all(&dist.to_le_bytes()) {
                        return error_result(format!("write F distance q={}: {}", q, e), start);
                    }
                }
                emitted += 1;
            }
            // Sentinel pad.
            let pad = k - emitted;
            survivors_total += emitted as u64;
            sentinels_total += pad as u64;
            for _ in 0..pad {
                if let Err(e) = idx_writer.write_all(&(-1i32).to_le_bytes()) {
                    return error_result(format!("write F sentinel q={}: {}", q, e), start);
                }
                if let Some(ref mut dw) = dist_writer
                    && let Err(e) = dw.write_all(&f32::INFINITY.to_le_bytes())
                {
                    return error_result(format!("write F sentinel distance q={}: {}", q, e), start);
                }
            }
            if (q + 1) % 1024 == 0 || q + 1 == n {
                pb.set_position((q + 1) as u64);
            }
        }
        pb.finish();

        if let Err(e) = idx_writer.finish() {
            return error_result(format!("finalise {}: {}", indices_path.display(), e), start);
        }
        if let Some(dw) = dist_writer
            && let Err(e) = dw.finish()
        {
            return error_result(format!("finalise distances: {}", e), start);
        }

        // Verified-count side-effect — matches the convention used by
        // compute prefiltered-knn so downstream bound checkers find the
        // count without re-opening the file.
        for xvec_path in &produced {
            let var_name = format!(
                "verified_count:{}",
                xvec_path.file_name().and_then(|n| n.to_str()).unwrap_or("output"),
            );
            let _ = crate::pipeline::variables::set_and_save(
                &ctx.workspace, &var_name, &n.to_string(),
            );
            ctx.defaults.insert(var_name, n.to_string());
        }

        let avg_survivors = if n > 0 { survivors_total as f64 / n as f64 } else { 0.0 };
        let avg_sentinels = if n > 0 { sentinels_total as f64 / n as f64 } else { 0.0 };

        CommandResult {
            status: Status::Ok,
            message: format!(
                "postfiltered KNN (F = G ∩ R): {} queries, k={}, avg survivors {:.2}, avg sentinels {:.2}",
                n, k, avg_survivors, avg_sentinels,
            ),
            produced,
            elapsed: start.elapsed(),
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            // Optional at the schema level: when the pipeline doesn't pass
            // an explicit path, the command resolves the `neighbor_indices`
            // facet from dataset.yaml (per-profile, ext-tolerant). Mirrors
            // how the verify_* commands declare facet-resolvable inputs.
            opt("ground-truth", "Path", false, None,
                "Unfiltered KNN indices (G facet, ivec/ivecs)", OptionRole::Input),
            opt("ground-truth-distances", "Path", false, None,
                "Unfiltered KNN distances (D facet, fvec/fvecs). When supplied, survivor distances are copied into the F output.",
                OptionRole::Input),
            opt("metadata-indices", "Path", true, None,
                "Predicate-matching base ordinals per query (R facet, slab or ivvec) from compute evaluate-predicates",
                OptionRole::Input),
            opt("indices", "Path", true, None,
                "Output post-filter neighbor indices (postfiltered_neighbor_indices.ivec)",
                OptionRole::Output),
            opt("distances", "Path", false, None,
                "Output post-filter neighbor distances (postfiltered_neighbor_distances.fvec). Requires --ground-truth-distances.",
                OptionRole::Output),
        ]
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        crate::pipeline::command::manifest_from_keys(
            step_id, self.command_path(), options,
            &["ground-truth", "ground-truth-distances", "metadata-indices"],
            &["indices", "distances"],
        )
    }
}

fn resolve_path(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

fn error_result(message: String, start: Instant) -> CommandResult {
    CommandResult {
        status: Status::Error,
        message,
        produced: vec![],
        elapsed: start.elapsed(),
    }
}

fn opt(
    name: &str, type_name: &str, required: bool, default: Option<&str>,
    desc: &str, role: OptionRole,
) -> OptionDesc {
    OptionDesc {
        name: name.to_string(),
        type_name: type_name.to_string(),
        required,
        default: default.map(|s| s.to_string()),
        description: desc.to_string(),
        extended_description: None,
        role,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet as StdHashSet;
    use std::fs;
    use indexmap::IndexMap;
    use tempfile::TempDir;

    /// Standard StreamContext shape used across the producer's tests —
    /// matches the helper in `compute_prefiltered_knn::tests`.
    fn test_ctx(dir: &std::path::Path) -> StreamContext {
        StreamContext {
            dataset_name: String::new(),
            profile: String::new(),
            profile_names: vec![],
            workspace: dir.to_path_buf(),
            cache: dir.join(".cache"),
            defaults: IndexMap::new(),
            dry_run: false,
            progress: crate::pipeline::progress::ProgressLog::new(),
            threads: 1,
            step_id: String::new(),
            governor: crate::pipeline::resource::ResourceGovernor::default_governor(),
            ui: veks_core::ui::UiHandle::new(std::sync::Arc::new(veks_core::ui::TestSink::new())),
            status_interval: std::time::Duration::from_secs(1),
            estimated_total_steps: 0,
            provenance_selector: crate::pipeline::provenance::ProvenanceFlags::STRICT,
        }
    }

    /// Write an ivec file: each row is `[k:i32 LE][k * i32 LE]`.
    fn write_ivec(path: &Path, rows: &[Vec<i32>]) {
        let mut f = fs::File::create(path).unwrap();
        for row in rows {
            let dim = row.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for v in row {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }

    /// Write an fvec file: each row is `[k:i32 LE][k * f32 LE]`.
    fn write_fvec(path: &Path, rows: &[Vec<f32>]) {
        let mut f = fs::File::create(path).unwrap();
        for row in rows {
            let dim = row.len() as i32;
            f.write_all(&dim.to_le_bytes()).unwrap();
            for v in row {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }

    /// Read an ivec back into rows, including sentinels.
    fn read_ivec(path: &Path) -> Vec<Vec<i32>> {
        let r = XvecReader::<i32>::open_path(path).unwrap();
        let n = <XvecReader<i32> as VectorReader<i32>>::count(&r);
        (0..n).map(|i| r.get_slice(i).to_vec()).collect()
    }

    /// Read an fvec back into rows, including sentinels.
    fn read_fvec(path: &Path) -> Vec<Vec<f32>> {
        let r = XvecReader::<f32>::open_path(path).unwrap();
        let n = <XvecReader<f32> as VectorReader<f32>>::count(&r);
        (0..n).map(|i| r.get_slice(i).to_vec()).collect()
    }

    /// Pack a Vec<Vec<i32>> as a slab so it can be opened by PredicateIndices.
    /// Mirrors the slab layout that `compute evaluate-predicates` writes
    /// (each record is a packed `[i32 LE]*` array of ordinals).
    fn write_predicate_slab(path: &Path, rows: &[Vec<i32>]) {
        use slabtastic::{SlabWriter, WriterConfig};
        let config = WriterConfig::new(512, 4096, u32::MAX, false).unwrap();
        let mut w = SlabWriter::new(path, config).unwrap();
        for row in rows {
            let mut bytes = Vec::with_capacity(row.len() * 4);
            for v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            w.add_record(&bytes).unwrap();
        }
        w.finish().unwrap();
    }

    /// Run the producer end-to-end on a tiny fixture and validate that
    /// F == G ∩ R byte-for-byte, with sentinel padding to k.
    #[test]
    fn postfilter_equals_g_intersect_r() {
        let tmp = TempDir::new().unwrap();

        // G: 3 queries, k=4. Use distinct ordinals so intersection is
        // easy to reason about.
        let g_rows: Vec<Vec<i32>> = vec![
            vec![10, 20, 30, 40],
            vec![ 5, 15, 25, 35],
            vec![ 1,  2,  3,  4],
        ];
        // D: matching distances, ascending (G is rank-sorted).
        let d_rows: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![1.1, 1.2, 1.3, 1.4],
            vec![2.1, 2.2, 2.3, 2.4],
        ];
        // R: per-query predicate matches. Query 0 keeps only 20 & 40;
        // query 1 keeps 5; query 2 keeps all four.
        let r_rows: Vec<Vec<i32>> = vec![
            vec![20, 40, 99],            // 99 not in G; ignored
            vec![5, 100],                // only 5 survives
            vec![1, 2, 3, 4, 50, 60],
        ];

        let g_path = tmp.path().join("g.ivec");
        let d_path = tmp.path().join("d.fvec");
        let r_path = tmp.path().join("r.slab");
        let f_idx_path = tmp.path().join("f.ivec");
        let f_dist_path = tmp.path().join("f.fvec");

        write_ivec(&g_path, &g_rows);
        write_fvec(&d_path, &d_rows);
        write_predicate_slab(&r_path, &r_rows);

        // Drive execute() through a real Options + StreamContext.
        let mut opts = Options::new();
        opts.set("ground-truth", g_path.to_string_lossy().to_string());
        opts.set("ground-truth-distances", d_path.to_string_lossy().to_string());
        opts.set("metadata-indices", r_path.to_string_lossy().to_string());
        opts.set("indices", f_idx_path.to_string_lossy().to_string());
        opts.set("distances", f_dist_path.to_string_lossy().to_string());

        let mut ctx = test_ctx(tmp.path());
        let mut op = ComputePostfilteredKnnOp;
        let res = op.execute(&opts, &mut ctx);
        assert_eq!(res.status, Status::Ok, "execute should succeed: {}", res.message);

        // Validate indices: survivors first (in G rank order), then -1 pad.
        let f_idx = read_ivec(&f_idx_path);
        assert_eq!(f_idx.len(), 3, "F should have 3 rows");
        assert_eq!(f_idx[0], vec![20, 40, -1, -1], "q0 survivors then pad");
        assert_eq!(f_idx[1], vec![5, -1, -1, -1], "q1 sole survivor then pad");
        assert_eq!(f_idx[2], vec![1, 2, 3, 4], "q2 all survive, no pad");

        // Validate distances: copied from D for survivors, +∞ for pad.
        let f_dist = read_fvec(&f_dist_path);
        assert_eq!(f_dist[0][0], 0.2, "q0 survivor 0 dist from D");
        assert_eq!(f_dist[0][1], 0.4, "q0 survivor 1 dist from D");
        assert!(f_dist[0][2].is_infinite() && f_dist[0][2] > 0.0, "q0 pad dist = +inf");
        assert!(f_dist[0][3].is_infinite() && f_dist[0][3] > 0.0, "q0 pad dist = +inf");
        assert_eq!(f_dist[1][0], 1.1, "q1 sole survivor dist from D");
        assert_eq!(f_dist[2], vec![2.1, 2.2, 2.3, 2.4], "q2 all distances copied");
    }

    /// A query whose predicate matches nothing produces an all-sentinel
    /// F row (still length k). Regression pin: the producer never emits
    /// short rows.
    #[test]
    fn postfilter_empty_predicate_yields_all_sentinel_row() {
        let tmp = TempDir::new().unwrap();
        let g_rows = vec![vec![10, 20, 30]];
        let r_rows = vec![Vec::<i32>::new()];

        let g_path = tmp.path().join("g.ivec");
        let r_path = tmp.path().join("r.slab");
        let f_idx_path = tmp.path().join("f.ivec");

        write_ivec(&g_path, &g_rows);
        write_predicate_slab(&r_path, &r_rows);

        let mut opts = Options::new();
        opts.set("ground-truth", g_path.to_string_lossy().to_string());
        opts.set("metadata-indices", r_path.to_string_lossy().to_string());
        opts.set("indices", f_idx_path.to_string_lossy().to_string());

        let mut ctx = test_ctx(tmp.path());
        let mut op = ComputePostfilteredKnnOp;
        let res = op.execute(&opts, &mut ctx);
        assert_eq!(res.status, Status::Ok);

        let f_idx = read_ivec(&f_idx_path);
        assert_eq!(f_idx[0], vec![-1, -1, -1]);
    }

    /// Asking for distance output without supplying D is a configuration
    /// error — we cannot synthesise distances from G alone.
    #[test]
    fn postfilter_rejects_distances_output_without_d_input() {
        let tmp = TempDir::new().unwrap();
        let g_rows = vec![vec![1, 2]];
        let r_rows = vec![vec![1, 2]];

        let g_path = tmp.path().join("g.ivec");
        let r_path = tmp.path().join("r.slab");
        let f_idx_path = tmp.path().join("f.ivec");
        let f_dist_path = tmp.path().join("f.fvec");

        write_ivec(&g_path, &g_rows);
        write_predicate_slab(&r_path, &r_rows);

        let mut opts = Options::new();
        opts.set("ground-truth", g_path.to_string_lossy().to_string());
        opts.set("metadata-indices", r_path.to_string_lossy().to_string());
        opts.set("indices", f_idx_path.to_string_lossy().to_string());
        opts.set("distances", f_dist_path.to_string_lossy().to_string());

        let mut ctx = test_ctx(tmp.path());
        let mut op = ComputePostfilteredKnnOp;
        let res = op.execute(&opts, &mut ctx);
        assert_eq!(res.status, Status::Error);
        assert!(res.message.contains("ground-truth-distances"),
            "error should explain the missing D input: {}", res.message);
    }

    /// F survivors preserve G's rank order. The producer must walk G in
    /// order and emit only the indices that pass the predicate — never
    /// resort by ordinal value.
    #[test]
    fn postfilter_preserves_g_rank_order() {
        let tmp = TempDir::new().unwrap();
        // G in non-monotonic ordinal order — only the rank order matters.
        let g_rows = vec![vec![50, 10, 30, 20, 40]];
        // Predicate keeps everything; survivors must come out in G order.
        let r_rows = vec![vec![10, 20, 30, 40, 50]];

        let g_path = tmp.path().join("g.ivec");
        let r_path = tmp.path().join("r.slab");
        let f_idx_path = tmp.path().join("f.ivec");

        write_ivec(&g_path, &g_rows);
        write_predicate_slab(&r_path, &r_rows);

        let mut opts = Options::new();
        opts.set("ground-truth", g_path.to_string_lossy().to_string());
        opts.set("metadata-indices", r_path.to_string_lossy().to_string());
        opts.set("indices", f_idx_path.to_string_lossy().to_string());

        let mut ctx = test_ctx(tmp.path());
        let mut op = ComputePostfilteredKnnOp;
        let res = op.execute(&opts, &mut ctx);
        assert_eq!(res.status, Status::Ok);

        let f_idx = read_ivec(&f_idx_path);
        assert_eq!(f_idx[0], vec![50, 10, 30, 20, 40],
            "survivors must come out in the order they appear in G");
    }

    /// G sentinels (-1) are skipped, never treated as ordinal 0 or a real
    /// neighbor candidate.
    #[test]
    fn postfilter_skips_g_sentinels() {
        let tmp = TempDir::new().unwrap();
        let g_rows = vec![vec![10, -1, 20, -1]];
        let r_rows = vec![vec![10, 20]];

        let g_path = tmp.path().join("g.ivec");
        let r_path = tmp.path().join("r.slab");
        let f_idx_path = tmp.path().join("f.ivec");

        write_ivec(&g_path, &g_rows);
        write_predicate_slab(&r_path, &r_rows);

        let mut opts = Options::new();
        opts.set("ground-truth", g_path.to_string_lossy().to_string());
        opts.set("metadata-indices", r_path.to_string_lossy().to_string());
        opts.set("indices", f_idx_path.to_string_lossy().to_string());

        let mut ctx = test_ctx(tmp.path());
        let mut op = ComputePostfilteredKnnOp;
        let res = op.execute(&opts, &mut ctx);
        assert_eq!(res.status, Status::Ok);

        let f_idx = read_ivec(&f_idx_path);
        assert_eq!(f_idx[0], vec![10, 20, -1, -1]);
    }

    /// F should be exactly the set-intersection of G and R (modulo ordering)
    /// — sanity-check against a hash-set computation on the same fixture.
    #[test]
    fn postfilter_set_equals_intersection_set() {
        let tmp = TempDir::new().unwrap();
        let g_rows = vec![vec![1, 2, 3, 4, 5, 6, 7]];
        let r_rows = vec![vec![2, 4, 6, 8, 10]];

        let g_path = tmp.path().join("g.ivec");
        let r_path = tmp.path().join("r.slab");
        let f_idx_path = tmp.path().join("f.ivec");

        write_ivec(&g_path, &g_rows);
        write_predicate_slab(&r_path, &r_rows);

        let mut opts = Options::new();
        opts.set("ground-truth", g_path.to_string_lossy().to_string());
        opts.set("metadata-indices", r_path.to_string_lossy().to_string());
        opts.set("indices", f_idx_path.to_string_lossy().to_string());

        let mut ctx = test_ctx(tmp.path());
        let mut op = ComputePostfilteredKnnOp;
        let res = op.execute(&opts, &mut ctx);
        assert_eq!(res.status, Status::Ok);

        let f_idx = read_ivec(&f_idx_path);
        let survivors: StdHashSet<i32> = f_idx[0].iter().copied().filter(|&v| v >= 0).collect();
        let g_set: StdHashSet<i32> = g_rows[0].iter().copied().collect();
        let r_set: StdHashSet<i32> = r_rows[0].iter().copied().collect();
        let expected: StdHashSet<i32> = g_set.intersection(&r_set).copied().collect();
        assert_eq!(survivors, expected,
            "F survivors must equal G ∩ R as sets");
    }
}
