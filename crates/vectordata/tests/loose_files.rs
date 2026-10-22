// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A directory of files vectordata did not produce, opened as the
//! dataset its files imply (`dataset::infer`, LF-1 to LF-12): one test
//! per worked case of `docs/design/srd-loose-files-overlays-integrity.md`
//! §3.2, and `datasets init`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use vectordata::dataset::StandardFacet;
use vectordata::dataset::infer::{InferError, Rule, infer_dir};
use vectordata::{DistanceFunction, TestDataGroup, TestDataViewExt};

fn fvecs(path: &Path, rows: &[Vec<f32>]) {
    let mut f = std::fs::File::create(path).unwrap();
    for r in rows {
        f.write_all(&(r.len() as i32).to_le_bytes()).unwrap();
        for v in r {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}

fn ivecs(path: &Path, rows: &[Vec<i32>]) {
    let mut f = std::fs::File::create(path).unwrap();
    for r in rows {
        f.write_all(&(r.len() as i32).to_le_bytes()).unwrap();
        for v in r {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}

/// `n` unit vectors of dimension 2.
fn unit(n: usize) -> Vec<Vec<f32>> {
    (0..n).map(|i| {
        let a = i as f32 * 0.1;
        vec![a.cos(), a.sin()]
    }).collect()
}

/// A ground truth of `q` rows of `k`, every index below `base`.
fn truth(q: usize, k: usize, base: usize) -> Vec<Vec<i32>> {
    (0..q).map(|r| (0..k).map(|c| ((r + c) % base) as i32).collect()).collect()
}

/// A directory named `name` under a fresh root.
fn dir(name: &str) -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let d = root.path().join(name);
    std::fs::create_dir(&d).unwrap();
    (root, d)
}

fn sources(inferred: &vectordata::dataset::infer::Inferred) -> Vec<(StandardFacet, String, Rule)> {
    inferred.classified.iter().map(|c| (c.facet, c.source.clone(), c.rule.clone())).collect()
}

/// **Canonical names (LF-4) read as the five-line dataset.yaml would.**
#[test]
fn canonical_names_open_as_their_facets() {
    let (_root, d) = dir("plain");
    fvecs(&d.join("base_vectors.fvec"), &unit(10));
    fvecs(&d.join("query_vectors.fvec"), &unit(3));
    ivecs(&d.join("neighbor_indices.ivec"), &truth(3, 4, 10));
    let group = TestDataGroup::load(d.to_str().unwrap()).unwrap();
    assert_eq!(group.name(), Some("plain"));
    assert_eq!(
        group.inferred(),
        Some("base_vectors=base_vectors.fvec, query_vectors=query_vectors.fvec, neighbor_indices=neighbor_indices.ivec")
    );
    let view = group.profile("default").unwrap();
    assert_eq!(view.base_vectors_as::<f32>().unwrap().count(), 10);
    assert_eq!(view.neighbor_indices_as::<i32>().unwrap().dim(), 4);
    assert_eq!(view.base_count(), Some(10));
    let inferred = infer_dir(&d).unwrap();
    assert!(sources(&inferred).iter().all(|(_, _, r)| *r == Rule::CanonicalName));
    assert!(inferred.yaml.contains("maxk: 4"));
}

/// **jvector names: one role token each (LF-5)**, and `ip` declares
/// the inner product (LF-12).
#[test]
fn role_tokens_classify_jvector_names_and_ip_declares_the_metric() {
    let (_root, d) = dir("ada_002_100k");
    fvecs(&d.join("ada_002_100k_base_99287.fvecs"), &unit(10));
    fvecs(&d.join("ada_002_100k_query_10000.fvecs"), &unit(3));
    ivecs(&d.join("ada_002_100k_gt_ip_100.ivecs"), &truth(3, 4, 10));
    let inferred = infer_dir(&d).unwrap();
    assert_eq!(
        sources(&inferred),
        [
            (StandardFacet::BaseVectors, "ada_002_100k_base_99287.fvecs".into(), Rule::RoleToken("base".into())),
            (StandardFacet::QueryVectors, "ada_002_100k_query_10000.fvecs".into(), Rule::RoleToken("query".into())),
            (StandardFacet::NeighborIndices, "ada_002_100k_gt_ip_100.ivecs".into(), Rule::RoleToken("gt".into())),
        ]
    );
    let view = TestDataGroup::load(d.to_str().unwrap()).unwrap().profile("default").unwrap();
    assert_eq!(view.distance(), Some(DistanceFunction::DotProduct));
}

/// **Shapes finish what names could not (LF-6)**, and are verified
/// (LF-9).
#[test]
fn shapes_classify_files_whose_names_say_nothing() {
    let (_root, d) = dir("shapes");
    fvecs(&d.join("vectors.fvecs"), &unit(10));
    fvecs(&d.join("q.fvecs"), &unit(3));
    ivecs(&d.join("truth.ivecs"), &truth(3, 4, 10));
    let inferred = infer_dir(&d).unwrap();
    assert_eq!(
        sources(&inferred),
        [
            (StandardFacet::BaseVectors, "vectors.fvecs".into(), Rule::Shape),
            (StandardFacet::QueryVectors, "q.fvecs".into(), Rule::Shape),
            (StandardFacet::NeighborIndices, "truth.ivecs".into(), Rule::Shape),
        ]
    );
}

/// **Shapes that do not settle the roles are refused**, listing every
/// file with its shape.
#[test]
fn shapes_that_tie_are_refused_listing_each_file() {
    let (_root, d) = dir("tie");
    fvecs(&d.join("a.fvecs"), &unit(10));
    fvecs(&d.join("b.fvecs"), &unit(10));
    fvecs(&d.join("q.fvecs"), &unit(3));
    ivecs(&d.join("truth.ivecs"), &truth(3, 4, 10));
    let err = infer_dir(&d).unwrap_err();
    assert!(matches!(err, InferError::Shapes { .. }), "{err}");
    let text = err.to_string();
    assert!(text.contains("two float files tie for the most records"), "{text}");
    assert!(text.contains("a.fvecs (10 × 2)") && text.contains("b.fvecs (10 × 2)"), "{text}");
}

/// **Two files for one role are refused naming both (LF-7)** — never
/// settled by order.
#[test]
fn two_files_for_one_role_are_refused_naming_both() {
    let (_root, d) = dir("two");
    fvecs(&d.join("a_base.fvecs"), &unit(10));
    fvecs(&d.join("b_base.fvecs"), &unit(10));
    fvecs(&d.join("q_query.fvecs"), &unit(3));
    ivecs(&d.join("gt.ivecs"), &truth(3, 4, 10));
    let err = infer_dir(&d).unwrap_err().to_string();
    assert!(err.contains("2 files could be base_vectors"), "{err}");
    assert!(err.contains("a_base.fvecs (by the role token 'base' (LF-5))") && err.contains("b_base.fvecs"), "{err}");
}

/// **A truth computed against another base is refused (LF-9)**, at the
/// row and column of the first index the base does not hold.
#[test]
fn a_truth_against_a_larger_base_is_refused_with_its_row_and_column() {
    let (_root, d) = dir("stale");
    fvecs(&d.join("base.fvecs"), &unit(10));
    fvecs(&d.join("query.fvecs"), &unit(3));
    ivecs(&d.join("gt.ivecs"), &[vec![0, 1], vec![2, 15], vec![3, 4]]);
    let err = infer_dir(&d).unwrap_err();
    assert!(matches!(err, InferError::Verification(_)), "{err}");
    assert!(err.to_string().contains("row 1 column 1 holds 15, which names no base vector of 10 base vectors"), "{err}");
    // The open says so too, rather than reading on.
    let open = TestDataGroup::load(d.to_str().unwrap()).unwrap_err().to_string();
    assert!(open.contains("its files make no dataset") && open.contains("row 1 column 1"), "{open}");
}

/// **Names that disagree on the metric declare none**, and say so.
#[test]
fn metric_tokens_that_disagree_declare_none() {
    let (_root, d) = dir("glove_25_angular");
    fvecs(&d.join("glove_25_angular_base.fvecs"), &unit(10));
    fvecs(&d.join("glove_25_angular_query.fvecs"), &unit(3));
    ivecs(&d.join("glove_25_angular_gt_ip_100.ivecs"), &truth(3, 4, 10));
    let inferred = infer_dir(&d).unwrap();
    assert_eq!(inferred.config.attributes.as_ref().and_then(|a| a.distance()), None);
    assert!(inferred.notes.iter().any(|n| n.contains("disagree on the metric")), "{:?}", inferred.notes);
}

/// **A veks layout without its descriptor (LF-10)**: each profile by
/// canonical names, `profiles/base/` lending what a profile lacks.
#[test]
fn a_veks_layout_without_its_descriptor_reads_by_profile() {
    let (_root, d) = dir("veks");
    std::fs::create_dir_all(d.join("profiles/base")).unwrap();
    std::fs::create_dir_all(d.join("profiles/default")).unwrap();
    fvecs(&d.join("profiles/base/base_vectors.fvec"), &unit(10));
    fvecs(&d.join("profiles/default/query_vectors.fvec"), &unit(3));
    ivecs(&d.join("profiles/default/neighbor_indices.ivec"), &truth(3, 4, 10));
    let inferred = infer_dir(&d).unwrap();
    assert!(
        sources(&inferred).contains(&(StandardFacet::BaseVectors, "profiles/base/base_vectors.fvec".into(), Rule::CanonicalName)),
        "{:?}",
        sources(&inferred)
    );
    let view = TestDataGroup::load(d.to_str().unwrap()).unwrap().profile("default").unwrap();
    assert_eq!(view.query_vectors_as::<f32>().unwrap().count(), 3);
}

/// **A role token in a format its facet cannot take is not used**:
/// `x_gt.fvecs` is not a ground truth (its indices are integers), the
/// note says why, and nothing else can be the truth.
#[test]
fn a_role_token_in_the_wrong_format_is_not_used() {
    let (_root, d) = dir("wrong");
    fvecs(&d.join("base.fvecs"), &unit(10));
    fvecs(&d.join("query.fvecs"), &unit(3));
    fvecs(&d.join("x_gt.fvecs"), &unit(3));
    let err = infer_dir(&d).unwrap_err();
    assert!(err.to_string().contains("no integer file has a record per query"), "{err}");

    // Integer queries are queries: the facet spec takes quantized ones.
    ivecs(&d.join("gt.ivecs"), &truth(3, 4, 10));
    std::fs::remove_file(d.join("x_gt.fvecs")).unwrap();
    ivecs(&d.join("codes_query.ivecs"), &[vec![1, 2], vec![3, 4], vec![5, 6]]);
    std::fs::remove_file(d.join("query.fvecs")).unwrap();
    let inferred = infer_dir(&d).unwrap();
    assert!(sources(&inferred).contains(&(StandardFacet::QueryVectors, "codes_query.ivecs".into(), Rule::RoleToken("query".into()))));
}

/// **A series is one candidate**, contiguous from `0000`.
#[test]
fn a_series_is_one_candidate_and_must_be_contiguous() {
    let (_root, d) = dir("series");
    fvecs(&d.join("base_vectors__0000.fvec"), &unit(5));
    fvecs(&d.join("base_vectors__0001.fvec"), &unit(5));
    fvecs(&d.join("query_vectors.fvec"), &unit(3));
    ivecs(&d.join("neighbor_indices.ivec"), &truth(3, 4, 10));
    let view = TestDataGroup::load(d.to_str().unwrap()).unwrap().profile("default").unwrap();
    assert_eq!(view.base_vectors_as::<f32>().unwrap().count(), 10);

    std::fs::rename(d.join("base_vectors__0001.fvec"), d.join("base_vectors__0002.fvec")).unwrap();
    let err = infer_dir(&d).unwrap_err().to_string();
    assert!(err.contains("has no file base_vectors__0001.fvec before base_vectors__0002.fvec"), "{err}");
}

/// **A `norm` token declares a normalized base only when a sample of it
/// is unit length.**
#[test]
fn a_norm_token_is_checked_against_a_sample() {
    let (_root, d) = dir("emb_norm");
    fvecs(&d.join("base.fvecs"), &unit(10));
    fvecs(&d.join("query.fvecs"), &unit(3));
    ivecs(&d.join("gt.ivecs"), &truth(3, 4, 10));
    let inferred = infer_dir(&d).unwrap();
    assert_eq!(inferred.config.attributes.as_ref().and_then(|a| a.is_normalized), Some(true));

    fvecs(&d.join("base.fvecs"), &(0..10).map(|i| vec![i as f32 + 2.0, 0.0]).collect::<Vec<_>>());
    let err = infer_dir(&d).unwrap_err().to_string();
    assert!(err.contains("V8 base_vectors") && err.contains("is named normalized"), "{err}");
}

/// **`datasets init` writes the descriptor, so later opens are
/// declared**, and refuses to replace one without `--force`.
#[test]
fn init_writes_the_descriptor_and_refuses_to_replace_one() {
    let (_root, d) = dir("init_me");
    fvecs(&d.join("base.fvecs"), &unit(10));
    fvecs(&d.join("query.fvecs"), &unit(3));
    ivecs(&d.join("gt.ivecs"), &truth(3, 4, 10));
    let written = vectordata::datasets::init::init(&d, false).unwrap();
    let text = std::fs::read_to_string(&written.path).unwrap();
    assert!(text.contains("base_vectors: base.fvecs") && text.contains("neighbor_indices: gt.ivecs"), "{text}");
    // Declared from now on: the file is what an open reads.
    let view = TestDataGroup::load(d.to_str().unwrap()).unwrap().profile("default").unwrap();
    assert_eq!(view.facet_source("query_vectors").map(|s| s.ends_with("query.fvecs")), Some(true));
    let err = vectordata::datasets::init::init(&d, false).unwrap_err();
    assert!(err.contains("exists; --force replaces it"), "{err}");
    vectordata::datasets::init::init(&d, true).unwrap();
}

fn vectordata_bin(home: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vectordata"))
        .args(args)
        .env("VECTORDATA_HOME", home)
        .env_remove(vectordata::settings::OFFLINE_ENV)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .output()
        .expect("run vectordata")
}

/// **`datasets describe <dir>` on three bare files describes them** —
/// the reported gap ("no catalog file found"), through the binary; and
/// `datasets init` through it writes the file.
#[test]
fn describe_and_init_take_a_directory_of_loose_files() {
    let (root, d) = dir("bare");
    fvecs(&d.join("base.fvecs"), &unit(10));
    fvecs(&d.join("query.fvecs"), &unit(3));
    ivecs(&d.join("gt.ivecs"), &truth(3, 4, 10));
    let home = root.path().join("home");
    std::fs::create_dir_all(home.join("cache")).unwrap();
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n", home.join("cache").display())).unwrap();

    let out = vectordata_bin(&home, &["datasets", "describe", d.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}{}", stdout, String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("base_vectors") && stdout.contains("gt.ivecs"), "{stdout}");
    assert!(stdout.contains("Type:    inferred from its files"), "{stdout}");
    assert!(stdout.contains("Read as: base_vectors=base.fvecs, query_vectors=query.fvecs, neighbor_indices=gt.ivecs"), "{stdout}");

    let out = vectordata_bin(&home, &["datasets", "init", d.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("neighbor_indices = gt.ivecs (by the role token 'gt' (LF-5))"), "{stdout}");
    assert!(d.join("dataset.yaml").is_file());
}

/// **A symlinked file is a candidate like a copy (LF-3)**: staging a
/// dataset by linking its files into a scratch directory is how one is
/// tried without duplicating it, and it infers and opens as the copies
/// would.
#[cfg(unix)]
#[test]
fn symlinked_files_are_candidates() {
    use std::os::unix::fs::symlink;
    let (_store, store) = dir("store");
    fvecs(&store.join("e5_base_99443.fvecs"), &unit(10));
    fvecs(&store.join("e5_query_10000.fvecs"), &unit(3));
    ivecs(&store.join("e5_gt_100.ivecs"), &truth(3, 4, 10));
    let (_root, d) = dir("staged");
    symlink(store.join("e5_base_99443.fvecs"), d.join("mydata_base_100000.fvecs")).unwrap();
    symlink(store.join("e5_query_10000.fvecs"), d.join("mydata_query_10000.fvecs")).unwrap();
    symlink(store.join("e5_gt_100.ivecs"), d.join("mydata_gt_100.ivecs")).unwrap();
    let inferred = infer_dir(&d).unwrap();
    assert_eq!(
        sources(&inferred),
        [
            (StandardFacet::BaseVectors, "mydata_base_100000.fvecs".into(), Rule::RoleToken("base".into())),
            (StandardFacet::QueryVectors, "mydata_query_10000.fvecs".into(), Rule::RoleToken("query".into())),
            (StandardFacet::NeighborIndices, "mydata_gt_100.ivecs".into(), Rule::RoleToken("gt".into())),
        ]
    );
    let view = TestDataGroup::load(d.to_str().unwrap()).unwrap().profile("default").unwrap();
    assert_eq!(view.base_vectors_as::<f32>().unwrap().count(), 10);
}

/// **A symlinked profile directory is a profile (LF-10).**
#[cfg(unix)]
#[test]
fn a_symlinked_profile_directory_is_read() {
    use std::os::unix::fs::symlink;
    let (_store, store) = dir("store");
    std::fs::create_dir_all(store.join("base")).unwrap();
    fvecs(&store.join("base/base_vectors.fvec"), &unit(10));
    let (_root, d) = dir("veks");
    std::fs::create_dir_all(d.join("profiles/default")).unwrap();
    symlink(store.join("base"), d.join("profiles/base")).unwrap();
    fvecs(&d.join("profiles/default/query_vectors.fvec"), &unit(3));
    ivecs(&d.join("profiles/default/neighbor_indices.ivec"), &truth(3, 4, 10));
    let inferred = infer_dir(&d).unwrap();
    assert!(
        sources(&inferred).contains(&(StandardFacet::BaseVectors, "profiles/base/base_vectors.fvec".into(), Rule::CanonicalName)),
        "{:?}",
        sources(&inferred)
    );
}

/// **A link to nothing is refused, naming it**, rather than skipped
/// into a misleading "no file is …".
#[cfg(unix)]
#[test]
fn a_dangling_link_is_refused_naming_it() {
    use std::os::unix::fs::symlink;
    let (_root, d) = dir("broken");
    fvecs(&d.join("x_base.fvecs"), &unit(10));
    fvecs(&d.join("x_query.fvecs"), &unit(3));
    symlink(d.join("gone.ivecs"), d.join("x_gt.ivecs")).unwrap();
    let err = infer_dir(&d).unwrap_err();
    assert!(matches!(err, InferError::Unreadable(_)), "{err:?}");
    assert!(err.to_string().contains("x_gt.ivecs"), "{err}");
}

/// **A directory with no vector files says so**, not "name the files
/// for their roles:" with nothing after it.
#[test]
fn a_directory_without_vector_files_says_so() {
    let (_root, d) = dir("empty-ish");
    std::fs::write(d.join("notes.txt"), "nothing to see").unwrap();
    let err = infer_dir(&d).unwrap_err().to_string();
    assert!(!err.trim_end().ends_with(':'), "{err}");
    assert!(err.contains("no vector files"), "{err}");
}
