// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The library fetch, end to end against a served dataset: the
//! acceptance cases of `docs/design/srd-api-discoverability.md` §9.

mod support;

use std::path::Path;
use std::sync::LazyLock;

use vectordata::catalog::Catalog;
use vectordata::fetch::{FetchEvent, FetchRequest, Silent};
use vectordata::merkle::MerkleRef;
use vectordata::{TestDataGroup, TestDataView};

use support::testserver::TestServer;

static TEST_CACHE_DIR: LazyLock<tempfile::TempDir> = LazyLock::new(|| {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).expect("create test cache root")
});

fn init_test_cache() {
    vectordata::settings::override_cache_dir_for_process(TEST_CACHE_DIR.path().to_path_buf());
}

fn make_tmp() -> tempfile::TempDir {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
    std::fs::create_dir_all(&base).unwrap();
    tempfile::tempdir_in(&base).unwrap()
}

/// `records` fvec records of `dim`, values offset by `seed` so that no
/// two tests share content — and therefore never share a cache slot.
fn write_fvec(path: &Path, records: usize, dim: usize, seed: f32) {
    let mut buf = Vec::new();
    for i in 0..records {
        buf.extend_from_slice(&(dim as i32).to_le_bytes());
        for d in 0..dim {
            buf.extend_from_slice(&(seed + (i * dim + d) as f32).to_le_bytes());
        }
    }
    std::fs::write(path, buf).unwrap();
}

/// Publish a `.mref` beside `path`, with chunks small enough that a
/// small fixture spans several of them.
fn write_mref(path: &Path) {
    let content = std::fs::read(path).unwrap();
    let mref = MerkleRef::from_content(&content, 4 * 1024);
    let mut p = path.to_path_buf().into_os_string();
    p.push(".mref");
    mref.save(Path::new(&p)).unwrap();
}

/// A served dataset with three merkle-published facets.
fn served(seed: f32) -> (tempfile::TempDir, TestServer) {
    let tmp = make_tmp();
    for (name, n) in [("base.fvec", 2000), ("query.fvec", 200), ("distances.fvec", 200)] {
        write_fvec(&tmp.path().join(name), n, 8, seed);
        write_mref(&tmp.path().join(name));
    }
    std::fs::write(
        tmp.path().join("dataset.yaml"),
        "name: fetch-ds\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         query_vectors: query.fvec\n    neighbor_distances: distances.fvec\n",
    )
    .unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    init_test_cache();
    (tmp, server)
}

fn resident(view: &dyn TestDataView, facet: &str) -> bool {
    view.open_facet_storage(facet).unwrap().is_complete()
}

/// **A two-facet fetch fetches those two, reports both, and the sink
/// hears both** (§9 case 1).
#[test]
fn fetching_a_subset_fetches_exactly_that_subset() {
    let (_tmp, server) = served(1.0e6);
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let mut progressed: Vec<String> = Vec::new();
    let mut total = None;
    let report = view
        .fetch(
            &FetchRequest::facets(["base_vectors", "query_vectors"]),
            &mut |e: &FetchEvent<'_>| match e {
                FetchEvent::Begin { bytes, .. } => total = Some(*bytes),
                FetchEvent::Progress { facet, .. } if !progressed.contains(&facet.facet) => {
                    progressed.push(facet.facet.clone());
                }
                _ => {}
            },
        )
        .unwrap();

    assert_eq!(report.facets.len(), 2);
    assert!(report.facet("base_vectors").is_some() && report.facet("query_vectors").is_some());
    assert!(resident(&*view, "base_vectors"));
    assert!(resident(&*view, "query_vectors"));
    assert!(!resident(&*view, "neighbor_distances"), "a facet not asked for is not fetched");
    assert_eq!(progressed, ["base_vectors", "query_vectors"], "progress for both, in order");
    for row in &report.facets {
        assert!(row.complete, "{} fetched whole", row.id);
    }
    // Fetched whole, the readers serve zero-copy slices.
    let base = view.base_vectors().unwrap();
    assert_eq!(base.get_slice(7).unwrap(), base.get(7).unwrap().as_slice());
    assert!(
        total.unwrap() >= report.bytes_fetched(),
        "the announced total covers the whole run"
    );
}

/// **The spec form fetches the same thing** (§9 case 2): a URL spec
/// with a selector, resolved by the catalog without any catalog loaded.
#[test]
fn fetching_by_spec_matches_fetching_the_view() {
    let (_tmp, server) = served(2.0e6);
    let spec = format!("{}:default", server.base_url());
    let report = Catalog::default()
        .fetch(&spec, &FetchRequest::facets(["query_vectors"]), &mut Silent)
        .unwrap();
    assert_eq!(report.facets.len(), 1);
    assert_eq!(report.facets[0].id.profile.as_deref(), Some("default"));
    assert_eq!(report.facets[0].id.to_string(), "default/query_vectors");

    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();
    assert!(resident(&*view, "query_vectors"));
    assert!(!resident(&*view, "base_vectors"));
}

/// Standard aliases name the facet they stand for, and a facet the
/// profile does not declare stops the fetch before anything moves.
#[test]
fn facet_names_resolve_aliases_and_unknown_names_are_refused_up_front() {
    let (_tmp, server) = served(3.0e6);
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let err = view
        .fetch(&FetchRequest::facets(["base", "nope"]), &mut Silent)
        .expect_err("an unknown facet fails the plan");
    match err {
        vectordata::Error::UnknownFacets { missing, declared, .. } => {
            assert_eq!(missing, ["nope"]);
            assert!(declared.contains(&"base_vectors".to_string()), "{declared:?}");
        }
        other => panic!("expected UnknownFacets, got {other}"),
    }
    assert!(!resident(&*view, "base_vectors"), "nothing was fetched");

    let report = view.fetch(&FetchRequest::facets(["base"]), &mut Silent).unwrap();
    assert_eq!(report.facets[0].id.facet, "base_vectors", "`base` is base_vectors");
}

/// **A plan reports its cost and moves nothing.** The cost of whole
/// files is their size: a file's last chunk is only as long as what is
/// left of the file, and is not charged as a whole chunk.
#[test]
fn planning_fetches_nothing() {
    let (_tmp, server) = served(4.0e6);
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();
    let plan = view.plan_fetch(&FetchRequest::all(), &mut Silent).unwrap();
    assert_eq!(plan.facets().len(), 3);
    // 2000 + 200 + 200 records of 8 f32s; none a multiple of the 4 KiB chunk.
    assert_eq!(plan.bytes_to_fetch(), (2000 + 200 + 200) * 36);
    assert!(!plan.is_resident());
    for f in ["base_vectors", "query_vectors", "neighbor_distances"] {
        assert!(!resident(&*view, f), "{f} was fetched by planning");
    }
    drop(plan);
    assert!(view.plan_fetch(&FetchRequest::all(), &mut Silent).unwrap().bytes_to_fetch() > 0);

    // Fetching the base file's short last chunk takes exactly its length
    // off what the rest of the file costs.
    let window = |w: &str| FetchRequest::facets(["base_vectors"]).window(vectordata::dataset::source::parse_window(w).unwrap());
    let cost = || view.plan_fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent).unwrap().bytes_to_fetch();
    view.fetch(&window("[0..1]"), &mut Silent).unwrap();
    let before = cost();
    view.fetch(&window("[1999..2000]"), &mut Silent).unwrap();
    assert_eq!(before - cost(), 72000 % 4096, "the last chunk, charged at its length");
}

/// Events arrive in the documented order: a plan pair per facet, then
/// one `Begin`, a begin/end pair per facet, and `End`.
#[test]
fn events_arrive_in_the_documented_order() {
    let (_tmp, server) = served(5.0e6);
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();
    let mut seen: Vec<String> = Vec::new();
    view.fetch(
        &FetchRequest::facets(["query_vectors", "neighbor_distances"]),
        &mut |e: &FetchEvent<'_>| {
            let tag = match e {
                FetchEvent::PlanBegin { facet, .. } => format!("plan-begin {facet}"),
                FetchEvent::PlanEnd { facet, .. } => format!("plan-end {facet}"),
                FetchEvent::Begin { facets, .. } => format!("begin {facets}"),
                FetchEvent::FacetBegin { facet, index, count, .. } => {
                    format!("facet-begin {index}/{count} {facet}")
                }
                FetchEvent::FacetEnd { facet, .. } => format!("facet-end {facet}"),
                FetchEvent::End { .. } => "end".to_string(),
                _ => return,
            };
            seen.push(tag);
        },
    )
    .unwrap();
    assert_eq!(
        seen,
        [
            "plan-begin query_vectors",
            "plan-end query_vectors",
            "plan-begin neighbor_distances",
            "plan-end neighbor_distances",
            "begin 2",
            "facet-begin 1/2 query_vectors",
            "facet-end query_vectors",
            "facet-begin 2/2 neighbor_distances",
            "facet-end neighbor_distances",
            "end",
        ]
    );
}

/// **A window that would silently become a whole-facet download is
/// refused for the whole run, before any of it is fetched**, and allowed
/// when the request says so.
#[test]
fn an_unresolvable_window_refuses_the_whole_run_up_front() {
    let tmp = make_tmp();
    write_fvec(&tmp.path().join("base.fvec"), 500, 8, 6.0e6);
    write_mref(&tmp.path().join("base.fvec"));
    // A variable-length facet with no published index: its records
    // cannot be mapped to bytes without the whole file.
    let mut buf = Vec::new();
    for i in 0..300i32 {
        let dim = i % 4 + 1;
        buf.extend_from_slice(&dim.to_le_bytes());
        for d in 0..dim {
            buf.extend_from_slice(&(6_000_000 + i * 10 + d).to_le_bytes());
        }
    }
    std::fs::write(tmp.path().join("results.ivvec"), buf).unwrap();
    write_mref(&tmp.path().join("results.ivvec"));
    std::fs::write(
        tmp.path().join("dataset.yaml"),
        "name: unmappable\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         metadata_results: results.ivvec\n",
    )
    .unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    init_test_cache();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    // Records 400..410 sit past the first chunk, which planning reads
    // for the header, so whether they were fetched is observable.
    let window = vectordata::dataset::source::parse_window("400..410").unwrap();
    let request = FetchRequest::facets(["base_vectors", "metadata_results"]).window(window);
    let err = view.fetch(&request, &mut Silent).expect_err("refused without consent");
    match &err {
        vectordata::Error::WindowUnresolvable { facets, bytes } => {
            assert_eq!(facets, &["metadata_results"]);
            assert!(*bytes > 0);
        }
        other => panic!("expected WindowUnresolvable, got {other}"),
    }
    let record = 4 + 8 * 4;
    let fill = view
        .open_facet_storage("base_vectors")
        .unwrap()
        .range_fill(400 * record, 410 * record)
        .unwrap();
    assert_eq!(fill.chunks_resident, 0, "the mappable facet's window was not fetched either");

    view.fetch(&request.allow_whole_facet(), &mut Silent).unwrap();
    assert!(resident(&*view, "metadata_results"), "allowed, the facet comes down whole");
}

/// **An unknown dataset comes back as a value carrying its
/// suggestions** (§9 case 10); printing them is the caller's choice.
#[test]
fn an_unknown_dataset_is_an_error_value_with_suggestions() {
    let tmp = make_tmp();
    std::fs::write(
        tmp.path().join("catalog.json"),
        r#"[{"name":"glove-100","path":"glove-100/dataset.yaml","dataset_type":"dataset.yaml","layout":{"profiles":{"default":{}}}},
            {"name":"glove-200","path":"glove-200/dataset.yaml","dataset_type":"dataset.yaml","layout":{"profiles":{"default":{}}}}]"#,
    )
    .unwrap();
    let sources = vectordata::catalog::CatalogSources::new()
        .add_catalogs(&[tmp.path().to_str().unwrap().to_string()]);
    let catalog = Catalog::of(&sources);
    assert!(catalog.diagnostics().is_empty(), "{:?}", catalog.diagnostics());

    match catalog.lookup("glove") {
        Err(vectordata::Error::UnknownDataset { name, suggestions }) => {
            assert_eq!(name, "glove");
            assert_eq!(suggestions, ["glove-100", "glove-200"]);
        }
        other => panic!("expected UnknownDataset, got {other:?}"),
    }
    assert!(catalog.lookup("GLOVE-100").is_ok(), "names match case-insensitively");
    let err = catalog.open_spec("glove-300:default").map(|_| ()).unwrap_err();
    assert!(matches!(err, vectordata::Error::UnknownDataset { .. }), "{err}");
}

/// A catalog location that cannot be loaded is a diagnostic value, not
/// a line on stderr.
#[test]
fn an_unloadable_catalog_is_a_diagnostic() {
    let tmp = make_tmp();
    let missing = tmp.path().join("no-such-catalog.json");
    let sources = vectordata::catalog::CatalogSources::new()
        .add_catalogs(&[missing.to_str().unwrap().to_string()]);
    let catalog = Catalog::of(&sources);
    assert!(catalog.is_empty());
    let diag = catalog.diagnostics();
    assert_eq!(diag.len(), 1, "{diag:?}");
    assert_eq!(diag[0].severity, vectordata::catalog::Severity::Error);
    assert!(diag[0].to_string().starts_with("error: could not load catalog"), "{}", diag[0]);
}

/// **A fetched series reads through one reader, zero-copy, across its
/// shards** — the access path a caller uses instead of opening the
/// shard files itself.
#[test]
fn a_fetched_series_is_read_zero_copy_through_its_reader() {
    let tmp = make_tmp();
    for s in 0..2 {
        write_fvec(&tmp.path().join(format!("base__{s:04}.fvec")), 100, 4, 7.0e6 + s as f32);
    }
    std::fs::write(
        tmp.path().join("dataset.yaml"),
        "format_version: 2\nname: series\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 2\n      \
         record_count: 200\n",
    )
    .unwrap();
    let group = TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let view = group.profile("default").unwrap();
    let report = view.fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent).unwrap();
    assert!(report.facet("base_vectors").unwrap().complete);
    let base = view.base_vectors().unwrap();
    assert_eq!(base.count(), 200);
    for i in [0usize, 99, 100, 199] {
        let slice = base.get_slice(i).expect("zero-copy across shards once fetched");
        assert_eq!(slice, base.get(i).unwrap().as_slice(), "record {i}");
    }
}

/// After a windowed fetch only the window is local, and the report
/// says so.
#[test]
fn a_windowed_fetch_reports_an_incomplete_file() {
    let (_tmp, server) = served(8.0e6);
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();
    let window = vectordata::dataset::source::parse_window("0..10").unwrap();
    let report = view
        .fetch(&FetchRequest::facets(["base_vectors"]).window(window), &mut Silent)
        .unwrap();
    let row = &report.facets[0];
    assert!(!row.complete, "only the window was fetched");
}

// ─── Bulk reads ────────────────────────────────────────────────────

/// The value `write_fvec` puts at element `d` of record `i`.
fn expected(seed: f32, dim: usize, i: usize, d: usize) -> f32 {
    seed + (i * dim + d) as f32
}

/// **`read_into` copies records contiguously, across a shard seam, and
/// clamps at the end** — the bulk read that replaces opening the files.
#[test]
fn read_into_fills_a_buffer_across_shards() {
    let tmp = make_tmp();
    // Two shards of 100 records, each numbered on from the last.
    let mut buf = Vec::new();
    for s in 0..2usize {
        buf.clear();
        for i in s * 100..(s + 1) * 100 {
            buf.extend_from_slice(&4i32.to_le_bytes());
            for d in 0..4 {
                buf.extend_from_slice(&expected(9.0e6, 4, i, d).to_le_bytes());
            }
        }
        std::fs::write(tmp.path().join(format!("base__{s:04}.fvec")), &buf).unwrap();
    }
    std::fs::write(
        tmp.path().join("dataset.yaml"),
        "format_version: 2\nname: series\nprofiles:\n  default:\n    base_vectors:\n      \
         source: base__NNNN.fvec\n      shard_stride: 100\n      shard_count: 2\n      \
         record_count: 200\n  window:\n    base_vectors: base__0000.fvec[10..60)\n",
    )
    .unwrap();
    let group = TestDataGroup::load(tmp.path().to_str().unwrap()).unwrap();
    let base = group.profile("default").unwrap().base_vectors().unwrap();

    // 30 records straddling the seam at 100.
    let mut out = vec![0f32; 30 * 4];
    assert_eq!(base.read_into(90, &mut out).unwrap(), 30);
    for k in 0..30 {
        for d in 0..4 {
            assert_eq!(out[k * 4 + d], expected(9.0e6, 4, 90 + k, d), "record {}", 90 + k);
        }
    }
    // Clamped at the end of the facet; nothing at the end; past it is an error.
    let mut out = vec![0f32; 50 * 4];
    assert_eq!(base.read_into(180, &mut out).unwrap(), 20);
    assert_eq!(base.read_into(200, &mut out).unwrap(), 0);
    assert!(base.read_into(201, &mut out).is_err());
    // A buffer that is not whole records is refused.
    assert!(base.read_into(0, &mut [0f32; 6]).is_err());

    // A windowed profile reads in its own ordinals.
    let windowed = group.profile("window").unwrap().base_vectors().unwrap();
    assert_eq!(windowed.count(), 50);
    let mut out = vec![0f32; 2 * 4];
    assert_eq!(windowed.read_into(0, &mut out).unwrap(), 2);
    assert_eq!(out[0], expected(9.0e6, 4, 10, 0), "window record 0 is file record 10");
    assert!(windowed.get_slice(0).is_some(), "a windowed reader passes zero-copy slices through");
}

/// Over HTTP, unfetched, `read_into` reads the span in one go and
/// agrees with per-record reads; after a fetch it copies from the map.
#[test]
fn read_into_matches_per_record_reads_over_http() {
    let (_tmp, server) = served(1.1e7);
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();
    let base = view.base_vectors().unwrap();
    let mut out = vec![0f32; 300 * 8];
    assert_eq!(base.read_into(500, &mut out).unwrap(), 300);
    for k in [0usize, 1, 150, 299] {
        assert_eq!(&out[k * 8..(k + 1) * 8], base.get(500 + k).unwrap().as_slice(), "record {}", 500 + k);
    }
    view.fetch(&FetchRequest::facets(["base_vectors"]), &mut Silent).unwrap();
    let mut again = vec![0f32; 300 * 8];
    assert_eq!(base.read_into(500, &mut again).unwrap(), 300);
    assert_eq!(out, again);
}

// ─── One cache entry per facet ─────────────────────────────────────

/// Write the catalog-served dataset the cross-process test uses.
fn anchored_dataset(root: &Path) {
    let ds = root.join("anchored");
    std::fs::create_dir_all(&ds).unwrap();
    write_fvec(&ds.join("base.fvec"), 2000, 8, 1.2e7);
    write_mref(&ds.join("base.fvec"));
    let mut results = Vec::new();
    for i in 0..300i32 {
        results.extend_from_slice(&1i32.to_le_bytes());
        results.extend_from_slice(&(12_000_000 + i).to_le_bytes());
    }
    std::fs::write(ds.join("results.ivvec"), &results).unwrap();
    write_mref(&ds.join("results.ivvec"));
    let labels: Vec<u8> = (0..2000u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(ds.join("labels.u8"), &labels).unwrap();
    write_mref(&ds.join("labels.u8"));
    std::fs::write(
        ds.join("dataset.yaml"),
        "name: anchored\nprofiles:\n  default:\n    base_vectors: base.fvec\n    \
         metadata_results: results.ivvec\n    metadata_content: labels.u8\n",
    )
    .unwrap();
    std::fs::write(
        root.join("catalog.json"),
        r#"[{"name":"anchored","path":"anchored/dataset.yaml","dataset_type":"dataset.yaml","layout":{"profiles":{"default":{}}}}]"#,
    )
    .unwrap();
}

const CHILD_CATALOG: &str = "VECTORDATA_TEST_CHILD_CATALOG";
const CHILD_CACHE: &str = "VECTORDATA_TEST_CHILD_CACHE";

/// **A fetched facet is the facet its readers read — in the next
/// process too.** `fetch` fills the dataset-anchored cache entry
/// (`<cache>/<dataset>/<file>`). The readers used to open the resolved
/// URL instead. Within one process that went unnoticed, because both
/// opens share a registry keyed by URL; in the run after a `precache`,
/// the URL open keyed a *second* entry by URL authority, downloaded
/// every byte again, and served every read as a range request against
/// it — while `is_complete()` and the chunk bitmap described the first.
///
/// So the fetch runs here and the reads run in a fresh process (this
/// test binary, re-run as the child below), against the same cache.
/// Each accessor shape is checked: uniform vectors, variable-length
/// records, and typed scalars.
#[test]
fn readers_in_a_later_process_read_the_entry_fetch_filled() {
    let tmp = make_tmp();
    anchored_dataset(tmp.path());
    let server = TestServer::start(tmp.path()).unwrap();
    // The override is process-wide, so this test uses the cache every
    // test in this file shares, and tells the child where it is.
    init_test_cache();
    let cache = TEST_CACHE_DIR.path();
    let catalog = Catalog::of(&vectordata::catalog::CatalogSources::new().add_catalogs(&[server.base_url()]));
    assert!(catalog.diagnostics().is_empty(), "{:?}", catalog.diagnostics());
    catalog
        .open_profile("anchored", "default")
        .unwrap()
        .fetch(&FetchRequest::all(), &mut Silent)
        .unwrap();

    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["read_the_fetched_facets_in_a_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_CATALOG, server.base_url())
        .env(CHILD_CACHE, cache)
        .output()
        .expect("run the reading process");
    assert!(
        child.status.success(),
        "the reading process failed:\n{}\n{}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );

    // No URL-keyed copy of this server's files beside the anchored one.
    let listing = vectordata::cache_admin::list_entries(cache).unwrap();
    let port = server.port().to_string();
    let doubled: Vec<_> = listing
        .url_derived
        .iter()
        .filter(|e| e.path.to_string_lossy().contains(&port))
        .collect();
    assert!(doubled.is_empty(), "a URL-keyed duplicate cache entry: {doubled:?}");
}

/// The reading half of the test above. Does nothing unless that test
/// launched it.
#[test]
fn read_the_fetched_facets_in_a_child() {
    let (Ok(catalog_url), Ok(cache)) = (std::env::var(CHILD_CATALOG), std::env::var(CHILD_CACHE)) else {
        return;
    };
    vectordata::settings::override_cache_dir_for_process(cache.into());
    let catalog = Catalog::of(&vectordata::catalog::CatalogSources::new().add_catalogs(&[catalog_url]));
    let view = catalog.open_profile("anchored", "default").unwrap();

    let base = view.base_vectors().unwrap();
    assert!(base.is_complete(), "the vector reader sees the fetched bytes");
    assert!(base.get_slice(1999).is_some(), "and borrows them from the mapping");
    let results = view.metadata_results().unwrap();
    assert!(results.is_complete(), "the variable-length reader sees them too");
    assert_eq!(results.get(299).unwrap(), vec![12_000_299]);
    let typed: vectordata::TypedReader<u8> =
        vectordata::open_facet_typed(&*view, "metadata_content").unwrap();
    assert!(typed.is_complete(), "and so does the typed reader");
    assert_eq!(typed.get_native(300).unwrap(), (300 % 251) as u8);
}

/// Bytes free on the filesystem holding `path`, as the cache check
/// measures them.
#[cfg(unix)]
fn free_bytes(path: &Path) -> u64 {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::statvfs(c.as_ptr(), &mut stat) }, 0);
    stat.f_bavail as u64 * stat.f_frsize as u64
}

/// **A fetch the cache cannot hold is refused before anything moves**
/// (§9 case 6). The served base file is sparse and larger than the free
/// space under the cache; the run, which also asks for a small facet, is
/// refused as a whole with the three numbers a user acts on, the sink
/// hears only the failure, and neither facet lands in the cache.
#[cfg(unix)]
#[test]
fn a_fetch_larger_than_the_cache_is_refused_before_anything_moves() {
    let tmp = make_tmp();
    init_test_cache();
    let free = free_bytes(TEST_CACHE_DIR.path());
    // One-element records: any multiple of 8 bytes is a whole file.
    let size = (free + (64 << 30)) / 8 * 8;
    let huge = tmp.path().join("huge.fvec");
    std::fs::write(&huge, 1i32.to_le_bytes()).unwrap();
    let f = std::fs::OpenOptions::new().write(true).open(&huge).unwrap();
    if let Err(e) = f.set_len(size) {
        // A filesystem whose largest file is smaller than its free space
        // cannot stage this case.
        eprintln!("skipping: cannot make a {size}-byte sparse file here: {e}");
        return;
    }
    drop(f);
    write_fvec(&tmp.path().join("small.fvec"), 100, 8, 9.0e6);
    std::fs::write(
        tmp.path().join("dataset.yaml"),
        "name: too-big\nprofiles:\n  default:\n    base_vectors: huge.fvec\n    query_vectors: small.fvec\n",
    )
    .unwrap();
    let server = TestServer::start(tmp.path()).unwrap();
    let group = TestDataGroup::load(&server.base_url()).unwrap();
    let view = group.profile("default").unwrap();

    let mut events = Vec::new();
    let err = view
        .fetch(&FetchRequest::facets(["query_vectors", "base_vectors"]), &mut |e: &FetchEvent<'_>| {
            events.push(format!("{e:?}"))
        })
        .expect_err("the cache cannot hold it");
    match &err {
        vectordata::Error::InsufficientCacheSpace(s) => {
            assert!(s.needed >= size - (1 << 20), "needed {} of {size}", s.needed);
            assert!(s.available < s.needed);
            assert!(s.cache_dir.starts_with(TEST_CACHE_DIR.path()) || TEST_CACHE_DIR.path().starts_with(&s.cache_dir));
        }
        other => panic!("expected InsufficientCacheSpace, got {other:?}"),
    }
    assert!(err.to_string().contains("not enough free space in cache directory"), "{err}");
    // Planning is heard; nothing that moves bytes is, and the failure ends it.
    assert!(events.iter().all(|e| e.starts_with("Plan") || e.starts_with("Failed")), "{events:?}");
    assert!(events.last().unwrap().starts_with("Failed"), "{events:?}");
    assert!(!resident(&*view, "query_vectors"), "the small facet was not fetched either");
}

/// **A selection that matches several profiles opens one view only when
/// asked for one.** `view` names the matches and asks to narrow; `views`
/// returns them all; a selector naming one profile gives its view.
#[test]
fn a_selection_of_several_profiles_refuses_a_single_view() {
    let tmp = make_tmp();
    write_fvec(&tmp.path().join("base.fvec"), 100, 4, 8.0e6);
    write_fvec(&tmp.path().join("small.fvec"), 10, 4, 8.5e6);
    std::fs::write(
        tmp.path().join("dataset.yaml"),
        "name: multi\nprofiles:\n  default:\n    base_vectors: base.fvec\n  \
         small:\n    base_vectors: small.fvec\n",
    )
    .unwrap();
    let catalog = Catalog::of(&vectordata::catalog::CatalogSources::new());
    let spec = tmp.path().to_str().unwrap();

    let all = catalog.open_spec(&format!("{spec}:profile=*")).unwrap();
    assert_eq!(all.profiles().len(), 2, "{:?}", all.profiles());
    let err = all.view().map(|_| ()).unwrap_err();
    match &err {
        vectordata::Error::Selection { message, .. } => {
            assert!(message.contains("matches 2 profiles"), "{message}");
            assert!(message.contains("default") && message.contains("small"), "{message}");
        }
        other => panic!("expected a Selection error, got {other:?}"),
    }
    let views = all.views().unwrap();
    let names: Vec<&str> = views.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"default") && names.contains(&"small"), "{names:?}");

    let one = catalog.open_spec(&format!("{spec}:small")).unwrap();
    assert_eq!(one.view().unwrap().base_vectors().unwrap().count(), 10);
}

/// **Problems met reading the sources reach the catalog's diagnostics**,
/// alongside the ones met loading: a configured directory with no
/// catalog file in it, then a catalog that cannot be loaded.
#[test]
fn source_problems_are_carried_into_the_catalog() {
    let tmp = make_tmp();
    let bare = tmp.path().join("bare");
    std::fs::create_dir_all(&bare).unwrap();
    let missing = tmp.path().join("missing.json");
    let sources = vectordata::catalog::CatalogSources::new().add_catalogs(&[
        bare.to_str().unwrap().to_string(),
        missing.to_str().unwrap().to_string(),
    ]);
    assert_eq!(sources.diagnostics().len(), 1);
    let catalog = Catalog::of(&sources);
    let diag: Vec<String> = catalog.diagnostics().iter().map(|d| d.to_string()).collect();
    assert_eq!(diag.len(), 2, "{diag:?}");
    assert!(diag[0].starts_with("warning: directory") && diag[0].contains("has no catalogs.yaml"), "{diag:?}");
    assert!(diag[1].starts_with("error: could not load catalog"), "{diag:?}");
}
