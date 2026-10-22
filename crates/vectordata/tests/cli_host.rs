// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A host program runs the `vectordata` command line as its own
//! (`shell::run`, `shell::run_with_catalogs`, `shell::Host`): the exit
//! status comes back instead of ending the process, the host's hooks run
//! for their own command only and after its output, a host's precache
//! specs expand into one run, and a host whose command line this is
//! answers completion under its own name.
//!
//! What a command prints is observed in a child: this test binary
//! re-runs itself as `host_child`, which runs one case through
//! `shell::run` and exits with its status. Every child gets its own
//! `VECTORDATA_HOME`; this process's environment is never changed.

#![cfg(feature = "cli")]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::rc::Rc;

use vectordata::Location;
use vectordata::catalog::{CatalogBuilder, CatalogSources};
use vectordata::dataset::{CatalogEntry, DatasetConfig, StandardFacet};
use vectordata::shell::{Host, run, run_with_catalogs};

/// The case a child runs, set only in the child's environment.
const CHILD_CASE: &str = "VECTORDATA_CLI_HOST_CASE";
/// The scratch directory a child's datasets and config are in.
const CHILD_DIR: &str = "VECTORDATA_CLI_HOST_DIR";

/// Three two-dimensional f32 vectors in `.fvec` form.
fn write_fvec(path: &Path) {
    let mut buf = Vec::new();
    for i in 0..3 {
        buf.extend(2i32.to_le_bytes());
        buf.extend((i as f32).to_le_bytes());
        buf.extend((i as f32 + 0.5).to_le_bytes());
    }
    std::fs::write(path, buf).unwrap();
}

/// Two local datasets, `alpha` and `beta`, under `dir`, an empty config
/// directory beside them, and the program catalog `bench` listing both.
/// Returns the catalogs and the config directory.
fn bench(dir: &Path) -> (CatalogSources, String) {
    let mut builder = CatalogBuilder::named("bench");
    for name in ["alpha", "beta"] {
        let home = dir.join(name);
        std::fs::create_dir_all(&home).unwrap();
        write_fvec(&home.join("b.fvec"));
        let description = DatasetConfig::builder(name)
            .profile("default", |p| p.facet(StandardFacet::BaseVectors, "b.fvec"))
            .build()
            .unwrap();
        builder = builder.entry(
            CatalogEntry::builder(name)
                .home(&Location::from_path(&home))
                .dataset(description)
                .build()
                .unwrap(),
        );
    }
    let config = dir.join("config");
    std::fs::create_dir_all(&config).unwrap();
    (CatalogSources::new().with(builder.build()), config.to_string_lossy().into_owned())
}

fn words(w: &[&str]) -> Vec<String> {
    w.iter().map(|s| s.to_string()).collect()
}

/// What every hook heard, in order, as one line each.
#[derive(Clone, Default)]
struct Heard(Rc<RefCell<Vec<String>>>);

impl Heard {
    fn lines(&self) -> Vec<String> {
        self.0.borrow().clone()
    }
}

/// A host over `catalogs` whose hooks record what they hear in `heard`:
/// `describe <entry> <profile> <catalog size>`, `list <names>`, and
/// `precache <spec>`. `group:pair` expands to both datasets; any other
/// `group:` is an error.
fn recording_host(catalogs: CatalogSources, heard: &Heard) -> Host {
    let (d, l, p) = (heard.clone(), heard.clone(), heard.clone());
    Host::new()
        .catalogs(catalogs)
        .after_describe(move |catalog, entry, profile| {
            d.0.borrow_mut().push(format!("describe {} {profile} {}", entry.name(), catalog.datasets().len()));
        })
        .after_list(move |_catalog, entries| {
            let names: Vec<&str> = entries.iter().map(|e| e.name()).collect();
            l.0.borrow_mut().push(format!("list {}", names.join(",")));
        })
        .before_precache(move |spec| {
            p.0.borrow_mut().push(format!("precache {spec}"));
            match spec.strip_prefix("group:") {
                Some("pair") => Ok(vec!["alpha:default".to_string(), "beta:default".to_string()]),
                Some(other) => Err(format!("no group named '{other}'")),
                None => Ok(vec![spec.to_string()]),
            }
        })
}

/// **The status comes back; the process goes on.** A usage error is 2,
/// a command that works is 0, and one that fails is 1, each returned to
/// this test rather than ending it, which the old dispatcher did on
/// every failure.
#[test]
fn run_with_catalogs_returns_the_status_instead_of_exiting() {
    let tmp = tempfile::tempdir().unwrap();
    let (catalogs, config) = bench(tmp.path());

    let usage = run_with_catalogs("bench", words(&["datasets", "describe", "--no-such-flag"]), catalogs.clone());
    assert_eq!(usage, 2, "a malformed command line");
    let usage = run_with_catalogs("bench", words(&["no-such-command"]), catalogs.clone());
    assert_eq!(usage, 2, "an unknown command");

    let ok = run_with_catalogs("bench", words(&["datasets", "describe", "alpha", "--configdir", &config]), catalogs.clone());
    assert_eq!(ok, 0, "a dataset the program registered describes");

    let failed = run_with_catalogs("bench", words(&["datasets", "describe", "gamma", "--configdir", &config]), catalogs.clone());
    assert_eq!(failed, 1, "a dataset nobody registered");
    let failed = run_with_catalogs(
        "bench",
        words(&["datasets", "list", "--with-min-count", "lots", "--configdir", &config]),
        catalogs,
    );
    assert_eq!(failed, 1, "a malformed size filter, which exited the process before");
}

/// **`after_describe` hears the dataset and profile described, and only
/// for describe that succeeded**: not for a failed describe, and no
/// other hook for describe.
#[test]
fn after_describe_hears_what_was_described() {
    let tmp = tempfile::tempdir().unwrap();
    let (catalogs, config) = bench(tmp.path());
    let heard = Heard::default();
    let host = recording_host(catalogs, &heard);

    assert_eq!(run("bench", words(&["datasets", "describe", "beta", "--configdir", &config]), &host), 0);
    assert_eq!(heard.lines(), ["describe beta default 2"]);

    assert_eq!(run("bench", words(&["datasets", "describe", "gamma", "--configdir", &config]), &host), 1);
    assert_eq!(heard.lines(), ["describe beta default 2"], "a failed describe calls nothing");

    // A path opens directly, with no catalog to have looked it up in.
    let path = tmp.path().join("ds");
    std::fs::create_dir_all(&path).unwrap();
    write_fvec(&path.join("b.fvec"));
    std::fs::write(path.join("dataset.yaml"), "name: ds\nprofiles:\n  default:\n    base_vectors: b.fvec\n").unwrap();
    let spec = path.to_string_lossy().into_owned();
    assert_eq!(run("bench", words(&["datasets", "describe", &spec]), &host), 0);
    assert_eq!(heard.lines()[1], "describe ds default 0");
}

/// **`after_list` hears exactly the entries listed**: all of them, the
/// ones a filter kept, and nothing for `--select` or for any other
/// command.
#[test]
fn after_list_hears_the_entries_listed() {
    let tmp = tempfile::tempdir().unwrap();
    let (catalogs, config) = bench(tmp.path());
    let heard = Heard::default();
    let host = recording_host(catalogs, &heard);

    assert_eq!(run("bench", words(&["datasets", "list", "--configdir", &config]), &host), 0);
    assert_eq!(run("bench", words(&["datasets", "list", "--dataset", "alpha", "--configdir", &config]), &host), 0);
    assert_eq!(run("bench", words(&["datasets", "list", "-f", "json", "--configdir", &config]), &host), 0);
    assert_eq!(heard.lines(), ["list alpha,beta", "list alpha", "list alpha,beta"]);

    assert_eq!(run("bench", words(&["datasets", "list", "--select", "beta:default", "--configdir", &config]), &host), 0);
    assert_eq!(run("bench", words(&["datasets", "list", "--dataset", "gamma", "--configdir", &config]), &host), 1);
    assert_eq!(run("bench", words(&["datasets", "ping", "--at", "bench", "alpha:default", "--configdir", &config]), &host), 0);
    assert_eq!(heard.lines().len(), 3, "no listing hook for --select, a failed list, or ping: {:?}", heard.lines());
}

/// **`before_precache` expands each spec before anything resolves, and
/// its error is the command's**: `group:pair` becomes two specs that
/// both run, an unknown group fails with 2 having resolved nothing, and
/// no other command calls it.
#[test]
fn before_precache_expands_specs_and_its_error_fails_the_command() {
    let tmp = tempfile::tempdir().unwrap();
    let (catalogs, config) = bench(tmp.path());
    let heard = Heard::default();
    let host = recording_host(catalogs, &heard);

    assert_eq!(run("bench", words(&["datasets", "precache", "group:pair", "--configdir", &config]), &host), 0);
    assert_eq!(heard.lines(), ["precache group:pair"]);

    // The error stops the run before the next spec is even expanded.
    assert_eq!(
        run("bench", words(&["datasets", "precache", "group:none", "alpha:default", "--configdir", &config]), &host),
        2
    );
    assert_eq!(heard.lines(), ["precache group:pair", "precache group:none"]);

    // Each expanded spec is resolved: one that names nothing fails.
    assert_eq!(run("bench", words(&["datasets", "precache", "gamma:default", "--configdir", &config]), &host), 1);
    assert_eq!(heard.lines().len(), 3);

    assert_eq!(run("bench", words(&["datasets", "describe", "alpha", "--configdir", &config]), &host), 0);
    assert!(!heard.lines().iter().skip(3).any(|l| l.starts_with("precache")), "{:?}", heard.lines());
}

/// Run this test binary as a child running `case` over a fresh scratch
/// directory, with `env` set in the child alone.
fn child(case: &str, env: &[(&str, &str)]) -> (Output, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(home.join("settings.yaml"), format!("cache_dir: {}\n", home.join("cache").display())).unwrap();
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", "host_child", "--nocapture", "--test-threads=1"])
        .env(CHILD_CASE, case)
        .env(CHILD_DIR, tmp.path())
        .env("VECTORDATA_HOME", &home)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .env_remove(vectordata::settings::OFFLINE_ENV)
        .env_remove("_VECTORDATA_COMPLETE")
        .env_remove("_FVEC_BENCH_COMPLETE");
    for (k, v) in env {
        cmd.env(k, v);
    }
    (cmd.output().unwrap(), tmp)
}

/// The child half of the tests below: a no-op unless this process was
/// started by [`child`]. It runs one case through `shell::run`, with
/// hooks that print what they hear, and exits with the case's status.
#[test]
fn host_child() {
    let Ok(case) = std::env::var(CHILD_CASE) else { return };
    let dir = PathBuf::from(std::env::var(CHILD_DIR).unwrap());
    let (catalogs, config) = bench(&dir);
    let host = Host::new()
        .catalogs(catalogs)
        .after_describe(|_, entry, profile| println!("HOOK describe {} {profile}", entry.name()))
        .after_list(|_, entries| {
            let names: Vec<&str> = entries.iter().map(|e| e.name()).collect();
            println!("HOOK list {}", names.join(","));
        })
        .before_precache(|spec| {
            println!("HOOK precache {spec}");
            Ok(match spec {
                "group:pair" => vec!["alpha:default".to_string(), "beta:default".to_string()],
                other => vec![other.to_string()],
            })
        });
    let code = match case.as_str() {
        "describe" => run("bench", words(&["datasets", "describe", "alpha", "--configdir", &config]), &host),
        "list" => run("bench", words(&["datasets", "list", "--configdir", &config]), &host),
        "precache-group" => run("bench", words(&["datasets", "precache", "group:pair", "--configdir", &config]), &host),
        "complete-script" => run("fvec-bench", words(&["completions", "--shell", "bash"]), &host.owns_completion()),
        "complete-handshake" => run("fvec-bench", Vec::new(), &host.owns_completion()),
        "complete-request" => {
            let line = "fvec-bench datasets desc";
            run("fvec-bench", words(&[line, &line.len().to_string()]), &host.owns_completion())
        }
        "complete-path" => run("fvec bench", words(&["completions", "--shell", "bash"]), &host.owns_completion()),
        "embedded-completions" => run("fvec bench vectordata", words(&["completions", "--shell", "bash"]), &host),
        other => panic!("no case {other}"),
    };
    std::process::exit(code);
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Where `needle` first appears in `haystack`, failing with both.
fn at(haystack: &str, needle: &str) -> usize {
    haystack.find(needle).unwrap_or_else(|| panic!("no {needle:?} in:\n{haystack}"))
}

/// **A hook prints after its command's output.** describe's description
/// and list's table come first, then the hook's line; neither command
/// calls another command's hook.
#[test]
fn hooks_print_after_their_commands_output() {
    let (out, _tmp) = child("describe", &[]);
    let stdout = text(&out.stdout);
    assert!(out.status.success(), "{stdout}\n{}", text(&out.stderr));
    assert!(at(&stdout, "Dataset: alpha") < at(&stdout, "HOOK describe alpha default"), "{stdout}");
    assert!(at(&stdout, "Facets:") < at(&stdout, "HOOK describe"), "{stdout}");
    assert!(!stdout.contains("HOOK list") && !stdout.contains("HOOK precache"), "{stdout}");

    let (out, _tmp) = child("list", &[]);
    let stdout = text(&out.stdout);
    assert!(out.status.success(), "{stdout}\n{}", text(&out.stderr));
    let hook = at(&stdout, "HOOK list alpha,beta");
    assert!(at(&stdout, "DATASET") < hook && stdout[..hook].contains("beta"), "{stdout}");
    assert!(!stdout.contains("HOOK describe") && !stdout.contains("HOOK precache"), "{stdout}");
}

/// **A spec the host expands into two runs both, in one report**: one
/// heading, one closing line counting both datasets' facets, and the
/// hook heard before any of it.
#[test]
fn an_expanded_precache_spec_runs_as_one_report() {
    let (out, _tmp) = child("precache-group", &[]);
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert!(out.status.success(), "{stdout}\n{stderr}");
    assert!(stdout.contains("HOOK precache group:pair"), "{stdout}");
    assert!(stderr.contains("Prebuffering 2 profiles of 2 dataset(s) (alpha:default, beta:default)"), "{stderr}");
    let done: Vec<&str> = stderr.lines().filter(|l| l.starts_with("Precache done:")).collect();
    assert_eq!(done, ["Precache done: 2 facet(s), 0 B in 0.0s (0 B/s)."], "{stderr}");
}

/// **A host that owns completion is completed under its own name.**
/// `completions --shell bash` registers `fvec-bench` and calls it back
/// with `_FVEC_BENCH_COMPLETE`; with that set, the host answers the
/// handshake with the script and a line with this CLI's candidates.
#[test]
fn an_owning_host_answers_completion_under_its_own_name() {
    let (out, _tmp) = child("complete-script", &[]);
    let script = text(&out.stdout);
    assert!(out.status.success(), "{script}\n{}", text(&out.stderr));
    assert!(script.contains("complete -o nosort -o nospace -F _fvec-bench_complete fvec-bench"), "{script}");
    assert!(script.contains("_FVEC_BENCH_COMPLETE=bash"), "{script}");
    assert!(
        !script.contains("_VECTORDATA_") && !script.contains("_complete vectordata"),
        "nothing bound to the vectordata executable: {script}"
    );

    let (out, _tmp) = child("complete-handshake", &[("_FVEC_BENCH_COMPLETE", "bash")]);
    assert!(out.status.success());
    assert!(text(&out.stdout).contains("-F _fvec-bench_complete fvec-bench"), "{}", text(&out.stdout));

    let (out, _tmp) = child("complete-request", &[("_FVEC_BENCH_COMPLETE", "bash")]);
    let candidates = text(&out.stdout);
    assert!(out.status.success(), "{candidates}\n{}", text(&out.stderr));
    // libtest's own `test host_child ... ` may lead the first line.
    let offers = |out: &str| out.lines().any(|c| c.trim_end() == "describe" || c.trim_end().ends_with(" describe"));
    assert!(offers(&candidates), "{candidates}");

    // The `vectordata` executable's variable is not this host's.
    let (out, _tmp) = child("complete-request", &[("_VECTORDATA_COMPLETE", "bash")]);
    assert!(!offers(&text(&out.stdout)), "{}", text(&out.stdout));

    // A command path is no name a shell completes.
    let (out, _tmp) = child("complete-path", &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(text(&out.stderr).contains("is a command path"), "{}", text(&out.stderr));
}

/// **Without `owns_completion`, an embedding name keeps today's help**:
/// `completions` says to graft `completion_node()` and fails, printing
/// no script.
#[test]
fn an_embedding_host_keeps_the_graft_help() {
    let (out, _tmp) = child("embedded-completions", &[]);
    let stderr = text(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("fvec bench vectordata: completion for an embedded vectordata CLI is the host program's to answer:"), "{stderr}");
    assert!(stderr.contains("graft vectordata::shell::completion_node()"), "{stderr}");
    assert!(!text(&out.stdout).contains("complete -o"), "{}", text(&out.stdout));
}
