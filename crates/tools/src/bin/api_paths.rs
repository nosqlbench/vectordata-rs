// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Release guard: no public API path disappears between releases
//! without being listed as an intended removal.
//!
//! This is the release check for AC-5 / CB-3 of
//! `docs/design/srd-crate-boundaries.md` ("paths users know keep
//! working"). It runs in two passes:
//!
//! 1. `cargo semver-checks check-release --baseline-rev <tag>` per
//!    publishable crate. Every item of a `*_missing` lint is a
//!    candidate removal. semver-checks reports an item as missing
//!    when it is now re-exported from another crate, so most
//!    candidates are false positives.
//! 2. One probe crate (`target/api-paths/probe`) that `use`s every
//!    candidate path, one per line, checked with `cargo check`. A path
//!    the compiler cannot resolve (E0432 / E0433) or finds private
//!    (E0603) is a real removal.
//!
//! A real removal must be listed for this baseline in
//! `crates/tools/api-removals.txt` (`<baseline-tag> <path>  # reason`),
//! or the tool exits 1. Inherent methods, variants and fields cannot be
//! probed with `use`; they are listed as items to check by hand and do
//! not fail the run. Each crate's semver verdict is printed so that a
//! version bump smaller than the changes is visible; the version itself
//! is the maintainer's decision.
//!
//! Usage (from anywhere in the workspace; it takes minutes):
//!   cargo run -p tools --bin api-paths -- [--baseline <tag>] [--allowlist <file>]
//!
//! The default baseline is the newest `v*` tag reachable from HEAD. The
//! default allowlist is `crates/tools/api-removals.txt` under the
//! workspace root (`git rev-parse --show-toplevel` of the current
//! directory). Exit status: 0 clean, 1 unexpected removals, 2 the check
//! itself could not run.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::Instant;

/// Features checked per crate, on top of its defaults.
///
/// Only features that add modules without native libraries belong here:
/// the probe and semver-checks must build them on any machine. Never
/// all features — `embed-cuda`, `faiss`, `blas-system`, `simsimd` and
/// `sqlcipher` need native toolchains or libraries.
const FEATURES: &[(&str, &[&str])] = &[
    ("veks-io", &["npy", "parquet", "slab"]),
    ("veks-pipeline", &["knnutils"]),
];

/// Compiler errors that mean a path no longer resolves for users:
/// unresolved import, failed to resolve, private item.
const REMOVAL_CODES: &[&str] = &["E0432", "E0433", "E0603"];

/// The allowlist, relative to the workspace root.
const ALLOWLIST: &str = "crates/tools/api-removals.txt";

/// The tool's scratch area, relative to the workspace root.
const SCRATCH: &str = "target/api-paths";

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("api-paths: {e}");
            ExitCode::from(2)
        }
    }
}

/// Parsed command line.
struct Args {
    baseline: Option<String>,
    allowlist: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args { baseline: None, allowlist: None };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--baseline" => args.baseline = Some(it.next().ok_or("--baseline needs a tag")?),
            "--allowlist" => {
                args.allowlist = Some(PathBuf::from(it.next().ok_or("--allowlist needs a file")?))
            }
            "-h" | "--help" => {
                println!(
                    "usage: api-paths [--baseline <tag>] [--allowlist <file>]\n\
                     Fails when a public path of a publishable crate stops resolving\n\
                     since <tag> (default: newest v* tag reachable from HEAD) and is\n\
                     not listed for <tag> in {ALLOWLIST}."
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}` (try --help)")),
        }
    }
    Ok(args)
}

/// Returns `Ok(true)` when every removed path is allowlisted.
fn run() -> Result<bool, String> {
    let args = parse_args()?;
    require_semver_checks()?;
    let root = PathBuf::from(command_stdout(
        Command::new("git").args(["rev-parse", "--show-toplevel"]),
    )?);
    let baseline = match args.baseline {
        Some(b) => b,
        None => command_stdout(Command::new("git").current_dir(&root).args([
            "describe", "--tags", "--abbrev=0", "--match", "v*",
        ]))?,
    };
    let head = command_stdout(Command::new("git").current_dir(&root).args([
        "rev-parse", "--short", "HEAD",
    ]))?;
    let allowlist_path = args.allowlist.unwrap_or_else(|| root.join(ALLOWLIST));
    let allowlist_text = std::fs::read_to_string(&allowlist_path)
        .map_err(|e| format!("reading {}: {e}", allowlist_path.display()))?;
    let allowlist = parse_allowlist(&allowlist_text)?;
    let target_dir = root.join(SCRATCH).join("target");

    println!("api-paths: HEAD {head} against baseline {baseline}");
    let (macros, crates): (Vec<_>, Vec<_>) =
        workspace_crates(&root)?.into_iter().partition(|c| c.proc_macro);
    if !macros.is_empty() {
        let names: Vec<_> = macros.iter().map(|c| c.name.as_str()).collect();
        println!("skipped (proc-macro, no library API to check): {}", names.join(", "));
    }
    let (crates, new): (Vec<_>, Vec<_>) = crates.into_iter().partition(|c| {
        Command::new("git")
            .current_dir(&root)
            .args(["cat-file", "-e", &format!("{baseline}:{}/Cargo.toml", c.dir)])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    });
    if !new.is_empty() {
        let names: Vec<_> = new.iter().map(|c| c.name.as_str()).collect();
        println!("skipped (new since {baseline}, nothing to compare): {}", names.join(", "));
    }

    // Pass 1: semver-checks per crate.
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut members: Vec<(String, String)> = Vec::new();
    let mut verdicts: Vec<(String, String)> = Vec::new();
    let total = crates.len();
    for (i, c) in crates.iter().enumerate() {
        let feats = features_for(&c.name);
        let shown = if feats.is_empty() { String::new() } else { format!(" +{}", feats.join(",")) };
        println!("[{}/{total}] {}{shown}: cargo semver-checks against {baseline}", i + 1, c.name);
        let started = Instant::now();
        let report = semver_checks(&root, &target_dir, c, &baseline, feats)?;
        let missing = parse_missing(&report.stdout);
        let mut probed = 0;
        for item in missing {
            match classify(&item, &c.lib) {
                Classified::Path(path) => {
                    probed += 1;
                    candidates.push(Candidate { krate: c.name.clone(), path });
                }
                Classified::Member(m) => members.push((c.name.clone(), m)),
            }
        }
        println!(
            "[{}/{total}] {} done in {:.0}s: {}; {probed} paths reported missing",
            i + 1,
            c.name,
            started.elapsed().as_secs_f64(),
            report.verdict
        );
        verdicts.push((c.name.clone(), report.verdict));
    }
    let mut seen = BTreeSet::new();
    candidates.retain(|c| seen.insert(c.path.clone()));

    // Pass 2: does each reported path still resolve?
    let mut removed: Vec<(Candidate, String)> = Vec::new();
    if !candidates.is_empty() {
        println!("probing {} paths with cargo check ({SCRATCH}/probe)", candidates.len());
        let started = Instant::now();
        let errors = probe(&root, &target_dir, &crates, &candidates)?;
        println!("probe done in {:.0}s", started.elapsed().as_secs_f64());
        for (idx, err) in errors {
            removed.push((candidates[idx].clone(), err));
        }
    }

    // Report.
    println!();
    println!("semver verdicts (the version decision is the maintainer's):");
    for (name, verdict) in &verdicts {
        println!("  {name:<24} {verdict}");
    }
    if !members.is_empty() {
        println!();
        println!("methods to check ({}; `use` cannot probe them, not a failure):", members.len());
        for (name, m) in &members {
            println!("  {name:<24} {m}");
        }
    }
    let allowed: BTreeMap<&str, &str> = allowlist
        .iter()
        .filter(|e| e.baseline == baseline)
        .map(|e| (e.path.as_str(), e.reason.as_str()))
        .collect();
    let (expected, unexpected): (Vec<_>, Vec<_>) =
        removed.iter().partition(|(c, _)| allowed.contains_key(c.path.as_str()));
    let removed_paths: BTreeSet<&str> = removed.iter().map(|(c, _)| c.path.as_str()).collect();
    let stale: Vec<&str> =
        allowed.keys().copied().filter(|p| !removed_paths.contains(p)).collect();

    println!();
    println!(
        "{} paths reported missing, {} still resolve, {} removed ({} allowlisted, {} unexpected)",
        candidates.len(),
        candidates.len() - removed.len(),
        removed.len(),
        expected.len(),
        unexpected.len()
    );
    if !stale.is_empty() {
        println!();
        println!("allowlist entries for {baseline} that still resolve or were not reported ({}):", stale.len());
        for p in &stale {
            println!("  {p}");
        }
    }
    if unexpected.is_empty() {
        println!("OK: no public path disappeared without an entry in the allowlist");
        return Ok(true);
    }
    println!();
    println!("FAIL: public paths removed since {baseline} and not in {}:", allowlist_path.display());
    for (c, err) in &unexpected {
        println!("  {:<24} {}", c.krate, c.path);
        println!("  {:<24}   {err}", "");
    }
    println!();
    println!("Restore each path (a `pub use` re-export keeps it working), or, if the");
    println!("removal is intended, add `{baseline} <path>  # reason` to the allowlist.");
    Ok(false)
}

/// Fails with the install command when cargo-semver-checks is absent.
fn require_semver_checks() -> Result<(), String> {
    let ok = Command::new("cargo")
        .args(["semver-checks", "--version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if ok {
        Ok(())
    } else {
        Err("cargo-semver-checks is not installed; install it with\n  \
             cargo install --locked cargo-semver-checks"
            .into())
    }
}

/// Runs a command and returns its trimmed stdout, or an error naming it.
fn command_stdout(cmd: &mut Command) -> Result<String, String> {
    let out = cmd.output().map_err(|e| format!("running {cmd:?}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{cmd:?} failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

// ─── Workspace manifests ─────────────────────────────────────────────

/// A publishable workspace crate.
#[derive(Debug, Clone, PartialEq)]
struct Crate {
    /// Package name (`veks-io`).
    name: String,
    /// Library name, the first segment of its paths (`veks_io`).
    lib: String,
    /// Member directory relative to the workspace root.
    dir: String,
    /// A proc-macro crate, which semver-checks cannot check (rustdoc
    /// documents no library API for it).
    proc_macro: bool,
}

/// The publishable members of the workspace at `root`, in member order.
fn workspace_crates(root: &Path) -> Result<Vec<Crate>, String> {
    let read = |p: &Path| std::fs::read_to_string(p).map_err(|e| format!("reading {}: {e}", p.display()));
    let members = workspace_members(&read(&root.join("Cargo.toml"))?);
    if members.is_empty() {
        return Err("no `members` in the root Cargo.toml [workspace]".into());
    }
    let mut out = Vec::new();
    for dir in members {
        let manifest = read(&root.join(&dir).join("Cargo.toml"))?;
        if let Some(c) = parse_crate(&manifest, &dir) {
            out.push(c);
        }
    }
    Ok(out)
}

/// The `members = [...]` list of the `[workspace]` table.
fn workspace_members(manifest: &str) -> Vec<String> {
    let mut in_workspace = false;
    let mut in_members = false;
    let mut members = Vec::new();
    for line in manifest.lines() {
        let line = strip_toml_comment(line).trim();
        if !in_members && line.starts_with('[') && !line.starts_with("[\"") {
            in_workspace = line == "[workspace]";
            continue;
        }
        if in_workspace && !in_members && line.starts_with("members") {
            if let Some(rest) = line.split_once('=').map(|(_, r)| r.trim()) {
                in_members = true;
                collect_quoted(rest, &mut members);
                if rest.contains(']') {
                    in_members = false;
                }
            }
            continue;
        }
        if in_members {
            collect_quoted(line, &mut members);
            if line.contains(']') {
                in_members = false;
            }
        }
    }
    members
}

/// The crate a member manifest declares, or `None` when it is
/// `publish = false`.
fn parse_crate(manifest: &str, dir: &str) -> Option<Crate> {
    let mut table = String::new();
    let mut name = None;
    let mut lib = None;
    let mut proc_macro = false;
    for line in manifest.lines() {
        let line = strip_toml_comment(line).trim();
        if line.starts_with('[') {
            table = line.to_string();
            continue;
        }
        let Some((key, value)) = line.split_once('=') else { continue };
        let (key, value) = (key.trim(), value.trim());
        match (table.as_str(), key) {
            ("[package]", "name") => name = Some(unquote(value)),
            ("[package]", "publish") if value == "false" => return None,
            ("[lib]", "name") => lib = Some(unquote(value)),
            ("[lib]", "proc-macro") => proc_macro = value == "true",
            _ => {}
        }
    }
    let name = name?;
    let lib = lib.unwrap_or_else(|| name.replace('-', "_"));
    Some(Crate { name, lib, dir: dir.to_string(), proc_macro })
}

fn strip_toml_comment(line: &str) -> &str {
    // Manifests here never put `#` inside a string on a line we parse.
    line.split_once('#').map_or(line, |(code, _)| code)
}

fn unquote(value: &str) -> String {
    value.trim().trim_matches('"').to_string()
}

fn collect_quoted(text: &str, out: &mut Vec<String>) {
    let mut parts = text.split('"');
    parts.next();
    while let Some(s) = parts.next() {
        out.push(s.to_string());
        parts.next();
    }
}

fn features_for(name: &str) -> &'static [&'static str] {
    FEATURES.iter().find(|(n, _)| *n == name).map_or(&[], |(_, f)| f)
}

// ─── Pass 1: cargo semver-checks ─────────────────────────────────────

/// What one semver-checks run produced.
struct SemverReport {
    stdout: String,
    /// "4.0.2 -> 4.1.0 (minor change): semver requires new major version: …"
    verdict: String,
}

/// Runs semver-checks for one crate, echoing its progress lines.
fn semver_checks(
    root: &Path,
    target_dir: &Path,
    c: &Crate,
    baseline: &str,
    feats: &[&str],
) -> Result<SemverReport, String> {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(root)
        .env("CARGO_TARGET_DIR", target_dir)
        .args(["semver-checks", "check-release", "--color", "never", "-p", &c.name])
        .args(["--baseline-rev", baseline, "--default-features"]);
    if !feats.is_empty() {
        cmd.args(["--features", &feats.join(",")]);
    }
    let (status, stdout, stderr) = run_streaming(&mut cmd, |line| {
        let t = line.trim_start();
        if ["Building", "Built", "Checking", "Checked"].iter().any(|w| t.starts_with(w)) {
            println!("      {t}");
        }
    })?;
    match parse_verdict(&stderr) {
        Some(verdict) => Ok(SemverReport { stdout, verdict }),
        None => Err(format!(
            "cargo semver-checks gave no verdict for {} (exit {status}):\n{}",
            c.name,
            tail(&stderr, 30)
        )),
    }
}

/// Spawns `cmd`, passes each stderr line to `on_line` as it arrives, and
/// returns the exit status with the full stdout and stderr.
fn run_streaming(
    cmd: &mut Command,
    mut on_line: impl FnMut(&str),
) -> Result<(String, String, String), String> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("running {cmd:?}: {e}"))?;
    let mut out = child.stdout.take().ok_or("no stdout pipe")?;
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out.read_to_string(&mut s);
        s
    });
    let mut stderr = String::new();
    let err = child.stderr.take().ok_or("no stderr pipe")?;
    for line in BufReader::new(err).lines() {
        let line = line.map_err(|e| format!("reading stderr: {e}"))?;
        on_line(&line);
        stderr.push_str(&line);
        stderr.push('\n');
    }
    let status = child.wait().map_err(|e| format!("waiting for {cmd:?}: {e}"))?;
    let stdout = reader.join().map_err(|_| "stdout reader panicked")?;
    let code = status.code().map_or_else(|| "signal".to_string(), |c| c.to_string());
    Ok((code, stdout, stderr))
}

fn tail(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// The verdict from semver-checks' stderr: its `Checking a -> b (…)`
/// version line joined with its `Summary` line.
fn parse_verdict(stderr: &str) -> Option<String> {
    let mut versions = None;
    let mut summary = None;
    for line in stderr.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("Checking ") {
            // "vectordata v4.0.2 -> v4.1.0 (minor change)"
            versions = rest.split_once(' ').map(|(_, v)| v.to_string());
        } else if let Some(rest) = t.strip_prefix("Summary ") {
            summary = Some(rest.trim().to_string());
        }
    }
    let summary = summary?;
    Some(match versions {
        Some(v) => format!("{v}: {summary}"),
        None => summary,
    })
}

/// One `Failed in:` entry of a `*_missing` lint.
#[derive(Debug, Clone, PartialEq)]
struct MissingItem {
    /// The lint, e.g. `enum_missing`.
    lint: String,
    /// The text before `, previously in file` / ` in file`, e.g.
    /// `enum vectordata::catalog::source::CatalogOrigin`.
    item: String,
    /// The baseline source file the item was defined in.
    file: String,
}

/// Every item of every `*_missing` lint in semver-checks' stdout.
fn parse_missing(stdout: &str) -> Vec<MissingItem> {
    let mut items = Vec::new();
    let mut lint: Option<String> = None;
    let mut in_failed = false;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("--- failure ") {
            let name = rest.split(':').next().unwrap_or("").trim();
            lint = name.ends_with("_missing").then(|| name.to_string());
            in_failed = false;
            continue;
        }
        if line.trim() == "Failed in:" {
            in_failed = true;
            continue;
        }
        if line.trim().is_empty() {
            in_failed = false;
            continue;
        }
        let (Some(lint), true) = (&lint, in_failed) else { continue };
        let entry = line.trim();
        let (item, file) = entry
            .split_once(", previously in file ")
            .or_else(|| entry.split_once(" in file "))
            .unwrap_or((entry, ""));
        let file = file.rsplit_once(':').map_or(file, |(f, _)| f);
        items.push(MissingItem { lint: lint.clone(), item: item.to_string(), file: file.to_string() });
    }
    items
}

/// What a missing item becomes.
#[derive(Debug, PartialEq)]
enum Classified {
    /// A path the probe can `use`.
    Path(String),
    /// A method, variant, field or other member, listed for a human.
    Member(String),
}

/// Turns a reported item into a probe path where it names one.
///
/// - `enum a::b::C` (any kind word, path rooted at the crate) → `a::b::C`
/// - `macro m` (exported `macro_rules!`, crate root) → `lib::m`
/// - `NAME` of a const/static lint → the module of its baseline file
///   (`src/x/y.rs` → `lib::x::y::NAME`)
/// - anything else (`Type::method`, `variant E::V`, …) → a member
fn classify(m: &MissingItem, lib: &str) -> Classified {
    let prefix = format!("{lib}::");
    match m.item.split_once(' ') {
        Some((_, rest)) if rest.starts_with(&prefix) && !rest.contains(' ') => {
            Classified::Path(rest.to_string())
        }
        Some(("macro", name)) if !name.contains("::") && !name.contains(' ') => {
            Classified::Path(format!("{lib}::{name}"))
        }
        None if (m.lint.contains("const") || m.lint.contains("static")) && !m.item.contains("::") => {
            match module_of_file(&m.file) {
                Some(module) if module.is_empty() => Classified::Path(format!("{lib}::{}", m.item)),
                Some(module) => Classified::Path(format!("{lib}::{module}::{}", m.item)),
                None => Classified::Member(m.item.clone()),
            }
        }
        _ => Classified::Member(m.item.clone()),
    }
}

/// The module path of a source file under a crate's `src/`
/// (`…/src/dataset/facet.rs` → `dataset::facet`, `…/src/lib.rs` → ``).
fn module_of_file(file: &str) -> Option<String> {
    let (_, rel) = file.rsplit_once("/src/")?;
    let rel = rel.strip_suffix(".rs")?;
    let mut parts: Vec<&str> = rel.split('/').collect();
    if matches!(parts.last(), Some(&"mod") | Some(&"lib")) {
        parts.pop();
    }
    Some(parts.join("::"))
}

// ─── Pass 2: the probe crate ─────────────────────────────────────────

/// A path to probe and the crate that reported it.
#[derive(Debug, Clone)]
struct Candidate {
    krate: String,
    path: String,
}

/// Writes the probe crate, checks it, and returns the compiler error for
/// each candidate (by index) that failed to resolve.
fn probe(
    root: &Path,
    target_dir: &Path,
    crates: &[Crate],
    candidates: &[Candidate],
) -> Result<BTreeMap<usize, String>, String> {
    let dir = root.join(SCRATCH).join("probe");
    std::fs::create_dir_all(dir.join("src")).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    let used: BTreeSet<&str> = candidates.iter().map(|c| c.krate.as_str()).collect();
    let mut manifest = String::from(
        "# Generated by `cargo run -p tools --bin api-paths`; overwritten each run.\n\
         [package]\nname = \"api-paths-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n\
         # Its own workspace, not a member of the one it sits in.\n[workspace]\n\n[dependencies]\n",
    );
    for c in crates.iter().filter(|c| used.contains(c.name.as_str())) {
        let path = root.join(&c.dir);
        let feats: Vec<String> = features_for(&c.name).iter().map(|f| format!("\"{f}\"")).collect();
        let _ = writeln!(
            manifest,
            "{} = {{ path = \"{}\", features = [{}] }}",
            c.name,
            path.display(),
            feats.join(", ")
        );
    }
    let (source, lines) = probe_source(candidates);
    write(&dir.join("Cargo.toml"), &manifest)?;
    write(&dir.join("src/lib.rs"), &source)?;
    // Start from the workspace's resolution so the probe builds the
    // dependency versions the workspace does (and works offline).
    if let Ok(lock) = std::fs::read_to_string(root.join("Cargo.lock")) {
        write(&dir.join("Cargo.lock"), &lock)?;
    }

    let mut cmd = Command::new("cargo");
    cmd.current_dir(&dir)
        .env("CARGO_TARGET_DIR", target_dir)
        .args(["check", "--message-format", "short"]);
    let mut built = 0usize;
    let (status, _, stderr) = run_streaming(&mut cmd, |line| {
        let t = line.trim_start();
        if t.starts_with("Compiling") || t.starts_with("Checking") {
            built += 1;
            if built.is_multiple_of(50) || t.contains("api-paths-probe") {
                println!("      probe: {built} crates built ({t})");
            }
        }
    })?;
    let mut errors = BTreeMap::new();
    for line in stderr.lines() {
        if let Some(e) = parse_probe_error(line)
            && let Some(&idx) = lines.get(&e.line)
        {
            errors.entry(idx).or_insert(e.message);
        }
    }
    if status != "0" && errors.is_empty() {
        return Err(format!("the probe failed to build for another reason:\n{}", tail(&stderr, 40)));
    }
    Ok(errors)
}

/// The probe's `lib.rs`, one `use` per line, and the map from line
/// number to candidate index.
fn probe_source(candidates: &[Candidate]) -> (String, BTreeMap<usize, usize>) {
    let mut source = String::from("// Generated by api-paths: one `use` per reported path.\n");
    let mut lines = BTreeMap::new();
    for (i, c) in candidates.iter().enumerate() {
        let _ = writeln!(source, "mod m{i} {{ #[allow(unused_imports)] use ::{}; }}", c.path);
        lines.insert(i + 2, i);
    }
    (source, lines)
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    std::fs::write(path, text).map_err(|e| format!("writing {}: {e}", path.display()))
}

/// A compiler error in the probe's `lib.rs`.
#[derive(Debug, PartialEq)]
struct ProbeError {
    line: usize,
    /// `error[E0603]: enum `CatalogOrigin` is private`
    message: String,
}

/// Parses a `--message-format short` line of a path-removal error:
/// `src/lib.rs:12:40: error[E0432]: unresolved import `…``.
fn parse_probe_error(line: &str) -> Option<ProbeError> {
    let rest = line.strip_prefix("src/lib.rs:")?;
    let (line_no, rest) = rest.split_once(':')?;
    let (_col, rest) = rest.split_once(':')?;
    let message = rest.trim();
    let code = message.strip_prefix("error[")?.split(']').next()?;
    if !REMOVAL_CODES.contains(&code) {
        return None;
    }
    Some(ProbeError { line: line_no.parse().ok()?, message: message.to_string() })
}

// ─── The allowlist ───────────────────────────────────────────────────

/// One intended removal.
#[derive(Debug, PartialEq)]
struct Allowed {
    baseline: String,
    path: String,
    reason: String,
}

/// Parses `<baseline-tag> <path>  # reason` lines; `#` lines and blank
/// lines are skipped.
fn parse_allowlist(text: &str) -> Result<Vec<Allowed>, String> {
    let mut out = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let (body, reason) = raw.split_once('#').unwrap_or((raw, ""));
        let mut words = body.split_whitespace();
        let (Some(baseline), path) = (words.next(), words.next()) else { continue };
        let (Some(path), None) = (path, words.next()) else {
            return Err(format!(
                "{ALLOWLIST}:{}: expected `<baseline-tag> <path>  # reason`, got `{}`",
                n + 1,
                raw.trim()
            ));
        };
        out.push(Allowed {
            baseline: baseline.to_string(),
            path: path.to_string(),
            reason: reason.trim().to_string(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured from cargo-semver-checks 0.51.0 (vectordata, v4.0.2 ->
    /// 4.1.0), file paths shortened.
    const SEMVER_STDOUT: &str = "\
--- failure declarative_macro_missing: macro_rules declaration removed or renamed ---

Description:
A `macro_rules!` declarative macro cannot be invoked by its prior name. The macro may have been renamed or removed entirely.
        ref: https://doc.rust-lang.org/reference/macros-by-example.html#path-based-scope
       impl: https://github.com/obi1kenobi/cargo-semver-checks/tree/v0.51.0/src/lints/declarative_macro_missing.ron

Failed in:
  macro dispatch_typed, previously in file /w/target/semver-checks/git-v4_0_2/98dd/crates/vectordata/src/typed_access.rs:562

--- failure enum_missing: pub enum removed or renamed ---

Description:
A publicly-visible enum cannot be imported by its prior path. A `pub use` may have been removed, or the enum itself may have been renamed or removed entirely.
        ref: https://doc.rust-lang.org/cargo/reference/semver.html#item-remove
       impl: https://github.com/obi1kenobi/cargo-semver-checks/tree/v0.51.0/src/lints/enum_missing.ron

Failed in:
  enum vectordata::catalog::source::CatalogOrigin, previously in file /w/target/semver-checks/git-v4_0_2/98dd/crates/vectordata/src/catalog/source.rs:63
  enum vectordata::Error, previously in file /w/target/semver-checks/git-v4_0_2/98dd/crates/vectordata/src/lib.rs:338

--- failure function_parameter_count_changed: pub fn parameter count changed ---

Description:
A publicly-visible function now takes a different number of parameters.

Failed in:
  vectordata::fetch::fetch now takes 3 parameters instead of 2, in /w/crates/vectordata/src/fetch.rs:10

--- failure inherent_method_missing: pub method removed or renamed ---

Description:
A publicly-visible method or associated fn is no longer available under its prior name. It may have been renamed or removed entirely.

Failed in:
  GenericTestDataView::open_facet_typed, previously in file /w/target/semver-checks/git-v4_0_2/98dd/crates/vectordata/src/view.rs:1631

--- failure module_missing: pub module removed or renamed ---

Failed in:
  mod vectordata::dataset::facet, previously in file /w/target/semver-checks/git-v4_0_2/98dd/crates/vectordata/src/dataset/facet.rs:1

--- failure pub_module_level_const_missing: pub module-level const is missing ---

Description:
A public const is missing or renamed

Failed in:
  SURVEY_NAMESPACE in file /w/target/semver-checks/git-v4_0_2/98dd/crates/vectordata/src/dataset/facet.rs:30
  ROOT_CONST in file /w/target/semver-checks/git-v4_0_2/98dd/crates/vectordata/src/lib.rs:12
  NESTED in file /w/target/semver-checks/git-v4_0_2/98dd/crates/vectordata/src/catalog/mod.rs:12
";

    /// Captured stderr of the same run.
    const SEMVER_STDERR: &str = "\
     Cloning v4.0.2
    Building vectordata v4.1.0 (current)
       Built [  25.021s] (current)
    Checking vectordata v4.0.2 -> v4.1.0 (minor change)
     Checked [   0.029s] 202 checks: 194 pass, 8 fail, 0 warn, 58 skip

     Summary semver requires new major version: 8 major and 0 minor checks failed
    Finished [  51.712s] vectordata
";

    #[test]
    fn only_missing_lints_are_collected() {
        let items = parse_missing(SEMVER_STDOUT);
        let lints: BTreeSet<&str> = items.iter().map(|i| i.lint.as_str()).collect();
        assert_eq!(
            lints,
            BTreeSet::from([
                "declarative_macro_missing",
                "enum_missing",
                "inherent_method_missing",
                "module_missing",
                "pub_module_level_const_missing",
            ])
        );
        assert_eq!(items.len(), 8);
        assert_eq!(
            items[1],
            MissingItem {
                lint: "enum_missing".into(),
                item: "enum vectordata::catalog::source::CatalogOrigin".into(),
                file: "/w/target/semver-checks/git-v4_0_2/98dd/crates/vectordata/src/catalog/source.rs".into(),
            }
        );
    }

    #[test]
    fn items_become_probe_paths_or_members() {
        let got: Vec<Classified> =
            parse_missing(SEMVER_STDOUT).iter().map(|m| classify(m, "vectordata")).collect();
        use Classified::{Member, Path};
        assert_eq!(
            got,
            vec![
                Path("vectordata::dispatch_typed".into()),
                Path("vectordata::catalog::source::CatalogOrigin".into()),
                Path("vectordata::Error".into()),
                Member("GenericTestDataView::open_facet_typed".into()),
                Path("vectordata::dataset::facet".into()),
                Path("vectordata::dataset::facet::SURVEY_NAMESPACE".into()),
                Path("vectordata::ROOT_CONST".into()),
                Path("vectordata::catalog::NESTED".into()),
            ]
        );
    }

    #[test]
    fn variants_and_paths_of_other_crates_are_members() {
        let m = |lint: &str, item: &str| MissingItem { lint: lint.into(), item: item.into(), file: String::new() };
        assert_eq!(
            classify(&m("enum_variant_missing", "variant Op:Lt"), "vectordata"),
            Classified::Member("variant Op:Lt".into())
        );
        assert_eq!(
            classify(&m("struct_missing", "struct veks_io::Reader"), "vectordata"),
            Classified::Member("struct veks_io::Reader".into())
        );
    }

    #[test]
    fn the_verdict_joins_versions_and_summary() {
        assert_eq!(
            parse_verdict(SEMVER_STDERR).as_deref(),
            Some(
                "v4.0.2 -> v4.1.0 (minor change): semver requires new major version: \
                 8 major and 0 minor checks failed"
            )
        );
        assert_eq!(
            parse_verdict("     Summary no semver update required\n").as_deref(),
            Some("no semver update required")
        );
        assert_eq!(parse_verdict("error: could not compile\n"), None);
    }

    #[test]
    fn removal_errors_map_to_probe_lines() {
        // Captured `cargo check --message-format short` output.
        let stderr = "\
src/lib.rs:3:44: error[E0603]: enum `CatalogOrigin` is private
src/lib.rs:7:38: error[E0432]: unresolved import `vectordata::model`
src/lib.rs:9:30: error[E0433]: failed to resolve: could not find `merkle` in `vectordata`
src/lib.rs:11:1: warning: use of deprecated struct `vectordata::Old`
src/lib.rs:12:5: error[E0308]: mismatched types
error: could not compile `api-paths-probe` (lib) due to 3 previous errors
";
        let got: Vec<ProbeError> = stderr.lines().filter_map(parse_probe_error).collect();
        assert_eq!(
            got,
            vec![
                ProbeError { line: 3, message: "error[E0603]: enum `CatalogOrigin` is private".into() },
                ProbeError { line: 7, message: "error[E0432]: unresolved import `vectordata::model`".into() },
                ProbeError {
                    line: 9,
                    message: "error[E0433]: failed to resolve: could not find `merkle` in `vectordata`".into()
                },
            ]
        );
    }

    #[test]
    fn probe_lines_index_candidates() {
        let c = |p: &str| Candidate { krate: "vectordata".into(), path: p.into() };
        let (source, lines) = probe_source(&[c("vectordata::a::B"), c("vectordata::C")]);
        let text: Vec<&str> = source.lines().collect();
        assert_eq!(text[1], "mod m0 { #[allow(unused_imports)] use ::vectordata::a::B; }");
        assert_eq!(text[2], "mod m1 { #[allow(unused_imports)] use ::vectordata::C; }");
        assert_eq!(lines, BTreeMap::from([(2, 0), (3, 1)]));
    }

    #[test]
    fn module_paths_come_from_baseline_files() {
        assert_eq!(module_of_file("/x/crates/v/src/lib.rs").as_deref(), Some(""));
        assert_eq!(module_of_file("/x/crates/v/src/a/mod.rs").as_deref(), Some("a"));
        assert_eq!(module_of_file("/x/crates/v/src/a/b.rs").as_deref(), Some("a::b"));
        assert_eq!(module_of_file("no-source"), None);
    }

    #[test]
    fn the_allowlist_keys_on_baseline_and_path() {
        let text = "\
# header comment

v4.0.2 vectordata::merkle  # moved to veks-transport
v4.1.0   veks_core::formats
";
        assert_eq!(
            parse_allowlist(text).unwrap(),
            vec![
                Allowed {
                    baseline: "v4.0.2".into(),
                    path: "vectordata::merkle".into(),
                    reason: "moved to veks-transport".into()
                },
                Allowed { baseline: "v4.1.0".into(), path: "veks_core::formats".into(), reason: String::new() },
            ]
        );
        assert!(parse_allowlist("v4.0.2\n").is_err());
        assert!(parse_allowlist("v4.0.2 a::b c::d\n").is_err());
    }

    #[test]
    fn manifests_yield_publishable_crates() {
        let root = "\
[workspace]
members = [
    \"crates/vectordata\",   # the client
    \"crates/tools\",
]
default-members = [\"crates/other\"]
";
        assert_eq!(workspace_members(root), vec!["crates/vectordata", "crates/tools"]);
        assert_eq!(
            parse_crate("[package]\nname = \"veks-io\"\nversion = \"1\"\n", "crates/veks-io"),
            Some(Crate {
                name: "veks-io".into(),
                lib: "veks_io".into(),
                dir: "crates/veks-io".into(),
                proc_macro: false
            })
        );
        assert_eq!(
            parse_crate("[package]\nname = \"vecd\"\n[lib]\nname = \"vecd_lib\"\n", "crates/vecd").map(|c| c.lib),
            Some("vecd_lib".into())
        );
        assert_eq!(parse_crate("[package]\nname = \"tools\"\npublish = false\n", "crates/tools"), None);
        assert!(parse_crate("[package]\nname = \"d\"\n[lib]\nproc-macro = true\n", "crates/d").unwrap().proc_macro);
    }

    #[test]
    fn the_feature_table_never_names_native_features() {
        for (_, feats) in FEATURES {
            for f in *feats {
                assert!(
                    !["embed", "embed-cuda", "faiss", "blas-system", "simsimd", "sqlcipher"].contains(f),
                    "{f} needs a native library"
                );
            }
        }
    }
}
