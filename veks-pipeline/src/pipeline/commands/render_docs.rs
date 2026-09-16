// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Pipeline command: render a dataset's Markdown documents to HTML.
//!
//! A dataset is documented in Markdown — `README.md`, `LICENSE.md` and
//! their kin at the root, and everything under `docs/` — and that
//! Markdown is the source of truth. Hosted as objects on S3 it is shown
//! as text, so this step projects it to HTML that needs nothing else:
//! no script, no stylesheet or font fetched from anywhere, so a page
//! opens the same from the bucket, from a local copy, or from any
//! static host.
//!
//! The rendered tree mirrors the Markdown tree from the dataset root,
//! rooted in its own folder: `README.md` → `docs/html/README.html`,
//! `docs/x.md` → `docs/html/docs/x.html`. Every link is rewritten
//! relative to the page it sits in: a link to a Markdown document
//! points at that document's page, a link to anything else — an image,
//! a data file — points back at the real file. Nothing is copied but
//! the text.
//!
//! The step runs in the finalize pass after the generated reference,
//! and the Markdown documents are an input of every finalize step by
//! content, so an edit re-renders on the next run.

use std::path::{Path, PathBuf};
use std::time::Instant;

use pulldown_cmark::{html, Event, Options as MdOptions, Parser, Tag, TagEnd};

use crate::pipeline::command::{
    ArtifactManifest, ArtifactState, CommandDoc, CommandOp, CommandResult, OptionDesc, OptionRole, Options,
    ResourceDesc, Status, StreamContext, render_options_table,
};

/// The default folder of the rendered tree, under the dataset root.
pub const DEFAULT_OUTPUT: &str = "docs/html";

pub struct RenderDocsOp;

pub fn factory() -> Box<dyn CommandOp> {
    Box::new(RenderDocsOp)
}

impl CommandOp for RenderDocsOp {
    fn command_path(&self) -> &str {
        "generate docs-html"
    }

    fn category(&self) -> &'static dyn veks_completion::CategoryTag {
        &crate::pipeline::command::CAT_ANALYZE
    }

    fn level(&self) -> &'static dyn veks_completion::LevelTag {
        &crate::pipeline::command::LVL_PRIMARY
    }

    fn command_doc(&self) -> CommandDoc {
        let options = self.describe_options();
        CommandDoc {
            summary: "Render the dataset's Markdown documents to a self-contained HTML mirror".into(),
            body: format!(
                r#"# generate docs-html

Render every Markdown document of the dataset — `README.md`, `LICENSE.md`
and their kin at the root, everything under `docs/` — to HTML under
`docs/html/`, mirroring the Markdown tree from the dataset root:
`README.md` becomes `docs/html/README.html`, `docs/x.md` becomes
`docs/html/docs/x.html`.

Each page is self-contained: the stylesheet is inline and nothing is
fetched from anywhere. Links are rewritten relative to the page: a link
to a Markdown document points at that document's page, a link to an
image or a data file points back at the real file, external links and
anchors are left alone.

The Markdown is the source of truth; the HTML is a projection of it,
regenerated whenever a document changes.

## Options

{}"#,
                render_options_table(&options)
            ),
        }
    }

    fn describe_resources(&self) -> Vec<ResourceDesc> {
        vec![]
    }

    fn execute(&mut self, options: &Options, ctx: &mut StreamContext) -> CommandResult {
        let start = Instant::now();
        let source = resolve(options.get("source").unwrap_or("."), &ctx.workspace);
        let output_rel = options.get("output").unwrap_or(DEFAULT_OUTPUT).to_string();
        match render_tree(&source, &output_rel) {
            Ok(pages) => {
                for p in &pages {
                    ctx.ui.log(&format!("  {} ← {}", p.page.display(), p.source.display()));
                }
                let produced: Vec<PathBuf> = pages.iter().map(|p| source.join(&p.page)).collect();
                CommandResult {
                    status: Status::Ok,
                    message: format!("rendered {} document(s) to {output_rel}/", pages.len()),
                    produced,
                    elapsed: start.elapsed(),
                }
            }
            Err(e) => CommandResult { status: Status::Error, message: e, produced: vec![], elapsed: start.elapsed() },
        }
    }

    fn describe_options(&self) -> Vec<OptionDesc> {
        vec![
            OptionDesc {
                name: "source".into(),
                type_name: "Path".into(),
                required: false,
                default: Some(".".into()),
                description: "Dataset directory; its root documents and docs/ are rendered".into(),
                extended_description: None,
                // A directory, not an input file: its mtime moves with every
                // write beside it. The documents themselves are the step's
                // input, by content, through the finalize pass.
                role: OptionRole::Config,
            },
            OptionDesc {
                name: "output".into(),
                type_name: "Path".into(),
                required: false,
                default: Some(DEFAULT_OUTPUT.into()),
                description: "Folder of the rendered tree, under the dataset directory".into(),
                extended_description: None,
                role: OptionRole::Output,
            },
        ]
    }

    /// The rendered tree is complete when every page a source projects
    /// to is there and non-empty; a page missing is partial (the step
    /// re-renders the whole tree); no tree at all is absent.
    fn check_artifact(&self, output: &Path, options: &Options) -> ArtifactState {
        if !output.is_dir() {
            return ArtifactState::Absent;
        }
        let source = PathBuf::from(options.get("source").unwrap_or("."));
        let output_rel = options.get("output").unwrap_or(DEFAULT_OUTPUT).to_string();
        // `output` is the tree under `source`; pages are located from it.
        let root = output.parent().and_then(|p| if output_rel.contains('/') { p.parent() } else { Some(p) });
        let root = match root {
            Some(r) => r.to_path_buf(),
            None => source,
        };
        let sources = markdown_sources(&root, &output_rel);
        if sources.is_empty() {
            return ArtifactState::Complete;
        }
        let all = sources.iter().all(|s| {
            std::fs::metadata(root.join(page_path(s, &output_rel))).map(|m| m.len() > 0).unwrap_or(false)
        });
        if all { ArtifactState::Complete } else { ArtifactState::Partial }
    }

    fn project_artifacts(&self, step_id: &str, options: &Options) -> ArtifactManifest {
        self.project_artifacts_in(step_id, options, Path::new("."))
    }

    /// The pages follow from the documents the dataset holds, so the
    /// projection reads the dataset directory the caller names; a
    /// relative `source` is under it.
    fn project_artifacts_in(&self, step_id: &str, options: &Options, workspace: &Path) -> ArtifactManifest {
        let source = resolve(options.get("source").unwrap_or("."), workspace);
        let output_rel = options.get("output").unwrap_or(DEFAULT_OUTPUT).to_string();
        let sources = markdown_sources(&source, &output_rel);
        ArtifactManifest {
            step_id: step_id.to_string(),
            command: self.command_path().to_string(),
            inputs: sources.iter().map(|s| s.to_string_lossy().replace('\\', "/")).collect(),
            outputs: sources.iter().map(|s| page_path(s, &output_rel).to_string_lossy().replace('\\', "/")).collect(),
            intermediates: vec![],
        }
    }
}

/// One rendered page: the Markdown it came from and where it went,
/// both relative to the dataset root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    pub source: PathBuf,
    pub page: PathBuf,
}

/// Every Markdown document of the dataset, relative to its root, in
/// sorted order: the static payload at the root that is Markdown, and
/// every `.md` under `docs/` except the rendered tree itself.
pub fn markdown_sources(root: &Path, output_rel: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if e.path().is_file() && vectordata::filters::is_static_payload(&name) && is_markdown(&name) {
                out.push(PathBuf::from(name));
            }
        }
    }
    let output_abs = root.join(output_rel);
    walk_markdown(root, &root.join("docs"), &output_abs, &mut out);
    out.sort();
    out.dedup();
    out
}

fn walk_markdown(root: &Path, dir: &Path, skip: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if vectordata::filters::is_excluded_dir(&name) || same_file(&path, skip) {
                continue;
            }
            walk_markdown(root, &path, skip, out);
        } else if path.is_file()
            && is_markdown(&name)
            && let Ok(rel) = path.strip_prefix(root)
        {
            out.push(rel.to_path_buf());
        }
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

fn is_markdown(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".md") || lower.ends_with(".markdown")
}

/// Where a Markdown document renders to: the mirrored path under the
/// output folder with the extension replaced.
pub fn page_path(source_rel: &Path, output_rel: &str) -> PathBuf {
    let mut p = PathBuf::from(output_rel);
    let mut mirrored = source_rel.to_path_buf();
    mirrored.set_extension("html");
    p.push(mirrored);
    p
}

/// Render every document under `root` into `output_rel`; the pages, in
/// source order.
pub fn render_tree(root: &Path, output_rel: &str) -> Result<Vec<Page>, String> {
    let sources = markdown_sources(root, output_rel);
    let source_set: std::collections::HashSet<String> =
        sources.iter().map(|s| slashed(s)).collect();
    let mut pages = Vec::new();
    for src in &sources {
        let text = std::fs::read_to_string(root.join(src)).map_err(|e| format!("{}: {e}", src.display()))?;
        let page = page_path(src, output_rel);
        let html = render_document(&text, &slashed(src), &slashed(&page), output_rel, &source_set);
        if let Some(parent) = root.join(&page).parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::write(root.join(&page), html).map_err(|e| format!("{}: {e}", page.display()))?;
        pages.push(Page { source: src.clone(), page });
    }
    Ok(pages)
}

fn slashed(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

/// One document to one self-contained page. `source_rel` and
/// `page_rel` are dataset-relative, forward-slashed; `sources` holds
/// every Markdown source, so a link to one becomes a link to its page.
pub fn render_document(
    markdown: &str,
    source_rel: &str,
    page_rel: &str,
    output_rel: &str,
    sources: &std::collections::HashSet<String>,
) -> String {
    let src_dir = dir_of(source_rel);
    let page_dir = dir_of(page_rel);
    let mut opts = MdOptions::empty();
    opts.insert(MdOptions::ENABLE_TABLES);
    opts.insert(MdOptions::ENABLE_FOOTNOTES);
    opts.insert(MdOptions::ENABLE_STRIKETHROUGH);
    opts.insert(MdOptions::ENABLE_TASKLISTS);
    opts.insert(MdOptions::ENABLE_HEADING_ATTRIBUTES);
    let parser = Parser::new_ext(markdown, opts);
    let mut title: Option<String> = None;
    let mut in_h1 = false;
    let mut events: Vec<Event> = Vec::new();
    for ev in parser {
        match ev {
            Event::Start(Tag::Link { link_type, dest_url, title: t, id }) => {
                let dest = rewrite_link(&dest_url, &src_dir, &page_dir, output_rel, sources);
                events.push(Event::Start(Tag::Link { link_type, dest_url: dest.into(), title: t, id }));
            }
            Event::Start(Tag::Image { link_type, dest_url, title: t, id }) => {
                let dest = rewrite_link(&dest_url, &src_dir, &page_dir, output_rel, sources);
                events.push(Event::Start(Tag::Image { link_type, dest_url: dest.into(), title: t, id }));
            }
            Event::Start(Tag::Heading { level: pulldown_cmark::HeadingLevel::H1, .. }) if title.is_none() => {
                in_h1 = true;
                events.push(ev);
            }
            Event::End(TagEnd::Heading(pulldown_cmark::HeadingLevel::H1)) if in_h1 => {
                in_h1 = false;
                events.push(ev);
            }
            Event::Text(ref t) | Event::Code(ref t) if in_h1 => {
                title.get_or_insert_with(String::new).push_str(t);
                events.push(ev);
            }
            other => events.push(other),
        }
    }
    let mut body = String::new();
    html::push_html(&mut body, events.into_iter());
    let title = title
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| Path::new(source_rel).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());
    page(&title, &body, source_rel)
}

fn dir_of(rel: &str) -> String {
    match rel.rsplit_once('/') {
        Some((d, _)) => d.to_string(),
        None => String::new(),
    }
}

/// A link as the page must write it. External links, anchors and
/// absolute paths are left alone. A relative link is resolved against
/// the document's directory to a dataset-relative target; a target
/// that is a Markdown source becomes its page, anything else stays the
/// real file; the result is written relative to the page's directory.
pub fn rewrite_link(
    dest: &str,
    src_dir: &str,
    page_dir: &str,
    output_rel: &str,
    sources: &std::collections::HashSet<String>,
) -> String {
    if dest.is_empty() || dest.starts_with('#') || dest.starts_with('/') || dest.contains("://") || dest.starts_with("mailto:") || dest.starts_with("data:") {
        return dest.to_string();
    }
    let (path, fragment) = match dest.find('#') {
        Some(i) => (&dest[..i], &dest[i..]),
        None => (dest, ""),
    };
    let target = normalize(&if src_dir.is_empty() { path.to_string() } else { format!("{src_dir}/{path}") });
    let target_out = if sources.contains(&target) {
        slashed(&page_path(Path::new(&target), output_rel))
    } else {
        target
    };
    format!("{}{fragment}", relative(page_dir, &target_out))
}

/// `a/./b/../c` → `a/c`; leading `..` beyond the root are dropped.
fn normalize(p: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            s => parts.push(s),
        }
    }
    parts.join("/")
}

/// The path of `to` as written from inside `from_dir`, both
/// dataset-relative.
fn relative(from_dir: &str, to: &str) -> String {
    let from: Vec<&str> = from_dir.split('/').filter(|s| !s.is_empty()).collect();
    let to_parts: Vec<&str> = to.split('/').filter(|s| !s.is_empty()).collect();
    let common = from.iter().zip(to_parts.iter()).take_while(|(a, b)| a == b).count();
    let mut out: Vec<String> = vec!["..".to_string(); from.len() - common];
    out.extend(to_parts[common..].iter().map(|s| s.to_string()));
    if out.is_empty() { ".".to_string() } else { out.join("/") }
}

/// The page around a rendered body: a title, an inline stylesheet, a
/// footer naming the Markdown it was rendered from. Nothing external.
fn page(title: &str, body: &str, source_rel: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{CSS}
</style>
</head>
<body>
<main>
{body}
</main>
<footer>Rendered from <code>{source}</code>. The Markdown is the source of truth; this page is regenerated from it.</footer>
</body>
</html>
"#,
        title = escape(title),
        body = body,
        source = escape(source_rel),
    )
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

const CSS: &str = r#"
:root { color-scheme: light; }
body { margin: 0; padding: 2rem 1rem 4rem; background: #fcfcfa; color: #1e1e1e;
       font: 16px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 52rem; margin: 0 auto; }
h1, h2, h3 { line-height: 1.25; margin-top: 1.8em; }
h1 { font-size: 2rem; margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: .3em; }
h2 { font-size: 1.45rem; border-bottom: 1px solid #eee; padding-bottom: .2em; }
h3 { font-size: 1.15rem; }
a { color: #0b57d0; text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre { font: 0.92em/1.45 Menlo, Consolas, "Liberation Mono", monospace; }
code { background: #f1f1ee; padding: .1em .3em; border-radius: 3px; }
pre { background: #f1f1ee; padding: .8em 1em; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 1em 0; display: block; overflow-x: auto; }
th, td { border: 1px solid #ddd; padding: .35em .6em; text-align: left; vertical-align: top; }
th { background: #f4f4f1; }
img { max-width: 100%; height: auto; }
blockquote { margin: 1em 0; padding: .2em 1em; border-left: 4px solid #ddd; color: #444; }
footer { max-width: 52rem; margin: 3rem auto 0; padding-top: 1rem; border-top: 1px solid #ddd;
         color: #666; font-size: .9em; }
"#;

fn resolve(path_str: &str, workspace: &Path) -> PathBuf {
    let p = PathBuf::from(path_str);
    if p.is_absolute() { p } else { workspace.join(p) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The rendered tree mirrors the Markdown tree and every link is
    /// rewritten relative to its page**: a Markdown link becomes a link
    /// to the page, an asset link points back at the real file, external
    /// links and anchors are untouched, nothing external is fetched, and
    /// the rendered tree is never a source of itself.
    #[test]
    fn the_mirror_follows_the_markdown_and_links_resolve() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("docs/html/docs")).unwrap();
        std::fs::write(root.join("README.md"), "# The Set\n\nSee [made](docs/how.md), the [licence](LICENSE.md#terms) and [out](https://example.org/x.md).\n\n![diagram](docs/pic.png)\n\nBack to [top](#the-set).\n").unwrap();
        std::fs::write(root.join("LICENSE.md"), "# Terms\n\nsome terms\n").unwrap();
        std::fs::write(root.join("NOTES.md"), "# not payload\n").unwrap();
        std::fs::write(root.join("docs/how.md"), "# How\n\n[home](../README.md) · [ref](dataset.md) · [licence](../LICENSE.md)\n\n![pic](pic.png) ![base](../profiles/base/x.png)\n").unwrap();
        std::fs::write(root.join("docs/dataset.md"), "reference without a heading\n").unwrap();
        std::fs::write(root.join("docs/pic.png"), b"png").unwrap();
        std::fs::write(root.join("docs/html/docs/stale.html"), "<p>old</p>").unwrap();
        std::fs::write(root.join("docs/html/README.md"), "# never a source\n").unwrap();

        let sources = markdown_sources(root, DEFAULT_OUTPUT);
        assert_eq!(
            sources,
            vec![PathBuf::from("LICENSE.md"), PathBuf::from("README.md"), PathBuf::from("docs/dataset.md"), PathBuf::from("docs/how.md")],
            "root static payload and docs/, never the rendered tree or a stray root file: {sources:?}"
        );
        let pages = render_tree(root, DEFAULT_OUTPUT).unwrap();
        assert_eq!(pages.iter().map(|p| slashed(&p.page)).collect::<Vec<_>>(), vec!["docs/html/LICENSE.html", "docs/html/README.html", "docs/html/docs/dataset.html", "docs/html/docs/how.html"]);

        let readme = std::fs::read_to_string(root.join("docs/html/README.html")).unwrap();
        assert!(readme.contains("<title>The Set</title>"), "{readme}");
        assert!(readme.contains(r#"href="docs/how.html""#), "{readme}");
        assert!(readme.contains(r#"href="LICENSE.html#terms""#), "{readme}");
        assert!(readme.contains(r#"href="https://example.org/x.md""#), "{readme}");
        assert!(readme.contains(r#"src="../pic.png""#), "{readme}");
        assert!(readme.contains(r##"href="#the-set""##), "{readme}");
        assert!(readme.contains("Rendered from <code>README.md</code>"), "{readme}");
        assert!(!readme.contains("<script") && !readme.contains("<link"), "self-contained: {readme}");

        let how = std::fs::read_to_string(root.join("docs/html/docs/how.html")).unwrap();
        assert!(how.contains(r#"href="../README.html""#), "{how}");
        assert!(how.contains(r#"href="dataset.html""#), "{how}");
        assert!(how.contains(r#"href="../LICENSE.html""#), "{how}");
        assert!(how.contains(r#"src="../../pic.png""#), "{how}");
        assert!(how.contains(r#"src="../../../profiles/base/x.png""#), "{how}");
        let reference = std::fs::read_to_string(root.join("docs/html/docs/dataset.html")).unwrap();
        assert!(reference.contains("<title>dataset</title>"), "no heading: the file stem names the page: {reference}");
    }

    /// **The tree answers for its own completeness**: absent, partial
    /// when a page is missing, complete when every page is there.
    #[test]
    fn the_tree_reports_its_state() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("docs")).unwrap();
        std::fs::write(root.join("README.md"), "# r\n").unwrap();
        std::fs::write(root.join("docs/a.md"), "# a\n").unwrap();
        let mut o = Options::new();
        o.set("source", root.to_string_lossy().as_ref());
        o.set("output", DEFAULT_OUTPUT);
        let op = RenderDocsOp;
        let out = root.join(DEFAULT_OUTPUT);
        assert!(matches!(op.check_artifact(&out, &o), ArtifactState::Absent));
        render_tree(root, DEFAULT_OUTPUT).unwrap();
        assert!(matches!(op.check_artifact(&out, &o), ArtifactState::Complete));
        std::fs::remove_file(root.join("docs/html/docs/a.html")).unwrap();
        assert!(matches!(op.check_artifact(&out, &o), ArtifactState::Partial));
    }

    #[test]
    fn paths_normalize_and_relativize() {
        assert_eq!(normalize("docs/../LICENSE.md"), "LICENSE.md");
        assert_eq!(normalize("./docs/./a/../b.md"), "docs/b.md");
        assert_eq!(relative("docs/html", "docs/pic.png"), "../pic.png");
        assert_eq!(relative("docs/html/docs", "docs/html/docs/x.html"), "x.html");
        assert_eq!(relative("docs/html/docs", "LICENSE.md"), "../../../LICENSE.md");
        assert_eq!(relative("", "docs/a.md"), "docs/a.md");
        assert_eq!(relative("docs/html", "docs/html"), ".");
    }
}
