// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. [`Location`]: where a dataset, catalog or facet lives.
//!
//! A location is a local path or a URL. Every place the crate joins a
//! relative path onto a base, takes a parent, or decides whether
//! something is remote goes through this type, so a dataset resolves
//! the same facet to the same place whichever surface asks: opening it,
//! listing it, filtering it, or dropping its cache.
//!
//! The resolving rules ([`Location::resolve`]), the same for every base:
//!
//! - **`scheme://…`** stands alone and is kept as written; a `file://`
//!   URI becomes its plain path.
//! - **A leading `/`** is an absolute local path, kept as written, also
//!   under a remote base. A `dataset.yaml` may point a facet at a local
//!   copy.
//! - **Anything else** is relative to the base, read as a directory.
//!   Under a URL it joins by RFC 3986, so `..` segments resolve and
//!   characters are percent-encoded the way the request will send them.
//!   Under a path it joins as [`Path::join`] does.
//!
//! Absolute references are kept as written because resolved strings
//! become cache identities: a facet's cache path is its resolved string
//! minus the dataset's home, and the home is recorded in the dataset's
//! `origin.json`. Re-spelling either would move or orphan cached bytes.
//! For the same reason the string helpers here change nothing but what
//! they say: [`dir_form`] adds a trailing `/`, [`parent_form`] cuts the
//! last segment, and [`concat_form`] is the plain concatenation that
//! `knn_entries.yaml` catalogs resolve facets by.

use std::fmt;
use std::path::{Path, PathBuf};

use url::Url;

use crate::{Error, Result};

/// Where a dataset, catalog or facet lives: a local path or a URL.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Location {
    /// A local filesystem path. `file://` URIs are read into this form.
    Path(PathBuf),
    /// A URL whose scheme is not `file`: `http`, `https`, `s3`, or a
    /// scheme the transport does not speak (kept, so opening it fails
    /// with a precise error rather than being taken for a path).
    Url(Url),
}

impl Location {
    /// Read a location from text.
    ///
    /// `scheme://…` is a URL (`file://` becomes a path); anything else,
    /// including a Windows drive path, is a local path. Fails only for
    /// text that names a scheme but is not a valid URL.
    pub fn parse(text: &str) -> Result<Self> {
        if text.starts_with("file://") {
            return Ok(Location::Path(PathBuf::from(file_uri_path(text).as_ref())));
        }
        if has_scheme(text) {
            return Url::parse(text)
                .map(Location::Url)
                .map_err(|e| Error::Other(format!("invalid location '{text}': {e}")));
        }
        Ok(Location::Path(PathBuf::from(text)))
    }

    /// A local path.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Location::Path(path.into())
    }

    /// Whether the transport fetches this location over the network:
    /// an `http`, `https` or `s3` URL.
    pub fn is_remote(&self) -> bool {
        match self {
            Location::Url(url) => is_remote_url(url.as_str()),
            Location::Path(_) => false,
        }
    }

    /// The local path, when this is one.
    pub fn as_path(&self) -> Option<&Path> {
        match self {
            Location::Path(path) => Some(path),
            Location::Url(_) => None,
        }
    }

    /// The URL, when this is one.
    pub fn as_url(&self) -> Option<&Url> {
        match self {
            Location::Url(url) => Some(url),
            Location::Path(_) => None,
        }
    }

    /// This location read as a directory: a URL gains a trailing `/`
    /// on its path, so joining onto it keeps its last segment.
    pub fn as_dir(&self) -> Self {
        match self {
            Location::Url(url) if !url.path().ends_with('/') => {
                let mut dir = url.clone();
                dir.set_path(&format!("{}/", url.path()));
                Location::Url(dir)
            }
            other => other.clone(),
        }
    }

    /// `reference` resolved against this location, read as a
    /// directory, as the string to open. Absolute references are kept
    /// as written. See the [module docs](self) for the rules.
    pub fn resolve(&self, reference: &str) -> Result<String> {
        if reference.starts_with("file://") {
            return Ok(file_uri_path(reference).into_owned());
        }
        if reference.starts_with('/') || has_scheme(reference) {
            return Ok(reference.to_string());
        }
        match self.as_dir() {
            Location::Url(base) => base
                .join(reference)
                .map(|url| url.to_string())
                .map_err(|e| Error::Other(format!("cannot join '{reference}' onto {base}: {e}"))),
            Location::Path(base) => Ok(base.join(reference).to_string_lossy().into_owned()),
        }
    }

    /// [`resolve`](Self::resolve), read back as a location.
    pub fn join(&self, reference: &str) -> Result<Self> {
        Self::parse(&self.resolve(reference)?)
    }

    /// The directory holding this location, or `None` at a root.
    /// For a URL, the result ends in `/`.
    pub fn parent(&self) -> Option<Self> {
        match self {
            Location::Url(url) => {
                let trimmed = url.path().trim_end_matches('/');
                let cut = trimmed.rfind('/')?;
                let mut parent = url.clone();
                parent.set_path(&trimmed[..=cut]);
                parent.set_query(None);
                parent.set_fragment(None);
                Some(Location::Url(parent))
            }
            Location::Path(path) => path.parent().map(|p| Location::Path(p.to_path_buf())),
        }
    }

    /// The last non-empty segment: a file or directory name.
    pub fn last_segment(&self) -> Option<&str> {
        match self {
            Location::Url(url) => url.path_segments()?.rfind(|s| !s.is_empty()),
            Location::Path(path) => path.file_name()?.to_str(),
        }
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Location::Url(url) => f.write_str(url.as_str()),
            Location::Path(path) => write!(f, "{}", path.display()),
        }
    }
}

impl std::str::FromStr for Location {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        Self::parse(text)
    }
}

/// `text` with a trailing `/` added when missing, and no other change.
///
/// For strings that identify something, a dataset's home URL above
/// all: it keys the cache directory's `origin.json`, so it is compared
/// as written and must not be re-spelled by URL parsing.
pub fn dir_form(text: &str) -> String {
    if text.ends_with('/') {
        text.to_string()
    } else {
        format!("{text}/")
    }
}

/// The directory part of `text`, ending in `/`, with no other change:
/// `https://h/a/catalog.yaml` gives `https://h/a/`. Text with no `/`
/// gives the empty string. On Windows a `\` separates a local path's
/// parts too, so `C:\data\catalog.yaml` gives `C:\data/`: the form
/// [`dir_form`] gives the directory's own path.
pub fn parent_form(text: &str) -> String {
    match text.rfind(is_path_separator) {
        Some(cut) => format!("{}/", &text[..cut]),
        None => String::new(),
    }
}

/// Whether `c` separates the parts of a location: `/` everywhere, and
/// `\` in a Windows path.
pub fn is_path_separator(c: char) -> bool {
    c == '/' || (cfg!(windows) && c == '\\')
}

/// `base` (a directory, ending in `/`) followed by `relative`, as
/// written: no parsing, no `..` resolution, no encoding.
///
/// This is how `knn_entries.yaml` catalogs resolve facets (the jvector
/// rule), and the strings it gives are the cache identities those
/// datasets were recorded under, so they must stay as they are. An
/// absolute `relative` (a scheme, or a leading `/`) stands alone.
pub fn concat_form(base: &str, relative: &str) -> String {
    if relative.starts_with('/') || has_scheme(relative) {
        return relative.to_string();
    }
    format!("{base}{relative}")
}

/// True when `text` starts with `<scheme>://`, the scheme being a letter
/// followed by letters, digits, `+`, `-` or `.` (RFC 3986). A Windows
/// drive path (`C:\x`, `C:/x`) has no `//` after its colon, so it is
/// not taken for a URL.
pub fn has_scheme(text: &str) -> bool {
    let Some(end) = text.find("://") else { return false };
    let scheme = &text[..end];
    let mut chars = scheme.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

/// The path a `file://` URI names: `file:///abs` and
/// `file://localhost/abs` give `/abs`, and on Windows `file:///C:/x`
/// gives `C:\x`, all decoded as a URI is. A URI the platform cannot
/// read as a local path (another host) keeps the path after the host;
/// `file://abs` gives `/abs`. Text without the prefix passes through.
pub fn file_uri_path(text: &str) -> std::borrow::Cow<'_, str> {
    let Some(rest) = text.strip_prefix("file://") else {
        return std::borrow::Cow::Borrowed(text);
    };
    // `file://name` with no path names `/name`, not a host, on every
    // platform; read as a URL, Windows would make it the share `\\name\`.
    if !rest.is_empty() && !rest.contains('/') {
        return std::borrow::Cow::Owned(format!("/{rest}"));
    }
    if let Some(path) = Url::parse(text).ok().and_then(|url| url.to_file_path().ok()) {
        return std::borrow::Cow::Owned(path.to_string_lossy().into_owned());
    }
    if rest.starts_with('/') {
        return std::borrow::Cow::Borrowed(rest);
    }
    match rest.find('/') {
        Some(slash) => std::borrow::Cow::Borrowed(&rest[slash..]),
        None => std::borrow::Cow::Owned(format!("/{rest}")),
    }
}

/// Whether `s` names a remote resource the transport fetches: an
/// `http://`, `https://` or `s3://` URL. `file://` and paths are local.
#[inline]
pub fn is_remote_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://") || s.starts_with("s3://")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(text: &str) -> Location {
        Location::parse(text).unwrap()
    }

    #[test]
    fn text_is_read_as_a_url_or_a_path() {
        assert!(matches!(loc("https://h/a/"), Location::Url(_)));
        assert!(matches!(loc("s3://bucket/key"), Location::Url(_)));
        assert!(matches!(loc("gs://bucket/key"), Location::Url(_)));
        assert_eq!(loc("file:///abs/x"), Location::from_path("/abs/x"));
        // A host is a UNC share on Windows; elsewhere it is not part of the path.
        #[cfg(windows)]
        assert_eq!(loc("file://host/abs/x"), Location::from_path(r"\\host\abs\x"));
        #[cfg(not(windows))]
        assert_eq!(loc("file://host/abs/x"), Location::from_path("/abs/x"));
        assert_eq!(loc("file://abs"), Location::from_path("/abs"));
        assert_eq!(loc("/abs/x"), Location::from_path("/abs/x"));
        assert_eq!(loc("rel/x"), Location::from_path("rel/x"));
        assert_eq!(loc(r"C:\data\x"), Location::from_path(r"C:\data\x"));
        assert_eq!(loc("C:/data/x"), Location::from_path("C:/data/x"));
    }

    #[test]
    fn only_http_https_and_s3_are_remote() {
        assert!(loc("http://h/").is_remote());
        assert!(loc("https://h/").is_remote());
        assert!(loc("s3://b/k").is_remote());
        assert!(!loc("gs://b/k").is_remote());
        assert!(!loc("/abs").is_remote());
        assert!(!loc("file:///abs").is_remote());
    }

    #[test]
    fn absolute_references_are_kept_as_written() {
        let remote = loc("https://h/cat/ds/");
        assert_eq!(remote.resolve("https://H/a%7e/x").unwrap(), "https://H/a%7e/x");
        assert_eq!(remote.resolve("/local/x").unwrap(), "/local/x");
        assert_eq!(remote.resolve("file:///local/x").unwrap(), "/local/x");
        assert_eq!(remote.resolve("gs://b/k").unwrap(), "gs://b/k");
    }

    #[test]
    fn string_forms_change_only_what_they_say() {
        assert_eq!(parent_form("https://H/a/catalog.yaml"), "https://H/a/");
        assert_eq!(parent_form("/data/cat/knn_entries.yaml"), "/data/cat/");
        assert_eq!(parent_form("catalog.yaml"), "");
        assert_eq!(concat_form("https://H/a/", "x/../b c.fvec"), "https://H/a/x/../b c.fvec");
        assert_eq!(concat_form("https://h/a/", "s3://o/k"), "s3://o/k");
        assert_eq!(concat_form("https://h/a/", "/abs"), "/abs");
    }

    #[test]
    fn joining_follows_one_rule_for_every_base() {
        let remote = loc("https://h/cat/ds");
        // The base is read as a directory, slash or not.
        assert_eq!(remote.join("base.fvec").unwrap().to_string(), "https://h/cat/ds/base.fvec");
        // `..` resolves; the request would not send it literally.
        assert_eq!(remote.join("../shared/b.fvec").unwrap().to_string(), "https://h/cat/shared/b.fvec");
        // Characters are encoded the way the request sends them.
        assert_eq!(remote.join("a b.fvec").unwrap().to_string(), "https://h/cat/ds/a%20b.fvec");
        // A leading `/` is a local path, even under a remote base.
        assert_eq!(remote.join("/local/b.fvec").unwrap(), Location::from_path("/local/b.fvec"));
        // A scheme replaces the base.
        assert_eq!(remote.join("s3://other/k").unwrap().to_string(), "s3://other/k");
        assert_eq!(remote.join("file:///x").unwrap(), Location::from_path("/x"));

        let s3 = loc("s3://bucket/ds/");
        assert_eq!(s3.join("profiles/base.fvec").unwrap().to_string(), "s3://bucket/ds/profiles/base.fvec");

        let local = loc("/data/ds");
        assert_eq!(local.join("base.fvec").unwrap(), Location::from_path("/data/ds/base.fvec"));
        assert_eq!(local.join("/elsewhere/b").unwrap(), Location::from_path("/elsewhere/b"));
    }

    #[test]
    fn parents_and_segments() {
        assert_eq!(loc("https://h/a/b.yaml").parent().unwrap().to_string(), "https://h/a/");
        assert_eq!(loc("https://h/a/b/").parent().unwrap().to_string(), "https://h/a/");
        assert_eq!(loc("https://h/").parent(), None);
        assert_eq!(loc("/a/b.yaml").parent().unwrap(), Location::from_path("/a"));
        assert_eq!(loc("https://h/a/ds/").last_segment(), Some("ds"));
        assert_eq!(loc("s3://bucket/a/ds").last_segment(), Some("ds"));
        assert_eq!(loc("/a/ds").last_segment(), Some("ds"));
        assert_eq!(loc("https://h/").last_segment(), None);
    }

    #[test]
    fn dir_form_only_adds_the_slash() {
        assert_eq!(dir_form("https://H/a%7Eb"), "https://H/a%7Eb/");
        assert_eq!(dir_form("https://h/a/"), "https://h/a/");
        assert_eq!(dir_form("/data/ds"), "/data/ds/");
    }
}
