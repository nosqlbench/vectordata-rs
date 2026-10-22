// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The API's errors: [`Error`](enum@Error) (and [`Result`]) for dataset, catalog
//! and fetch calls; [`TransportError`], a request that failed; and
//! [`InsufficientCacheSpace`]. The reader-level error, which `Error`
//! wraps, is [`IoError`](crate::IoError).

use std::path::PathBuf;

use thiserror::Error;

/// Why a request failed, as far as what a caller does next: retry it,
/// fall back to a local copy, or give up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportErrorKind {
    /// Retrying cannot succeed: a 4xx status other than 408, 425 and
    /// 429, a malformed URL, a refused redirect.
    Permanent,
    /// Retrying might succeed: a reset, a body cut short, a 5xx, 408,
    /// 425 or 429.
    Transient,
    /// The server could not be reached at all — the connection was
    /// refused or timed out, or its certificate was not trusted — so a
    /// local copy, if there is one, is what there is. Retrying might
    /// succeed.
    Unreachable,
    /// Offline mode is on and the request was not made. Decided
    /// locally, so retrying cannot change it.
    Offline,
}

impl TransportErrorKind {
    /// Whether no retry of the request can succeed.
    pub fn is_permanent(self) -> bool {
        matches!(self, TransportErrorKind::Permanent | TransportErrorKind::Offline)
    }
}

/// A request that failed, carrying what a caller decides on: the URL,
/// the HTTP status when the server answered, and whether retrying could
/// succeed (`srd-crate-boundaries.md` PR-6, TR-6). Built only on the
/// error path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportError {
    /// The URL asked for, when there was one.
    pub url: Option<String>,
    /// The HTTP status, when the server answered.
    pub status: Option<u16>,
    /// Whether retrying could succeed.
    pub kind: TransportErrorKind,
    /// What went wrong, as the transport said it.
    pub message: String,
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.kind == TransportErrorKind::Offline {
            // The refusal says what happened and what to do; it was not
            // an HTTP request.
            return f.write_str(&self.message);
        }
        f.write_str("HTTP request failed: ")?;
        if let Some(status) = self.status {
            write!(f, "HTTP {status}: ")?;
        }
        f.write_str(&self.message)?;
        if let Some(url) = &self.url
            && !self.message.contains(url.as_str())
        {
            write!(f, " ({url})")?;
        }
        Ok(())
    }
}

impl std::error::Error for TransportError {}

impl TransportError {
    /// Whether no retry of the request can succeed.
    pub fn is_permanent(&self) -> bool {
        self.kind.is_permanent()
    }

    /// Whether the server could not be reached, so a local copy stands
    /// in for it.
    pub fn is_unreachable(&self) -> bool {
        self.kind == TransportErrorKind::Unreachable
    }

    /// Whether offline mode refused the request.
    pub fn is_offline(&self) -> bool {
        self.kind == TransportErrorKind::Offline
    }
}

/// Raised when the configured cache directory cannot hold a planned
/// download. Carries the three numbers a user needs to act on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsufficientCacheSpace {
    /// The configured cache directory the download would write into.
    pub cache_dir: PathBuf,
    /// Bytes the download still needs to fetch (already-resident
    /// bytes are excluded by the caller).
    pub needed: u64,
    /// Bytes currently free on the cache directory's filesystem.
    pub available: u64,
}

impl std::fmt::Display for InsufficientCacheSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "not enough free space in cache directory {}: need {} ({} B) but only {} ({} B) available. \
             Free up space, or point the cache elsewhere (set `cache_dir` in settings or VECTORDATA_HOME).",
            self.cache_dir.display(),
            fmt_gib(self.needed),
            self.needed,
            fmt_gib(self.available),
            self.available,
        )
    }
}

impl std::error::Error for InsufficientCacheSpace {}

/// Compact binary-unit rendering for capacity messages (GiB/MiB).
/// Local to the cache module so the one error string doesn't reach
/// across into the `datasets` formatter.
fn fmt_gib(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let b = bytes as f64;
    if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else {
        format!("{:.1} MiB", b / MIB)
    }
}

/// The API's error: what every dataset, catalog and fetch call returns.
#[derive(Error, Debug)]
pub enum Error {
    /// A vector I/O operation failed (read, mmap, HTTP fetch).
    #[error("Vector IO error: {0}")]
    VectorIo(#[from] crate::reader::IoError),
    /// The `dataset.yaml` file could not be read from disk or network.
    #[error("Failed to read dataset configuration: {0}")]
    ConfigIo(#[source] std::io::Error),
    /// The `dataset.yaml` content is not valid YAML or does not match the schema.
    #[error("Failed to parse dataset configuration: {0}")]
    ConfigParse(#[from] serde_yaml::Error),
    /// A URL string could not be parsed.
    #[error("Invalid URL: {0}")]
    UrlParse(#[from] url::ParseError),
    /// A request to a remote dataset failed: its status, whether a
    /// retry could succeed, and the URL.
    #[error("{0}")]
    Transport(#[from] TransportError),
    /// A required facet (e.g., `base_vectors`) is not defined in the profile.
    #[error("Required facet not defined: {0}")]
    MissingFacet(String),
    /// A facet that must hold one record per record of another holds
    /// a different number: a quantization column against its base.
    #[error("facet '{facet}' holds {found} records where {expected} were expected, one per record of '{against}'")]
    FacetCardinality {
        /// The facet with the wrong count.
        facet: String,
        /// The facet it must match.
        against: String,
        /// The count it must have.
        expected: usize,
        /// The count it has.
        found: usize,
    },
    /// The facet was opened through a reader for the wrong shape.
    ///
    /// Not a missing capability: the facet is readable, through the
    /// other path. A `metadata_content.slab` holds opaque records and
    /// has no element width, so asking a vector reader for it is asking
    /// a question it cannot answer — and saying "cannot infer element
    /// size" describes the symptom rather than the situation.
    ///
    /// Carries the shape so a caller can branch on it rather than parse
    /// a message; [`crate::dataset::facet::FacetShape`] is also
    /// available up front through `facet_shape()`, which is the way to
    /// avoid the error entirely.
    #[error(
        "facet '{facet}' holds {shape} and was opened as {attempted}; \
         open it with {reader}"
    )]
    WrongFacetShape {
        /// The facet as declared.
        facet: String,
        /// What the facet actually holds.
        shape: crate::dataset::facet::FacetShape,
        /// The shape the caller's reader expects.
        attempted: crate::dataset::facet::FacetShape,
        /// The reader that does open this facet.
        reader: &'static str,
    },
    /// A fetch named facets the profile does not declare. Nothing was
    /// fetched: fetching the facets that do exist and reporting success
    /// would hide the typo.
    #[error(
        "no such facet(s){}: {}; the profile declares: {}",
        .profile.as_deref().map(|p| format!(" in profile '{p}'")).unwrap_or_default(),
        .missing.join(", "),
        .declared.join(", ")
    )]
    UnknownFacets {
        /// The profile, when the fetch spanned several.
        profile: Option<String>,
        /// The names asked for that the profile does not declare, sorted.
        missing: Vec<String>,
        /// Every facet the profile declares, sorted.
        declared: Vec<String>,
    },
    /// A fetch window cannot be resolved for these facets' formats, so
    /// honouring it means fetching them whole, and the request did not
    /// allow that. Refused before anything was fetched.
    #[error(
        "the window cannot be resolved for {}, so honouring it means fetching \
         the whole facet ({bytes} bytes); allow that with \
         FetchRequest::allow_whole_facet (--allow-whole-facet on the command \
         line), or drop the window",
        .facets.iter().map(|f| format!("'{f}'")).collect::<Vec<_>>().join(", ")
    )]
    WindowUnresolvable {
        /// The facets, as `facet` or `profile/facet`.
        facets: Vec<String>,
        /// Their combined size — what allowing it means fetching, less
        /// whatever is already cached. The number the decision turns on.
        bytes: u64,
    },
    /// The cache directory cannot hold what a fetch would download.
    /// Refused before anything was fetched.
    #[error("{0}")]
    InsufficientCacheSpace(#[from] InsufficientCacheSpace),
    /// No configured catalog has a dataset by this name.
    #[error(
        "dataset '{name}' not found{}",
        if .suggestions.is_empty() { String::new() } else { format!("; did you mean {}?", .suggestions.join(", ")) }
    )]
    UnknownDataset {
        /// The name looked up.
        name: String,
        /// Datasets whose names contain it, for a "did you mean".
        suggestions: Vec<String>,
    },
    /// More than one dataset matches this name (case-insensitively).
    ///
    /// Two catalogs that both declare the name are a collision to fix —
    /// narrow the catalogs consulted — not something resolved by
    /// precedence.
    #[error("multiple datasets match '{name}': {}", .matches.join("; "))]
    AmbiguousDataset {
        /// The name looked up.
        name: String,
        /// Every dataset it matches, each with the catalog it came from
        /// and that catalog's location when known:
        /// `ds (catalog 'lab': /data/lab/catalog.json)`.
        matches: Vec<String>,
    },
    /// A dataset spec or profile selector is malformed, or matches no
    /// profile of the dataset.
    #[error("dataset '{dataset}': {message}")]
    Selection {
        /// The dataset the selector was applied to.
        dataset: String,
        /// What is wrong with it.
        message: String,
    },
    /// A profile's facets do not belong together: each check that
    /// failed, what it found and the file
    /// ([`TestDataView::verify`](crate::TestDataView::verify)).
    #[error("{}", .failures.iter().map(|f| f.to_string()).collect::<Vec<_>>().join("; "))]
    Verification {
        /// Every check that failed, in the order they ran.
        failures: Vec<crate::verify::VerifyFailure>,
    },
    /// Catch-all for errors that do not fit other variants.
    #[error("{0}")]
    Other(String),
}

/// A specialized Result type for the library.
pub type Result<T> = std::result::Result<T, Error>;
