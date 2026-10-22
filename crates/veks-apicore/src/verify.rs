// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! What verifying a profile reports: the checks, their failures and the
//! report, and the normalization measurement (V8).
//!
//! [`TestDataView::verify`](crate::TestDataView::verify) proves a
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

use std::fmt;

pub use crate::progress::{NoProgress, ProgressSink as VerifyProgress};

/// One of the checks [`TestDataView::verify`](crate::TestDataView::verify) makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Check {
    /// Every facet opens.
    V1Opens,
    /// The queries have the base's dimension.
    V2QueryDimension,
    /// A ground truth has one row per query.
    V3TruthRows,
    /// A ground truth's rows are not empty.
    V4TruthWidth,
    /// Every ground-truth index names a base vector.
    V5TruthIndices,
    /// A ground truth's distances have its rows and width.
    V6Distances,
    /// A declared count is the facet's.
    V7DeclaredCount,
    /// A declared `is_normalized` is what a sample of the base shows.
    V8Normalized,
    /// Every cached chunk still hashes to what it was verified or
    /// recorded as when it landed (opt-in: it reads every cached byte).
    V9CachedChunks,
}

impl Check {
    /// The check's code: `V1` … `V7`.
    pub fn code(self) -> &'static str {
        match self {
            Check::V1Opens => "V1",
            Check::V2QueryDimension => "V2",
            Check::V3TruthRows => "V3",
            Check::V4TruthWidth => "V4",
            Check::V5TruthIndices => "V5",
            Check::V6Distances => "V6",
            Check::V7DeclaredCount => "V7",
            Check::V8Normalized => "V8",
            Check::V9CachedChunks => "V9",
        }
    }
}

/// A check that failed: which, on which facet and file, and what it
/// found against what it expected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyFailure {
    /// The check.
    pub check: Check,
    /// The facet it failed on.
    pub facet: String,
    /// The facet's file, as the view resolves it, when it has one.
    pub file: Option<String>,
    /// What was found, as a sentence about the facet: `holds 4 rows
    /// for 3 queries`.
    pub found: String,
}

impl fmt::Display for VerifyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.check.code(), self.facet)?;
        if let Some(file) = &self.file {
            write!(f, " {file}")?;
        }
        write!(f, " {}", self.found)
    }
}

/// What a verification ran and found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerifyReport {
    /// The checks made, in order; a check with nothing to check (no
    /// ground truth, say) is not made.
    pub checked: Vec<Check>,
    /// The checks that failed.
    pub failures: Vec<VerifyFailure>,
}

impl VerifyReport {
    /// Whether every check passed.
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

/// How a verification is run.
#[derive(Debug, Clone, Default)]
pub struct VerifyOptions {
    /// Skip a facet whose local file is not there, rather than fail V1
    /// on it: for checking a dataset still being built, whose later
    /// facets are not written yet.
    pub skip_absent: bool,
    /// Also check every cached chunk of a remote facet against the
    /// hash it was verified or recorded as (V9): local corruption of
    /// the cache. Reads every cached byte, so it is asked for.
    pub cached_chunks: bool,
}

/// A fact about a facet measured from its bytes, with how it was
/// measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured<T> {
    /// What the measurement found.
    pub value: T,
    /// How many records it read.
    pub sample: usize,
    /// The largest distance from the ideal among them (for unit length,
    /// `|‖v‖ − 1|`).
    pub worst_deviation: f64,
}

/// Records a normalization sample reads, at most: a stride apart
/// across the whole base.
pub const NORMALIZATION_SAMPLE: usize = 1000;

/// How far from 1 a sampled norm may be and still count as unit length.
pub const UNIT_NORM_TOLERANCE: f64 = 1e-3;

/// What a dataset's base is, as to unit length: the declared value (the
/// dataset's, an overlay's, or one inferred from file names) checked
/// against a sample, or the sample alone when nothing declares it.
#[derive(Debug, Clone, PartialEq)]
pub struct Normalization {
    /// Whether the base is unit length: the declared value when there is
    /// one, even contradicted, else the measured one.
    pub value: bool,
    /// Where the value came from.
    pub source: crate::overlay::AttributeSource,
    /// The sample, recorded or just taken.
    pub measured: Measured<bool>,
    /// Whether the sample disagrees with the declared value.
    pub contradicted: bool,
}
