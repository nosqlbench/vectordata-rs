// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Bring a profile's facets into the local cache — all of
//! them or a chosen few, optionally only a window of records — with
//! live progress, in one call.
//!
//! This is the module to reach for when the words in your head are
//! *fetch*, *download*, *precache*, *prebuffer* or *prefetch*. The call
//! is [`TestDataView::fetch`] on a profile
//! view, [`TestDataGroup::fetch`](crate::TestDataGroup::fetch) across
//! several profiles, or [`Catalog::fetch`](crate::catalog::resolver::Catalog::fetch)
//! straight from a `name:selector` spec. `vectordata datasets precache`
//! is this module with a [`TextMeter`] attached.
//!
//! ```no_run
//! use vectordata::catalog::resolver::Catalog;
//! use vectordata::catalog::sources::CatalogSources;
//! use vectordata::fetch::{FetchRequest, TextMeter};
//!
//! # fn main() -> vectordata::Result<()> {
//! let catalog = Catalog::of(&CatalogSources::new().configure_default());
//! let view = catalog.open_profile("my-dataset", "default")?;
//!
//! // Base and query vectors only, with the same meter the CLI draws.
//! let request = FetchRequest::facets(["base_vectors", "query_vectors"]);
//! let report = view.fetch(&request, &mut TextMeter::stderr("Fetch"))?;
//! println!("{} bytes fetched", report.bytes_fetched());
//! # Ok(()) }
//! ```
//!
//! A fetch runs in two steps, and either can be called on its own:
//!
//! 1. **Plan** ([`TestDataView::plan_fetch`]).
//!    Each facet is opened and its window resolved to the byte ranges
//!    and chunks it needs, net of what is already resident. Nothing is
//!    downloaded, so the plan is where a caller learns the cost.
//! 2. **Execute** ([`FetchPlan::execute`]). The plan is checked — a
//!    window that cannot be honoured without fetching a facet whole is
//!    refused unless the request allowed it, and the cache must have
//!    room — and then each facet is fetched, reporting through the
//!    [`FetchProgress`] sink the caller passed.
//!
//! A sink is anything that implements [`FetchProgress`]: [`Silent`],
//! [`TextMeter`], or a closure taking `&FetchEvent<'_>`. The library
//! never writes to the terminal on its own; whatever a fetch shows, the
//! caller chose.

use std::fmt;
use std::time::{Duration, Instant};

use crate::dataset::source::DSWindow;
use crate::view::{FacetStorage, PrefetchPlan, WholeFacetFallback, facet_declared_window};
use crate::{Error, Result, TestDataView};

pub(crate) mod meter;
pub use meter::TextMeter;

// ─── Request ─────────────────────────────────────────────────────────

/// What to fetch: which facets, which records, and whether a window
/// that cannot be honoured may fall back to the whole facet.
///
/// ```
/// use vectordata::fetch::FetchRequest;
/// use vectordata::dataset::source::parse_window;
///
/// // Everything the profile declares, each facet against its own window.
/// let all = FetchRequest::all();
///
/// // Two facets, records 0..1000 of each.
/// let some = FetchRequest::facets(["base_vectors", "metadata_content"])
///     .window(parse_window("0..1000").unwrap());
/// # let _ = (all, some);
/// ```
#[derive(Debug, Clone, Default)]
pub struct FetchRequest {
    facets: Vec<String>,
    window: Option<DSWindow>,
    fallback: WholeFacetFallback,
}

impl FetchRequest {
    /// Every facet that holds data, each against the window it declares
    /// for itself. What `vectordata datasets precache` does without
    /// `--facet` or `--window`.
    pub fn all() -> Self {
        FetchRequest::default()
    }

    /// The named facets only, in the order given; a name repeated is
    /// fetched once. A standard alias (`base`, `gt`, `metadata_indices`)
    /// names the facet it stands for. Naming a facet the profile does
    /// not declare fails
    /// the plan with [`Error::UnknownFacets`] rather than fetching the
    /// rest and reporting success. An empty list means every facet, as
    /// [`FetchRequest::all`].
    pub fn facets<I, S>(facets: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut seen = std::collections::HashSet::new();
        FetchRequest {
            facets: facets
                .into_iter()
                .map(Into::into)
                .filter(|f: &String| seen.insert(f.clone()))
                .collect(),
            ..FetchRequest::default()
        }
    }

    /// Fetch only `window`, in record ordinals, of every selected facet,
    /// in place of each facet's declared window.
    ///
    /// The unit of fetch is a chunk, so a small window can still be a
    /// large download; the plan says how large before anything moves.
    /// Parse the text form with
    /// [`parse_window`](crate::dataset::source::parse_window).
    pub fn window(mut self, window: DSWindow) -> Self {
        self.window = Some(window);
        self
    }

    /// Accept fetching a whole facet when its window cannot be resolved
    /// for its format — parquet, or a variable-length file with no
    /// published offset index. Refused by default: asking for a window
    /// and silently receiving the whole facet is the surprise windowed
    /// fetch exists to prevent.
    pub fn allow_whole_facet(self) -> Self {
        self.fallback(WholeFacetFallback::Allow)
    }

    /// Set the whole-facet fallback explicitly.
    pub fn fallback(mut self, fallback: WholeFacetFallback) -> Self {
        self.fallback = fallback;
        self
    }

    /// The facets named, or empty for every facet.
    pub fn facet_names(&self) -> &[String] {
        &self.facets
    }

    /// The window that overrides each facet's own, if one was set.
    pub fn record_window(&self) -> Option<&DSWindow> {
        self.window.as_ref()
    }

    /// Whether an unresolvable window may fall back to the whole facet.
    pub fn whole_facet_fallback(&self) -> WholeFacetFallback {
        self.fallback
    }
}

// ─── Identity ────────────────────────────────────────────────────────

/// Which facet an event or a report row is about.
///
/// `profile` is set when the fetch spans profiles — a
/// [`TestDataGroup::fetch`](crate::TestDataGroup::fetch) — and `None`
/// for a fetch on one view, which has no profile name to give.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FacetId {
    /// The profile the facet belongs to, when the fetch spans several.
    pub profile: Option<String>,
    /// The facet's name, as the profile declares it.
    pub facet: String,
}

impl FacetId {
    fn new(profile: Option<&str>, facet: &str) -> Self {
        FacetId {
            profile: profile.map(str::to_string),
            facet: facet.to_string(),
        }
    }
}

/// `profile/facet` when the profile is known, else `facet`.
impl fmt::Display for FacetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.profile {
            Some(p) => write!(f, "{p}/{}", self.facet),
            None => f.write_str(&self.facet),
        }
    }
}

// ─── Progress ────────────────────────────────────────────────────────

/// Something a fetch is doing, as it does it.
///
/// Events arrive in this order: a `PlanBegin`/`PlanEnd` pair per facet
/// while planning; then `Begin` once; then per facet `FacetBegin`, any
/// number of `Progress`, and `FacetEnd`; then `End`. `Failed` replaces
/// whatever would have come next when something goes wrong.
#[derive(Debug)]
#[non_exhaustive]
pub enum FetchEvent<'a> {
    /// About to open and plan facet `index` of `count` (1-based).
    ///
    /// Opening is the slow part of planning: on a sharded remote facet
    /// it fetches one merkle reference per shard, and for a slab its
    /// offset index. A sink that says nothing until planning ends shows
    /// a blank screen for as long as that takes.
    PlanBegin {
        /// Position of this facet in the plan, from 1.
        index: usize,
        /// Facets being planned.
        count: usize,
        /// The facet.
        facet: &'a FacetId,
        /// Files the facet spans, for a series; `None` for one file.
        files: Option<usize>,
    },
    /// Facet planned.
    PlanEnd {
        /// The facet.
        facet: &'a FacetId,
        /// What fetching it will cost.
        plan: &'a PrefetchPlan,
    },
    /// The plan passed its checks and fetching starts.
    Begin {
        /// Facets the run will visit.
        facets: usize,
        /// Bytes the whole run will fetch, net of resident chunks.
        bytes: u64,
    },
    /// Fetching starts on facet `index` of `count` (1-based).
    FacetBegin {
        /// Position of this facet in the run, from 1.
        index: usize,
        /// Facets in the run.
        count: usize,
        /// The facet.
        facet: &'a FacetId,
        /// Bytes this facet will fetch, net of resident chunks.
        bytes: u64,
        /// Chunks this facet will fetch; zero for local storage and for
        /// a whole-facet fetch, whose chunking the plan does not know.
        chunks: u32,
        /// Everything the facet needs is already in the cache.
        resident: bool,
    },
    /// Bytes arrived for the current facet. Cumulative for the facet
    /// and never more than its planned `bytes`.
    Progress {
        /// The facet.
        facet: &'a FacetId,
        /// Bytes fetched for this facet so far.
        bytes: u64,
        /// Bytes this facet will fetch in all.
        total: u64,
        /// Chunks fetched and verified for this facet so far.
        chunks: u32,
    },
    /// The current facet is resident.
    FacetEnd {
        /// The facet.
        facet: &'a FacetId,
        /// Bytes the transport moved for it.
        bytes: u64,
    },
    /// Every facet is resident.
    End {
        /// What the run did.
        report: &'a FetchReport,
    },
    /// The fetch stopped. `facet` is the facet being planned or fetched
    /// at the time, if any; the same error is returned to the caller.
    Failed {
        /// The facet in progress when it failed.
        facet: Option<&'a FacetId>,
        /// What went wrong.
        error: &'a Error,
    },
}

/// Where a fetch reports what it is doing.
///
/// Implemented by [`Silent`], [`TextMeter`], and every
/// `FnMut(&FetchEvent<'_>)`, so a closure is a sink:
///
/// ```no_run
/// # use vectordata::fetch::{FetchEvent, FetchRequest};
/// # fn demo(view: &dyn vectordata::TestDataView) -> vectordata::Result<()> {
/// view.fetch(&FetchRequest::all(), &mut |event: &FetchEvent<'_>| {
///     if let FetchEvent::Progress { facet, bytes, total, .. } = event {
///         eprintln!("{facet}: {bytes}/{total}");
///     }
/// })?;
/// # Ok(()) }
/// ```
pub trait FetchProgress {
    /// Hear one event. Called on the fetching thread, between transport
    /// callbacks, so it should return promptly.
    fn on_event(&mut self, event: &FetchEvent<'_>);
}

impl<F: FnMut(&FetchEvent<'_>)> FetchProgress for F {
    fn on_event(&mut self, event: &FetchEvent<'_>) {
        self(event)
    }
}

/// A sink that ignores every event.
#[derive(Debug, Clone, Copy, Default)]
pub struct Silent;

impl FetchProgress for Silent {
    fn on_event(&mut self, _event: &FetchEvent<'_>) {}
}

// ─── Plan ────────────────────────────────────────────────────────────

/// One facet of a [`FetchPlan`]: what it will fetch, and the open
/// handle the fetch will use.
///
/// Holding the handle is what makes planning and fetching load a
/// variable-length facet's offset index once between them instead of
/// twice.
pub struct PlannedFacet {
    id: FacetId,
    window: DSWindow,
    plan: PrefetchPlan,
    storage: FacetStorage,
    upstream_checked: bool,
}

impl PlannedFacet {
    /// Which facet this is.
    pub fn id(&self) -> &FacetId {
        &self.id
    }

    /// The record window it was planned against: the request's, or the
    /// facet's own declared window. Empty means the whole facet.
    pub fn window(&self) -> &DSWindow {
        &self.window
    }

    /// The byte ranges, chunks and cost the window resolved to.
    pub fn plan(&self) -> &PrefetchPlan {
        &self.plan
    }

    /// Whether the upstream was reached to confirm a complete local
    /// copy is current. `false` means it was unreachable and the copy is
    /// used as cached — the offline case.
    pub fn upstream_checked(&self) -> bool {
        self.upstream_checked
    }
}

impl fmt::Debug for PlannedFacet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlannedFacet")
            .field("id", &self.id)
            .field("window", &self.window)
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

/// A fetch, planned and not yet run.
///
/// Returned by [`TestDataView::plan_fetch`]
/// and [`TestDataGroup::plan_fetch`](crate::TestDataGroup::plan_fetch).
/// Read what it will cost, then [`execute`](Self::execute) it or drop
/// it; dropping a plan fetches nothing.
#[derive(Debug)]
pub struct FetchPlan {
    facets: Vec<PlannedFacet>,
    fallback: WholeFacetFallback,
}

impl FetchPlan {
    /// The facets, in the order they will be fetched.
    pub fn facets(&self) -> &[PlannedFacet] {
        &self.facets
    }

    /// Whether the plan has no facets at all.
    pub fn is_empty(&self) -> bool {
        self.facets.is_empty()
    }

    /// Bytes the run will fetch, net of resident chunks. A facet whose
    /// window cannot be resolved counts at its full size, since that is
    /// what fetching it would cost.
    pub fn bytes_to_fetch(&self) -> u64 {
        self.facets.iter().map(|f| f.plan.bytes_to_fetch()).sum()
    }

    /// Whether every facet is already resident, so executing would move
    /// nothing.
    pub fn is_resident(&self) -> bool {
        self.facets.iter().all(|f| f.plan.is_resident())
    }

    /// Facets whose window cannot be resolved for their format, so that
    /// fetching them means fetching them whole.
    pub fn unresolvable(&self) -> Vec<&FacetId> {
        self.facets
            .iter()
            .filter(|f| f.plan.degrades_to_full_download)
            .map(|f| &f.id)
            .collect()
    }

    /// Check the plan the way [`execute`](Self::execute) will, without
    /// fetching: an unresolvable window is refused unless the request
    /// allowed the whole facet ([`Error::WindowUnresolvable`]), and the
    /// cache directory must have room for
    /// [`bytes_to_fetch`](Self::bytes_to_fetch)
    /// ([`Error::InsufficientCacheSpace`]).
    ///
    /// Both are checked for the whole plan before any facet is fetched,
    /// so a refused run leaves nothing half done.
    pub fn check(&self) -> Result<()> {
        if self.fallback == WholeFacetFallback::Refuse {
            let refused: Vec<&PlannedFacet> =
                self.facets.iter().filter(|f| f.plan.degrades_to_full_download).collect();
            if !refused.is_empty() {
                return Err(Error::WindowUnresolvable {
                    facets: refused.iter().map(|f| f.id.to_string()).collect(),
                    bytes: refused.iter().map(|f| f.plan.facet_bytes).sum(),
                });
            }
        }
        crate::cache::ensure_cache_capacity(self.bytes_to_fetch())?;
        Ok(())
    }

    /// Run the plan: [`check`](Self::check) it, then fetch each facet in
    /// order, reporting through `progress`.
    ///
    /// Returning `Ok` means every planned byte is resident and verified
    /// (merkle-verified when the source publishes a `.mref`). A failure
    /// stops the run at the facet that failed; facets already fetched
    /// stay in the cache.
    pub fn execute(self, progress: &mut dyn FetchProgress) -> Result<FetchReport> {
        if let Err(error) = self.check() {
            progress.on_event(&FetchEvent::Failed { facet: None, error: &error });
            return Err(error);
        }
        let started = Instant::now();
        let count = self.facets.len();
        progress.on_event(&FetchEvent::Begin {
            facets: count,
            bytes: self.bytes_to_fetch(),
        });

        let mut rows = Vec::with_capacity(count);
        for (i, planned) in self.facets.into_iter().enumerate() {
            let PlannedFacet { id, plan, storage, upstream_checked, .. } = planned;
            let total = plan.bytes_to_fetch();
            progress.on_event(&FetchEvent::FacetBegin {
                index: i + 1,
                count,
                facet: &id,
                bytes: total,
                chunks: plan.chunks_to_fetch(),
                resident: plan.is_resident(),
            });
            let fetched = fetch_planned(
                &storage,
                &plan,
                &|| false,
                &mut |bytes, chunks| {
                    progress.on_event(&FetchEvent::Progress {
                        facet: &id,
                        bytes: bytes.min(total),
                        total,
                        chunks,
                    })
                },
                &mut |_| {},
            );
            let fetched = match fetched {
                Ok(f) => f,
                Err(e) => {
                    let error = Error::Other(format!("facet '{id}': {e}"));
                    progress.on_event(&FetchEvent::Failed { facet: Some(&id), error: &error });
                    return Err(error);
                }
            };
            // What crossed the network is at most what the plan found
            // missing: the transports also count chunks that were already
            // resident when a range is asked for, which moved nothing.
            let moved = fetched.bytes.min(total);
            progress.on_event(&FetchEvent::FacetEnd {
                facet: &id,
                bytes: moved,
            });
            rows.push(FacetFetch {
                id,
                planned: plan,
                ranges_fetched: fetched.ranges,
                bytes_fetched: moved,
                complete: storage.is_complete(),
                upstream_checked,
            });
        }
        let report = FetchReport {
            facets: rows,
            elapsed: started.elapsed(),
        };
        progress.on_event(&FetchEvent::End { report: &report });
        Ok(report)
    }
}

// ─── Report ──────────────────────────────────────────────────────────

/// What a fetch did, per facet.
#[derive(Debug, Clone)]
pub struct FetchReport {
    /// One row per facet, in the order fetched.
    pub facets: Vec<FacetFetch>,
    /// Wall time from the first byte requested to the last resident.
    pub elapsed: Duration,
}

impl FetchReport {
    /// Bytes the transport moved, across every facet.
    pub fn bytes_fetched(&self) -> u64 {
        self.facets.iter().map(|f| f.bytes_fetched).sum()
    }

    /// The row for `facet`, by name (and profile, for a fetch that
    /// spans profiles: pass `"profile/facet"`).
    pub fn facet(&self, facet: &str) -> Option<&FacetFetch> {
        self.facets
            .iter()
            .find(|f| f.id.facet == facet || f.id.to_string() == facet)
    }
}

/// What a fetch did for one facet.
#[derive(Debug, Clone)]
pub struct FacetFetch {
    /// Which facet.
    pub id: FacetId,
    /// The plan as it stood before fetching.
    pub planned: PrefetchPlan,
    /// Byte ranges handed to the transport. Fewer than the plan's
    /// intervals when nearby ones were merged.
    pub ranges_fetched: usize,
    /// Bytes the transport moved. Zero for a facet that was already
    /// resident or is local.
    pub bytes_fetched: u64,
    /// Whether every byte of the facet is resident, so its readers make
    /// no network requests and serve zero-copy slices
    /// ([`VectorReader::get_slice`](crate::VectorReader::get_slice)).
    /// True after a fetch without a window; after a windowed fetch only
    /// the window's chunks are local.
    pub complete: bool,
    /// Whether the upstream was reached to confirm the copy is current.
    /// `false` when it was unreachable and a complete cached copy was
    /// used as it stands — a fetch on a warmed cache works offline.
    pub upstream_checked: bool,
}

// ─── Planning ────────────────────────────────────────────────────────

/// Plan `request` against one view, appending to `out`. `profile`
/// qualifies the facet ids when the caller is planning across profiles.
pub(crate) fn plan_view<V: TestDataView + ?Sized>(
    view: &V,
    profile: Option<&str>,
    request: &FetchRequest,
    progress: &mut dyn FetchProgress,
    out: &mut Vec<PlannedFacet>,
) -> Result<()> {
    let manifest = view.facet_manifest();
    let names: Vec<String> = if request.facets.is_empty() {
        let mut names: Vec<String> = manifest
            .keys()
            .filter(|name| view.facet_holds_data(name))
            .cloned()
            .collect();
        names.sort();
        names
    } else {
        // A name the profile declares, or a standard alias of one
        // (`base`, `gt`, `metadata_indices`, …), which is how the same
        // facet is spelled in a `dataset.yaml`.
        let resolve = |f: &str| -> Option<String> {
            if manifest.contains_key(f) {
                return Some(f.to_string());
            }
            crate::dataset::facet::resolve_standard_key(f).filter(|k| manifest.contains_key(k.as_str()))
        };
        let mut missing: Vec<String> = request
            .facets
            .iter()
            .filter(|f| resolve(f).is_none())
            .cloned()
            .collect();
        if !missing.is_empty() {
            missing.sort();
            let mut declared: Vec<String> = manifest.keys().cloned().collect();
            declared.sort();
            let error = Error::UnknownFacets {
                profile: profile.map(str::to_string),
                missing,
                declared,
            };
            progress.on_event(&FetchEvent::Failed { facet: None, error: &error });
            return Err(error);
        }
        let mut seen = std::collections::HashSet::new();
        request
            .facets
            .iter()
            .filter_map(|f| resolve(f))
            .filter(|k| seen.insert(k.clone()))
            .collect()
    };

    let count = names.len();
    for (i, name) in names.iter().enumerate() {
        let id = FacetId::new(profile, name);
        let desc = &manifest[name.as_str()];
        progress.on_event(&FetchEvent::PlanBegin {
            index: i + 1,
            count,
            facet: &id,
            files: desc.shard_count.map(|n| n as usize),
        });
        let planned = (|| {
            let window = match &request.window {
                Some(w) => w.clone(),
                None => facet_declared_window(desc)?,
            };
            let storage = view.open_facet_storage(name)?;
            // A complete copy opened from disk without asking the
            // server; a fetch is where it asks. Unreachable is fine —
            // the copy stands — but a changed upstream is stale.
            let upstream_checked = if storage.is_complete() {
                storage.revalidate().map_err(|e| Error::Other(e.to_string()))?
            } else {
                true
            };
            let plan = view.prefetch_plan_on(&storage, name, &window)?;
            Ok::<_, Error>((window, storage, plan, upstream_checked))
        })();
        let (window, storage, plan, upstream_checked) = match planned {
            Ok(p) => p,
            Err(e) => {
                let error = Error::Other(format!("facet '{id}': {e}"));
                progress.on_event(&FetchEvent::Failed { facet: Some(&id), error: &error });
                return Err(error);
            }
        };
        progress.on_event(&FetchEvent::PlanEnd { facet: &id, plan: &plan });
        out.push(PlannedFacet { id, window, plan, storage, upstream_checked });
    }
    Ok(())
}

/// Assemble a plan from planned facets.
pub(crate) fn plan_from(facets: Vec<PlannedFacet>, request: &FetchRequest) -> FetchPlan {
    FetchPlan {
        facets,
        fallback: request.fallback,
    }
}

// ─── Fetching ────────────────────────────────────────────────────────

/// What fetching one facet moved.
pub(crate) struct Fetched {
    /// Ranges handed to the transport.
    pub(crate) ranges: usize,
    /// Bytes the transport reported, summed across ranges and parts.
    pub(crate) bytes: u64,
}

/// Fetch one planned facet: every range of its plan, or the whole facet
/// when the plan degrades to that.
///
/// The one fetch loop behind [`FetchPlan::execute`] and the single-facet
/// prefetch forms, so they cannot drift apart. `on_bytes` hears the
/// facet's cumulative byte count: each range (and each part of a
/// series) reports its own totals, and summing them is done here, once,
/// rather than by every caller. `on_progress` hears cumulative bytes
/// and chunks. `stop` is consulted between ranges; `on_range` hears the
/// count of ranges completed.
pub(crate) fn fetch_planned(
    storage: &FacetStorage,
    plan: &PrefetchPlan,
    stop: &dyn Fn() -> bool,
    on_progress: &mut dyn FnMut(u64, u32),
    on_range: &mut dyn FnMut(usize),
) -> std::io::Result<Fetched> {
    // Bytes and chunks from the parts or ranges already finished, and
    // from the one in flight. A range that was already resident reports
    // itself as done — its full size downloaded, out of zero chunks — so
    // a display can show its size; none of those bytes moved.
    let moved = |p: &crate::transport::DownloadProgress| {
        if p.total_chunks() == 0 { (0, 0) } else { (p.downloaded_bytes(), p.completed_chunks()) }
    };
    let (mut done, mut done_chunks) = (0u64, 0u32);
    if plan.degrades_to_full_download {
        let mut part = None;
        let (mut in_part, mut in_part_chunks) = (0u64, 0u32);
        storage.prebuffer_parts(&mut |i, p| {
            if part != Some(i) {
                done += in_part;
                done_chunks += in_part_chunks;
                (in_part, in_part_chunks) = (0, 0);
                part = Some(i);
            }
            (in_part, in_part_chunks) = moved(p);
            on_progress(done + in_part, done_chunks + in_part_chunks);
        })?;
        on_range(1);
        return Ok(Fetched {
            ranges: 1,
            bytes: done + in_part,
        });
    }
    let mut ranges = 0usize;
    for r in &plan.byte_ranges {
        if stop() {
            break;
        }
        let (mut in_range, mut in_range_chunks) = (0u64, 0u32);
        storage.prebuffer_shard_range(r.shard, r.start, r.end, |p| {
            (in_range, in_range_chunks) = moved(p);
            on_progress(done + in_range, done_chunks + in_range_chunks);
        })?;
        done += in_range;
        done_chunks += in_range_chunks;
        ranges += 1;
        on_range(ranges);
    }
    Ok(Fetched { ranges, bytes: done })
}
