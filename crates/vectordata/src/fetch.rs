// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. Bring a profile's facets into the local cache — all of
//! them or a chosen few, optionally only a window of records — with
//! live progress, in one call.
//!
//! This is the module to reach for when the words in your head are
//! *fetch*, *download*, *precache*, *prebuffer* or *prefetch*. The call
//! is [`TestDataView::fetch`](crate::TestDataView::fetch) on a profile
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
//! 1. **Plan** ([`TestDataView::plan_fetch`](crate::TestDataView::plan_fetch)).
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
//! [`TextMeter`] for a terminal, [`LogMeter`] for a log — whole lines,
//! throttled to a percent step and a minimum interval, with a warning
//! when the fetch stalls — or a closure taking `&FetchEvent<'_>`. The library
//! never writes to the terminal on its own; whatever a fetch shows, the
//! caller chose.

pub use veks_apicore::fetch::*;

pub(crate) mod meter;
pub use meter::{LogMeter, ProgressStep, TextMeter};

/// Fetch a small remote file — a catalog, a list, a manifest — as text,
/// with the read-side auth and the client every dataset fetch uses. In
/// offline mode it is an error saying so; an answer that is not a
/// success is an error naming its status.
///
/// For a dataset's facets use [`fetch`](crate::TestDataView::fetch):
/// this is for the small files a program reads beside them.
pub fn fetch_text(url: &str) -> std::io::Result<String> {
    crate::transport::ensure_online(url)?;
    crate::transport::shared().fetch_text(url)
}

/// Keep `local_path` a copy of the small remote file at `url`: fetched
/// when there is no copy or the server's `Last-Modified` is newer than
/// it, left alone otherwise. `Ok(true)` when it was fetched. The copy's
/// mtime is the server's `Last-Modified`, so the next call compares
/// like with like. Same auth and client as [`fetch_text`]: the client for
/// the URL's origin, so a self-signed origin trusted in `settings.yaml`
/// (`trust_self_signed`) is reachable here too.
pub fn fetch_if_modified(url: &str, local_path: &std::path::Path) -> std::io::Result<bool> {
    crate::transport::ensure_online(url)?;
    crate::transport::shared().fetch_if_modified(url, local_path)
}
