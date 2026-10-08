// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

//! The canonical fetch, in the single-facet shape most tests ask for.

use vectordata::dataset::source::DSWindow;
use vectordata::fetch::{FacetFetch, FetchRequest, Silent};
use vectordata::{TestDataView, WholeFacetFallback};

/// Fetch `window` of one facet through [`TestDataView::fetch`] and
/// return its report row.
pub fn fetch_window(
    view: &dyn TestDataView,
    facet: &str,
    window: &DSWindow,
    fallback: WholeFacetFallback,
) -> vectordata::Result<FacetFetch> {
    let request = FetchRequest::facets([facet])
        .window(window.clone())
        .fallback(fallback);
    view.fetch(&request, &mut Silent).map(|mut r| r.facets.remove(0))
}
