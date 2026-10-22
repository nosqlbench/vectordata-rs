// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. A facet that spans several files (`<stem>__NNNN.<ext>`):
//! the naming, declaration and addressing rules are the API core's
//! ([`veks_apicore::dataset::shards`]), and finding and measuring a
//! series' files on disk is veks-io's ([`veks_io::series`]); both are
//! re-exported here.

pub use veks_apicore::dataset::shards::*;
pub use veks_io::series::{ObservedSeries, discover_shards, observe_series};
