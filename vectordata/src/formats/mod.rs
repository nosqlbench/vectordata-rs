// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Wire format codecs for metadata and predicate records.
//!
//! - [`mnode`]: Self-describing binary metadata records (MNode/MValue).
//! - [`pnode`]: Binary predicate expression trees (PNode).
//! - [`anode`]: Annotation nodes combining mnode and pnode data.

pub mod mnode;
pub mod pnode;
pub mod anode;
pub mod anode_vernacular;
