// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Re-exports pipeline schema types from the `dataset` crate.
//!
//! The `upstream` block schema (`PipelineConfig`, `StepDef`, `OnPartial`)
//! is now owned by the `dataset` crate. This module re-exports them for
//! backwards compatibility within veks.

pub use vectordata::dataset::pipeline::*;
