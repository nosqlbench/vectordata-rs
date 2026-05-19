// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `analyze survey` — incremental metadata survey.
//!
//! Two-pass, type-driven, sketch-backed survey of an ANode/MNode slab
//! file. See `docs/sysref/13-metadata-survey.md` for the full design.
//!
//! This module is being built incrementally per §13.13 of the sysref.
//! At the time of this commit, only the bounded-memory streaming
//! sketches that everything else builds on are present. The
//! orchestrator, measure trait, per-type measure suites, and
//! cross-field analyzers land in subsequent steps.

pub mod command;
pub mod crossfield;
pub mod findings;
pub mod governor;
pub mod measure;
pub mod measures;
pub mod orchestrator;
pub mod probes;
pub mod progress;
pub mod sketches;
pub mod template;
pub mod types;

pub use command::{factory, SurveyOp};
pub use findings::{render_findings, Finding, FindingsConfig, FindingsReport, Severity};
pub use governor::{Downscaler, DownscaleAction, GovernorAdapter};
pub use measure::{Measure, MeasureCtx, MeasureKind, MeasureReport};
pub use orchestrator::{
    survey, CrossFieldReport, FieldProfile, SamplingInfo, SourceInfo, SurveyConfig, SurveyReport,
    Warning,
};
pub use progress::{ProgressDriver, SurveyPass, SurveyProgress};
pub use template::{ExplorationProbe, FieldTemplate, TemplateConfig};
pub use types::{
    BinaryKind, CardinalityRegime, CategoricalKind, IdentifierKind, NumberKind, NumericWidth,
    SemanticType, StructuredKind, TemporalKind, TimestampGranularity, WireEncoding,
    WireEncodingKind,
};
