// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `analyze survey` — incremental metadata survey.
//!
//! Type-driven, sketch-backed survey of an ANode/MNode slab file in
//! two sampled passes — discovery, then profiling — followed by an
//! optional exhaustive third pass, the census, that counts declared
//! fields, hierarchies and pairs exactly over every record. See
//! `docs/sysref/13-metadata-survey.md` for the full design.

pub mod census;
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

pub use census::{
    CensusConfig, CensusInfo, DroppedField, ExactIntegerHistogramReport, ExactValueCensusReport,
    HierarchyCensusReport, HierarchyNode, PairCensusReport, DEFAULT_CENSUS_CAP,
    DEFAULT_PAIR_CELLS_CAP,
};
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
