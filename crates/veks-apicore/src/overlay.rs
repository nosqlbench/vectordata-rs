// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Where a dataset attribute's value came from, and what an overlay
//! states: the vocabulary of attribute overlays
//! (`srd-loose-files-overlays-integrity.md` §4). Loading, merging and
//! editing overlays is vectordata's (`vectordata::overlay`).

use std::fmt;

use serde_yaml::Value;

/// Who stated an overlay.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OverlayKind {
    /// The user, in `<config_dir>/overlays.yaml`.
    User,
    /// A program, by the name it registered the overlay under.
    Program(String),
}

/// Where an attribute's value came from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AttributeSource {
    /// The dataset's own descriptor or catalog listing.
    Dataset,
    /// Its loose files' names (`dataset::infer`, LF-12).
    Inferred,
    /// An overlay.
    Overlay {
        /// The overlay's file, or the program's name for one built in
        /// code.
        file: String,
        /// Who stated it.
        kind: OverlayKind,
    },
    /// Measured from the data, and recorded in this file
    /// (`<cache_root>/<dataset>/measured.yaml`), or not recorded (a
    /// local base, measured each time).
    Measured {
        /// The file the measurement is recorded in, when it is.
        file: Option<String>,
    },
}

impl fmt::Display for AttributeSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttributeSource::Dataset => write!(f, "dataset"),
            AttributeSource::Inferred => write!(f, "inferred from file names"),
            AttributeSource::Overlay { file, kind: OverlayKind::User } => write!(f, "overlay {file}"),
            AttributeSource::Overlay { file, kind: OverlayKind::Program(name) } => {
                write!(f, "program overlay '{name}' ({file})")
            }
            AttributeSource::Measured { file: Some(file) } => write!(f, "measured, recorded in {file}"),
            AttributeSource::Measured { file: None } => write!(f, "measured"),
        }
    }
}

/// One attribute an overlay sets, with its source.
#[derive(Debug, Clone, PartialEq)]
pub struct Stated {
    /// The key.
    pub key: String,
    /// The value.
    pub value: Value,
    /// Who stated it.
    pub source: AttributeSource,
}

/// An overlay's value that the data or the dataset's own declaration
/// says otherwise (OV-5). The overlay's value stays in force; the
/// implementation that found it decides how to say so (vectordata
/// warns once per process for each dataset and key).
#[derive(Debug, Clone, PartialEq)]
pub struct Contradiction {
    /// The dataset.
    pub dataset: String,
    /// The attribute.
    pub key: String,
    /// What the overlay states, rendered.
    pub claimed: String,
    /// What was observed instead (`the dataset's own L2`).
    pub observed: String,
    /// Who stated it.
    pub source: AttributeSource,
}
