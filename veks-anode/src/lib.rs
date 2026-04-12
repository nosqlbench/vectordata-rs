// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! # veks-anode — Self-describing binary wire formats
//!
//! Provides two core record types for structured data interchange:
//!
//! - [`mnode::MNode`]: An ordered map of named, typed fields (metadata records).
//!   Each field carries a [`mnode::TypeTag`] discriminant and a value from the
//!   [`mnode::MValue`] enum (29 scalar, container, and temporal types).
//!
//! - [`pnode::PNode`]: A boolean predicate tree with field references, comparison
//!   operators, and typed comparands. Supports both indexed (positional) and named
//!   (string) field references with two wire sub-formats.
//!
//! Both types are self-describing: each record carries enough type information to
//! be decoded without external schema. Dialect leader bytes (`0x01` for MNode,
//! `0x02` for PNode) allow mixed-format streams.
//!
//! ## Quick Start
//!
//! ```rust
//! use veks_anode::mnode::{MNode, MValue};
//!
//! let mut node = MNode::new();
//! node.insert("name".into(), MValue::Text("alice".into()));
//! node.insert("age".into(), MValue::Int(30));
//!
//! // Encode to bytes
//! let bytes = node.to_bytes();
//!
//! // Decode back
//! let decoded = MNode::from_bytes(&bytes).unwrap();
//! assert_eq!(node, decoded);
//! ```
//!
//! ```rust
//! use veks_anode::pnode::*;
//!
//! let pred = PNode::Predicate(PredicateNode {
//!     field: FieldRef::Named("score".into()),
//!     op: OpType::Ge,
//!     comparands: vec![Comparand::Int(90)],
//! });
//!
//! let bytes = pred.to_bytes_named();
//! let decoded = PNode::from_bytes_named(&bytes).unwrap();
//! assert_eq!(pred, decoded);
//! ```
//!
//! ## Dependencies
//!
//! Only `byteorder` (LE serialization), `indexmap` (ordered field maps),
//! and `serde_json` (JSON vernacular parsing).
//! No networking, no I/O beyond `std::io::Read`/`Write`.

pub mod mnode;
pub mod pnode;
pub mod anode;
pub mod anode_vernacular;
