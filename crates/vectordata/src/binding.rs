// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Binding records to operation parameters.
//!
//! A workload generator addresses a facet per cycle: cycle, ordinal,
//! record, bound operation. [`crate::records`] delivers the first three.
//! This delivers the fourth.
//!
//! ## Binding is not rendering
//!
//! The vernacular codecs produce text for a human — `to_cql` inlines
//! literals, quotes strings by hand, and drops the field names. That is
//! right for `inspect` and wrong for a prepared statement, where the
//! statement is prepared once and only values move per cycle. Both
//! contracts are legitimate and neither is a degraded form of the
//! other; this module is the second one.
//!
//! ## What it costs
//!
//! Names are resolved to **positions** once, from the facet's schema.
//! Per record the names are skipped and values are read in place, so
//! binding a record allocates nothing per field name — the mechanism is
//! `veks_anode::mnode::scan`, built for predicate evaluation and reused
//! here unchanged.
//!
//! See `docs/design/srd-record-binding-and-forms.md`.

use crate::formats::mnode::TypeTag;

/// The type a parameter binds as.
///
/// **Distinct from the rendering mapping**, deliberately. `cql_type`
/// answers "what column would hold this" and is correct as DDL; it maps
/// `Half` to `smallint`, gives `Null` a type of `text`, and collapses
/// container element types. Binding a value through those answers would
/// send a float's bit pattern as an integer and every collection's
/// elements as strings.
///
/// Dialect-neutral: this names what the value *is*, and an adapter maps
/// it to a driver's type. The mapping from [`TypeTag`] is exhaustive
/// with no wildcard arm, so a new tag fails to compile until it is
/// handled here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindType {
    /// UTF-8 text.
    Text,
    /// Text constrained to ASCII.
    Ascii,
    /// True or false.
    Bool,
    /// Opaque bytes.
    Blob,
    /// Signed integers, by width.
    Int16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// IEEE binary16.
    Float16,
    /// IEEE binary32.
    Float32,
    /// IEEE binary64.
    Float64,
    /// Arbitrary-precision decimal.
    Decimal,
    /// Arbitrary-precision integer.
    Varint,
    /// A date with no time.
    Date,
    /// A time with no date.
    Time,
    /// An instant, in milliseconds.
    TimestampMillis,
    /// An instant, in seconds plus a nanosecond adjustment.
    TimestampNanos,
    /// An instant carried as text. Distinct from the numeric
    /// timestamps because that is what it is on the wire, and a
    /// parameter typed as a number would be bound from a string.
    TimestampText,
    /// A UUID whose ordering carries a timestamp.
    TimeUuid,
    /// A UUID.
    Uuid,
    /// A ULID.
    Ulid,
    /// Absent. Carries no type of its own — the parameter's type comes
    /// from the schema, not from the null.
    Null,
    /// An ordered collection. `None` when the element type is not
    /// determined by the tag alone, which is the schema-level case:
    /// only a value knows what it holds.
    List(Option<Box<BindType>>),
    /// An unordered collection.
    Set(Option<Box<BindType>>),
    /// Key/value pairs.
    Map(Option<Box<BindType>>, Option<Box<BindType>>),
}

impl BindType {
    /// The bind type a tag implies.
    ///
    /// Container element types come back as `None`: a tag says a field
    /// holds a list, and only the value says a list of what. Reporting
    /// that honestly is the point — the rendering mapping guesses
    /// `text` there, which is fine in a `CREATE TABLE` and wrong in a
    /// bind.
    pub fn of_tag(tag: TypeTag) -> BindType {
        match tag {
            TypeTag::Text | TypeTag::TextValidated | TypeTag::EnumStr => BindType::Text,
            TypeTag::Ascii => BindType::Ascii,
            TypeTag::Bool => BindType::Bool,
            TypeTag::Bytes => BindType::Blob,
            TypeTag::Short => BindType::Int16,
            TypeTag::Int32 | TypeTag::EnumOrd => BindType::Int32,
            TypeTag::Int => BindType::Int64,
            // A 16-bit float is a float. The rendering mapping calls it
            // a smallint, which would bind the bit pattern.
            TypeTag::Half => BindType::Float16,
            TypeTag::Float32 => BindType::Float32,
            TypeTag::Float => BindType::Float64,
            TypeTag::Decimal => BindType::Decimal,
            TypeTag::Varint => BindType::Varint,
            TypeTag::Date => BindType::Date,
            TypeTag::Time => BindType::Time,
            TypeTag::Millis => BindType::TimestampMillis,
            TypeTag::Nanos => BindType::TimestampNanos,
            // Textual on the wire, whatever the name suggests.
            TypeTag::DateTime => BindType::TimestampText,
            TypeTag::UuidV1 => BindType::TimeUuid,
            TypeTag::UuidV7 => BindType::Uuid,
            TypeTag::Ulid => BindType::Ulid,
            TypeTag::Null => BindType::Null,
            TypeTag::List | TypeTag::Array => BindType::List(None),
            TypeTag::Set => BindType::Set(None),
            TypeTag::Map | TypeTag::TypedMap => BindType::Map(None, None),
        }
    }

    /// Whether this type is still missing an element type.
    ///
    /// A schema built from tags alone leaves containers open; a caller
    /// preparing a statement against one needs either a form that says
    /// what they hold, or a sample value.
    pub fn is_underdetermined(&self) -> bool {
        match self {
            BindType::List(e) | BindType::Set(e) => e.is_none(),
            BindType::Map(k, v) => k.is_none() || v.is_none(),
            _ => false,
        }
    }
}

use crate::formats::mnode::scan::{self, Field};
use crate::records::{RecordError, RecordFacet};
use std::collections::HashMap;

/// What went wrong preparing or binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindError {
    /// The facet could not be read.
    Facet(String),
    /// No record was available to learn the layout from.
    NoLayout(String),
    /// A record could not be walked.
    Record(String),
    /// A parameter names a field the facet does not have.
    NoSuchField {
        /// The field asked for.
        field: String,
        /// The fields the facet does have, in order.
        available: Vec<String>,
    },
    /// A form was asked for that this facet does not offer.
    NoSuchForm {
        /// The form asked for.
        form: String,
        /// The forms the facet does offer.
        available: Vec<String>,
    },
}

impl std::fmt::Display for BindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Facet(s) => write!(f, "{s}"),
            Self::NoLayout(s) => write!(f, "{s}: no record to learn the field layout from"),
            Self::Record(s) => write!(f, "{s}"),
            Self::NoSuchField { field, available } => write!(
                f,
                "no field '{field}' in this facet; it has: {}",
                available.join(", ")
            ),
            Self::NoSuchForm { form, available } => write!(
                f,
                "this facet offers no form '{form}'; it offers: {}",
                if available.is_empty() {
                    "(only its implicit form)".to_string()
                } else {
                    available.join(", ")
                }
            ),
        }
    }
}

impl std::error::Error for BindError {}

impl From<RecordError> for BindError {
    fn from(e: RecordError) -> Self {
        BindError::Facet(e.to_string())
    }
}

type Result<T> = std::result::Result<T, BindError>;

/// A facet's field layout: names in wire order, with the type each
/// binds as.
///
/// Learned **once**. The names are the metadata's own, in the order the
/// records carry them, and that order is the position a binder
/// addresses fields by for the rest of the run.
#[derive(Debug, Clone)]
pub struct Layout {
    names: Vec<String>,
    types: Vec<BindType>,
}

impl Layout {
    /// Learn a facet's layout from its first record.
    ///
    /// A facet publishing a schema namespace should be preferred over
    /// this, because a first record that omits an optional field would
    /// otherwise define the layout for every record after it. That
    /// preference is the caller's to express until a schema-namespace
    /// reader exists; this is the discovery path.
    pub fn discover(facet: &RecordFacet) -> Result<Self> {
        if facet.count()? == 0 {
            return Err(BindError::NoLayout(facet.name().to_string()));
        }
        let bytes = facet.record_bytes(0)?;
        let mut names = Vec::new();
        let mut types = Vec::new();
        for field in scan::fields(&bytes).map_err(|e| BindError::Record(e.to_string()))? {
            let field = field.map_err(|e| BindError::Record(e.to_string()))?;
            names.push(
                field
                    .name_str()
                    .ok_or_else(|| BindError::Record("field name is not UTF-8".into()))?
                    .to_string(),
            );
            types.push(
                TypeTag::from_u8(field.tag)
                    .map(BindType::of_tag)
                    .unwrap_or(BindType::Blob),
            );
        }
        Ok(Layout { names, types })
    }

    /// Field names, in wire order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// What each field binds as, by position.
    pub fn types(&self) -> &[BindType] {
        &self.types
    }

    /// The position of a named field.
    pub fn position_of(&self, field: &str) -> Option<usize> {
        self.names.iter().position(|n| n == field)
    }
}

/// A compiled binding: which positions to read, and what to call them.
///
/// Built once from a [`Layout`], then used for every record. The
/// per-record path never looks at a field name — that work is done
/// here, which is what keeps binding free of per-record allocation.
#[derive(Debug, Clone)]
pub struct Binder {
    /// Positions to bind, in parameter order.
    positions: Vec<usize>,
    /// Parameter names, in the same order.
    parameters: Vec<String>,
    /// What each parameter binds as.
    types: Vec<BindType>,
}

impl Binder {
    /// Bind every field of the layout, under its own name.
    pub fn all(layout: &Layout) -> Self {
        Binder {
            positions: (0..layout.names.len()).collect(),
            parameters: layout.names.clone(),
            types: layout.types.clone(),
        }
    }

    /// Bind the named fields, in the order given.
    ///
    /// A field the facet does not have is refused here, at compile
    /// time, naming what the facet does have — rather than binding a
    /// missing value on the first cycle.
    pub fn select(layout: &Layout, fields: &[&str]) -> Result<Self> {
        let mut positions = Vec::with_capacity(fields.len());
        for f in fields {
            positions.push(layout.position_of(f).ok_or_else(|| {
                BindError::NoSuchField {
                    field: (*f).to_string(),
                    available: layout.names.clone(),
                }
            })?);
        }
        Ok(Binder {
            parameters: positions.iter().map(|i| layout.names[*i].clone()).collect(),
            types: positions.iter().map(|i| layout.types[*i].clone()).collect(),
            positions,
        })
    }

    /// Rename parameters for substitution.
    ///
    /// The metadata name is the parameter name unless a runtime says
    /// otherwise. An override is applied here, against resolved
    /// positions, so it costs nothing per record and cannot reach the
    /// data — the field keeps its name in the facet.
    pub fn with_overrides(mut self, overrides: &HashMap<String, String>) -> Self {
        for p in &mut self.parameters {
            if let Some(to) = overrides.get(p.as_str()) {
                *p = to.clone();
            }
        }
        self
    }

    /// Parameter names, in bind order. Read once, to prepare a
    /// statement.
    pub fn parameters(&self) -> &[String] {
        &self.parameters
    }

    /// What each parameter binds as, in bind order.
    pub fn types(&self) -> &[BindType] {
        &self.types
    }

    /// Bind one record, handing each parameter to `f` in bind order.
    ///
    /// The form for a cycle loop: nothing is allocated, nothing is
    /// copied, and no field name is looked at. Values borrow the
    /// record, so they are handed over rather than returned — a buffer
    /// living across cycles could not hold borrows of two different
    /// records, which is a property worth having rather than working
    /// around.
    pub fn bind_each<'a, F>(&self, record: &'a [u8], mut f: F) -> Result<()>
    where
        F: FnMut(usize, Field<'a>),
    {
        let mut seen = 0usize;
        for field in scan::fields(record).map_err(|e| BindError::Record(e.to_string()))? {
            let field = field.map_err(|e| BindError::Record(e.to_string()))?;
            for (slot, want) in self.positions.iter().enumerate() {
                if *want == field.index {
                    f(slot, field);
                    seen += 1;
                }
            }
        }
        if seen < self.positions.len() {
            return Err(BindError::Record(format!(
                "record has {seen} of the {} bound fields — its layout differs \
                 from the one this binder was compiled against",
                self.positions.len()
            )));
        }
        Ok(())
    }

    /// Bind one record into `out`, in bind order.
    ///
    /// For a caller that wants random access to the bound values.
    /// `out` borrows the record, so it is scoped to it — see
    /// [`Self::bind_each`] for the loop form.
    pub fn bind<'a>(&self, record: &'a [u8], out: &mut Vec<Field<'a>>) -> Result<()> {
        out.clear();
        // One walk, picking out the wanted positions. The positions are
        // sorted for the walk and restored to parameter order after, so
        // a template may name fields in any order without costing a
        // second pass.
        let mut found: Vec<Option<Field<'a>>> = vec![None; self.positions.len()];
        for field in scan::fields(record).map_err(|e| BindError::Record(e.to_string()))? {
            let field = field.map_err(|e| BindError::Record(e.to_string()))?;
            for (slot, want) in self.positions.iter().enumerate() {
                if *want == field.index {
                    found[slot] = Some(field);
                }
            }
        }
        for (slot, f) in found.into_iter().enumerate() {
            out.push(f.ok_or_else(|| BindError::NoSuchField {
                field: self.parameters[slot].clone(),
                available: Vec::new(),
            })?);
        }
        Ok(())
    }
}

/// Slab namespace enumerating the op-template forms a facet offers.
///
/// Absent from every dataset written so far, and absent means one
/// implicit form rather than none.
pub const FORMS_NAMESPACE: &str = "forms";

/// One op-template form a facet offers.
///
/// Open by construction: unknown keys are kept in `extra` rather than
/// rejected. A writer recording a form this build does not implement is
/// recording, not misbehaving.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Form {
    /// The form's name, as selected.
    pub name: String,
    /// Which dialect of record this form is for — `metadata` or
    /// `predicate`. Advisory: the record's leader byte remains the
    /// authority on what a record is.
    #[serde(default)]
    pub kind: Option<String>,
    /// The record encoding this form consumes, e.g. `mnode:v1`.
    #[serde(default)]
    pub wire_format: Option<String>,
    /// What operation the bound record becomes, e.g. `insert`.
    #[serde(default)]
    pub operation: Option<String>,
    /// Fields this form binds, in bind order. Absent means all of them,
    /// in the facet's own order.
    #[serde(default)]
    pub fields: Option<Vec<String>>,
    /// Parameter renames, keyed by field name.
    #[serde(default)]
    pub parameters: HashMap<String, String>,
    /// Anything this build does not recognise, preserved.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Form {
    /// The name an unnamed, implicit form goes by.
    pub const IMPLICIT: &'static str = "default";

    /// The form a facet with no `forms` namespace offers: every field,
    /// under its own name.
    pub fn implicit() -> Self {
        Form {
            name: Self::IMPLICIT.to_string(),
            kind: None,
            wire_format: None,
            operation: None,
            fields: None,
            parameters: HashMap::new(),
            extra: HashMap::new(),
        }
    }

    /// Compile this form against a layout.
    pub fn binder(&self, layout: &Layout) -> Result<Binder> {
        let binder = match &self.fields {
            None => Binder::all(layout),
            Some(names) => {
                let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                Binder::select(layout, &refs)?
            }
        };
        Ok(binder.with_overrides(&self.parameters))
    }
}

/// The forms a facet offers.
///
/// Read from the `forms` namespace when a facet has one. A facet
/// without it offers exactly one form — the implicit one its records
/// already have — which is every dataset in existence and must keep
/// working unchanged. Absence is not an empty set.
pub fn forms_of(facet: &RecordFacet) -> Result<Vec<Form>> {
    let ns = facet.namespace(FORMS_NAMESPACE);
    let count = ns.count()?;
    if count == 0 {
        return Ok(vec![Form::implicit()]);
    }
    let mut out = Vec::with_capacity(count as usize);
    for o in 0..count {
        let bytes = ns.record_bytes(o)?;
        // A form record this build cannot parse is skipped rather than
        // failing the facet: the forms it *can* read are still usable,
        // and refusing all of them over one would make a facet
        // unreadable for describing a capability it also has.
        match serde_json::from_slice::<Form>(&bytes) {
            Ok(form) => out.push(form),
            Err(e) => log::warn!(
                "facet '{}': form record {o} not understood, skipping: {e}",
                facet.name()
            ),
        }
    }
    if out.is_empty() {
        out.push(Form::implicit());
    }
    Ok(out)
}

/// Select a form by name.
///
/// A name this facet does not offer is refused naming what it does —
/// the same rule as a wrong-door reader error, for the same reason.
pub fn form_by_name(facet: &RecordFacet, name: &str) -> Result<Form> {
    let forms = forms_of(facet)?;
    forms
        .iter()
        .find(|f| f.name == name)
        .cloned()
        .ok_or_else(|| BindError::NoSuchForm {
            form: name.to_string(),
            available: forms
                .iter()
                .map(|f| f.name.clone())
                .filter(|n| n != Form::IMPLICIT)
                .collect(),
        })
}

// ---------------------------------------------------------------------------
// Predicate binding
// ---------------------------------------------------------------------------

use crate::formats::anode::{self, ANode};
use crate::formats::pnode::{Comparand, OpType, PNode};

/// One condition of a predicate, prepared.
///
/// The statement fragment is `field op ?`; only the comparands move per
/// record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Condition {
    /// The metadata field this condition constrains.
    pub field: String,
    /// What to call the parameter. The field's name unless a runtime
    /// overrode it.
    pub parameter: String,
    /// The comparison.
    pub op: OpType,
    /// What the parameter binds as.
    ///
    /// **From the field's tag, not the comparand's variant.** `MValue`
    /// has 29 variants and `Comparand` has six, so a predicate over a
    /// `DateTime` field carries an `Int` comparand — typing the
    /// parameter from that would collapse every temporal and UUID
    /// column to a bigint or a string.
    pub bind_type: BindType,
    /// How many comparands this condition carries. One for the
    /// scalar operators; a membership set for `In`.
    pub arity: usize,
}

/// A compiled predicate template: the shape of a filter, prepared once.
///
/// Compiled against **two** things — a sample predicate, which supplies
/// the fields and operators, and the metadata layout, which supplies
/// the types. Neither alone is enough: a predicate knows what it
/// constrains and not what type that field is.
#[derive(Debug, Clone)]
pub struct PredicateBinder {
    conditions: Vec<Condition>,
    /// The shape every bound record must match, values removed.
    shape: PNode,
}

impl PredicateBinder {
    /// Compile a predicate template.
    ///
    /// Only a flat conjunction compiles: a nested or disjunctive
    /// predicate has no flat parameter list, and pretending otherwise
    /// would bind values into the wrong conditions. Such a predicate is
    /// refused by name rather than partially flattened.
    pub fn compile(predicate: &PNode, layout: &Layout) -> Result<Self> {
        let flat = crate::formats::mnode::scan::flatten_and(predicate).ok_or_else(|| {
            BindError::Record(
                "this predicate is not a flat conjunction of named field \
                 comparisons, so it has no parameter list — bind its \
                 conditions individually, or use a template that is"
                    .to_string(),
            )
        })?;
        let mut conditions = Vec::with_capacity(flat.len());
        for (field, op, comparands) in flat {
            let position = layout.position_of(&field).ok_or_else(|| {
                BindError::NoSuchField {
                    field: field.clone(),
                    available: layout.names().to_vec(),
                }
            })?;
            conditions.push(Condition {
                parameter: field.clone(),
                field,
                op,
                // The field's type, not the comparand's.
                bind_type: layout.types()[position].clone(),
                arity: comparands.len(),
            });
        }
        Ok(PredicateBinder {
            conditions,
            shape: predicate.fingerprint(),
        })
    }

    /// Rename parameters for substitution, as the metadata binder does.
    pub fn with_overrides(mut self, overrides: &HashMap<String, String>) -> Self {
        for c in &mut self.conditions {
            if let Some(to) = overrides.get(c.field.as_str()) {
                c.parameter = to.clone();
            }
        }
        self
    }

    /// The conditions, in order. Read once, to build the filter.
    pub fn conditions(&self) -> &[Condition] {
        &self.conditions
    }

    /// Bind one predicate record, handing each condition's comparands
    /// to `f` in condition order.
    ///
    /// The record is checked against the compiled shape first: a
    /// predicate of a different shape would otherwise bind values into
    /// conditions they do not belong to, which no downstream check
    /// would catch.
    pub fn bind_each<F>(&self, record: &[u8], mut f: F) -> Result<()>
    where
        F: FnMut(&Condition, &[Comparand]),
    {
        let node = anode::decode(record).map_err(BindError::Record)?;
        // The leader byte is the authority on what a record is; a
        // template cannot make an MNode into a predicate.
        let ANode::PNode(pnode) = node else {
            return Err(BindError::Record(
                "this record is not a predicate — its dialect byte says MNode, \
                 and a form does not override what a record is"
                    .to_string(),
            ));
        };
        if !pnode.is_congruent(&self.shape) {
            return Err(BindError::Record(
                "this predicate's shape differs from the one this binder was \
                 compiled against; its values would bind to the wrong conditions"
                    .to_string(),
            ));
        }
        let flat = crate::formats::mnode::scan::flatten_and(&pnode).ok_or_else(|| {
            BindError::Record("predicate is not a flat conjunction".to_string())
        })?;
        for (condition, (_, _, comparands)) in self.conditions.iter().zip(flat.iter()) {
            f(condition, comparands);
        }
        Ok(())
    }
}
