// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Ordinal-addressed record facets: a facet of opaque records (a slab
//! of MNodes or PNodes, a schema document) read by ordinal
//! ([`RecordFacet`]), typed by a codec ([`RecordCodec`], [`Records`]).
//!
//! The containers behind a facet are an implementation's
//! ([`RecordContainer`]): vectordata reads slabs through its storage,
//! so a record costs its page, fetched and verified like any other
//! read. Its codecs (`Anode`, `Text`, `Serde`) are in
//! `vectordata::records`.

use std::sync::OnceLock;

/// What went wrong reading or decoding a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordError {
    /// The facet holds element runs, not opaque records.
    ///
    /// The mirror of [`crate::Error::WrongFacetShape`]: a caller that
    /// brings an `.fvec` here would otherwise get a slab parse failure
    /// about a footer, which says nothing about what actually happened.
    WrongShape {
        /// The facet as declared.
        facet: String,
        /// The reader that does open it.
        reader: &'static str,
    },
    /// The facet declares no readable container.
    NotAContainer(String),
    /// The container could not be opened or read.
    Container(String),
    /// The ordinal lies outside the facet.
    OutOfBounds(u64),
    /// The record's bytes could not be decoded by the chosen codec.
    Decode(String),
}

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongShape { facet, reader } => write!(
                f,
                "facet '{facet}' holds element runs, not opaque records; \
                 open it with {reader}"
            ),
            Self::NotAContainer(s) => write!(f, "not a record container: {s}"),
            Self::Container(s) => write!(f, "{s}"),
            Self::OutOfBounds(o) => write!(f, "ordinal {o} is past the end of this facet"),
            Self::Decode(s) => write!(f, "decode: {s}"),
        }
    }
}

impl std::error::Error for RecordError {}

/// The result of a record read.
pub type Result<T> = std::result::Result<T, RecordError>;

/// Turns a record's bytes into a value.
///
/// Implementors compose the stages above. A codec is a **value**, not
/// just a type, so a codec chosen at runtime — a vernacular named in a
/// setting — is the same kind of thing as one chosen at compile time,
/// and both reach the same `decode`.
pub trait RecordCodec {
    /// What this codec produces.
    type Out;
    /// Decode one record.
    fn decode(&self, bytes: &[u8]) -> Result<Self::Out>;
}

/// One container of records — a slab, or one shard of a series of
/// them — addressed from zero: what an implementation hands a
/// [`RecordFacet`] to read through. A codec ([`RecordCodec`]) types
/// what comes out:
///
/// ```
/// use std::borrow::Cow;
/// use veks_apicore::records::{RecordCodec, RecordContainer, RecordError, RecordFacet};
///
/// /// Records held in memory.
/// struct Lines(Vec<&'static str>);
///
/// impl RecordContainer for Lines {
///     fn count(&self) -> Result<u64, RecordError> {
///         Ok(self.0.len() as u64)
///     }
///     fn record(&self, local: u64) -> Result<Cow<'_, [u8]>, RecordError> {
///         self.0.get(local as usize).map(|s| Cow::Borrowed(s.as_bytes())).ok_or(RecordError::OutOfBounds(local))
///     }
///     fn namespaced(&self, _name: &str) -> Box<dyn RecordContainer> {
///         Box::new(Lines(Vec::new()))
///     }
/// }
///
/// /// Each record as UTF-8 text.
/// struct Utf8;
///
/// impl RecordCodec for Utf8 {
///     type Out = String;
///     fn decode(&self, bytes: &[u8]) -> Result<String, RecordError> {
///         String::from_utf8(bytes.to_vec()).map_err(|e| RecordError::Decode(e.to_string()))
///     }
/// }
///
/// // Two shards, one facet: ordinals run across them.
/// let facet = RecordFacet::new("labels", vec![Box::new(Lines(vec!["a", "b"])), Box::new(Lines(vec!["c"]))]).unwrap();
/// assert_eq!(facet.count().unwrap(), 3);
/// assert_eq!(facet.decode(Utf8).get(2).unwrap(), "c");
/// ```
pub trait RecordContainer: Send + Sync {
    /// The records it holds.
    fn count(&self) -> Result<u64>;

    /// Record `local`'s bytes: borrowed when mapped, owned when fetched.
    fn record(&self, local: u64) -> Result<std::borrow::Cow<'_, [u8]>>;

    /// The same container's sibling namespace `name`.
    fn namespaced(&self, name: &str) -> Box<dyn RecordContainer>;
}

impl RecordFacet {
    /// The facet `facet` over `containers`, in ordinal order: one for a
    /// single file, one per shard of a series. For an implementation
    /// opening a facet as records.
    pub fn new(facet: impl Into<String>, containers: Vec<Box<dyn RecordContainer>>) -> Result<Self> {
        let facet = facet.into();
        if containers.is_empty() {
            return Err(RecordError::NotAContainer(format!("facet '{facet}'")));
        }
        Ok(Self { facet, containers, starts: OnceLock::new() })
    }
}

impl std::fmt::Debug for RecordFacet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordFacet").field("facet", &self.facet).field("containers", &self.containers.len()).finish()
    }
}

/// A facet of ordinal-addressed records, before a codec is chosen.
///
/// Holds the containers behind the facet — one for a single file, one
/// per shard for a series — and answers a facet ordinal by finding the
/// container that owns it and asking for its local ordinal.
pub struct RecordFacet {
    facet: String,
    containers: Vec<Box<dyn RecordContainer>>,
    /// First facet ordinal of each container, plus a final total.
    starts: OnceLock<Vec<u64>>,
}

impl RecordFacet {
    /// The number of records across every container.
    pub fn count(&self) -> Result<u64> {
        Ok(*self.offsets()?.last().unwrap_or(&0))
    }

    /// Cumulative first-ordinals, computed once.
    ///
    /// Counts come from the containers rather than from the shard
    /// declaration: a slab knows how many records it holds, and asking
    /// it is what keeps this from being a second place the answer
    /// lives (SH-78).
    fn offsets(&self) -> Result<&Vec<u64>> {
        if let Some(o) = self.starts.get() {
            return Ok(o);
        }
        let mut acc = Vec::with_capacity(self.containers.len() + 1);
        let mut at = 0u64;
        acc.push(0);
        for c in &self.containers {
            at += c.count()?;
            acc.push(at);
        }
        Ok(self.starts.get_or_init(|| acc))
    }

    /// The container holding facet ordinal `o`, and its local ordinal.
    ///
    /// Each shard of a slab series is an ordinary slab based at zero,
    /// so the local ordinal is what the container is asked for and the
    /// global base never reaches it (SH-96).
    fn locate(&self, o: u64) -> Result<(&dyn RecordContainer, u64)> {
        let starts = self.offsets()?;
        // Containers are few; a scan beats the machinery to avoid one.
        for (i, c) in self.containers.iter().enumerate() {
            if o < starts[i + 1] {
                return Ok((c.as_ref(), o - starts[i]));
            }
        }
        Err(RecordError::OutOfBounds(o))
    }

    /// Measure this facet's record lengths, for sizing a shard.
    ///
    /// A slab's records carry their own extents, so there is no record
    /// size to divide a file-size cap by — only a measurement. This
    /// takes one at evenly spaced ordinals and hands back a basis a
    /// [`ShardPlan`](crate::dataset::ShardPlan) can carry a margin on.
    ///
    /// Costs the pages the sampled ordinals fall in, not the facet: a
    /// thousand records spread across a terabyte slab is a thousand
    /// page reads, and against a remote facet those are ranges like
    /// any other read here.
    pub fn sample_record_size(
        &self,
        target: u64,
    ) -> Result<Option<crate::dataset::RecordSize>> {
        let total = self.count()?;
        crate::dataset::shard_sizing::sample_spread(total, target, |o| {
            self.record_bytes(o).map(|b| b.len() as u64)
        })
    }

    /// One record's bytes.
    ///
    /// Borrowed when the source is mapped — a local file, or a remote
    /// one already resident — and owned when the page had to be
    /// fetched. The same borrow-where-mapped rule the vector readers
    /// follow, and the reason reading one record from a remote facet
    /// costs one page rather than one file.
    ///
    /// The escape hatch beneath every codec: a caller that wants to
    /// decode a record some other way is not obliged to go through one.
    ///
    /// Reads from whichever shard holds `ordinal`; past the last record
    /// it errors with [`RecordError::OutOfBounds`].
    pub fn record_bytes(&self, ordinal: u64) -> Result<std::borrow::Cow<'_, [u8]>> {
        let (container, local) = self.locate(ordinal)?;
        container.record(local)
    }

    /// A sibling namespace of this facet as a facet of its own.
    ///
    /// The schema, the layout copy and the survey report are records in
    /// named namespaces of the same containers, so reading them is the
    /// same operation against a different name rather than a special
    /// case per document.
    pub fn namespace(&self, name: &str) -> RecordFacet {
        RecordFacet {
            facet: format!("{}:{name}", self.facet),
            containers: self.containers.iter().map(|c| c.namespaced(name)).collect(),
            starts: OnceLock::new(),
        }
    }

    /// Apply a codec, producing a typed reader.
    ///
    /// This is the curry: the facet supplies ordinals and bytes, the
    /// codec supplies the type, and their composition is the reader a
    /// caller actually wants. `decode(Anode)` is the untyped level and
    /// not a separate path.
    pub fn decode<C: RecordCodec>(&self, codec: C) -> Records<'_, C> {
        Records { facet: self, codec }
    }

    /// The facet's name, for diagnostics.
    pub fn name(&self) -> &str {
        &self.facet
    }
}

/// A record facet with a codec applied: values by ordinal.
pub struct Records<'a, C> {
    facet: &'a RecordFacet,
    codec: C,
}

impl<C: RecordCodec> Records<'_, C> {
    /// The number of records.
    pub fn count(&self) -> Result<u64> {
        self.facet.count()
    }

    /// The record at `ordinal`, decoded.
    pub fn get(&self, ordinal: u64) -> Result<C::Out> {
        self.codec.decode(&self.facet.record_bytes(ordinal)?)
    }

    /// Every record in order, decoded lazily.
    pub fn iter(&self) -> impl Iterator<Item = Result<C::Out>> + '_ {
        let n = self.facet.count().unwrap_or(0);
        (0..n).map(move |o| self.get(o))
    }
}
