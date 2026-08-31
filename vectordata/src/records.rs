// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Ordinal-addressed record facets, and the codecs that type them.
//!
//! A metadata or predicate facet is a **slab**: a container of
//! variable-length records addressed by ordinal, with named sibling
//! namespaces carrying the schema and other per-facet documents. Until
//! now nothing in this crate could read one — the readers in [`crate::io`]
//! are built on fixed-width elements, and a slab record is neither fixed
//! nor an element run.
//!
//! ## The layers
//!
//! The codecs were already here; only the container was missing. What
//! this module adds is the bottom layer and the composition:
//!
//! ```text
//! slab ──[container]──▶ &[u8] ──[stage 1]──▶ ANode ──[stage 2]──▶ text ──[serde]──▶ T
//!         this module          formats::anode      formats::anode_vernacular
//! ```
//!
//! **Stage 1 is self-describing.** A record's leading byte names its
//! dialect — `0x01` for an MNode, `0x02` for a PNode — so the reader
//! never infers the dialect from the facet it came from. The same
//! container in predicate position holds PNodes and in content position
//! holds MNodes, and taking that from the facet table would put record
//! identity in two places when the bytes already carry it.
//!
//! ## Currying, not a second implementation
//!
//! Applying a codec to a facet yields a typed reader:
//!
//! ```rust,ignore
//! let facet = view.open_facet_records("metadata_content")?;
//!
//! let nodes = facet.decode(Anode);                  // get() -> ANode
//! let cql   = facet.decode(Text(Vernacular::Cql));  // get() -> String
//! let rows  = facet.decode(Serde::<MyRow>::new());  // get() -> MyRow
//! ```
//!
//! There is one read path. The untyped level is not a separate
//! implementation but [`Records<Anode>`] — the codec that stops after
//! stage 1. A by-name codec resolves to the same impls, so the dynamic
//! entry point is a lookup in front of one implementation rather than a
//! parallel one.
//!
//! A series needs nothing extra: the shard model resolves a facet
//! ordinal to a shard and a local ordinal, and the slab resolves the
//! local one — which is what makes shards of a slab series ordinary
//! slabs based at zero (`docs/design/srd-multifile-facet-shards.md`,
//! SH-96).

use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::formats::anode::{self, ANode};
use crate::formats::anode_vernacular::{self, Vernacular};

/// What went wrong reading or decoding a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordError {
    /// The facet declares no readable container.
    NotAContainer(String),
    /// The facet's bytes are not local, so the container cannot be
    /// opened. Slab access is by memory map, which needs a real file.
    NotResident(String),
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
            Self::NotAContainer(s) => write!(f, "not a record container: {s}"),
            Self::NotResident(s) => write!(
                f,
                "{s} is not resident locally; precache the facet before reading \
                 records — a slab is read by memory map and needs a real file"
            ),
            Self::Container(s) => write!(f, "{s}"),
            Self::OutOfBounds(o) => write!(f, "ordinal {o} is past the end of this facet"),
            Self::Decode(s) => write!(f, "decode: {s}"),
        }
    }
}

impl std::error::Error for RecordError {}

type Result<T> = std::result::Result<T, RecordError>;

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

/// Stage 1 only: the record as the node it says it is.
///
/// The dialect comes from the record's leading byte, so a facet holding
/// a mix of MNodes and PNodes reads without the caller declaring which
/// is which.
#[derive(Debug, Clone, Copy, Default)]
pub struct Anode;

impl RecordCodec for Anode {
    type Out = ANode;
    fn decode(&self, bytes: &[u8]) -> Result<ANode> {
        anode::decode(bytes).map_err(RecordError::Decode)
    }
}

/// Stages 1 and 2: the record rendered in a human-readable vernacular.
///
/// The output is text by construction — `Vernacular` is a rendering,
/// not a type. A caller wanting a structured value wants [`Serde`].
#[derive(Debug, Clone, Copy)]
pub struct Text(pub Vernacular);

impl RecordCodec for Text {
    type Out = String;
    fn decode(&self, bytes: &[u8]) -> Result<String> {
        let node = anode::decode(bytes).map_err(RecordError::Decode)?;
        Ok(anode_vernacular::render(&node, self.0))
    }
}

/// Stages 1, 2 and serde: the record as a caller's own type.
///
/// Routed through the JSON vernacular, which is the bridge between the
/// node model and anything with a `Deserialize` — so a caller names the
/// struct it wants and gets it by ordinal, without this crate knowing
/// anything about that struct.
#[derive(Debug)]
pub struct Serde<T>(PhantomData<fn() -> T>);

impl<T> Serde<T> {
    /// A codec producing `T`.
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Default for Serde<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for Serde<T> {
    fn clone(&self) -> Self {
        *self
    }
}

// Hand-written rather than derived: `derive` would demand `T: Clone`
// and `T: Copy`, which the codec never needs — it holds no `T`, only
// the intent to produce one.
impl<T> Copy for Serde<T> {}

impl<T: serde::de::DeserializeOwned> RecordCodec for Serde<T> {
    type Out = T;
    fn decode(&self, bytes: &[u8]) -> Result<T> {
        let node = anode::decode(bytes).map_err(RecordError::Decode)?;
        let json = anode_vernacular::render(&node, Vernacular::Jsonl);
        serde_json::from_str(&json).map_err(|e| RecordError::Decode(e.to_string()))
    }
}

/// Resolve a codec named in a setting or a declaration.
///
/// The name is a vernacular's, parsed by the same
/// [`Vernacular::parse`] every other by-name surface uses — so a codec
/// selected at runtime and one written in a type reach identical
/// decoding. `"anode"` names the stage-1 codec, which has no vernacular
/// because it produces no text.
pub fn codec_by_name(name: &str) -> Option<Box<dyn RecordCodec<Out = String>>> {
    Vernacular::parse(name).map(|v| Box::new(Text(v)) as Box<dyn RecordCodec<Out = String>>)
}

/// One slab file backing part or all of a facet.
struct Container {
    path: PathBuf,
    namespace: Option<String>,
    /// Opened on first read. A facet may name many shards and a caller
    /// may touch few, so the memory map is paid for per file used.
    ///
    /// `Some(None)` means the file opened but does not carry the
    /// namespace asked for — a normal state, not a failure.
    reader: OnceLock<Option<slabtastic::SlabReader>>,
    /// Records this container contributes, known after it is opened.
    count: OnceLock<u64>,
}

impl Container {
    /// The reader for this container, or `None` when the file is a
    /// readable slab that simply lacks the namespace asked for.
    ///
    /// The file is opened first so a real I/O or format failure stays a
    /// failure, and only the namespace probe is allowed to answer
    /// "absent" — the same separation `dataset::layout` makes, and for
    /// the same reason: an optional document's absence is a normal
    /// state and must not read as a broken file.
    fn open(&self) -> Result<Option<&slabtastic::SlabReader>> {
        if let Some(r) = self.reader.get() {
            return Ok(r.as_ref());
        }
        let opened = match self.namespace.as_deref() {
            None => Some(slabtastic::SlabReader::open(&self.path).map_err(|e| {
                RecordError::Container(format!("open slab {}: {e}", self.path.display()))
            })?),
            Some(ns) => {
                slabtastic::SlabReader::open(&self.path).map_err(|e| {
                    RecordError::Container(format!("open slab {}: {e}", self.path.display()))
                })?;
                slabtastic::SlabReader::open_namespace(&self.path, Some(ns)).ok()
            }
        };
        // A race here loses the duplicate map rather than the read.
        let _ = self.reader.set(opened);
        Ok(self.reader.get().expect("just set").as_ref())
    }

    /// Records this container contributes — zero when it does not carry
    /// the namespace.
    fn count(&self) -> Result<u64> {
        if let Some(n) = self.count.get() {
            return Ok(*n);
        }
        let n = self.open()?.map_or(0, |r| r.total_records());
        let _ = self.count.set(n);
        Ok(n)
    }
}

/// A facet of ordinal-addressed records, before a codec is chosen.
///
/// Holds the containers behind the facet — one for a single file, one
/// per shard for a series — and answers a facet ordinal by finding the
/// container that owns it and asking for its local ordinal.
pub struct RecordFacet {
    facet: String,
    containers: Vec<Container>,
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
        let _ = self.starts.set(acc);
        Ok(self.starts.get().expect("just set"))
    }

    /// The container holding facet ordinal `o`, and its local ordinal.
    ///
    /// Each shard of a slab series is an ordinary slab based at zero,
    /// so the local ordinal is what the container is asked for and the
    /// global base never reaches it (SH-96).
    fn locate(&self, o: u64) -> Result<(&Container, u64)> {
        let starts = self.offsets()?;
        // Containers are few; a scan beats the machinery to avoid one.
        for (i, c) in self.containers.iter().enumerate() {
            if o < starts[i + 1] {
                return Ok((c, o - starts[i]));
            }
        }
        Err(RecordError::OutOfBounds(o))
    }

    /// One record's bytes, borrowed from the container's mapping.
    ///
    /// The escape hatch beneath every codec: a caller that wants to
    /// decode a record some other way is not obliged to go through one.
    pub fn record_bytes(&self, ordinal: u64) -> Result<&[u8]> {
        let (container, local) = self.locate(ordinal)?;
        // `locate` only returns a container the ordinal falls inside,
        // and an absent namespace contributes none — so reaching here
        // means the reader exists.
        let reader = container
            .open()?
            .ok_or(RecordError::OutOfBounds(ordinal))?;
        reader.get_ref(local as i64).map_err(|e| {
            RecordError::Container(format!(
                "read ordinal {ordinal} (local {local}) from {}: {e}",
                container.path.display()
            ))
        })
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
            containers: self
                .containers
                .iter()
                .map(|c| Container {
                    path: c.path.clone(),
                    namespace: Some(name.to_string()),
                    reader: OnceLock::new(),
                    count: OnceLock::new(),
                })
                .collect(),
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
        self.codec.decode(self.facet.record_bytes(ordinal)?)
    }

    /// Every record in order, decoded lazily.
    pub fn iter(&self) -> impl Iterator<Item = Result<C::Out>> + '_ {
        let n = self.facet.count().unwrap_or(0);
        (0..n).map(move |o| self.get(o))
    }
}

impl RecordFacet {
    /// Open a facet's record containers through a view.
    ///
    /// Resolves the facet's files exactly as every other reader does —
    /// through the storage handle, so catalog anchoring, caching and
    /// the shard model all apply — and then hands each one to the slab
    /// reader.
    ///
    /// A slab is read by memory map, so its bytes must be on disk. A
    /// remote facet that has not been precached is refused by name
    /// rather than partially read: the alternative is a map over a
    /// sparse file, which reads holes as zeroes and decodes them as a
    /// dialect error somewhere unrelated.
    pub(crate) fn open(
        storage: &crate::view::FacetStorage,
        facet: &str,
        namespace: Option<&str>,
    ) -> Result<Self> {
        let mut containers = Vec::new();
        match storage.series_ref() {
            None => {
                let path = storage.local_file().ok_or_else(|| {
                    RecordError::NotResident(format!("facet '{facet}'"))
                })?;
                containers.push(Container {
                    path,
                    namespace: namespace.map(str::to_string),
                    reader: OnceLock::new(),
                    count: OnceLock::new(),
                });
            }
            Some(series) => {
                // One container per **shard**, in ordinal order, so the
                // facet's ordinal space is the concatenation the shard
                // map describes. Two shards drawn from one file open it
                // twice, which is the price of a container that maps
                // its own file.
                for shard in 0..series.shards().entries().len() {
                    let i = series.file_index_of_shard(shard).map_err(|e| {
                        RecordError::Container(format!("facet '{facet}': {e}"))
                    })?;
                    let handle = series.file(i).map_err(|e| {
                        RecordError::Container(format!("facet '{facet}': {e}"))
                    })?;
                    // Complete, not merely present: a sparse cache file
                    // exists from the moment a download starts and reads
                    // as zeroes until it finishes.
                    let path = handle
                        .is_complete()
                        .then(|| handle.local_path())
                        .flatten()
                        .ok_or_else(|| {
                            RecordError::NotResident(format!(
                                "facet '{facet}' shard {shard} ({})",
                                series.file_source(i)
                            ))
                        })?;
                    containers.push(Container {
                        path,
                        namespace: namespace.map(str::to_string),
                        reader: OnceLock::new(),
                        count: OnceLock::new(),
                    });
                }
            }
        }
        if containers.is_empty() {
            return Err(RecordError::NotAContainer(format!("facet '{facet}'")));
        }
        Ok(Self {
            facet: facet.to_string(),
            containers,
            starts: OnceLock::new(),
        })
    }
}
