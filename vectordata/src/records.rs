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
//!
//! ## Incremental by the same means as everything else
//!
//! Reads go through [`crate::storage::Storage`], not a memory map of
//! this module's own. A slab ends with a pages-page indexing every data
//! page by start ordinal, so opening a facet costs its tail and reading
//! a record costs that record's page — each fetched and merkle-verified
//! as a byte range by the same chunked source the vector readers use.
//! Nothing here requires a facet to be downloaded first, and a facet
//! spread over shards costs only the shards touched.

use std::marker::PhantomData;
use std::sync::OnceLock;

use crate::formats::anode::{self, ANode};
use crate::formats::anode_vernacular::{self, Vernacular};

/// What went wrong reading or decoding a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordError {
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
            Self::NotAContainer(s) => write!(f, "not a record container: {s}"),
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

/// The ordinal index of one slab, read from its tail.
///
/// A slab ends with a pages-page listing `(start_ordinal, file_offset)`
/// for every data page, so the whole index is three short reads from
/// the end of the file — the footer, the tail page, and for a
/// multi-namespace file the pages-page that page points at. Nothing
/// before the tail is touched to build it.
struct SlabIndex {
    /// Page entries in ordinal order.
    entries: Vec<slabtastic::PageEntry>,
    /// Records across every page.
    total: u64,
}

/// One slab backing part or all of a facet.
///
/// Reads through [`crate::storage::Storage`] rather than a memory map
/// of its own. That is what keeps a slab facet incremental like every
/// other format: a chunked, merkle-verified source fetches and verifies
/// the chunks covering each range as it is asked for, so opening a
/// facet costs its tail and reading a record costs its page — not the
/// file.
struct Container {
    storage: std::sync::Arc<crate::storage::Storage>,
    /// What the container is called, for diagnostics.
    label: String,
    namespace: Option<String>,
    /// Built on first use from the tail of the file.
    index: OnceLock<SlabIndex>,
}

impl Container {
    /// Read `len` bytes at `offset`, fetching what is missing.
    fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.storage.read_bytes(offset, len).map_err(|e| {
            RecordError::Container(format!(
                "{}: read {len} bytes at {offset}: {e}",
                self.label
            ))
        })
    }

    /// The page bytes at `offset`, borrowed when the source is mapped.
    ///
    /// A local file — or a remote one already fully resident — serves
    /// the page without copying; anything else pays one copy of one
    /// page. The same borrow-where-mapped rule the vector readers use.
    fn page(&self, offset: u64) -> Result<std::borrow::Cow<'_, [u8]>> {
        // The page's own header states its size (4 bytes at +4).
        let size = u32::from_le_bytes(
            self.read(offset + 4, 4)?
                .try_into()
                .map_err(|_| RecordError::Container(format!("{}: short page header", self.label)))?,
        ) as u64;
        if let Some(mapped) = self.storage.mmap_slice(offset, size) {
            return Ok(std::borrow::Cow::Borrowed(mapped));
        }
        Ok(std::borrow::Cow::Owned(self.read(offset, size)?))
    }

    /// The ordinal index, built from the tail on first use.
    fn index(&self) -> Result<&SlabIndex> {
        if let Some(i) = self.index.get() {
            return Ok(i);
        }
        let built = self.build_index()?;
        let _ = self.index.set(built);
        Ok(self.index.get().expect("just set"))
    }

    fn build_index(&self) -> Result<SlabIndex> {
        const FOOTER: u64 = 16;
        let file_len = self.storage.total_size();
        if file_len < FOOTER {
            return Err(RecordError::Container(format!(
                "{}: too small to be a slab ({file_len} bytes)",
                self.label
            )));
        }
        // 1. The trailing footer says what the last page is and how big.
        let footer = slabtastic::Footer::read_from(&self.read(file_len - FOOTER, FOOTER)?)
            .map_err(|e| RecordError::Container(format!("{}: footer: {e}", self.label)))?;
        let tail_start = file_len - footer.page_size as u64;
        let tail = self.read(tail_start, footer.page_size as u64)?;

        // 2. That page is either the pages-page itself, or the
        //    namespaces page naming where each namespace's pages-page
        //    lives. An absent namespace is a normal state and reported
        //    as an empty index, not as a broken file.
        let pages = match footer.page_type {
            slabtastic::PageType::Pages => {
                if self.namespace.as_deref().is_some_and(|n| !n.is_empty()) {
                    return Ok(SlabIndex { entries: Vec::new(), total: 0 });
                }
                slabtastic::PagesPage::deserialize(&tail).map_err(|e| {
                    RecordError::Container(format!("{}: pages page: {e}", self.label))
                })?
            }
            _ => {
                let ns = slabtastic::NamespacesPage::deserialize(&tail).map_err(|e| {
                    RecordError::Container(format!("{}: namespaces page: {e}", self.label))
                })?;
                let entries = ns.entries().map_err(|e| {
                    RecordError::Container(format!("{}: namespace entries: {e}", self.label))
                })?;
                let wanted = match self.namespace.as_deref() {
                    Some(name) if !name.is_empty() => entries.iter().find(|e| e.name == name),
                    _ => entries.iter().find(|e| e.name.is_empty()),
                };
                let Some(entry) = wanted else {
                    return Ok(SlabIndex { entries: Vec::new(), total: 0 });
                };
                let at = entry.pages_page_offset as u64;
                let bytes = self.page(at)?;
                slabtastic::PagesPage::deserialize(&bytes).map_err(|e| {
                    RecordError::Container(format!("{}: pages page: {e}", self.label))
                })?
            }
        };

        // 3. Record counts come from consecutive entries; only the last
        //    page has to be read for its own count.
        let entries = pages.sorted_entries_ref().to_vec();
        let total = match entries.last() {
            None => 0,
            Some(last) => {
                let bytes = self.page(last.file_offset as u64)?;
                let n = slabtastic::Page::record_count_from_buf(&bytes).map_err(|e| {
                    RecordError::Container(format!("{}: last page: {e}", self.label))
                })?;
                (last.start_ordinal as u64) + n as u64
            }
        };
        Ok(SlabIndex { entries, total })
    }

    fn count(&self) -> Result<u64> {
        Ok(self.index()?.total)
    }

    /// The record at this container's own ordinal `local`.
    fn record(&self, local: u64) -> Result<std::borrow::Cow<'_, [u8]>> {
        let index = self.index()?;
        // The page whose start ordinal is the greatest not exceeding
        // `local`. Entries are sorted, so this is a binary search.
        let at = match index
            .entries
            .binary_search_by_key(&(local as i64), |e| e.start_ordinal)
        {
            Ok(i) => i,
            Err(0) => return Err(RecordError::OutOfBounds(local)),
            Err(i) => i - 1,
        };
        let entry = &index.entries[at];
        let bytes = self.page(entry.file_offset as u64)?;
        let within = (local as i64 - entry.start_ordinal) as usize;
        let count = slabtastic::Page::record_count_from_buf(&bytes).map_err(|e| {
            RecordError::Container(format!("{}: page record count: {e}", self.label))
        })?;
        if within >= count {
            return Err(RecordError::OutOfBounds(local));
        }
        match bytes {
            // Mapped: the record is a borrow into the source.
            std::borrow::Cow::Borrowed(b) => {
                slabtastic::Page::get_record_ref_from_buf(b, within, count)
                    .map(std::borrow::Cow::Borrowed)
                    .map_err(|e| {
                        RecordError::Container(format!("{}: record {local}: {e}", self.label))
                    })
            }
            // Fetched: one copy of one page, and the record out of it.
            std::borrow::Cow::Owned(b) => {
                slabtastic::Page::get_record_from_buf(&b, within)
                    .map(std::borrow::Cow::Owned)
                    .map_err(|e| {
                        RecordError::Container(format!("{}: record {local}: {e}", self.label))
                    })
            }
        }
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
            containers: self
                .containers
                .iter()
                .map(|c| Container {
                    storage: c.storage.clone(),
                    label: format!("{}:{name}", c.label),
                    namespace: Some(name.to_string()),
                    index: OnceLock::new(),
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
        self.codec.decode(&self.facet.record_bytes(ordinal)?)
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
            None => containers.push(Container {
                storage: storage.storage_handle(),
                label: format!("facet '{facet}'"),
                namespace: namespace.map(str::to_string),
                index: OnceLock::new(),
            }),
            Some(series) => {
                // One container per **shard**, in ordinal order, so the
                // facet's ordinal space is the concatenation the shard
                // map describes.
                for shard in 0..series.shards().entries().len() {
                    let i = series.file_index_of_shard(shard).map_err(|e| {
                        RecordError::Container(format!("facet '{facet}': {e}"))
                    })?;
                    let handle = series.file(i).map_err(|e| {
                        RecordError::Container(format!("facet '{facet}': {e}"))
                    })?;
                    containers.push(Container {
                        storage: handle,
                        label: format!("facet '{facet}' shard {shard}"),
                        namespace: namespace.map(str::to_string),
                        index: OnceLock::new(),
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
