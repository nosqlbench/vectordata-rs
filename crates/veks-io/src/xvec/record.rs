// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! One xvec record in a buffer the caller owns: encoding a row into a
//! record-sized slot ([`encode_record_into`]), reading a slot's header
//! ([`record_dim`]), elements ([`decode_record_into`]) or element bytes
//! ([`record_payload`]), walking the records of mapped bytes strictly
//! ([`records`]), reading one record of a file by `pread`
//! ([`pread_record`]), and putting headers before element bytes that
//! arrive untyped ([`RecordHeader`]).
//!
//! These are for code that lays records out itself — a partition
//! buffer filled out of order, a slot array compacted in place, a read
//! that keeps resident memory bounded by not mapping the file — so the
//! record layout (`[dim: i32]` then `dim` little-endian elements) is
//! still written down only here. A front-to-back write goes through
//! [`XvecWriter`](crate::XvecWriter) instead, and a mapped read through
//! [`XvecReader`](crate::XvecReader).
//!
//! ```
//! use veks_io::xvec::record::{decode_record_into, encode_record_into, record_dim};
//!
//! // Two dim-3 f32 slots, the second left empty.
//! let mut slots = vec![0u8; 2 * (4 + 3 * 4)];
//! let (first, second) = slots.split_at_mut(16);
//! encode_record_into(&[1.0f32, 2.0, 3.0], first).unwrap();
//! assert_eq!(record_dim(first), Some(3));
//! assert_eq!(record_dim(second), None, "an all-zero slot holds no record");
//!
//! let mut row = [0f32; 3];
//! decode_record_into(first, &mut row).unwrap();
//! assert_eq!(row, [1.0, 2.0, 3.0]);
//! ```

use std::fs::File;

use veks_datatypes::Element;

use crate::reader::IoError;

/// Write `row` into `slot` as one xvec record: the dimension, then the
/// elements, little-endian. `slot` must be exactly one record long
/// (`4 + row.len() * T::SIZE` bytes).
#[inline]
pub fn encode_record_into<T: Element>(row: &[T], slot: &mut [u8]) -> Result<(), IoError> {
    let dim = header_of(row.len())?;
    if slot.len() != 4 + row.len() * T::SIZE {
        return Err(IoError::InvalidFormat(format!(
            "a {}-byte slot does not hold one record of {} {}-byte elements",
            slot.len(),
            row.len(),
            T::SIZE
        )));
    }
    let (header, body) = slot.split_at_mut(4);
    header.copy_from_slice(&dim.to_le_bytes());
    for (v, out) in row.iter().zip(body.chunks_exact_mut(T::SIZE)) {
        v.write_le_bytes(out);
    }
    Ok(())
}

/// The dimension a record's header states, or `None` when `bytes` is
/// shorter than a header or the header is not positive — the state of
/// a slot no record was written into.
#[inline]
pub fn record_dim(bytes: &[u8]) -> Option<usize> {
    let header = bytes.get(..4)?;
    let dim = i32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    (dim > 0).then_some(dim as usize)
}

/// Read the elements of the record in `slot` into `out`. The header
/// must state `out.len()` and `slot` be exactly one record long.
#[inline]
pub fn decode_record_into<T: Element>(slot: &[u8], out: &mut [T]) -> Result<(), IoError> {
    if record_dim(slot) != Some(out.len()) || slot.len() != 4 + out.len() * T::SIZE {
        return Err(IoError::InvalidFormat(format!(
            "a {}-byte slot with header {:?} is not one record of {} {}-byte elements",
            slot.len(),
            record_dim(slot),
            out.len(),
            T::SIZE
        )));
    }
    for (v, bytes) in out.iter_mut().zip(slot[4..].chunks_exact(T::SIZE)) {
        *v = T::from_le_bytes(bytes);
    }
    Ok(())
}

/// The element bytes of `record`, one whole record of `elem_size`-byte
/// elements: what follows the header, which must state their number.
/// For a reader that decodes the elements itself, at a type it learns
/// at run time.
#[inline]
pub fn record_payload(record: &[u8], elem_size: usize) -> Result<&[u8], IoError> {
    let payload = record.get(4..).unwrap_or_default();
    match record_dim(record) {
        Some(dim) if elem_size > 0 && dim.checked_mul(elem_size) == Some(payload.len()) => Ok(payload),
        found => Err(IoError::InvalidFormat(format!(
            "a {}-byte record with header {found:?} is not whole in {elem_size}-byte elements",
            record.len()
        ))),
    }
}

/// The records of `bytes` — an xvec file of `elem_size`-byte elements,
/// uniform or variable-length — front to back, as each record's
/// dimension and element bytes, borrowed. Strict where the streaming
/// reader is lenient: a header that is not a positive dimension, or a
/// record cut short, is an error item, and the walk ends after it.
/// Nothing is written beside the file (no offset index).
///
/// ```
/// use veks_io::xvec::record::records;
///
/// let mut bytes = Vec::new();
/// bytes.extend(1i32.to_le_bytes());
/// bytes.extend(7i32.to_le_bytes());
/// bytes.extend(2i32.to_le_bytes());
/// bytes.extend([8i32.to_le_bytes(), 9i32.to_le_bytes()].concat());
/// let dims: Vec<usize> = records(&bytes, 4).map(|r| r.unwrap().0).collect();
/// assert_eq!(dims, [1, 2]);
///
/// bytes.extend(3i32.to_le_bytes()); // a third record, cut short
/// assert!(records(&bytes, 4).last().unwrap().is_err());
/// ```
pub fn records(bytes: &[u8], elem_size: usize) -> Records<'_> {
    Records { bytes, elem_size, offset: 0, failed: false }
}

/// The iterator [`records`] returns.
#[derive(Debug, Clone)]
pub struct Records<'a> {
    bytes: &'a [u8],
    elem_size: usize,
    offset: usize,
    failed: bool,
}

impl<'a> Iterator for Records<'a> {
    type Item = Result<(usize, &'a [u8]), IoError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.offset == self.bytes.len() {
            return None;
        }
        let at = self.offset;
        let rest = &self.bytes[at..];
        let whole = record_dim(rest)
            .and_then(|dim| dim.checked_mul(self.elem_size).filter(|_| self.elem_size > 0).map(|n| (dim, n)))
            .and_then(|(dim, n)| rest.get(4..4 + n).map(|payload| (dim, payload)));
        match whole {
            Some((dim, payload)) => {
                self.offset += 4 + payload.len();
                Some(Ok((dim, payload)))
            }
            None => {
                self.failed = true;
                Some(Err(IoError::InvalidFormat(format!(
                    "the record at byte {at} has header {:?} and {} bytes after it: not a whole record of {}-byte elements",
                    rest.get(..4).map(|h| i32::from_le_bytes([h[0], h[1], h[2], h[3]])),
                    rest.len().saturating_sub(4),
                    self.elem_size
                ))))
            }
        }
    }
}

/// Read record `index` of a uniform xvec file of `out.len()`-element
/// records into `out`, by `pread` — nothing is mapped, so resident
/// memory stays at the caller's buffers. `scratch` holds the record's
/// bytes and is reused across calls. The record's header must state
/// `out.len()`.
pub fn pread_record<T: Element>(file: &File, index: u64, scratch: &mut Vec<u8>, out: &mut [T]) -> Result<(), IoError> {
    let record_bytes = 4 + out.len() * T::SIZE;
    header_of(out.len())?;
    scratch.resize(record_bytes, 0);
    let offset = index
        .checked_mul(record_bytes as u64)
        .ok_or_else(|| IoError::InvalidFormat(format!("record {index} lies past any file")))?;
    crate::file::pread_exact(file, scratch, offset)?;
    decode_record_into(scratch, out)
}

/// A record header for one dimension, checked once, then put before
/// element bytes of any type as they arrive — for a byte-level sink
/// whose records come as raw payloads and whose writes cannot fail.
///
/// ```
/// use veks_io::xvec::record::{RecordHeader, records};
///
/// let header = RecordHeader::new(2).unwrap();
/// let payloads = [1u16.to_le_bytes(), 2u16.to_le_bytes(), 3u16.to_le_bytes(), 4u16.to_le_bytes()].concat();
/// let mut out = Vec::new();
/// header.append_records(&payloads, 4, &mut out); // two 2-element u16 rows
/// let walked: Vec<_> = records(&out, 2).map(|r| r.unwrap()).collect();
/// assert_eq!(walked, [(2, &payloads[..4]), (2, &payloads[4..])]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordHeader([u8; 4]);

impl RecordHeader {
    /// The header of `dim`-element records; `dim` must be 1 to
    /// `i32::MAX`.
    pub fn new(dim: usize) -> Result<Self, IoError> {
        Ok(Self(header_of(dim)?.to_le_bytes()))
    }

    /// The header's four bytes.
    #[inline]
    pub fn bytes(&self) -> [u8; 4] {
        self.0
    }

    /// Append one record to `out`: the header, then `payload`.
    #[inline]
    pub fn append_record(&self, payload: &[u8], out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
        out.extend_from_slice(payload);
    }

    /// Append the records of `packed` — payloads of `payload_bytes`
    /// each, end to end — to `out`, each after the header. `packed` is
    /// whole payloads; the sizing is the caller's, checked in debug
    /// builds.
    #[inline]
    pub fn append_records(&self, packed: &[u8], payload_bytes: usize, out: &mut Vec<u8>) {
        debug_assert!(payload_bytes > 0 && packed.len().is_multiple_of(payload_bytes));
        let stride = 4 + payload_bytes;
        let start = out.len();
        // The zero fill vectorizes, and the loop overwrites every byte.
        out.resize(start + packed.len() / payload_bytes * stride, 0);
        for (slot, payload) in out[start..].chunks_exact_mut(stride).zip(packed.chunks_exact(payload_bytes)) {
            slot[..4].copy_from_slice(&self.0);
            slot[4..].copy_from_slice(payload);
        }
    }
}

/// The header of a `dim`-element record.
#[inline]
fn header_of(dim: usize) -> Result<i32, IoError> {
    i32::try_from(dim)
        .ok()
        .filter(|&d| d > 0)
        .ok_or_else(|| IoError::InvalidFormat(format!("dimension {dim} is not 1 to {}", i32::MAX)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::XvecWriter;

    #[test]
    fn slots_round_trip_every_element_type() {
        fn round_trip<T: Element + PartialEq + std::fmt::Debug>(row: &[T]) {
            let mut slot = vec![0u8; 4 + row.len() * T::SIZE];
            encode_record_into(row, &mut slot).unwrap();
            let mut encoded = Vec::new();
            crate::writer::encode_records(row, row.len(), &mut encoded).unwrap();
            assert_eq!(slot, encoded, "{:?}: a slot is the record encode_records writes", T::TYPE);
            let mut back = vec![row[0]; row.len()];
            decode_record_into(&slot, &mut back).unwrap();
            assert_eq!(back, row);
        }
        round_trip(&[1u8, 255]);
        round_trip(&[-1i8, 7]);
        round_trip(&[1u16, 65535]);
        round_trip(&[-2i16, 3]);
        round_trip(&[1u32, u32::MAX]);
        round_trip(&[-1i32, i32::MAX]);
        round_trip(&[1u64, u64::MAX]);
        round_trip(&[-1i64, 5]);
        round_trip(&[veks_datatypes::f16::from_f32(1.5), veks_datatypes::f16::from_f32(-2.0)]);
        round_trip(&[veks_datatypes::bf16::from_f32(1.5), veks_datatypes::bf16::from_f32(-2.0)]);
        round_trip(&[1.5f32, -2.0]);
        round_trip(&[1.5f64, -2.0]);
    }

    #[test]
    fn a_slot_of_the_wrong_size_or_header_is_refused() {
        let mut short = vec![0u8; 8];
        assert!(encode_record_into(&[1.0f32, 2.0], &mut short).is_err());
        assert!(encode_record_into::<f32>(&[], &mut short).is_err());
        let mut slot = vec![0u8; 12];
        encode_record_into(&[1.0f32, 2.0], &mut slot).unwrap();
        let mut three = [0f32; 3];
        assert!(decode_record_into(&slot, &mut three).is_err(), "the header says 2");
        assert!(decode_record_into(&[0u8; 12], &mut [0f32; 2]).is_err(), "an empty slot");
        assert_eq!(record_dim(&[1, 0, 0]), None);
        assert_eq!(record_dim(&(-1i32).to_le_bytes()), None);
    }

    #[test]
    fn a_payload_is_the_bytes_its_header_counts() {
        let mut slot = vec![0u8; 12];
        encode_record_into(&[1.0f32, 2.0], &mut slot).unwrap();
        assert_eq!(record_payload(&slot, 4).unwrap(), &slot[4..]);
        assert!(record_payload(&slot, 2).is_err(), "8 bytes are 4 two-byte elements, not 2");
        assert!(record_payload(&slot[..10], 4).is_err());
        assert!(record_payload(&[0u8; 12], 4).is_err());
        assert!(record_payload(&slot, 0).is_err());
    }

    #[test]
    fn a_walk_yields_every_record_and_stops_at_the_first_bad_one() {
        let mut bytes = Vec::new();
        for dim in [1usize, 3, 2] {
            let row: Vec<i32> = (0..dim as i32).collect();
            crate::writer::encode_records(&row, dim, &mut bytes).unwrap();
        }
        let walked: Vec<_> = records(&bytes, 4).map(|r| r.unwrap()).collect();
        assert_eq!(walked.iter().map(|r| r.0).collect::<Vec<_>>(), [1, 3, 2]);
        assert_eq!(walked[1].1, &bytes[12..24]);
        assert_eq!(records(&[], 4).count(), 0);

        let mut zero = bytes.clone();
        zero.extend(0i32.to_le_bytes());
        zero.extend(1i32.to_le_bytes());
        zero.extend(5i32.to_le_bytes());
        let items: Vec<_> = records(&zero, 4).collect();
        assert_eq!(items.len(), 4, "three records, then the error, then nothing");
        assert!(items[3].is_err());

        let mut cut = bytes.clone();
        cut.extend([2, 0]);
        assert!(records(&cut, 4).last().unwrap().is_err());
        assert!(records(&bytes, 0).next().unwrap().is_err());
    }

    #[test]
    fn pread_reads_the_record_the_writer_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rows.fvecs");
        let mut w = XvecWriter::<f32>::create(&path, 2).unwrap();
        w.write_packed(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        w.finish().unwrap();

        let file = File::open(&path).unwrap();
        let mut scratch = Vec::new();
        let mut row = [0f32; 2];
        pread_record(&file, 2, &mut scratch, &mut row).unwrap();
        assert_eq!(row, [5.0, 6.0]);
        pread_record(&file, 0, &mut scratch, &mut row).unwrap();
        assert_eq!(row, [1.0, 2.0]);
        assert!(pread_record(&file, 3, &mut scratch, &mut row).is_err(), "past the end");
        assert!(pread_record(&file, 0, &mut scratch, &mut [0f32; 3]).is_err(), "the header says 2");
    }
}
