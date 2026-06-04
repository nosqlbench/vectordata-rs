// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The received-range tracker — the linearization core of resumable,
//! concurrent, sparse uploads.
//!
//! An upload of an `N`-byte object is filled by one or more `PATCH`es, each
//! carrying a byte range `[offset, offset+len)`. Per the resumable-upload
//! convention (`local/vecd-resumable-upload-SPEC.md` §4) those `PATCH`es
//! may arrive **out of order and concurrently** — a client may stream the
//! tail before the head, or split the object across several connections.
//!
//! [`ReceivedRanges`] turns that unordered set of written intervals into a
//! single **linearized acknowledgement**: the length of the contiguous
//! prefix that has been durably written from byte 0. That prefix length is
//! exactly the `Upload-Offset` the server acks — it advances only as the
//! gaps before it fill, so a resuming client always learns the furthest
//! point from which the rest of the object is still needed. Chunks written
//! *ahead* of a gap are remembered (so they are not re-requested) but do
//! not move the ack until the gap closes.
//!
//! The structure is **pure and synchronous** — no I/O — so it is directly
//! unit-testable and is the first thing built. The HTTP layer wraps a
//! per-upload instance behind a mutex (only the interval-merge and
//! contiguous-prefix read need mutual exclusion; the sparse byte-writes to
//! disjoint backend regions proceed in parallel).

use std::collections::BTreeMap;

use rusqlite::{params, OptionalExtension};

use crate::db::Db;
use crate::model::VecdError;

/// Tracks which byte ranges of an `N`-byte object have been durably
/// written and reports the contiguous-prefix length (the acknowledgeable
/// `Upload-Offset`).
///
/// Intervals are stored merged and non-overlapping, keyed by start offset,
/// so the contiguous prefix is just the end of the interval anchored at 0
/// (if any). Every mutation re-establishes the merged invariant, so reads
/// are O(1) on the first interval and adds are O(log n + overlap).
#[derive(Clone, Debug)]
pub struct ReceivedRanges {
    /// Object length in bytes. Adds are clamped to this bound.
    total: u64,
    /// Merged, non-overlapping, non-adjacent `[start, end)` intervals,
    /// keyed by `start`. Adjacent intervals are always coalesced, so two
    /// entries never touch.
    intervals: BTreeMap<u64, u64>,
}

impl ReceivedRanges {
    /// A fresh tracker for an `total`-byte object with nothing received.
    pub fn new(total: u64) -> Self {
        ReceivedRanges { total, intervals: BTreeMap::new() }
    }

    /// The object's declared total length.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Record that `[start, start+len)` has been written. Idempotent and
    /// commutative: re-adding an overlapping or duplicate range (a chunk
    /// re-sent on resume) is a no-op, and adjacent ranges coalesce. A range
    /// extending past `total` is clamped; a zero-length add (including one
    /// at/after `total`) does nothing.
    pub fn add(&mut self, start: u64, len: u64) {
        if len == 0 || start >= self.total {
            return;
        }
        let mut lo = start;
        let mut hi = (start.saturating_add(len)).min(self.total);

        // Absorb every existing interval that overlaps or is adjacent to
        // `[lo, hi)`, widening `[lo, hi)` to their union. The candidates are
        // the interval starting at-or-before `hi` and any starting within
        // `[lo, hi]`; an interval can also extend into `[lo, ..)` from a
        // start strictly below `lo`.
        let mut to_remove: Vec<u64> = Vec::new();
        // The interval immediately at or before `lo` may reach into [lo, ..).
        if let Some((&s, &e)) = self.intervals.range(..=lo).next_back() {
            if e >= lo {
                lo = lo.min(s);
                hi = hi.max(e);
                to_remove.push(s);
            }
        }
        // Intervals starting within [lo, hi] are adjacent/overlapping.
        for (&s, &e) in self.intervals.range(lo..=hi) {
            hi = hi.max(e);
            to_remove.push(s);
        }
        for s in to_remove {
            self.intervals.remove(&s);
        }
        self.intervals.insert(lo, hi);
    }

    /// The contiguous bytes written from offset 0 — the linearized
    /// `Upload-Offset` to acknowledge. Zero when byte 0 has not been
    /// written; otherwise the end of the interval anchored at 0.
    pub fn contiguous_prefix(&self) -> u64 {
        match self.intervals.get(&0) {
            Some(&end) => end,
            None => 0,
        }
    }

    /// Has the whole object been received (the contiguous prefix spans it)?
    pub fn is_complete(&self) -> bool {
        self.total == 0 || self.contiguous_prefix() >= self.total
    }

    /// The merged received intervals as `(start, end)` pairs, ascending.
    /// Used to persist/restore upload state across a daemon restart.
    pub fn intervals(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.intervals.iter().map(|(&s, &e)| (s, e))
    }
}

// ── persisted resumable-upload state ────────────────────────────────
//
// A resumable upload's durable record. The received byte ranges live in a
// side table (`upload_ranges`), kept merged; their contiguous prefix from 0
// is the acknowledged `Upload-Offset`. Persisting both lets an upload resume
// across a daemon restart. All mutations run under the server's single DB
// mutex, so the interval merge is naturally serialized even when PATCHes
// arrive concurrently — only the disjoint backend byte-writes run in
// parallel.

/// A persisted in-progress (or just-completed) resumable upload.
#[derive(Clone, Debug)]
pub struct Upload {
    pub upload_id: String,
    /// Storage namespace and namespace-relative key the bytes resolve to.
    pub storage_ns: String,
    pub key: String,
    /// Declared object length (`Upload-Length`).
    pub total_length: u64,
    /// The backend staging blob accumulating the bytes.
    pub staging_key: String,
    /// True once the contiguous prefix reached `total_length` and the object
    /// was committed.
    pub complete: bool,
    /// The committed object's ETag, set at finalize.
    pub etag: Option<String>,
    /// CAS precondition captured at create, evaluated at finalize: an
    /// `If-Match` ETag, or `If-None-Match: *` (must-not-exist).
    pub if_match: Option<String>,
    pub if_none_match: bool,
}

impl Upload {
    /// The full logical request path (`<ns>/<key>`), for re-resolving the
    /// backend and authorizing the write.
    pub fn path(&self) -> String {
        if self.storage_ns.is_empty() {
            self.key.clone()
        } else {
            format!("{}/{}", self.storage_ns, self.key)
        }
    }

    /// The CAS precondition to enforce at finalize.
    pub fn precondition(&self) -> crate::store::Precondition {
        if self.if_none_match {
            crate::store::Precondition::IfNoneMatchStar
        } else if let Some(e) = &self.if_match {
            crate::store::Precondition::IfMatch(e.clone())
        } else {
            crate::store::Precondition::None
        }
    }
}

/// Create a new upload resource. Returns nothing; the caller already holds
/// the generated `upload_id` and `staging_key`.
#[allow(clippy::too_many_arguments)]
pub fn create(
    db: &Db,
    upload_id: &str,
    storage_ns: &str,
    key: &str,
    total_length: u64,
    staging_key: &str,
    if_match: Option<&str>,
    if_none_match: bool,
    created: i64,
) -> Result<(), VecdError> {
    db.conn().execute(
        "INSERT INTO uploads(upload_id,namespace_path,key,total_length,staging_key,created,complete,if_match,if_none_match)
         VALUES(?1,?2,?3,?4,?5,?6,0,?7,?8)",
        params![upload_id, storage_ns, key, total_length as i64, staging_key, created, if_match, if_none_match as i64],
    )?;
    Ok(())
}

/// Look up an upload by id.
pub fn find(db: &Db, upload_id: &str) -> Result<Option<Upload>, VecdError> {
    Ok(db
        .conn()
        .query_row(
            "SELECT upload_id,namespace_path,key,total_length,staging_key,complete,etag,if_match,if_none_match
             FROM uploads WHERE upload_id=?1",
            params![upload_id],
            |r| {
                Ok(Upload {
                    upload_id: r.get(0)?,
                    storage_ns: r.get(1)?,
                    key: r.get(2)?,
                    total_length: r.get::<_, i64>(3)?.max(0) as u64,
                    staging_key: r.get(4)?,
                    // State column: 0 = in progress, 1 = committed,
                    // 2 = finalizing (claimed). Only 1 is "complete".
                    complete: r.get::<_, i64>(5)? == 1,
                    etag: r.get(6)?,
                    if_match: r.get(7)?,
                    if_none_match: r.get::<_, i64>(8)? != 0,
                })
            },
        )
        .optional()?)
}

/// Rebuild the [`ReceivedRanges`] for an upload from its persisted intervals.
pub fn received(db: &Db, upload_id: &str, total: u64) -> Result<ReceivedRanges, VecdError> {
    let mut rr = ReceivedRanges::new(total);
    let mut stmt = db
        .conn()
        .prepare("SELECT start,end FROM upload_ranges WHERE upload_id=?1")?;
    let rows = stmt.query_map(params![upload_id], |r| {
        Ok((r.get::<_, i64>(0)?.max(0) as u64, r.get::<_, i64>(1)?.max(0) as u64))
    })?;
    for row in rows {
        let (s, e) = row?;
        rr.add(s, e.saturating_sub(s));
    }
    Ok(rr)
}

/// Record a received chunk `[start, start+len)`: merge it into the upload's
/// ranges, persist the merged set, and return `(prefix, claimed_finalize)` —
/// the new contiguous-prefix length (the `Upload-Offset` to acknowledge),
/// and whether *this* call atomically won the right to finalize the now-full
/// upload (state `0 → 2`). Idempotent for re-sent chunks. The whole thing
/// runs in one transaction, serialized by the server DB mutex, so concurrent
/// PATCHes that both fill the object cannot both finalize it.
pub fn record_chunk(
    db: &mut Db,
    upload_id: &str,
    start: u64,
    len: u64,
    total: u64,
) -> Result<(u64, bool), VecdError> {
    let mut rr = received(db, upload_id, total)?;
    rr.add(start, len);
    let prefix = rr.contiguous_prefix();
    let merged: Vec<(u64, u64)> = rr.intervals().collect();
    let tx = db.conn_mut().transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    tx.execute("DELETE FROM upload_ranges WHERE upload_id=?1", params![upload_id])?;
    for (s, e) in &merged {
        tx.execute(
            "INSERT INTO upload_ranges(upload_id,start,end) VALUES(?1,?2,?3)",
            params![upload_id, *s as i64, *e as i64],
        )?;
    }
    let mut claimed = false;
    if prefix >= total {
        // Claim finalization exactly once: in progress (0) → finalizing (2).
        let n = tx.execute(
            "UPDATE uploads SET complete=2 WHERE upload_id=?1 AND complete=0",
            params![upload_id],
        )?;
        claimed = n == 1;
    }
    tx.commit()?;
    Ok((prefix, claimed))
}

/// Mark an upload committed and stamp the resulting object ETag (state `2 →
/// 1`). The staging blob is gone by now (promoted to the object); the row +
/// ranges are kept so `HEAD`/idempotent `PATCH` report completion until the
/// upload is deleted.
pub fn mark_complete(db: &Db, upload_id: &str, etag: &str) -> Result<(), VecdError> {
    db.conn().execute(
        "UPDATE uploads SET complete=1, etag=?2 WHERE upload_id=?1",
        params![upload_id, etag],
    )?;
    Ok(())
}

/// Release a finalization claim that failed (CAS/quota/error), returning the
/// upload to in-progress (state `2 → 0`) so it can be retried or deleted.
pub fn reset_finalizing(db: &Db, upload_id: &str) -> Result<(), VecdError> {
    db.conn().execute(
        "UPDATE uploads SET complete=0 WHERE upload_id=?1 AND complete=2",
        params![upload_id],
    )?;
    Ok(())
}

/// Delete an upload and its ranges (abandon, or post-finalize cleanup).
pub fn delete(db: &Db, upload_id: &str) -> Result<(), VecdError> {
    db.conn().execute("DELETE FROM upload_ranges WHERE upload_id=?1", params![upload_id])?;
    db.conn().execute("DELETE FROM uploads WHERE upload_id=?1", params![upload_id])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_object_is_complete() {
        let r = ReceivedRanges::new(0);
        assert_eq!(r.contiguous_prefix(), 0);
        assert!(r.is_complete());
    }

    #[test]
    fn single_full_chunk_completes() {
        let mut r = ReceivedRanges::new(100);
        assert!(!r.is_complete());
        r.add(0, 100);
        assert_eq!(r.contiguous_prefix(), 100);
        assert!(r.is_complete());
    }

    #[test]
    fn in_order_chunks_advance_prefix() {
        let mut r = ReceivedRanges::new(30);
        r.add(0, 10);
        assert_eq!(r.contiguous_prefix(), 10);
        r.add(10, 10);
        assert_eq!(r.contiguous_prefix(), 20);
        r.add(20, 10);
        assert_eq!(r.contiguous_prefix(), 30);
        assert!(r.is_complete());
    }

    #[test]
    fn out_of_order_sparse_prefix_jumps_when_gap_fills() {
        let mut r = ReceivedRanges::new(20);
        // Tail arrives first — prefix can't advance past the leading gap.
        r.add(10, 10);
        assert_eq!(r.contiguous_prefix(), 0);
        assert!(!r.is_complete());
        // Head fills the gap — prefix jumps across both, 0 → 20.
        r.add(0, 10);
        assert_eq!(r.contiguous_prefix(), 20);
        assert!(r.is_complete());
    }

    #[test]
    fn duplicate_and_overlapping_adds_are_idempotent() {
        let mut r = ReceivedRanges::new(100);
        r.add(0, 50);
        r.add(0, 50); // exact duplicate (resend)
        r.add(10, 20); // fully contained
        r.add(40, 20); // overlaps the boundary [40,50) + extends to 60
        assert_eq!(r.contiguous_prefix(), 60);
        // The whole received region is one merged interval.
        assert_eq!(r.intervals().collect::<Vec<_>>(), vec![(0, 60)]);
    }

    #[test]
    fn adjacent_intervals_coalesce() {
        let mut r = ReceivedRanges::new(100);
        r.add(0, 10);
        r.add(20, 10); // [20,30) — a gap [10,20) remains
        assert_eq!(r.intervals().collect::<Vec<_>>(), vec![(0, 10), (20, 30)]);
        assert_eq!(r.contiguous_prefix(), 10);
        // Fill exactly the gap; all three coalesce into one interval.
        r.add(10, 10);
        assert_eq!(r.intervals().collect::<Vec<_>>(), vec![(0, 30)]);
        assert_eq!(r.contiguous_prefix(), 30);
    }

    #[test]
    fn gap_leaves_prefix_at_gap_start() {
        let mut r = ReceivedRanges::new(50);
        r.add(0, 20);
        r.add(30, 20); // [30,50), gap [20,30)
        assert_eq!(r.contiguous_prefix(), 20);
        assert!(!r.is_complete());
    }

    #[test]
    fn add_beyond_total_is_clamped() {
        let mut r = ReceivedRanges::new(30);
        r.add(20, 100); // clamped to [20,30)
        assert_eq!(r.intervals().collect::<Vec<_>>(), vec![(20, 30)]);
        r.add(0, 1000); // clamped to [0,30) — fills everything
        assert_eq!(r.contiguous_prefix(), 30);
        assert!(r.is_complete());
    }

    #[test]
    fn zero_len_and_out_of_bounds_adds_are_noops() {
        let mut r = ReceivedRanges::new(30);
        r.add(5, 0); // zero length
        r.add(30, 10); // starts at total
        r.add(40, 10); // starts past total
        assert_eq!(r.intervals().collect::<Vec<_>>(), Vec::new());
        assert_eq!(r.contiguous_prefix(), 0);
    }

    #[test]
    fn many_out_of_order_fragments_merge_to_complete() {
        // A stress-ish ordering: fill [0,100) via shuffled 10-byte chunks.
        let mut r = ReceivedRanges::new(100);
        for &start in &[30, 0, 90, 10, 70, 50, 20, 80, 40, 60] {
            r.add(start, 10);
        }
        assert_eq!(r.intervals().collect::<Vec<_>>(), vec![(0, 100)]);
        assert_eq!(r.contiguous_prefix(), 100);
        assert!(r.is_complete());
    }

    #[test]
    fn overlapping_spans_bridge_multiple_intervals() {
        let mut r = ReceivedRanges::new(100);
        r.add(0, 10);
        r.add(20, 10);
        r.add(40, 10); // three disjoint islands + the [0,10) prefix
        assert_eq!(r.contiguous_prefix(), 10);
        // One big span swallows all of them and the gaps between.
        r.add(5, 40); // [5,45) bridges [0,10),[20,30),[40,50)
        assert_eq!(r.intervals().collect::<Vec<_>>(), vec![(0, 50)]);
        assert_eq!(r.contiguous_prefix(), 50);
    }
}
