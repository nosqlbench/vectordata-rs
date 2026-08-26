// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! A virtual page cache, so the model can say what actually reaches the
//! device.
//!
//! The perfscripts measurements this crate prices against are unbuffered
//! (`direct=1`), which makes them the *cold* floor: every read goes to
//! the device. Real pipelines do not run that way. Between the algorithm
//! and the device sits a page cache, and it changes two things the cost
//! model otherwise gets wrong.
//!
//! **It creates the container.** Reading one record faults in the whole
//! page containing it, so neighbouring records come along free. With the
//! page size set to the container size, container amplification stops
//! being an assumption and becomes an *outcome* — the same
//! `A(P) = P · (1 − exp(−w / P))` curve falls out of replaying the access
//! sequence through [`PageCache`], or it does not, and then the formula
//! is wrong. See [`crate::study`] for that comparison.
//!
//! **It breaks the pass model.** `A(P)` assumes each pass re-reads its
//! containers from the device. With enough resident memory, a page
//! touched in one pass is still there in the next, and the re-read never
//! happens. How much memory that takes, and how sharply the benefit
//! falls off, is what the cache simulation is for.
//!
//! Readahead policy — which lives in [`crate::io::Readahead`] and drives
//! this cache — follows Linux's: a window that doubles toward `ra_pages`
//! (128 KiB by default, 256 KiB after `POSIX_FADV_SEQUENTIAL`) and fires
//! once per window when a read crosses an async marker. See
//! [the crate bibliography](crate#sources) for what grounds the rest of
//! the storage path.
//!
//! Residency is a bitmask over the address space — one bit per page —
//! and recency is an intrusive list over cache slots, so both membership
//! and eviction are constant-time and the whole thing stays cheap enough
//! to replay millions of operations.

use crate::model::{Op, Trace};

/// Where a page lives. Input and output occupy one flat page space so
/// that a streaming writer competes with the reader for residency, as it
/// would in a real page cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Input,
    Output,
    /// Scratch the rewrite writes and reads back — the bucket streams a
    /// staged rewrite spills into. It is a real extent on the same
    /// volume, so it contends for cache and for the device exactly as the
    /// other two do.
    Spill,
}

impl Region {
    fn index(self) -> usize {
        match self {
            Region::Input => 0,
            Region::Output => 1,
            Region::Spill => 2,
        }
    }
}

/// How much memory the cache may use, and at what granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheConfig {
    /// Memory available for cached pages.
    pub ram_bytes: u64,
    /// Page size. Sweeping this is the point — it sets how much comes
    /// along free with each fault.
    pub page_bytes: u64,
    /// Whether written pages occupy cache and evict read pages. Real
    /// page caches do; turning it off isolates read behaviour.
    pub writes_occupy: bool,
}

impl CacheConfig {
    pub fn new(ram_bytes: u64, page_bytes: u64) -> Self {
        CacheConfig {
            ram_bytes,
            page_bytes,
            writes_occupy: true,
        }
    }

    /// A cache that retains nothing at all, so every page a request
    /// touches is fetched even if the previous request just touched it.
    /// This is a floor for comparison, not a model of any real system.
    pub fn uncached(page_bytes: u64) -> Self {
        CacheConfig {
            ram_bytes: 0,
            page_bytes,
            writes_occupy: false,
        }
    }

    /// One page of retention: the current block is held until the reader
    /// moves off it, and nothing else is.
    ///
    /// This is the unbuffered case the fio data was measured under —
    /// `direct=1` bypasses the page cache, but a read of `bs` bytes still
    /// delivers all `bs` bytes, and consecutive records inside one block
    /// arrive together. It is also exactly what the `container_touches`
    /// metric counts, which makes it the configuration where the
    /// simulated and table-driven costs should agree.
    pub fn single_page(page_bytes: u64) -> Self {
        CacheConfig {
            ram_bytes: page_bytes,
            page_bytes,
            writes_occupy: false,
        }
    }

    pub fn capacity_pages(&self) -> usize {
        (self.ram_bytes / self.page_bytes.max(1)) as usize
    }
}

/// What a replay cost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Page touches served from memory.
    pub read_hits: u64,
    /// Page touches that had to reach the device.
    pub read_misses: u64,
    pub write_hits: u64,
    pub write_misses: u64,
    pub evictions: u64,
    /// Dirty pages written back to the device: by the background
    /// flusher, by expiry, or because a dirty page reached the cold end
    /// of the LRU and had to be cleaned before its frame could be reused.
    pub writebacks: u64,
    /// Writebacks forced by eviction rather than by the flusher. These
    /// are the expensive ones: they happen on the allocation path, in
    /// whatever order the LRU produces, rather than in a batch the
    /// flusher could have sorted.
    pub eviction_writebacks: u64,
    /// High-water mark of dirty pages held in the cache.
    pub peak_dirty_pages: u64,
    /// Distinct pages faulted in — the cold-cache lower bound on misses.
    pub compulsory_misses: u64,
    /// Pages brought in speculatively by readahead.
    pub readahead_pages: u64,
    /// Readahead pages that were subsequently asked for. The difference
    /// between this and `readahead_pages` is bandwidth spent on data
    /// nobody wanted — the cost of guessing.
    pub readahead_hits: u64,
    pub page_bytes: u64,
    pub capacity_pages: u64,
}

impl CacheStats {
    /// Bytes that actually reached the device on the read path.
    pub fn read_bytes_from_device(&self) -> u64 {
        self.read_misses * self.page_bytes
    }

    pub fn read_hit_rate(&self) -> f64 {
        let total = self.read_hits + self.read_misses;
        if total == 0 {
            0.0
        } else {
            self.read_hits as f64 / total as f64
        }
    }

    /// Misses beyond the unavoidable first touch of each page. Zero means
    /// the cache was large enough that nothing was ever re-fetched.
    pub fn capacity_misses(&self) -> u64 {
        self.read_misses.saturating_sub(self.compulsory_misses)
    }

    /// Pages readahead fetched that were never asked for.
    pub fn wasted_readahead_pages(&self) -> u64 {
        self.readahead_pages.saturating_sub(self.readahead_hits)
    }

    /// Share of readahead that paid for itself.
    pub fn readahead_precision(&self) -> f64 {
        if self.readahead_pages == 0 {
            0.0
        } else {
            self.readahead_hits as f64 / self.readahead_pages as f64
        }
    }
}

const NONE: u32 = u32::MAX;

/// An LRU page cache over a flat virtual address space.
pub struct PageCache {
    config: CacheConfig,
    /// One bit per page of the address space: is it resident?
    resident: Vec<u64>,
    /// Pages ever faulted in, for separating compulsory from capacity
    /// misses.
    ever_seen: Vec<u64>,
    /// Pages that arrived speculatively and have not yet been asked for.
    speculative: Vec<u64>,
    /// Pages modified in memory and not yet written back.
    dirty: Vec<u64>,
    /// Dirty pages in the order they were first dirtied, with the time
    /// each was dirtied — what the periodic flusher walks to find the
    /// ones older than `dirty_expire_centisecs`.
    dirty_queue: std::collections::VecDeque<(u64, f64)>,
    dirty_pages: u64,
    /// Pages the caller must write back before the run can proceed,
    /// drained by [`Self::take_writebacks`].
    pending_writeback: Vec<u64>,
    /// Which page occupies each slot.
    slot_page: Vec<u64>,
    /// Which slot holds each resident page. Sized to the page space
    /// because residency is already tracked there; only entries whose
    /// `resident` bit is set are meaningful.
    page_slot: Vec<u32>,
    prev: Vec<u32>,
    next: Vec<u32>,
    head: u32,
    tail: u32,
    used: usize,
    /// First page index of each region, indexed by `Region::index`.
    region_base: [u64; 3],
    stats: CacheStats,
}

impl PageCache {
    /// Build a cache over an address space of `input_bytes` of source
    /// followed by `output_bytes` of destination.
    pub fn new(config: CacheConfig, input_bytes: u64, output_bytes: u64) -> Self {
        Self::with_spill(config, input_bytes, output_bytes, 0)
    }

    /// A cache over three extents: source, output, and the spill scratch a
    /// staged rewrite uses.
    pub fn with_spill(
        config: CacheConfig,
        input_bytes: u64,
        output_bytes: u64,
        spill_bytes: u64,
    ) -> Self {
        let page_bytes = config.page_bytes.max(1);
        let input_pages = input_bytes.div_ceil(page_bytes);
        let output_pages = output_bytes.div_ceil(page_bytes);
        let spill_pages = spill_bytes.div_ceil(page_bytes);
        let region_base = [0, input_pages, input_pages + output_pages];
        let total_pages = (input_pages + output_pages + spill_pages) as usize;
        let capacity = config.capacity_pages().min(total_pages);

        PageCache {
            config,
            resident: vec![0u64; total_pages.div_ceil(64)],
            ever_seen: vec![0u64; total_pages.div_ceil(64)],
            speculative: vec![0u64; total_pages.div_ceil(64)],
            dirty: vec![0u64; total_pages.div_ceil(64)],
            dirty_queue: std::collections::VecDeque::new(),
            dirty_pages: 0,
            pending_writeback: Vec::new(),
            slot_page: vec![0; capacity],
            page_slot: vec![NONE; total_pages],
            prev: vec![NONE; capacity],
            next: vec![NONE; capacity],
            head: NONE,
            tail: NONE,
            used: 0,
            region_base,
            stats: CacheStats {
                page_bytes,
                capacity_pages: capacity as u64,
                ..CacheStats::default()
            },
        }
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    fn bit(set: &[u64], page: u64) -> bool {
        set[(page / 64) as usize] & (1u64 << (page % 64)) != 0
    }

    fn set_bit(set: &mut [u64], page: u64, on: bool) {
        let word = &mut set[(page / 64) as usize];
        let mask = 1u64 << (page % 64);
        if on { *word |= mask } else { *word &= !mask }
    }

    /// Number of resident pages, counted from the bitmask.
    pub fn resident_pages(&self) -> u32 {
        self.resident.iter().map(|w| w.count_ones()).sum()
    }

    fn unlink(&mut self, slot: u32) {
        let (p, n) = (self.prev[slot as usize], self.next[slot as usize]);
        if p != NONE {
            self.next[p as usize] = n
        } else {
            self.head = n
        }
        if n != NONE {
            self.prev[n as usize] = p
        } else {
            self.tail = p
        }
        self.prev[slot as usize] = NONE;
        self.next[slot as usize] = NONE;
    }

    fn push_front(&mut self, slot: u32) {
        self.next[slot as usize] = self.head;
        self.prev[slot as usize] = NONE;
        if self.head != NONE {
            self.prev[self.head as usize] = slot;
        }
        self.head = slot;
        if self.tail == NONE {
            self.tail = slot;
        }
    }

    /// Touch one page. Returns true on a hit.
    fn touch_page(&mut self, page: u64, write: bool, now: f64) -> bool {
        let counts_for_residency = !write || self.config.writes_occupy;

        if Self::bit(&self.resident, page) {
            let slot = self.page_slot[page as usize];
            self.unlink(slot);
            self.push_front(slot);
            if Self::bit(&self.speculative, page) {
                Self::set_bit(&mut self.speculative, page, false);
                self.stats.readahead_hits += 1;
            }
            if write {
                self.stats.write_hits += 1;
                self.mark_dirty(page, now);
            } else {
                self.stats.read_hits += 1
            }
            return true;
        }

        if write {
            self.stats.write_misses += 1
        } else {
            self.stats.read_misses += 1
        }
        if !Self::bit(&self.ever_seen, page) {
            Self::set_bit(&mut self.ever_seen, page, true);
            if !write {
                self.stats.compulsory_misses += 1;
            }
        }

        if !counts_for_residency || self.stats.capacity_pages == 0 {
            return false;
        }

        let slot = if self.used < self.slot_page.len() {
            let s = self.used as u32;
            self.used += 1;
            s
        } else {
            self.evict_tail()
        };

        self.slot_page[slot as usize] = page;
        self.page_slot[page as usize] = slot;
        Self::set_bit(&mut self.resident, page, true);
        self.push_front(slot);
        if write {
            self.mark_dirty(page, now);
        }
        false
    }

    /// Evict the coldest page and return its freed slot.
    ///
    /// **A dirty page cannot simply be dropped.** Its frame is only
    /// reusable once its contents have reached the device, so evicting
    /// one turns into a write on the allocation path — issued in LRU
    /// order, which is not an order the device likes, and issued while
    /// something is waiting for the frame. This is precisely the cost
    /// the background flusher exists to avoid, and counting it
    /// separately from flusher writebacks is what makes the difference
    /// visible.
    fn evict_tail(&mut self) -> u32 {
        let victim = self.tail;
        let old = self.slot_page[victim as usize];
        if Self::bit(&self.dirty, old) {
            self.clean_page(old);
            self.stats.eviction_writebacks += 1;
            self.stats.writebacks += 1;
            self.pending_writeback.push(old);
        }
        Self::set_bit(&mut self.resident, old, false);
        Self::set_bit(&mut self.speculative, old, false);
        self.page_slot[old as usize] = NONE;
        self.stats.evictions += 1;
        self.unlink(victim);
        victim
    }

    /// Mark a resident page modified in memory.
    fn mark_dirty(&mut self, page: u64, now: f64) {
        if Self::bit(&self.dirty, page) {
            return;
        }
        Self::set_bit(&mut self.dirty, page, true);
        self.dirty_pages += 1;
        self.stats.peak_dirty_pages = self.stats.peak_dirty_pages.max(self.dirty_pages);
        self.dirty_queue.push_back((page, now));
    }

    /// Clear a page's dirty bit, without deciding who writes it back.
    fn clean_page(&mut self, page: u64) {
        if !Self::bit(&self.dirty, page) {
            return;
        }
        Self::set_bit(&mut self.dirty, page, false);
        self.dirty_pages = self.dirty_pages.saturating_sub(1);
    }

    /// Pages currently dirty in memory.
    pub fn dirty_pages(&self) -> u64 {
        self.dirty_pages
    }

    /// Bytes currently dirty in memory — what `balance_dirty_pages`
    /// compares against its thresholds.
    pub fn dirty_bytes(&self) -> u64 {
        self.dirty_pages * self.config.page_bytes
    }

    /// Take the pages that must be written back before their frames can
    /// be reused. These are already marked clean; the caller owes the
    /// device write.
    pub fn take_writebacks(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.pending_writeback)
    }

    /// Hand the flusher dirty pages as **coalesced runs**, bounded both
    /// by how many pages it may take and by how many device requests
    /// those pages are allowed to become.
    ///
    /// Both bounds are real. Linux's flusher works in batches
    /// (`nr_to_write`, typically 1024 pages) and its submissions share
    /// the device queue with everything else, so a batch that fragments
    /// into a thousand separate writes is not one it can issue at once.
    /// Pages that do not fit inside `max_runs` are handed back — marked
    /// dirty again, at the front of the queue — rather than silently
    /// dropped, because a page the flusher declined to write is still
    /// dirty and still has to go somewhere.
    ///
    /// The asymmetry this produces is the point. A sequential writer's
    /// dirty pages are contiguous, so a thousand of them coalesce into
    /// one request and the whole batch goes at once. A scattered
    /// writer's are not, so the same thousand pages need a thousand
    /// requests, the run cap bites, and the drain rate collapses to what
    /// the device can do with small scattered writes.
    pub fn flush_dirty_runs(
        &mut self,
        max_pages: usize,
        max_runs: usize,
        older_than: Option<f64>,
    ) -> Vec<(u64, u64)> {
        if max_pages == 0 || max_runs == 0 {
            return Vec::new();
        }
        let mut pages = self.flush_dirty(max_pages, older_than);
        if pages.is_empty() {
            return Vec::new();
        }
        pages.sort_unstable();

        let mut runs: Vec<(u64, u64)> = Vec::new();
        let mut taken = 0usize;
        for (index, page) in pages.iter().copied().enumerate() {
            match runs.last_mut() {
                Some((start, count)) if *start + *count == page => *count += 1,
                _ => {
                    if runs.len() == max_runs {
                        // Everything from here on is handed back.
                        for &page in &pages[index..] {
                            self.redirty(page);
                        }
                        return runs;
                    }
                    runs.push((page, 1));
                }
            }
            taken = index + 1;
        }
        let _ = taken;
        runs
    }

    /// Put a page back on the dirty list — the flusher looked at it and
    /// could not take it.
    fn redirty(&mut self, page: u64) {
        if Self::bit(&self.dirty, page) || !Self::bit(&self.resident, page) {
            return;
        }
        Self::set_bit(&mut self.dirty, page, true);
        self.dirty_pages += 1;
        self.stats.writebacks = self.stats.writebacks.saturating_sub(1);
        self.dirty_queue.push_front((page, 0.0));
    }

    /// Hand the flusher up to `limit` dirty pages, oldest first,
    /// optionally only those dirtied before `older_than`.
    ///
    /// Two callers want this and they want different things from it. The
    /// periodic `kupdate` flusher passes an age, because its job is to
    /// bound how long data sits unwritten
    /// (`dirty_expire_centisecs`, 30 s). Background writeback passes
    /// none, because its job is to get the dirty total back under the
    /// threshold and it does not care how old the pages are.
    pub fn flush_dirty(&mut self, limit: usize, older_than: Option<f64>) -> Vec<u64> {
        let mut out = Vec::new();
        while out.len() < limit {
            let Some(&(page, dirtied_at)) = self.dirty_queue.front() else {
                break;
            };
            if let Some(cutoff) = older_than
                && dirtied_at > cutoff
            {
                break;
            }
            self.dirty_queue.pop_front();
            // The page may have been cleaned by an eviction already.
            if Self::bit(&self.dirty, page) {
                self.clean_page(page);
                self.stats.writebacks += 1;
                out.push(page);
            }
        }
        out
    }

    /// Bring a page in speculatively.
    ///
    /// Unlike [`Self::access`] this is nobody's request: it does not
    /// count as a hit or a miss, because the cost of fetching it is
    /// charged to the readahead request the caller issues. What it does
    /// record is that the page arrived on spec, so that a later hit on
    /// it can be credited to readahead and an eviction without a hit can
    /// be counted as waste.
    fn insert_speculative(&mut self, page: u64) -> bool {
        if Self::bit(&self.resident, page) {
            return false;
        }
        if self.stats.capacity_pages == 0 {
            return false;
        }
        Self::set_bit(&mut self.ever_seen, page, true);

        let slot = if self.used < self.slot_page.len() {
            let s = self.used as u32;
            self.used += 1;
            s
        } else {
            self.evict_tail()
        };

        self.slot_page[slot as usize] = page;
        self.page_slot[page as usize] = slot;
        Self::set_bit(&mut self.resident, page, true);
        Self::set_bit(&mut self.speculative, page, true);
        self.push_front(slot);
        self.stats.readahead_pages += 1;
        true
    }

    /// Prefetch the pages covering a byte range, returning how many were
    /// not already resident — that is, how much the readahead request
    /// actually has to move.
    pub fn prefetch(&mut self, region: Region, offset: u64, len: u64) -> u64 {
        if len == 0 {
            return 0;
        }
        let page_bytes = self.config.page_bytes;
        let base = self.region_base[region.index()];
        let first = base + offset / page_bytes;
        let last = base + (offset + len - 1) / page_bytes;
        let total = (self.resident.len() * 64) as u64;
        let mut fetched = 0;
        for page in first..=last.min(total.saturating_sub(1)) {
            if self.insert_speculative(page) {
                fetched += 1;
            }
        }
        fetched
    }

    /// Which pages covering a range are absent, as contiguous runs of
    /// `(page_index, count)`.
    ///
    /// Runs matter: a request spanning several missing pages is one I/O,
    /// not one per page, and treating it otherwise inflates the operation
    /// count of every large read.
    pub fn missing_runs(&self, region: Region, offset: u64, len: u64) -> Vec<(u64, u64)> {
        if len == 0 {
            return Vec::new();
        }
        let page_bytes = self.config.page_bytes;
        let base = self.region_base[region.index()];
        let first = base + offset / page_bytes;
        let last = base + (offset + len - 1) / page_bytes;

        let mut runs: Vec<(u64, u64)> = Vec::new();
        for page in first..=last {
            if Self::bit(&self.resident, page) {
                continue;
            }
            match runs.last_mut() {
                Some((start, count)) if *start + *count == page => *count += 1,
                _ => runs.push((page, 1)),
            }
        }
        runs
    }

    /// Byte offset of a page within its region.
    pub fn page_offset(&self, page: u64) -> u64 {
        let base = self
            .region_base
            .iter()
            .copied()
            .filter(|b| *b <= page)
            .max()
            .unwrap_or(0);
        (page - base) * self.config.page_bytes
    }

    /// Touch every page covering `[offset, offset + len)` in `region`.
    ///
    /// `now` timestamps any page this dirties, so the flusher can later
    /// find the ones that have aged past `dirty_expire_centisecs`.
    pub fn access_at(&mut self, region: Region, offset: u64, len: u64, write: bool, now: f64) {
        if len == 0 {
            return;
        }
        let page_bytes = self.config.page_bytes;
        let base = self.region_base[region.index()];
        let first = base + offset / page_bytes;
        let last = base + (offset + len - 1) / page_bytes;
        for page in first..=last {
            self.touch_page(page, write, now);
        }
    }

    /// The same, on a run with no clock — every page dirtied is stamped
    /// at time zero, so an expiry-driven flush treats them all as due.
    pub fn access(&mut self, region: Region, offset: u64, len: u64, write: bool) {
        self.access_at(region, offset, len, write, 0.0);
    }
}

/// Replay a trace through a page cache.
///
/// The algorithms record what they asked for, not how it was served, so
/// the same trace can be replayed at any RAM size and page size. That
/// separation is deliberate: it means a change in cache behaviour cannot
/// quietly change what an algorithm did.
pub fn replay(trace: &Trace, config: CacheConfig) -> CacheStats {
    let g = trace.geometry;
    let payload = g.payload_bytes();
    let mut cache = PageCache::new(config, payload, payload);

    for op in &trace.ops {
        match *op {
            Op::ReadRecord { ordinal } => {
                cache.access(
                    Region::Input,
                    ordinal * g.record_bytes,
                    g.record_bytes,
                    false,
                );
            }
            Op::WriteRange {
                first_slot,
                records,
            } => {
                cache.access(
                    Region::Output,
                    first_slot * g.record_bytes,
                    records * g.record_bytes,
                    true,
                );
            }
            _ => {}
        }
    }
    cache.stats()
}

#[cfg(all(test, feature = "heavy-tests"))]
mod tests {
    use super::*;
    use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather};
    use crate::model::{Geometry, Map};

    fn geo() -> Geometry {
        Geometry {
            records: 20_000,
            record_bytes: 512,
            container_bytes: 65_536,
        }
    }

    #[test]
    fn residency_never_exceeds_the_configured_ram() {
        let g = geo();
        let payload = g.payload_bytes();
        let config = CacheConfig::new(payload / 8, 4_096);
        let mut cache = PageCache::new(config, payload, payload);

        for ordinal in 0..g.records {
            cache.access(
                Region::Input,
                ordinal * g.record_bytes,
                g.record_bytes,
                false,
            );
            assert!(
                cache.resident_pages() as u64 * config.page_bytes <= config.ram_bytes,
                "cache exceeded its RAM budget"
            );
        }
    }

    #[test]
    fn the_bitmask_and_the_slot_list_agree() {
        let g = geo();
        let payload = g.payload_bytes();
        let config = CacheConfig::new(payload / 4, 4_096);
        let mut cache = PageCache::new(config, payload, payload);
        let map = Map::shuffled(g.records, 7);

        for i in 0..g.records {
            cache.access(
                Region::Input,
                map.0[i as usize] * g.record_bytes,
                g.record_bytes,
                false,
            );
        }
        assert_eq!(cache.resident_pages() as usize, cache.used);
    }

    /// With no memory to retain anything, every page touch reaches the
    /// device — which is exactly the condition the fio numbers were
    /// measured under.
    #[test]
    fn an_empty_cache_reproduces_the_unbuffered_case() {
        let g = geo();
        let map = Map::identity(g.records);
        let (_, trace) = NaiveGather.run(g, &map, g.record_bytes * 64);
        let stats = replay(&trace, CacheConfig::uncached(4_096));

        assert_eq!(stats.read_hits, 0, "nothing can be retained");
        assert_eq!(stats.read_misses, g.records, "one miss per record read");
    }

    /// A larger page brings more neighbours along, so sequential access
    /// gets cheaper in device traffic terms as the page grows — the
    /// mechanism the container model describes.
    #[test]
    fn bigger_pages_serve_sequential_access_with_fewer_faults() {
        let g = geo();
        let map = Map::identity(g.records);
        let (_, trace) = NaiveGather.run(g, &map, g.record_bytes * 64);

        let mut previous = u64::MAX;
        for page_bytes in [512u64, 4_096, 16_384, 65_536] {
            let stats = replay(&trace, CacheConfig::new(page_bytes * 4, page_bytes));
            assert!(
                stats.read_misses < previous,
                "page {page_bytes}: {} misses, expected fewer than {previous}",
                stats.read_misses
            );
            previous = stats.read_misses;
        }
    }

    /// The claim that matters for the pass model: give the cache enough
    /// memory to hold the source and multi-pass re-reads stop costing
    /// anything.
    #[test]
    fn a_cache_that_holds_the_source_eliminates_repeat_passes() {
        let g = geo();
        let map = Map::shuffled(g.records, 42);
        let (_, trace) = Gsplat::new().run(g, &map, g.payload_bytes() / 8);
        let m = trace.metrics();
        assert!(
            m.passes >= 4,
            "need a multi-pass run to test reuse, got {}",
            m.passes
        );

        let page = 4_096;
        let starved = replay(&trace, CacheConfig::new(page * 4, page));
        let ample = replay(&trace, CacheConfig::new(g.payload_bytes() * 2, page));

        assert!(starved.capacity_misses() > 0, "a tiny cache must re-fetch");
        assert_eq!(ample.capacity_misses(), 0, "an ample cache must not");
        assert!(
            ample.read_misses < starved.read_misses,
            "ample {} vs starved {}",
            ample.read_misses,
            starved.read_misses
        );
    }

    /// **A partial cache is worth nothing here.** Each gsplat pass scans
    /// the source in ascending order, so the access pattern is a cyclic
    /// sequential scan — the case LRU handles worst. Every page is
    /// evicted shortly before it would next be wanted, and the hit rate
    /// collapses to zero at any size below the whole source.
    ///
    /// This matters more than it looks. It says the cold-cache
    /// amplification model is not merely a pessimistic bound for
    /// multi-pass runs: it is what actually happens, unless memory can
    /// hold the entire source — in which case there was no need to make
    /// multiple passes at all.
    #[test]
    fn an_lru_cache_smaller_than_the_source_gives_a_multipass_scan_nothing() {
        let g = geo();
        let map = Map::shuffled(g.records, 99);
        let (_, trace) = Gsplat::new().run(g, &map, g.payload_bytes() / 8);
        let payload = g.payload_bytes();
        let page = 4_096;

        // Within a page, consecutive records hit regardless of cache size,
        // so raw hit counts are not the measure. Cross-pass reuse is, and
        // that is what stays at zero.
        let misses_at: Vec<u64> = [8u64, 4, 2]
            .iter()
            .map(|d| {
                replay(
                    &trace,
                    CacheConfig {
                        ram_bytes: payload / d,
                        page_bytes: page,
                        writes_occupy: false,
                    },
                )
                .read_misses
            })
            .collect();

        assert!(
            misses_at.windows(2).all(|w| w[0] == w[1]),
            "quadrupling the cache changed nothing, as LRU on a cyclic scan must: {misses_at:?}"
        );

        // What the re-fetching costs is not `passes × pages`: it is the
        // amplification formula, evaluated at *page* granularity. The
        // cache knows nothing of that formula — it just runs LRU over a
        // recorded access sequence — so agreement here is an independent
        // confirmation, at a granularity the formula was never fitted to.
        let passes = trace.metrics().passes as f64;
        let records_per_page = (page / g.record_bytes) as f64;
        let predicted = passes * (1.0 - (-records_per_page / passes).exp());

        let starved = replay(
            &trace,
            CacheConfig {
                ram_bytes: payload / 8,
                page_bytes: page,
                writes_occupy: false,
            },
        );
        let observed = starved.read_misses as f64 / starved.compulsory_misses as f64;
        assert!(
            (observed - predicted).abs() / predicted < 0.10,
            "page-level amplification observed {observed:.2}, formula predicts {predicted:.2}"
        );

        let whole = replay(
            &trace,
            CacheConfig {
                ram_bytes: payload * 2,
                page_bytes: page,
                writes_occupy: false,
            },
        );
        assert_eq!(
            whole.capacity_misses(),
            0,
            "holding the whole source finally eliminates re-fetching"
        );
        assert!(whole.read_misses < starved.read_misses / 4);
    }

    /// A streaming writer evicts the reader's pages. This is the page
    /// cache's version of the starvation the mixed fio sweep measures at
    /// the device — and it shows up precisely where the cache was
    /// otherwise working.
    #[test]
    fn a_streaming_writer_evicts_the_readers_pages() {
        let g = geo();
        let map = Map::shuffled(g.records, 99);
        let (_, trace) = Gsplat::new().run(g, &map, g.payload_bytes() / 8);
        let page = 4_096;
        // Sized so the source alone fits, which is the only regime where
        // the reader had anything to lose.
        let ram = g.payload_bytes() + page * 16;

        let shared = replay(
            &trace,
            CacheConfig {
                ram_bytes: ram,
                page_bytes: page,
                writes_occupy: true,
            },
        );
        let reads_only = replay(
            &trace,
            CacheConfig {
                ram_bytes: ram,
                page_bytes: page,
                writes_occupy: false,
            },
        );

        assert!(
            reads_only.read_hits > 0,
            "the reader must be getting hits to lose any"
        );
        assert!(
            shared.read_misses > reads_only.read_misses,
            "writes competing for cache should cost the reader: {} vs {}",
            shared.read_misses,
            reads_only.read_misses
        );
    }
}
