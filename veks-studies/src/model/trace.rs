// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The record of what an algorithm did, and the cost computed from it.
//!
//! Every algorithm in this crate reports its work as a sequence of
//! [`Op`]s. Nothing is timed and nothing is measured against a device;
//! costs are *derived* from the trace, which is what makes them exact
//! and reproducible. A device model can be layered on later by assigning
//! prices to the counts in [`Metrics`].

use super::Geometry;

/// One virtual operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// A pass begins. Everything until the next `PassStart` belongs to it.
    PassStart { pass: u64 },
    /// A window of the reorder map was read: `count` ordinals.
    ReadMap { from: u64, count: u64 },
    /// One source record was requested by ordinal.
    ReadRecord { ordinal: u64 },
    /// One record was placed into the segment buffer at a local slot.
    Scatter { local: u64 },
    /// A contiguous range of output was written, in records.
    WriteRange { first_slot: u64, records: u64 },
    /// A run of records was appended to a spill bucket's stream.
    ///
    /// Spilling is what separates a staged rewrite from one that
    /// re-reads: instead of sweeping the source once per destination
    /// segment, a single sweep routes every record into the bucket it
    /// belongs to, and each bucket is buffered so its writes are large.
    SpillWrite { bucket: u64, records: u64 },
    /// A run of records was read back from a spill bucket's stream.
    SpillRead { bucket: u64, records: u64 },
    /// A durability barrier was taken.
    Barrier,
}

/// Bytes of destination ordinal carried with each spilled record.
pub const SPILL_TAG_BYTES: u64 = 8;

/// An ordered log of operations plus the geometry they ran against.
#[derive(Debug, Clone)]
pub struct Trace {
    pub geometry: Geometry,
    pub ops: Vec<Op>,
    /// Peak virtual resident bytes the algorithm claimed at any moment.
    pub peak_resident_bytes: u64,
}

impl Trace {
    pub fn new(geometry: Geometry) -> Self {
        Trace {
            geometry,
            ops: Vec::new(),
            peak_resident_bytes: 0,
        }
    }

    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    pub fn claim_resident(&mut self, bytes: u64) {
        self.peak_resident_bytes = self.peak_resident_bytes.max(bytes);
    }

    /// The ops of each pass, in order. Ops before the first `PassStart`
    /// belong to no pass and are ignored here.
    pub fn passes(&self) -> Vec<Vec<Op>> {
        let mut out: Vec<Vec<Op>> = Vec::new();
        for op in &self.ops {
            match op {
                Op::PassStart { .. } => out.push(Vec::new()),
                other => {
                    if let Some(current) = out.last_mut() {
                        current.push(*other);
                    }
                }
            }
        }
        out
    }

    /// Source ordinals read during each pass, in the order requested.
    pub fn reads_per_pass(&self) -> Vec<Vec<u64>> {
        self.passes()
            .into_iter()
            .map(|ops| {
                ops.iter()
                    .filter_map(|op| match op {
                        Op::ReadRecord { ordinal } => Some(*ordinal),
                        _ => None,
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute costs from the log.
    pub fn metrics(&self) -> Metrics {
        let g = self.geometry;
        let mut m = Metrics {
            geometry: g,
            ..Default::default()
        };

        for ops in self.passes() {
            m.passes += 1;
            // A container is "touched" each time the read sequence
            // arrives at a container it was not just in. Under ascending
            // access that counts each needed container exactly once;
            // under scattered access it counts re-entries too, which is
            // precisely the cost being modelled.
            let mut last_container: Option<u64> = None;
            let mut last_ordinal: Option<u64> = None;
            for op in ops {
                match op {
                    Op::ReadRecord { ordinal } => {
                        m.record_reads += 1;
                        let c = g.container_of(ordinal);
                        if last_container != Some(c) {
                            m.container_touches += 1;
                            last_container = Some(c);
                        }
                        if let Some(prev) = last_ordinal
                            && ordinal < prev
                        {
                            m.backward_steps += 1;
                        }
                        last_ordinal = Some(ordinal);
                    }
                    Op::ReadMap { count, .. } => m.map_ordinals_read += count,
                    Op::Scatter { .. } => m.scatters += 1,
                    Op::WriteRange { records, .. } => {
                        m.write_ranges += 1;
                        m.records_written += records;
                    }
                    Op::SpillWrite { records, .. } => {
                        m.spill_runs += 1;
                        m.records_spilled += records;
                    }
                    Op::SpillRead { records, .. } => {
                        m.spill_runs += 1;
                        m.records_unspilled += records;
                    }
                    Op::Barrier => m.barriers += 1,
                    Op::PassStart { .. } => {}
                }
            }
        }

        m.peak_resident_bytes = self.peak_resident_bytes;
        m
    }
}

/// Costs derived from a trace. Counts, not seconds — a device model can
/// price them, but the counts are what the formulas predict.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Metrics {
    pub geometry: Geometry,
    pub passes: u64,
    /// Records requested from the source, summed over passes.
    pub record_reads: u64,
    /// Container fetches implied by the read sequence.
    pub container_touches: u64,
    /// Times a read went backwards relative to the previous read within
    /// the same pass. The monotone-access invariant says zero.
    pub backward_steps: u64,
    /// Ordinals read from the reorder map.
    pub map_ordinals_read: u64,
    pub scatters: u64,
    pub write_ranges: u64,
    pub records_written: u64,
    /// Contiguous runs written to or read back from spill buckets.
    pub spill_runs: u64,
    /// Records routed into spill buckets.
    pub records_spilled: u64,
    /// Records read back out of spill buckets.
    pub records_unspilled: u64,
    pub barriers: u64,
    pub peak_resident_bytes: u64,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            records: 1,
            record_bytes: 1,
            container_bytes: 1,
        }
    }
}

impl Metrics {
    /// Bytes the tier moves to serve the reads, at container granularity.
    pub fn bytes_read(&self) -> u64 {
        self.container_touches * self.geometry.container_bytes
    }

    /// Bytes written to the output, which is the live payload exactly
    /// once. Spill traffic is counted separately by [`Self::spill_bytes`]
    /// so that a staged rewrite's scratch is never mistaken for output.
    pub fn bytes_written(&self) -> u64 {
        self.records_written * self.geometry.record_bytes
    }

    /// Bytes moved to and from the spill extent. A spilled record carries
    /// its destination ordinal alongside its payload, which is the cost
    /// of not having to consult the map again on the way back.
    pub fn spill_bytes(&self) -> u64 {
        (self.records_spilled + self.records_unspilled)
            * (self.geometry.record_bytes + SPILL_TAG_BYTES)
    }

    /// Measured read amplification: tier bytes moved per live byte.
    pub fn amplification(&self) -> f64 {
        let live = self.geometry.payload_bytes() as f64;
        if live == 0.0 {
            return 0.0;
        }
        self.bytes_read() as f64 / live
    }

    /// What the published formula predicts for this pass count.
    pub fn predicted_amplification(&self) -> f64 {
        self.geometry.predicted_amplification(self.passes)
    }
}

#[cfg(all(test, feature = "heavy-tests"))]
mod tests {
    use super::*;

    fn geom() -> Geometry {
        // 10 records per container, 100 records, 10 containers.
        Geometry::new(100, 10, 100)
    }

    #[test]
    fn ascending_reads_touch_each_container_once() {
        let mut t = Trace::new(geom());
        t.push(Op::PassStart { pass: 0 });
        for ordinal in 0..100 {
            t.push(Op::ReadRecord { ordinal });
        }
        let m = t.metrics();
        assert_eq!(m.record_reads, 100);
        assert_eq!(m.container_touches, 10, "one touch per container");
        assert_eq!(m.backward_steps, 0);
    }

    #[test]
    fn scattered_reads_re_enter_containers_and_step_backwards() {
        let mut t = Trace::new(geom());
        t.push(Op::PassStart { pass: 0 });
        // Alternate between the first and last container.
        for i in 0..10 {
            t.push(Op::ReadRecord { ordinal: i });
            t.push(Op::ReadRecord { ordinal: 90 + i });
        }
        let m = t.metrics();
        assert_eq!(m.record_reads, 20);
        assert_eq!(m.container_touches, 20, "every read re-enters a container");
        assert!(m.backward_steps > 0, "alternating access moves backwards");
    }

    #[test]
    fn touches_are_counted_per_pass_not_globally() {
        let mut t = Trace::new(geom());
        // The same container read in two different passes is two touches:
        // nothing is assumed to survive between passes.
        for pass in 0..2 {
            t.push(Op::PassStart { pass });
            t.push(Op::ReadRecord { ordinal: 0 });
        }
        assert_eq!(t.metrics().container_touches, 2);
        assert_eq!(t.metrics().passes, 2);
    }
}
