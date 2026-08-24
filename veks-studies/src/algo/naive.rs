// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The two baselines gsplat is measured against.
//!
//! Both apply the map directly, with no segment buffer. They differ in
//! which side absorbs the scatter: gather walks the output in order and
//! reads at random; scatter walks the source in order and writes at
//! random.

use super::Rewrite;
use crate::model::{Geometry, Map, Op, Sink, Trace};

/// Walk the output in order, reading each record where the map points.
///
/// Reads land wherever the permutation says, so the read sequence is as
/// scattered as the map is. Writes are sequential and free.
pub struct NaiveGather;

impl Rewrite for NaiveGather {
    fn name(&self) -> &'static str {
        "naive-gather"
    }

    fn run(&self, geometry: Geometry, map: &Map, _budget_bytes: u64) -> (Sink, Trace) {
        let mut sink = Sink::new(map.len());
        let mut trace = Trace::new(geometry);

        // One notional pass: the output is walked once, start to end.
        trace.push(Op::PassStart { pass: 0 });
        // Resident state is a single record in flight.
        trace.claim_resident(geometry.record_bytes);

        for (slot, &source) in map.0.iter().enumerate() {
            trace.push(Op::ReadMap { from: slot as u64, count: 1 });
            trace.push(Op::ReadRecord { ordinal: source });
            sink.slots[slot] = Some(source);
            trace.push(Op::WriteRange { first_slot: slot as u64, records: 1 });
        }
        trace.push(Op::Barrier);

        (sink, trace)
    }
}

/// Walk the source in order, writing each record where its inverse says.
///
/// Reads are sequential; writes are scattered. Modelled for completeness
/// and to make the asymmetry visible: it needs the inverse map, which is
/// resident state proportional to the store.
pub struct NaiveScatter;

impl Rewrite for NaiveScatter {
    fn name(&self) -> &'static str {
        "naive-scatter"
    }

    fn run(&self, geometry: Geometry, map: &Map, _budget_bytes: u64) -> (Sink, Trace) {
        let mut sink = Sink::new(map.len());
        let mut trace = Trace::new(geometry);

        // Inverting the map is the price of entry: one slot per record.
        let mut inverse = vec![u64::MAX; geometry.records as usize];
        for (slot, &source) in map.0.iter().enumerate() {
            inverse[source as usize] = slot as u64;
        }
        trace.push(Op::PassStart { pass: 0 });
        trace.push(Op::ReadMap { from: 0, count: map.len() });
        trace.claim_resident(geometry.record_bytes + geometry.records * 8);

        for source in 0..geometry.records {
            let slot = inverse[source as usize];
            if slot == u64::MAX {
                continue;
            }
            trace.push(Op::ReadRecord { ordinal: source });
            sink.slots[slot as usize] = Some(source);
            // Each write lands wherever the inverse points — one range
            // of one record, at an unrelated position.
            trace.push(Op::WriteRange { first_slot: slot, records: 1 });
        }
        trace.push(Op::Barrier);

        (sink, trace)
    }
}
