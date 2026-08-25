// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! I/O schedulers — the queue between the issuer and the device.
//!
//! There are two reordering opportunities in a storage path and they are
//! easy to confuse. The device reorders the commands it has already
//! accepted, which is native command queueing and lives in
//! [`super::hw::ServicePolicy`]. This module is the other one: the queue
//! *before* the device, where an operating system can hold requests back
//! and hand them over in an order the device would not have chosen for
//! itself.
//!
//! That second queue only does anything when the issuer wants more
//! outstanding than the device will accept. Below that point requests
//! pass straight through and the scheduler is a formality — which is
//! itself worth knowing, because it means a workload at `iodepth=10`
//! against a 32-slot device is measuring the device's reordering, not
//! the kernel's.

use super::Request;
use std::collections::VecDeque;

/// A queue of accepted-but-not-yet-dispatched requests.
pub trait Scheduler {
    fn name(&self) -> &'static str;
    fn push(&mut self, req: Request);
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Take the request this scheduler would hand over next.
    fn pop_first(&mut self) -> Option<Request> {
        self.pop_first_where(&|_| true)
    }

    /// The same, restricted to requests the device can actually accept
    /// right now — a die that is mid-program will not take another
    /// command, and the queue has to skip past it rather than stall.
    fn pop_first_where(&mut self, allowed: &dyn Fn(&Request) -> bool) -> Option<Request>;

    /// Take whichever queued request minimises `cost`. Used when the
    /// *device* is choosing, so the scheduler is only holding the pool.
    fn pop_best(&mut self, cost: &dyn Fn(&Request) -> f64) -> Option<Request>;

    /// The same, but considering only the `window` oldest queued
    /// requests — firmware that commits some way ahead rather than
    /// re-evaluating the whole queue on every completion.
    fn pop_best_within(
        &mut self,
        window: usize,
        cost: &dyn Fn(&Request) -> f64,
    ) -> Option<Request> {
        self.pop_best_within_where(window, cost, &|_| true)
    }

    /// Take the oldest request that has waited beyond `expiry`, if any.
    ///
    /// Reordering without a bound starves whatever it keeps passing over.
    /// Real firmware will not do that indefinitely — a command that has
    /// waited long enough goes next whatever it costs — and without this
    /// an aggressive reorder policy drives a competing random reader to
    /// zero, which is not what the mixed-workload measurements show.
    fn pop_oldest_beyond(
        &mut self,
        now: f64,
        expiry: f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request>;

    /// The same, restricted to requests the device can accept right now.
    fn pop_best_within_where(
        &mut self,
        window: usize,
        cost: &dyn Fn(&Request) -> f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request>;
}

/// No reordering: hand requests over exactly as they arrived.
///
/// This is the right default for flash, and the right model for what fio
/// was measuring, since a modern kernel uses `none` for NVMe.
#[derive(Default)]
pub struct Noop {
    queue: VecDeque<Request>,
}

impl Scheduler for Noop {
    fn name(&self) -> &'static str {
        "noop"
    }

    fn push(&mut self, req: Request) {
        self.queue.push_back(req);
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn pop_first_where(&mut self, allowed: &dyn Fn(&Request) -> bool) -> Option<Request> {
        let idx = self.queue.iter().position(allowed)?;
        self.queue.remove(idx)
    }

    fn pop_oldest_beyond(
        &mut self,
        now: f64,
        expiry: f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        let idx = self
            .queue
            .iter()
            .enumerate()
            .filter(|(_, r)| allowed(r) && now - r.submitted_at >= expiry)
            .min_by(|(_, a), (_, b)| a.submitted_at.partial_cmp(&b.submitted_at).unwrap())
            .map(|(i, _)| i)?;
        self.queue.remove(idx)
    }

    fn pop_best(&mut self, cost: &dyn Fn(&Request) -> f64) -> Option<Request> {
        let best = self
            .queue
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| cost(a).partial_cmp(&cost(b)).unwrap())
            .map(|(i, _)| i)?;
        self.queue.remove(best)
    }

    fn pop_best_within_where(
        &mut self,
        window: usize,
        cost: &dyn Fn(&Request) -> f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        if self.queue.is_empty() {
            return None;
        }
        let limit = window.clamp(1, self.queue.len());
        let best = self
            .queue
            .iter()
            .take(limit)
            .enumerate()
            .filter(|(_, r)| allowed(r))
            .min_by(|(_, a), (_, b)| cost(a).partial_cmp(&cost(b)).unwrap())
            .map(|(i, _)| i)?;
        self.queue.remove(best)
    }
}

/// Sweep in one direction through the address space, serving requests as
/// the sweep passes them, then jump back and sweep again.
///
/// One-directional sweeping rather than back-and-forth is deliberate: it
/// gives every region the same wait, where a bidirectional elevator
/// favours the middle. The cost is one long seek per sweep.
#[derive(Default)]
pub struct Elevator {
    queue: Vec<Request>,
    position: u64,
}

impl Scheduler for Elevator {
    fn name(&self) -> &'static str {
        "elevator"
    }

    fn push(&mut self, req: Request) {
        self.queue.push(req);
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn pop_first_where(&mut self, allowed: &dyn Fn(&Request) -> bool) -> Option<Request> {
        if self.queue.is_empty() {
            return None;
        }
        // The next request at or after the sweep position, or the lowest
        // one if the sweep has run off the end.
        let ahead = self
            .queue
            .iter()
            .enumerate()
            .filter(|(_, r)| r.offset >= self.position && allowed(r))
            .min_by_key(|(_, r)| r.offset)
            .map(|(i, _)| i);
        let idx = match ahead {
            Some(i) => i,
            None => {
                self.position = 0;
                self.queue
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| allowed(r))
                    .min_by_key(|(_, r)| r.offset)
                    .map(|(i, _)| i)?
            }
        };
        let req = self.queue.swap_remove(idx);
        self.position = req.offset + req.len;
        Some(req)
    }

    fn pop_best(&mut self, cost: &dyn Fn(&Request) -> f64) -> Option<Request> {
        let best = self
            .queue
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| cost(a).partial_cmp(&cost(b)).unwrap())
            .map(|(i, _)| i)?;
        let req = self.queue.swap_remove(best);
        self.position = req.offset + req.len;
        Some(req)
    }

    fn pop_oldest_beyond(
        &mut self,
        now: f64,
        expiry: f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        let idx = self
            .queue
            .iter()
            .enumerate()
            .filter(|(_, r)| allowed(r) && now - r.submitted_at >= expiry)
            .min_by(|(_, a), (_, b)| a.submitted_at.partial_cmp(&b.submitted_at).unwrap())
            .map(|(i, _)| i)?;
        let req = self.queue.swap_remove(idx);
        self.position = req.offset + req.len;
        Some(req)
    }

    fn pop_best_within_where(
        &mut self,
        window: usize,
        cost: &dyn Fn(&Request) -> f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        let req = pop_best_within_slice(&mut self.queue, window, cost, allowed)?;
        self.position = req.offset + req.len;
        Some(req)
    }
}

/// An elevator that will not let a request wait past a deadline.
///
/// Pure sweeping starves anything behind the sweep while new work keeps
/// arriving in front of it. The deadline bounds that: once a request has
/// waited long enough it goes next regardless of where the head is. The
/// throughput cost of the occasional out-of-order seek is the price of
/// a bounded tail.
pub struct Deadline {
    inner: Elevator,
    expiry_s: f64,
    now: f64,
}

impl Deadline {
    pub fn new(expiry_s: f64) -> Self {
        Deadline {
            inner: Elevator::default(),
            expiry_s,
            now: 0.0,
        }
    }

    /// The engine advances this so the scheduler can tell what has aged.
    pub fn set_clock(&mut self, now: f64) {
        self.now = now;
    }

    fn expired(&self) -> Option<usize> {
        self.inner
            .queue
            .iter()
            .enumerate()
            .filter(|(_, r)| self.now - r.submitted_at >= self.expiry_s)
            .min_by(|(_, a), (_, b)| a.submitted_at.partial_cmp(&b.submitted_at).unwrap())
            .map(|(i, _)| i)
    }
}

impl Scheduler for Deadline {
    fn name(&self) -> &'static str {
        "deadline"
    }

    fn push(&mut self, req: Request) {
        self.inner.push(req);
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn pop_oldest_beyond(
        &mut self,
        now: f64,
        expiry: f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        self.inner.pop_oldest_beyond(now, expiry, allowed)
    }

    fn pop_first_where(&mut self, allowed: &dyn Fn(&Request) -> bool) -> Option<Request> {
        if let Some(i) = self.expired()
            && allowed(&self.inner.queue[i])
        {
            let req = self.inner.queue.swap_remove(i);
            self.inner.position = req.offset + req.len;
            return Some(req);
        }
        self.inner.pop_first_where(allowed)
    }

    fn pop_best(&mut self, cost: &dyn Fn(&Request) -> f64) -> Option<Request> {
        if let Some(i) = self.expired() {
            let req = self.inner.queue.swap_remove(i);
            self.inner.position = req.offset + req.len;
            return Some(req);
        }
        self.inner.pop_best(cost)
    }

    fn pop_best_within_where(
        &mut self,
        window: usize,
        cost: &dyn Fn(&Request) -> f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        if let Some(i) = self.expired()
            && allowed(&self.inner.queue[i])
        {
            let req = self.inner.queue.swap_remove(i);
            self.inner.position = req.offset + req.len;
            return Some(req);
        }
        self.inner.pop_best_within_where(window, cost, allowed)
    }
}

/// Choose among only the first `window` queued requests.
///
/// Default implementation for schedulers that keep insertion order.
pub fn pop_best_within_slice(
    queue: &mut Vec<Request>,
    window: usize,
    cost: &dyn Fn(&Request) -> f64,
    allowed: &dyn Fn(&Request) -> bool,
) -> Option<Request> {
    if queue.is_empty() {
        return None;
    }
    let limit = window.clamp(1, queue.len());
    let best = queue[..limit]
        .iter()
        .enumerate()
        .filter(|(_, r)| allowed(r))
        .min_by(|(_, a), (_, b)| cost(a).partial_cmp(&cost(b)).unwrap())
        .map(|(i, _)| i)?;
    Some(queue.remove(best))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: u64, offset: u64, submitted_at: f64) -> Request {
        Request {
            id,
            stream: 0,
            offset,
            len: 4_096,
            write: false,
            submitted_at,
        }
    }

    #[test]
    fn noop_preserves_arrival_order() {
        let mut s = Noop::default();
        for i in 0..4 {
            s.push(req(i, (4 - i) * 1_000_000, 0.0));
        }
        let order: Vec<u64> = (0..4).map(|_| s.pop_first().unwrap().id).collect();
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn the_elevator_sweeps_forward() {
        let mut s = Elevator::default();
        for (i, off) in [900u64, 100, 500, 300].iter().enumerate() {
            s.push(req(i as u64, off * 1_000_000, 0.0));
        }
        let offsets: Vec<u64> = (0..4).map(|_| s.pop_first().unwrap().offset).collect();
        let mut sorted = offsets.clone();
        sorted.sort();
        assert_eq!(offsets, sorted, "a sweep must visit offsets in order");
    }

    /// Sweeping is what makes the total distance travelled small; that is
    /// the entire reason to reorder on positional media.
    #[test]
    fn sweeping_travels_less_than_arrival_order() {
        let offsets: Vec<u64> = (0..32)
            .map(|i| ((i * 2_654_435_761u64) % 1_000) * 1_000_000)
            .collect();

        let travel = |order: Vec<u64>| -> u64 {
            let mut head = 0u64;
            let mut total = 0u64;
            for o in order {
                total += head.abs_diff(o);
                head = o;
            }
            total
        };

        let mut fifo = Noop::default();
        let mut lift = Elevator::default();
        for (i, &o) in offsets.iter().enumerate() {
            fifo.push(req(i as u64, o, 0.0));
            lift.push(req(i as u64, o, 0.0));
        }
        let fifo_order: Vec<u64> = (0..32).map(|_| fifo.pop_first().unwrap().offset).collect();
        let lift_order: Vec<u64> = (0..32).map(|_| lift.pop_first().unwrap().offset).collect();

        assert!(
            travel(lift_order) < travel(fifo_order) / 5,
            "sweeping should cut head travel dramatically"
        );
    }

    /// Without a deadline a sweep can leave a request behind indefinitely.
    /// With one, it cannot.
    #[test]
    fn the_deadline_rescues_a_request_the_sweep_passed() {
        let mut s = Deadline::new(0.1);
        // Submitted early, sitting behind where the sweep now is.
        s.push(req(99, 10_000_000, 0.0));
        s.pop_first();
        s.push(req(99, 10_000_000, 0.0));
        // Newer work, all ahead of it.
        for i in 0..8 {
            s.push(req(i, 500_000_000 + i * 1_000_000, 0.5));
        }

        s.set_clock(0.05);
        assert_ne!(s.pop_first().unwrap().id, 99, "not yet expired");

        s.set_clock(0.6);
        assert_eq!(s.pop_first().unwrap().id, 99, "expired, so it goes next");
    }
}
