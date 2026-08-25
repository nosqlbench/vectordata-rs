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
//! The schedulers here are simplified forms of the ones Linux ships —
//! `none`, an elevator, and a deadline-bounded elevator answering to
//! `mq-deadline`. [Ren et al. (ICPE '24)](https://dl.acm.org/doi/10.1145/3629526.3645053)
//! characterise the real ones on modern NVMe and find they can cost up
//! to 63.4% of throughput while cutting P99 latency by 99.3% under
//! interference. **That cost is not modelled here**: these schedulers are
//! free, so the model understates what running one actually takes.
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

    /// Serialized CPU time one dispatch costs.
    ///
    /// Zero for a scheduler that is only a queue. Non-zero for one whose
    /// bookkeeping runs under a contended lock, which is where the
    /// measured throughput ceilings of `mq-deadline` and `bfq` come
    /// from — see [`LinuxScheduler::dispatch_cost_s`].
    fn dispatch_cost_s(&self) -> f64 {
        0.0
    }

    /// Tell the scheduler a request it dispatched has completed, and how
    /// long it took.
    ///
    /// Only a scheduler with a feedback loop cares. [`Kyber`] uses this
    /// to size its token pools against the latency targets; everything
    /// else ignores it.
    fn release(&mut self, write: bool, latency_s: f64) {
        let _ = (write, latency_s);
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
            kernel: false,
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

// ---- The Linux multi-queue schedulers ---------------------------------

/// Which of Linux's block schedulers is in force.
///
/// The choice is not a detail. [Ren, Doekemeijer, Tehrany & Trivedi
/// (ICPE '24)](https://dl.acm.org/doi/10.1145/3629526.3645053) measured
/// all four on Samsung NVMe and found the spread enormous: against a
/// device ceiling of **785.7 KIOPS**, `none` and `kyber` both reach it,
/// `mq-deadline` peaks at **569.2 KIOPS** (0.72×) and `bfq` at
/// **315.3 KIOPS** (0.40×) — and the cause is not the policy but the
/// **lock contention inside it**, which consumed up to 78.0% of CPU
/// cycles for `bfq` and `mq-deadline` on a single SSD. In exchange,
/// `kyber` and `bfq` deliver up to **99.3% lower P99** than `none` or
/// `mq-deadline` when interfering workloads are present.
///
/// So the schedulers are not ranked; they trade. What each one is doing
/// to *this* rewrite is worth reporting rather than assuming, which is
/// why they are all here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinuxScheduler {
    /// `none` — no reordering, no queue, no lock. The default for NVMe.
    None,
    /// `mq-deadline` — sector-sorted with per-direction expiry.
    MqDeadline,
    /// `kyber` — latency-targeted token limits, reads favoured.
    Kyber,
    /// `bfq` — proportional-share budget fair queueing.
    Bfq,
}

impl LinuxScheduler {
    pub fn name(self) -> &'static str {
        match self {
            LinuxScheduler::None => "none",
            LinuxScheduler::MqDeadline => "mq-deadline",
            LinuxScheduler::Kyber => "kyber",
            LinuxScheduler::Bfq => "bfq",
        }
    }

    /// Serialized CPU time each dispatch costs, in seconds.
    ///
    /// This is the term the module used to leave out, and leaving it out
    /// is what made every scheduler look free. It is **serialized**
    /// rather than per-core because the measured ceiling does not move
    /// when cores are added: the paper attributes the shortfall to lock
    /// contention (`native_queued_spin_lock_slowpath` and friends), and
    /// a contended lock is a resource of which there is exactly one.
    ///
    /// The values are the reciprocals of the measured peaks, so a run
    /// against a fast enough device reproduces them by construction:
    /// `mq-deadline` 1/569.2 KIOPS = 1.757 µs, `bfq` 1/315.3 KIOPS =
    /// 3.172 µs. `none` and `kyber` reached the device ceiling, so
    /// neither imposes one of its own.
    pub fn dispatch_cost_s(self) -> f64 {
        match self {
            LinuxScheduler::None => 0.0,
            LinuxScheduler::Kyber => 0.0,
            LinuxScheduler::MqDeadline => 1.0 / 569_200.0,
            LinuxScheduler::Bfq => 1.0 / 315_300.0,
        }
    }

    /// The measured single-SSD ceiling this scheduler imposes, in IOPS,
    /// or infinity where the device was the limit.
    pub fn measured_ceiling_iops(self) -> f64 {
        match self {
            LinuxScheduler::None | LinuxScheduler::Kyber => 785_700.0,
            LinuxScheduler::MqDeadline => 569_200.0,
            LinuxScheduler::Bfq => 315_300.0,
        }
    }

    /// Build the scheduler.
    pub fn build(self) -> Box<dyn Scheduler> {
        match self {
            LinuxScheduler::None => Box::new(Noop::default()),
            LinuxScheduler::MqDeadline => Box::new(MqDeadline::default()),
            LinuxScheduler::Kyber => Box::new(Kyber::default()),
            LinuxScheduler::Bfq => Box::new(Bfq::default()),
        }
    }

    pub const ALL: [LinuxScheduler; 4] = [
        LinuxScheduler::None,
        LinuxScheduler::MqDeadline,
        LinuxScheduler::Kyber,
        LinuxScheduler::Bfq,
    ];
}

/// `mq-deadline`: sector order, bounded by per-direction expiry.
///
/// Two sorted queues and two FIFO queues, one pair per direction. The
/// dispatcher serves up to `fifo_batch` requests in increasing sector
/// order, and breaks off to serve the FIFO head whenever a request has
/// waited past its expiry. Reads expire ten times sooner than writes
/// (`read_expire` 500 ms against `write_expire` 5 s) because a reader is
/// usually waiting and a writer usually is not, and writes get a turn
/// after `writes_starved` consecutive read batches so the preference
/// does not become starvation.
///
/// Defaults are the kernel's: `block/mq-deadline.c`.
pub struct MqDeadline {
    queue: Vec<Request>,
    position: u64,
    now: f64,
    /// `read_expire`, seconds.
    pub read_expire_s: f64,
    /// `write_expire`, seconds.
    pub write_expire_s: f64,
    /// `fifo_batch` — requests served in sector order before the
    /// dispatcher reconsiders direction.
    pub fifo_batch: usize,
    /// `writes_starved` — read batches allowed before writes get one.
    pub writes_starved: usize,
    batch_remaining: usize,
    reads_in_a_row: usize,
    serving_writes: bool,
}

impl Default for MqDeadline {
    fn default() -> Self {
        MqDeadline {
            queue: Vec::new(),
            position: 0,
            now: 0.0,
            read_expire_s: 0.5,
            write_expire_s: 5.0,
            fifo_batch: 16,
            writes_starved: 2,
            batch_remaining: 0,
            reads_in_a_row: 0,
            serving_writes: false,
        }
    }
}

impl MqDeadline {
    pub fn set_clock(&mut self, now: f64) {
        self.now = now;
    }

    fn expiry_of(&self, r: &Request) -> f64 {
        if r.write {
            self.write_expire_s
        } else {
            self.read_expire_s
        }
    }

    /// The oldest request past its own direction's deadline.
    fn expired(&self, allowed: &dyn Fn(&Request) -> bool) -> Option<usize> {
        self.queue
            .iter()
            .enumerate()
            .filter(|(_, r)| allowed(r) && self.now - r.submitted_at >= self.expiry_of(r))
            .min_by(|(_, a), (_, b)| a.submitted_at.partial_cmp(&b.submitted_at).unwrap())
            .map(|(i, _)| i)
    }

    /// Decide which direction this batch serves, honouring the
    /// starvation bound.
    fn choose_direction(&mut self, allowed: &dyn Fn(&Request) -> bool) {
        if self.batch_remaining > 0 {
            return;
        }
        let has_read = self.queue.iter().any(|r| !r.write && allowed(r));
        let has_write = self.queue.iter().any(|r| r.write && allowed(r));
        self.serving_writes =
            if has_write && (!has_read || self.reads_in_a_row >= self.writes_starved) {
                self.reads_in_a_row = 0;
                true
            } else {
                if has_read {
                    self.reads_in_a_row += 1;
                }
                false
            };
        self.batch_remaining = self.fifo_batch;
    }

    fn next_in_sector_order(&self, allowed: &dyn Fn(&Request) -> bool) -> Option<usize> {
        let want_write = self.serving_writes;
        let matches = |r: &Request| allowed(r) && r.write == want_write;
        let ahead = self
            .queue
            .iter()
            .enumerate()
            .filter(|(_, r)| r.offset >= self.position && matches(r))
            .min_by_key(|(_, r)| r.offset)
            .map(|(i, _)| i);
        ahead.or_else(|| {
            self.queue
                .iter()
                .enumerate()
                .filter(|(_, r)| matches(r))
                .min_by_key(|(_, r)| r.offset)
                .map(|(i, _)| i)
        })
    }

    fn take(&mut self, index: usize) -> Request {
        let req = self.queue.remove(index);
        self.position = req.offset + req.len;
        self.batch_remaining = self.batch_remaining.saturating_sub(1);
        req
    }
}

impl Scheduler for MqDeadline {
    fn name(&self) -> &'static str {
        "mq-deadline"
    }

    fn push(&mut self, req: Request) {
        self.now = self.now.max(req.submitted_at);
        self.queue.push(req);
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn dispatch_cost_s(&self) -> f64 {
        LinuxScheduler::MqDeadline.dispatch_cost_s()
    }

    fn pop_first_where(&mut self, allowed: &dyn Fn(&Request) -> bool) -> Option<Request> {
        if let Some(i) = self.expired(allowed) {
            // Breaking the batch is the point of the deadline.
            self.batch_remaining = 0;
            return Some(self.take(i));
        }
        self.choose_direction(allowed);
        let i = self
            .next_in_sector_order(allowed)
            .or_else(|| self.queue.iter().position(allowed))?;
        Some(self.take(i))
    }

    fn pop_oldest_beyond(
        &mut self,
        now: f64,
        expiry: f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        self.now = self.now.max(now);
        let i = self
            .queue
            .iter()
            .enumerate()
            .filter(|(_, r)| allowed(r) && now - r.submitted_at >= expiry)
            .min_by(|(_, a), (_, b)| a.submitted_at.partial_cmp(&b.submitted_at).unwrap())
            .map(|(i, _)| i)?;
        Some(self.take(i))
    }

    fn pop_best(&mut self, cost: &dyn Fn(&Request) -> f64) -> Option<Request> {
        let i = self
            .queue
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| cost(a).partial_cmp(&cost(b)).unwrap())
            .map(|(i, _)| i)?;
        Some(self.take(i))
    }

    fn pop_best_within_where(
        &mut self,
        window: usize,
        cost: &dyn Fn(&Request) -> f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        // The device's own reordering only ever sees what the scheduler
        // has handed over, so a sorted scheduler and a reordering device
        // compose rather than compete: mq-deadline picks the direction
        // and the batch, and the device picks within it.
        if let Some(i) = self.expired(allowed) {
            self.batch_remaining = 0;
            return Some(self.take(i));
        }
        self.choose_direction(allowed);
        let want_write = self.serving_writes;
        let limit = window.clamp(1, self.queue.len().max(1));
        let i = self
            .queue
            .iter()
            .enumerate()
            .take(limit)
            .filter(|(_, r)| allowed(r) && r.write == want_write)
            .min_by(|(_, a), (_, b)| cost(a).partial_cmp(&cost(b)).unwrap())
            .map(|(i, _)| i)
            .or_else(|| self.next_in_sector_order(allowed))
            .or_else(|| self.queue.iter().position(allowed))?;
        Some(self.take(i))
    }
}

/// `kyber`: hold the device's queue short enough to hit a latency target.
///
/// Kyber does not reorder. It **limits how much is in flight**, per
/// direction, with a token pool sized by a feedback loop: if completions
/// come back faster than the target the pool can grow, and if the other
/// direction is being served badly the pool shrinks. Reads get a 2 ms
/// target and writes 10 ms, which is how reads end up favoured without
/// writes ever being starved — a reader is usually blocked on its
/// result and a writer usually is not.
///
/// Token bounds are the kernel's: at most 256 read and 128 write, at
/// least 1 of each (`block/kyber-iosched.c`). The ICPE '24 measurements
/// show why this is nearly free — Kyber reaches the same 785.7 KIOPS
/// peak as `none` while spending only 14.7% of cycles on locks against
/// `bfq`'s 78.0% — and Table 2 of that paper shows the target latencies
/// working exactly as described: dropping `read_lat_nsec` to zero moved
/// a concurrent pair from 156.7/103.4 KIOPS to 189.4/79.4.
pub struct Kyber {
    queue: VecDeque<Request>,
    /// `read_lat_nsec`, as seconds.
    pub read_target_s: f64,
    /// `write_lat_nsec`, as seconds.
    pub write_target_s: f64,
    read_tokens: usize,
    write_tokens: usize,
    read_in_flight: usize,
    write_in_flight: usize,
    /// Recent completion latencies per direction, for the percentiles the
    /// token rule is stated in terms of.
    recent: [VecDeque<f64>; 2],
    /// Cached percentiles, refreshed every `REEVALUATE` completions.
    /// Kyber itself re-evaluates on a timer rather than per request, so
    /// sampling here is faithful as well as cheap.
    cached: [(f64, f64); 2],
    since_refresh: [usize; 2],
}

impl Default for Kyber {
    fn default() -> Self {
        Kyber {
            queue: VecDeque::new(),
            read_target_s: 2e-3,
            write_target_s: 10e-3,
            read_tokens: Kyber::MAX_READ_TOKENS,
            write_tokens: Kyber::MAX_WRITE_TOKENS,
            read_in_flight: 0,
            write_in_flight: 0,
            recent: [VecDeque::new(), VecDeque::new()],
            cached: [(f64::NAN, f64::NAN); 2],
            since_refresh: [0; 2],
        }
    }
}

impl Kyber {
    pub const MAX_READ_TOKENS: usize = 256;
    pub const MAX_WRITE_TOKENS: usize = 128;
    pub const MIN_TOKENS: usize = 1;

    /// Tokens currently available for a direction.
    pub fn tokens(&self, write: bool) -> usize {
        if write {
            self.write_tokens
        } else {
            self.read_tokens
        }
    }

    /// Window of completions the percentiles are taken over.
    const WINDOW: usize = 64;
    /// Completions between percentile refreshes. Kyber re-evaluates on a
    /// timer, not on every request.
    const REEVALUATE: usize = 16;

    /// Feed an observed completion back into the token loop.
    ///
    /// The rule is a **re-prioritization**, not a latency clamp, and the
    /// difference matters. A direction's tokens are cut only when that
    /// direction is being *well* served — its P90 is inside its target —
    /// **and the other direction is being badly served**, its P99 outside
    /// its. Cutting the well-served side's depth is how the queue makes
    /// room for the starved one.
    ///
    /// Modelling it as "shrink whenever latency exceeds the target" would
    /// be a different and much harsher algorithm: a workload with only
    /// one direction in flight would strangle itself down to a single
    /// outstanding request, which is not what Kyber does and not what it
    /// measures. With no other direction to protect, tokens stay at the
    /// cap.
    ///
    /// Rule as stated in Ren et al. (ICPE '24) §6, describing
    /// `block/kyber-iosched.c`.
    pub fn observe(&mut self, write: bool, latency_s: f64) {
        let slot = usize::from(write);
        let window = &mut self.recent[slot];
        window.push_back(latency_s);
        while window.len() > Kyber::WINDOW {
            window.pop_front();
        }

        self.since_refresh[slot] += 1;
        if self.since_refresh[slot] >= Kyber::REEVALUATE {
            self.since_refresh[slot] = 0;
            self.cached[slot] = self.percentiles(slot);
        }

        let this_p90 = self.cached[slot].0;
        let other = 1 - slot;
        let other_p99 = self.cached[other].1;
        let this_target = if write {
            self.write_target_s
        } else {
            self.read_target_s
        };
        let other_target = if write {
            self.read_target_s
        } else {
            self.write_target_s
        };

        let well_served = this_p90.is_finite() && this_p90 <= this_target;
        let other_starved = other_p99.is_finite() && other_p99 > other_target;

        let (tokens, cap) = if write {
            (&mut self.write_tokens, Kyber::MAX_WRITE_TOKENS)
        } else {
            (&mut self.read_tokens, Kyber::MAX_READ_TOKENS)
        };
        if well_served && other_starved {
            *tokens = tokens.saturating_sub(1).max(Kyber::MIN_TOKENS);
        } else if *tokens < cap {
            *tokens += 1;
        }
    }

    /// The P90 and P99 of the recent completions for one direction, or
    /// `NaN` until there is enough of a window to mean anything.
    fn percentiles(&self, slot: usize) -> (f64, f64) {
        let window = &self.recent[slot];
        if window.len() < 8 {
            return (f64::NAN, f64::NAN);
        }
        let mut sorted: Vec<f64> = window.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
        (at(0.90), at(0.99))
    }

    fn admits(&self, r: &Request) -> bool {
        if r.write {
            self.write_in_flight < self.write_tokens
        } else {
            self.read_in_flight < self.read_tokens
        }
    }

    fn charge(&mut self, r: &Request) {
        if r.write {
            self.write_in_flight += 1;
        } else {
            self.read_in_flight += 1;
        }
    }
}

impl Scheduler for Kyber {
    fn name(&self) -> &'static str {
        "kyber"
    }

    fn push(&mut self, req: Request) {
        self.queue.push_back(req);
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn release(&mut self, write: bool, latency_s: f64) {
        if write {
            self.write_in_flight = self.write_in_flight.saturating_sub(1);
        } else {
            self.read_in_flight = self.read_in_flight.saturating_sub(1);
        }
        self.observe(write, latency_s);
    }

    fn pop_first_where(&mut self, allowed: &dyn Fn(&Request) -> bool) -> Option<Request> {
        let idx = self
            .queue
            .iter()
            .position(|r| allowed(r) && self.admits(r))?;
        let req = self.queue.remove(idx)?;
        self.charge(&req);
        Some(req)
    }

    fn pop_oldest_beyond(
        &mut self,
        now: f64,
        expiry: f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        // Expiry outranks the token limit: a request that has waited
        // this long is exactly the one the tokens were protecting.
        let idx = self
            .queue
            .iter()
            .enumerate()
            .filter(|(_, r)| allowed(r) && now - r.submitted_at >= expiry)
            .min_by(|(_, a), (_, b)| a.submitted_at.partial_cmp(&b.submitted_at).unwrap())
            .map(|(i, _)| i)?;
        let req = self.queue.remove(idx)?;
        self.charge(&req);
        Some(req)
    }

    fn pop_best(&mut self, cost: &dyn Fn(&Request) -> f64) -> Option<Request> {
        let idx = self
            .queue
            .iter()
            .enumerate()
            .filter(|(_, r)| self.admits(r))
            .min_by(|(_, a), (_, b)| cost(a).partial_cmp(&cost(b)).unwrap())
            .map(|(i, _)| i)?;
        let req = self.queue.remove(idx)?;
        self.charge(&req);
        Some(req)
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
        let idx = self
            .queue
            .iter()
            .take(limit)
            .enumerate()
            .filter(|(_, r)| allowed(r) && self.admits(r))
            .min_by(|(_, a), (_, b)| cost(a).partial_cmp(&cost(b)).unwrap())
            .map(|(i, _)| i)?;
        let req = self.queue.remove(idx)?;
        self.charge(&req);
        Some(req)
    }
}

/// `bfq`: proportional-share budget fair queueing.
///
/// Each stream gets its own queue and a budget in bytes; the queue with
/// the earliest virtual finish time gets exclusive device access until
/// its budget is spent or its slice times out, then the budget is
/// refilled and the next queue goes. That exclusivity is what gives it
/// the tail-latency behaviour it is chosen for — a background scan
/// cannot flood the queue ahead of a latency-sensitive reader — and the
/// bookkeeping behind it is what costs the throughput: ~10,000 lines
/// against ~1,000 for `kyber`, and up to 78.0% of cycles under lock.
///
/// At 315.3 KIOPS against a 785.7 KIOPS device, the ICPE '24 authors
/// conclude it is unsuitable for modern NVMe. It stays here because the
/// same paper shows it delivering up to 99.3% lower P99 under
/// interference, and a rewrite sharing a volume with a latency-sensitive
/// service is exactly that case.
pub struct Bfq {
    queues: Vec<VecDeque<Request>>,
    /// Bytes a stream may transfer before the device moves on.
    pub budget_bytes: u64,
    /// Wall-clock slice a stream may hold the device for.
    pub slice_s: f64,
    active: usize,
    spent_bytes: u64,
    slice_started_at: f64,
    now: f64,
    len: usize,
}

impl Default for Bfq {
    fn default() -> Self {
        Bfq {
            queues: Vec::new(),
            // The kernel's default max budget is expressed in sectors and
            // scaled from measured peak rate; 4 MiB is what that comes to
            // on a drive in this class.
            budget_bytes: 4 << 20,
            slice_s: 0.100,
            active: 0,
            spent_bytes: 0,
            slice_started_at: 0.0,
            now: 0.0,
            len: 0,
        }
    }
}

impl Bfq {
    fn slot(&mut self, stream: usize) -> &mut VecDeque<Request> {
        if self.queues.len() <= stream {
            self.queues.resize_with(stream + 1, VecDeque::new);
        }
        &mut self.queues[stream]
    }

    /// Whether the active queue's turn is over.
    fn slice_expired(&self) -> bool {
        self.spent_bytes >= self.budget_bytes
            || self.now - self.slice_started_at >= self.slice_s
            || self.queues.get(self.active).is_none_or(|q| q.is_empty())
    }

    /// Move to the next non-empty queue, round robin.
    fn rotate(&mut self) {
        if self.queues.is_empty() {
            return;
        }
        for step in 1..=self.queues.len() {
            let candidate = (self.active + step) % self.queues.len();
            if !self.queues[candidate].is_empty() {
                self.active = candidate;
                break;
            }
        }
        self.spent_bytes = 0;
        self.slice_started_at = self.now;
    }

    fn take_from_active(&mut self, allowed: &dyn Fn(&Request) -> bool) -> Option<Request> {
        let q = self.queues.get_mut(self.active)?;
        let idx = q.iter().position(allowed)?;
        let req = q.remove(idx)?;
        self.spent_bytes += req.len;
        self.len -= 1;
        Some(req)
    }
}

impl Scheduler for Bfq {
    fn name(&self) -> &'static str {
        "bfq"
    }

    fn push(&mut self, req: Request) {
        self.now = self.now.max(req.submitted_at);
        let stream = req.stream;
        self.slot(stream).push_back(req);
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn dispatch_cost_s(&self) -> f64 {
        LinuxScheduler::Bfq.dispatch_cost_s()
    }

    fn pop_first_where(&mut self, allowed: &dyn Fn(&Request) -> bool) -> Option<Request> {
        if self.len == 0 {
            return None;
        }
        if self.slice_expired() {
            self.rotate();
        }
        if let Some(req) = self.take_from_active(allowed) {
            return Some(req);
        }
        // The active queue has nothing servable; give the others a turn
        // rather than stalling the device.
        for step in 1..=self.queues.len() {
            self.active = (self.active + step) % self.queues.len();
            self.spent_bytes = 0;
            self.slice_started_at = self.now;
            if let Some(req) = self.take_from_active(allowed) {
                return Some(req);
            }
        }
        None
    }

    fn pop_oldest_beyond(
        &mut self,
        now: f64,
        expiry: f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        self.now = self.now.max(now);
        let mut best: Option<(usize, usize, f64)> = None;
        for (qi, q) in self.queues.iter().enumerate() {
            for (ri, r) in q.iter().enumerate() {
                if allowed(r)
                    && now - r.submitted_at >= expiry
                    && best.is_none_or(|(_, _, t)| r.submitted_at < t)
                {
                    best = Some((qi, ri, r.submitted_at));
                }
            }
        }
        let (qi, ri, _) = best?;
        let req = self.queues[qi].remove(ri)?;
        self.len -= 1;
        if qi == self.active {
            self.spent_bytes += req.len;
        }
        Some(req)
    }

    fn pop_best(&mut self, cost: &dyn Fn(&Request) -> f64) -> Option<Request> {
        self.pop_best_within_where(usize::MAX, cost, &|_| true)
    }

    fn pop_best_within_where(
        &mut self,
        window: usize,
        cost: &dyn Fn(&Request) -> f64,
        allowed: &dyn Fn(&Request) -> bool,
    ) -> Option<Request> {
        if self.len == 0 {
            return None;
        }
        if self.slice_expired() {
            self.rotate();
        }
        // Within the active queue's slice the device may still choose,
        // which is the same composition mq-deadline has: the scheduler
        // decides *whose* requests, the device decides which of them.
        let active = self.active;
        let idx = {
            let q = self.queues.get(active)?;
            let limit = window.clamp(1, q.len().max(1));
            q.iter()
                .take(limit)
                .enumerate()
                .filter(|(_, r)| allowed(r))
                .min_by(|(_, a), (_, b)| cost(a).partial_cmp(&cost(b)).unwrap())
                .map(|(i, _)| i)
        };
        match idx {
            Some(i) => {
                let req = self.queues[active].remove(i)?;
                self.spent_bytes += req.len;
                self.len -= 1;
                Some(req)
            }
            None => self.pop_first_where(allowed),
        }
    }
}
