// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Parallel work-pump: a fixed pool of scoped worker threads draining a
//! shared queue, with first-error abort and a calling-thread progress/
//! checkpoint tick.
//!
//! Both chunk-download paths — the merkle-verified
//! [`crate::cache::CachedChannel`] and the no-`.mref`
//! [`crate::chunked_http::ChunkStore`] — drive their fetches through the
//! exact same orchestration: reverse the work list into a LIFO
//! `Mutex<Vec<_>>` queue, spawn `download_concurrency()` workers that
//! pop-fetch-report, flip an `AtomicBool` on the first failure so no new
//! work starts, and collect completions on the main thread with a
//! `recv_timeout` tick that also drives progress and throttled
//! checkpoints. That skeleton lived in two copies; this module is the
//! single owner.
//!
//! Critically, the pump is **integrity-agnostic**: it knows nothing
//! about merkle verification, cache files, or sidecars. Every
//! path-specific step — `verify_chunk` before write, `mark_valid`, the
//! `.mrkl`/`.chunks` checkpoint cadence, in-flight cleanup for resume —
//! lives entirely in the caller's `work` and `on_event` closures, so the
//! merkle-verified download's guarantees are exactly as before.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Mutex};
use std::time::Duration;

/// Drive `items` through `concurrency` scoped OS-thread workers that pull
/// from a shared queue and run `work` on each item out of order.
///
/// - **Order**: items are processed in roughly the order given — workers
///   pop a reversed LIFO queue, so the lowest-indexed work goes out first
///   (near-sequential range requests, which S3 / CloudFront read-ahead
///   prefers). Completion is still out of order.
/// - **Abort**: the first `work` that returns `Err` flips an internal
///   abort flag; workers that haven't started their next item skip it
///   (so the completion counter still reaches `items.len()` and the loop
///   terminates) and the first error is the returned value. In-flight
///   items are allowed to finish. Any per-item cleanup an aborted item
///   needs (e.g. clearing an in-flight entry so blocked readers wake) is
///   the caller's responsibility *after* this returns — the pump only
///   owns the mechanics.
/// - **Ticks**: `on_event` runs on the **calling thread** after every
///   completion and on every `tick` timeout, so callers render progress
///   and run throttled checkpoints without a `Send` bound on that state.
///   The pump fires no final event — callers that want a terminal 100%
///   frame emit it themselves after this returns.
///
/// A `work` closure is shared across workers (`Fn + Sync`); it owns all
/// per-item I/O, verification, and progress accounting. `items.is_empty()`
/// is a no-op (`Ok(())`, no `on_event`).
pub(crate) fn drain_parallel<W, Wk, Ev>(
    mut items: Vec<W>,
    concurrency: usize,
    tick: Duration,
    work: Wk,
    mut on_event: Ev,
) -> io::Result<()>
where
    W: Send,
    Wk: Fn(&W) -> io::Result<()> + Sync,
    Ev: FnMut(),
{
    let total = items.len();
    if total == 0 {
        return Ok(());
    }
    // Pop from the back yields ascending order.
    items.reverse();
    let queue: Mutex<Vec<W>> = Mutex::new(items);
    let abort = AtomicBool::new(false);
    let (tx, rx) = mpsc::channel::<io::Result<()>>();

    std::thread::scope(|scope| {
        for _ in 0..concurrency.max(1) {
            let queue = &queue;
            let abort = &abort;
            let work = &work;
            let tx = tx.clone();
            scope.spawn(move || loop {
                let item = match queue.lock().unwrap().pop() {
                    Some(i) => i,
                    None => break,
                };
                // Once aborted, keep draining so the main loop's done-count
                // still reaches `total` and terminates — just skip the work.
                if abort.load(Ordering::Relaxed) {
                    let _ = tx.send(Ok(()));
                    continue;
                }
                let r = work(&item);
                if r.is_err() {
                    abort.store(true, Ordering::Relaxed);
                }
                let _ = tx.send(r);
            });
        }
        // Drop the spawner's handle so the channel closes once every worker
        // exits — guards against a hang if `done` were to undercount.
        drop(tx);

        let mut first_err: io::Result<()> = Ok(());
        let mut done = 0usize;
        while done < total {
            match rx.recv_timeout(tick) {
                Ok(r) => {
                    done += 1;
                    if r.is_err() && first_err.is_ok() {
                        first_err = r;
                    }
                    on_event();
                }
                Err(mpsc::RecvTimeoutError::Timeout) => on_event(),
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        first_err
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Every item runs exactly once and `on_event` fires once per
    /// completion (no final synthetic event).
    #[test]
    fn runs_every_item_once() {
        let ran = Arc::new(AtomicUsize::new(0));
        let events = std::cell::Cell::new(0usize);
        let r = ran.clone();
        let out = drain_parallel(
            (0..50).collect::<Vec<u32>>(),
            8,
            Duration::from_millis(50),
            move |_| {
                r.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
            || events.set(events.get() + 1),
        );
        assert!(out.is_ok());
        assert_eq!(ran.load(Ordering::Relaxed), 50);
    }

    /// The first error is returned, the abort flag stops new work from
    /// starting, and the completion loop still terminates (every item is
    /// accounted for, erroring or skipped).
    #[test]
    fn first_error_aborts_and_returns() {
        let ran = Arc::new(AtomicUsize::new(0));
        let r = ran.clone();
        let out = drain_parallel(
            (0..200).collect::<Vec<u32>>(),
            4,
            Duration::from_millis(10),
            move |&i| {
                r.fetch_add(1, Ordering::Relaxed);
                if i == 0 {
                    // Item 0 is dispatched first (ascending order), so the
                    // abort takes effect early.
                    Err(io::Error::new(io::ErrorKind::Other, "boom"))
                } else {
                    Ok(())
                }
            },
            || {},
        );
        assert!(out.is_err());
        // Abort spares most items: far fewer than all 200 should have run.
        assert!(ran.load(Ordering::Relaxed) < 200, "abort must stop new work");
    }

    /// An empty work list is a no-op: `Ok(())`, and `on_event` never fires.
    #[test]
    fn empty_is_noop() {
        let events = std::cell::Cell::new(0usize);
        let out = drain_parallel(
            Vec::<u32>::new(),
            4,
            Duration::from_millis(10),
            |_: &u32| Ok(()),
            || events.set(events.get() + 1),
        );
        assert!(out.is_ok());
        assert_eq!(events.get(), 0);
    }
}
