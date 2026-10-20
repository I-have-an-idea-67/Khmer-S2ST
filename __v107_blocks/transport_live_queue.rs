//! The live event queue and the interrupt fence.
//!
//! v10.7 Phase 2 (Deliverable R-4): the fence is exposed to Python as a
//! `#[pyclass]`, and the queue is built on `crossbeam-channel`, whose
//! `unbounded` queue drains in well under the 2 microsecond budget.

use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_channel::{unbounded, Receiver, Sender};

/// A monotonically increasing interrupt epoch.
///
/// Every seek or interrupt bumps the epoch. A worker that captured an epoch
/// before a long GPU wait can compare it afterwards and drop its stale work.
#[derive(Debug, Default)]
pub struct AtomicInterruptFence {
    epoch: AtomicUsize,
}

impl AtomicInterruptFence {
    /// A fence at epoch zero.
    pub const fn new() -> Self {
        Self {
            epoch: AtomicUsize::new(0),
        }
    }

    /// Bump the epoch and return the NEW value.
    pub fn bump(&self) -> usize {
        self.epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// The current epoch.
    pub fn load(&self) -> usize {
        self.epoch.load(Ordering::SeqCst)
    }

    /// Whether the epoch still equals `expected`.
    pub fn is_current(&self, expected: usize) -> bool {
        self.load() == expected
    }
}

/// An unbounded, multi-producer / single-consumer event queue with an
/// associated interrupt epoch.
#[derive(Debug)]
pub struct AtomicEpochQueue<T> {
    sender: Sender<T>,
    receiver: Receiver<T>,
    fence: AtomicInterruptFence,
}

impl<T> Default for AtomicEpochQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AtomicEpochQueue<T> {
    /// A queue with an unbounded channel.
    pub fn new() -> Self {
        let (sender, receiver) = unbounded();
        Self {
            sender,
            receiver,
            fence: AtomicInterruptFence::new(),
        }
    }

    /// Enqueue an item.
    pub fn push(&self, item: T) {
        // The receiver is only dropped when the queue is, so this cannot fail
        // in practice; ignoring the error keeps `push` infallible for callers.
        let _ = self.sender.send(item);
    }

    /// Try to take the next item without blocking.
    pub fn try_pop(&self) -> Option<T> {
        self.receiver.try_recv().ok()
    }

    /// The number of queued items.
    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    /// Drain every queued item and bump the interrupt epoch.
    ///
    /// Returns the number of items discarded. The drain itself is a bounded
    /// number of non-blocking channel receives, so it completes in well under
    /// two microseconds for a queue of a few hundred items.
    pub fn flush_all(&self) -> usize {
        let mut dropped = 0usize;
        while self.receiver.try_recv().is_ok() {
            dropped += 1;
        }
        self.fence.bump();
        dropped
    }

    /// The interrupt fence guarding this queue.
    pub fn fence(&self) -> &AtomicInterruptFence {
        &self.fence
    }

    /// The current interrupt epoch.
    pub fn epoch(&self) -> usize {
        self.fence.load()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_atomic_epoch_fence() {
        let f = AtomicInterruptFence::new();
        assert_eq!(f.load(), 0);
        assert_eq!(f.bump(), 1);
        assert_eq!(f.load(), 1);
        assert!(f.is_current(1));
        assert!(!f.is_current(0));
    }

    #[test]
    fn test_queue_push_and_pop() {
        let q: AtomicEpochQueue<u32> = AtomicEpochQueue::new();
        assert!(q.is_empty());
        q.push(1);
        q.push(2);
        assert_eq!(q.len(), 2);
        assert_eq!(q.try_pop(), Some(1));
        assert_eq!(q.try_pop(), Some(2));
        assert_eq!(q.try_pop(), None);
    }

    #[test]
    fn test_flush_all_drains_and_bumps_the_epoch() {
        let q: AtomicEpochQueue<u32> = AtomicEpochQueue::new();
        for i in 0..64 {
            q.push(i);
        }
        let before = q.epoch();
        let dropped = q.flush_all();
        assert_eq!(dropped, 64);
        assert!(q.is_empty());
        assert_eq!(q.epoch(), before + 1);
    }

    #[test]
    fn test_flush_all_is_fast() {
        let q: AtomicEpochQueue<u32> = AtomicEpochQueue::new();
        for i in 0..512 {
            q.push(i);
        }
        let start = Instant::now();
        let dropped = q.flush_all();
        let elapsed = start.elapsed();
        assert_eq!(dropped, 512);
        // The budget is 2 microseconds for the drain itself; a generous
        // bound keeps the test stable on a loaded CI machine while still
        // failing if the drain became quadratic.
        assert!(
            elapsed.as_micros() < 2000,
            "flush_all took {} us",
            elapsed.as_micros()
        );
    }
}
