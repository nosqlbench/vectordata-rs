// Copyright (c) nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! Retry policy with exponential backoff and jitter.

use std::io;
use std::thread;
use std::time::Duration;

/// Configurable retry policy with exponential backoff and jitter.
///
/// Delay formula: `min(base_delay * 2^attempt, max_delay) * (1 + random(0, jitter))`.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries, just one attempt).
    pub max_retries: u32,
    /// Base delay in milliseconds before first retry.
    pub base_delay_ms: u64,
    /// Maximum delay cap in milliseconds.
    pub max_delay_ms: u64,
    /// Jitter fraction (0.0 to 1.0). Applied as multiplicative jitter.
    pub jitter_fraction: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy {
            max_retries: 10,
            base_delay_ms: 1000,
            max_delay_ms: 30_000,
            jitter_fraction: 0.10,
        }
    }
}

impl RetryPolicy {
    /// Execute a fallible operation with retries.
    ///
    /// Returns the first successful result, or the last error after all
    /// retries are exhausted.
    pub fn execute<F>(&self, mut op: F) -> io::Result<Vec<u8>>
    where
        F: FnMut() -> io::Result<Vec<u8>>,
    {
        let mut last_err = None;

        for attempt in 0..=self.max_retries {
            match op() {
                Ok(data) => return Ok(data),
                Err(e) => {
                    last_err = Some(e);
                    if attempt < self.max_retries {
                        let delay = self.delay_for_attempt(attempt);
                        thread::sleep(delay);
                    }
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "retry policy exhausted")
        }))
    }

    /// Compute the delay for a given attempt number.
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base = self.base_delay_ms as f64 * (2.0f64).powi(attempt as i32);
        let capped = base.min(self.max_delay_ms as f64);
        let jitter = 1.0 + self.jitter_fraction * pseudo_random();
        Duration::from_millis((capped * jitter) as u64)
    }
}

/// Simple pseudo-random [0, 1) based on current time nanos.
/// Not cryptographic — just enough to decorrelate retries.
fn pseudo_random() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_succeeds_first_try() {
        let policy = RetryPolicy::default();
        let result = policy.execute(|| Ok(vec![1, 2, 3]));
        assert_eq!(result.unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_succeeds_after_retries() {
        let attempts = AtomicU32::new(0);
        let policy = RetryPolicy {
            max_retries: 5,
            base_delay_ms: 1,
            max_delay_ms: 1,
            jitter_fraction: 0.0,
        };

        let result = policy.execute(|| {
            let n = attempts.fetch_add(1, Ordering::Relaxed);
            if n < 2 {
                Err(io::Error::new(io::ErrorKind::ConnectionReset, "fail"))
            } else {
                Ok(vec![42])
            }
        });

        assert_eq!(result.unwrap(), vec![42]);
        assert_eq!(attempts.load(Ordering::Relaxed), 3); // 2 fails + 1 success
    }

    #[test]
    fn test_exhausts_retries() {
        let policy = RetryPolicy {
            max_retries: 2,
            base_delay_ms: 1,
            max_delay_ms: 1,
            jitter_fraction: 0.0,
        };

        let attempts = AtomicU32::new(0);
        let result = policy.execute(|| {
            attempts.fetch_add(1, Ordering::Relaxed);
            Err(io::Error::new(io::ErrorKind::ConnectionReset, "always fail"))
        });

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::Relaxed), 3); // 1 initial + 2 retries
    }

    #[test]
    fn test_delay_exponential_backoff() {
        let policy = RetryPolicy {
            max_retries: 10,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            jitter_fraction: 0.0,
        };

        // Without jitter, delays should be exact powers of 2
        let d0 = policy.delay_for_attempt(0);
        let d1 = policy.delay_for_attempt(1);
        let d2 = policy.delay_for_attempt(2);

        assert_eq!(d0.as_millis(), 100);
        assert_eq!(d1.as_millis(), 200);
        assert_eq!(d2.as_millis(), 400);

        // Should cap at max_delay
        let d10 = policy.delay_for_attempt(10);
        assert_eq!(d10.as_millis(), 5000);
    }
}
