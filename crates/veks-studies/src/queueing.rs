// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Operational analysis of the rewrite: what bounds the transform rate,
//! and which resources are pegged when it is bound.
//!
//! Reporting a single "bound by" winner is the wrong shape. A storage
//! subsystem can peg at essentially full utilization on **command
//! processing and on bandwidth at once** — they are different resources
//! and a strategy can exhaust both. Which one is fractionally larger is
//! not the interesting fact; that both are at 1.0 is, because it says
//! there is no headroom anywhere and only a change of strategy will help.
//!
//! So the model here is a closed queueing network in the operational
//! sense (Denning & Buzen): each transformed record places a **service
//! demand** `D_k` on each resource `k`, and the standard bounds follow
//! without any distributional assumptions at all.
//!
//! ```text
//!   D_max   = max_k D_k                 the bottleneck demand
//!   D_total = Σ_k D_k                    total demand per record
//!
//!   X(n)   ≤ min( n / D_total , 1 / D_max )      throughput bound
//!   R(n)   ≥ max( D_total , n · D_max )          residence bound
//!   U_k    = X · D_k                             utilization
//!   n*     = D_total / D_max                     the knee
//! ```
//!
//! `n*` is worth stating plainly: below it the system is
//! concurrency-limited and more requests in flight buy throughput
//! proportionally; above it the bottleneck is saturated and more
//! concurrency buys only queueing delay. It is the number that decides
//! how deep to issue.
//!
//! # What this says about ordering
//!
//! Written this way, what an ordered rewrite does becomes a single
//! sentence: **it trades command demand for bandwidth demand.** A gather
//! places one command per record on the controller; an ordered pass
//! places one per container, so `D_controller` falls by a factor of `w`.
//! In exchange it reads the source `A(P)` times over, so `D_bandwidth`
//! rises by that factor.
//!
//! It therefore wins exactly when the controller was the bottleneck and
//! bandwidth had headroom to absorb the trade — and it loses when
//! bandwidth was already the bottleneck, which is the same crossover the
//! cost model reaches from the other direction.

/// A resource a transformed record consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resource {
    /// The controller's command-processing rate.
    Controller,
    /// Device and link bandwidth.
    Bandwidth,
    /// Host CPU spent issuing and completing.
    HostCpu,
    /// Time the medium itself is occupied, divided by how many units can
    /// work at once.
    Media,
}

impl Resource {
    pub fn label(self) -> &'static str {
        match self {
            Resource::Controller => "controller",
            Resource::Bandwidth => "bandwidth",
            Resource::HostCpu => "host cpu",
            Resource::Media => "media",
        }
    }

    pub const ALL: [Resource; 4] = [
        Resource::Controller,
        Resource::Bandwidth,
        Resource::HostCpu,
        Resource::Media,
    ];
}

/// Seconds of each resource consumed per record transformed.
#[derive(Debug, Clone, Copy, Default)]
pub struct Demand {
    pub controller: f64,
    pub bandwidth: f64,
    pub host_cpu: f64,
    pub media: f64,
}

impl Demand {
    pub fn get(&self, r: Resource) -> f64 {
        match r {
            Resource::Controller => self.controller,
            Resource::Bandwidth => self.bandwidth,
            Resource::HostCpu => self.host_cpu,
            Resource::Media => self.media,
        }
    }

    /// Total demand per record — the residence time of a record with no
    /// contention at all.
    pub fn total(&self) -> f64 {
        self.controller + self.bandwidth + self.host_cpu + self.media
    }

    /// The bottleneck demand and the resource that carries it.
    pub fn bottleneck(&self) -> (Resource, f64) {
        Resource::ALL.into_iter().map(|r| (r, self.get(r))).fold(
            (Resource::Controller, 0.0),
            |best, next| {
                if next.1 > best.1 { next } else { best }
            },
        )
    }

    /// Maximum transform rate, records per second: `1 / D_max`.
    pub fn max_transform_rate(&self) -> f64 {
        let (_, d_max) = self.bottleneck();
        if d_max <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / d_max
        }
    }

    /// Throughput bound at a given concurrency:
    /// `X(n) ≤ min(n / D_total, 1 / D_max)`.
    pub fn transform_rate_at(&self, concurrency: f64) -> f64 {
        let by_concurrency = if self.total() <= 0.0 {
            f64::INFINITY
        } else {
            concurrency.max(1.0) / self.total()
        };
        by_concurrency.min(self.max_transform_rate())
    }

    /// Residence bound: `R(n) ≥ max(D_total, n · D_max)`.
    pub fn residence_at(&self, concurrency: f64) -> f64 {
        let (_, d_max) = self.bottleneck();
        self.total().max(concurrency.max(1.0) * d_max)
    }

    /// **The knee**: `n* = D_total / D_max`, the concurrency at which the
    /// bottleneck saturates.
    ///
    /// Below it, throughput rises proportionally with requests in flight.
    /// Above it, the bottleneck is already busy and additional
    /// concurrency converts directly into queueing delay.
    pub fn saturation_concurrency(&self) -> f64 {
        let (_, d_max) = self.bottleneck();
        if d_max <= 0.0 {
            f64::INFINITY
        } else {
            self.total() / d_max
        }
    }

    /// Utilization of every resource at a given transform rate.
    ///
    /// **Several of these can be at 1.0 together.** That is the case
    /// worth noticing: it means no resource has headroom and no amount of
    /// tuning will help, only a strategy that places different demands.
    pub fn utilizations(&self, transform_rate: f64) -> [(Resource, f64); 4] {
        Resource::ALL.map(|r| (r, (transform_rate * self.get(r)).min(1.0)))
    }

    /// Resources within `slack` of full utilization at the maximum
    /// transform rate.
    pub fn pegged(&self, slack: f64) -> Vec<Resource> {
        let rate = self.max_transform_rate();
        self.utilizations(rate)
            .into_iter()
            .filter(|(_, u)| *u >= 1.0 - slack)
            .map(|(r, _)| r)
            .collect()
    }

    /// Seconds to transform `records` at the bound, given concurrency.
    pub fn completion_seconds(&self, records: u64, concurrency: f64) -> f64 {
        let rate = self.transform_rate_at(concurrency);
        if rate <= 0.0 {
            f64::INFINITY
        } else {
            records as f64 / rate
        }
    }
}

/// Render the operational analysis of one demand vector.
pub fn render(label: &str, demand: &Demand, records: u64, concurrency: f64) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let rate = demand.transform_rate_at(concurrency);
    let (bottleneck, d_max) = demand.bottleneck();

    let _ = writeln!(s, "\n  {label}");
    let _ = writeln!(
        s,
        "    D_total {:.3} us/record   D_max {:.3} us ({})   n* {:.1}",
        demand.total() * 1e6,
        d_max * 1e6,
        bottleneck.label(),
        demand.saturation_concurrency()
    );
    let _ = writeln!(
        s,
        "    X(n={concurrency:.0}) {:.0} records/s   completion {:.2} h",
        rate,
        demand.completion_seconds(records, concurrency) / 3600.0
    );
    let _ = write!(s, "    utilization ");
    for (resource, u) in demand.utilizations(rate) {
        let _ = write!(s, "{} {:.0}%  ", resource.label(), u * 100.0);
    }
    let _ = writeln!(s);
    s
}

#[cfg(all(test, feature = "heavy-tests"))]
mod tests {
    use super::*;

    fn balanced() -> Demand {
        Demand {
            controller: 1e-6,
            bandwidth: 1e-6,
            host_cpu: 1e-6,
            media: 1e-6,
        }
    }

    #[test]
    fn the_bottleneck_is_the_largest_demand() {
        let d = Demand {
            controller: 5e-6,
            bandwidth: 1e-6,
            ..Demand::default()
        };
        assert_eq!(d.bottleneck().0, Resource::Controller);
        assert!((d.max_transform_rate() - 200_000.0).abs() < 1.0);
    }

    /// The classic asymptotic bounds: throughput rises with concurrency
    /// until the bottleneck saturates, then stops.
    #[test]
    fn throughput_rises_to_the_knee_and_then_stops() {
        let d = balanced();
        let knee = d.saturation_concurrency();
        assert!((knee - 4.0).abs() < 1e-9, "four equal demands knee at n*=4");

        let below = d.transform_rate_at(2.0);
        let at = d.transform_rate_at(knee);
        let above = d.transform_rate_at(knee * 8.0);

        assert!(at > below, "below the knee, concurrency buys throughput");
        assert!(
            (above - at).abs() < 1.0,
            "above it, concurrency buys nothing: {above:.0} against {at:.0}"
        );
    }

    /// Residence time is flat below the knee and rises linearly above it —
    /// which is what makes issuing deeper than `n*` a way to add latency
    /// without adding throughput.
    #[test]
    fn residence_is_flat_below_the_knee_and_linear_above() {
        let d = balanced();
        let knee = d.saturation_concurrency();
        assert!((d.residence_at(1.0) - d.total()).abs() < 1e-12);
        assert!((d.residence_at(knee) - d.total()).abs() < 1e-9);
        let deep = d.residence_at(knee * 4.0);
        assert!(
            (deep / d.residence_at(knee) - 4.0).abs() < 1e-6,
            "four times the knee is four times the residence"
        );
    }

    /// **Both resources can be pegged.** A strategy that exhausts
    /// commands and bandwidth together has no headroom anywhere, and
    /// reporting only whichever is fractionally larger hides that.
    #[test]
    fn more_than_one_resource_can_be_saturated() {
        let d = Demand {
            controller: 4.0e-6,
            bandwidth: 3.98e-6,
            host_cpu: 0.5e-6,
            media: 0.2e-6,
        };
        let pegged = d.pegged(0.02);
        assert!(pegged.contains(&Resource::Controller));
        assert!(
            pegged.contains(&Resource::Bandwidth),
            "bandwidth is within 1% of the bottleneck and is equally the problem"
        );
        assert_eq!(pegged.len(), 2);

        let utilizations = d.utilizations(d.max_transform_rate());
        let high: Vec<f64> = utilizations
            .iter()
            .map(|(_, u)| *u)
            .filter(|u| *u > 0.9)
            .collect();
        assert_eq!(high.len(), 2, "two resources above 90%");
    }

    /// Little's law holds across the bound: `n = X · R`.
    #[test]
    fn littles_law_holds_at_the_bound() {
        let d = balanced();
        for n in [1.0f64, 2.0, 4.0, 16.0, 64.0] {
            let x = d.transform_rate_at(n);
            let r = d.residence_at(n);
            assert!(
                (x * r - n).abs() / n < 1e-9,
                "n={n}: X·R = {:.6} should equal n",
                x * r
            );
        }
    }

    /// Completion is the record count divided by the achieved rate, and
    /// at the bound that is `N · D_max` — linear in the ordinal count and
    /// in the bottleneck demand, which is the whole scaling story.
    #[test]
    fn completion_is_linear_in_ordinals_and_bottleneck_demand() {
        let d = Demand {
            controller: 8e-6,
            ..balanced()
        };
        let knee = d.saturation_concurrency();
        let one = d.completion_seconds(1_000_000, knee);
        let ten = d.completion_seconds(10_000_000, knee);
        assert!((ten / one - 10.0).abs() < 1e-6);
        assert!((one - 1_000_000.0 * 8e-6).abs() / one < 1e-6);
    }
}
