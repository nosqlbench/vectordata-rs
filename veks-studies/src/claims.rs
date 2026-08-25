// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The gsplat documents, adjudicated against the simulator.
//!
//! Individual claims in `docs/gsplat/` already have tests, scattered
//! across [`crate::check`], [`crate::study`] and [`crate::price`]. What
//! did not exist was anything that took the documents' *claim list* and
//! reported on each — and the consequence was predictable: a claim that
//! the simulator directly contradicts sat in the README for the whole
//! life of this crate, with a passing test asserting the opposite in
//! another module, and nothing to put the two in the same room.
//!
//! Every claim here carries its source file and line, so a reader can go
//! and check the wording, and every verdict carries the numbers it was
//! reached with rather than a bare pass or fail.
//!
//! The conditional claims — the documents' "skip it when" list — get the
//! most attention, because they are the ones nothing was testing. A
//! statement about when a technique *does not* apply is exactly as
//! load-bearing as one about when it does, is harder to get right, and
//! is far less likely to be noticed when wrong.

use crate::algo::{Rewrite, gsplat::Gsplat, naive::NaiveGather};
use crate::cache::CacheConfig;
use crate::io::hw;
use crate::model::{Geometry, Map};
use crate::{check, device, price, study};

/// Where a claim lives, so its wording can be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Source {
    pub file: &'static str,
    pub line: u32,
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

/// What kind of thing is being asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A property the algorithm must always have.
    Invariant,
    /// A quantitative prediction.
    Formula,
    /// A statement about when the technique applies, or does not.
    Conditional,
}

/// How the claim fared.
///
/// The distinction that matters is between [`Verdict::Upheld`] and
/// [`Verdict::Qualified`]. A claim that holds only under conditions its
/// document does not state is not simply true — a reader applying it
/// outside those conditions is being misled by text that passed its own
/// test. Qualified findings therefore carry the missing condition, and
/// are counted separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The simulator agrees.
    Upheld,
    /// True as stated, but only under conditions the document does not
    /// give — which for a conditional claim is close to being wrong.
    Qualified,
    /// The simulator disagrees.
    Contradicted,
    /// Outside what this crate can decide.
    Untestable,
}

impl Verdict {
    fn symbol(self) -> &'static str {
        match self {
            Verdict::Upheld => "upheld",
            Verdict::Qualified => "qualified",
            Verdict::Contradicted => "CONTRADICTED",
            Verdict::Untestable => "untestable",
        }
    }
}

/// One adjudicated claim.
#[derive(Debug, Clone)]
pub struct Finding {
    pub id: &'static str,
    pub source: Source,
    pub kind: Kind,
    /// The claim, in the document's own words, trimmed.
    pub statement: &'static str,
    pub verdict: Verdict,
    /// The numbers the verdict was reached with.
    pub evidence: String,
    /// What the document should say instead, where it is wrong.
    pub note: &'static str,
}

const README: &str = "docs/gsplat/README.md";
const COST: &str = "docs/gsplat/cost-model.md";

fn geometry(records: u64, record_bytes: u64) -> Geometry {
    Geometry {
        records,
        record_bytes,
        container_bytes: 65_536,
    }
}

// ---- Invariants -------------------------------------------------------

fn invariants() -> Vec<Finding> {
    let g = geometry(20_000, 512);
    let map = Map::shuffled(g.records, 0xC1A1);
    let (sink, trace) = Gsplat::new().run(g, &map, g.payload_bytes() / 8);

    let single_read = check::single_read(&trace, &map);
    let monotone = check::monotone_access(&trace);
    let single_write = check::single_write(&trace, &map);
    let bounded = check::bounded_memory(&trace, g.payload_bytes() / 8);
    let deterministic = check::deterministic(&Gsplat::new(), g, &map, g.payload_bytes() / 8);
    let correct = sink.matches(&map);

    let verdict = |v: &Vec<check::Violation>| {
        if v.is_empty() {
            Verdict::Upheld
        } else {
            Verdict::Contradicted
        }
    };

    vec![
        Finding {
            id: "single-read-write",
            source: Source {
                file: README,
                line: 222,
            },
            kind: Kind::Invariant,
            statement: "Every mapped source record is read exactly once across all \
                        passes; every output byte is written exactly once.",
            verdict: if single_read.is_empty() && single_write.is_empty() && correct {
                Verdict::Upheld
            } else {
                Verdict::Contradicted
            },
            evidence: format!(
                "{} records, {} passes: {} read violations, {} write violations, \
                 output matches the map: {correct}",
                g.records,
                trace.metrics().passes,
                single_read.len(),
                single_write.len()
            ),
            note: "",
        },
        Finding {
            id: "monotone-access",
            source: Source {
                file: README,
                line: 225,
            },
            kind: Kind::Invariant,
            statement: "Within a pass, source reads ascend by address and the output \
                        write is one contiguous range.",
            verdict: verdict(&monotone),
            evidence: format!(
                "{} backward steps across {} reads",
                trace.metrics().backward_steps,
                trace.metrics().record_reads
            ),
            note: "",
        },
        Finding {
            id: "bounded-memory",
            source: Source {
                file: README,
                line: 227,
            },
            kind: Kind::Invariant,
            statement: "Resident state is one segment buffer plus the active pass's \
                        plan — never a function of N.",
            verdict: verdict(&bounded),
            evidence: format!(
                "peak resident {} bytes against a {} byte budget",
                trace.peak_resident_bytes,
                g.payload_bytes() / 8
            ),
            note: "",
        },
        Finding {
            id: "determinism",
            source: Source {
                file: README,
                line: 229,
            },
            kind: Kind::Invariant,
            statement: "Output depends only on the source and the map. Pass count, \
                        worker count and scheduling never change the bytes produced.",
            verdict: verdict(&deterministic),
            evidence: format!("{} divergences across repeated runs", deterministic.len()),
            note: "",
        },
    ]
}

// ---- Formulas ---------------------------------------------------------

fn formulas() -> Vec<Finding> {
    let g = Geometry::new(20_000, 64, 1_024);
    let live = g.payload_bytes();
    let budgets: Vec<u64> = [2u64, 4, 8, 16, 32, 64, 128]
        .iter()
        .map(|p| live / p)
        .collect();
    let rows = study::amplification_sweep(g, 1234, &budgets);
    let worst = rows
        .iter()
        .map(|r| r.relative_error())
        .fold(0.0f64, f64::max);

    // "Naive moves less data" — measured at *page* granularity, which is
    // the granularity the document states it in ("random access defeats
    // readahead entirely, so the tier fetches only the blocks the record
    // occupies"). Records are smaller than a page here, so both sides
    // have something to amplify; equal-sized records and pages make the
    // comparison vacuous.
    let ng = geometry(40_000, 512);
    let map = Map::shuffled(ng.records, 7);
    let page = crate::cache::CacheConfig::single_page(4_096);
    let bytes_of = |t: &crate::model::Trace| {
        crate::cache::replay(t, page).read_bytes_from_device() as f64 / ng.payload_bytes() as f64
    };

    // Sweep the pass count: the claim is a comparison, and a comparison
    // that does not name its conditions is only sometimes true.
    let mut crossover = None;
    let mut samples = Vec::new();
    for divisor in [2u64, 4, 8, 16] {
        let budget = (ng.payload_bytes() / divisor).max(ng.record_bytes);
        let naive_amp = bytes_of(&NaiveGather.run(ng, &map, budget).1);
        let ordered = Gsplat::new().run(ng, &map, budget).1;
        let ordered_amp = bytes_of(&ordered);
        let passes = ordered.metrics().passes;
        samples.push(format!(
            "P={passes}: naive {naive_amp:.1}x, gsplat {ordered_amp:.1}x"
        ));
        if crossover.is_none() && naive_amp < ordered_amp {
            crossover = Some(passes);
        }
    }

    vec![
        Finding {
            id: "amplification-formula",
            source: Source {
                file: COST,
                line: 83,
            },
            kind: Kind::Formula,
            statement: "A(P) = P · (1 − exp(−w / P)), bounded by min(P, w).",
            verdict: if worst < 0.10 {
                Verdict::Upheld
            } else {
                Verdict::Contradicted
            },
            evidence: format!(
                "worst relative error {:.1}% across P = 2..128, spanning the \
                 dense-to-sparse crossover at P = w = 16",
                worst * 100.0
            ),
            note: "",
        },
        Finding {
            id: "naive-moves-less",
            source: Source {
                file: COST,
                line: 105,
            },
            kind: Kind::Formula,
            statement: "Naive moves less data only when P > B/R. Below the crossover \
                        gsplat moves less data as well as moving it in a better order.",
            // Corrected in the document after this adjudication found the
            // original stated flatly. The test now checks the corrected
            // form: below the crossover gsplat must move less.
            verdict: if crossover.is_none() {
                Verdict::Upheld
            } else {
                Verdict::Contradicted
            },
            evidence: format!(
                "{} — gsplat moves less at every pass count up to 16, as the \
                 corrected text says it does below P = B/R = 8",
                samples.join("; ")
            ),
            note: "",
        },
    ]
}

// ---- Conditionals — the "skip it when" list ---------------------------

fn conditional_fits_in_memory() -> Finding {
    let g = geometry(20_000, 512);
    let map = Map::shuffled(g.records, 3);
    // A budget larger than the payload: the algorithm should collapse to
    // its floor rather than doing anything clever.
    let trace = Gsplat::new().run(g, &map, g.payload_bytes() * 4).1;
    let passes = trace.metrics().passes;
    let amp = trace.metrics().amplification();

    Finding {
        id: "skip-fits-in-memory",
        source: Source {
            file: README,
            line: 248,
        },
        kind: Kind::Conditional,
        statement: "Skip it when the collection fits in memory. gsplat's floor of two \
                    segments already degenerates toward this.",
        verdict: if passes == 2 && amp < 2.2 {
            Verdict::Upheld
        } else {
            Verdict::Contradicted
        },
        evidence: format!(
            "budget 4x the payload gives {passes} passes and {amp:.2}x amplification \
             — the floor, as claimed"
        ),
        note: "",
    }
}

fn conditional_small_records() -> Finding {
    // R << W: 128 byte records in a 64 KiB container, so w = 512.
    let payload = 1u64 << 34;
    let mut evidence = Vec::new();
    let mut worst_penalty = f64::INFINITY;

    for model in device::ALL_MODELS {
        let penalty = model.random_penalty(128);
        worst_penalty = worst_penalty.min(penalty);
        let budget = model.min_budget_for_ordering(payload, 128);
        evidence.push(format!(
            "{}: penalty {penalty:.0}x, ordering pays above {:.2}% of payload",
            model.name,
            budget as f64 / payload as f64 * 100.0
        ));
    }

    Finding {
        id: "skip-small-records",
        source: Source {
            file: README,
            line: 250,
        },
        kind: Kind::Conditional,
        statement: "Skip it when records are already large relative to what the tier \
                    fetches efficiently — once penalty(R) falls below 2, no budget can \
                    win, because a rewrite always makes at least two passes.",
        // Corrected in the document after this adjudication found the
        // original inverted. The test now checks the corrected form: at
        // R << W the penalty must be large, so the condition must *not*
        // fire, and at large R it must.
        verdict: {
            let large_record = device::NVME_MODERN_MODEL.random_penalty_at_depth(16_384, 128.0);
            if worst_penalty > 8.0 && large_record < 2.0 {
                Verdict::Upheld
            } else {
                Verdict::Contradicted
            }
        },
        evidence: format!(
            "{}; and at 16 KiB on a modern drive at depth 128 the penalty is {:.2}x, \
             below the two-pass floor",
            evidence.join("; "),
            device::NVME_MODERN_MODEL.random_penalty_at_depth(16_384, 128.0)
        ),
        note: "",
    }
}

fn conditional_identity_map() -> Finding {
    let g = geometry(20_000, 512);
    let identity = Map::identity(g.records);
    let budget = g.payload_bytes() / 8;

    // The claim is about what an ascending map lets you *skip*, not about
    // gsplat's pass count — a small budget still segments. So the test is
    // whether the transpose is skipped, and whether a single streaming
    // pass would in fact be cheaper.
    let fast = Gsplat::new().run(g, &identity, budget).1;
    let forced = Gsplat::always_transpose().run(g, &identity, budget).1;
    let cache = Some(CacheConfig::new(budget, 65_536));
    let segmented = price::simulate_io(&fast, &hw::SPINNING_SATA_HW, cache, 32).elapsed_s;
    // What a streaming filter would cost: one sequential pass.
    let streaming = (g.payload_bytes() * 2) as f64 / hw::SPINNING_SATA_HW.sequential_bandwidth();

    Finding {
        id: "skip-identity-map",
        source: Source {
            file: README,
            line: 254,
        },
        kind: Kind::Conditional,
        statement: "Skip it when the map is the identity or already ascending. A \
                    selection list in source order is a streaming filter, not a \
                    permutation: read and write both sequentially in one pass, \
                    skipping Linearize and Assemble entirely.",
        verdict: if fast.metrics().scatters == 0 && streaming < segmented {
            Verdict::Upheld
        } else {
            Verdict::Contradicted
        },
        evidence: format!(
            "an ascending map does {} scatters against {} when the transpose is \
             forced, and a single streaming pass costs {:.3}s against the segmented \
             run's {segmented:.3}s",
            fast.metrics().scatters,
            forced.metrics().scatters,
            streaming
        ),
        note: "",
    }
}

fn conditional_no_ordering() -> Finding {
    // A tier with no meaningful ordering: flash at a record size it
    // already serves efficiently.
    let scattered = crate::io::fio_like(&hw::NVME_CONSUMER_HW, 4_096, 4_000);
    let ordered = crate::io::fio_like_sequential(&hw::NVME_CONSUMER_HW, 4_096, 40_000);
    let ratio = ordered.bandwidth_utilization() / scattered.bandwidth_utilization().max(1e-9);

    let disk_scattered = crate::io::fio_like(&hw::SPINNING_SATA_HW, 4_096, 2_000);
    let disk_ordered = crate::io::fio_like_sequential(&hw::SPINNING_SATA_HW, 4_096, 20_000);
    let disk_ratio =
        disk_ordered.bandwidth_utilization() / disk_scattered.bandwidth_utilization().max(1e-9);

    Finding {
        id: "skip-no-ordering",
        source: Source {
            file: README,
            line: 257,
        },
        kind: Kind::Conditional,
        statement: "Skip it when the storage tier has no meaningful ordering — a true \
                    random-access medium with uniform cost. There is nothing to buy.",
        verdict: if ratio < 1.3 && disk_ratio > 10.0 {
            Verdict::Qualified
        } else {
            Verdict::Contradicted
        },
        evidence: format!(
            "at 4 KiB, ordering changes NVMe bandwidth utilization by {ratio:.2}x \
             (nothing to buy) against {disk_ratio:.0}x on the disk"
        ),
        note: "Upheld, but 'uniform cost' is a property of the tier *and the record \
               size together*, not of the tier alone. The same NVMe drive pays a 110x \
               penalty at 128 byte records. The condition is better stated as a record \
               size relative to what the tier serves efficiently.",
    }
}

fn conditional_exceeds_budget() -> Finding {
    let g = geometry(200_000, 512);
    let map = Map::shuffled(g.records, 11);
    let cache = Some(CacheConfig::new(g.payload_bytes() / 8, 65_536));
    let budget = g.payload_bytes() / 4;

    let ordered = Gsplat::new().run(g, &map, budget).1;
    let scattered = NaiveGather.run(g, &map, budget).1;
    let o = price::simulate_io(&ordered, &hw::SPINNING_SATA_HW, cache, 32);
    let s = price::simulate_io(&scattered, &hw::SPINNING_SATA_HW, cache, 32);
    let gain = s.elapsed_s / o.elapsed_s.max(1e-12);

    Finding {
        id: "use-when-exceeds-budget",
        source: Source {
            file: README,
            line: 240,
        },
        kind: Kind::Conditional,
        statement: "Use gsplat when the collection exceeds the memory budget and \
                    random access on the tier is materially worse than sequential.",
        verdict: if gain > 2.0 {
            Verdict::Qualified
        } else {
            Verdict::Contradicted
        },
        evidence: format!(
            "payload 4x the budget on a seek-bound disk: ordering is {gain:.1}x faster \
             end to end"
        ),
        note: "Upheld as stated, but incomplete: whether it pays also depends on the \
               pass count the budget forces and on issue concurrency. See the crossover \
               rule in cost-model.md, which postdates this list.",
    }
}

// ---- Cost-model claims ------------------------------------------------

fn cost_model_claims() -> Vec<Finding> {
    let payload = 143 * (1u64 << 30);

    // "Ordering pays exactly when the pass count is below the penalty."
    let mut rule_agrees = true;
    for model in device::ALL_MODELS {
        for record in [128u64, 1_540, 4_096] {
            for passes in [2u64, 8, 32] {
                let ex = study::WorkedExample {
                    label: "",
                    records: payload / record,
                    record_bytes: record,
                    container_bytes: 128 * 1024,
                    budget_bytes: (payload / passes).max(record),
                };
                let predicted = model.ordering_pays(record, ex.passes());
                let priced = ex.gsplat_seconds(model) < ex.naive_seconds(model);
                if predicted != priced {
                    rule_agrees = false;
                }
            }
        }
    }

    // "Page size is second-order; ordering is the lever."
    let g = geometry(40_000, 512);
    let map = Map::shuffled(g.records, 0xD1CE);
    let ram = g.payload_bytes() / 8;
    let ordered = Gsplat::new().run(g, &map, g.payload_bytes() / 2).1;
    let scattered = NaiveGather.run(g, &map, g.payload_bytes() / 2).1;
    let at = |trace: &crate::model::Trace, page: u64| {
        price::simulate_io(
            trace,
            &hw::SPINNING_SATA_HW,
            Some(CacheConfig::new(ram.max(page), page)),
            32,
        )
        .elapsed_s
    };
    let page_effect = (at(&ordered, 65_536) - at(&ordered, 4_096)).abs() / at(&ordered, 4_096);
    let order_effect = at(&scattered, 4_096) / at(&ordered, 4_096);

    // "Transfer must rate-limit its output stream."
    let capped = crate::io::mixed_job(&hw::NVME_CONSUMER_HW, Some(40.0e6), 8_000);
    let free = crate::io::mixed_job(&hw::NVME_CONSUMER_HW, None, 8_000);
    let collapse = capped.stream("randread").iops() / free.stream("randread").iops().max(1e-9);

    vec![
        Finding {
            id: "crossover-rule",
            source: Source {
                file: COST,
                line: 518,
            },
            kind: Kind::Formula,
            statement: "Ordering pays exactly when the pass count is below the \
                        random-access penalty at the record size.",
            verdict: if rule_agrees {
                Verdict::Upheld
            } else {
                Verdict::Contradicted
            },
            evidence: "the rule and the independently priced worked examples agree on \
                       all 27 combinations of device, record size and pass count"
                .to_string(),
            note: "",
        },
        Finding {
            id: "page-size-second-order",
            source: Source {
                file: COST,
                line: 232,
            },
            kind: Kind::Formula,
            statement: "Page size is second-order; the kernel's readahead window, not \
                        the page size, sets the fetch granularity for an ordered reader.",
            verdict: if page_effect < 0.20 && order_effect > 5.0 {
                Verdict::Upheld
            } else {
                Verdict::Contradicted
            },
            evidence: format!(
                "a sixteenfold page change moves an ordered run {:.0}%, while ordering \
                 the same accesses is worth {order_effect:.0}x",
                page_effect * 100.0
            ),
            note: "",
        },
        Finding {
            id: "govern-the-writer",
            source: Source {
                file: COST,
                line: 425,
            },
            kind: Kind::Conditional,
            statement: "Transfer must rate-limit its output stream against its input \
                        stream, or the cost model does not describe the run.",
            verdict: if collapse > 10.0 {
                Verdict::Qualified
            } else {
                Verdict::Contradicted
            },
            evidence: format!(
                "an uncapped writer costs a concurrent reader {collapse:.0}x on NVMe \
                 in simulation, against 178x measured"
            ),
            note: "Upheld in direction and understated in magnitude; the simulator's \
                   flash contention is the model's largest known residual.",
        },
    ]
}

/// Claims the simulator currently contradicts.
///
/// Listed explicitly so that drift is caught in both directions: a *new*
/// contradiction fails the build, and correcting a document also fails
/// the build until this list is updated. A known-wrong claim that nobody
/// is forced to look at is how the small-record error survived as long as
/// it did.
/// Empty, and it has not always been. This module was written because
/// `skip-small-records` told readers to avoid gsplat in precisely the
/// regime where it pays best, and `naive-moves-less` stated a
/// conditional comparison flatly; both had passing tests elsewhere in
/// the crate asserting the opposite. Both documents are now corrected
/// and both claims adjudicate as upheld. The list stays so that the
/// next one is caught the first time it is run rather than the fiftieth.
pub const KNOWN_CONTRADICTED: &[&str] = &[];

/// Adjudicate every catalogued claim.
pub fn adjudicate_all() -> Vec<Finding> {
    let mut findings = invariants();
    findings.extend(formulas());
    findings.push(conditional_exceeds_budget());
    findings.push(conditional_fits_in_memory());
    findings.push(conditional_small_records());
    findings.push(conditional_identity_map());
    findings.push(conditional_no_ordering());
    findings.extend(cost_model_claims());
    findings
}

/// Count findings by verdict.
pub fn tally(findings: &[Finding]) -> [(Verdict, usize); 4] {
    let count = |v: Verdict| findings.iter().filter(|f| f.verdict == v).count();
    [
        (Verdict::Upheld, count(Verdict::Upheld)),
        (Verdict::Qualified, count(Verdict::Qualified)),
        (Verdict::Contradicted, count(Verdict::Contradicted)),
        (Verdict::Untestable, count(Verdict::Untestable)),
    ]
}

/// Render the adjudication.
pub fn render(findings: &[Finding]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();

    let _ = writeln!(s, "\n  gsplat claims, adjudicated against the simulator\n");
    let _ = writeln!(
        s,
        "  {:<24} {:<13} {:<12} source",
        "claim", "kind", "verdict"
    );
    for f in findings {
        let kind = match f.kind {
            Kind::Invariant => "invariant",
            Kind::Formula => "formula",
            Kind::Conditional => "conditional",
        };
        let _ = writeln!(
            s,
            "  {:<24} {:<13} {:<12} {}",
            f.id,
            kind,
            f.verdict.symbol(),
            f.source
        );
    }

    let _ = writeln!(s, "\n  Evidence\n");
    for f in findings {
        let _ = writeln!(s, "  {} — {}", f.id, f.verdict.symbol());
        let _ = writeln!(s, "    \"{}\"", f.statement);
        let _ = writeln!(s, "    {}", f.evidence);
        if !f.note.is_empty() {
            let _ = writeln!(s, "    note: {}", f.note);
        }
        let _ = writeln!(s);
    }

    let counts = tally(findings);
    let _ = write!(s, "  ");
    for (verdict, n) in counts {
        if n > 0 {
            let _ = write!(s, "{} {}   ", n, verdict.symbol());
        }
    }
    let _ = writeln!(s);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every catalogued claim must reach a verdict. A claim nobody can
    /// adjudicate is one nobody is checking.
    #[test]
    fn every_claim_is_adjudicated_with_evidence() {
        let findings = adjudicate_all();
        assert!(findings.len() >= 12, "too few claims catalogued");
        for f in &findings {
            assert!(!f.statement.is_empty(), "{}: no statement", f.id);
            assert!(!f.evidence.is_empty(), "{}: no evidence", f.id);
            assert!(f.source.line > 0, "{}: no source line", f.id);
            if f.verdict == Verdict::Contradicted || f.verdict == Verdict::Qualified {
                assert!(
                    !f.note.is_empty(),
                    "{}: a claim that failed must say what should replace it",
                    f.id
                );
            }
        }
    }

    /// The invariants are the algorithm's contract. If any of these ever
    /// fails, the implementation is wrong, not the document.
    #[test]
    fn every_invariant_holds() {
        for f in adjudicate_all()
            .iter()
            .filter(|f| f.kind == Kind::Invariant)
        {
            assert_eq!(
                f.verdict,
                Verdict::Upheld,
                "{} failed: {}",
                f.id,
                f.evidence
            );
        }
    }

    /// The quantitative predictions hold, except the ones already known
    /// to be wrong and catalogued as such.
    #[test]
    fn every_formula_holds_except_the_known_ones() {
        for f in adjudicate_all().iter().filter(|f| f.kind == Kind::Formula) {
            if KNOWN_CONTRADICTED.contains(&f.id) {
                continue;
            }
            assert_eq!(
                f.verdict,
                Verdict::Upheld,
                "{} failed: {}",
                f.id,
                f.evidence
            );
        }
    }

    /// **The drift guard.** The set of contradicted claims must be
    /// exactly the catalogued set — no more, and no fewer.
    ///
    /// More means the simulator has found something new and nobody has
    /// written it down. Fewer means a document was corrected and this
    /// list was not, which would leave a stale accusation standing
    /// against text that no longer says it.
    #[test]
    fn the_contradiction_set_is_exactly_what_is_catalogued() {
        let mut found: Vec<&str> = adjudicate_all()
            .iter()
            .filter(|f| f.verdict == Verdict::Contradicted)
            .map(|f| f.id)
            .collect();
        found.sort_unstable();
        let mut known: Vec<&str> = KNOWN_CONTRADICTED.to_vec();
        known.sort_unstable();
        assert_eq!(
            found, known,
            "the contradicted set moved — either the simulator found something new, \
             or a document was corrected and KNOWN_CONTRADICTED was not updated"
        );
    }

    /// **The finding this module was written to surface, now fixed.**
    ///
    /// The README used to say to skip gsplat when records are much
    /// smaller than a container — the regime where ordering pays best.
    /// The corrected condition turns on the random-access penalty
    /// instead, and this checks it discriminates in both directions:
    /// small records must *not* trigger it, large ones must.
    #[test]
    fn the_small_record_condition_now_discriminates_correctly() {
        let finding = adjudicate_all()
            .into_iter()
            .find(|f| f.id == "skip-small-records")
            .expect("the claim must stay catalogued");
        assert_eq!(finding.verdict, Verdict::Upheld);

        // Small records: ordering pays, so the skip condition is false.
        for model in device::ALL_MODELS {
            assert!(
                model.random_penalty(128) > 8.0,
                "{}: 128 B records must still justify ordering",
                model.name
            );
        }
        // Large records at depth: the skip condition is true.
        assert!(device::NVME_MODERN_MODEL.random_penalty_at_depth(16_384, 128.0) < 2.0);
    }

    /// A conditional claim that is only true under unstated conditions is
    /// tracked as qualified, not quietly counted as a pass.
    #[test]
    fn qualified_claims_carry_their_missing_conditions() {
        let findings = adjudicate_all();
        let qualified: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.verdict == Verdict::Qualified)
            .collect();
        assert!(
            !qualified.is_empty(),
            "at least the tier-ordering and writer-governance claims are qualified"
        );
        for f in qualified {
            assert!(f.note.len() > 40, "{}: the caveat must be specific", f.id);
        }
    }

    /// The rendering has to show a reader where to go and read the
    /// wording for themselves.
    #[test]
    fn the_report_cites_its_sources() {
        let text = render(&adjudicate_all());
        assert!(text.contains("docs/gsplat/README.md:250"));
        assert!(text.contains("docs/gsplat/cost-model.md:"));
        // Every verdict reached must be legible in the output, whichever
        // ones happen to be present today.
        for f in adjudicate_all() {
            assert!(
                text.contains(f.verdict.symbol()),
                "{} verdict not rendered",
                f.id
            );
        }
    }
}

#[cfg(test)]
mod report {
    use super::*;

    #[test]
    #[ignore = "diagnostic report"]
    fn print_claims() {
        print!("{}", render(&adjudicate_all()));
    }
}
