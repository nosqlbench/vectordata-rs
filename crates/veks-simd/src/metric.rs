// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Distance metrics.

/// A distance metric, in the "smaller is closer" convention.
///
/// - [`Metric::L2`]: squared Euclidean distance. The square root is
///   monotonic, so leaving it out preserves every ordering.
/// - [`Metric::Cosine`]: `1 − a·b / (|a|·|b|)`. A zero-length operand
///   gives `1.0`.
/// - [`Metric::DotProduct`]: `−a·b`, so a larger inner product sorts as
///   a smaller distance.
/// - [`Metric::L1`]: Manhattan distance, `Σ|aᵢ − bᵢ|`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Squared Euclidean distance.
    L2,
    /// Cosine distance, `1 − cos θ`.
    Cosine,
    /// Negated inner product.
    DotProduct,
    /// Manhattan distance.
    L1,
}

impl Metric {
    /// Every metric, in table order.
    pub const ALL: [Metric; 4] = [Metric::L2, Metric::Cosine, Metric::DotProduct, Metric::L1];

    /// Parse a metric name, case-insensitively.
    ///
    /// Accepts `L2`/`EUCLIDEAN`, `COSINE`, `DOT_PRODUCT`/`DOTPRODUCT`/`DOT`
    /// and `L1`/`MANHATTAN`. Named `parse` rather than `from_str` so it
    /// can't be confused with `std::str::FromStr::from_str`, which
    /// returns `Result`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "L2" | "EUCLIDEAN" => Some(Metric::L2),
            "COSINE" => Some(Metric::Cosine),
            "DOT_PRODUCT" | "DOTPRODUCT" | "DOT" => Some(Metric::DotProduct),
            "L1" | "MANHATTAN" => Some(Metric::L1),
            _ => None,
        }
    }

    /// This metric's position in a per-metric kernel table.
    pub(crate) const fn index(self) -> usize {
        match self {
            Metric::L2 => 0,
            Metric::Cosine => 1,
            Metric::DotProduct => 2,
            Metric::L1 => 3,
        }
    }
}
