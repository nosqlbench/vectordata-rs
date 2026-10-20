// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Distance metrics.

use veks_datatypes::DistanceFunction;

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

    /// The metric a spelling names: any spelling of
    /// [`DistanceFunction::parse`] for one of the four this crate
    /// computes, so `euclidean`, `angular`, `ip` and `taxicab` are read
    /// as the dataset attribute reads them. `None` for an unknown
    /// spelling and for a function with no kernel here (Hamming,
    /// Jaccard). Named `parse` rather than `from_str` so it can't be
    /// confused with `std::str::FromStr::from_str`, which returns
    /// `Result`.
    ///
    /// ```
    /// use veks_simd::Metric;
    /// assert_eq!(Metric::parse("angular"), Some(Metric::Cosine));
    /// assert_eq!(Metric::parse("IP"), Some(Metric::DotProduct));
    /// assert_eq!(Metric::parse("hamming"), None);
    /// ```
    pub fn parse(s: &str) -> Option<Self> {
        DistanceFunction::parse(s).and_then(|f| Metric::try_from(f).ok())
    }

    /// The spelling a dataset's `distance_function` attribute writes
    /// for this metric ([`DistanceFunction::name`]).
    pub fn name(self) -> &'static str {
        DistanceFunction::from(self).name()
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

impl From<Metric> for DistanceFunction {
    fn from(m: Metric) -> Self {
        match m {
            Metric::L2 => DistanceFunction::L2,
            Metric::Cosine => DistanceFunction::Cosine,
            Metric::DotProduct => DistanceFunction::DotProduct,
            Metric::L1 => DistanceFunction::L1,
        }
    }
}

/// A declared distance function no kernel here computes; the error of
/// `Metric::try_from(DistanceFunction)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoKernel(pub DistanceFunction);

impl std::fmt::Display for NoKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = Metric::ALL.iter().map(|m| m.name()).collect();
        write!(f, "no distance kernel computes {}; the kernels are {}", self.0, names.join(", "))
    }
}

impl std::error::Error for NoKernel {}

impl TryFrom<DistanceFunction> for Metric {
    type Error = NoKernel;

    fn try_from(f: DistanceFunction) -> Result<Self, NoKernel> {
        match f {
            DistanceFunction::L2 => Ok(Metric::L2),
            DistanceFunction::Cosine => Ok(Metric::Cosine),
            DistanceFunction::DotProduct => Ok(Metric::DotProduct),
            DistanceFunction::L1 => Ok(Metric::L1),
            other => Err(NoKernel(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every metric is a declared function and back**, and every
    /// declared function either is a metric or is refused by name: the
    /// two tables cannot drift.
    #[test]
    fn every_metric_round_trips_through_its_distance_function() {
        for &m in &Metric::ALL {
            assert_eq!(Metric::try_from(DistanceFunction::from(m)), Ok(m));
            for &s in DistanceFunction::from(m).spellings() {
                assert_eq!(Metric::parse(s), Some(m), "{s}");
            }
        }
        for &f in DistanceFunction::ALL {
            match Metric::try_from(f) {
                Ok(m) => assert_eq!(DistanceFunction::from(m), f),
                Err(e) => {
                    assert_eq!(e, NoKernel(f));
                    assert!(e.to_string().contains(f.name()), "{e}");
                    assert_eq!(Metric::parse(f.name()), None);
                }
            }
        }
    }
}
