// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The distance function a dataset declares, and the one table of the
//! spellings that name it.

/// The distance function a dataset's `distance_function` attribute
/// names: the measure its ground truth was computed under.
///
/// Catalogs spell these many ways (`L2` or `euclidean`, `COSINE` or
/// `angular`, `DOT_PRODUCT`, `ip` or `mips`, …). [`parse`](Self::parse)
/// reads every spelling in the workspace's one table, and
/// [`name`](Self::name) is the spelling written back. This is the
/// declaration, not a kernel: `veks_simd::Metric` is what the
/// distance kernels compute, and it converts from the four of these it
/// implements.
///
/// ```
/// use veks_datatypes::DistanceFunction;
///
/// assert_eq!(DistanceFunction::parse("angular"), Some(DistanceFunction::Cosine));
/// assert_eq!(DistanceFunction::parse("Inner-Product"), Some(DistanceFunction::DotProduct));
/// assert_eq!(DistanceFunction::parse("taxicab"), Some(DistanceFunction::L1));
/// assert_eq!(DistanceFunction::parse("chebyshev"), None);
/// assert_eq!(DistanceFunction::DotProduct.name(), "DOT_PRODUCT");
/// assert_eq!("euclidean".parse::<DistanceFunction>()?.to_string(), "L2");
/// # Ok::<(), veks_datatypes::UnknownDistanceFunction>(())
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceFunction {
    /// Euclidean distance. Ground truth may record it squared; the
    /// ordering is the same.
    L2,
    /// Cosine distance or similarity: the angle between the vectors.
    Cosine,
    /// The inner product, larger nearer (maximum inner product search).
    DotProduct,
    /// Manhattan distance, `Σ|aᵢ − bᵢ|`.
    L1,
    /// Hamming distance, over bits or codes.
    Hamming,
    /// Jaccard distance, over sets.
    Jaccard,
}

impl DistanceFunction {
    /// Every distance function, in table order.
    pub const ALL: &'static [DistanceFunction] = &[
        DistanceFunction::L2,
        DistanceFunction::Cosine,
        DistanceFunction::DotProduct,
        DistanceFunction::L1,
        DistanceFunction::Hamming,
        DistanceFunction::Jaccard,
    ];

    /// The spelling written to a `distance_function` attribute: `L2`,
    /// `COSINE`, `DOT_PRODUCT`, `L1`, `HAMMING`, `JACCARD`.
    pub fn name(self) -> &'static str {
        self.spellings()[0]
    }

    /// Every spelling [`parse`](Self::parse) reads as this function,
    /// [`name`](Self::name) first, in the upper-case, underscore form
    /// parse compares against.
    ///
    /// ```
    /// use veks_datatypes::DistanceFunction;
    /// assert_eq!(DistanceFunction::Cosine.spellings(), ["COSINE", "COS", "ANGULAR"]);
    /// ```
    pub fn spellings(self) -> &'static [&'static str] {
        match self {
            DistanceFunction::L2 => &["L2", "EUCLIDEAN"],
            DistanceFunction::Cosine => &["COSINE", "COS", "ANGULAR"],
            DistanceFunction::DotProduct => &["DOT_PRODUCT", "DOTPRODUCT", "DOT", "IP", "INNER_PRODUCT", "MIPS"],
            DistanceFunction::L1 => &["L1", "MANHATTAN", "TAXICAB"],
            DistanceFunction::Hamming => &["HAMMING"],
            DistanceFunction::Jaccard => &["JACCARD"],
        }
    }

    /// The function a spelling names, or `None` for one the table does
    /// not hold. Case is ignored, surrounding space is trimmed, and a
    /// `-` or a space inside counts as `_`, so `Inner-Product`,
    /// `inner product` and `INNER_PRODUCT` are one spelling.
    pub fn parse(s: &str) -> Option<Self> {
        let key: String = s
            .trim()
            .chars()
            .map(|c| if c == '-' || c == ' ' { '_' } else { c.to_ascii_uppercase() })
            .collect();
        DistanceFunction::ALL.iter().copied().find(|f| f.spellings().contains(&key.as_str()))
    }
}

impl std::fmt::Display for DistanceFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A spelling [`DistanceFunction::parse`] does not hold; the
/// [`FromStr`](std::str::FromStr) error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownDistanceFunction(pub String);

impl std::fmt::Display for UnknownDistanceFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = DistanceFunction::ALL.iter().map(|d| d.name()).collect();
        write!(f, "unknown distance function '{}'; expected one of {}", self.0, names.join(", "))
    }
}

impl std::error::Error for UnknownDistanceFunction {}

impl std::str::FromStr for DistanceFunction {
    type Err = UnknownDistanceFunction;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        DistanceFunction::parse(s).ok_or_else(|| UnknownDistanceFunction(s.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every spelling names exactly one function**, and its own: two
    /// rows holding one spelling would make parse depend on table order.
    #[test]
    fn every_spelling_names_one_function() {
        for &f in DistanceFunction::ALL {
            for &s in f.spellings() {
                let owners: Vec<_> = DistanceFunction::ALL.iter().filter(|g| g.spellings().contains(&s)).collect();
                assert_eq!(owners, [&f], "{s}");
                assert_eq!(DistanceFunction::parse(s), Some(f), "{s}");
                assert_eq!(DistanceFunction::parse(&s.to_lowercase()), Some(f), "{s} lower-case");
                assert_eq!(DistanceFunction::parse(&s.replace('_', "-")), Some(f), "{s} hyphenated");
            }
        }
    }

    /// **The written name parses back to its function**, through
    /// `Display` and `FromStr` alike.
    #[test]
    fn the_name_round_trips() {
        for &f in DistanceFunction::ALL {
            assert_eq!(f.to_string(), f.name());
            assert_eq!(f.name().parse::<DistanceFunction>(), Ok(f));
        }
    }

    /// **An unknown spelling is `None`**, and its `FromStr` error names
    /// the spelling and what is accepted.
    #[test]
    fn an_unknown_spelling_is_refused_by_name() {
        for s in ["", " ", "chebyshev", "l3", "cosine_similarity", "dot product product"] {
            assert_eq!(DistanceFunction::parse(s), None, "{s:?}");
        }
        let err = "chebyshev".parse::<DistanceFunction>().unwrap_err().to_string();
        assert!(err.contains("'chebyshev'") && err.contains("DOT_PRODUCT"), "{err}");
    }

    /// **Surrounding space is not part of a spelling.**
    #[test]
    fn surrounding_space_is_trimmed() {
        assert_eq!(DistanceFunction::parse("  cosine \n"), Some(DistanceFunction::Cosine));
        assert_eq!(DistanceFunction::parse(" inner product "), Some(DistanceFunction::DotProduct));
    }
}
