// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The `quantization` attribute: how a pre-quantized
//! base's codes relate to the vectors they stand for.
//!
//! A quantized vector is not self-describing. The codes in
//! `base_vectors` (`.bvecs`, `.i8vecs`) need their parameters — per
//! vector, a scale, a zero point and the sum of the codes — and the
//! scheme that says how to apply them. The parameters are the three
//! standard column facets `quant_scale`, `quant_zero_point` and
//! `quant_sum`, index-aligned with the base; the scheme is this
//! attribute, at the dataset or the profile level:
//!
//! ```yaml
//! attributes:
//!   quantization:
//!     scheme: affine            # affine | symmetric
//!     granularity: per-vector   # per-vector | per-dimension | per-block
//!     codes: u8                 # the element type of base_vectors
//!     range: [0, 255]
//! ```
//!
//! [`TestDataViewExt::quantized_base`](crate::TestDataViewExt::quantized_base)
//! resolves the codes and the three columns as one thing, checked
//! against each other.

use serde::{Deserialize, Serialize};

use crate::typed_access::ElementType;

/// How codes map back to values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuantScheme {
    /// `value = scale * (code - zero_point)`.
    Affine,
    /// `value = scale * code`, with the zero point fixed at zero.
    Symmetric,
}

/// What one scale and zero point cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuantGranularity {
    /// One scale and zero point per vector: the columns hold one value
    /// per base vector.
    PerVector,
    /// One per dimension, shared by every vector.
    PerDimension,
    /// One per block of [`block`](Quantization::block) dimensions.
    PerBlock,
}

/// The `quantization` attribute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quantization {
    /// How codes map back to values.
    pub scheme: QuantScheme,
    /// What one scale and zero point cover.
    pub granularity: QuantGranularity,
    /// Dimensions per block, for [`QuantGranularity::PerBlock`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block: Option<u32>,
    /// The element type of the codes in `base_vectors`, by its name
    /// (`u8`, `i8`).
    #[serde(with = "element_name")]
    pub codes: ElementType,
    /// The inclusive range of code values.
    pub range: [i64; 2],
}

impl Quantization {
    /// What is wrong with the attribute, if anything: a non-integer
    /// code type, a reversed range, a per-block granularity without a
    /// block, or a block on another granularity.
    pub fn validate(&self) -> Result<(), String> {
        if !self.codes.is_integer() {
            return Err(format!("codes must be an integer type, not {}", self.codes));
        }
        if self.range[0] > self.range[1] {
            return Err(format!("range [{}, {}] is reversed", self.range[0], self.range[1]));
        }
        match (self.granularity, self.block) {
            (QuantGranularity::PerBlock, None) => Err("per-block granularity needs a block size".into()),
            (QuantGranularity::PerBlock, Some(0)) => Err("a block of 0 dimensions covers nothing".into()),
            (g, Some(_)) if g != QuantGranularity::PerBlock => {
                Err(format!("a block size means nothing with {} granularity", kebab(g)))
            }
            _ => Ok(()),
        }
    }

    /// The attribute as found in a `quantization` value, validated.
    pub fn from_value(value: &serde_yaml::Value) -> Result<Self, String> {
        let q: Self = serde_yaml::from_value(value.clone()).map_err(|e| e.to_string())?;
        q.validate()?;
        Ok(q)
    }
}

fn kebab(g: QuantGranularity) -> &'static str {
    match g {
        QuantGranularity::PerVector => "per-vector",
        QuantGranularity::PerDimension => "per-dimension",
        QuantGranularity::PerBlock => "per-block",
    }
}

/// `codes` as the type's name.
mod element_name {
    use serde::{Deserialize, Deserializer, Serializer};

    use crate::typed_access::ElementType;

    pub fn serialize<S: Serializer>(ty: &ElementType, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(ty.name())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ElementType, D::Error> {
        let name = String::deserialize(d)?;
        ElementType::from_name(&name)
            .ok_or_else(|| serde::de::Error::custom(format!("'{name}' is not an element type")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(yaml: &str) -> Result<Quantization, String> {
        Quantization::from_value(&serde_yaml::from_str(yaml).unwrap())
    }

    #[test]
    fn the_attribute_round_trips_through_serde() {
        let q = parse("scheme: affine\ngranularity: per-vector\ncodes: u8\nrange: [0, 255]\n").unwrap();
        assert_eq!(q.scheme, QuantScheme::Affine);
        assert_eq!(q.granularity, QuantGranularity::PerVector);
        assert_eq!(q.codes, ElementType::U8);
        assert_eq!(q.range, [0, 255]);
        let text = serde_yaml::to_string(&q).unwrap();
        assert!(text.contains("codes: u8") && text.contains("granularity: per-vector"), "{text}");
        assert_eq!(serde_yaml::from_str::<Quantization>(&text).unwrap(), q);

        let blocked = parse("scheme: symmetric\ngranularity: per-block\nblock: 64\ncodes: i8\nrange: [-127, 127]\n").unwrap();
        assert_eq!((blocked.block, blocked.codes), (Some(64), ElementType::I8));
    }

    #[test]
    fn a_malformed_attribute_says_what_is_wrong() {
        let err = parse("scheme: affine\ngranularity: per-vector\ncodes: f32\nrange: [0, 255]\n").unwrap_err();
        assert!(err.contains("integer type"), "{err}");
        let err = parse("scheme: affine\ngranularity: per-vector\ncodes: u8\nrange: [255, 0]\n").unwrap_err();
        assert!(err.contains("reversed"), "{err}");
        let err = parse("scheme: affine\ngranularity: per-block\ncodes: u8\nrange: [0, 255]\n").unwrap_err();
        assert!(err.contains("block size"), "{err}");
        let err = parse("scheme: affine\ngranularity: per-vector\nblock: 8\ncodes: u8\nrange: [0, 255]\n").unwrap_err();
        assert!(err.contains("means nothing with per-vector"), "{err}");
        let err = parse("scheme: affine\ngranularity: per-vector\ncodes: bytes\nrange: [0, 255]\n").unwrap_err();
        assert!(err.contains("not an element type"), "{err}");
        let err = parse("scheme: linear\ngranularity: per-vector\ncodes: u8\nrange: [0, 255]\n").unwrap_err();
        assert!(err.contains("linear"), "{err}");
    }
}
