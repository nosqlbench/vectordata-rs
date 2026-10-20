// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. The element model shared by every reader in the
//! vectordata workspace.
//!
//! A vector file holds runs of one element type. Three things about
//! that type have to be agreed on by everything that reads or writes
//! such a file, and this crate is where they are stated once:
//!
//! - **What the types are.** [`ElementType`] names them: the eight
//!   integer widths, IEEE `f16`, brain `bf16`, `f32` and `f64`. The
//!   [`Element`] trait is the one bound a generic reader puts on its
//!   element parameter.
//! - **How a file says which it holds.** The file extension, through
//!   one table: [`ElementType::extension`], [`ElementType::aliases`]
//!   and [`ElementType::parse_extension`], for the three
//!   [`Container`] layouts.
//! - **What a reader may do when the caller asks for a different
//!   type.** [`ElementType::conversion_to`] answers with an
//!   [`ElementConversion`]: the file's own type is read in place;
//!   a lossless conversion widens; a same-width sign change is
//!   checked value by value; everything else is refused. A reader
//!   never reinterprets bytes as a type the file does not hold.
//!
//! One more name is shared the same way: the distance function a
//! dataset declares. [`DistanceFunction`] holds the one table of its
//! spellings, so a catalog's `angular` and a pipeline's `COSINE` are
//! read as the same thing everywhere.
//!
//! Nothing here touches a file. The readers live in `veks-io` and
//! `vectordata`; both depend on this crate so that they agree.
//!
//! ```
//! use veks_datatypes::{Container, ElementConversion, ElementType, bf16};
//!
//! let (ty, container) = ElementType::parse_extension("bf16vecs").unwrap();
//! assert_eq!((ty, container), (ElementType::BF16, Container::Vec));
//!
//! // bf16 widens to f32 without loss; f32 never narrows to bf16.
//! assert_eq!(ElementType::BF16.conversion_to(ElementType::F32), ElementConversion::Widening);
//! assert_eq!(ElementType::F32.conversion_to(ElementType::BF16), ElementConversion::Refused);
//!
//! let x = bf16::from_f32(1.5);
//! assert_eq!(x.to_f32(), 1.5);
//! ```

#![warn(missing_docs)]
// An example that calls a deprecated item teaches the old API; only a
// deprecated item's own example may, and it says so with an allow.
#![doc(test(attr(deny(deprecated))))]

mod conversion;
mod distance;
mod element;
mod element_type;

pub use conversion::{ElementConversion, Overflow, convert};
pub use distance::{DistanceFunction, UnknownDistanceFunction};
pub use element::{Element, Value};
pub use element_type::{Container, ElementKind, ElementType, FloatFormat};

/// The 16-bit floating point types, re-exported so that no crate
/// downstream needs its own `half` dependency to name them.
///
/// [`struct@bf16`] is `#[repr(transparent)]` over `u16`: a program
/// with its own transparent bf16 wrapper may reinterpret a `&[bf16]`
/// as a slice of it. It is never a `u16` to this crate;
/// [`ElementType::BF16`] carries the kind, and
/// [`ElementType::conversion_to`] keeps it apart from `u16` and from
/// [`struct@f16`].
pub use half::{bf16, f16};

/// Run one body at the Rust type of a runtime [`ElementType`].
///
/// `$T` is bound, inside `$body`, to the element type the value names,
/// so a reader generic over [`Element`] is reached without writing the
/// twelve-arm match at every site:
///
/// ```
/// use veks_datatypes::{Element, ElementType, dispatch_element};
///
/// fn width_of(ty: ElementType) -> usize {
///     dispatch_element!(ty, T => <T as Element>::SIZE)
/// }
/// assert_eq!(width_of(ElementType::BF16), 2);
/// assert_eq!(width_of(ElementType::F64), 8);
/// ```
#[macro_export]
macro_rules! dispatch_element {
    ($ty:expr, $T:ident => $body:expr) => {{
        #[allow(unreachable_patterns)]
        match $ty {
            $crate::ElementType::U8 => { type $T = u8; $body }
            $crate::ElementType::I8 => { type $T = i8; $body }
            $crate::ElementType::U16 => { type $T = u16; $body }
            $crate::ElementType::I16 => { type $T = i16; $body }
            $crate::ElementType::U32 => { type $T = u32; $body }
            $crate::ElementType::I32 => { type $T = i32; $body }
            $crate::ElementType::U64 => { type $T = u64; $body }
            $crate::ElementType::I64 => { type $T = i64; $body }
            $crate::ElementType::F16 => { type $T = $crate::f16; $body }
            $crate::ElementType::BF16 => { type $T = $crate::bf16; $body }
            $crate::ElementType::F32 => { type $T = f32; $body }
            $crate::ElementType::F64 => { type $T = f64; $body }
            // `ElementType` is non-exhaustive for crates downstream;
            // every variant this crate defines has an arm above.
            _ => unreachable!("element type without a dispatch arm"),
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_reaches_every_type_at_its_own_width() {
        for ty in ElementType::ALL {
            let width = dispatch_element!(*ty, T => <T as Element>::SIZE);
            assert_eq!(width, ty.byte_width(), "{ty}");
            let named = dispatch_element!(*ty, T => <T as Element>::TYPE);
            assert_eq!(named, *ty);
        }
    }
}
