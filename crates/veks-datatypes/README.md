# veks-datatypes

The element model shared by every reader in the vectordata-rs
workspace.

A vector file holds runs of one element type. Three things about that
type have to be agreed on by everything that reads or writes such a
file, and this crate states them once:

- **What the types are.** `ElementType` names them: the eight integer
  widths, IEEE `f16`, brain `bf16`, `f32` and `f64`. The `Element` trait
  is the one bound a generic reader puts on its element parameter.
- **How a file says which it holds.** The file extension, through one
  table (`ElementType::extension`, `aliases`, `parse_extension`) for the
  three `Container` layouts: header-free scalar columns, uniform xvec
  files and variable-length vvec files.
- **What a reader may do when asked for another type.**
  `ElementType::conversion_to` answers with an `ElementConversion`: the
  file's own type is read in place, a lossless conversion widens, a
  same-width sign change is checked value by value, and everything else
  is refused. A reader never reinterprets bytes as a type the file does
  not hold.

It also holds the one table of distance-function spellings
(`DistanceFunction`), so a catalog's `angular` and a pipeline's
`COSINE` are read as the same thing everywhere.

Nothing here touches a file. The readers live in `veks-io` and
`vectordata`, which both depend on this crate so that they agree.

```rust
use veks_datatypes::{Container, ElementConversion, ElementType};

let (ty, container) = ElementType::parse_extension("bf16vecs").unwrap();
assert_eq!((ty, container), (ElementType::BF16, Container::Vec));
assert_eq!(ElementType::BF16.conversion_to(ElementType::F32), ElementConversion::Widening);
assert_eq!(ElementType::F32.conversion_to(ElementType::BF16), ElementConversion::Refused);
```

License: Apache-2.0
