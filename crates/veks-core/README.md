# veks-core

Foundation modules shared by the [`veks`](https://crates.io/crates/veks)
CLI and its [`veks-pipeline`](https://crates.io/crates/veks-pipeline)
engine: what both need but neither should own. It sits on top of
[`vectordata`](https://crates.io/crates/vectordata), the dataset-access
library.

- **`formats`** covers vector and record file formats:
  - the `VecFormat` taxonomy and extension detection;
  - readers and writers for xvec, npy, parquet and slab sources and sinks;
  - f16/f32/f64 element conversion on the
    [`veks-simd`](https://crates.io/crates/veks-simd) kernels;
  - the parquet compilers that turn columnar metadata into MNode records.
- **`ui`** is the progress and logging abstraction every long-running
  command reports through: a `UiHandle` in front of a pluggable sink
  (ratatui, plain text, headless, or a capturing test sink).
- **`filters`** decides which files in a dataset directory are content,
  which are infrastructure and which are excluded. It is re-exported from
  `vectordata`, which owns the rules.
- **`term`** and **`paths`** provide terminal styling and path display.
- **`legacy_sweep`** removes the singular-extension xvec links that
  predate extension normalization.

```rust
use veks_core::formats::{VecFormat, convert::convert_elements};

assert_eq!(VecFormat::from_extension("fvecs"), Some(VecFormat::Fvec));

// Widen one f16 element (1.0) to f32, as `veks transform convert` does.
let f16_one = half::f16::from_f32(1.0).to_le_bytes();
let f32_bytes = convert_elements(&f16_one, 2, 4).unwrap();
assert_eq!(f32::from_le_bytes(f32_bytes.try_into().unwrap()), 1.0);
```

This is an internal layer of the toolkit. To read datasets, use
`vectordata`; to build them, use `veks`.

License: Apache-2.0
