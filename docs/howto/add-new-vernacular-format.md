<!-- Copyright (c) Jonathan Shook -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# How to Add a New Vernacular Format

To add a new human-readable format to the vernacular codec, you need to touch
three places.

## 1. Add the variant to `Vernacular`

In `vectordata/src/formats/anode_vernacular.rs`, add your variant to the enum:

```rust
pub enum Vernacular {
    // ... existing variants ...
    MyFormat,
}
```

## 2. Update `Vernacular::from_str`

Add a mapping so the CLI can parse the format name:

```rust
"myformat" => Some(Self::MyFormat),
```

## 3. Add rendering logic

In the `render_mnode` and `render_pnode` functions, add match arms for your
format. You can create a new helper function:

```rust
fn render_mnode(m: &MNode, vernacular: Vernacular) -> String {
    match vernacular {
        // ... existing arms ...
        Vernacular::MyFormat => render_mnode_myformat(m),
    }
}

fn render_mnode_myformat(m: &MNode) -> String {
    // Build your output string from m.fields
    todo!()
}
```

Do the same for `render_pnode` if your format applies to predicate trees.

## 4. Add parsing logic (optional)

If your format should support text → ANode parsing, add a match arm in the
`parse` function:

```rust
Vernacular::MyFormat => parse_myformat(text),
```

If parsing is not supported, the existing fallback returns a descriptive error.

## 5. Add tests

Add tests in the `#[cfg(test)] mod tests` block:

```rust
#[test]
fn test_render_mnode_myformat() {
    let m = sample_mnode();
    let text = render(&ANode::MNode(m), Vernacular::MyFormat);
    assert!(text.contains("expected content"));
}
```

## 6. The format is automatically available in `slab inspect`

The `slab inspect` command uses `Vernacular::from_str` to resolve the
`format` option, so your new format is immediately available:

```
veks pipeline run --steps '
  - slab inspect:
      input: data.slab
      ordinals: "0"
      format: myformat
'
```
