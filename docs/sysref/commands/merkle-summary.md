# merkle summary

Display merkle hash tree structure for an .mref file.

## Usage

```bash
veks pipeline merkle summary --source <file.mref>
```

## Example

```bash
veks pipeline merkle summary --source profiles/base/base_vectors.fvec.mref
```

```
MERKLE REFERENCE FILE SUMMARY
============================
File: ./profiles/base/base_vectors.fvec.mref
File Size: 77 bytes
Content File Size: 516000 bytes
Chunk Size: 1048576 bytes
Number of Chunks: 1

Tree Shape:
  Leaf Nodes: 1
  Padded Leaves: 1
  Internal Nodes: 0
  Total Nodes: 1
  Tree Depth: 1
```
