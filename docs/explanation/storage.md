# Storage Layer (Slab Files)

The **Storage Layer** (provided by the `slabtastic` crate) is a specialized, page-aligned record container format designed for storing non-uniform, randomly-accessible data.

## Key Properties

- **Page-aligned**: Records are organized into fixed-size pages (e.g., 64 KB). Pages can be read independently, allowing for efficient I/O and random access without scanning the entire file.
- **Ordinal-addressed**: Every record in a slab file is assigned a sequential ordinal starting at 0. This ordinal serves as the unique address for the record.
- **Opaque Payloads**: The slab format treats record contents as raw byte slices. The interpretation of these bytes happens at the codec layer above (e.g., [MNode or PNode](./data-model.md)).
- **Namespace Support**: A single slab file can contain multiple independent namespaces, each with its own page index and ordinal sequence.
- **High Scale**: Designed to support files up to 2^63 bytes.

## Page Structure

Each page in a slab file consists of:
- **Header**: Contains page metadata such as magic bytes, flags, and namespace ID.
- **Record Data**: The raw bytes of the records.
- **Footer**: Contains a record index for the page, enabling O(1) lookup of any record within the page by its intra-page index.

## Record Identification: Dialect Leader Byte

Since slab files store opaque byte payloads, the system needs a way to identify the record type without external metadata. This is achieved using a **Dialect Leader Byte** at the start of every record:

| Leader Byte | Record Type |
|-------------|-------------|
| `0x01` | MNode (Metadata Node) |
| `0x02` | PNode (Predicate Node) |
| `0x03` | Generic Data (e.g., JSON) |

This simple tagging system allows the `slab inspect` tool and other components to automatically select the correct codec for decoding and rendering any record in a slab file.

## Decoupling Storage and Logic

The primary advantage of the storage layer is the decoupling of physical storage from logical data interpretation. The `SlabReader` and `SlabWriter` provide a robust, high-performance foundation that can store any type of record, while the `ANode` codec (see [Two-Stage Codec](./two-stage-codec.md)) handles the complex task of binary-to-logical transformation.
