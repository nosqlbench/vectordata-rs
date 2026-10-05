// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Vector data writers for various formats.

pub mod mmap_xvec;
pub mod sharded;
pub mod slab;
pub mod xvec;

use std::path::Path;

use super::VecFormat;

/// Trait for writing vector records to an output format.
///
/// Records are raw element bytes (no dimension prefix). The dimension is
/// configured at writer construction time.
pub trait VecSink {
    /// Write a record with the given ordinal. Data is raw element bytes.
    fn write_record(&mut self, ordinal: i64, data: &[u8]);

    /// Finalize the output (e.g., write pages page for slab).
    ///
    /// Consumes self — an unfinished slab file is corrupt, so the caller
    /// must handle any I/O errors.
    fn finish(self: Box<Self>) -> Result<(), String>;
}

/// Adapter to wrap a `veks_io::VecSink` as a `veks_core::formats::writer::VecSink`.
struct IoSinkAdapter(Option<Box<dyn veks_io::VecSink>>);

impl VecSink for IoSinkAdapter {
    fn write_record(&mut self, ordinal: i64, data: &[u8]) {
        if let Some(ref mut inner) = self.0 {
            inner.write_record(ordinal, data);
        }
    }
    fn finish(mut self: Box<Self>) -> Result<(), String> {
        if let Some(inner) = self.0.take() {
            inner.finish()
        } else {
            Ok(())
        }
    }
}

/// Configuration for opening a sink writer
pub struct SinkConfig {
    pub dimension: u32,
    pub source_format: VecFormat,
    /// Preferred slab page size override. `None` uses the slabtastic default.
    pub slab_page_size: Option<u32>,
    pub slab_namespace: u8,
    /// Optional schema descriptor to emit alongside the content
    /// records as a `:schema` namespace sidecar. Honored only by slab
    /// sinks; other formats ignore it.
    pub schema_sidecar: Option<vectordata::metadata_schema::MetadataSchema>,
    /// Cap on one output file. A facet that would exceed it is
    /// written as a multi-file series instead (SH-35).
    ///
    /// `None` writes a single file whatever its size — which is what
    /// a caller wants for an output that is not a facet, and what
    /// every caller got before the cap existed.
    pub max_shard_bytes: Option<u64>,
}

impl SinkConfig {
    /// Bytes one written record will occupy, framing included.
    ///
    /// The question [`VecFormat::record_bytes`] cannot answer alone
    /// for a slab: the payload is `dimension` elements of the
    /// **source** format's width, because that is what gets written
    /// into it, and a slab has no element width of its own.
    pub fn record_bytes(&self, format: VecFormat) -> Option<u64> {
        if format == VecFormat::Slab {
            Some(self.slab_record_payload() as u64 + super::SLAB_RECORD_FRAMING)
        } else {
            format.record_bytes(self.dimension)
        }
    }

    /// The payload length of one record written into a slab.
    fn slab_record_payload(&self) -> usize {
        let element_size = if self.source_format.is_xvec() || self.source_format.is_scalar() {
            self.source_format.element_size()
        } else {
            4 // npy/parquet sources land as f32
        };
        self.dimension as usize * element_size
    }

    /// The shard stride this config implies, if the output should be
    /// written as a series.
    ///
    /// `None` when there is no cap, when the format's record size is
    /// not a fixed quantity, or when the cap is roomy enough that a
    /// stride would be meaningless — in every case the caller writes
    /// one file, which is what it did before.
    pub fn shard_plan(&self, format: VecFormat) -> Option<vectordata::dataset::ShardPlan> {
        let max = self.max_shard_bytes?;
        let record = self.record_bytes(format)?;
        vectordata::dataset::shard_sizing::plan_fixed(max, record)
    }
}

/// Open a sink writer for the given path, format, and configuration.
///
/// When [`SinkConfig::max_shard_bytes`] implies a series, the sink
/// returned writes across shard files and collapses back to `path` if
/// the run fits in one (SH-83) — so a caller writes records the same
/// way whether or not the output is sharded, and the only visible
/// difference is what ends up on disk.
///
/// Use [`open_sharded_sink`] instead when the caller needs the
/// declaration the output produced.
pub fn open_sink(
    path: &Path,
    format: VecFormat,
    config: &SinkConfig,
) -> Result<Box<dyn VecSink>, String> {
    if let Some(plan) = config.shard_plan(format) {
        return Ok(Box::new(sharded_sink(path, format, config, plan.stride)?));
    }
    open_one_file(path, format, config)
}

/// Open a sharding sink and keep a handle to its outcome.
///
/// The same sink [`open_sink`] would return, typed so the caller can
/// read back what was written — the files, the stride, and whether it
/// collapsed — which is what an emitted `dataset.yaml` needs (SH-37).
pub fn open_sharded_sink(
    path: &Path,
    format: VecFormat,
    config: &SinkConfig,
    stride: u64,
) -> Result<Box<sharded::ShardedSink>, String> {
    Ok(Box::new(sharded_sink(path, format, config, stride)?))
}

fn sharded_sink(
    path: &Path,
    format: VecFormat,
    config: &SinkConfig,
    stride: u64,
) -> Result<sharded::ShardedSink, String> {
    // The per-shard config is this one with the cap removed: a shard
    // is an ordinary file, and leaving the cap on would make each
    // shard try to shard itself.
    let per_shard = SinkConfig {
        max_shard_bytes: None,
        schema_sidecar: config.schema_sidecar.clone(),
        ..*config
    };
    sharded::ShardedSink::open(
        path,
        stride,
        Box::new(move |p: &Path| open_one_file(p, format, &per_shard)),
    )
}

/// Open a sink for exactly one file, ignoring any shard cap.
fn open_one_file(
    path: &Path,
    format: VecFormat,
    config: &SinkConfig,
) -> Result<Box<dyn VecSink>, String> {
    match format {
        VecFormat::Slab => slab::SlabWriter::open(
            path,
            config.slab_record_payload(),
            config.slab_page_size,
            config.slab_namespace,
            config.schema_sidecar.clone(),
        ),
        _ if format.is_xvec() => xvec::XvecWriter::open(path, config.dimension),
        _ if format.is_scalar() => {
            let io_fmt = veks_io::VecFormat::from_extension(format.name())
                .unwrap_or(veks_io::VecFormat::ScalarU8);
            let io_writer = veks_io::scalar::writer::open(path, io_fmt)?;
            Ok(Box::new(IoSinkAdapter(Some(io_writer))))
        }
        _ => Err(format!("{} is not a supported output format", format)),
    }
}

#[cfg(test)]
mod sizing_tests {
    use super::*;

    fn cfg(dim: u32, source: VecFormat, cap: Option<u64>) -> SinkConfig {
        SinkConfig {
            dimension: dim,
            source_format: source,
            slab_page_size: None,
            slab_namespace: 0,
            schema_sidecar: None,
            max_shard_bytes: cap,
        }
    }

    /// A uniform xvec record is its dim header plus its elements, and
    /// the answer agrees with the forward calculation that was already
    /// here — the two must not drift.
    #[test]
    fn a_record_size_is_the_inverse_of_the_file_size() {
        for (fmt, dim) in [
            (VecFormat::Fvec, 384u32),
            (VecFormat::Mvec, 768),
            (VecFormat::Ivec, 100),
            (VecFormat::ScalarU8, 1),
        ] {
            let c = cfg(dim, fmt, None);
            let per = c.record_bytes(fmt).unwrap();
            assert_eq!(
                fmt.expected_file_size(1_000, dim),
                Some(per * 1_000),
                "{fmt} at dim {dim}"
            );
        }
        assert_eq!(cfg(384, VecFormat::Fvec, None).record_bytes(VecFormat::Fvec), Some(1540));
    }

    /// **A slab's record size comes from its source, not from `Slab`.**
    /// A slab has no element width of its own, so the same target
    /// format sizes differently depending on what is written into it.
    #[test]
    fn a_slab_record_is_sized_from_what_is_written_into_it() {
        // f32 source: 4 bytes an element, plus the offset entry.
        assert_eq!(
            cfg(100, VecFormat::Fvec, None).record_bytes(VecFormat::Slab),
            Some(404)
        );
        // f16 source: half that payload, same framing.
        assert_eq!(
            cfg(100, VecFormat::Mvec, None).record_bytes(VecFormat::Slab),
            Some(204)
        );
        // And `VecFormat` alone declines, rather than guessing f32.
        assert_eq!(VecFormat::Slab.record_bytes(100), None);
    }

    /// A vvec record carries its own dimension, so there is no fixed
    /// size to state and nothing pretends otherwise.
    #[test]
    fn a_variable_format_has_no_fixed_record_size() {
        assert_eq!(VecFormat::Fvvec.record_bytes(384), None);
        assert_eq!(VecFormat::Ivvec.record_bytes(384), None);
        assert_eq!(VecFormat::Parquet.record_bytes(384), None);
        assert_eq!(cfg(384, VecFormat::Fvec, None).record_bytes(VecFormat::Fvvec), None);
    }

    /// **No cap means no plan**, which is what every caller got before
    /// the cap existed.
    #[test]
    fn without_a_cap_nothing_is_sharded() {
        assert!(cfg(384, VecFormat::Fvec, None).shard_plan(VecFormat::Fvec).is_none());
    }

    /// A cap produces a decade stride whose projected file stays under
    /// it — the property the whole feature exists for.
    #[test]
    fn a_cap_produces_a_stride_that_fits_under_it() {
        let cap = 1_000_000_000_000u64;
        let c = cfg(384, VecFormat::Fvec, Some(cap));
        let plan = c.shard_plan(VecFormat::Fvec).unwrap();

        assert_eq!(plan.stride, 100_000_000);
        assert_eq!(plan.record_bytes, 1540);
        assert!(plan.projected_bytes() <= cap);
    }

    /// A capped facet whose records vary is written whole rather than
    /// sharded at a guessed stride — the size is not knowable from the
    /// config alone, and inventing one would be the wrong answer
    /// silently.
    #[test]
    fn a_capped_variable_format_is_not_sharded_from_a_guess() {
        let c = cfg(384, VecFormat::Fvec, Some(1_000_000_000_000));
        assert!(c.shard_plan(VecFormat::Fvvec).is_none());
    }

    /// **`open_sink` shards on its own** when the config says to, so a
    /// caller writes records the same way either way.
    #[test]
    fn open_sink_returns_a_series_when_the_cap_calls_for_one() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.fvec");
        // dim 4 → 20 bytes a record; a 200-byte cap gives a stride
        // of 10.
        let c = cfg(4, VecFormat::Fvec, Some(200));
        assert_eq!(c.shard_plan(VecFormat::Fvec).unwrap().stride, 10);

        let mut sink = open_sink(&path, VecFormat::Fvec, &c).unwrap();
        for o in 0..25i64 {
            let data: Vec<u8> = (0..4).flat_map(|d| (o as f32 + d as f32).to_le_bytes()).collect();
            sink.write_record(o, &data);
        }
        sink.finish().unwrap();

        assert!(tmp.path().join("base__0000.fvec").exists());
        assert!(tmp.path().join("base__0002.fvec").exists());
        assert!(!path.exists(), "a series does not also write the plain name");
    }

    /// The same call with no cap writes exactly one file — so adding
    /// the cap is what changes behaviour, not the code path.
    #[test]
    fn open_sink_without_a_cap_writes_one_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.fvec");
        let mut sink = open_sink(&path, VecFormat::Fvec, &cfg(4, VecFormat::Fvec, None)).unwrap();
        for o in 0..25i64 {
            let data: Vec<u8> = (0..4).flat_map(|d| (o as f32 + d as f32).to_le_bytes()).collect();
            sink.write_record(o, &data);
        }
        sink.finish().unwrap();

        assert!(path.exists());
        assert!(!tmp.path().join("base__0000.fvec").exists());
    }

    /// A shard does not shard itself: the per-shard config drops the
    /// cap, or each shard would try to split again at the same stride.
    #[test]
    fn a_shard_does_not_shard_itself() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("base.fvec");
        let c = cfg(4, VecFormat::Fvec, Some(200));
        let mut sink = open_sink(&path, VecFormat::Fvec, &c).unwrap();
        for o in 0..25i64 {
            sink.write_record(o, &[0u8; 16]);
        }
        sink.finish().unwrap();

        // Exactly three shards, and no nested names beneath them.
        let names: Vec<String> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names.len(), 3, "{names:?}");
        assert!(!names.iter().any(|n| n.matches("__").count() > 1), "{names:?}");
    }
}
