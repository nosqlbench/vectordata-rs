// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Ordinal mapping for multi-file facets.
//!
//! A facet may be one file or a series of them. Either way its records
//! form one dense, gapless ordinal space, and this module is what turns
//! a global ordinal in that space into the file that holds it and the
//! offset within that file.
//!
//! Three coordinate levels, never conflated (SRD SH-64):
//!
//! ```text
//! global ordinal  o
//!    ↓  [OrdinalMap::locate]          — which shard, how far into it
//! local ordinal   l   within shard s
//!    ↓  + entries[s].file_base        — the entry window's lower bound
//! file ordinal    f   within the file s is drawn from
//!    ↓  the format's record→byte rule — elsewhere
//! byte offset
//! ```
//!
//! **Shards and files are counted separately** (SH-81). A shard is a
//! contiguous run of ordinals; a file is where bytes live. Two shards
//! may be drawn from one file at different windows, in which case they
//! are two shards and one file.
//!
//! See `docs/design/srd-multifile-facet-shards.md` for the requirements
//! this implements.

// The first landed piece of multi-file facet support: the readers,
// prefetch planner, and serde realization that consume this arrive in
// the following steps. Remove this attribute once they do — it exists to
// keep the workspace lint-clean across a staged landing, not to excuse a
// permanently unused module.
#![allow(dead_code)]

use crate::dataset::source::DSSource;

/// How a global ordinal becomes a shard index and a local ordinal.
///
/// The two arms are resolved **once**, when the declaration is realized,
/// and are thereafter a single dispatch — never a general path with a
/// fast case tested for on every lookup (SH-54).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OrdinalMap {
    /// Every shard but the last holds exactly `stride` ordinals, so the
    /// lookup is division and remainder: **O(1)**, no allocation, no
    /// search (SH-11).
    Uniform {
        /// Ordinals per shard, for every shard but the last.
        stride: u64,
        /// Number of shards.
        count: u32,
        /// Total ordinals across the series.
        total: u64,
    },
    /// Shard lengths are uneven, so the lookup is a binary search over
    /// prefix sums (SH-55). `starts` has `count + 1` entries: it opens
    /// at `0`, closes at `total`, and is strictly increasing.
    Explicit {
        /// Prefix sums of the shard lengths.
        starts: Vec<u64>,
        /// Total ordinals across the series — equals `starts[count]`.
        total: u64,
    },
}

impl OrdinalMap {
    /// Build the map from per-shard lengths, collapsing to [`Self::Uniform`]
    /// whenever the lengths permit it.
    ///
    /// **Uniformity is a property of the lengths, not of how the series
    /// was spelled** (SH-68). A list of evenly-sized files — what an
    /// importer routinely produces — gets the O(1) map, because
    /// penalizing a regular series for having been written as a list
    /// would be exactly backwards.
    ///
    /// Returns `None` if any length is zero (SH-56): a zero-length shard
    /// contributes no ordinals and would put two shards at one prefix-sum
    /// boundary. An empty facet is not a series at all (SH-5).
    pub(crate) fn from_lengths(lens: &[u64]) -> Option<Self> {
        if lens.is_empty() || lens.contains(&0) {
            return None;
        }
        let total: u64 = lens.iter().sum();
        let stride = lens[0];
        // Uniform when every shard but the last is `stride`, and the last
        // does not exceed it.
        let uniform = lens[..lens.len() - 1].iter().all(|&l| l == stride)
            && *lens.last().expect("non-empty") <= stride;
        if uniform {
            return Some(Self::Uniform {
                stride,
                count: lens.len() as u32,
                total,
            });
        }
        let mut starts = Vec::with_capacity(lens.len() + 1);
        let mut at = 0u64;
        starts.push(0);
        for &l in lens {
            at += l;
            starts.push(at);
        }
        Some(Self::Explicit { starts, total })
    }

    /// Total ordinals across the series.
    pub(crate) fn total(&self) -> u64 {
        match self {
            Self::Uniform { total, .. } | Self::Explicit { total, .. } => *total,
        }
    }

    /// Number of shards.
    pub(crate) fn shard_count(&self) -> usize {
        match self {
            Self::Uniform { count, .. } => *count as usize,
            Self::Explicit { starts, .. } => starts.len() - 1,
        }
    }

    /// First global ordinal of shard `s`, or `None` if out of range.
    pub(crate) fn shard_base(&self, s: usize) -> Option<u64> {
        if s >= self.shard_count() {
            return None;
        }
        Some(match self {
            Self::Uniform { stride, .. } => s as u64 * stride,
            Self::Explicit { starts, .. } => starts[s],
        })
    }

    /// Ordinals held by shard `s`, or `None` if out of range.
    pub(crate) fn shard_len(&self, s: usize) -> Option<u64> {
        let base = self.shard_base(s)?;
        let end = match self {
            Self::Uniform { stride, total, .. } => (base + stride).min(*total),
            Self::Explicit { starts, .. } => starts[s + 1],
        };
        Some(end - base)
    }

    /// Map a global ordinal to `(shard, local ordinal)`.
    ///
    /// `None` when `o` is past the end of the series — never a clamp,
    /// because a silently clamped ordinal reads the wrong record.
    pub(crate) fn locate(&self, o: u64) -> Option<(usize, u64)> {
        if o >= self.total() {
            return None;
        }
        Some(match self {
            // The whole point: two integer operations (SH-11).
            Self::Uniform { stride, .. } => ((o / stride) as usize, o % stride),
            Self::Explicit { starts, .. } => {
                // Last index whose start is <= o. `starts` opens at 0 and
                // o < total, so this never underflows.
                let s = starts.partition_point(|&x| x <= o) - 1;
                (s, o - starts[s])
            }
        })
    }
}

/// One shard's binding to the file it is drawn from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Entry {
    /// The file, plus any namespace — the window has already been
    /// folded into `file_base` and `len`.
    pub(crate) source: DSSource,
    /// First **file** ordinal this shard reads. Zero for a whole-file
    /// shard; the entry window's lower bound for a sliced one.
    pub(crate) file_base: u64,
    /// Ordinals this shard holds.
    pub(crate) len: u64,
}

/// A facet's shards: the ordinal map, and what each shard reads from.
///
/// Ordinal mapping is deliberately separate from source resolution
/// (SH-69). The map answers "which shard, and how far into it"; the
/// entries answer "which file, and how far into that". Both are indexed
/// by shard number, so either map arm serves arbitrary per-shard
/// sources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Shards {
    map: OrdinalMap,
    entries: Vec<Entry>,
}

/// Where a global ordinal lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Located {
    /// Index of the shard holding it.
    pub(crate) shard: usize,
    /// Its offset within that shard.
    pub(crate) local: u64,
    /// Its ordinal within the file the shard is drawn from — `local`
    /// plus the entry's `file_base`.
    pub(crate) file_ordinal: u64,
}

/// One shard's slice of a decomposed window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SubWindow {
    /// Index of the shard this covers.
    pub(crate) shard: usize,
    /// Half-open bounds in the shard's local ordinals.
    pub(crate) local: (u64, u64),
    /// The same bounds in the file's ordinals, which is what a format's
    /// record→byte rule wants.
    pub(crate) file: (u64, u64),
}

impl Shards {
    /// Build from per-shard entries, deriving the map from their lengths.
    ///
    /// `None` when the entries are empty or any is zero-length (SH-56).
    pub(crate) fn new(entries: Vec<Entry>) -> Option<Self> {
        let lens: Vec<u64> = entries.iter().map(|e| e.len).collect();
        let map = OrdinalMap::from_lengths(&lens)?;
        Some(Self { map, entries })
    }

    /// The ordinal map.
    pub(crate) fn map(&self) -> &OrdinalMap {
        &self.map
    }

    /// The shard entries, indexed by shard number.
    pub(crate) fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Total records across the series.
    pub(crate) fn count(&self) -> u64 {
        self.map.total()
    }

    /// Whether this facet is a single whole file — the canonical shape
    /// for everything written before sharding existed (SH-4).
    pub(crate) fn is_single_file(&self) -> bool {
        self.entries.len() == 1 && self.entries[0].file_base == 0
    }

    /// Resolve a global ordinal through all three coordinate levels.
    pub(crate) fn locate(&self, o: u64) -> Option<Located> {
        let (shard, local) = self.map.locate(o)?;
        Some(Located {
            shard,
            local,
            file_ordinal: self.entries[shard].file_base + local,
        })
    }

    /// Decompose the window `[lo, hi)` into per-shard sub-windows
    /// (SH-14).
    ///
    /// Returns an empty vector for an empty or out-of-range window.
    /// Never emits an empty sub-window, so a window ending exactly on a
    /// shard boundary does not produce a trailing no-op.
    pub(crate) fn decompose(&self, lo: u64, hi: u64) -> Vec<SubWindow> {
        let hi = hi.min(self.map.total());
        if lo >= hi {
            return Vec::new();
        }
        let (first, _) = self.map.locate(lo).expect("lo < hi <= total");
        let (last, _) = self.map.locate(hi - 1).expect("hi - 1 < total");
        (first..=last)
            .map(|s| {
                let base = self.map.shard_base(s).expect("s <= last");
                let len = self.map.shard_len(s).expect("s <= last");
                let local_lo = lo.max(base) - base;
                let local_hi = hi.min(base + len) - base;
                let fb = self.entries[s].file_base;
                SubWindow {
                    shard: s,
                    local: (local_lo, local_hi),
                    file: (fb + local_lo, fb + local_hi),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(path: &str) -> DSSource {
        crate::dataset::source::parse_source_string(path).unwrap()
    }

    /// Whole-file shards of the given lengths.
    fn whole(lens: &[u64]) -> Shards {
        Shards::new(
            lens.iter()
                .enumerate()
                .map(|(i, &len)| Entry {
                    source: src(&format!("part_{i}.u8")),
                    file_base: 0,
                    len,
                })
                .collect(),
        )
        .unwrap()
    }

    // ── the map ────────────────────────────────────────────────────

    /// **Uniform lengths take the O(1) arm** — division and remainder,
    /// no prefix-sum table to allocate and nothing to search (SH-11,
    /// SH-54). Asserted on the variant itself, because "it is fast" is
    /// not a property a timing test can pin.
    #[test]
    fn uniform_lengths_map_without_allocating_or_searching() {
        let m = OrdinalMap::from_lengths(&[100, 100, 100]).unwrap();
        assert_eq!(
            m,
            OrdinalMap::Uniform {
                stride: 100,
                count: 3,
                total: 300
            },
            "an even series must not carry a prefix-sum table"
        );
    }

    /// A short final shard is still uniform: the stride governs every
    /// shard but the last (SH-12).
    #[test]
    fn a_shorter_last_shard_is_still_uniform() {
        let m = OrdinalMap::from_lengths(&[100, 100, 37]).unwrap();
        assert_eq!(
            m,
            OrdinalMap::Uniform {
                stride: 100,
                count: 3,
                total: 237
            }
        );
    }

    /// **Uniformity is a property of the lengths, not of the spelling**
    /// (SH-68). An explicitly-listed series of even files is exactly as
    /// uniform as a generated one, and must not be penalized for how it
    /// was declared.
    #[test]
    fn an_explicitly_listed_even_series_still_collapses_to_o1() {
        let listed = whole(&[64, 64, 64, 64]);
        assert!(
            matches!(listed.map(), OrdinalMap::Uniform { .. }),
            "a list of evenly-sized files is what an importer produces; \
             it must get the O(1) map, not binary search"
        );
    }

    /// Genuinely uneven lengths keep the prefix-sum form.
    #[test]
    fn uneven_lengths_keep_prefix_sums() {
        let m = OrdinalMap::from_lengths(&[10, 25, 7]).unwrap();
        assert_eq!(
            m,
            OrdinalMap::Explicit {
                starts: vec![0, 10, 35, 42],
                total: 42
            },
            "starts open at 0, close at total, and are strictly increasing"
        );
    }

    /// A last shard *longer* than the stride is not a uniform series —
    /// it is uneven, and gets the map that can describe it.
    #[test]
    fn an_overlong_last_shard_is_not_uniform() {
        let m = OrdinalMap::from_lengths(&[10, 10, 15]).unwrap();
        assert!(matches!(m, OrdinalMap::Explicit { .. }));
    }

    /// Zero-length shards are refused (SH-56): they contribute no
    /// ordinals and would put two shards at one prefix-sum boundary.
    #[test]
    fn a_zero_length_shard_is_refused() {
        assert!(OrdinalMap::from_lengths(&[10, 0, 10]).is_none());
        assert!(OrdinalMap::from_lengths(&[0]).is_none());
        assert!(OrdinalMap::from_lengths(&[]).is_none());
    }

    /// **The two arms answer identically.** This is the property that
    /// lets the collapse in SH-68 be an optimization rather than a
    /// behaviour change.
    #[test]
    fn both_map_arms_agree_on_every_ordinal() {
        let lens = [7u64, 7, 7, 4];
        let uniform = OrdinalMap::from_lengths(&lens).unwrap();
        assert!(matches!(uniform, OrdinalMap::Uniform { .. }));

        // The same lengths, forced onto the prefix-sum arm.
        let mut starts = vec![0u64];
        for &l in &lens {
            starts.push(starts.last().unwrap() + l);
        }
        let explicit = OrdinalMap::Explicit {
            starts,
            total: lens.iter().sum(),
        };

        for o in 0..uniform.total() {
            assert_eq!(
                uniform.locate(o),
                explicit.locate(o),
                "arms disagree at ordinal {o}"
            );
        }
        assert_eq!(uniform.locate(uniform.total()), None);
        assert_eq!(explicit.locate(explicit.total()), None);
    }

    /// Past the end is `None`, never a clamp — a clamped ordinal reads
    /// the wrong record and reports success.
    #[test]
    fn an_ordinal_past_the_end_does_not_clamp() {
        let m = OrdinalMap::from_lengths(&[10, 10]).unwrap();
        assert_eq!(m.locate(19), Some((1, 9)));
        assert_eq!(m.locate(20), None);
        assert_eq!(m.locate(u64::MAX), None);
    }

    /// Every boundary lands on the shard that owns it, both arms.
    #[test]
    fn boundaries_belong_to_the_shard_that_starts_there() {
        for lens in [vec![5u64, 5, 5], vec![5, 9, 2]] {
            let m = OrdinalMap::from_lengths(&lens).unwrap();
            let mut base = 0u64;
            for (s, &len) in lens.iter().enumerate() {
                assert_eq!(m.locate(base), Some((s, 0)), "start of shard {s}");
                assert_eq!(
                    m.locate(base + len - 1),
                    Some((s, len - 1)),
                    "end of shard {s}"
                );
                assert_eq!(m.shard_base(s), Some(base));
                assert_eq!(m.shard_len(s), Some(len));
                base += len;
            }
            assert_eq!(m.shard_base(lens.len()), None);
        }
    }

    // ── three coordinate levels ────────────────────────────────────

    /// **A sliced shard's file ordinal is offset by its window** (SH-64).
    /// The local ordinal is where the record sits in the shard; the file
    /// ordinal is where it sits in the file, and they differ by exactly
    /// the entry's lower bound.
    #[test]
    fn a_sliced_shard_offsets_into_its_file() {
        let shards = Shards::new(vec![
            Entry {
                source: src("a.u8"),
                file_base: 0,
                len: 10,
            },
            Entry {
                source: src("b.u8"),
                file_base: 500,
                len: 10,
            },
        ])
        .unwrap();

        assert_eq!(
            shards.locate(0),
            Some(Located {
                shard: 0,
                local: 0,
                file_ordinal: 0
            })
        );
        // First record of the sliced shard: local 0, but file ordinal 500.
        assert_eq!(
            shards.locate(10),
            Some(Located {
                shard: 1,
                local: 0,
                file_ordinal: 500
            })
        );
        assert_eq!(
            shards.locate(19),
            Some(Located {
                shard: 1,
                local: 9,
                file_ordinal: 509
            })
        );
        assert_eq!(shards.locate(20), None);
    }

    /// One file may back two shards at disjoint windows (SH-66) — the
    /// ordinal space is dense across them regardless.
    #[test]
    fn one_file_may_back_two_shards() {
        let shards = Shards::new(vec![
            Entry {
                source: src("corpus.u8"),
                file_base: 0,
                len: 100,
            },
            Entry {
                source: src("corpus.u8"),
                file_base: 900,
                len: 100,
            },
        ])
        .unwrap();
        assert_eq!(shards.count(), 200);
        assert_eq!(shards.locate(0).unwrap().file_ordinal, 0);
        assert_eq!(shards.locate(100).unwrap().file_ordinal, 900);
        assert_eq!(shards.locate(199).unwrap().file_ordinal, 999);
        assert_eq!(
            shards.entries()[0].source,
            shards.entries()[1].source,
            "both shards name one file"
        );
    }

    /// A single whole file is recognisable as such — the canonical shape
    /// for every dataset written before sharding existed (SH-4, SH-73).
    #[test]
    fn a_single_whole_file_is_recognised() {
        assert!(whole(&[1000]).is_single_file());
        assert!(!whole(&[500, 500]).is_single_file());
        let sliced = Shards::new(vec![Entry {
            source: src("a.u8"),
            file_base: 10,
            len: 90,
        }])
        .unwrap();
        assert!(
            !sliced.is_single_file(),
            "one entry, but a window into a file is not the whole file"
        );
    }

    // ── window decomposition ───────────────────────────────────────

    /// A window inside one shard touches only that shard.
    #[test]
    fn a_window_inside_one_shard_touches_only_it() {
        let s = whole(&[100, 100, 100]);
        assert_eq!(
            s.decompose(110, 150),
            vec![SubWindow {
                shard: 1,
                local: (10, 50),
                file: (10, 50)
            }]
        );
    }

    /// **A window across a seam splits with no gap and no overlap**
    /// (SH-14) — the case the whole design exists to get right.
    #[test]
    fn a_window_across_a_seam_splits_exactly() {
        let s = whole(&[100, 100, 100]);
        assert_eq!(
            s.decompose(80, 220),
            vec![
                SubWindow {
                    shard: 0,
                    local: (80, 100),
                    file: (80, 100)
                },
                SubWindow {
                    shard: 1,
                    local: (0, 100),
                    file: (0, 100)
                },
                SubWindow {
                    shard: 2,
                    local: (0, 20),
                    file: (0, 20)
                },
            ]
        );
        let covered: u64 = s
            .decompose(80, 220)
            .iter()
            .map(|w| w.local.1 - w.local.0)
            .sum();
        assert_eq!(covered, 140, "the parts must sum to the whole window");
    }

    /// A window ending exactly on a boundary emits no empty trailing
    /// sub-window.
    #[test]
    fn a_boundary_aligned_window_has_no_empty_tail() {
        let s = whole(&[100, 100, 100]);
        let parts = s.decompose(0, 200);
        assert_eq!(parts.len(), 2, "got {parts:?}");
        assert!(parts.iter().all(|w| w.local.1 > w.local.0));
    }

    /// Decomposition works the same over an uneven series.
    #[test]
    fn an_uneven_series_decomposes_at_its_own_seams() {
        let s = whole(&[10, 25, 7]);
        assert_eq!(
            s.decompose(5, 40),
            vec![
                SubWindow {
                    shard: 0,
                    local: (5, 10),
                    file: (5, 10)
                },
                SubWindow {
                    shard: 1,
                    local: (0, 25),
                    file: (0, 25)
                },
                SubWindow {
                    shard: 2,
                    local: (0, 5),
                    file: (0, 5)
                },
            ]
        );
    }

    /// Sub-window file bounds carry the entry's offset, so a caller can
    /// hand them straight to a format's record→byte rule.
    #[test]
    fn sub_windows_report_file_bounds_not_just_local_ones() {
        let s = Shards::new(vec![
            Entry {
                source: src("a.u8"),
                file_base: 1000,
                len: 50,
            },
            Entry {
                source: src("b.u8"),
                file_base: 7,
                len: 50,
            },
        ])
        .unwrap();
        assert_eq!(
            s.decompose(40, 60),
            vec![
                SubWindow {
                    shard: 0,
                    local: (40, 50),
                    file: (1040, 1050)
                },
                SubWindow {
                    shard: 1,
                    local: (0, 10),
                    file: (7, 17)
                },
            ]
        );
    }

    /// The full range covers every shard completely and nothing more.
    #[test]
    fn the_full_range_covers_every_shard_exactly_once() {
        let s = whole(&[10, 25, 7]);
        let parts = s.decompose(0, s.count());
        assert_eq!(parts.len(), 3);
        for (i, w) in parts.iter().enumerate() {
            assert_eq!(w.shard, i);
            assert_eq!(w.local, (0, s.map().shard_len(i).unwrap()));
        }
    }

    /// An empty or reversed window decomposes to nothing, and a window
    /// running past the end is clipped rather than refused — the clip is
    /// safe here because it removes ordinals rather than inventing them.
    #[test]
    fn empty_and_overlong_windows_are_handled() {
        let s = whole(&[10, 10]);
        assert!(s.decompose(5, 5).is_empty());
        assert!(s.decompose(9, 3).is_empty());
        assert!(s.decompose(25, 30).is_empty());
        assert_eq!(
            s.decompose(15, 999),
            vec![SubWindow {
                shard: 1,
                local: (5, 10),
                file: (5, 10)
            }]
        );
    }

    /// **A single file behaves identically to a series of the same
    /// records** — the anchor property (SH-48). Every ordinal resolves
    /// to the same file ordinal, and every window covers the same
    /// records, however the facet was laid out.
    #[test]
    fn one_file_and_a_series_of_the_same_records_agree() {
        let single = whole(&[300]);
        let series = whole(&[100, 100, 100]);
        assert_eq!(single.count(), series.count());

        for o in 0..300 {
            let a = single.locate(o).unwrap();
            let b = series.locate(o).unwrap();
            // Different shards, but the same record: shard 0 of the
            // single-file layout at file ordinal `o`, and shard `o/100`
            // of the series at file ordinal `o % 100`.
            assert_eq!(a.file_ordinal, o);
            assert_eq!(b.file_ordinal, o % 100);
            assert_eq!(b.shard, (o / 100) as usize);
        }

        // A window covers the same count of records either way.
        let n = |w: &[SubWindow]| -> u64 { w.iter().map(|x| x.local.1 - x.local.0).sum() };
        for (lo, hi) in [(0, 300), (0, 1), (150, 151), (99, 202), (299, 300)] {
            assert_eq!(
                n(&single.decompose(lo, hi)),
                n(&series.decompose(lo, hi)),
                "window [{lo}..{hi}) differs between layouts"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Realization — declaration to model
// ═══════════════════════════════════════════════════════════════════

/// A fault in a facet's shard declaration.
///
/// Declaration faults are raised where they can still be fixed: at load,
/// naming the facet and the entry (SH-94). Every variant carries the
/// shard index or the source it concerns, because a message that says
/// *which* is the difference between a fixable report and a puzzle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ShardError {
    /// `NNNN` without `shard_stride`/`shard_count`, or the reverse
    /// (SH-47).
    DeclarationIncomplete { facet: String, detail: String },
    /// An array `source` alongside `shard_stride`/`shard_count` — the
    /// array already says everything those fields would (SH-57).
    MixedDeclaration { facet: String },
    /// An entry's `=<count>` disagrees with the length it annotates
    /// (SH-62).
    SliceCountMismatch {
        facet: String,
        index: usize,
        declared: u64,
        implied: u64,
    },
    /// The declared total disagrees with what the shards hold (SH-8).
    RecordCountMismatch {
        facet: String,
        declared: u64,
        derived: u64,
    },
    /// An entry resolves to zero length (SH-56).
    EmptyEntry { facet: String, index: usize },
    /// An entry carries more than one interval — equivalent to listing
    /// the file once per interval, and one spelling is better than two
    /// (SH-65).
    MultiIntervalEntry { facet: String, index: usize },
    /// A remote entry with neither a window nor a count, whose length
    /// could only be learned by fetching (SH-63).
    UnboundedRemoteEntry {
        facet: String,
        index: usize,
        source: String,
    },
    /// An entry's source string is malformed.
    MalformedEntry {
        facet: String,
        index: usize,
        detail: String,
    },
    /// A length could not be established for an entry.
    LengthUnavailable {
        facet: String,
        index: usize,
        detail: String,
    },
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeclarationIncomplete { facet, detail } => {
                write!(
                    f,
                    "facet '{facet}': incomplete shard declaration — {detail}"
                )
            }
            Self::MixedDeclaration { facet } => write!(
                f,
                "facet '{facet}': an array source cannot also carry shard_stride \
                 or shard_count — the array already states the layout"
            ),
            Self::SliceCountMismatch {
                facet,
                index,
                declared,
                implied,
            } => write!(
                f,
                "facet '{facet}' shard {index}: declared count {declared} does not \
                 match its length {implied}"
            ),
            Self::RecordCountMismatch {
                facet,
                declared,
                derived,
            } => write!(
                f,
                "facet '{facet}': declared record_count {declared} does not match \
                 the {derived} records its shards hold"
            ),
            Self::EmptyEntry { facet, index } => {
                write!(f, "facet '{facet}' shard {index}: resolves to zero records")
            }
            Self::MultiIntervalEntry { facet, index } => write!(
                f,
                "facet '{facet}' shard {index}: an entry carries at most one \
                 interval — list the file once per interval instead"
            ),
            Self::UnboundedRemoteEntry {
                facet,
                index,
                source,
            } => write!(
                f,
                "facet '{facet}' shard {index}: remote entry '{source}' states no \
                 window or count, and its length cannot be learned without fetching"
            ),
            Self::MalformedEntry {
                facet,
                index,
                detail,
            } => write!(f, "facet '{facet}' shard {index}: {detail}"),
            Self::LengthUnavailable {
                facet,
                index,
                detail,
            } => write!(
                f,
                "facet '{facet}' shard {index}: cannot establish record count — {detail}"
            ),
        }
    }
}

impl std::error::Error for ShardError {}

/// Answers how many records a file holds.
///
/// Consulted **only** for an entry that declares neither a window nor a
/// count — the local-convenience spelling (SH-63). Every other entry is
/// self-describing and never reaches this.
pub(crate) type Cardinality<'a> = &'a dyn Fn(&DSSource) -> Result<u64, String>;

/// The literal token marking the shard-index field in a uniform source.
///
/// Exactly four `N`s: the width is fixed (SH-2), so there is no `NNN` or
/// `NNNNN` form to accept.
pub(crate) const SHARD_FIELD: &str = "NNNN";

/// Whether a source string declares a uniform series.
pub(crate) fn has_shard_field(source: &str) -> bool {
    source.contains(SHARD_FIELD)
}

/// Substitute the shard index into a uniform source's `NNNN` field.
pub(crate) fn shard_filename(source: &str, index: u32) -> String {
    source.replacen(SHARD_FIELD, &format!("{index:04}"), 1)
}

/// The all-digit token immediately before the shard field, if there is
/// one (SH-101).
///
/// The shard field is always last before the extension and always four
/// digits, so `p__0010__NNNN.ivecs` derives `p__0010__0000.ivecs` —
/// which reads equally well as shard 10 of `p` with a stray suffix, or
/// shard 0 of `p__0010`. Nothing downstream parses these names, but a
/// human and a directory listing both do, and a generator
/// interpolating a numeric profile name into a basename produces
/// exactly this.
fn ambiguous_token_before_shard_field(pattern: &str) -> Option<&str> {
    let path = pattern.split(':').next().unwrap_or(pattern);
    let stem = match path.rfind('.') {
        Some(dot) => &path[..dot],
        None => path,
    };
    let before = stem.strip_suffix(SHARD_FIELD)?.strip_suffix("__")?;
    let token = before.rsplit("__").next()?;
    (!token.is_empty() && token.bytes().all(|b| b.is_ascii_digit())).then_some(token)
}

/// What a facet's declaration says about its layout, independent of how
/// it was spelled.
///
/// This is the shape [`realize`] consumes. Extracting it keeps the
/// realization logic testable without constructing a `FacetConfig`, and
/// keeps the serde types out of the ordinal model.
#[derive(Debug, Clone, Default)]
pub(crate) struct Declaration<'a> {
    /// Source strings in ordinal order: one for a single file or a
    /// uniform series, several for an explicit one.
    pub(crate) sources: &'a [String],
    /// Whether `sources` came from an array (SH-50).
    pub(crate) is_array: bool,
    /// Ordinals per shard, for a uniform series.
    pub(crate) shard_stride: Option<u64>,
    /// Shard count, for a uniform series.
    pub(crate) shard_count: Option<u32>,
    /// Declared total records.
    pub(crate) record_count: Option<u64>,
}

/// Whether a source string names a remote resource.
fn is_remote(path: &str) -> bool {
    crate::transport::is_remote_url(path)
}

/// Realize a declaration into the ordinal model (SH-85).
///
/// This is where declaration shape stops mattering. Whatever form the
/// facet was written in — single file, uniform series, explicit series,
/// pinned or bare — the result is one [`Shards`], and no stage above
/// this branches on the spelling again (SH-86).
///
/// A **non-canonical single-shard declaration is accepted** and realized
/// as the single-file facet it describes (SH-72). Reporting it is the
/// validator's job, not the loader's; see [`canonical_violations`].
pub(crate) fn realize(
    facet: &str,
    decl: &Declaration<'_>,
    cardinality: Cardinality<'_>,
) -> Result<Shards, ShardError> {
    let uniform = decl.sources.first().is_some_and(|s| has_shard_field(s));

    if decl.is_array && (decl.shard_stride.is_some() || decl.shard_count.is_some()) {
        return Err(ShardError::MixedDeclaration {
            facet: facet.to_string(),
        });
    }
    if uniform {
        return realize_uniform(facet, decl);
    }
    if decl.shard_stride.is_some() || decl.shard_count.is_some() {
        return Err(ShardError::DeclarationIncomplete {
            facet: facet.to_string(),
            detail: format!(
                "shard_stride/shard_count without a '{SHARD_FIELD}' field in the source"
            ),
        });
    }
    realize_entries(facet, decl, cardinality)
}

/// The uniform form: filenames derived from `NNNN`, lengths from stride
/// and the declared total.
fn realize_uniform(facet: &str, decl: &Declaration<'_>) -> Result<Shards, ShardError> {
    let incomplete = |detail: &str| ShardError::DeclarationIncomplete {
        facet: facet.to_string(),
        detail: detail.to_string(),
    };
    if decl.is_array {
        return Err(incomplete(
            "a '{SHARD_FIELD}' field belongs to a single source string, not an array",
        ));
    }
    let pattern = &decl.sources[0];
    if let Some(ambiguous) = ambiguous_token_before_shard_field(pattern) {
        return Err(incomplete(&format!(
            "the token '{ambiguous}' before the '{SHARD_FIELD}' field is all digits, \
             so the derived filenames have two readings and neither is decidable \
             (SH-101); give it a non-numeric prefix"
        )));
    }
    let stride = decl
        .shard_stride
        .ok_or_else(|| incomplete(&format!("'{SHARD_FIELD}' without shard_stride")))?;
    let count = decl
        .shard_count
        .ok_or_else(|| incomplete(&format!("'{SHARD_FIELD}' without shard_count")))?;
    let total = decl
        .record_count
        .ok_or_else(|| incomplete("a sharded facet must declare record_count"))?;
    if stride == 0 {
        return Err(incomplete("shard_stride must be greater than zero"));
    }
    if count == 0 {
        return Err(incomplete("shard_count must be greater than zero"));
    }

    // Lengths follow from the stride and the declared total: every shard
    // but the last is `stride`, and the last is whatever remains. A
    // remainder outside `1..=stride` means the declaration disagrees
    // with itself.
    let full = (count as u64 - 1) * stride;
    if total <= full || total > full + stride {
        return Err(ShardError::RecordCountMismatch {
            facet: facet.to_string(),
            declared: total,
            derived: full + stride,
        });
    }
    let mut lens: Vec<u64> = vec![stride; count as usize - 1];
    lens.push(total - full);

    let entries = lens
        .iter()
        .enumerate()
        .map(|(i, &len)| {
            let name = shard_filename(pattern, i as u32);
            crate::dataset::source::parse_source_string(&name)
                .map(|source| Entry {
                    source,
                    file_base: 0,
                    len,
                })
                .map_err(|e| ShardError::MalformedEntry {
                    facet: facet.to_string(),
                    index: i,
                    detail: e,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Shards::new(entries).ok_or_else(|| ShardError::EmptyEntry {
        facet: facet.to_string(),
        index: 0,
    })
}

/// The single-file and explicit forms: one entry per listed source.
fn realize_entries(
    facet: &str,
    decl: &Declaration<'_>,
    cardinality: Cardinality<'_>,
) -> Result<Shards, ShardError> {
    let mut entries = Vec::with_capacity(decl.sources.len());
    for (i, raw) in decl.sources.iter().enumerate() {
        let parsed = crate::dataset::source::parse_source_string(raw).map_err(|e| {
            ShardError::MalformedEntry {
                facet: facet.to_string(),
                index: i,
                detail: e,
            }
        })?;
        if parsed.window.0.len() > 1 {
            return Err(ShardError::MultiIntervalEntry {
                facet: facet.to_string(),
                index: i,
            });
        }
        let interval = parsed.window.0.first().cloned();
        let (file_base, len) = match &interval {
            Some(iv) => (iv.min_incl, iv.max_excl.saturating_sub(iv.min_incl)),
            None => match parsed.declared_count {
                Some(n) => (0, n),
                None => {
                    // Nothing self-describing: the length can only come
                    // from the file.
                    //
                    // Refused for a remote *series* (SH-63) — building
                    // the map would open every shard before a single
                    // record is read, which is the expense a declaration
                    // exists to avoid. A single remote file is not that
                    // case: the reader must open it to read anything, so
                    // its count is the same open rather than an extra
                    // one, and requiring a declared count would break
                    // every remote facet ever written.
                    if decl.is_array && is_remote(&parsed.path) {
                        return Err(ShardError::UnboundedRemoteEntry {
                            facet: facet.to_string(),
                            index: i,
                            source: parsed.path.clone(),
                        });
                    }
                    let n = cardinality(&parsed).map_err(|e| ShardError::LengthUnavailable {
                        facet: facet.to_string(),
                        index: i,
                        detail: e,
                    })?;
                    (0, n)
                }
            },
        };
        // The edifying count checks whatever it annotates: the interval's
        // length when there is one, the file's cardinality when there is
        // not (SH-62).
        if let Some(declared) = parsed.declared_count
            && interval.is_some()
            && declared != len
        {
            return Err(ShardError::SliceCountMismatch {
                facet: facet.to_string(),
                index: i,
                declared,
                implied: len,
            });
        }
        if len == 0 {
            return Err(ShardError::EmptyEntry {
                facet: facet.to_string(),
                index: i,
            });
        }
        entries.push(Entry {
            source: parsed,
            file_base,
            len,
        });
    }

    let shards = Shards::new(entries).ok_or_else(|| ShardError::EmptyEntry {
        facet: facet.to_string(),
        index: 0,
    })?;

    // A declared total is checked, never preferred (SH-8). Only a series
    // is required to carry one; a plain single-file facet has always
    // been spelled without.
    if let Some(declared) = decl.record_count
        && declared != shards.count()
    {
        return Err(ShardError::RecordCountMismatch {
            facet: facet.to_string(),
            declared,
            derived: shards.count(),
        });
    }
    if decl.is_array && decl.record_count.is_none() {
        return Err(ShardError::DeclarationIncomplete {
            facet: facet.to_string(),
            detail: "a sharded facet must declare record_count".to_string(),
        });
    }
    Ok(shards)
}

/// Marker detail for a length that only the file can answer.
const UNPROBED: &str = "deferred: needs the file";

/// Check everything about a declaration that can be checked **without
/// touching a file**.
///
/// This is what runs at deserialization: it catches a declaration that
/// disagrees with itself — mixed forms, half-stated uniform fields, a
/// count that contradicts its interval, a total that contradicts its
/// entries — at the earliest moment, before a dataset root even exists
/// to resolve relative paths against.
///
/// A length that only the file can answer is **not** a fault here. It is
/// the local-convenience spelling (SH-63), resolved later by [`realize`]
/// when there is a root to resolve against.
///
/// Deliberately implemented by running [`realize`] with a probe that
/// declines, rather than by restating its rules: two copies of these
/// checks would be two chances to disagree, which is the same hazard
/// SH-90 names for the two loaders.
pub(crate) fn validate_declaration(facet: &str, decl: &Declaration<'_>) -> Result<(), ShardError> {
    let unprobed = |_: &DSSource| Err(UNPROBED.to_string());
    match realize(facet, decl, &unprobed) {
        Ok(_) => Ok(()),
        Err(ShardError::LengthUnavailable { detail, .. }) if detail == UNPROBED => Ok(()),
        Err(e) => Err(e),
    }
}

/// Non-canonical spellings a validator should report but a reader must
/// accept (SH-72).
///
/// Currently one: a sharded declaration describing a single shard, which
/// SH-4 requires be spelled as a plain single file so that every reader
/// predating sharding can still open it.
pub(crate) fn canonical_violations(facet: &str, decl: &Declaration<'_>) -> Vec<String> {
    let mut out = Vec::new();
    let single_shard = if decl.is_array {
        decl.sources.len() == 1
    } else {
        decl.shard_count == Some(1)
    };
    if single_shard {
        out.push(format!(
            "facet '{facet}': a series of one shard must be spelled as a single \
             file, so that readers predating multi-file facets can open it"
        ));
    }
    out
}

#[cfg(test)]
mod realization {
    use super::*;

    /// A probe that answers from a canned table, and fails loudly for
    /// anything else — so a test that unexpectedly reaches the file
    /// system says so instead of quietly succeeding.
    fn probe<'a>(table: &'a [(&'a str, u64)]) -> impl Fn(&DSSource) -> Result<u64, String> + 'a {
        move |s: &DSSource| {
            table
                .iter()
                .find(|(p, _)| *p == s.path)
                .map(|(_, n)| *n)
                .ok_or_else(|| format!("no canned cardinality for {}", s.path))
        }
    }

    fn decl<'a>(sources: &'a [String], is_array: bool) -> Declaration<'a> {
        Declaration {
            sources,
            is_array,
            ..Default::default()
        }
    }

    fn v(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    // ── the canonical single file ──────────────────────────────────

    /// **The overwhelmingly common case still works, unchanged.** A
    /// plain filename with no shard fields realizes to one whole-file
    /// shard whose length comes from the file (SH-4, SH-73).
    #[test]
    fn a_plain_filename_realizes_to_one_whole_file() {
        let srcs = v(&["base.fvec"]);
        let p = probe(&[("base.fvec", 1000)]);
        let s = realize("base_vectors", &decl(&srcs, false), &p).unwrap();
        assert!(s.is_single_file());
        assert_eq!(s.count(), 1000);
        assert_eq!(s.entries()[0].file_base, 0);
        assert!(matches!(s.map(), OrdinalMap::Uniform { count: 1, .. }));
    }

    /// A single file may state its own cardinality, and is then never
    /// probed at all.
    #[test]
    fn a_declared_count_replaces_the_probe() {
        let srcs = v(&["base.fvec=1000"]);
        // A probe that would fail if consulted.
        let never = |_: &DSSource| Err("must not be probed".to_string());
        let s = realize("base_vectors", &decl(&srcs, false), &never).unwrap();
        assert_eq!(s.count(), 1000);
    }

    // ── the uniform form ───────────────────────────────────────────

    /// **Ordinals follow the declared order, not the filenames**
    /// (SH-52).
    ///
    /// An explicit series is a concatenation of what the declaration
    /// lists, in the order it lists it. Sorting the names — or assuming
    /// they sort — would silently reorder the dataset for anyone whose
    /// parts are named by content rather than by position.
    #[test]
    fn ordinals_follow_declared_order_not_alphabetical_order() {
        let sources = v(&["zulu.u8=10", "alpha.u8=10", "mike.u8=10"]);
        let shards = realize(
            "metadata_content",
            &Declaration {
                record_count: Some(30),
                ..decl(&sources, true)
            },
            &probe(&[]),
        )
        .expect("counted entries need no probe");

        let names: Vec<&str> = shards
            .entries()
            .iter()
            .map(|e| e.source.path.as_str())
            .collect();
        assert_eq!(names, vec!["zulu.u8", "alpha.u8", "mike.u8"]);
        assert_eq!(shards.locate(0).unwrap().shard, 0, "ordinal 0 is zulu's");
        assert_eq!(shards.locate(15).unwrap().shard, 1, "ordinal 15 is alpha's");
        assert_eq!(shards.locate(25).unwrap().shard, 2, "ordinal 25 is mike's");
    }

    /// **A bare entry is probed once** (SH-87).
    ///
    /// Resolution happens at load, in one place. A probe per read — or
    /// per shard per read — turns a length lookup into an I/O pattern,
    /// and for a remote entry into a fetch.
    #[test]
    fn a_bare_entry_is_probed_exactly_once_per_shard() {
        use std::cell::RefCell;
        let calls = RefCell::new(Vec::<String>::new());
        let counting = |s: &DSSource| {
            calls.borrow_mut().push(s.path.clone());
            Ok(10u64)
        };
        let sources = v(&["a.u8", "b.u8", "c.u8"]);
        let shards = realize(
            "metadata_content",
            &Declaration {
                record_count: Some(30),
                ..decl(&sources, true)
            },
            &counting,
        )
        .expect("bare entries realize by probing");
        assert_eq!(shards.count(), 30);
        assert_eq!(
            calls.borrow().as_slice(),
            &["a.u8".to_string(), "b.u8".to_string(), "c.u8".to_string()],
            "one probe per shard, in order, and no more"
        );
    }

    /// **A counted entry is not probed at all** (SH-87). The count in
    /// the declaration is the answer; going to the file anyway would
    /// make the edifying suffix cost what it was meant to save.
    #[test]
    fn a_counted_entry_is_never_probed() {
        let sources = v(&["a.u8=10", "b.u8=10"]);
        let shards = realize(
            "metadata_content",
            &Declaration {
                record_count: Some(20),
                ..decl(&sources, true)
            },
            // Any probe call fails this test loudly.
            &probe(&[]),
        )
        .expect("counted entries need no probe");
        assert_eq!(shards.count(), 20);
    }

    /// **An all-digit token before the shard field is refused**
    /// (SH-101).
    ///
    /// `p__0010__NNNN.ivecs` derives `p__0010__0000.ivecs`, which reads
    /// as shard 10 of `p` or shard 0 of `p__0010` and gives no way to
    /// choose. A generator interpolating a numeric profile name into a
    /// basename produces exactly this, so it is refused where the
    /// declaration is realized rather than left to be noticed later.
    #[test]
    fn an_all_digit_token_before_the_shard_field_is_refused() {
        let sources = v(&["p__0010__NNNN.ivecs"]);
        let err = realize(
            "metadata_results",
            &Declaration {
                shard_stride: Some(100),
                shard_count: Some(3),
                record_count: Some(300),
                ..decl(&sources, false)
            },
            &probe(&[]),
        )
        .expect_err("an ambiguous basename must not realize");
        let msg = err.to_string();
        assert!(msg.contains("0010"), "{msg}");
        assert!(msg.contains("two readings"), "{msg}");
    }

    /// A non-numeric token in the same position is fine — the rule is
    /// about digits, not about how many `__`-separated parts a name has.
    #[test]
    fn a_named_token_before_the_shard_field_is_accepted() {
        let sources = v(&["p__profile10__NNNN.ivecs"]);
        let shards = realize(
            "metadata_results",
            &Declaration {
                shard_stride: Some(100),
                shard_count: Some(2),
                record_count: Some(200),
                ..decl(&sources, false)
            },
            &probe(&[]),
        )
        .expect("a non-numeric token is unambiguous");
        assert_eq!(shards.entries()[1].source.path, "p__profile10__0001.ivecs");
    }

    /// **A namespace selector and the shard field do not reach into
    /// each other** (SH-97). The namespace follows the path, the index
    /// sits inside the filename, and each parse leaves the other alone.
    #[test]
    fn a_slab_namespace_survives_shard_derivation() {
        let sources = v(&["metadata_content__NNNN.slab:mnodes"]);
        let shards = realize(
            "metadata_content",
            &Declaration {
                shard_stride: Some(50),
                shard_count: Some(3),
                record_count: Some(140),
                ..decl(&sources, false)
            },
            &probe(&[]),
        )
        .expect("a namespaced pattern realizes");
        for (i, e) in shards.entries().iter().enumerate() {
            assert_eq!(e.source.path, format!("metadata_content__{i:04}.slab"));
            assert_eq!(
                e.source.namespace.as_deref(),
                Some("mnodes"),
                "every shard keeps the selector"
            );
        }
        assert_eq!(shards.entries()[2].len, 40, "the last shard is short");
    }

    /// Filenames come from the `NNNN` field, four digits, and lengths
    /// from stride plus the declared total (SH-2, SH-49).
    #[test]
    fn a_uniform_series_derives_its_filenames_and_lengths() {
        let srcs = v(&["base__NNNN.fvec"]);
        let d = Declaration {
            sources: &srcs,
            is_array: false,
            shard_stride: Some(1000),
            shard_count: Some(3),
            record_count: Some(2500),
        };
        let never = |_: &DSSource| Err("must not be probed".to_string());
        let s = realize("base_vectors", &d, &never).unwrap();

        assert_eq!(s.count(), 2500);
        assert_eq!(s.entries().len(), 3);
        assert_eq!(s.entries()[0].source.path, "base__0000.fvec");
        assert_eq!(s.entries()[1].source.path, "base__0001.fvec");
        assert_eq!(s.entries()[2].source.path, "base__0002.fvec");
        assert_eq!(
            (s.map().shard_len(0), s.map().shard_len(2)),
            (Some(1000), Some(500)),
            "every shard but the last is the stride"
        );
        assert!(matches!(s.map(), OrdinalMap::Uniform { .. }));
    }

    /// `NNNN` without the numbers, and the numbers without `NNNN`, are
    /// both errors — one spelling, no inference (SH-47).
    #[test]
    fn a_half_stated_uniform_declaration_is_refused() {
        let never = |_: &DSSource| Err("unused".to_string());

        let srcs = v(&["base__NNNN.fvec"]);
        let no_numbers = decl(&srcs, false);
        assert!(matches!(
            realize("f", &no_numbers, &never),
            Err(ShardError::DeclarationIncomplete { .. })
        ));

        let plain = v(&["base.fvec"]);
        let no_field = Declaration {
            sources: &plain,
            shard_stride: Some(10),
            shard_count: Some(2),
            record_count: Some(20),
            is_array: false,
        };
        assert!(matches!(
            realize("f", &no_field, &never),
            Err(ShardError::DeclarationIncomplete { .. })
        ));
    }

    /// A declared total that cannot be reached by the stride and count
    /// is a disagreement, not a preference (SH-8).
    #[test]
    fn a_total_the_layout_cannot_produce_is_refused() {
        let srcs = v(&["b__NNNN.fvec"]);
        let never = |_: &DSSource| Err("unused".to_string());
        for total in [1999u64, 3001, 2000] {
            let d = Declaration {
                sources: &srcs,
                is_array: false,
                shard_stride: Some(1000),
                shard_count: Some(3),
                record_count: Some(total),
            };
            // 3 shards of stride 1000 hold 2001..=3000 records.
            let ok = (2001..=3000).contains(&total);
            assert_eq!(
                realize("f", &d, &never).is_ok(),
                ok,
                "total {total} should {}",
                if ok {
                    "be reachable"
                } else {
                    "not be reachable"
                }
            );
        }
    }

    // ── the explicit form ──────────────────────────────────────────

    /// Whole files with stated counts realize without touching disk,
    /// which is what makes the form legal remotely (SH-63).
    #[test]
    fn an_explicit_series_of_counted_files_needs_no_probe() {
        let srcs = v(&["a.u8=100", "b.u8=100", "c.u8=40"]);
        let d = Declaration {
            sources: &srcs,
            is_array: true,
            record_count: Some(240),
            ..Default::default()
        };
        let never = |_: &DSSource| Err("must not be probed".to_string());
        let s = realize("metadata_content", &d, &never).unwrap();
        assert_eq!(s.count(), 240);
        // Even lengths but for a shorter last: still the O(1) arm (SH-68).
        assert!(matches!(s.map(), OrdinalMap::Uniform { stride: 100, .. }));
    }

    /// **Windows slice existing files into an ordinal view** — the same
    /// file may appear twice, at disjoint windows (SH-66).
    #[test]
    fn windowed_entries_compose_an_ordinal_view() {
        let srcs = v(&["corpus.u8[0..100]", "corpus.u8[900..1000]"]);
        let d = Declaration {
            sources: &srcs,
            is_array: true,
            record_count: Some(200),
            ..Default::default()
        };
        let never = |_: &DSSource| Err("must not be probed".to_string());
        let s = realize("metadata_content", &d, &never).unwrap();

        assert_eq!(s.count(), 200);
        assert_eq!(s.entries()[0].file_base, 0);
        assert_eq!(s.entries()[1].file_base, 900);
        assert_eq!(s.locate(0).unwrap().file_ordinal, 0);
        assert_eq!(s.locate(100).unwrap().file_ordinal, 900);
        assert_eq!(
            s.entries()[0].source.path,
            s.entries()[1].source.path,
            "one file, two shards"
        );
    }

    /// Bare names are probed — the local convenience — and the probe is
    /// consulted once per entry.
    #[test]
    fn bare_names_are_resolved_by_probing() {
        let srcs = v(&["a.u8", "b.u8"]);
        let d = Declaration {
            sources: &srcs,
            is_array: true,
            record_count: Some(75),
            ..Default::default()
        };
        let p = probe(&[("a.u8", 50), ("b.u8", 25)]);
        let s = realize("metadata_content", &d, &p).unwrap();
        assert_eq!(s.count(), 75);
        assert_eq!(s.map().shard_len(0), Some(50));
        assert_eq!(s.map().shard_len(1), Some(25));
    }

    /// **A bare remote entry is refused before anything is fetched**
    /// (SH-63): resolving it costs a round trip per shard, which is the
    /// exact expense a declaration exists to avoid.
    #[test]
    fn a_bare_remote_entry_is_refused_without_fetching() {
        let srcs = v(&["https://h/a.u8=10", "https://h/b.u8"]);
        let d = Declaration {
            sources: &srcs,
            is_array: true,
            record_count: Some(20),
            ..Default::default()
        };
        let never = |_: &DSSource| panic!("a refusal must not reach the network");
        match realize("metadata_content", &d, &never) {
            Err(ShardError::UnboundedRemoteEntry { index, .. }) => assert_eq!(index, 1),
            other => panic!("expected UnboundedRemoteEntry, got {other:?}"),
        }
    }

    /// **A plain remote file stays legal bare** — the restriction is on
    /// the series, not on remoteness (SH-63).
    ///
    /// Almost every remote facet in existence is written this way. Its
    /// reader must open the file to read anything, so learning the count
    /// is the same open rather than an extra one, and there is no
    /// per-shard multiplication to avoid.
    #[test]
    fn a_plain_remote_file_needs_no_declared_count() {
        let srcs = v(&["s3://bucket/prefix/base.fvecs"]);
        let p = probe(&[("s3://bucket/prefix/base.fvecs", 4242)]);
        let s = realize("base_vectors", &decl(&srcs, false), &p).unwrap();
        assert_eq!(s.count(), 4242);
        assert!(s.is_single_file());
    }

    /// An `=<count>` that disagrees with its interval is caught
    /// (SH-62) — the typo the suffix exists to find.
    #[test]
    fn a_count_disagreeing_with_its_interval_is_caught() {
        let srcs = v(&["a.u8[0..100]=99"]);
        let d = Declaration {
            sources: &srcs,
            is_array: true,
            record_count: Some(100),
            ..Default::default()
        };
        let never = |_: &DSSource| Err("unused".to_string());
        match realize("f", &d, &never) {
            Err(ShardError::SliceCountMismatch {
                declared, implied, ..
            }) => {
                assert_eq!((declared, implied), (99, 100));
            }
            other => panic!("expected SliceCountMismatch, got {other:?}"),
        }
    }

    /// A declared total that disagrees with the entries is caught, and
    /// neither number silently wins (SH-8).
    #[test]
    fn a_total_disagreeing_with_the_entries_is_caught() {
        let srcs = v(&["a.u8=10", "b.u8=10"]);
        let d = Declaration {
            sources: &srcs,
            is_array: true,
            record_count: Some(21),
            ..Default::default()
        };
        let never = |_: &DSSource| Err("unused".to_string());
        match realize("f", &d, &never) {
            Err(ShardError::RecordCountMismatch {
                declared, derived, ..
            }) => assert_eq!((declared, derived), (21, 20)),
            other => panic!("expected RecordCountMismatch, got {other:?}"),
        }
    }

    /// An array source cannot also carry the uniform form's fields —
    /// the array already states the layout (SH-57).
    #[test]
    fn an_array_with_uniform_fields_is_refused() {
        let srcs = v(&["a.u8=10", "b.u8=10"]);
        let d = Declaration {
            sources: &srcs,
            is_array: true,
            shard_stride: Some(10),
            record_count: Some(20),
            ..Default::default()
        };
        let never = |_: &DSSource| Err("unused".to_string());
        assert!(matches!(
            realize("f", &d, &never),
            Err(ShardError::MixedDeclaration { .. })
        ));
    }

    /// A series must state its total (SH-58); a plain single file need
    /// not, and never has.
    #[test]
    fn a_series_must_state_its_total_but_a_single_file_need_not() {
        let never = |_: &DSSource| Err("unused".to_string());

        let series = v(&["a.u8=10", "b.u8=10"]);
        let no_total = decl(&series, true);
        assert!(matches!(
            realize("f", &no_total, &never),
            Err(ShardError::DeclarationIncomplete { .. })
        ));

        let single = v(&["a.u8=10"]);
        assert!(realize("f", &decl(&single, false), &never).is_ok());
    }

    /// Multi-interval entries are refused — equivalent to listing the
    /// file once per interval, and one spelling is better than two
    /// (SH-65).
    #[test]
    fn a_multi_interval_entry_is_refused() {
        let srcs = v(&["a.u8[0..10, 20..30]"]);
        let d = Declaration {
            sources: &srcs,
            is_array: true,
            record_count: Some(20),
            ..Default::default()
        };
        let never = |_: &DSSource| Err("unused".to_string());
        assert!(matches!(
            realize("f", &d, &never),
            Err(ShardError::MultiIntervalEntry { index: 0, .. })
        ));
    }

    /// Zero-length entries are refused (SH-56).
    #[test]
    fn a_zero_length_entry_is_refused() {
        let srcs = v(&["a.u8=10", "b.u8"]);
        let d = Declaration {
            sources: &srcs,
            is_array: true,
            record_count: Some(10),
            ..Default::default()
        };
        let p = probe(&[("b.u8", 0)]);
        assert!(matches!(
            realize("f", &d, &p),
            Err(ShardError::EmptyEntry { index: 1, .. })
        ));
    }

    // ── the collapse rule ──────────────────────────────────────────

    /// **A reader accepts a one-shard series; a validator reports it**
    /// (SH-72). Rejecting at read would help nobody, and accepting
    /// silently would let the non-canonical spelling spread.
    #[test]
    fn a_one_shard_series_is_accepted_by_the_reader_and_reported_by_the_validator() {
        let srcs = v(&["only.u8=10"]);
        let d = Declaration {
            sources: &srcs,
            is_array: true,
            record_count: Some(10),
            ..Default::default()
        };
        let never = |_: &DSSource| Err("unused".to_string());

        let s = realize("f", &d, &never).expect("readers accept it");
        assert_eq!(s.count(), 10);
        assert!(s.is_single_file());

        let found = canonical_violations("f", &d);
        assert_eq!(found.len(), 1, "validators report it: {found:?}");
        assert!(found[0].contains("single file"));
    }

    /// The same for a uniform declaration of one shard.
    #[test]
    fn a_uniform_declaration_of_one_shard_is_reported_too() {
        let srcs = v(&["b__NNNN.fvec"]);
        let d = Declaration {
            sources: &srcs,
            is_array: false,
            shard_stride: Some(1000),
            shard_count: Some(1),
            record_count: Some(1000),
        };
        assert_eq!(canonical_violations("f", &d).len(), 1);
    }

    /// A canonical declaration draws no complaint — in either form.
    #[test]
    fn canonical_declarations_are_silent() {
        let single = v(&["base.fvec"]);
        assert!(canonical_violations("f", &decl(&single, false)).is_empty());

        let series = v(&["a.u8=10", "b.u8=10"]);
        assert!(canonical_violations("f", &decl(&series, true)).is_empty());
    }

    /// **The anchor: one file, a uniform series, and an explicit series
    /// of the same records are indistinguishable** through the realized
    /// model (SH-48).
    #[test]
    fn the_three_layouts_realize_to_the_same_ordinal_space() {
        let never = |_: &DSSource| Err("unused".to_string());

        let single = v(&["all.u8=250"]);
        let one = realize("f", &decl(&single, false), &never).unwrap();

        let upat = v(&["p__NNNN.u8"]);
        let uni = realize(
            "f",
            &Declaration {
                sources: &upat,
                is_array: false,
                shard_stride: Some(100),
                shard_count: Some(3),
                record_count: Some(250),
            },
            &never,
        )
        .unwrap();

        let elist = v(&["x.u8=60", "y.u8=90", "z.u8=100"]);
        let exp = realize(
            "f",
            &Declaration {
                sources: &elist,
                is_array: true,
                record_count: Some(250),
                ..Default::default()
            },
            &never,
        )
        .unwrap();

        assert_eq!(one.count(), 250);
        assert_eq!(uni.count(), 250);
        assert_eq!(exp.count(), 250);

        // Every ordinal resolves in all three; a window covers the same
        // number of records in all three.
        for o in [0u64, 1, 99, 100, 149, 249] {
            for (name, s) in [("single", &one), ("uniform", &uni), ("explicit", &exp)] {
                assert!(s.locate(o).is_some(), "{name} lost ordinal {o}");
            }
        }
        let covered = |s: &Shards, lo, hi| -> u64 {
            s.decompose(lo, hi)
                .iter()
                .map(|w| w.local.1 - w.local.0)
                .sum()
        };
        for (lo, hi) in [(0u64, 250u64), (50, 200), (99, 101), (249, 250)] {
            assert_eq!(covered(&one, lo, hi), hi - lo);
            assert_eq!(covered(&uni, lo, hi), hi - lo);
            assert_eq!(covered(&exp, lo, hi), hi - lo);
        }
    }
}

#[cfg(test)]
mod loader_wiring {
    use crate::model::DatasetConfig;

    fn load(yaml: &str) -> Result<DatasetConfig, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    /// **A declaration that disagrees with itself fails at load**, not
    /// at first read (SH-85). The dataset root is not known here, so
    /// this is everything checkable without touching a file.
    #[test]
    fn a_self_contradicting_declaration_fails_at_load() {
        let err = load(
            "profiles:\n  default:\n    base_vectors:\n      source: b__NNNN.fvec\n\
             \x20     shard_stride: 1000\n      shard_count: 3\n      record_count: 99\n",
        )
        .expect_err("a total the layout cannot produce must not load");
        let msg = err.to_string();
        assert!(msg.contains("record_count"), "{msg}");
        assert!(msg.contains("base_vectors"), "names the facet: {msg}");
        assert!(msg.contains("default"), "names the profile: {msg}");
    }

    /// `NNNN` without the numbers is caught at load too (SH-47).
    #[test]
    fn a_half_stated_uniform_form_fails_at_load() {
        let err = load("profiles:\n  default:\n    base_vectors:\n      source: b__NNNN.fvec\n")
            .expect_err("NNNN without stride/count must not load");
        assert!(err.to_string().contains("shard_stride"), "{err}");
    }

    /// **A malformed shard field cannot produce a quietly-valid facet.**
    ///
    /// The deserializer tolerates unparseable profile entries because
    /// the compact `sized:` shorthand shares the map, and extending that
    /// tolerance to a broken shard declaration would make the facet
    /// vanish rather than complain — the silent shape SH-74 forbids.
    ///
    /// With the versioned shape this is caught by the *type*: a
    /// non-numeric `shard_stride` no longer matches the sharded case at
    /// all, so what remains is a plain source carrying an `NNNN` field
    /// with nothing to interpret it — which the declaration check names
    /// directly (V-21, SH-47).
    #[test]
    fn a_broken_shard_declaration_is_not_silently_skipped() {
        let err = load(
            "profiles:\n  default:\n    base_vectors:\n      source: b__NNNN.fvec\n\
             \x20     shard_stride: not-a-number\n      shard_count: 3\n",
        )
        .expect_err("a malformed shard field must not vanish the profile");
        let msg = err.to_string();
        assert!(
            msg.contains("shard_stride"),
            "names the missing piece: {msg}"
        );
        assert!(msg.contains("base_vectors"), "names the facet: {msg}");
    }

    /// A valid series loads, and an explicit one loads too.
    #[test]
    fn valid_declarations_load() {
        let uniform = load(
            "profiles:\n  default:\n    base_vectors:\n      source: b__NNNN.fvec\n\
             \x20     shard_stride: 1000\n      shard_count: 3\n      record_count: 2500\n",
        )
        .expect("a consistent uniform series loads");
        assert!(uniform.profiles["default"].base_vectors.is_some());

        let explicit = load(
            "profiles:\n  default:\n    metadata_content:\n      source:\n\
             \x20       - a.u8=100\n        - b.u8=40\n      record_count: 140\n",
        )
        .expect("a consistent explicit series loads");
        let f = explicit.profiles["default"]
            .metadata_content
            .as_ref()
            .unwrap();
        assert!(f.is_explicit_series());
        assert_eq!(f.sources().len(), 2);
        assert_eq!(f.source(), None, "a series has no single source");
    }

    /// **Every dataset written before sharding still loads**, unchanged
    /// — the compatibility anchor (SH-70, test 33).
    #[test]
    fn pre_sharding_declarations_are_untouched() {
        let cfg = load(
            "attributes:\n  distance_function: COSINE\nprofiles:\n  default:\n\
             \x20   base_vectors: base.fvec\n    query_vectors:\n      source: q.fvec\n\
             \x20     window: 0..1000\n",
        )
        .expect("plain declarations must keep loading");
        let d = &cfg.profiles["default"];
        assert_eq!(d.base_vectors.as_ref().unwrap().source(), Some("base.fvec"));
        assert_eq!(d.query_vectors.as_ref().unwrap().window(), Some("0..1000"));
        assert!(d.base_vectors.as_ref().unwrap().shard_stride().is_none());
    }

    /// The `sized:` shorthand is still skipped rather than rejected —
    /// the tolerance that made the silent-skip hazard necessary in the
    /// first place stays exactly as wide as it was.
    #[test]
    fn the_sized_shorthand_still_parses_around() {
        let cfg = load(
            "profiles:\n  sized:\n    - \"mul:1m/2\"\n  default:\n    base_vectors: base.fvec\n",
        )
        .expect("the sized shorthand must not break loading");
        assert!(cfg.profiles.contains_key("default"));
        assert!(!cfg.profiles.contains_key("sized"));
    }
}

#[cfg(test)]
mod loader_parity {
    use super::*;
    use crate::dataset::profile::DSProfileGroup;
    use crate::model::DatasetConfig;

    /// Realize a facet through the `dataset.yaml` loader.
    fn via_dataset_config(yaml: &str, profile: &str, facet: &str) -> Result<Shards, String> {
        let cfg: DatasetConfig = serde_yaml::from_str(yaml).map_err(|e| e.to_string())?;
        let p = cfg.profiles.get(profile).ok_or("no such profile")?;
        let (_, f) = p
            .facets()
            .into_iter()
            .find(|(n, _)| *n == facet)
            .ok_or("no such facet")?;
        let never = |_: &DSSource| Err("must not probe".to_string());
        realize(facet, &f.declaration(), &never).map_err(|e| e.to_string())
    }

    /// Realize the same facet through the catalog loader.
    fn via_profile_group(yaml: &str, profile: &str, facet: &str) -> Result<Shards, String> {
        // The group deserializer takes the `profiles:` map directly.
        let doc: serde_yaml::Value = serde_yaml::from_str(yaml).map_err(|e| e.to_string())?;
        let profiles = doc.get("profiles").ok_or("no profiles key")?.clone();
        let group: DSProfileGroup = serde_yaml::from_value(profiles).map_err(|e| e.to_string())?;
        let p = group.profiles.get(profile).ok_or("no such profile")?;
        let v = p.views.get(facet).ok_or("no such view")?;
        let sources = v.declaration_sources();
        let never = |_: &DSSource| Err("must not probe".to_string());
        realize(facet, &v.declaration(&sources), &never).map_err(|e| e.to_string())
    }

    /// **The parity requirement** (SH-90): one dataset realizes to the
    /// same ordinal space through both loaders.
    ///
    /// The two have always had to agree about profile inheritance, and
    /// the comment marking that duplication in `model.rs` is a standing
    /// warning. Two loaders that drift here would produce a dataset that
    /// reads one way through `TestDataGroup::load` and another through a
    /// catalog — silently, and differently per transport.
    #[test]
    fn both_loaders_realize_a_uniform_series_identically() {
        let yaml = "profiles:\n  default:\n    base_vectors:\n      source: b__NNNN.fvec\n\
                    \x20     shard_stride: 1000\n      shard_count: 3\n      record_count: 2500\n";
        let a = via_dataset_config(yaml, "default", "base_vectors").unwrap();
        let b = via_profile_group(yaml, "default", "base_vectors").unwrap();
        assert_eq!(a.count(), b.count());
        assert_eq!(a.map(), b.map());
        let names = |s: &Shards| -> Vec<String> {
            s.entries().iter().map(|e| e.source.path.clone()).collect()
        };
        assert_eq!(names(&a), names(&b), "derived filenames must match");
        assert_eq!(names(&a)[2], "b__0002.fvec");
    }

    /// The same for the explicit form, windows and counts included.
    #[test]
    fn both_loaders_realize_an_explicit_series_identically() {
        let yaml = "profiles:\n  default:\n    metadata_content:\n      source:\n\
                    \x20       - corpus.u8[0..100]=100\n        - corpus.u8[900..1000]=100\n\
                    \x20     record_count: 200\n";
        let a = via_dataset_config(yaml, "default", "metadata_content").unwrap();
        let b = via_profile_group(yaml, "default", "metadata_content").unwrap();
        assert_eq!(a.count(), 200);
        assert_eq!(a.map(), b.map());
        let bases = |s: &Shards| -> Vec<u64> { s.entries().iter().map(|e| e.file_base).collect() };
        assert_eq!(bases(&a), vec![0, 900]);
        assert_eq!(
            bases(&a),
            bases(&b),
            "entry windows must survive both paths"
        );
    }

    /// **A declaration rejected by one loader is rejected by the
    /// other.** Accepting through a catalog what `dataset.yaml` refuses
    /// would make the transport decide whether a dataset is valid.
    #[test]
    fn both_loaders_reject_the_same_declarations() {
        let bad = [
            // total the layout cannot produce
            "profiles:\n  default:\n    base_vectors:\n      source: b__NNNN.fvec\n\
             \x20     shard_stride: 1000\n      shard_count: 3\n      record_count: 99\n",
            // NNNN without the numbers
            "profiles:\n  default:\n    base_vectors:\n      source: b__NNNN.fvec\n",
            // a count that contradicts its interval
            "profiles:\n  default:\n    metadata_content:\n      source:\n\
             \x20       - a.u8[0..100]=99\n        - b.u8=10\n      record_count: 110\n",
        ];
        for yaml in bad {
            assert!(
                via_dataset_config(yaml, "default", "base_vectors").is_err()
                    || via_dataset_config(yaml, "default", "metadata_content").is_err(),
                "dataset.yaml loader accepted:\n{yaml}"
            );
            assert!(
                via_profile_group(yaml, "default", "base_vectors").is_err()
                    || via_profile_group(yaml, "default", "metadata_content").is_err(),
                "catalog loader accepted:\n{yaml}"
            );
        }
    }

    /// A plain single-file facet realizes identically too — the shape
    /// every existing dataset is in (SH-70).
    #[test]
    fn both_loaders_realize_a_plain_facet_identically() {
        let yaml = "profiles:\n  default:\n    base_vectors: base.fvec=1000\n";
        let a = via_dataset_config(yaml, "default", "base_vectors").unwrap();
        let b = via_profile_group(yaml, "default", "base_vectors").unwrap();
        assert_eq!(a.count(), 1000);
        assert_eq!(a.map(), b.map());
        assert!(a.is_single_file() && b.is_single_file());
    }

    /// A series survives a serialize/deserialize round trip through the
    /// view, so a catalog can carry one (SH-41).
    #[test]
    fn a_series_view_round_trips_through_yaml() {
        let yaml = "profiles:\n  default:\n    metadata_content:\n      source:\n\
                    \x20       - a.u8=100\n        - b.u8=40\n      record_count: 140\n";
        let doc: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let group: DSProfileGroup =
            serde_yaml::from_value(doc.get("profiles").unwrap().clone()).unwrap();
        let rendered = serde_yaml::to_string(&group).unwrap();
        let again: DSProfileGroup = serde_yaml::from_str(&rendered).unwrap();

        let v = again.profiles["default"]
            .views
            .get("metadata_content")
            .unwrap();
        assert!(v.is_series(), "the series must survive: {rendered}");
        assert_eq!(v.sources().len(), 2);
        assert_eq!(v.record_count, Some(140));
        let sources = v.declaration_sources();
        let never = |_: &DSSource| Err("must not probe".to_string());
        let s = realize("metadata_content", &v.declaration(&sources), &never).unwrap();
        assert_eq!(s.count(), 140);
    }
}
