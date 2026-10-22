// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Library API. A dataset made of files vectordata did not produce: the
//! `dataset.yaml` a directory of loose files implies.
//!
//! A local directory holding neither `dataset.yaml` nor
//! `knn_entries.yaml` is read as the descriptor these rules infer, in
//! order, and `vectordata datasets init <dir>` writes that descriptor
//! beside the files, after which opens are declared, not inferred.
//!
//! | Rule | |
//! |------|-|
//! | LF-3 | Candidates: regular files with an extension a facet format knows; not hidden files, `IDXFOR__` sidecars, `.chunks`, `.mrkl`, `.mref`, `.provenance.json`, or names holding `#`. |
//! | LF-4 | A canonical facet name classifies a file (`base_vectors.fvec`, `neighbor_indices.ivecs`, …), or a `__NNNN` series of them. |
//! | LF-5 | Else one role token in the `_`/`-`-split stem: `base`, `train`; `query`, `queries`, `test`; `gt`, `groundtruth`, `ground`+`truth`, `neighbors`, `indices`; `distances`. The format must be one the facet takes. Two roles in one stem classify nothing. |
//! | LF-6 | Else, only to fill a role of the minimum set, the shapes: an integer file with a float file's record count is the ground truth and that float file the queries; the float file with the most records is the base. One candidate per role, or a refusal listing every leftover file's dimension and count. |
//! | LF-7 | One file per facet: two candidates are refused, naming both; a series must run from `0000` without a gap. |
//! | LF-8 | The base, queries and ground truth are required, distances are taken when present, every other facet only by its canonical name. |
//! | LF-9 | The result is verified (V1–V7, [`crate::verify`]) before it is used. |
//! | LF-10 | A directory with `profiles/` and no descriptor is a veks dataset that lost it: each `profiles/<p>/` a profile by canonical names, `profiles/base/` lending its facets to the others. |
//! | LF-11 | The name is the directory's; the profile is `default`; `maxk` is the ground truth's width and `base_count` the base's count. |
//! | LF-12 | A metric token (`angular`, `cosine`, `cos`; `euclidean`, `l2`; `dot`, `ip`, `mips`) in the directory or a file name declares the distance function when every one found agrees; a `norm`/`normalized` token declares a normalized base when a sample of it is unit length. Nothing else is inferred. |
//!
//! (`docs/design/srd-loose-files-overlays-integrity.md` §3.)

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::dataset::facet::{FacetFormat, StandardFacet};
use crate::DistanceFunction;

/// How a file got its facet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rule {
    /// LF-4: its name is the facet's canonical name.
    CanonicalName,
    /// LF-5: its stem holds this role token.
    RoleToken(String),
    /// LF-6: its shape, once names had classified what they could.
    Shape,
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rule::CanonicalName => write!(f, "its canonical name (LF-4)"),
            Rule::RoleToken(t) => write!(f, "the role token '{t}' (LF-5)"),
            Rule::Shape => write!(f, "its shape (LF-6)"),
        }
    }
}

/// One file the inference gave a facet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classified {
    /// The profile it belongs to.
    pub profile: String,
    /// The facet.
    pub facet: StandardFacet,
    /// The source as the descriptor writes it: a filename, relative to
    /// the dataset, or a series pattern.
    pub source: String,
    /// The rule that classified it.
    pub rule: Rule,
}

/// What a directory of loose files infers to.
#[derive(Debug, Clone)]
pub struct Inferred {
    /// The `dataset.yaml` text, as `datasets init` writes it.
    pub yaml: String,
    /// The descriptor, parsed from [`yaml`](Self::yaml).
    pub config: crate::dataset::DatasetConfig,
    /// Every file given a facet, and why.
    pub classified: Vec<Classified>,
    /// What the inference saw and did not use: unclassified files,
    /// metric tokens that disagree.
    pub notes: Vec<String>,
}

impl Inferred {
    /// One line naming each profile's files: `base=…, query=…,
    /// neighbor_indices=…`.
    pub fn summary(&self) -> String {
        let mut by_profile: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for c in &self.classified {
            by_profile.entry(&c.profile).or_default().push(format!("{}={}", c.facet.key(), c.source));
        }
        by_profile
            .into_iter()
            .map(|(p, files)| if p == "default" { files.join(", ") } else { format!("{p}: {}", files.join(", ")) })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Why a directory does not infer to a dataset. Each names the
/// candidates, never settling by order, size or time.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InferError {
    /// The directory could not be read.
    Unreadable(String),
    /// A required facet has no file (LF-8).
    Missing {
        /// The facet.
        facet: StandardFacet,
        /// The candidates nothing classified, for renaming one.
        unclassified: Vec<String>,
    },
    /// Two files for one facet (LF-7).
    Ambiguous {
        /// The facet.
        facet: StandardFacet,
        /// Each candidate, with the rule that matched it.
        candidates: Vec<String>,
    },
    /// The shapes do not settle the roles names left open (LF-6).
    Shapes {
        /// Every leftover file with its dimension and record count.
        files: Vec<String>,
        /// Why not.
        reason: String,
    },
    /// A series has a gap, or does not start at `0000` (LF-7).
    Series(String),
    /// The files do not belong together (LF-9).
    Verification(Vec<crate::verify::VerifyFailure>),
    /// A `profiles/` directory without one profile to call `default`
    /// (LF-10).
    Profiles(Vec<String>),
}

impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferError::Unreadable(e) => write!(f, "cannot read the directory: {e}"),
            InferError::Missing { facet, unclassified } => {
                write!(f, "no file is {} (LF-8)", facet.key())?;
                if !unclassified.is_empty() {
                    write!(f, "; name one of {} for its role (base, query, gt)", unclassified.join(", "))?;
                }
                Ok(())
            }
            InferError::Ambiguous { facet, candidates } => {
                write!(f, "{} files could be {}: {} (LF-7); rename all but one", candidates.len(), facet.key(), candidates.join(", "))
            }
            InferError::Shapes { files, reason } => {
                write!(f, "{reason} (LF-6); name the files for their roles: {}", files.join(", "))
            }
            InferError::Series(e) => write!(f, "{e} (LF-7)"),
            InferError::Verification(failures) => {
                let text: Vec<String> = failures.iter().map(|x| x.to_string()).collect();
                write!(f, "the files do not belong together (LF-9): {}", text.join("; "))
            }
            InferError::Profiles(names) => {
                write!(f, "profiles/ holds {} profiles and none is default: {} (LF-10)", names.len(), names.join(", "))
            }
        }
    }
}

impl std::error::Error for InferError {}

/// The roles a token names (LF-5), restricted to the four facets a set
/// of loose files is made of.
fn role_of(token: &str) -> Option<StandardFacet> {
    match token {
        "base" | "train" => Some(StandardFacet::BaseVectors),
        "query" | "queries" | "test" => Some(StandardFacet::QueryVectors),
        "gt" | "groundtruth" | "neighbors" | "indices" => Some(StandardFacet::NeighborIndices),
        "distances" => Some(StandardFacet::NeighborDistances),
        _ => None,
    }
}

/// The distance functions a token names (LF-12).
fn metric_of(token: &str) -> Option<DistanceFunction> {
    match token {
        "angular" | "cosine" | "cos" | "euclidean" | "l2" | "dot" | "ip" | "mips" => DistanceFunction::parse(token),
        _ => None,
    }
}

/// The `_`/`-` tokens of a stem, lowercased.
fn tokens(stem: &str) -> Vec<String> {
    stem.split(['_', '-']).filter(|t| !t.is_empty()).map(str::to_lowercase).collect()
}

/// Whether `dir` holds anything to infer from: a candidate file, or a
/// `profiles/` directory (LF-10). A directory with neither is not a
/// dataset of loose files, and an open of it says the descriptor is
/// missing, as it always has.
pub fn has_loose_files(dir: &Path) -> bool {
    dir.join("profiles").is_dir()
        || std::fs::read_dir(dir).is_ok_and(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| e.file_type().is_ok_and(|t| t.is_file()) && e.file_name().to_str().is_some_and(is_candidate))
        })
}

/// LF-3: whether a directory entry is a candidate.
fn is_candidate(name: &str) -> bool {
    !(name.starts_with('.')
        || name.starts_with("IDXFOR__")
        || name.contains('#')
        || name.ends_with(".chunks")
        || name.ends_with(".mrkl")
        || name.ends_with(".mref")
        || name.ends_with(".provenance.json"))
        && name.rsplit_once('.').is_some_and(|(_, ext)| FacetFormat::from_extension(ext).is_some())
}

/// A candidate file, or a series of them.
#[derive(Debug, Clone)]
struct Candidate {
    /// The source as a descriptor writes it.
    source: String,
    /// The files, in ordinal order: one, or a series.
    files: Vec<PathBuf>,
    /// The stem without a shard suffix, and the extension.
    stem: String,
    ext: String,
    format: FacetFormat,
}

impl Candidate {
    fn is_series(&self) -> bool {
        self.files.len() > 1 || self.source.contains(crate::dataset::shards::SHARD_FIELD)
    }

    /// The records and dimension of a vector file, from its headers:
    /// `None` for anything but a uniform vector file.
    fn shape(&self) -> Option<(usize, usize)> {
        let mut total = 0usize;
        let mut dim = None;
        for f in &self.files {
            let r = veks_io::open_vec_native(&f.to_string_lossy()).ok()?;
            dim.get_or_insert(r.dim());
            total += r.count();
        }
        Some((total, dim?))
    }

    fn is_integer_vectors(&self) -> bool {
        matches!(self.format, FacetFormat::IntegerXvec)
    }

    fn is_float_vectors(&self) -> bool {
        matches!(self.format, FacetFormat::FloatXvec)
    }
}

/// The candidates of `dir`, a `__NNNN` series folded into one (LF-3,
/// LF-7).
fn candidates(dir: &Path) -> Result<Vec<Candidate>, InferError> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| InferError::Unreadable(format!("{}: {e}", dir.display())))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|n| is_candidate(n))
        .collect();
    names.sort();
    let mut singles = Vec::new();
    let mut series: BTreeMap<(String, String), Vec<(u32, String)>> = BTreeMap::new();
    for name in names {
        let (stem, ext) = name.rsplit_once('.').expect("a candidate has an extension");
        let base = StandardFacet::strip_shard_suffix(stem);
        if base != stem {
            let index: u32 = stem[base.len() + 2..].parse().expect("four digits");
            series.entry((base.to_string(), ext.to_string())).or_default().push((index, name.clone()));
        } else {
            singles.push(Candidate {
                source: name.clone(),
                files: vec![dir.join(&name)],
                stem: stem.to_string(),
                ext: ext.to_string(),
                format: FacetFormat::from_extension(ext).expect("a candidate's format"),
            });
        }
    }
    for ((stem, ext), mut parts) in series {
        parts.sort();
        for (expected, (index, name)) in parts.iter().enumerate() {
            if *index != expected as u32 {
                return Err(InferError::Series(format!(
                    "the series {stem}__NNNN.{ext} has no file {} before {name}",
                    crate::dataset::shards::shard_name(&stem, &ext, expected as u32)
                )));
            }
        }
        singles.push(Candidate {
            source: crate::dataset::shards::shard_source_spec(&stem, &ext),
            files: parts.iter().map(|(_, n)| dir.join(n)).collect(),
            stem,
            format: FacetFormat::from_extension(&ext).expect("a candidate's format"),
            ext,
        });
    }
    Ok(singles)
}

/// The files of one profile, classified (LF-4 to LF-8).
fn classify(
    profile: &str,
    cands: Vec<Candidate>,
    names_only: bool,
    notes: &mut Vec<String>,
) -> Result<indexmap::IndexMap<StandardFacet, (Candidate, Rule)>, InferError> {
    let mut by_facet: indexmap::IndexMap<StandardFacet, Vec<(Candidate, Rule)>> = indexmap::IndexMap::new();
    let mut left = Vec::new();
    for c in cands {
        // LF-4: the canonical name, through the one classifier.
        if let Some((facet, _)) = StandardFacet::classify(&format!("{}.{}", c.stem, c.ext)) {
            by_facet.entry(facet).or_default().push((c, Rule::CanonicalName));
            continue;
        }
        if names_only {
            notes.push(format!("{}: not a canonical facet name; not used", c.source));
            continue;
        }
        // LF-5: one role token, in a format the facet takes.
        let toks = tokens(&c.stem);
        let mut roles: Vec<(StandardFacet, String)> = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            let role = role_of(t).or_else(|| {
                (t == "ground" && toks.get(i + 1).is_some_and(|n| n == "truth")).then_some(StandardFacet::NeighborIndices)
            });
            if let Some(r) = role
                && !roles.iter().any(|(f, _)| *f == r)
            {
                roles.push((r, if t == "ground" { "ground_truth".to_string() } else { t.clone() }));
            }
        }
        match roles.as_slice() {
            [(facet, token)] if facet.accepts_format(c.format) => {
                by_facet.entry(*facet).or_default().push((c, Rule::RoleToken(token.clone())));
            }
            [(facet, token)] => {
                notes.push(format!("{}: the role token '{token}' names {}, which a .{} file cannot be; not classified", c.source, facet.key(), c.ext));
                left.push(c);
            }
            [] => left.push(c),
            many => {
                let names: Vec<&str> = many.iter().map(|(_, t)| t.as_str()).collect();
                notes.push(format!("{}: two role tokens ({}); not classified", c.source, names.join(", ")));
                left.push(c);
            }
        }
    }

    // LF-7: one file per facet.
    for (facet, found) in &by_facet {
        if found.len() > 1 {
            return Err(InferError::Ambiguous {
                facet: *facet,
                candidates: found.iter().map(|(c, r)| format!("{} (by {r})", c.source)).collect(),
            });
        }
    }
    let mut chosen: indexmap::IndexMap<StandardFacet, (Candidate, Rule)> =
        by_facet.into_iter().map(|(f, mut v)| (f, v.remove(0))).collect();

    // LF-6: shapes, only to finish the minimum set.
    let required = [StandardFacet::BaseVectors, StandardFacet::QueryVectors, StandardFacet::NeighborIndices];
    if !names_only && required.iter().any(|f| !chosen.contains_key(f)) {
        shape_fill(&mut chosen, &mut left)?;
    }

    // LF-8: the minimum set.
    for facet in required {
        if !chosen.contains_key(&facet) {
            return Err(InferError::Missing { facet, unclassified: left.iter().map(|c| c.source.clone()).collect() });
        }
    }
    for c in &left {
        notes.push(format!("{}: not classified; not used", c.source));
    }
    let _ = profile;
    Ok(chosen)
}

/// LF-6: fill the roles of the minimum set names left open, from the
/// leftover files' shapes, or refuse.
fn shape_fill(chosen: &mut indexmap::IndexMap<StandardFacet, (Candidate, Rule)>, left: &mut Vec<Candidate>) -> Result<(), InferError> {
    let shaped: Vec<(Candidate, usize, usize)> =
        left.iter().filter_map(|c| c.shape().map(|(n, d)| (c.clone(), n, d))).collect();
    let listing = || -> Vec<String> { shaped.iter().map(|(c, n, d)| format!("{} ({n} × {d})", c.source)).collect() };
    let refuse = |reason: &str| InferError::Shapes { files: listing(), reason: reason.to_string() };
    let count_of = |facet: StandardFacet, chosen: &indexmap::IndexMap<StandardFacet, (Candidate, Rule)>| {
        chosen.get(&facet).and_then(|(c, _)| c.shape()).map(|(n, _)| n)
    };
    let floats: Vec<&(Candidate, usize, usize)> = shaped.iter().filter(|(c, _, _)| c.is_float_vectors()).collect();
    let ints: Vec<&(Candidate, usize, usize)> = shaped.iter().filter(|(c, _, _)| c.is_integer_vectors()).collect();
    let mut taken: Vec<String> = Vec::new();

    if !chosen.contains_key(&StandardFacet::NeighborIndices) {
        // An integer file whose count is the queries'.
        let query_n = count_of(StandardFacet::QueryVectors, chosen);
        let pairs: Vec<(&Candidate, Option<&Candidate>)> = ints
            .iter()
            .flat_map(|(i, n, _)| match query_n {
                Some(q) => (*n == q).then_some((i, None)).into_iter().collect::<Vec<_>>(),
                None => floats.iter().filter(|(_, fn_, _)| fn_ == n).map(|(f, _, _)| (i, Some(f))).collect(),
            })
            .collect();
        match pairs.as_slice() {
            [(gt, query)] => {
                chosen.insert(StandardFacet::NeighborIndices, ((*gt).clone(), Rule::Shape));
                taken.push(gt.source.clone());
                if let Some(q) = query {
                    chosen.insert(StandardFacet::QueryVectors, ((*q).clone(), Rule::Shape));
                    taken.push(q.source.clone());
                }
            }
            [] => return Err(refuse("no integer file has a record per query")),
            _ => return Err(refuse("more than one integer file could be the ground truth")),
        }
    }
    if !chosen.contains_key(&StandardFacet::QueryVectors) {
        let gt_n = count_of(StandardFacet::NeighborIndices, chosen);
        let q: Vec<&&(Candidate, usize, usize)> =
            floats.iter().filter(|(c, n, _)| Some(*n) == gt_n && !taken.contains(&c.source)).collect();
        match q.as_slice() {
            [(c, _, _)] => {
                chosen.insert(StandardFacet::QueryVectors, (c.clone(), Rule::Shape));
                taken.push(c.source.clone());
            }
            [] => return Err(refuse("no float file has a record per ground-truth row")),
            _ => return Err(refuse("more than one float file has a record per ground-truth row")),
        }
    }
    if !chosen.contains_key(&StandardFacet::BaseVectors) {
        let rest: Vec<&&(Candidate, usize, usize)> = floats.iter().filter(|(c, _, _)| !taken.contains(&c.source)).collect();
        let most = rest.iter().map(|(_, n, _)| *n).max();
        let biggest: Vec<&&&(Candidate, usize, usize)> = rest.iter().filter(|(_, n, _)| Some(*n) == most).collect();
        let query_n = count_of(StandardFacet::QueryVectors, chosen).unwrap_or(0);
        match biggest.as_slice() {
            [(c, n, _)] if *n > query_n => {
                chosen.insert(StandardFacet::BaseVectors, (c.clone(), Rule::Shape));
                taken.push(c.source.clone());
            }
            [_] => return Err(refuse("the largest float file has no more records than the queries")),
            [] => return Err(refuse("no float file is left for the base")),
            _ => return Err(refuse("two float files tie for the most records")),
        }
    }
    left.retain(|c| !taken.contains(&c.source));
    Ok(())
}

/// LF-12: the distance function the names declare, when they agree.
fn declared_metric(dir_name: &str, chosen: &[&Candidate], notes: &mut Vec<String>) -> Option<DistanceFunction> {
    let mut found: Vec<(DistanceFunction, String)> = Vec::new();
    let mut scan = |what: &str, stem: &str| {
        for t in tokens(stem) {
            if let Some(m) = metric_of(&t) {
                found.push((m, format!("'{t}' in {what}")));
            }
        }
    };
    scan(dir_name, dir_name);
    for c in chosen {
        scan(&c.source, &c.stem);
    }
    let first = found.first()?.0;
    if found.iter().all(|(m, _)| *m == first) {
        Some(first)
    } else {
        let all: Vec<String> = found.iter().map(|(m, w)| format!("{m} ({w})")).collect();
        notes.push(format!("the names disagree on the metric: {}; none declared (LF-12)", all.join(", ")));
        None
    }
}

/// The `source:` lines of one facet, at `indent`.
fn source_yaml(facet: StandardFacet, c: &Candidate, indent: &str) -> String {
    if !c.is_series() {
        return format!("{indent}{}: {}\n", facet.key(), c.source);
    }
    let count = |f: &PathBuf| veks_io::open_vec_native(&f.to_string_lossy()).map(|r| r.count()).unwrap_or(0);
    let stride = c.files.first().map(count).unwrap_or(0);
    let records: usize = c.files.iter().map(count).sum();
    format!(
        "{indent}{}:\n{indent}  source: {}\n{indent}  shard_stride: {stride}\n{indent}  shard_count: {}\n{indent}  record_count: {records}\n",
        facet.key(),
        c.source,
        c.files.len()
    )
}

/// Infer the dataset a directory of loose files makes (LF-1 to LF-12),
/// verified (LF-9). Does not look at `dataset.yaml` or
/// `knn_entries.yaml`; a caller that infers only in their absence
/// checks for them.
///
/// ```
/// use vectordata::dataset::infer::{infer_dir, Rule};
/// use vectordata::dataset::StandardFacet;
/// use vectordata::DistanceFunction;
/// # use std::io::Write;
/// # fn rows(path: &std::path::Path, rows: &[Vec<f32>]) { let mut f = std::fs::File::create(path).unwrap(); for r in rows { f.write_all(&(r.len() as i32).to_le_bytes()).unwrap(); for v in r { f.write_all(&v.to_le_bytes()).unwrap(); } } }
/// # fn ids(path: &std::path::Path, rows: &[Vec<i32>]) { let mut f = std::fs::File::create(path).unwrap(); for r in rows { f.write_all(&(r.len() as i32).to_le_bytes()).unwrap(); for v in r { f.write_all(&v.to_le_bytes()).unwrap(); } } }
/// # let root = tempfile::tempdir().unwrap();
/// # let dir = root.path().join("ada_002_100k");
/// # std::fs::create_dir(&dir).unwrap();
/// # rows(&dir.join("ada_002_100k_base_4.fvecs"), &[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.6, 0.8], vec![0.8, 0.6]]);
/// # rows(&dir.join("ada_002_100k_query_2.fvecs"), &[vec![1.0, 0.0], vec![0.0, 1.0]]);
/// # ids(&dir.join("ada_002_100k_gt_ip_2.ivecs"), &[vec![0, 3], vec![1, 2]]);
/// let inferred = infer_dir(&dir).unwrap();
/// assert_eq!(inferred.summary(), "base_vectors=ada_002_100k_base_4.fvecs, query_vectors=ada_002_100k_query_2.fvecs, neighbor_indices=ada_002_100k_gt_ip_2.ivecs");
/// assert!(inferred.classified.iter().all(|c| matches!(c.rule, Rule::RoleToken(_))));
/// let attrs = inferred.config.attributes.as_ref().unwrap();
/// assert_eq!(attrs.distance(), Some(DistanceFunction::DotProduct)); // from `ip`
/// assert!(inferred.yaml.contains("base_count: 4") && inferred.yaml.contains("maxk: 2"));
/// ```
pub fn infer_dir(dir: &Path) -> Result<Inferred, InferError> {
    let dir_name = dir
        .canonicalize()
        .ok()
        .and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "dataset".to_string());
    let mut notes = Vec::new();

    // LF-10: a veks layout that lost its descriptor.
    let profiles_dir = dir.join("profiles");
    let profiles: Vec<(String, indexmap::IndexMap<StandardFacet, (Candidate, Rule)>)> = if profiles_dir.is_dir() {
        let mut names: Vec<String> = std::fs::read_dir(&profiles_dir)
            .map_err(|e| InferError::Unreadable(format!("{}: {e}", profiles_dir.display())))?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .collect();
        names.sort();
        let shared: Vec<Candidate> = if names.iter().any(|n| n == "base") {
            candidates(&profiles_dir.join("base"))?
                .into_iter()
                .map(|mut c| {
                    c.source = format!("profiles/base/{}", c.source);
                    c
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut own: Vec<String> = names.iter().filter(|n| *n != "base").cloned().collect();
        if own.is_empty() && !shared.is_empty() {
            own.push("base".to_string());
        }
        if own.len() > 1 && !own.iter().any(|n| n == "default") {
            return Err(InferError::Profiles(own));
        }
        let mut out = Vec::new();
        for p in own {
            let mut cands: Vec<Candidate> = if p == "base" {
                Vec::new()
            } else {
                candidates(&profiles_dir.join(&p))?
                    .into_iter()
                    .map(|mut c| {
                        c.source = format!("profiles/{p}/{}", c.source);
                        c
                    })
                    .collect()
            };
            // profiles/base/ lends what the profile does not have.
            for s in &shared {
                let facet = StandardFacet::classify(&format!("{}.{}", s.stem, s.ext)).map(|(f, _)| f);
                let has = cands
                    .iter()
                    .any(|c| StandardFacet::classify(&format!("{}.{}", c.stem, c.ext)).map(|(f, _)| f) == facet);
                if !has {
                    cands.push(s.clone());
                }
            }
            let name = if p == "base" { "default".to_string() } else { p.clone() };
            let chosen = classify(&name, cands, true, &mut notes)?;
            out.push((name, chosen));
        }
        out
    } else {
        let chosen = classify("default", candidates(dir)?, false, &mut notes)?;
        vec![("default".to_string(), chosen)]
    };

    // LF-12: the metric, from the names.
    let all: Vec<&Candidate> = profiles.iter().flat_map(|(_, c)| c.values().map(|(c, _)| c)).collect();
    let metric = declared_metric(&dir_name, &all, &mut notes);
    let normalized_token = std::iter::once(dir_name.as_str())
        .chain(all.iter().map(|c| c.stem.as_str()))
        .any(|s| tokens(s).iter().any(|t| t == "norm" || t == "normalized"));

    // LF-11: the descriptor.
    let any_series = all.iter().any(|c| c.is_series());
    let mut yaml = String::from(
        "# Inferred by vectordata from the files in this directory\n# (docs/design/srd-loose-files-overlays-integrity.md §3).\n",
    );
    if any_series {
        yaml.push_str("format_version: 2\n");
    }
    yaml.push_str(&format!("name: {dir_name}\n"));
    let mut attrs = String::new();
    if let Some(m) = metric {
        attrs.push_str(&format!("  distance_function: {}\n", m.name()));
    }
    if normalized_token {
        attrs.push_str("  is_normalized: true\n");
    }
    if !attrs.is_empty() {
        yaml.push_str("attributes:\n");
        yaml.push_str(&attrs);
    }
    yaml.push_str("profiles:\n");
    let mut classified = Vec::new();
    for (name, chosen) in &profiles {
        yaml.push_str(&format!("  {name}:\n"));
        if let Some((n, _)) = chosen.get(&StandardFacet::BaseVectors).and_then(|(c, _)| c.shape()) {
            yaml.push_str(&format!("    base_count: {n}\n"));
        }
        if let Some((_, k)) = chosen.get(&StandardFacet::NeighborIndices).and_then(|(c, _)| c.shape()) {
            yaml.push_str(&format!("    maxk: {k}\n"));
        }
        for facet in StandardFacet::PREFERRED_ORDER {
            if let Some((c, rule)) = chosen.get(facet) {
                yaml.push_str(&source_yaml(*facet, c, "    "));
                classified.push(Classified { profile: name.clone(), facet: *facet, source: c.source.clone(), rule: rule.clone() });
            }
        }
    }
    let config = crate::dataset::DatasetConfig::parse(&yaml)
        .map_err(|e| InferError::Unreadable(format!("the inferred descriptor does not parse: {e}")))?;

    // LF-9 (and LF-12's normalization claim): verified before use.
    let group = crate::TestDataGroup::from_config(&dir_name, crate::Location::from_path(dir), config.clone());
    let mut failures = Vec::new();
    for (name, _) in &profiles {
        let Some(view) = group.profile(name) else { continue };
        let report = view.verify_report(&crate::verify::VerifyOptions::default(), &mut crate::verify::NoProgress);
        failures.extend(report.failures);
        if normalized_token
            && let Ok(m) = crate::verify::measure_normalization(&*view)
            && !m.value
        {
            return Err(InferError::Verification(vec![crate::verify::VerifyFailure {
                check: crate::verify::Check::V8Normalized,
                facet: "base_vectors".to_string(),
                file: view.facet_source("base_vectors"),
                found: format!(
                    "is named normalized but a sample of {} vectors has a norm {} from 1",
                    m.sample, m.worst_deviation
                ),
            }]));
        }
    }
    if !failures.is_empty() {
        return Err(InferError::Verification(failures));
    }
    Ok(Inferred { yaml, config, classified, notes })
}
