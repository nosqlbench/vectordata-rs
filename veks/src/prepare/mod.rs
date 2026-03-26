// Copyright (c) DataStax, Inc.
// SPDX-License-Identifier: Apache-2.0

//! `veks prepare` — dataset preparation commands.
//!
//! These commands create, modify, and maintain dataset artifacts:
//! importing source data, adding sized profiles, generating catalogs,
//! and managing cache compression. They are the producer-side
//! counterpart to the consumer-side `veks datasets` commands.

pub(crate) mod cache_compress;
pub(crate) mod cache_gc;
pub mod import;
pub mod stratify;
pub(crate) mod wizard;

use std::path::PathBuf;

use clap::{Args, Subcommand};

/// Dataset preparation and pipeline management
#[derive(Args)]
#[command(disable_help_subcommand = true)]
pub struct PrepareArgs {
    #[command(subcommand)]
    pub command: PrepareCommand,
}

/// Available subcommands under `veks prepare`.
#[derive(Subcommand)]
pub enum PrepareCommand {
    /// Bootstrap a new dataset directory from source files
    Bootstrap {
        /// Interactive wizard mode — prompts for each option
        #[arg(long, short = 'i')]
        interactive: bool,

        /// Accept all defaults without prompting (use with -i)
        #[arg(long, short = 'y')]
        yes: bool,

        /// Dataset name (required unless --interactive)
        #[arg(long)]
        name: Option<String>,

        /// Output directory for the new dataset (required unless --interactive)
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,

        /// Path to base vectors (file or directory)
        #[arg(long)]
        base_vectors: Option<PathBuf>,

        /// Path to separate query vectors (file or directory)
        #[arg(long)]
        query_vectors: Option<PathBuf>,

        /// Extract queries from base via shuffle (default when no --query-vectors)
        #[arg(long)]
        self_search: bool,

        /// Number of query vectors in self-search mode
        #[arg(long, default_value = "10000")]
        query_count: u32,

        /// Path to metadata (file or directory)
        #[arg(long)]
        metadata: Option<PathBuf>,

        /// Pre-computed ground truth indices (ivec file)
        #[arg(long)]
        ground_truth: Option<PathBuf>,

        /// Pre-computed ground truth distances (fvec file)
        #[arg(long)]
        ground_truth_distances: Option<PathBuf>,

        /// Distance metric for KNN computation (auto-detected from data if not specified)
        #[arg(long, default_value = "auto")]
        metric: String,

        /// Number of neighbors for KNN ground truth
        #[arg(long, default_value = "100")]
        neighbors: u32,

        /// Random seed for shuffle
        #[arg(long, default_value = "42")]
        seed: u32,

        /// Dataset description
        #[arg(long)]
        description: Option<String>,

        /// Skip deduplication stage
        #[arg(long)]
        no_dedup: bool,

        /// Skip zero-vector check and clean ordinals
        #[arg(long)]
        no_zero_check: bool,

        /// Skip filtered KNN even when metadata is present
        #[arg(long)]
        no_filtered: bool,

        /// L2-normalize vectors during extraction
        #[arg(long)]
        normalize: bool,

        /// Overwrite existing dataset.yaml
        #[arg(long)]
        force: bool,

        /// Start fresh — ignore existing dataset.yaml, variables.yaml,
        /// and .cache state. Equivalent to --force but also removes
        /// variables.yaml and the progress log.
        #[arg(long, short = 'r')]
        restart: bool,

        /// Fraction of base vectors to use. Use "%" suffix for percentages
        /// (e.g., "1%", "50%", "100%") or decimal fractions < 1 (e.g., "0.01", "0.5").
        /// Bare whole numbers like "1" are rejected — use "1%" instead.
        #[arg(long, default_value = "100%")]
        base_fraction: String,

        /// Required dataset facets (e.g., "BQGD", "base,query,gt,dist").
        /// Controls which pipeline steps are generated. When omitted,
        /// facets are inferred from available inputs (B->BQGD, M->BQGDMPR, B+M->BQGDMPRF).
        #[arg(long)]
        required_facets: Option<String>,

        /// Significant digits for computed counts (base_end, etc.).
        /// Default 2 produces clean sizes (180M instead of 184623729).
        /// Set to 10+ to disable rounding.
        #[arg(long, default_value = "2")]
        round_digits: u32,

        /// Run dedup+zeros on the full input before subsetting (slower but
        /// stable). Default: subset first, then dedup on the subset.
        #[arg(long)]
        pedantic_dedup: bool,

        /// Fully automatic mode — implies --interactive --restart --yes (-iry).
        /// Detects files by name, accepts all defaults, and starts fresh.
        #[arg(long)]
        auto: bool,
    },
    /// Add sized profiles to an existing dataset for multi-scale benchmarking
    Stratify {
        /// Dataset directory or path to dataset.yaml
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Size range specification (e.g., "mul:1m..400m/2")
        #[arg(long)]
        spec: Option<String>,

        /// Interactive wizard to choose sized profile options
        #[arg(long, short = 'i')]
        interactive: bool,

        /// Overwrite existing sized profiles
        #[arg(long)]
        force: bool,

        /// Accept defaults without prompting
        #[arg(long, short = 'y')]
        yes: bool,

        /// Fully automatic — use standard spec, overwrite existing (implies -iy --force)
        #[arg(long)]
        auto: bool,
    },
    /// Generate and manage dataset catalog index files
    #[command(disable_help_subcommand = true)]
    Catalog {
        #[command(subcommand)]
        command: CatalogSubcommand,
    },
    /// Remove orphaned cache files from .cache/ that are no longer
    /// referenced by the current pipeline configuration.
    CacheGc {
        /// Dataset directory or path to dataset.yaml
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Dry run — show what would be removed without deleting
        #[arg(long)]
        dry_run: bool,
    },
    /// Compress eligible cache files to save disk space
    CacheCompress {
        /// Cache directory to compress (default: .cache/ in current directory)
        #[arg(default_value = ".cache")]
        cache_dir: PathBuf,

        /// Compression level (0-9, default 9 = maximum)
        #[arg(long, default_value = "9")]
        level: u32,

        /// Dry run — show what would be compressed without changing files
        #[arg(long)]
        dry_run: bool,
    },
    /// Execute the pipeline defined in dataset.yaml
    Run(crate::pipeline::RunArgs),
    /// Pre-flight checks for dataset readiness
    Check(crate::check::CheckArgs),
    /// Emit a dataset pipeline as an equivalent shell script
    GenerateScript(crate::pipeline::ScriptArgs),
    /// Publish dataset to S3
    Publish(crate::publish::PublishArgs),
    /// Decompress cache files back to their original form
    CacheUncompress {
        /// Cache directory to uncompress (default: .cache/ in current directory)
        #[arg(default_value = ".cache")]
        cache_dir: PathBuf,

        /// Dry run — show what would be uncompressed without changing files
        #[arg(long)]
        dry_run: bool,
    },
}

/// Subcommands under `veks prepare catalog`.
#[derive(Subcommand)]
pub enum CatalogSubcommand {
    /// Generate catalog.json and catalog.yaml from dataset directories
    Generate {
        /// Root directory to scan for datasets (default: current directory)
        #[arg(default_value = ".")]
        input: PathBuf,

        /// Base filename for catalog files (without extension)
        #[arg(long, default_value = "catalog")]
        basename: String,

        /// Walk up to the .publish_url root and generate catalogs for the
        /// entire publish hierarchy.
        #[arg(long)]
        for_publish_url: bool,

        /// Only update catalog files that already exist (skip new directories)
        #[arg(long)]
        update: bool,
    },
}

/// Parse a fraction string like "50%", "1%", "0.5", or "1.0" into 0.0–1.0.
/// Parse a fraction string. Accepts:
/// - "50%" → 0.50  (percentage with suffix)
/// - "1%" → 0.01
/// - "0.5" → 0.50  (decimal fraction, must be < 1.0)
/// - "0.01" → 0.01
///
/// Bare whole numbers like "1" or "50" are rejected to avoid ambiguity.
/// Use "1%" or "0.01" instead.
fn parse_fraction(s: &str) -> f64 {
    let s = s.trim();
    if let Some(pct) = s.strip_suffix('%') {
        let pct_val = match pct.trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("Error: --base-fraction '{}' — '{}' is not a valid number", s, pct.trim());
                std::process::exit(1);
            }
        };
        if pct_val < 0.0 || pct_val > 100.0 {
            eprintln!("Error: --base-fraction '{}' — percentage must be between 0% and 100%", s);
            std::process::exit(1);
        }
        pct_val / 100.0
    } else {
        let v = match s.parse::<f64>() {
            Ok(v) => v,
            Err(_) => {
                eprintln!("Error: --base-fraction '{}' is not a valid number. Use e.g. '5%' or '0.05'", s);
                std::process::exit(1);
            }
        };
        if v >= 1.0 {
            if v == 1.0 && !s.contains('.') {
                // Bare "1" is ambiguous — could mean 1% or 100%.
                // Require "1%" or "0.01" instead.
                eprintln!(
                    "Error: --base-fraction '{}' is ambiguous. Use '{}%' for {}% or '{}' for a decimal fraction.",
                    s, s, s, format!("0.{:02}", v as u32)
                );
                std::process::exit(1);
            }
            // Bare whole numbers > 1 are also ambiguous
            eprintln!(
                "Error: --base-fraction '{}' is ambiguous. Use '{}%' for {}%.",
                s, s, s
            );
            std::process::exit(1);
        } else {
            if v < 0.0 {
                eprintln!("Error: --base-fraction '{}' — fraction must be positive", s);
                std::process::exit(1);
            }
            v
        }
    }
}

/// Dispatch a prepare subcommand.
pub fn run(args: PrepareArgs) {
    match args.command {
        PrepareCommand::Bootstrap {
            interactive, yes, name, output, base_vectors, query_vectors,
            self_search, query_count, metadata, ground_truth,
            ground_truth_distances, metric, neighbors, seed, description,
            no_dedup, no_zero_check, no_filtered, normalize, force, restart,
            base_fraction, required_facets, round_digits, pedantic_dedup, auto,
        } => {
            // --auto implies -i -r -y
            let interactive = interactive || auto;
            let yes = yes || auto;
            let base_fraction = parse_fraction(&base_fraction);
            let metric = if metric == "auto" && !interactive {
                // Non-interactive: detect metric from base vectors now.
                // Interactive mode: the wizard handles detection after
                // resolving the actual file path.
                let (m, reason) = base_vectors.as_ref()
                    .map(|p| import::detect_metric(p))
                    .unwrap_or_else(|| ("Cosine".to_string(), "default".to_string()));
                eprintln!("Metric: {} ({})", m, reason);
                m
            } else {
                metric
            };
            let restart = restart || auto;

            // Validate explicitly provided paths exist before proceeding.
            // Without this, a typo or misplaced flag value (e.g. --base-vectors '5%')
            // silently falls through to auto-detection in the wizard.
            fn validate_path_arg(name: &str, path: &Option<std::path::PathBuf>) {
                if let Some(p) = path {
                    if !p.exists() {
                        eprintln!("Error: --{name} path '{}' does not exist", p.display());
                        std::process::exit(1);
                    }
                }
            }
            validate_path_arg("base-vectors", &base_vectors);
            validate_path_arg("query-vectors", &query_vectors);
            validate_path_arg("metadata", &metadata);
            validate_path_arg("ground-truth", &ground_truth);
            validate_path_arg("ground-truth-distances", &ground_truth_distances);
            if let Some(ref o) = output {
                // output directory is allowed to not exist yet (we'll create it),
                // but its parent must exist
                if let Some(parent) = o.parent() {
                    if !parent.as_os_str().is_empty() && !parent.exists() {
                        eprintln!("Error: --output parent directory '{}' does not exist", parent.display());
                        std::process::exit(1);
                    }
                }
            }

            if restart {
                let out_dir = output.as_deref()
                    .unwrap_or_else(|| std::path::Path::new("."));
                let yaml_path = out_dir.join("dataset.yaml");

                // Back up existing dataset.yaml before overwriting
                if yaml_path.exists() {
                    match crate::check::fix::create_backup(&yaml_path) {
                        Ok(bp) => eprintln!("Backed up {} → {}", yaml_path.display(), bp.display()),
                        Err(e) => eprintln!("Warning: backup failed: {}", e),
                    }
                }

                // Capture the old config and state — if the regenerated config
                // is identical, we restore the progress log and variables so
                // completed steps and their outputs are preserved.
                let old_yaml = std::fs::read_to_string(&yaml_path).ok();
                let progress_path = out_dir.join(".cache/.upstream.progress.yaml");
                let variables_path = out_dir.join("variables.yaml");
                let old_progress = std::fs::read(&progress_path).ok();
                let old_variables = std::fs::read(&variables_path).ok();
                let _ = std::fs::remove_file(&yaml_path);
                let _ = std::fs::remove_file(&variables_path);
                let _ = std::fs::remove_file(&progress_path);
                eprintln!("Restarting: removed dataset.yaml, variables.yaml, and progress log");

                // After regeneration, check if dataset.yaml changed.
                // If identical, restore progress log AND variables.yaml.
                let check_and_restore = move |dir: &std::path::Path| {
                    if let Some(ref old) = old_yaml {
                        if let Ok(new) = std::fs::read_to_string(dir.join("dataset.yaml")) {
                            if *old == new {
                                // Only restore if BOTH progress and variables were captured.
                                // If either is missing, the state is inconsistent — don't
                                // restore a partial state that would cause "variable not defined"
                                // errors for steps marked as fresh.
                                if old_progress.is_some() && old_variables.is_some() {
                                    if let Some(ref progress) = old_progress {
                                        let pp = dir.join(".cache/.upstream.progress.yaml");
                                        let _ = std::fs::create_dir_all(pp.parent().unwrap_or(dir));
                                        let _ = std::fs::write(&pp, progress);
                                    }
                                    if let Some(ref vars) = old_variables {
                                        let _ = std::fs::write(dir.join("variables.yaml"), vars);
                                    }
                                    eprintln!("Config unchanged — restored progress log and variables (completed steps preserved)");
                                } else {
                                    eprintln!("Config unchanged but prior state incomplete — steps will re-execute");
                                }
                            } else {
                                eprintln!("Config changed — steps will re-execute");
                            }
                        }
                    }
                };

                if interactive {
                    // CLI flags pre-seed wizard defaults. The wizard presents
                    // them as the default choice; --auto accepts them all.
                    let seeds = wizard::WizardSeeds {
                        name: name.clone(),
                        output: output.clone(),
                        base_vectors: base_vectors.clone(),
                        query_vectors: query_vectors.clone(),
                        self_search: if self_search { Some(true) } else { None },
                        query_count: None, // use wizard default (10000)
                        metadata: metadata.clone(),
                        ground_truth: ground_truth.clone(),
                        ground_truth_distances: ground_truth_distances.clone(),
                        metric: if metric != "auto" { Some(metric.clone()) } else { None },
                        neighbors: None, // use wizard default
                        seed: None,
                        description: description.clone(),
                        no_dedup: if no_dedup { Some(true) } else { None },
                        no_zero_check: if no_zero_check { Some(true) } else { None },
                        no_filtered: if no_filtered { Some(true) } else { None },
                        normalize: if normalize { Some(true) } else { None },
                        base_fraction: if base_fraction < 1.0 { Some(base_fraction) } else { None },
                        pedantic_dedup: if pedantic_dedup { Some(true) } else { None },
                        required_facets: required_facets.clone(),
                        round_digits: Some(round_digits),
                        selectivity: None,
                    };
                    let args = wizard::run_wizard_with_options(yes, auto, seeds);
                    let out = args.output.clone();
                    import::run(args);
                    check_and_restore(&out);
                } else {
                    let name = name.unwrap_or_else(|| {
                        eprintln!("Error: --name is required (or use --interactive)");
                        std::process::exit(1);
                    });
                    let out = output.clone().unwrap_or_else(|| {
                        eprintln!("Error: --output is required (or use --interactive)");
                        std::process::exit(1);
                    });
                    import::run(import::ImportArgs {
                        name, output: out.clone(), base_vectors, query_vectors, self_search,
                        query_count, metadata, ground_truth, ground_truth_distances,
                        metric, neighbors, seed, description, no_dedup, no_zero_check,
                        no_filtered, normalize, force: force || restart,
                        base_convert_format: None,
                        query_convert_format: None,
                        compress_cache: false,
                        sized_profiles: None,
                        base_fraction,
                        required_facets: required_facets.clone(),
                        round_digits,
                        pedantic_dedup,
                        selectivity: 0.0001,
                    });
                    check_and_restore(&out);
                }
            } else if interactive {
                let seeds = wizard::WizardSeeds {
                    name: name.clone(),
                    output: output.clone(),
                    base_vectors: base_vectors.clone(),
                    query_vectors: query_vectors.clone(),
                    self_search: if self_search { Some(true) } else { None },
                    query_count: None,
                    metadata: metadata.clone(),
                    ground_truth: ground_truth.clone(),
                    ground_truth_distances: ground_truth_distances.clone(),
                    metric: if metric != "auto" { Some(metric.clone()) } else { None },
                    neighbors: None,
                    seed: None,
                    description: description.clone(),
                    no_dedup: if no_dedup { Some(true) } else { None },
                    no_zero_check: if no_zero_check { Some(true) } else { None },
                    no_filtered: if no_filtered { Some(true) } else { None },
                    normalize: if normalize { Some(true) } else { None },
                    base_fraction: if base_fraction < 1.0 { Some(base_fraction) } else { None },
                    pedantic_dedup: if pedantic_dedup { Some(true) } else { None },
                    required_facets: required_facets.clone(),
                    round_digits: Some(round_digits),
                    selectivity: None,
                };
                let args = wizard::run_wizard_with_options(yes, false, seeds);
                import::run(args);
            } else {
                let name = name.unwrap_or_else(|| {
                    eprintln!("Error: --name is required (or use --interactive)");
                    std::process::exit(1);
                });
                let output = output.unwrap_or_else(|| {
                    eprintln!("Error: --output is required (or use --interactive)");
                    std::process::exit(1);
                });
                import::run(import::ImportArgs {
                    name, output, base_vectors, query_vectors, self_search,
                    query_count, metadata, ground_truth, ground_truth_distances,
                    metric, neighbors, seed, description, no_dedup, no_zero_check,
                    no_filtered, normalize, force,
                    base_convert_format: None,
                    query_convert_format: None,
                    compress_cache: false,
                    sized_profiles: None,
                    base_fraction,
                    required_facets,
                    round_digits,
                    pedantic_dedup,
                    selectivity: 0.0001,
                });
            }
        }
        PrepareCommand::Run(args) => {
            crate::pipeline::run_pipeline(args);
        }
        PrepareCommand::GenerateScript(args) => {
            crate::pipeline::run_script(args);
        }
        PrepareCommand::Check(args) => {
            crate::check::run(args);
        }
        PrepareCommand::Publish(args) => {
            crate::publish::run(args);
        }
        PrepareCommand::Stratify { path, spec, interactive, force, yes, auto } => {
            let interactive = interactive || auto;
            let yes = yes || auto;
            let force = force || auto;
            let spec = if spec.is_some() {
                spec
            } else if interactive && !yes {
                Some(stratify::interactive_spec_wizard(&path))
            } else {
                // --yes or --auto: use standard spec without prompting
                None // stratify::run handles this via the yes path
            };
            stratify::run(&path, spec.as_deref(), force, yes);
        }
        PrepareCommand::Catalog { command } => {
            match command {
                CatalogSubcommand::Generate { input, basename, for_publish_url, update } => {
                    crate::catalog::generate::run(&input, &basename, for_publish_url, update);
                }
            }
        }
        PrepareCommand::CacheGc { path, dry_run } => {
            cache_gc::run(&path, dry_run);
        }
        PrepareCommand::CacheCompress { cache_dir, level, dry_run } => {
            crate::pipeline::gz_cache::set_compression_level(level);
            cache_compress::run(&cache_dir, dry_run);
        }
        PrepareCommand::CacheUncompress { cache_dir, dry_run } => {
            cache_compress::run_uncompress(&cache_dir, dry_run);
        }
    }
}
