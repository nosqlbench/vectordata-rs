// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! `vectordata explore` — interactive visualization and exploration.
//!
//! Owns the ratatui-based dataset browser, raw-values grid, and REPL
//! command engine. Originally lived in `veks/src/explore/`; migrated
//! into vectordata to make `vectordata explore` self-contained — the
//! TUI never had a real pipeline-command-framework dependency (the
//! wiring that looked like one was dead code).
//!
//! The source argument accepts either a local file path or a
//! `dataset:profile:facet` specifier from the catalog.

pub mod shared;
mod dataset_picker;
pub(crate) mod palette;
mod unified;

/// Resolve the configured cache directory or exit the process with a
/// helpful error message. Used as the entry-point fallback when the
/// explore TUI needs to know where remote-cache blobs live and has
/// nowhere sensible to proceed without one.
pub(crate) fn cache_dir_or_exit() -> std::path::PathBuf {
    match crate::settings::cache_dir() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: cannot resolve cache_dir from settings: {e}");
            std::process::exit(1);
        }
    }
}

/// Deterministic seeded RNG used by sampling code in the explore
/// TUI. Mirrors the `veks-pipeline::rng::seeded_rng` shape that the
/// pre-migration code called into — same xoshiro256++ generator, so
/// any test that pins a sample under a given seed continues to
/// produce identical output.
pub(crate) fn seeded_rng(seed: u64) -> rand_xoshiro::Xoshiro256PlusPlus {
    // SeedableRng is re-exported by rand_xoshiro itself so this works
    // without a separate `rand` dependency in vectordata's
    // (non-dev) deps.
    use rand_xoshiro::rand_core::SeedableRng;
    rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed)
}

pub use shared::SampleMode;

/// Session UI theme, derived from the active (palette, curve) pair.
/// Initialized from settings/flags by [`run`] and mutable at runtime
/// — the picker's settings screen (Ctrl-G) and the explorer's
/// `p`/`f` cycles write through here so one pair themes everything
/// live. Consumers that render before initialization get the
/// resolution chain (settings → project standard).
#[allow(clippy::type_complexity)]
static THEME_STATE: std::sync::RwLock<Option<(palette::Palette, palette::Curve, palette::Theme)>> =
    std::sync::RwLock::new(None);

pub(crate) fn theme() -> palette::Theme {
    if let Some((_, _, t)) = *THEME_STATE.read().unwrap() {
        return t;
    }
    let (p, c) = resolve_palette_curve(None, None);
    set_theme(p, c);
    palette::Theme::derive(p, c)
}

/// Replace the session theme. The derived chrome is cached alongside
/// the pair so `theme()` is a lock-read + copy.
pub(crate) fn set_theme(p: palette::Palette, c: palette::Curve) {
    *THEME_STATE.write().unwrap() = Some((p, c, palette::Theme::derive(p, c)));
}

/// The active (palette, curve) pair, initializing on first use.
pub(crate) fn theme_palette_curve() -> (palette::Palette, palette::Curve) {
    if let Some((p, c, _)) = *THEME_STATE.read().unwrap() {
        return (p, c);
    }
    let (p, c) = resolve_palette_curve(None, None);
    set_theme(p, c);
    (p, c)
}

/// Persist the active (palette, curve) pair to `settings.yaml` as the
/// user's default theme — the "save theme" action on the picker's
/// settings screen (Ctrl-G) and behind Ctrl-S in the explorer.
/// Comment-preserving line edits.
pub(crate) fn save_theme_to_settings() -> Result<std::path::PathBuf, String> {
    let (p, c) = theme_palette_curve();
    crate::settings::write_setting("palette", p.name())?;
    crate::settings::write_setting("curve", c.name())
}

/// "Reset Display Options": remove the display-related settings keys
/// (`palette`, `curve`, `disabled_columns`) so the project standard
/// applies again, and reset the session theme to match. Catalog
/// enablement is data scope, not display — deliberately untouched.
pub(crate) fn reset_display_options() -> Result<(), String> {
    crate::settings::remove_setting("palette")?;
    crate::settings::remove_setting("curve")?;
    crate::settings::remove_setting("disabled_columns")?;
    set_theme(palette::Palette::default(), palette::Curve::default());
    Ok(())
}

/// Resolve the (palette, curve) pair: CLI flag > settings.yaml key >
/// project standard. Unknown names warn and fall through so a typo'd
/// setting degrades to the standard instead of breaking the UI.
pub(crate) fn resolve_palette_curve(
    flag_palette: Option<&str>,
    flag_curve: Option<&str>,
) -> (palette::Palette, palette::Curve) {
    let parse_p = |s: &str, origin: &str| {
        palette::Palette::parse(s).or_else(|| {
            eprintln!("warning: unknown palette '{s}' ({origin}); valid: {}. Using standard.",
                palette::ALL_PALETTES.iter().map(|p| p.name()).collect::<Vec<_>>().join(", "));
            None
        })
    };
    let parse_c = |s: &str, origin: &str| {
        palette::Curve::parse(s).or_else(|| {
            eprintln!("warning: unknown curve '{s}' ({origin}); valid: {}. Using standard.",
                palette::ALL_CURVES.iter().map(|c| c.name()).collect::<Vec<_>>().join(", "));
            None
        })
    };
    let p = flag_palette.and_then(|s| parse_p(s, "--palette"))
        .or_else(|| crate::settings::setting_value("palette")
            .and_then(|s| parse_p(&s, "settings.yaml")))
        .unwrap_or_default();
    let c = flag_curve.and_then(|s| parse_c(s, "--curve"))
        .or_else(|| crate::settings::setting_value("curve")
            .and_then(|s| parse_c(&s, "settings.yaml")))
        .unwrap_or_default();
    (p, c)
}

/// Unified vector space explorer — norms, distances, eigenvalues, PCA
/// in one TUI. Run without any source flag to pop the catalog picker.
#[derive(veks_completion_derive::VeksCli)]
pub struct ExploreArgs {
    /// Dataset from catalog (e.g., img-search or img-search:default)
    #[arg(long, group = "input")]
    pub dataset: Option<String>,
    /// Any data source: local file path or dataset:profile:facet
    #[arg(long, group = "input")]
    pub source: Option<String>,
    /// Profile name (used with --dataset; overrides profile in dataset:profile)
    #[arg(long)]
    pub profile: Option<String>,
    /// Number of vectors to sample
    #[arg(long, default_value = "50000")]
    pub sample: usize,
    /// Random seed
    #[arg(long, default_value = "42")]
    pub seed: u64,
    /// Sampling mode [clumped, streaming, sparse]. Clumped is the
    /// default: whole transfer chunks, evenly spaced across the file,
    /// so a remote sample covers the full distribution while wasting
    /// none of the downloaded blocks. Streaming reads the head of the
    /// file (fastest, distribution-biased); sparse is fully random
    /// (best coverage, one chunk download per vector when remote).
    #[arg(long, default_value = "clumped", value_parser = ["streaming", "clumped", "sparse"])]
    pub sample_mode: SampleMode,
    /// Color palette for visualization AND the UI theme (overrides
    /// the `palette:` settings key; standard: turbo).
    #[arg(long)]
    pub palette: Option<String>,
    /// Intensity curve for visualization AND the UI theme (overrides
    /// the `curve:` settings key; standard: square).
    #[arg(long)]
    pub curve: Option<String>,
}

/// Resolve the data source from mutually exclusive --dataset / --source options.
///
/// When `--profile` is given with `--dataset`, it's appended as `dataset:profile`.
/// If the dataset already contains a `:`, the explicit `--profile` overrides it.
fn resolve_input(dataset: Option<String>, source: Option<String>, profile: Option<String>) -> Option<String> {
    let base = match (dataset, source) {
        (Some(ds), None) => ds,
        (None, Some(src)) => src,
        (None, None) => return None,
        (Some(_), Some(_)) => {
            eprintln!("vectordata explore: --dataset and --source are mutually exclusive — pass only one");
            std::process::exit(2);
        }
    };

    Some(match profile {
        Some(p) => {
            let name = base.split(':').next().unwrap_or(&base);
            format!("{}:{}", name, p)
        }
        None => {
            if !base.contains(':') && !base.contains('/') && !base.contains('.') {
                format!("{}:default", base)
            } else {
                base
            }
        }
    })
}

/// Launch the unified explore TUI. When no source is supplied, the
/// catalog picker pops first; each picker selection routes through a
/// per-row action menu (visualize / precache / purge / ping). Actions
/// run in-place with the picker's chrome temporarily suspended so the
/// picker's UI state — cursor, expanded set, filter, scroll, last
/// menu cursor — is preserved across every action.
pub fn run(args: ExploreArgs) -> i32 {
    use dataset_picker::{ActionFlow, PickerAction, PickerOutcome};

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(
            std::io::stdout(),
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::cursor::Show,
        );
        original_hook(info);
    }));

    let ExploreArgs { dataset, source, profile, sample, seed, sample_mode, palette: palette_flag, curve: curve_flag } = args;
    // Resolve the theme once for the whole session — the picker's
    // chrome and the explorer's data colors derive from the same
    // (palette, curve) pair.
    let (resolved_palette, resolved_curve) =
        resolve_palette_curve(palette_flag.as_deref(), curve_flag.as_deref());
    set_theme(resolved_palette, resolved_curve);

    // Non-interactive path: explicit source or dataset → straight to
    // the explorer, no picker, no menu.
    if dataset.is_some() || source.is_some() {
        let src = resolve_input(dataset, source, profile)
            .expect("guarded above: at least one of --dataset/--source is set here");
        return match unified::run_interactive_explore(&src, sample, seed, sample_mode, resolved_palette, resolved_curve) {
            unified::ExploreExit::Quit | unified::ExploreExit::Back => 0,
        };
    }

    // Interactive path: picker owns the loop and dispatches actions
    // inline. Visualize → Quit exits the picker too; Visualize → Back
    // keeps the picker open. locate/download/purge/ping always return
    // to the picker so the user can chain operations.
    let dispatch = |specifier: &str, action: PickerAction, pause: bool| -> ActionFlow {
        match action {
            PickerAction::Visualize => {
                // Interactive viewer — `pause` is irrelevant; the
                // viewer owns the terminal until the user exits it.
                let _ = pause;
                match unified::run_interactive_explore(specifier, sample, seed, sample_mode, resolved_palette, resolved_curve) {
                    unified::ExploreExit::Quit => ActionFlow::Exit,
                    unified::ExploreExit::Back => ActionFlow::Stay,
                }
            }
            PickerAction::Locate => {
                run_locate(specifier, pause);
                ActionFlow::Stay
            }
            PickerAction::Download => {
                run_precache(specifier, pause);
                ActionFlow::Stay
            }
            PickerAction::Purge => {
                run_purge(specifier, pause);
                ActionFlow::Stay
            }
            PickerAction::Ping => {
                run_ping(specifier, pause);
                ActionFlow::Stay
            }
            PickerAction::Describe | PickerAction::Source => {
                // Picker-local — the action menu handles them
                // directly via `is_picker_local()` and never reaches
                // dispatch. These arms are unreachable in practice
                // but the match needs to be exhaustive.
                let _ = pause;
                ActionFlow::Stay
            }
        }
    };
    match dataset_picker::run_picker(dispatch) {
        PickerOutcome::Done => 0,
        PickerOutcome::Failed => 1,
    }
}

/// The Locate action: print the dataset's cache location with the
/// picker's chrome suspended, then return to the picker (pausing
/// after the last item like every other action). Output is
/// shell-friendly — a `#`-commented header per dataset and either
/// the bare path or a commented not-cached note, so real paths
/// filter out with `grep -v '^#'`.
fn run_locate(specifier: &str, pause: bool) {
    for line in locate_lines(specifier, &cache_dir_or_exit()) {
        println!("{line}");
    }
    if pause { pause_for_keypress(); }
}

/// Pure body of [`run_locate`].
fn locate_lines(specifier: &str, cache_root: &std::path::Path) -> Vec<String> {
    let (dataset, _profile) = split_specifier(specifier);
    let dir = cache_root.join(dataset);
    vec![
        format!("# location of {dataset} on local system:"),
        if dir.is_dir() {
            dir.display().to_string()
        } else {
            format!("# {dataset}: not cached")
        },
    ]
}

/// Split a `dataset:profile` specifier. Profile defaults to `default`.
fn split_specifier(specifier: &str) -> (&str, &str) {
    match specifier.split_once(':') {
        Some((d, p)) if !p.is_empty() => (d, p),
        _ => (specifier, "default"),
    }
}

/// Pause for a keystroke so the user can read the action's stderr
/// output before the picker re-takes the screen.
fn pause_for_keypress() {
    eprintln!();
    eprintln!("Press Enter to return to the picker…");
    let mut buf = String::new();
    let _ = std::io::stdin().read_line(&mut buf);
}

fn run_precache(specifier: &str, pause: bool) {
    // Picker-initiated precache drives the *highlighted profile only*.
    // Earlier this stripped the `:profile` suffix and walked every
    // profile via `ProfileSelection::AllProfiles`, but for datasets
    // with many sized profiles (e.g. `example-1b` has ~130)
    // that's two pathologies at once:
    //   1. `plan_prebuffer` opens FacetStorage for every facet of
    //      every profile to learn its total_size, which means O(130
    //      × ~5) `.mref` HEAD probes serialised behind the planner
    //      before any progress line prints — looked like the
    //      precache had hung.
    //   2. `total_bytes` double-counts every windowed profile against
    //      the same underlying file, producing wildly inflated
    //      headlines ("169 TiB to download" for a 1.3 TiB dataset).
    // Sized/windowed profiles share base URLs, so the registry +
    // cache dedup means precaching the default already fills the
    // shared file for every window. Partition profiles have their
    // own bytes and need to be precached explicitly — but that's
    // what the picker rows are for: the user highlights the partition
    // profile and runs precache against it directly.
    let code = crate::datasets::precache::run(
        specifier,
        &crate::catalog::sources::config_dir(),
        &[],
        &[],
        None,
    );
    if code != 0 {
        eprintln!("(precache exited with status {code})");
    }
    if pause { pause_for_keypress(); }
}

fn run_purge(specifier: &str, pause: bool) {
    let (dataset, _profile) = split_specifier(specifier);
    // Purge is per-dataset: every cache leaf whose origin URL belongs
    // to ANY facet of ANY profile is removed. The profile component of
    // the specifier is ignored on purpose — the runtime's cache is
    // content-addressed by URL, not by profile, so per-profile purge
    // can't actually exist without re-introducing the dataset-named
    // layout we already moved away from.
    let sources = crate::catalog::sources::CatalogSources::new().configure_default();
    let catalog = crate::catalog::resolver::Catalog::of(&sources);
    let entry = match catalog.datasets().iter().find(|e| e.name == dataset) {
        Some(e) => e.clone(),
        None => {
            eprintln!("error: dataset '{dataset}' not found in any configured catalog");
            if pause { pause_for_keypress(); }
            return;
        }
    };
    let cache_dir = crate::settings::cache_dir().unwrap_or_else(|e| {
        eprintln!("error: cannot resolve cache_dir: {e}");
        std::process::exit(1);
    });
    let (removed, freed, skipped) = dataset_picker::purge_cache_for_entry(&entry, &cache_dir);
    for note in &skipped {
        eprintln!("skipped: {note}");
    }
    if removed.is_empty() {
        println!("No cached entries found for '{dataset}'.");
    } else {
        println!("Purged {} cache entr{} for '{dataset}' ({}):",
            removed.len(),
            if removed.len() == 1 { "y" } else { "ies" },
            format_bytes_short(freed));
        for path in &removed {
            println!("  - {}", path.display());
        }
    }
    if pause { pause_for_keypress(); }
}

pub(crate) fn format_bytes_short(n: u64) -> String {
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    const KIB: u64 = 1 << 10;
    if n >= GIB { format!("{:.1} GiB", n as f64 / GIB as f64) }
    else if n >= MIB { format!("{:.1} MiB", n as f64 / MIB as f64) }
    else if n >= KIB { format!("{:.1} KiB", n as f64 / KIB as f64) }
    else { format!("{n} B") }
}

fn run_ping(specifier: &str, pause: bool) {
    let (dataset, profile) = split_specifier(specifier);
    // Use the same union catalog the picker built its row list from
    // so a ping from inside the picker hits exactly the catalogs the
    // user can see.
    let sources = crate::catalog::sources::CatalogSources::new().configure_default();
    let catalog = crate::catalog::resolver::Catalog::of(&sources);
    let code = crate::datasets::ping::run_via_catalog(&catalog, dataset, profile);
    if code != 0 {
        eprintln!("(ping exited with status {code})");
    }
    if pause { pause_for_keypress(); }
}

// `run_pipeline_command` lived here as `#[allow(dead_code)]`
// scaffolding for invoking pipeline `CommandOp` instances from the
// explore TUI. It was the only consumer of `veks-pipeline`'s
// `StreamContext` / `ProgressLog` / `ResourceGovernor` / `ui::*`
// inside this module, and nothing ever called it. It was deleted
// during the migration into vectordata — explore is now leaf-crate
// code and has no business reaching back up into the pipeline
// command framework. Re-add a vectordata-local equivalent if a
// future need surfaces.

#[cfg(test)]
mod tests {
    use super::locate_lines;

    /// Locate prints a commented per-dataset header, then the cache
    /// directory when present or a `#`-commented note when not — so
    /// shell consumers can filter real paths with `grep -v '^#'`.
    #[test]
    fn locate_lines_header_then_path_or_note() {
        let tmp = tempfile::tempdir().unwrap();
        let cached = tmp.path().join("ds-a");
        std::fs::create_dir_all(&cached).unwrap();

        assert_eq!(locate_lines("ds-a:default", tmp.path()), vec![
            "# location of ds-a on local system:".to_string(),
            cached.display().to_string(),
        ]);
        // Profile suffix is irrelevant — the cache is dataset-keyed.
        assert_eq!(locate_lines("ds-a:100k", tmp.path())[1], cached.display().to_string());
        assert_eq!(locate_lines("ds-b:default", tmp.path()), vec![
            "# location of ds-b on local system:".to_string(),
            "# ds-b: not cached".to_string(),
        ]);
    }
}
