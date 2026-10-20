// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! The explorer's base open, through the compiled binary (SRD element
//! access EA-13, §13.10): a base the conversion table refuses as f32
//! is reported by both type names before the terminal is taken over,
//! and a base of a narrower float type is browsed with its values
//! widened — the unit tests in `src/explore/shared.rs` cover the open
//! itself for every type the table widens.

mod support;
#[cfg(unix)]
mod tui_common;

use std::path::Path;
use std::process::{Command, Output};

use support::fixtures::write_xvec;
use vectordata::typed_access::{Container, Element, ElementConversion, ElementType};

/// An isolated client home with a cache and settings.yaml, so the
/// binary never reads the real configuration.
fn init_home() -> tempfile::TempDir {
    let home = tempfile::tempdir().unwrap();
    let cache = home.path().join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(home.path().join("settings.yaml"), format!("cache_dir: {}\n", cache.display())).unwrap();
    home
}

/// Run `vectordata explore --source <path>` in `home`, with no terminal
/// attached: the open happens before raw mode, so a refused base is
/// reported on stderr and the process exits.
fn explore_source(home: &Path, path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vectordata"))
        .args(["explore", "--source"])
        .arg(path)
        .env("VECTORDATA_HOME", home)
        .env("VECTORDATA_NO_UPDATE_CHECK", "1")
        .env("CI", "1")
        .output()
        .expect("run vectordata explore")
}

/// **A base the table refuses as f32 is reported by name, not
/// reinterpreted and not a panic**: for every such type — i32 among
/// them — `vectordata explore --source <file>` exits with status 1 and
/// says on stderr which type the file holds and that f32 was asked.
#[test]
fn a_base_the_table_refuses_as_f32_is_reported_by_both_type_names() {
    let home = init_home();
    let dir = tempfile::tempdir().unwrap();
    let refused: Vec<ElementType> = ElementType::ALL
        .iter()
        .copied()
        .filter(|ty| ty.conversion_to(ElementType::F32) == ElementConversion::Refused)
        .collect();
    assert!(refused.contains(&ElementType::I32), "an i32 base is one the explorer cannot show as f32");
    for ty in refused {
        let ext = ty.legacy_extension(Container::Vec).unwrap_or(ty.extension(Container::Vec));
        let path = dir.path().join(format!("base.{ext}"));
        // Any bytes of the right layout: the refusal is by extension,
        // before a value is read.
        veks_datatypes::dispatch_element!(ty, T => write_xvec::<T>(&path, 3, 2, |i, d| {
            T::from_value(veks_datatypes::Value::Int((i * 2 + d) as i128)).unwrap()
        }));
        let out = explore_source(home.path(), &path);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{ext}: a refused open exits 1, stderr:\n{stderr}");
        assert!(!stderr.contains("panicked"), "{ext}: the refusal is reported, not a panic:\n{stderr}");
        assert!(stderr.contains("failed to open"), "{ext}: the report names the open:\n{stderr}");
        assert!(
            stderr.contains(&format!("holds {ty} elements")),
            "{ext}: the report names the file's type:\n{stderr}"
        );
        assert!(stderr.contains("cannot be read as f32"), "{ext}: the report names the requested type:\n{stderr}");
    }
}

#[cfg(unix)]
mod pty {
    //! The frame the explorer renders over a narrow-float base, driven
    //! in a headless PTY as `tests/explore_tui.rs` drives the picker.

    use shadow_terminal::shadow_terminal::Config;
    use shadow_terminal::steppable_terminal::{Input, SteppableTerminal};

    use super::support::fixtures::write_xvec;
    use super::tui_common::init_home;
    use vectordata::typed_access::{bf16, f16, ElementType};

    const RECORDS: usize = 8;
    const DIM: usize = 4;

    /// Element `d` of record `i`: halves along the records, sixteens
    /// along the dimensions — exact in f16 and bf16, and distinct in
    /// every dimension so a reinterpretation could not reproduce them.
    fn value(i: usize, d: usize) -> f32 {
        i as f32 * 0.5 + d as f32 * 16.0
    }

    /// Launch `vectordata explore --source <path> --sample 8` in a PTY
    /// against `home` (inherited through `$VECTORDATA_HOME`).
    async fn start_explore_source(home: &std::path::Path, path: &std::path::Path) -> SteppableTerminal {
        unsafe {
            std::env::set_var("VECTORDATA_HOME", home);
            std::env::remove_var("VECTORDATA_TOKEN");
            std::env::set_var("CI", "1");
        }
        let bin = env!("CARGO_BIN_EXE_vectordata");
        let cfg = Config {
            width: 110,
            height: 40,
            command: vec![
                bin.into(),
                "explore".into(),
                "--source".into(),
                path.as_os_str().to_owned(),
                "--sample".into(),
                RECORDS.to_string().into(),
            ],
            scrollback_size: 1000,
            scrollback_step: 100,
        };
        SteppableTerminal::start(cfg).await.expect("start vectordata explore in a PTY")
    }

    /// Open `file` and check the title names the file, its dimension
    /// and the full count, then switch to the per-dimension view and
    /// check the statistics of dimension 0 are those of the written
    /// values: 0, 0.5, ..., 3.5.
    async fn browse(home: &std::path::Path, file: &std::path::Path) {
        let name = file.file_name().unwrap().to_string_lossy().into_owned();
        let mut term = start_explore_source(home, file).await;
        term.wait_for_string(&format!("Explore: {name} — {DIM} dims"), Some(15_000))
            .await
            .unwrap_or_else(|e| panic!("{name}: the title names the file and its dimension: {e}"));
        term.wait_for_string(&format!("{RECORDS}/{RECORDS} vectors"), Some(15_000))
            .await
            .unwrap_or_else(|e| panic!("{name}: every record loads: {e}"));
        // Tab → F2:DimDist, whose stats line is over the loaded values.
        term.send_input(Input::Event("\t".to_string())).expect("send Tab");
        term.wait_for_string("dim[0]: mean=1.750000 std=1.145644 min=0.000000 max=3.500000", Some(15_000))
            .await
            .unwrap_or_else(|e| {
                let _ = term.dump_screen();
                panic!("{name}: dimension 0 holds the written values widened: {e}")
            });
        term.send_input(Input::Event("\u{03}".to_string())).expect("send Ctrl-C");
        term.kill().expect("shut down the PTY");
    }

    /// **The explorer opens an f16 and a bf16 base** (EA-13): each is
    /// browsed under its own extension, with the full count and
    /// dimension in the title and the per-dimension statistics of the
    /// values as written — widened, not reinterpreted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_explorer_opens_an_f16_and_a_bf16_base() {
        let home = init_home();
        let dir = tempfile::tempdir().unwrap();
        let half = dir.path().join("base.mvecs");
        write_xvec::<f16>(&half, RECORDS, DIM, |i, d| f16::from_f32(value(i, d)));
        assert_eq!(ElementType::from_extension("mvecs"), Some(ElementType::F16));
        browse(home.path(), &half).await;

        let brain = dir.path().join("base.bf16vecs");
        write_xvec::<bf16>(&brain, RECORDS, DIM, |i, d| bf16::from_f32(value(i, d)));
        assert_eq!(ElementType::from_extension("bf16vecs"), Some(ElementType::BF16));
        browse(home.path(), &brain).await;
    }
}
