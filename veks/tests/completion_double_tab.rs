// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end checks for the value-position double-tab help UX that are
//! **deterministic** — no wall-clock timing race.
//!
//! The tier table (tap 2 → short help, tap 3 → extended help, else
//! nothing) and the window→tap-count derivation are unit-tested directly
//! in `veks-completion` (`value_position_help_tier_table` and
//! `value_position_rapid_tap_reaches_tap_count_two` / `next_tap_state`),
//! where they can't flake. The earlier wall-clock E2E tests — which raced
//! two real `veks` process spawns against the 200ms rapid-tap window and
//! flaked under parallel load — were retired in favour of that unit
//! coverage. What remains here are the genuinely deterministic E2E
//! assertions: the window *reset* (a non-rapid second tap shows no help)
//! and the upstream tree wiring (the flag advertises its help).

use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

/// Per-test tap-state isolation: the engine derives the state path from
/// `_COMP_SHELL_PID`, so a unique value keeps tests from contaminating
/// each other's state file.
fn unique_ppid() -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    format!("{}_{}", std::process::id(), nanos)
}

/// Invoke the real `veks` binary the way the bash completion shim does.
fn invoke(line: &str, ppid: &str) -> (String, String) {
    let bin = env!("CARGO_BIN_EXE_veks");
    let point = line.len().to_string();
    let out = Command::new(bin)
        .arg(line)
        .arg(&point)
        .env("_VEKS_COMPLETE", "bash")
        .env("_COMP_SHELL_PID", ppid)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn veks");
    (
        String::from_utf8(out.stdout).expect("stdout utf8"),
        String::from_utf8(out.stderr).expect("stderr utf8"),
    )
}

fn clear_tap_state(ppid: &str) {
    let _ = std::fs::remove_file(format!("/tmp/.veks_tap_{ppid}"));
}

/// A *non-rapid* second tap (well outside the rapid window) must NOT emit
/// help — the rule is "rapid double-tap" specifically. This stays an E2E
/// test because it's robust to load: a 350ms sleep only ever *widens* the
/// gap, so a busy machine can't turn a no-help into a false help.
#[test]
fn slow_second_tap_does_not_emit_help() {
    let ppid = unique_ppid();
    clear_tap_state(&ppid);

    let line = "veks analyze survey --top-k ";
    let (_stdout1, stderr1) = invoke(line, &ppid);
    assert!(!stderr1.contains("HeavyHitters"));

    std::thread::sleep(std::time::Duration::from_millis(350));
    let (_stdout2, stderr2) = invoke(line, &ppid);
    assert!(
        !stderr2.contains("HeavyHitters"),
        "slow second tap must not emit help; stderr was: {stderr2:?}"
    );
}

/// `--top-k` carries help text in the tree dump — guards the upstream
/// wiring the value-position UX depends on. Deterministic (single spawn,
/// no timing). If this fails, look at `walk_clap_command` in
/// `veks/src/cli/dyncomp.rs` and the `arg.help(...)` calls in
/// `veks-pipeline/src/pipeline/cli.rs`.
#[test]
fn survey_flag_help_appears_in_tree_dump() {
    let bin = env!("CARGO_BIN_EXE_veks");
    let out = Command::new(bin).arg("---dump-tree").output().expect("spawn veks ---dump-tree");
    let stdout = String::from_utf8(out.stdout).expect("stdout utf8");
    let survey_line = stdout
        .lines()
        .find(|l| l.starts_with("/analyze/survey "))
        .unwrap_or_else(|| panic!("no /analyze/survey line in dump:\n{stdout}"));
    assert!(
        survey_line.contains("\"--top-k\""),
        "survey leaf should advertise --top-k flag: {survey_line}"
    );
    assert!(
        survey_line.contains("flag_help=") && survey_line.contains("\"--top-k\""),
        "survey leaf should carry flag_help for --top-k: {survey_line}"
    );
}
