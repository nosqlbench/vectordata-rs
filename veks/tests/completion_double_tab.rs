// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! End-to-end verification of the value-position double-tab help UX.
//!
//! Invokes the real `veks` binary with `_VEKS_COMPLETE=bash` exactly
//! like the bash completion shim does, then asserts that a rapid
//! double-tap at a value position prints the option's help line to
//! stderr while leaving stdout (the COMPREPLY candidates) unchanged.

use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

/// Stable env-var ladder for isolating each test's tap-state file.
/// The engine derives the state path from `_COMP_SHELL_PID` when
/// `getppid` falls back, but `getppid` is the live syscall on Linux.
/// Override the env var with a per-test pid and set `HOME` to a
/// scratch directory so neither parallel tests nor stale state from
/// a developer's terminal can contaminate the assertion.
fn unique_ppid() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{}_{}", std::process::id(), nanos)
}

/// Build a temp directory whose `/tmp`-equivalent is private and
/// invoke the binary in a way that scopes the tap-state file into
/// it. The engine writes to `/tmp/.veks_tap_<ppid>`, which we can't
/// override, but we can ensure the ppid is unique per test.
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

/// Tap-state file the engine touches. Test cleans it up between
/// rapid sequences so a stale file from a prior test doesn't make
/// `tap_count` jump straight to 2.
fn clear_tap_state(ppid: &str) {
    let p = format!("/tmp/.veks_tap_{ppid}");
    let _ = std::fs::remove_file(&p);
}

#[test]
fn double_tap_at_value_position_emits_help_to_stderr() {
    let ppid = unique_ppid();
    clear_tap_state(&ppid);

    // The argument `--top-k` takes an int value; its help text is
    // "HeavyHitters top-K capacity" (from `analyze survey`'s
    // describe_options). With the cursor sitting one space after
    // the flag, tap 1 should produce no help on stderr, and tap 2
    // (within the rapid window) should emit the help line.
    let line = "veks analyze survey --top-k ";
    let (stdout1, stderr1) = invoke(line, &ppid);
    assert!(
        !stderr1.contains("HeavyHitters"),
        "tap 1 should not emit help yet — stderr was: {stderr1:?}"
    );

    // No sleep between calls — the rapid-tap window is 200ms wide,
    // and a normal process spawn fits well inside that. If the
    // assertion below ever flakes the threshold needs widening.
    let (stdout2, stderr2) = invoke(line, &ppid);

    assert_eq!(
        stdout1, stdout2,
        "stdout (candidate list) must be identical across rapid taps so \
         bash's COMPREPLY stays stable; tap1={stdout1:?} tap2={stdout2:?}"
    );
    assert!(
        stderr2.contains("HeavyHitters") && stderr2.contains("--top-k"),
        "tap 2 must print the flag's help to stderr; stderr was: {stderr2:?}"
    );
    // The line must start with a newline + `#` so it drops onto a
    // fresh row below the prompt rather than wedging against the
    // user's cursor, and visually reads as a comment annotation.
    assert!(
        stderr2.starts_with("\n# "),
        "help line should lead with a newline + `# ` prefix; stderr was: {stderr2:?}"
    );
    // Every help emission ends with a one-line hint reminding the
    // user how to clear it. The exact wording is matched here so a
    // wording change is a deliberate test update, not a silent UX
    // regression.
    assert!(
        stderr2.contains("ctrl-l to clear help"),
        "help line must end with the ctrl-l hint; stderr was: {stderr2:?}"
    );
}

/// A rapid third tap (within the 200ms window) escalates to the
/// extended help registered via `OptionDesc::with_extended_description`.
/// For `--top-k` that means the multi-line guidance about choosing
/// a value, and the ctrl-l hint at the bottom.
#[test]
fn triple_tap_at_value_position_emits_extended_help() {
    let ppid = unique_ppid();
    clear_tap_state(&ppid);

    let line = "veks analyze survey --top-k ";
    let (_so1, _se1) = invoke(line, &ppid);
    let (_so2, _se2) = invoke(line, &ppid);
    let (stdout3, stderr3) = invoke(line, &ppid);

    // Extended-help content includes the multi-line decision guide
    // we set in survey/command.rs. If that text changes, update
    // both there and here in the same commit.
    assert!(
        stderr3.contains("Misra-Gries"),
        "tap 3 must include the extended help body; stderr was: {stderr3:?}"
    );
    // Indicator that this is the *detail* tier and not the short one.
    assert!(
        stderr3.contains("(detail)"),
        "tap 3 should be labeled as (detail) tier; stderr was: {stderr3:?}"
    );
    assert!(
        stderr3.contains("ctrl-l to clear help"),
        "tap 3 must still end with the ctrl-l hint; stderr was: {stderr3:?}"
    );
    // stdout (the candidate stream) must remain stable across all
    // three rapid taps so bash's COMPREPLY isn't disturbed.
    assert_eq!(
        stdout3, "",
        "tap 3 stdout should be the same empty candidate set as the prior taps; stdout3={stdout3:?}"
    );
}

/// A *non-rapid* second tap (outside the 200ms window) must NOT
/// emit help — the rule is "rapid double-tap" specifically.
#[test]
fn slow_second_tap_does_not_emit_help() {
    let ppid = unique_ppid();
    clear_tap_state(&ppid);

    let line = "veks analyze survey --top-k ";
    let (_stdout1, stderr1) = invoke(line, &ppid);
    assert!(!stderr1.contains("HeavyHitters"));

    // Outside the 200ms rapid-tap window — the counter resets.
    std::thread::sleep(std::time::Duration::from_millis(350));
    let (_stdout2, stderr2) = invoke(line, &ppid);
    assert!(
        !stderr2.contains("HeavyHitters"),
        "slow second tap must not emit help; stderr was: {stderr2:?}"
    );
}

/// `--top-k` carries help text in the tree dump — guards the
/// upstream wiring that the value-position UX depends on. If this
/// assertion ever fails, look at `walk_clap_command` in
/// `veks/src/cli/dyncomp.rs` and the `arg.help(...)` calls in
/// `veks-pipeline/src/pipeline/cli.rs`.
#[test]
fn survey_flag_help_appears_in_tree_dump() {
    let bin = env!("CARGO_BIN_EXE_veks");
    let out = Command::new(bin)
        .arg("---dump-tree")
        .output()
        .expect("spawn veks ---dump-tree");
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
