// Copyright 2026 nosqlbench contributors
// SPDX-License-Identifier: Apache-2.0

//! `slab get` subcommand — retrieve records by ordinal.

use std::io::{self, Write};

use crate::SlabReader;

use super::ordinal_range;

/// Run the `get` subcommand.
///
/// Retrieves records by ordinal. Each element of `ordinals` can be a
/// plain integer or an ordinal range specifier (see [`ordinal_range`]).
/// Output format is controlled by `raw`, `as_hex`, and `as_base64`.
pub fn run(
    file: &str,
    ordinals: &[String],
    raw: bool,
    as_hex: bool,
    as_base64: bool,
    namespace: &Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = SlabReader::open_namespace(file, namespace.as_deref())?;

    // Resolve ordinals: each argument is either a single ordinal or a range
    let mut resolved: Vec<i64> = Vec::new();
    for arg in ordinals {
        if let Ok(n) = arg.parse::<i64>() {
            // Plain integer — treat as a single ordinal (not a count)
            resolved.push(n);
        } else {
            // Try as a range specifier
            let (start, end) = ordinal_range::parse_ordinal_range(arg)?;
            for o in start..end {
                resolved.push(o);
            }
        }
    }

    for &ordinal in &resolved {
        let data = reader.get(ordinal)?;

        if raw {
            io::stdout().write_all(&data)?;
        } else if as_hex {
            let hex: Vec<String> = data.iter().map(|b| format!("{b:02x}")).collect();
            println!("{}", hex.join(" "));
        } else if as_base64 {
            println!("{}", base64_encode(&data));
        } else {
            println!("ordinal {ordinal} ({} bytes):", data.len());
            hex_dump(&data);
            println!();
        }
    }

    Ok(())
}

/// Simple base64 encoder (standard alphabet, with padding).
pub fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        result.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(ALPHABET[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Print a hex dump of data: 16-byte lines with offset, hex, and ASCII sidebar.
fn hex_dump(data: &[u8]) {
    for (i, chunk) in data.chunks(16).enumerate() {
        let offset = i * 16;
        print!("  {offset:08x}  ");

        // Hex bytes
        for (j, byte) in chunk.iter().enumerate() {
            if j == 8 {
                print!(" ");
            }
            print!("{byte:02x} ");
        }

        // Padding for short last line
        let padding = 16 - chunk.len();
        for _ in 0..padding {
            print!("   ");
        }
        if chunk.len() <= 8 {
            print!(" ");
        }

        // ASCII sidebar
        print!(" |");
        for byte in chunk {
            if byte.is_ascii_graphic() || *byte == b' ' {
                print!("{}", *byte as char);
            } else {
                print!(".");
            }
        }
        println!("|");
    }
}
