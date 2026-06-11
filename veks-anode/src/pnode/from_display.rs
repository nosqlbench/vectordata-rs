// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Reverse direction of the PNode Display renderer (`PNode::Display`
//! in `pnode::mod`): parse a Display-format string back into a
//! concrete [`PNode`].
//!
//! The grammar is the one [`PNode::Display`] emits:
//!
//! ```text
//! pnode      := conjugate | predicate
//! conjugate  := '(' pnode (' AND ' pnode)+ ')'
//!             | '(' pnode (' OR '  pnode)+ ')'
//! predicate  := field ' ' op_symbol ' ' comparand_list
//! field      := bare_name | 'field[' digits ']'
//! op_symbol  := '>' | '<' | '=' | '!=' | '>=' | '<=' | 'IN' | 'MATCHES'
//! comparand_list := comparand | '(' comparand (', ' comparand)* ')'
//! comparand  := int | float | text | bool | bytes | 'NULL'
//! int        := optional_sign digits
//! float      := optional_sign digits '.' digits ('e' | 'E' optional_sign digits)?
//! text       := "'" content "'"            (no escape sequence handling — keep
//!                                            in sync with the renderer)
//! bool       := 'true' | 'false'
//! bytes      := "X'" hex_pairs "'"
//! ```
//!
//! The parser is intentionally strict: it accepts exactly what
//! the renderer emits and rejects everything else. A successful
//! round trip — `from_display(&node.to_string())` returning a
//! `PNode` equal to `node` — is the codec contract.
//!
//! Wrappers that need extensions (e.g. `?` placeholders for
//! generation templates) layer them on top of this parser at a
//! higher level rather than relaxing the grammar here.

use super::{Comparand, ConjugateNode, ConjugateType, FieldRef, OpType, PNode, PredicateNode};

/// Position-tagged parse error. `pos` is the byte offset into the
/// original input where the parser bailed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub msg: String,
    pub pos: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PNode parse error at byte {}: {}", self.pos, self.msg)
    }
}

impl std::error::Error for ParseError {}

/// Parse a Display-format PNode string. See module docs for the
/// grammar. Returns the parsed [`PNode`] or a position-tagged
/// error.
pub fn from_display(text: &str) -> Result<PNode, ParseError> {
    let mut p = Parser::new(text);
    let node = p.parse_node()?;
    p.skip_ws();
    if p.cursor < p.input.len() {
        return Err(p.err("trailing input after predicate tree"));
    }
    Ok(node)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parser internals
// ─────────────────────────────────────────────────────────────────────────────

struct Parser<'a> {
    input: &'a str,
    cursor: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, cursor: 0 }
    }

    fn err(&self, msg: &str) -> ParseError {
        ParseError { msg: msg.to_string(), pos: self.cursor }
    }

    fn rest(&self) -> &'a str { &self.input[self.cursor..] }

    fn peek_char(&self) -> Option<char> { self.rest().chars().next() }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() { self.cursor += c.len_utf8(); } else { break; }
        }
    }

    fn consume_char(&mut self, expected: char) -> Result<(), ParseError> {
        match self.peek_char() {
            Some(c) if c == expected => {
                self.cursor += c.len_utf8();
                Ok(())
            }
            Some(c) => Err(self.err(&format!("expected '{expected}', got '{c}'"))),
            None => Err(self.err(&format!("expected '{expected}', got EOF"))),
        }
    }

    fn try_consume_literal(&mut self, lit: &str) -> bool {
        if self.rest().starts_with(lit) {
            self.cursor += lit.len();
            true
        } else {
            false
        }
    }

    // ── Top-level node ────────────────────────────────────────────

    fn parse_node(&mut self) -> Result<PNode, ParseError> {
        self.skip_ws();
        if self.peek_char() == Some('(') {
            self.parse_conjugate_or_paren_predicate()
        } else {
            self.parse_predicate().map(PNode::Predicate)
        }
    }

    /// A leading `(` may begin either a conjugate (`(a AND b)`) or a
    /// parenthesised predicate (rare — `PNode::Display` doesn't emit
    /// that shape, but the grammar tolerates it). We disambiguate by
    /// parsing the first inner node, then looking for `AND`/`OR`.
    fn parse_conjugate_or_paren_predicate(&mut self) -> Result<PNode, ParseError> {
        self.consume_char('(')?;
        let first = self.parse_node()?;
        self.skip_ws();
        // If the next token is `AND` or `OR`, this is a conjugate.
        let conj_kind = if self.try_consume_keyword("AND") {
            Some(ConjugateType::And)
        } else if self.try_consume_keyword("OR") {
            Some(ConjugateType::Or)
        } else {
            None
        };
        match conj_kind {
            None => {
                // Just a parenthesised inner node.
                self.skip_ws();
                self.consume_char(')')?;
                Ok(first)
            }
            Some(kind) => {
                let mut children = vec![first];
                self.skip_ws();
                children.push(self.parse_node()?);
                loop {
                    self.skip_ws();
                    let kw = match kind {
                        ConjugateType::And => "AND",
                        ConjugateType::Or => "OR",
                        ConjugateType::Pred => unreachable!(),
                    };
                    if !self.try_consume_keyword(kw) { break; }
                    self.skip_ws();
                    children.push(self.parse_node()?);
                }
                self.skip_ws();
                self.consume_char(')')?;
                Ok(PNode::Conjugate(ConjugateNode {
                    conjugate_type: kind,
                    children,
                }))
            }
        }
    }

    fn try_consume_keyword(&mut self, kw: &str) -> bool {
        if !self.rest().starts_with(kw) { return false; }
        // Must be followed by whitespace, '(', or end — so that
        // `AND` doesn't eat the leading chars of e.g. a field
        // named `ANDREW`. (Field names in Display position can
        // be any non-whitespace word.)
        let after = &self.rest()[kw.len()..];
        let next = after.chars().next();
        let ok = matches!(next, None | Some(' ') | Some('\t') | Some('\n') | Some('('));
        if ok { self.cursor += kw.len(); }
        ok
    }

    // ── Leaf predicate ────────────────────────────────────────────

    fn parse_predicate(&mut self) -> Result<PredicateNode, ParseError> {
        let field = self.parse_field_ref()?;
        self.skip_ws();
        let op = self.parse_op()?;
        self.skip_ws();
        let comparands = self.parse_comparand_list()?;
        Ok(PredicateNode { field, op, comparands })
    }

    fn parse_field_ref(&mut self) -> Result<FieldRef, ParseError> {
        if self.rest().starts_with("field[") {
            self.cursor += "field[".len();
            let start = self.cursor;
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() { self.cursor += 1; } else { break; }
            }
            let digits = &self.input[start..self.cursor];
            if digits.is_empty() {
                return Err(self.err("expected digits after 'field['"));
            }
            let n: u8 = digits.parse()
                .map_err(|_| self.err("field index does not fit in u8"))?;
            self.consume_char(']')?;
            return Ok(FieldRef::Index(n));
        }
        // Bare name — any run of non-whitespace, non-operator chars.
        // The renderer emits the field name verbatim, so we accept
        // the same. Stops at the first space (the renderer always
        // emits a single space before the op).
        let start = self.cursor;
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() || c == '(' || c == ')' { break; }
            self.cursor += c.len_utf8();
        }
        if self.cursor == start {
            return Err(self.err("expected field name"));
        }
        Ok(FieldRef::Named(self.input[start..self.cursor].to_string()))
    }

    fn parse_op(&mut self) -> Result<OpType, ParseError> {
        // Longest-match order matters: `>=` before `>`, `<=` before `<`, `!=` before `!`.
        if self.try_consume_literal("!=") { return Ok(OpType::Ne); }
        if self.try_consume_literal(">=") { return Ok(OpType::Ge); }
        if self.try_consume_literal("<=") { return Ok(OpType::Le); }
        if self.try_consume_literal(">")  { return Ok(OpType::Gt); }
        if self.try_consume_literal("<")  { return Ok(OpType::Lt); }
        if self.try_consume_literal("=")  { return Ok(OpType::Eq); }
        if self.try_consume_keyword("IN") { return Ok(OpType::In); }
        if self.try_consume_keyword("MATCHES") { return Ok(OpType::Matches); }
        Err(self.err("expected operator (>, <, =, !=, >=, <=, IN, MATCHES)"))
    }

    fn parse_comparand_list(&mut self) -> Result<Vec<Comparand>, ParseError> {
        self.skip_ws();
        if self.peek_char() == Some('(') {
            self.consume_char('(')?;
            let mut out = Vec::new();
            self.skip_ws();
            if self.peek_char() == Some(')') {
                self.consume_char(')')?;
                return Ok(out);
            }
            out.push(self.parse_comparand()?);
            loop {
                self.skip_ws();
                if !self.try_consume_literal(",") { break; }
                self.skip_ws();
                out.push(self.parse_comparand()?);
            }
            self.skip_ws();
            self.consume_char(')')?;
            Ok(out)
        } else {
            Ok(vec![self.parse_comparand()?])
        }
    }

    fn parse_comparand(&mut self) -> Result<Comparand, ParseError> {
        self.skip_ws();
        let c = self.peek_char().ok_or_else(|| self.err("expected comparand, got EOF"))?;
        // NULL
        if self.rest().starts_with("NULL") {
            let after = &self.rest()["NULL".len()..];
            // Word boundary: followed by ws / ',' / ')' / end.
            let next = after.chars().next();
            if matches!(next, None | Some(' ') | Some('\t') | Some(',') | Some(')')) {
                self.cursor += "NULL".len();
                return Ok(Comparand::Null);
            }
        }
        // Booleans
        if self.try_consume_keyword_value("true") { return Ok(Comparand::Bool(true)); }
        if self.try_consume_keyword_value("false") { return Ok(Comparand::Bool(false)); }
        // Bytes: X'…'
        if self.rest().starts_with("X'") {
            return self.parse_bytes_literal();
        }
        // Text: '…'
        if c == '\'' {
            return self.parse_text_literal();
        }
        // Numbers: sign + digits with optional fractional / exponent.
        if c == '-' || c == '+' || c.is_ascii_digit() {
            return self.parse_number();
        }
        Err(self.err(&format!("unexpected character '{c}' in comparand")))
    }

    fn try_consume_keyword_value(&mut self, kw: &str) -> bool {
        if !self.rest().starts_with(kw) { return false; }
        let after = &self.rest()[kw.len()..];
        let next = after.chars().next();
        let ok = matches!(next, None | Some(' ') | Some('\t') | Some(',') | Some(')'));
        if ok { self.cursor += kw.len(); }
        ok
    }

    fn parse_bytes_literal(&mut self) -> Result<Comparand, ParseError> {
        // X'<hex pairs>'
        self.cursor += 2; // skip X'
        let start = self.cursor;
        while let Some(c) = self.peek_char() {
            if c == '\'' { break; }
            self.cursor += c.len_utf8();
        }
        let hex = &self.input[start..self.cursor];
        self.consume_char('\'')?;
        if !hex.len().is_multiple_of(2) {
            return Err(ParseError { msg: "odd-length hex literal".into(), pos: start });
        }
        let mut bytes = Vec::with_capacity(hex.len() / 2);
        let bytes_input = hex.as_bytes();
        for i in (0..hex.len()).step_by(2) {
            let hi = decode_hex_digit(bytes_input[i])
                .ok_or_else(|| ParseError { msg: "non-hex character".into(), pos: start + i })?;
            let lo = decode_hex_digit(bytes_input[i + 1])
                .ok_or_else(|| ParseError { msg: "non-hex character".into(), pos: start + i + 1 })?;
            bytes.push((hi << 4) | lo);
        }
        Ok(Comparand::Bytes(bytes))
    }

    fn parse_text_literal(&mut self) -> Result<Comparand, ParseError> {
        self.consume_char('\'')?;
        let start = self.cursor;
        while let Some(c) = self.peek_char() {
            if c == '\'' { break; }
            self.cursor += c.len_utf8();
        }
        let body = &self.input[start..self.cursor];
        self.consume_char('\'')?;
        Ok(Comparand::Text(body.to_string()))
    }

    fn parse_number(&mut self) -> Result<Comparand, ParseError> {
        let start = self.cursor;
        if matches!(self.peek_char(), Some('+') | Some('-')) {
            self.cursor += 1;
        }
        let int_start = self.cursor;
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() { self.cursor += 1; } else { break; }
        }
        if self.cursor == int_start {
            return Err(ParseError { msg: "expected digits".into(), pos: int_start });
        }
        let mut is_float = false;
        if self.peek_char() == Some('.') {
            is_float = true;
            self.cursor += 1;
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() { self.cursor += 1; } else { break; }
            }
        }
        if matches!(self.peek_char(), Some('e') | Some('E')) {
            is_float = true;
            self.cursor += 1;
            if matches!(self.peek_char(), Some('+') | Some('-')) {
                self.cursor += 1;
            }
            let exp_start = self.cursor;
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() { self.cursor += 1; } else { break; }
            }
            if self.cursor == exp_start {
                return Err(ParseError { msg: "expected exponent digits".into(), pos: exp_start });
            }
        }
        let slice = &self.input[start..self.cursor];
        if is_float {
            let v: f64 = slice.parse()
                .map_err(|_| ParseError { msg: "invalid float".into(), pos: start })?;
            Ok(Comparand::Float(v))
        } else {
            let v: i64 = slice.parse()
                .map_err(|_| ParseError { msg: "invalid integer".into(), pos: start })?;
            Ok(Comparand::Int(v))
        }
    }
}

fn decode_hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pred(field: &str, op: OpType, c: Comparand) -> PNode {
        PNode::Predicate(PredicateNode {
            field: FieldRef::Named(field.to_string()),
            op,
            comparands: vec![c],
        })
    }

    fn assert_round_trip(node: &PNode) {
        let s = node.to_string();
        let parsed = from_display(&s)
            .unwrap_or_else(|e| panic!("parse '{s}': {e}"));
        assert_eq!(&parsed, node, "round trip differs: rendered={s:?}");
    }

    #[test]
    fn round_trip_simple_int_eq() {
        assert_round_trip(&pred("age", OpType::Eq, Comparand::Int(42)));
    }

    #[test]
    fn round_trip_inequalities() {
        for op in [OpType::Gt, OpType::Lt, OpType::Ge, OpType::Le, OpType::Ne] {
            assert_round_trip(&pred("score", op, Comparand::Int(10)));
        }
    }

    #[test]
    fn round_trip_float() {
        assert_round_trip(&pred("ratio", OpType::Lt, Comparand::Float(3.25)));
        assert_round_trip(&pred("zero", OpType::Eq, Comparand::Float(0.0)));
        assert_round_trip(&pred("whole", OpType::Eq, Comparand::Float(1.0)));
    }

    #[test]
    fn round_trip_text_with_spaces() {
        assert_round_trip(&pred("name", OpType::Eq, Comparand::Text("alice bob".into())));
    }

    #[test]
    fn round_trip_text_matches() {
        assert_round_trip(&pred("blob", OpType::Matches, Comparand::Text("(^|, )foo(,|$)".into())));
    }

    #[test]
    fn round_trip_bool() {
        assert_round_trip(&pred("active", OpType::Eq, Comparand::Bool(true)));
        assert_round_trip(&pred("active", OpType::Eq, Comparand::Bool(false)));
    }

    #[test]
    fn round_trip_null() {
        assert_round_trip(&pred("missing", OpType::Eq, Comparand::Null));
    }

    #[test]
    fn round_trip_bytes() {
        assert_round_trip(&pred("blob", OpType::Eq, Comparand::Bytes(vec![0xde, 0xad, 0xbe, 0xef])));
    }

    #[test]
    fn round_trip_field_index() {
        let n = PNode::Predicate(PredicateNode {
            field: FieldRef::Index(3),
            op: OpType::Gt,
            comparands: vec![Comparand::Int(7)],
        });
        assert_round_trip(&n);
    }

    #[test]
    fn round_trip_in_list() {
        let n = PNode::Predicate(PredicateNode {
            field: FieldRef::Named("status".into()),
            op: OpType::In,
            comparands: vec![Comparand::Int(1), Comparand::Int(2), Comparand::Int(3)],
        });
        assert_round_trip(&n);
    }

    #[test]
    fn round_trip_conjugate_and() {
        let n = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                pred("age", OpType::Ge, Comparand::Int(18)),
                pred("name", OpType::Matches, Comparand::Text("ali".into())),
            ],
        });
        assert_round_trip(&n);
    }

    #[test]
    fn round_trip_conjugate_or() {
        let n = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::Or,
            children: vec![
                pred("a", OpType::Eq, Comparand::Int(1)),
                pred("b", OpType::Eq, Comparand::Int(2)),
                pred("c", OpType::Eq, Comparand::Int(3)),
            ],
        });
        assert_round_trip(&n);
    }

    #[test]
    fn round_trip_nested_conjugate() {
        let n = PNode::Conjugate(ConjugateNode {
            conjugate_type: ConjugateType::And,
            children: vec![
                pred("age", OpType::Ge, Comparand::Int(18)),
                PNode::Conjugate(ConjugateNode {
                    conjugate_type: ConjugateType::Or,
                    children: vec![
                        pred("status", OpType::Eq, Comparand::Int(1)),
                        pred("status", OpType::Eq, Comparand::Int(2)),
                    ],
                }),
            ],
        });
        assert_round_trip(&n);
    }

    #[test]
    fn rejects_trailing_garbage() {
        let r = from_display("age = 1 extra");
        assert!(r.is_err());
    }

    #[test]
    fn rejects_unknown_op() {
        let r = from_display("age @ 1");
        assert!(r.is_err());
    }

    #[test]
    fn rejects_empty_input() {
        let r = from_display("");
        assert!(r.is_err());
    }

    /// Field names that look like keywords (e.g. `INDEX`) must still
    /// parse — `try_consume_keyword` requires a word boundary so
    /// `INDEX` isn't accidentally tokenised as `IN`.
    #[test]
    fn field_name_starting_with_keyword_chars() {
        let n = pred("INDEX_NO", OpType::Eq, Comparand::Int(5));
        assert_round_trip(&n);
    }

    /// Negative integers and floats survive round trip.
    #[test]
    fn round_trip_negative_numbers() {
        assert_round_trip(&pred("delta", OpType::Lt, Comparand::Int(-7)));
        assert_round_trip(&pred("delta", OpType::Lt, Comparand::Float(-1.5)));
    }
}
