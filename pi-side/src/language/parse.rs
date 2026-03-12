//! S-expression parser for the LISP interpreter.
//!
//! Parses a single `Value` from a `&str` using nom.
//!
//! Syntax:
//!   (a b c)              — list (cons chain ending in nil)
//!   '(a b c)             — quote sugar: desugars to (list a b c)
//!   @name                — syscall (case-insensitive): get32, put32, dsb, prefetch_flush
//!   #42  #0xFF  #0b101   — address literal (decimal, 0x hex, 0b binary)
//!   u42  u0xFF  u0b101   — unsigned literal (decimal, 0x hex, 0b binary)
//!   42  -7               — integer (decimal)
//!   0xFF                 — integer (hex)
//!   0b1010               — integer (binary)
//!   3.14  -0.5           — float
//!   "hello"              — string (supports \n \t \\ \")
//!   + - * / > < ~ | &   — operator specials
//!   nil true false       — literal values
//!   defun lambda if ...  — named special forms
//!   anything-else        — symbol
//!   ; comment            — line comment (to end of line)

use alloc::rc::Rc;
use alloc::string::String as AllocString;
use alloc::vec::Vec;

use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::take_while1,
    character::complete::{char, digit1, hex_digit1, multispace0, satisfy},
    combinator::opt,
};

use super::ast::{Special, Syscall, Value};
use super::constants::SYMB_NAME_LEN;
use super::number::Number;

type Symbol = heapless::String<SYMB_NAME_LEN>;

// ---------------------------------------------------------------------------
// whitespace & comments
// ---------------------------------------------------------------------------

/// Consume whitespace and `;` line comments (repeating until neither remains).
fn ws(mut input: &str) -> IResult<&str, ()> {
    loop {
        let (rest, _) = multispace0(input)?;
        if rest.starts_with(';') {
            let eol = rest.find('\n').map(|p| p + 1).unwrap_or(rest.len());
            input = &rest[eol..];
        } else {
            return Ok((rest, ()));
        }
    }
}

// ---------------------------------------------------------------------------
// atoms
// ---------------------------------------------------------------------------

/// Parse a double-quoted string with escape sequences (\n \t \\ \").
fn parse_string(input: &str) -> IResult<&str, Value> {
    let (input, _) = char('"')(input)?;
    let mut s = AllocString::new();
    let mut rest = input;
    let mut escape = false;
    loop {
        if rest.is_empty() {
            return Err(nom::Err::Failure(nom::error::Error::new(
                rest,
                nom::error::ErrorKind::Char,
            )));
        }
        let ch = rest.chars().next().unwrap();
        rest = &rest[ch.len_utf8()..];
        if escape {
            match ch {
                'n' => s.push('\n'),
                't' => s.push('\t'),
                '\\' => s.push('\\'),
                '"' => s.push('"'),
                other => {
                    s.push('\\');
                    s.push(other);
                }
            }
            escape = false;
        } else if ch == '\\' {
            escape = true;
        } else if ch == '"' {
            return Ok((rest, Value::String(s)));
        } else {
            s.push(ch);
        }
    }
}

/// Parse `#` address literals.
/// `#0xFF` → hex, `#0b101` → binary, `#42` → decimal.
fn parse_address(input: &str) -> IResult<&str, Value> {
    let (rest, _) = char('#')(input)?;
    let make_err = || {
        nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::HexDigit,
        ))
    };
    let (rest, addr) = if rest.starts_with("0x") || rest.starts_with("0X") {
        let r = &rest[2..];
        let (r, digits) = hex_digit1(r)?;
        let v = usize::from_str_radix(digits, 16).map_err(|_| make_err())?;
        (r, v)
    } else if rest.starts_with("0b") || rest.starts_with("0B") {
        let r = &rest[2..];
        let (r, digits) = take_while1(|c: char| c == '0' || c == '1')(r)?;
        let v = usize::from_str_radix(digits, 2).map_err(|_| make_err())?;
        (r, v)
    } else {
        // Decimal: #42 means address 42
        let (r, digits) = digit1(rest)?;
        let v: usize = digits.parse().map_err(|_| make_err())?;
        (r, v)
    };
    Ok((rest, Value::Number(Number::Addr(addr))))
}

/// Parse `u` unsigned literals.
/// `u0xFF` → hex, `u0b101` → binary, `u42` → decimal.
fn parse_unsigned(input: &str) -> IResult<&str, Value> {
    let (rest, _) = char('u')(input)?;
    // must be followed by a digit or 0x/0b prefix, not an ident char
    // (to avoid consuming symbols starting with 'u')
    if rest.is_empty() || !(rest.starts_with(|c: char| c.is_ascii_digit())) {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Digit,
        )));
    }
    let make_err = || {
        nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Digit,
        ))
    };
    let (rest, val) = if rest.starts_with("0x") || rest.starts_with("0X") {
        let r = &rest[2..];
        let (r, digits) = hex_digit1(r)?;
        let v = u32::from_str_radix(digits, 16).map_err(|_| make_err())?;
        (r, v)
    } else if rest.starts_with("0b") || rest.starts_with("0B") {
        let r = &rest[2..];
        let (r, digits) = take_while1(|c: char| c == '0' || c == '1')(r)?;
        let v = u32::from_str_radix(digits, 2).map_err(|_| make_err())?;
        (r, v)
    } else {
        let (r, digits) = digit1(rest)?;
        let v: u32 = digits.parse().map_err(|_| make_err())?;
        (r, v)
    };
    // reject if followed by an ident char (e.g. `u42foo` shouldn't parse as unsigned 42)
    if rest.starts_with(|c: char| is_ident_char(c)) {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Digit,
        )));
    }
    Ok((rest, Value::Number(Number::Unsigned(val))))
}

/// Parse a floating-point literal: [-]digits.digits
fn parse_float(input: &str) -> IResult<&str, Value> {
    let start = input;
    let (i, _) = opt(char('-')).parse(input)?;
    let (i, _) = digit1(i)?;
    let (i, _) = char('.')(i)?;
    let (rest, _) = digit1(i)?;
    let slice = &start[..start.len() - rest.len()];
    let f: f32 = slice.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Float))
    })?;
    Ok((rest, Value::Number(Number::Float(f))))
}

/// Parse a hex integer: 0xFF or 0XFF
fn parse_hex_integer(input: &str) -> IResult<&str, Value> {
    let start = input;
    let (i, neg) = opt(char('-')).parse(input)?;
    if !(i.starts_with("0x") || i.starts_with("0X")) {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }
    let i = &i[2..];
    let (rest, digits) = hex_digit1(i)?;
    let v = i32::from_str_radix(digits, 16).map_err(|_| {
        nom::Err::Error(nom::error::Error::new(
            start,
            nom::error::ErrorKind::HexDigit,
        ))
    })?;
    let v = if neg.is_some() { -v } else { v };
    Ok((rest, Value::Number(Number::Integer(v))))
}

/// Parse a binary integer: 0b0110 or 0B0110
fn parse_bin_integer(input: &str) -> IResult<&str, Value> {
    let start = input;
    let (i, neg) = opt(char('-')).parse(input)?;
    if !(i.starts_with("0b") || i.starts_with("0B")) {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }
    let i = &i[2..];
    let (rest, digits) = take_while1(|c: char| c == '0' || c == '1')(i)?;
    let v = i32::from_str_radix(digits, 2).map_err(|_| {
        nom::Err::Error(nom::error::Error::new(start, nom::error::ErrorKind::Digit))
    })?;
    let v = if neg.is_some() { -v } else { v };
    Ok((rest, Value::Number(Number::Integer(v))))
}

/// Parse a decimal integer literal: [-]digits
fn parse_integer(input: &str) -> IResult<&str, Value> {
    let start = input;
    let (i, _) = opt(char('-')).parse(input)?;
    let (rest, _) = digit1(i)?;
    let slice = &start[..start.len() - rest.len()];
    let i: i32 = slice.parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit))
    })?;
    Ok((rest, Value::Number(Number::Integer(i))))
}

/// Parse a number: try hex/binary first (0x/0b prefixes), then float, then decimal integer.
fn parse_number(input: &str) -> IResult<&str, Value> {
    alt((
        parse_hex_integer,
        parse_bin_integer,
        parse_float,
        parse_integer,
    ))
    .parse(input)
}

// ---------------------------------------------------------------------------
// syscalls
// ---------------------------------------------------------------------------

/// Returns true if `c` can start a symbol/identifier name.
/// Does NOT include `/` — a bare `/` is the div operator.
fn is_ident_start(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '?' | '!')
}

/// Returns true if `c` can appear in continuation position of a symbol name.
/// Includes `/` so that names like `my/func` parse as one symbol.
fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '?' | '!' | '/')
}

/// Parse @name as a syscall value (case-insensitive).
fn parse_syscall(input: &str) -> IResult<&str, Value> {
    let (rest, _) = char('@')(input)?;
    let (rest, name) = take_while1(|c: char| is_ident_char(c))(rest)?;
    let syscall = Syscall::from_name(name).ok_or_else(|| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))
    })?;
    Ok((rest, Value::Syscall(syscall)))
}

// ---------------------------------------------------------------------------
// quote sugar
// ---------------------------------------------------------------------------

/// Parse 'expr — desugars to (list ...).
/// '(a b c) → (list a b c) — prepend List as head of the cons chain.
/// 'atom    → (list atom)  — wrap in a single-element list call.
fn parse_quote(input: &str) -> IResult<&str, Value> {
    let (rest, _) = char('\'')(input)?;
    let (rest, inner) = parse_value(rest)?;
    let quoted = match inner {
        Value::Cons(_, _) => Value::Cons(Rc::new(Value::Special(Special::List)), Rc::new(inner)),
        _ => Value::cons(
            Value::Special(Special::List),
            Value::cons(inner, Value::Nil),
        ),
    };
    Ok((rest, quoted))
}

// ---------------------------------------------------------------------------
// lists
// ---------------------------------------------------------------------------

/// Check if a symbol matches the c[ad]{2,}r pattern (case-insensitive).
/// `car`/`cdr` (single middle char) are already Special forms and won't reach here.
fn is_cxr_pattern(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 4
        && b[0].eq_ignore_ascii_case(&b'c')
        && b[b.len() - 1].eq_ignore_ascii_case(&b'r')
        && b[1..b.len() - 1]
            .iter()
            .all(|c| c.eq_ignore_ascii_case(&b'a') || c.eq_ignore_ascii_case(&b'd'))
}

/// Parse a parenthesised list: ( value* )
fn parse_list(input: &str) -> IResult<&str, Value> {
    let (mut rest, _) = char('(')(input)?;
    let mut items: Vec<Value> = Vec::new();
    loop {
        let (r, _) = ws(rest)?;
        rest = r;
        if rest.starts_with(')') {
            rest = &rest[1..];
            break;
        }
        if rest.is_empty() {
            return Err(nom::Err::Failure(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Char,
            )));
        }
        let (r, val) = parse_value(rest)?;
        items.push(val);
        rest = r;
    }

    // desugar c[ad]+r: (cadr x) → (car (cdr x))
    if items.len() == 2 {
        let is_cxr = matches!(&items[0], Value::Symbol(sym) if is_cxr_pattern(sym.as_str()));
        if is_cxr {
            let head = items.remove(0);
            let arg = items.remove(0);
            let sym = match head {
                Value::Symbol(s) => s,
                _ => unreachable!(),
            };
            let mut result = arg;
            let s = sym.as_str();
            for b in s[1..s.len() - 1].bytes().rev() {
                let special = if b.eq_ignore_ascii_case(&b'a') {
                    Special::Car
                } else {
                    Special::Cdr
                };
                result = Value::cons(Value::Special(special), Value::cons(result, Value::Nil));
            }
            return Ok((rest, result));
        }
    }

    // build cons list from items (right fold)
    let mut result = Value::Nil;
    for item in items.into_iter().rev() {
        result = Value::Cons(Rc::new(item), Rc::new(result));
    }
    Ok((rest, result))
}

// ---------------------------------------------------------------------------
// single-char operator specials
// ---------------------------------------------------------------------------

/// Parse operator specials: multi-char (>=, <=) then single-char (+, *, etc.).
fn parse_operator(input: &str) -> IResult<&str, Value> {
    // try two-char operators first
    if input.starts_with(">=") {
        return Ok((&input[2..], Value::Special(Special::Gte)));
    }
    if input.starts_with("<=") {
        return Ok((&input[2..], Value::Special(Special::Lte)));
    }
    let (rest, ch) =
        satisfy(|c| matches!(c, '+' | '*' | '/' | '>' | '<' | '~' | '|' | '&'))(input)?;
    // reject if followed by an identifier-start char (e.g. `/foo` shouldn't be Div)
    if rest.starts_with(|c: char| is_ident_start(c)) {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Satisfy,
        )));
    }
    let special = match ch {
        '+' => Special::Add,
        '*' => Special::Mul,
        '/' => Special::Div,
        '>' => Special::Gt,
        '<' => Special::Lt,
        '~' => Special::BinNot,
        '|' => Special::BinOr,
        '&' => Special::BinAnd,
        _ => unreachable!(),
    };
    Ok((rest, Value::Special(special)))
}

/// Parse standalone `-` as Sub (not followed by a digit or ident char).
fn parse_minus(input: &str) -> IResult<&str, Value> {
    let (rest, _) = char('-')(input)?;
    if rest.starts_with(|c: char| c.is_ascii_digit() || is_ident_start(c)) {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Satisfy,
        )));
    }
    Ok((rest, Value::Special(Special::Sub)))
}

// ---------------------------------------------------------------------------
// identifiers (keywords, specials, symbols)
// ---------------------------------------------------------------------------

/// Try to match a word as a literal (case-insensitive).
fn match_literal(word: &str) -> Option<Value> {
    if word.eq_ignore_ascii_case("nil") {
        return Some(Value::Nil);
    }
    if word.eq_ignore_ascii_case("true") {
        return Some(Value::Bool(true));
    }
    if word.eq_ignore_ascii_case("false") {
        return Some(Value::Bool(false));
    }
    None
}

/// Parse a bare word: named specials (case-insensitive), literals, or user
/// symbols (case-sensitive, preserved exactly as written).
fn parse_identifier(input: &str) -> IResult<&str, Value> {
    // first char must be an ident-start (not `/` or `-`)
    if !input.starts_with(|c: char| is_ident_start(c)) {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Alpha,
        )));
    }
    let (rest, word) = take_while1(|c: char| is_ident_char(c))(input)?;

    // literals (case-insensitive)
    if let Some(val) = match_literal(word) {
        return Ok((rest, val));
    }

    // special forms (case-insensitive, defined in special.rs)
    if let Some(sp) = Special::from_name(word) {
        return Ok((rest, Value::Special(sp)));
    }

    // user-defined symbol — preserved exactly as written
    let sym = Symbol::try_from(word).map_err(|_| {
        nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::TooLarge,
        ))
    })?;
    Ok((rest, Value::Symbol(Rc::new(sym))))
}

// ---------------------------------------------------------------------------
// top-level value parser
// ---------------------------------------------------------------------------

/// Parse a single value, consuming leading whitespace/comments.
fn parse_value(input: &str) -> IResult<&str, Value> {
    let (input, _) = ws(input)?;
    alt((
        parse_string,
        parse_address,
        parse_unsigned,
        parse_quote,
        parse_list,
        parse_syscall,
        parse_number,
        parse_operator,
        parse_minus,
        parse_identifier,
    ))
    .parse(input)
}

/// Public entry point: parse one value from the input.
/// Returns the parsed Value, or a descriptive error string on failure.
pub fn parse(input: &str) -> Result<Value, AllocString> {
    let (_, val) = parse_value(input).map_err(|e| AllocString::from(alloc::format!("{}", e)))?;
    Ok(val)
}

/// Parse one value and return both the value and remaining input.
/// Useful for REPL-style incremental parsing.
#[allow(unused)]
pub fn parse_with_rest(input: &str) -> Result<(Value, &str), AllocString> {
    let (rest, val) = parse_value(input).map_err(|e| AllocString::from(alloc::format!("{}", e)))?;
    Ok((val, rest))
}
