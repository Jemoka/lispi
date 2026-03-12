//! S-expression parser for the LISP interpreter.
//!
//! Parses a single `Value` from a `&str` using nom.
//!
//! Syntax:
//!   (a b c)              — list (cons chain ending in nil)
//!   '(a b c)             — quote sugar: desugars to (list a b c)
//!   @name                — syscall (case-insensitive): get32, set32, dsb, prefetch_flush
//!   #FF                  — address literal (hex digits)
//!   42  -7               — integer
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

/// Parse #hex_digits as Number::Addr.
fn parse_address(input: &str) -> IResult<&str, Value> {
    let (rest, _) = char('#')(input)?;
    let (rest, digits) = hex_digit1(rest)?;
    let addr = usize::from_str_radix(digits, 16).map_err(|_| {
        nom::Err::Failure(nom::error::Error::new(
            input,
            nom::error::ErrorKind::HexDigit,
        ))
    })?;
    Ok((rest, Value::Number(Number::Addr(addr))))
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

/// Parse an integer literal: [-]digits
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

/// Parse a number: try float first (has `.`), then integer.
fn parse_number(input: &str) -> IResult<&str, Value> {
    alt((parse_float, parse_integer)).parse(input)
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
/// Returns the parsed Value, or an error string on failure.
pub fn parse(input: &str) -> Result<Value, &'static str> {
    let (_, val) = parse_value(input).map_err(|_| "Parse error.")?;
    Ok(val)
}

/// Parse one value and return both the value and remaining input.
/// Useful for REPL-style incremental parsing.
pub fn parse_with_rest(input: &str) -> Result<(Value, &str), &'static str> {
    let (rest, val) = parse_value(input).map_err(|_| "Parse error.")?;
    Ok((val, rest))
}
