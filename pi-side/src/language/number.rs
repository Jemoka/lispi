//! Numeric type for the LISP interpreter.
//! Unifies integers, unsigned integers, floats, and addresses.
//! Integer/float operations lift to float if either operand is float.
//! Unsigned promotes when mixed with Integer.
//! Addr is a separate kind — it represents a raw memory address and
//! does not participate in float promotion.
//!
//! Arithmetic rules:
//!   Int op Int       → Int
//!   Unsigned op Unsigned → Unsigned
//!   Unsigned op Int  → Unsigned (promotes)
//!   Int op Unsigned  → Unsigned (promotes)
//!   Float op any     → Float (lifts)
//!   Addr +/- Int     → Addr  (pointer offset)
//!   Addr +/- Unsigned → Addr (pointer offset)
//!   Int + Addr       → Addr  (pointer offset, commutative)
//!   Unsigned + Addr  → Addr  (pointer offset, commutative)
//!   Addr - Addr      → Int   (distance)
//!   Addr * or / anything → error
//!   Div by zero      → error

use core::fmt;

/// A number: exact integer, unsigned integer, float, or raw address.
/// Implements PartialEq and PartialOrd so you can use ==, <, >, <=, >=.
#[derive(Clone, Copy, Debug)]
pub enum Number {
    Integer(i32),
    Unsigned(u32),
    Float(f32),
    /// A raw memory address. Kept separate from Integer so that address
    /// arithmetic is explicit and you can't accidentally float-promote
    /// a pointer.
    Addr(usize),
}

impl Number {
    pub fn as_f32(&self) -> f32 {
        match self {
            Number::Integer(i) => *i as f32,
            Number::Unsigned(u) => *u as f32,
            Number::Float(f) => *f,
            Number::Addr(a) => *a as f32,
        }
    }

    /// Coerce to Addr. Identity if already Addr, converts non-negative
    /// integers and unsigned, rejects floats.
    pub fn as_addr(&self) -> Result<Number, &'static str> {
        match self {
            Number::Addr(_) => Ok(*self),
            Number::Integer(i) => {
                if *i < 0 {
                    Err("Cannot convert negative integer to address.")
                } else {
                    Ok(Number::Addr(*i as usize))
                }
            }
            Number::Unsigned(u) => Ok(Number::Addr(*u as usize)),
            Number::Float(_) => Err("Cannot convert float to address."),
        }
    }

    /// Extract the inner i32, or cast from unsigned. Error for float/addr.
    pub fn as_i32(&self) -> Result<i32, &'static str> {
        match self {
            Number::Integer(i) => Ok(*i),
            Number::Unsigned(u) => Ok(*u as i32),
            _ => Err("Expected integer."),
        }
    }

    /// Extract the inner u32, or cast from integer. Error for float/addr.
    pub fn as_u32(&self) -> Result<u32, &'static str> {
        match self {
            Number::Unsigned(u) => Ok(*u),
            Number::Integer(i) => Ok(*i as u32),
            _ => Err("Expected unsigned integer."),
        }
    }

    /// Add two numbers. See module docs for lifting rules.
    pub fn add(self, other: Number) -> Result<Number, &'static str> {
        match (self, other) {
            (Number::Integer(a), Number::Integer(b)) => Ok(Number::Integer(a.wrapping_add(b))),
            (Number::Unsigned(a), Number::Unsigned(b)) => Ok(Number::Unsigned(a.wrapping_add(b))),
            (Number::Unsigned(a), Number::Integer(b)) => {
                Ok(Number::Unsigned(a.wrapping_add(b as u32)))
            }
            (Number::Integer(a), Number::Unsigned(b)) => {
                Ok(Number::Unsigned((a as u32).wrapping_add(b)))
            }
            (Number::Addr(a), Number::Integer(b)) => Ok(Number::Addr(a.wrapping_add(b as usize))),
            (Number::Integer(a), Number::Addr(b)) => Ok(Number::Addr((a as usize).wrapping_add(b))),
            (Number::Addr(a), Number::Unsigned(b)) => {
                Ok(Number::Addr(a.wrapping_add(b as usize)))
            }
            (Number::Unsigned(a), Number::Addr(b)) => {
                Ok(Number::Addr((a as usize).wrapping_add(b)))
            }
            (Number::Addr(_), Number::Addr(_)) => Err("Cannot add two addresses."),
            (Number::Addr(_), Number::Float(_)) | (Number::Float(_), Number::Addr(_)) => {
                Err("Cannot mix address and float.")
            }
            _ => Ok(Number::Float(self.as_f32() + other.as_f32())),
        }
    }

    /// Subtract two numbers. Addr - Addr yields the integer distance.
    pub fn sub(self, other: Number) -> Result<Number, &'static str> {
        match (self, other) {
            (Number::Integer(a), Number::Integer(b)) => Ok(Number::Integer(a.wrapping_sub(b))),
            (Number::Unsigned(a), Number::Unsigned(b)) => Ok(Number::Unsigned(a.wrapping_sub(b))),
            (Number::Unsigned(a), Number::Integer(b)) => {
                Ok(Number::Unsigned(a.wrapping_sub(b as u32)))
            }
            (Number::Integer(a), Number::Unsigned(b)) => {
                Ok(Number::Unsigned((a as u32).wrapping_sub(b)))
            }
            (Number::Addr(a), Number::Integer(b)) => Ok(Number::Addr(a.wrapping_sub(b as usize))),
            (Number::Addr(a), Number::Unsigned(b)) => {
                Ok(Number::Addr(a.wrapping_sub(b as usize)))
            }
            (Number::Addr(a), Number::Addr(b)) => Ok(Number::Integer(a.wrapping_sub(b) as i32)),
            (Number::Integer(_), Number::Addr(_)) | (Number::Unsigned(_), Number::Addr(_)) => {
                Err("Cannot subtract address from integer.")
            }
            (Number::Addr(_), Number::Float(_)) | (Number::Float(_), Number::Addr(_)) => {
                Err("Cannot mix address and float.")
            }
            _ => Ok(Number::Float(self.as_f32() - other.as_f32())),
        }
    }

    /// Multiply two numbers. Addresses cannot be multiplied.
    pub fn mul(self, other: Number) -> Result<Number, &'static str> {
        match (self, other) {
            (Number::Integer(a), Number::Integer(b)) => Ok(Number::Integer(a.wrapping_mul(b))),
            (Number::Unsigned(a), Number::Unsigned(b)) => Ok(Number::Unsigned(a.wrapping_mul(b))),
            (Number::Unsigned(a), Number::Integer(b)) => {
                Ok(Number::Unsigned(a.wrapping_mul(b as u32)))
            }
            (Number::Integer(a), Number::Unsigned(b)) => {
                Ok(Number::Unsigned((a as u32).wrapping_mul(b)))
            }
            (Number::Addr(_), _) | (_, Number::Addr(_)) => Err("Cannot multiply addresses."),
            _ => Ok(Number::Float(self.as_f32() * other.as_f32())),
        }
    }

    /// Divide two numbers. Addresses cannot be divided. Division by zero errors.
    pub fn div(self, other: Number) -> Result<Number, &'static str> {
        match (self, other) {
            (Number::Addr(_), _) | (_, Number::Addr(_)) => Err("Cannot divide addresses."),
            (Number::Integer(a), Number::Integer(b)) => {
                if b == 0 {
                    Err("Division by zero.")
                } else {
                    Ok(Number::Integer(a / b))
                }
            }
            (Number::Unsigned(a), Number::Unsigned(b)) => {
                if b == 0 {
                    Err("Division by zero.")
                } else {
                    Ok(Number::Unsigned(a / b))
                }
            }
            (Number::Unsigned(a), Number::Integer(b)) => {
                if b == 0 {
                    Err("Division by zero.")
                } else {
                    Ok(Number::Unsigned(a / b as u32))
                }
            }
            (Number::Integer(a), Number::Unsigned(b)) => {
                if b == 0 {
                    Err("Division by zero.")
                } else {
                    Ok(Number::Unsigned(a as u32 / b))
                }
            }
            _ => {
                let d = other.as_f32();
                if d == 0.0 {
                    Err("Division by zero.")
                } else {
                    Ok(Number::Float(self.as_f32() / d))
                }
            }
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Integer(i) => write!(f, "{}", i),
            Number::Unsigned(u) => write!(f, "u{}", u),
            Number::Float(v) => write!(f, "{}", v),
            Number::Addr(a) => write!(f, "0x{:x}", a),
        }
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Number::Integer(a), Number::Integer(b)) => a == b,
            (Number::Unsigned(a), Number::Unsigned(b)) => a == b,
            (Number::Addr(a), Number::Addr(b)) => a == b,
            // unsigned/int: compare as u32
            (Number::Unsigned(a), Number::Integer(b)) => *b >= 0 && *a == *b as u32,
            (Number::Integer(a), Number::Unsigned(b)) => *a >= 0 && *a as u32 == *b,
            // addr/int: compare as usize
            (Number::Addr(a), Number::Integer(b)) => *b >= 0 && *a == *b as usize,
            (Number::Integer(a), Number::Addr(b)) => *a >= 0 && *a as usize == *b,
            // addr/unsigned: compare as usize
            (Number::Addr(a), Number::Unsigned(b)) => *a == *b as usize,
            (Number::Unsigned(a), Number::Addr(b)) => *a as usize == *b,
            // float mixed: lift to float
            (Number::Float(_), Number::Float(_))
            | (Number::Integer(_), Number::Float(_))
            | (Number::Float(_), Number::Integer(_))
            | (Number::Unsigned(_), Number::Float(_))
            | (Number::Float(_), Number::Unsigned(_)) => self.as_f32() == other.as_f32(),
            // addr vs float: incomparable
            _ => false,
        }
    }
}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (self, other) {
            (Number::Integer(a), Number::Integer(b)) => a.partial_cmp(b),
            (Number::Unsigned(a), Number::Unsigned(b)) => a.partial_cmp(b),
            (Number::Addr(a), Number::Addr(b)) => a.partial_cmp(b),
            // unsigned/int: compare as i64 to handle sign correctly
            (Number::Unsigned(a), Number::Integer(b)) => {
                (*a as i64).partial_cmp(&(*b as i64))
            }
            (Number::Integer(a), Number::Unsigned(b)) => {
                (*a as i64).partial_cmp(&(*b as i64))
            }
            // float mixed: lift to float
            (Number::Float(_), Number::Float(_))
            | (Number::Integer(_), Number::Float(_))
            | (Number::Float(_), Number::Integer(_))
            | (Number::Unsigned(_), Number::Float(_))
            | (Number::Float(_), Number::Unsigned(_)) => {
                self.as_f32().partial_cmp(&other.as_f32())
            }
            // addr vs int/float: incomparable
            _ => None,
        }
    }
}
