//! LISP language infrastructure

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::fmt;
use heapless::String;
use alloc::string::String as AllocString;

use super::constants::SYMB_NAME_LEN;
use super::environment;
use super::number::Number;
pub use super::special::Special;
pub use super::syscalls::Syscall;

/// a symbol, which is just a string with a maximum length
pub type Symbol = String<SYMB_NAME_LEN>;

/// A closure captures the environment at the time of `lambda`, so it can
/// reference bindings from its defining scope.  Because Environment now
/// holds Rc<RefCell<..>> Bindings, cloning the env is cheap (refcount
/// bumps) and `set!` mutations in the defining scope are visible to the
/// closure and vice-versa.
///
/// NOTE: PartialEq/Eq compare environments by *value* (borrowing the
/// RefCells).  If a closure is stored in its own captured environment
/// (e.g. recursive `define`), equality comparison would recurse forever.
/// This is fine as long as you never compare closures for deep equality —
/// most LISPs use pointer identity (`eq?`) for procedures anyway.
#[derive(Clone, Debug, PartialEq)]
pub struct Closure {
    pub params: Vec<Rc<Symbol>>,
    pub body: Rc<Value>,
    pub env: environment::Environment,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Macro {
    pub params: Vec<Rc<Symbol>>,
    pub closure: Closure,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    Nil,
    Bool(bool),
    Closure(Closure),
    Special(Special),
    Symbol(Rc<Symbol>),
    Cons(Rc<Value>, Rc<Value>),
    Number(Number),
    String(AllocString),
    Macro(Macro),
    Syscall(Syscall),
}

impl Value {
    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    pub fn cons(car: Value, cdr: Value) -> Self {
        Value::Cons(Rc::new(car), Rc::new(cdr))
    }

    pub fn car(&self) -> Rc<Value> {
        match self {
            Value::Cons(car, _) => Rc::clone(car),
            _ => Value::Nil.into(),
        }
    }

    pub fn cdr(&self) -> Rc<Value> {
        match self {
            Value::Cons(_, cdr) => Rc::clone(cdr),
            _ => Value::Nil.into(),
        }
    }

    pub fn nth(&self, n: usize) -> Rc<Value> {
        let mut current = self;
        for _ in 0..n {
            match current {
                Value::Cons(_, cdr) => current = cdr,
                _ => return Value::Nil.into(),
            }
        }
        match current {
            Value::Cons(car, _) => Rc::clone(car),
            _ => Value::Nil.into(),
        }
    }
}

/// Display a Value in a human-readable form, used by the `print` special form.
/// Lists are printed as `(a b c)`, nil as `nil`, etc.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{}", s),
            Value::Symbol(s) => write!(f, "{}", s.as_str()),
            Value::Special(s) => write!(f, "<special:{:?}>", s),
            Value::Closure(_) => write!(f, "<closure>"),
            Value::Macro(_) => write!(f, "<macro>"),
            Value::Syscall(s) => write!(f, "<syscall:{:?}>", s),
            Value::Cons(_, _) => {
                write!(f, "(")?;
                let mut current: &Value = self;
                let mut first = true;
                loop {
                    match current {
                        Value::Cons(car, cdr) => {
                            if !first { write!(f, " ")?; }
                            first = false;
                            write!(f, "{}", car)?;
                            current = cdr;
                        }
                        Value::Nil => break,
                        // dotted pair tail
                        other => {
                            write!(f, " . {}", other)?;
                            break;
                        }
                    }
                }
                write!(f, ")")
            }
        }
    }
}
