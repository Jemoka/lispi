//! LISP language infrastructure

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;
use heapless::String;

use super::constants::SYMB_NAME_LEN;
use super::environment;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Special {
    Quote,
    Lambda,
    If,
    Set,
    Begin,
    Car,
    Cdr,
    Nullp,
    Eq,
    Not,
    Print,
    Gt,
    Lt,
    Add,
    Sub,
    Mul,
    Div,
}

/// a symbol, which is just a string with a maximum length
pub type Symbol = String<SYMB_NAME_LEN>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Closure {
    pub params: Vec<Rc<Symbol>>,
    pub body: Rc<Value>,
    pub env: environment::Environment,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Value {
    Nil,
    Closure(Closure),
    Special(Special),
    Symbol(Rc<Symbol>),
    Cons(Rc<Value>, Rc<Value>),
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
