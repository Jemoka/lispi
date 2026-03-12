//! LISP Syscall Infrastructure

use alloc::rc::Rc;

use super::ast::Value;
use super::environment::{Environment, Image};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Syscall {
    GET32,
    SET32,
}

pub fn execute_syscall(syscall: Syscall, sexp: Rc<Value>, image: &mut Image) -> Result<(Value, Environment), &'static str> {
    todo!()
}
