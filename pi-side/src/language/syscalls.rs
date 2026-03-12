//! LISP Syscall Infrastructure

use alloc::rc::Rc;

use super::ast::Value;
use super::environment::{Environment, Image};
use super::execute::evaluate;

use crate::comm::uart;
use crate::utils::memory::{dsb, get32, prefetch_flush, put32};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Syscall {
    Get32,
    Put32,
    DSB,
    PrefetchFlush,
    UartInit,
    UartPut8,
    UartGet8,
}

impl Syscall {
    /// Look up a syscall by name (case-insensitive).
    /// Returns None for unknown names.
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("get32") {
            return Some(Self::Get32);
        }
        if name.eq_ignore_ascii_case("put32") {
            return Some(Self::Put32);
        }
        if name.eq_ignore_ascii_case("dsb") {
            return Some(Self::DSB);
        }
        if name.eq_ignore_ascii_case("prefetch_flush") {
            return Some(Self::PrefetchFlush);
        }
        if name.eq_ignore_ascii_case("uart/init") {
            return Some(Self::UartInit);
        }
        if name.eq_ignore_ascii_case("uart/put8") {
            return Some(Self::UartPut8);
        }
        if name.eq_ignore_ascii_case("uart/get8") {
            return Some(Self::UartGet8);
        }
        None
    }
}

pub fn execute_syscall(
    syscall: Syscall,
    sexp: Rc<Value>,
    image: &mut Image,
) -> Result<(Value, Environment), &'static str> {
    let env = image.e.clone();
    match syscall {
        Syscall::Get32 => {
            let addr = evaluate(sexp.nth(1), image)?.0;
            if let Value::Number(n) = &addr {
                let raw_addr = n
                    .as_addr()
                    .map_err(|_| "GET32: argument must be an address or non-negative integer.")?;
                if let super::number::Number::Addr(a) = raw_addr {
                    let val = unsafe { get32(a) };
                    Ok((
                        Value::Number(super::number::Number::Integer(val as i32)),
                        env,
                    ))
                } else {
                    unreachable!()
                }
            } else {
                Err("GET32 requires a number argument.")
            }
        }
        Syscall::Put32 => {
            let addr_val = evaluate(sexp.nth(1), image)?.0;
            let val_val = evaluate(sexp.nth(2), image)?.0;
            if let (Value::Number(n_addr), Value::Number(n_val)) = (&addr_val, &val_val) {
                let raw_addr = n_addr.as_addr().map_err(
                    |_| "PUT32: first argument must be an address or non-negative integer.",
                )?;
                let raw_val = n_val
                    .as_i32()
                    .map_err(|_| "PUT32: second argument must be an integer.")?;
                if let super::number::Number::Addr(a) = raw_addr {
                    unsafe { put32(a, raw_val as u32) };
                    prefetch_flush();
                    dsb();
                    Ok((Value::Nil, env))
                } else {
                    unreachable!()
                }
            } else {
                Err("PUT32 requires two number arguments.")
            }
        }
        Syscall::DSB => {
            dsb();
            Ok((Value::Nil, env))
        }
        Syscall::PrefetchFlush => {
            prefetch_flush();
            Ok((Value::Nil, env))
        }
        Syscall::UartInit => {
            uart::init();
            Ok((Value::Nil, env))
        }
        Syscall::UartPut8 => {
            let val = evaluate(sexp.nth(1), image)?.0;
            if let Value::Number(n) = &val {
                let byte = n.as_i32().map_err(|_| "uart/put8: argument must be an integer.")?;
                uart::put8(byte as u8);
                Ok((Value::Nil, env))
            } else {
                Err("uart/put8 requires a number argument.")
            }
        }
        Syscall::UartGet8 => {
            let byte = uart::get8();
            Ok((Value::Number(super::number::Number::Integer(byte as i32)), env))
        }
    }
}
