//! LISP Interpreter

use alloc::rc::Rc;
use alloc::vec::Vec;

use super::ast::{Closure, Value};
use super::environment::Image;
use super::special::execute_special;
use super::syscalls::execute_syscall;

/// Call a closure with already-prepared argument values.
/// Contains the trampoline loop for tail-call optimization:
/// if the body returns a TailCall token, rebind params and loop
/// instead of recursing.
fn call_closure(
    c: &Closure,
    arg_vals: Vec<Rc<Value>>,
    image: &mut Image,
) -> Result<Value, &'static str> {
    let mut closure = c.clone();
    let mut args = arg_vals;

    loop {
        // push captured env as a shared frame — O(1), just Rc::clone
        image.push_env(&closure.env);
        // push a fresh frame for parameter bindings
        image.push_frame();

        closure
            .params
            .iter()
            .zip(args.into_iter())
            .for_each(|(param, val)| {
                image.insert((**param).clone(), val);
            });

        // body is always in tail position
        let result = eval(Rc::clone(&closure.body), image, true)?;

        image.pop_frame(); // params
        image.pop_frame(); // captured env

        match result {
            Value::TailCall(next_closure, next_args) => {
                // tail call — reuse this stack frame
                closure = next_closure;
                args = next_args;
            }
            other => return Ok(other),
        }
    }
}

/// Public API: evaluate an expression (never in tail position).
#[allow(unused)]
pub fn evaluate(sexp: Rc<Value>, image: &mut Image) -> Result<Value, &'static str> {
    eval(sexp, image, false)
}

/// Internal: evaluate with tail-position flag.
/// When `tail` is true and the result would be a closure call,
/// returns a TailCall token instead of actually calling.
pub(super) fn eval(
    sexp: Rc<Value>,
    image: &mut Image,
    tail: bool,
) -> Result<Value, &'static str> {
    match &*sexp {
        // fundamental values — return as-is
        Value::Nil
        | Value::Bool(_)
        | Value::Number(_)
        | Value::String(_)
        | Value::Closure(_)
        | Value::Macro(_)
        | Value::Special(_)
        | Value::Syscall(_)
        | Value::Array(_) => Ok((*sexp).clone()),

        // TailCall should never appear in source — only as trampoline tokens
        Value::TailCall(_, _) => Ok((*sexp).clone()),

        // symbol — look up in environment
        Value::Symbol(s) => match image.get(s) {
            Some(v) => Ok((*v).clone()),
            None => Err("Unknown symbol."),
        },

        // list — execute it
        Value::Cons(..) => exec(sexp, image, tail),
    }
}

/// Execute a list sexp: the car is the action, dispatch on its type.
fn exec(sexp: Rc<Value>, image: &mut Image, tail: bool) -> Result<Value, &'static str> {
    let action = sexp.car();

    // evaluate the head to figure out what we're calling (never tail)
    let resolved = evaluate(action, image)?;

    // match by value so we can move Closure into TailCall without cloning
    match resolved {
        Value::Closure(c) => {
            // evaluate arguments in caller's scope (never tail)
            let arg_vals: Vec<Rc<Value>> = c
                .params
                .iter()
                .enumerate()
                .map(|(i, _)| evaluate(sexp.nth(i + 1), image).map(|v| Rc::new(v)))
                .collect::<Result<_, _>>()?;

            if tail {
                // in tail position: return token, move closure — no clone
                Ok(Value::TailCall(c, arg_vals))
            } else {
                call_closure(&c, arg_vals, image)
            }
        }
        Value::Macro(m) => {
            // call the closure with UNEVALUATED args (raw sexps)
            let arg_vals: Vec<Rc<Value>> = m
                .params
                .iter()
                .enumerate()
                .map(|(i, _)| sexp.nth(i + 1))
                .collect();

            // the closure returns the expanded sexp — then evaluate it
            // propagate tail: if macro call is in tail position, so is its expansion
            let expanded = call_closure(&m.closure, arg_vals, image)?;
            eval(Rc::new(expanded), image, tail)
        }
        Value::Special(s) => execute_special(s.clone(), sexp, image, tail),
        Value::Syscall(s) => execute_syscall(s.clone(), sexp, image),
        _ => Err("Cannot execute: head is not callable."),
    }
}
