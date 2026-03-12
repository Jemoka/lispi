//! LISP Interpreter

use alloc::rc::Rc;
use alloc::vec::Vec;

use super::ast::Value;
use super::ast::Syscall;
use super::environment::{Environment, Image};
use super::special::execute_special;
use super::syscalls::execute_syscall;

/// Call a closure with already-prepared argument values.
/// Handles environment save/restore and scope layering.
fn call_closure(c: &super::ast::Closure, arg_vals: Vec<Rc<Value>>, image: &mut Image) -> Result<(Value, Environment), &'static str> {
    let orig_env = image.e.clone();

    // build closure scope: (1) caller -> (2) captured -> (3) params
    let mut temp_env = orig_env.clone();
    c.env.iter().for_each(|(k, v)| {
        temp_env.insert(k.clone(), Rc::clone(v));
    });
    image.e = temp_env;

    c.params
        .iter()
        .zip(arg_vals.into_iter())
        .for_each(|(param, val)| {
            image.insert((**param).clone(), val);
        });

    let body_result = evaluate(Rc::clone(&c.body), image);

    let return_env = orig_env.clone();
    image.e = orig_env;

    body_result.map(|(v, _)| (v, return_env))
}

/// Evaluate a value: if it's a fundamental value (nil, bool, number, string,
/// closure, macro, special), return it as-is. If it's a symbol, look it up.
/// If it's a list (cons), execute it.
#[allow(unused)]
pub fn evaluate(sexp: Rc<Value>, image: &mut Image) -> Result<(Value, Environment), &'static str> {
    let env = image.e.clone();
    match &*sexp {
        // fundamental values — return as-is
        Value::Nil
        | Value::Bool(_)
        | Value::Number(_)
        | Value::String(_)
        | Value::Closure(_)
        | Value::Macro(_)
        | Value::Special(_)
        | Value::Syscall(_) => Ok(((*sexp).clone(), env)),

        // symbol — look up in environment
        Value::Symbol(s) => match image.get(s) {
            Some(v) => Ok(((*v).clone(), env)),
            None => Err("Unknown symbol."),
        },

        // list — execute it
        Value::Cons(..) => execute(sexp, image),
    }
}

/// Execute a list sexp: the car is the action, dispatch on its type.
fn execute(sexp: Rc<Value>, image: &mut Image) -> Result<(Value, Environment), &'static str> {
    let action = sexp.car();

    // evaluate the head to figure out what we're calling
    let (resolved, _) = evaluate(action, image)?;

    match &resolved {
        Value::Closure(c) => {
            // evaluate arguments in caller's scope
            let arg_vals: Vec<Rc<Value>> = c
                .params
                .iter()
                .enumerate()
                .map(|(i, _)| evaluate(sexp.nth(i + 1), image).map(|(v, _)| Rc::new(v)))
                .collect::<Result<_, _>>()?;

            call_closure(c, arg_vals, image)
        }
        Value::Macro(m) => {
            // call the closure with UNEVALUATED args (raw sexps)
            let arg_vals: Vec<Rc<Value>> = m.params
                .iter()
                .enumerate()
                .map(|(i, _)| sexp.nth(i + 1))
                .collect();

            // the closure returns the expanded sexp — then evaluate it
            let (expanded, _) = call_closure(&m.closure, arg_vals, image)?;
            evaluate(Rc::new(expanded), image)
        }
        Value::Special(s) => execute_special(s.clone(), sexp, image),
        Value::Syscall(s) => execute_syscall(s.clone(), sexp, image),
        _ => Err("Cannot execute: head is not callable."),
    }
}
