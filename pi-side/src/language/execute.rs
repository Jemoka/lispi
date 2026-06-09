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
        | Value::Array(_)
        | Value::JittedClosure(_) => Ok((*sexp).clone()),

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
        Value::JittedClosure(jc) => {
            // Evaluate args in caller scope (never tail — the JIT body
            // is one straight-line trip, no interpreter trampoline).
            let arg_vals: Vec<Value> = jc
                .params
                .iter()
                .enumerate()
                .map(|(i, _)| evaluate(sexp.nth(i + 1), image))
                .collect::<Result<_, _>>()?;

            // Type guard — the JIT body is specialized; refuse args
            // that don't match the compile-time dummy types.
            for (i, val) in arg_vals.iter().enumerate() {
                if !jc.input_types[i].accepts(val) {
                    return Err(
                        "jitted-closure: argument type mismatch — recompile (jit ...) with the new types.",
                    );
                }
            }

            // Save the current values of the pinned param Bindings,
            // then overwrite with this call's args. Restoring at exit
            // is what makes **recursion** work: each call sees its own
            // n, and when the recursive frame returns, the caller's n
            // is back where it left it. The JIT-emitted code holds raw
            // pointers into these `RefCell`s (via `LoadCapture`); the
            // RefCells themselves are pointer-stable across writes.
            let saved_param_vals: alloc::vec::Vec<Rc<Value>> = jc
                .param_bindings
                .iter()
                .map(|b| b.borrow().clone())
                .collect();
            for (i, val) in arg_vals.into_iter().enumerate() {
                *jc.param_bindings[i].borrow_mut() = Rc::new(val);
            }

            // Make the closure's captures + params visible to the
            // interpreter while the JIT runs — if the compiled body
            // `Escape`s back to the interpreter (e.g. to dispatch a
            // call to another `JittedClosure`), name lookups need to
            // resolve to the same Bindings the JIT used.
            image.push_env(&jc.env);
            let mut param_frame: alloc::collections::BTreeMap<_, _> =
                alloc::collections::BTreeMap::new();
            for (param, binding) in jc.params.iter().zip(jc.param_bindings.iter()) {
                param_frame.insert((**param).clone(), binding.clone());
            }
            image.push_env(&Rc::new(param_frame));

            let result = jc.executor.run(image);

            image.pop_frame();
            image.pop_frame();

            // Restore the pinned param Bindings to their pre-call
            // values, so the caller's view of these names is intact.
            for (i, val) in saved_param_vals.into_iter().enumerate() {
                *jc.param_bindings[i].borrow_mut() = val;
            }

            result
        }
        _ => Err("Cannot execute: head is not callable."),
    }
}
