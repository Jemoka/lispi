//! LISP Interpreter

use alloc::rc::Rc;
use alloc::vec::Vec;

use super::ast::Value;
use super::environment::{Environment, Image};

/// Executes a LISP expression and return the result and new enviroment
/// importantly, we modify the image's STATE but not the environment
/// useful for e.g. closures, which need to capture the environment at
/// the time of definition
pub fn execute(sexp: Rc<Value>, image: &mut Image) -> Result<(Value, Environment), &'static str> {
    let action = sexp.car();
    match &*action {
        Value::Nil => Err("Cannot execute nil."),
        Value::Symbol(s) => match image.get(&s) {
            Some(v) => execute(
                Value::Cons(Rc::new(execute(v, image)?.0), sexp.cdr()).into(),
                image,
            ),
            _ => Err("Unknown symbol."),
        },
        Value::Cons(..) => execute(
            Value::Cons(Rc::new(execute(action, image)?.0), sexp.cdr()).into(),
            image,
        ),
        Value::Closure(c) => {
            let orig_env = image.e.clone();

            // evaluate arguments first (caller scope), keeping their side effects
            let arg_vals: Vec<Rc<Value>> = c
                .params
                .iter()
                .enumerate()
                .map(|(i, _)| execute(sexp.nth(i + 1), image).map(|(v, _)| Rc::new(v)))
                .collect::<Result<_, _>>()?;

            // build closure scope: caller -> captured -> params
            let mut temp_env = orig_env.clone();
            c.env.iter().for_each(|(k, v)| {
                temp_env.insert(k.clone(), *v);
            });
            image.e = temp_env; // TODO how do we know the closed environment symbol ids are found
            c.params
                .iter()
                .zip(arg_vals.into_iter())
                .for_each(|(param, val)| {
                    image.insert((**param).clone(), val);
                });

            let body_result = execute(Rc::clone(&c.body), image);
            let return_env = orig_env.clone();
            image.e = orig_env;

            body_result.map(|(v, _)| (v, return_env))
        }
        _ => todo!(),
    }
}
