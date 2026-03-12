//! Special form definitions and implementations for the LISP interpreter.

use alloc::rc::Rc;
use alloc::string::String as AllocString;
use alloc::vec::Vec;

use super::ast::{Closure, Macro, Symbol, Value};
use super::environment::{Environment, Image};
use super::execute::evaluate;
use super::number::Number;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Special {
    Defun,
    Defmacro,
    Lambda,
    If,
    Set,
    Begin,
    Car,
    Cdr,
    Nullp,
    Eq,
    Not,
    And,
    Or,
    Xor,
    BinNot,
    BinOr,
    BinAnd,
    Print,
    Gt,
    Lt,
    Gte,
    Lte,
    Add,
    Sub,
    Mul,
    Div,
    Addr,
    Let,
    List,
    Macroexpand,
}

impl Special {
    /// Look up a special form by name (case-insensitive).
    /// Returns None for user-defined symbols.
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("add") {
            return Some(Self::Add);
        }
        if name.eq_ignore_ascii_case("sub") {
            return Some(Self::Sub);
        }
        if name.eq_ignore_ascii_case("mul") {
            return Some(Self::Mul);
        }
        if name.eq_ignore_ascii_case("div") {
            return Some(Self::Div);
        }
        if name.eq_ignore_ascii_case("gt") {
            return Some(Self::Gt);
        }
        if name.eq_ignore_ascii_case("lt") {
            return Some(Self::Lt);
        }
        if name.eq_ignore_ascii_case("gte") {
            return Some(Self::Gte);
        }
        if name.eq_ignore_ascii_case("lte") {
            return Some(Self::Lte);
        }
        if name.eq_ignore_ascii_case("eq") {
            return Some(Self::Eq);
        }
        if name.eq_ignore_ascii_case("not") {
            return Some(Self::Not);
        }
        if name.eq_ignore_ascii_case("and") {
            return Some(Self::And);
        }
        if name.eq_ignore_ascii_case("or") {
            return Some(Self::Or);
        }
        if name.eq_ignore_ascii_case("xor") {
            return Some(Self::Xor);
        }
        if name.eq_ignore_ascii_case("binnot") {
            return Some(Self::BinNot);
        }
        if name.eq_ignore_ascii_case("binor") {
            return Some(Self::BinOr);
        }
        if name.eq_ignore_ascii_case("binand") {
            return Some(Self::BinAnd);
        }
        if name.eq_ignore_ascii_case("defun") {
            return Some(Self::Defun);
        }
        if name.eq_ignore_ascii_case("defmacro") {
            return Some(Self::Defmacro);
        }
        if name.eq_ignore_ascii_case("lambda") || name.eq_ignore_ascii_case("fn") {
            return Some(Self::Lambda);
        }
        if name.eq_ignore_ascii_case("if") {
            return Some(Self::If);
        }
        if name.eq_ignore_ascii_case("set") {
            return Some(Self::Set);
        }
        if name.eq_ignore_ascii_case("begin") {
            return Some(Self::Begin);
        }
        if name.eq_ignore_ascii_case("car") {
            return Some(Self::Car);
        }
        if name.eq_ignore_ascii_case("cdr") {
            return Some(Self::Cdr);
        }
        if name.eq_ignore_ascii_case("null?") || name.eq_ignore_ascii_case("nullp") {
            return Some(Self::Nullp);
        }
        if name.eq_ignore_ascii_case("print") {
            return Some(Self::Print);
        }
        if name.eq_ignore_ascii_case("addr") {
            return Some(Self::Addr);
        }
        if name.eq_ignore_ascii_case("let") {
            return Some(Self::Let);
        }
        if name.eq_ignore_ascii_case("list") {
            return Some(Self::List);
        }
        if name.eq_ignore_ascii_case("macroexpand") {
            return Some(Self::Macroexpand);
        }
        None
    }
}

/// Extract and evaluate two numeric arguments from sexp (special left right).
fn extract_numeric_binop(
    sexp: Rc<Value>,
    image: &mut Image,
) -> Result<(Number, Number), &'static str> {
    let left = sexp.nth(1);
    let right = sexp.nth(2);

    if right.is_nil() {
        return Err("Expected 2 arguments, got fewer.");
    }
    if !sexp.nth(3).is_nil() {
        return Err("Expected 2 arguments, got more.");
    }

    let left_val = evaluate(left, image)?.0;
    let right_val = evaluate(right, image)?.0;

    let l = match &left_val {
        Value::Number(n) => *n,
        _ => return Err("Left operand is not a number."),
    };
    let r = match &right_val {
        Value::Number(n) => *n,
        _ => return Err("Right operand is not a number."),
    };

    Ok((l, r))
}

/// Extract and evaluate a single argument of any type from sexp (special arg).
fn extract_unary(sexp: Rc<Value>, image: &mut Image) -> Result<Value, &'static str> {
    let arg = sexp.nth(1);
    if arg.is_nil() {
        return Err("Expected 1 argument, got none.");
    }
    if !sexp.nth(2).is_nil() {
        return Err("Expected 1 argument, got more.");
    }
    Ok(evaluate(arg, image)?.0)
}

/// Extract and evaluate a single numeric argument from sexp (special arg).
/// Delegates to extract_unary, then checks that the result is a number.
fn extract_numeric_unary(sexp: Rc<Value>, image: &mut Image) -> Result<Number, &'static str> {
    match &extract_unary(sexp, image)? {
        Value::Number(n) => Ok(*n),
        _ => Err("Argument is not a number."),
    }
}

/// A value is falsy if it is nil, false, or integer 0.
pub fn is_falsy(v: &Value) -> bool {
    matches!(
        v,
        Value::Nil | Value::Bool(false) | Value::Number(Number::Integer(0))
    )
}

/// Walk a cons list and collect each element as an Rc<Symbol>.
/// Nil (empty list) returns an empty Vec.
fn collect_symbol_list(list: &Value) -> Result<Vec<Rc<Symbol>>, &'static str> {
    let mut params = Vec::new();
    let mut current = list;
    loop {
        match current {
            Value::Nil => break,
            Value::Cons(car, cdr) => {
                match &**car {
                    Value::Symbol(s) => params.push(Rc::clone(s)),
                    _ => return Err("Expected symbol in parameter list."),
                }
                current = cdr;
            }
            _ => return Err("Malformed parameter list (not a proper list)."),
        }
    }
    Ok(params)
}

pub fn execute_special(
    form: Special,
    sexp: Rc<Value>,
    image: &mut Image,
) -> Result<(Value, Environment), &'static str> {
    let env = image.e.clone();
    match form {
        // --- comparators ---
        Special::Gt => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Bool(l > r), env))
        }
        Special::Lt => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Bool(l < r), env))
        }
        Special::Gte => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Bool(l >= r), env))
        }
        Special::Lte => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Bool(l <= r), env))
        }
        Special::Eq => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Bool(l == r), env))
        }

        // --- arithmetic ---
        Special::Add => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Number(l.add(r)?), env))
        }
        Special::Sub => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Number(l.sub(r)?), env))
        }
        Special::Mul => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Number(l.mul(r)?), env))
        }
        Special::Div => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Number(l.div(r)?), env))
        }

        // --- type coercion ---
        Special::Addr => {
            let n = extract_numeric_unary(sexp, image)?;
            Ok((Value::Number(n.as_addr()?), env))
        }

        // --- logic ---

        // `not`: logical negation.
        //   bool  → flipped bool
        //   int 0 → Integer(1), non-zero int → Integer(0)
        //   everything else → error
        Special::Not => {
            let val = extract_unary(sexp, image)?;
            match &val {
                Value::Bool(b) => Ok((Value::Bool(!b), env)),
                Value::Number(Number::Integer(0)) => Ok((Value::Number(Number::Integer(1)), env)),
                Value::Number(Number::Integer(_)) => Ok((Value::Number(Number::Integer(0)), env)),
                _ => Err("not: expected bool or integer."),
            }
        }

        // `binnot`: bitwise NOT on an integer.
        Special::BinNot => {
            let n = extract_numeric_unary(sexp, image)?;
            let i = n.as_i32()?;
            Ok((Value::Number(Number::Integer(!i)), env))
        }

        // `binor`: bitwise OR on two integers.
        Special::BinOr => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            let li = l.as_i32()?;
            let ri = r.as_i32()?;
            Ok((Value::Number(Number::Integer(li | ri)), env))
        }

        // `binand`: bitwise AND on two integers.
        Special::BinAnd => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            let li = l.as_i32()?;
            let ri = r.as_i32()?;
            Ok((Value::Number(Number::Integer(li & ri)), env))
        }

        // --- list ops ---

        // `car`: head of a cons cell. nil if not a cons.
        Special::Car => {
            let val = extract_unary(sexp, image)?;
            Ok(((*val.car()).clone(), env))
        }

        // `cdr`: tail of a cons cell. nil if not a cons.
        Special::Cdr => {
            let val = extract_unary(sexp, image)?;
            Ok(((*val.cdr()).clone(), env))
        }

        // `nullp`: true iff the argument is exactly nil.
        Special::Nullp => {
            let val = extract_unary(sexp, image)?;
            Ok((Value::Bool(val.is_nil()), env))
        }

        // `list` makes a list
        Special::List => {
            let mut result = Value::Nil;
            let mut i = 1;
            loop {
                let arg = sexp.nth(i);
                if arg.is_nil() {
                    break;
                }
                let val = evaluate(arg, image)?.0;
                result = Value::cons(val, result);
                i += 1;
            }
            Ok((result, env))
        }

        // --- IO ---

        // `print`: prints to UART. First arg is a format string, remaining
        // args are substituted for `{}` placeholders (left to right).
        // Returns nil. If no format string, prints each arg separated by spaces.
        //
        // Examples:
        //   (print "hello")           → hello
        //   (print "x = {}" 42)       → x = 42
        //   (print "{} + {} = {}" 1 2 3) → 1 + 2 = 3
        Special::Print => {
            // collect and evaluate all arguments
            let mut args: Vec<Value> = Vec::new();
            let mut i = 1;
            loop {
                let arg = sexp.nth(i);
                if arg.is_nil() {
                    break;
                }
                args.push(evaluate(arg, image)?.0);
                i += 1;
            }

            if args.is_empty() {
                crate::println!();
            } else if let Value::String(ref fmt_str) = args[0] {
                // format string mode: replace {} with successive args
                let mut result = AllocString::new();
                let mut arg_idx = 1;
                let mut chars = fmt_str.chars().peekable();
                while let Some(ch) = chars.next() {
                    if ch == '{' && chars.peek() == Some(&'}') {
                        chars.next(); // consume '}'
                        if arg_idx < args.len() {
                            use core::fmt::Write;
                            let _ = write!(result, "{}", args[arg_idx]);
                            arg_idx += 1;
                        } else {
                            result.push_str("{}");
                        }
                    } else {
                        result.push(ch);
                    }
                }
                crate::println!("{}", result);
            } else {
                // no format string: print all args space-separated
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        crate::print!(" ");
                    }
                    crate::print!("{}", arg);
                }
                crate::println!();
            }

            Ok((Value::Nil, env))
        }

        // --- binding / closures ---

        // `lambda`: create a closure that captures the current environment.
        //   (lambda (params...) body)
        // params is a cons list of symbols (may be empty / nil).
        // body is a single sexp (not evaluated until the closure is called).
        Special::Lambda => {
            let param_list = sexp.nth(1);
            let body = sexp.nth(2);
            if body.is_nil() {
                return Err("lambda: missing body.");
            }
            if !sexp.nth(3).is_nil() {
                return Err("lambda: too many arguments (expected params and body).");
            }

            let params = collect_symbol_list(&param_list)?;

            Ok((
                Value::Closure(Closure {
                    params,
                    body,
                    env: env.clone(),
                }),
                env,
            ))
        }

        // `set`: bind a name in the current environment.
        //   (set name value)
        // name is a symbol (not evaluated). value is evaluated.
        // If the name already has a Binding, mutates it in place (visible
        // to all scopes sharing that Binding). Otherwise creates a new one.
        Special::Set => {
            let name_val = sexp.nth(1);
            let val_expr = sexp.nth(2);
            if name_val.is_nil() || val_expr.is_nil() {
                return Err("set: expected name and value.");
            }
            if !sexp.nth(3).is_nil() {
                return Err("set: too many arguments.");
            }

            let name = match &*name_val {
                Value::Symbol(s) => (**s).clone(),
                _ => return Err("set: first argument must be a symbol."),
            };

            let val = evaluate(val_expr, image)?.0;

            // if binding exists, mutate in place; otherwise create fresh
            if let Some(binding) = image.binding(&name) {
                *binding.borrow_mut() = Rc::new(val.clone());
            } else {
                image.insert(name, Rc::new(val.clone()));
            }

            let env = image.e.clone();
            Ok((val, env))
        }

        // `defun`: desugars (defun name (params) body) into
        // (set name (lambda (params) body)) and evaluates that.
        Special::Defun => {
            let name = sexp.nth(1);
            let params = sexp.nth(2);
            let body = sexp.nth(3);
            if name.is_nil() || body.is_nil() {
                return Err("defun: expected name, params, and body.");
            }
            if !sexp.nth(4).is_nil() {
                return Err("defun: too many arguments.");
            }

            // build (set name (lambda (params) body))
            let lambda_sexp = Value::cons(
                Value::Special(Special::Lambda),
                Value::cons((*params).clone(), Value::cons((*body).clone(), Value::Nil)),
            );
            let set_sexp = Value::cons(
                Value::Special(Special::Set),
                Value::cons((*name).clone(), Value::cons(lambda_sexp, Value::Nil)),
            );

            evaluate(Rc::new(set_sexp), image)
        }

        // --- short-circuit logic ---

        // `and`: short-circuit. Eval first arg; if falsy, return it.
        // Otherwise eval and return second arg.
        Special::And => {
            let left = sexp.nth(1);
            let right = sexp.nth(2);
            if left.is_nil() || right.is_nil() {
                return Err("and: expected 2 arguments.");
            }
            let l = evaluate(left, image)?.0;
            if is_falsy(&l) {
                Ok((l, image.e.clone()))
            } else {
                let r = evaluate(right, image)?.0;
                Ok((r, image.e.clone()))
            }
        }

        // `or`: short-circuit. Eval first arg; if truthy, return it.
        // Otherwise eval and return second arg.
        Special::Or => {
            let left = sexp.nth(1);
            let right = sexp.nth(2);
            if left.is_nil() || right.is_nil() {
                return Err("or: expected 2 arguments.");
            }
            let l = evaluate(left, image)?.0;
            if !is_falsy(&l) {
                Ok((l, image.e.clone()))
            } else {
                let r = evaluate(right, image)?.0;
                Ok((r, image.e.clone()))
            }
        }

        // `xor`: eval both args, return truthy iff exactly one is truthy.
        Special::Xor => {
            let left = sexp.nth(1);
            let right = sexp.nth(2);
            if left.is_nil() || right.is_nil() {
                return Err("xor: expected 2 arguments.");
            }
            let l = evaluate(left, image)?.0;
            let r = evaluate(right, image)?.0;
            Ok((Value::Bool(is_falsy(&l) != is_falsy(&r)), image.e.clone()))
        }

        // --- control flow ---

        // `if`: (if cond then else)
        // Evaluates cond; if truthy, evaluates and returns then;
        // otherwise evaluates and returns else.
        Special::If => {
            let cond = sexp.nth(1);
            let then_branch = sexp.nth(2);
            let else_branch = sexp.nth(3);
            if cond.is_nil() || then_branch.is_nil() || else_branch.is_nil() {
                return Err("if: expected condition, then, and else.");
            }
            let c = evaluate(cond, image)?.0;
            if !is_falsy(&c) {
                let v = evaluate(then_branch, image)?.0;
                Ok((v, image.e.clone()))
            } else {
                let v = evaluate(else_branch, image)?.0;
                Ok((v, image.e.clone()))
            }
        }

        // `begin`: evaluate each element seperately, returning the last value
        Special::Begin => {
            let mut last_val = Value::Nil;
            let mut i = 1;
            loop {
                let arg = sexp.nth(i);
                if arg.is_nil() {
                    break;
                }
                last_val = evaluate(arg, image)?.0;
                i += 1;
            }
            Ok((last_val, image.e.clone()))
        }
        // `let`: introduce local bindings, then evaluate body in that scope.
        //   (let (a 1 b 2 c 3) body)
        // The binding list is a flat cons list of name/value pairs.
        // Each value is evaluated in order (left to right) so that earlier
        // bindings are visible to later values. After body executes, the
        // original environment is restored — bindings don't leak out.
        Special::Let => {
            let bindings_list = sexp.nth(1);
            let body = sexp.nth(2);
            if body.is_nil() {
                return Err("let: expected bindings and body.");
            }
            if !sexp.nth(3).is_nil() {
                return Err("let: too many arguments.");
            }

            // save the current environment so we can restore it after
            let orig_env = image.e.clone();

            // walk the flat binding list: (name1 val1 name2 val2 ...)
            let mut current: &Value = &bindings_list;
            loop {
                match current {
                    Value::Nil => break,
                    Value::Cons(name_rc, rest) => {
                        // name must be a symbol
                        let name = match &**name_rc {
                            Value::Symbol(s) => (**s).clone(),
                            _ => return Err("let: binding name must be a symbol."),
                        };

                        // next element is the value expression
                        let (val_expr, tail) = match &**rest {
                            Value::Cons(val, tail) => (Rc::clone(val), &**tail),
                            _ => return Err("let: odd number of elements in binding list."),
                        };

                        // evaluate value in current (evolving) environment
                        let val = evaluate(val_expr, image)?.0;
                        image.insert(name, Rc::new(val));

                        current = tail;
                    }
                    _ => return Err("let: malformed binding list."),
                }
            }

            // evaluate body in the extended environment
            let result = evaluate(body, image)?.0;

            // restore original environment — let bindings don't escape
            // but side effects (mutations to shared Bindings) do
            image.e = orig_env;

            Ok((result, image.e.clone()))
        }

        // --- macros ---

        // `defmacro`: define a macro.
        //   (defmacro name (params...) body)
        // Creates a Macro wrapping a closure. When called, the macro's
        // closure receives unevaluated sexps as arguments, returns an
        // expanded sexp, which is then executed.
        Special::Defmacro => {
            let name_val = sexp.nth(1);
            let param_list = sexp.nth(2);
            let body = sexp.nth(3);
            if name_val.is_nil() || body.is_nil() {
                return Err("defmacro: expected name, params, and body.");
            }
            if !sexp.nth(4).is_nil() {
                return Err("defmacro: too many arguments.");
            }

            let name = match &*name_val {
                Value::Symbol(s) => (**s).clone(),
                _ => return Err("defmacro: first argument must be a symbol."),
            };

            let params = collect_symbol_list(&param_list)?;

            let closure = Closure {
                params: params.clone(),
                body,
                env: env.clone(),
            };

            let mac = Value::Macro(Macro { params, closure });

            if let Some(binding) = image.binding(&name) {
                *binding.borrow_mut() = Rc::new(mac.clone());
            } else {
                image.insert(name, Rc::new(mac.clone()));
            }

            let env = image.e.clone();
            Ok((mac, env))
        }

        // `macroexpand`: call the macro's closure with unevaluated args
        // and return the expanded sexp as a value (without executing it).
        //   (macroexpand (some-macro arg1 arg2))
        // Useful for debugging macros.
        Special::Macroexpand => {
            let arg = sexp.nth(1);
            if arg.is_nil() {
                return Err("macroexpand: expected 1 argument.");
            }
            if !sexp.nth(2).is_nil() {
                return Err("macroexpand: too many arguments.");
            }

            // arg should be a macro call sexp like (my-macro x y)
            // resolve the head to find the macro
            let head = arg.car();
            let mac_val = match &*head {
                Value::Symbol(s) => match image.get(s) {
                    Some(v) => v,
                    None => return Err("macroexpand: unknown symbol."),
                },
                _ => head,
            };

            let m = match &*mac_val {
                Value::Macro(m) => m.clone(),
                _ => return Err("macroexpand: argument is not a macro call."),
            };

            // collect unevaluated args
            let arg_vals: Vec<Rc<Value>> = m
                .params
                .iter()
                .enumerate()
                .map(|(i, _)| arg.nth(i + 1))
                .collect();

            // call the closure to get the expanded form
            let orig_env = image.e.clone();
            let mut temp_env = orig_env.clone();
            m.closure.env.iter().for_each(|(k, v)| {
                temp_env.insert(k.clone(), Rc::clone(v));
            });
            image.e = temp_env;
            m.params
                .iter()
                .zip(arg_vals.into_iter())
                .for_each(|(param, val)| {
                    image.insert((**param).clone(), val);
                });

            let expanded = evaluate(Rc::clone(&m.closure.body), image)?.0;
            image.e = orig_env;

            // return the expanded sexp as a value, don't execute it
            Ok((expanded, image.e.clone()))
        }
    }
}
