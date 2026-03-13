//! Special form definitions and implementations for the LISP interpreter.

use alloc::rc::Rc;
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
    Gt,
    Lt,
    Gte,
    Lte,
    Add,
    Sub,
    Mul,
    Div,
    Addr,
    Signed,
    Unsigned,
    Let,
    List,
    Macroexpand,
    Lshift,
    Rshift,
    Mod,
    Cons,
    Array,
    Full,
    Unpack,
    GetIdx,
    PutIdx,
    ReadIdx,
    FillIdx,
    FullIdx,
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
        if name.eq_ignore_ascii_case("addr") {
            return Some(Self::Addr);
        }
        if name.eq_ignore_ascii_case("signed") {
            return Some(Self::Signed);
        }
        if name.eq_ignore_ascii_case("unsigned") {
            return Some(Self::Unsigned);
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
        if name.eq_ignore_ascii_case("cons") {
            return Some(Self::Cons);
        }
        if name.eq_ignore_ascii_case("lshift") {
            return Some(Self::Lshift);
        }
        if name.eq_ignore_ascii_case("rshift") {
            return Some(Self::Rshift);
        }
        if name.eq_ignore_ascii_case("mod") {
            return Some(Self::Mod);
        }
        if name.eq_ignore_ascii_case("array") {
            return Some(Self::Array);
        }
        if name.eq_ignore_ascii_case("full") {
            return Some(Self::Full);
        }
        if name.eq_ignore_ascii_case("unpack") {
            return Some(Self::Unpack);
        }
        if name.eq_ignore_ascii_case("getidx") {
            return Some(Self::GetIdx);
        }
        if name.eq_ignore_ascii_case("putidx") {
            return Some(Self::PutIdx);
        }
        if name.eq_ignore_ascii_case("readidx") {
            return Some(Self::ReadIdx);
        }
        if name.eq_ignore_ascii_case("fillidx") {
            return Some(Self::FillIdx);
        }
        if name.eq_ignore_ascii_case("fullidx") {
            return Some(Self::FullIdx);
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

    if !sexp.nth_exists(2) {
        return Err("Expected 2 arguments, got fewer.");
    }
    if sexp.nth_exists(3) {
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
    if !sexp.nth_exists(1) {
        return Err("Expected 1 argument, got none.");
    }
    if sexp.nth_exists(2) {
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

/// Extract a usize from an already-evaluated Value.
fn extract_usize(val: &Value, ctx: &'static str) -> Result<usize, &'static str> {
    if let Value::Number(n) = val {
        Ok(n.as_i32().map_err(|_| ctx)? as usize)
    } else {
        Err(ctx)
    }
}

/// Extract a u32 from an already-evaluated Value.
fn extract_u32(val: &Value, ctx: &'static str) -> Result<u32, &'static str> {
    if let Value::Number(n) = val {
        n.as_u32().map_err(|_| ctx)
    } else {
        Err(ctx)
    }
}

/// Extract a raw *mut u32 base pointer from a Number (must be Addr).
/// The returned pointer is to the base; callers offset by index * sizeof(u32).
fn extract_addr(n: &Number, ctx: &'static str) -> Result<*mut u32, &'static str> {
    let a = n.as_addr().map_err(|_| ctx)?;
    if let Number::Addr(a) = a {
        Ok(a as *mut u32)
    } else {
        Err(ctx)
    }
}

/// A value is falsy if it is nil, false, or integer 0.
pub fn is_falsy(v: &Value) -> bool {
    matches!(
        v,
        Value::Nil
            | Value::Bool(false)
            | Value::Number(Number::Integer(0))
            | Value::Number(Number::Unsigned(0))
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
        Special::Mod => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Number(l.modulo(r)?), env))
        }
        Special::Lshift => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Number(l.lshift(r)?), env))
        }
        Special::Rshift => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            Ok((Value::Number(l.rshift(r)?), env))
        }

        // --- type coercion ---
        Special::Addr => {
            let n = extract_numeric_unary(sexp, image)?;
            Ok((Value::Number(n.as_addr()?), env))
        }
        Special::Signed => {
            let n = extract_numeric_unary(sexp, image)?;
            Ok((Value::Number(Number::Integer(n.as_i32()?)), env))
        }
        Special::Unsigned => {
            let n = extract_numeric_unary(sexp, image)?;
            Ok((Value::Number(Number::Unsigned(n.as_u32()?)), env))
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
                Value::Number(Number::Unsigned(0)) => Ok((Value::Number(Number::Unsigned(1)), env)),
                Value::Number(Number::Unsigned(_)) => Ok((Value::Number(Number::Unsigned(0)), env)),
                _ => Err("not: expected bool or integer."),
            }
        }

        // `binnot`: bitwise NOT on an integer or unsigned.
        Special::BinNot => {
            let n = extract_numeric_unary(sexp, image)?;
            match n {
                Number::Integer(i) => Ok((Value::Number(Number::Integer(!i)), env)),
                Number::Unsigned(u) => Ok((Value::Number(Number::Unsigned(!u)), env)),
                _ => Err("binnot: expected integer or unsigned."),
            }
        }

        // `binor`: bitwise OR on integers/unsigned.
        Special::BinOr => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            match (l, r) {
                (Number::Integer(a), Number::Integer(b)) => {
                    Ok((Value::Number(Number::Integer(a | b)), env))
                }
                (Number::Unsigned(a), Number::Unsigned(b)) => {
                    Ok((Value::Number(Number::Unsigned(a | b)), env))
                }
                (Number::Unsigned(a), Number::Integer(b)) => {
                    Ok((Value::Number(Number::Unsigned(a | b as u32)), env))
                }
                (Number::Integer(a), Number::Unsigned(b)) => {
                    Ok((Value::Number(Number::Unsigned(a as u32 | b)), env))
                }
                _ => Err("binor: expected integers or unsigned."),
            }
        }

        // `binand`: bitwise AND on integers/unsigned.
        Special::BinAnd => {
            let (l, r) = extract_numeric_binop(sexp, image)?;
            match (l, r) {
                (Number::Integer(a), Number::Integer(b)) => {
                    Ok((Value::Number(Number::Integer(a & b)), env))
                }
                (Number::Unsigned(a), Number::Unsigned(b)) => {
                    Ok((Value::Number(Number::Unsigned(a & b)), env))
                }
                (Number::Unsigned(a), Number::Integer(b)) => {
                    Ok((Value::Number(Number::Unsigned(a & b as u32)), env))
                }
                (Number::Integer(a), Number::Unsigned(b)) => {
                    Ok((Value::Number(Number::Unsigned(a as u32 & b)), env))
                }
                _ => Err("binand: expected integers or unsigned."),
            }
        }

        // --- list ops ---

        // `cons`: construct a cons cell from two evaluated arguments.
        Special::Cons => {
            let left = sexp.nth(1);
            let right = sexp.nth(2);
            if !sexp.nth_exists(2) {
                return Err("cons: expected 2 arguments.");
            }
            if sexp.nth_exists(3) {
                return Err("cons: too many arguments.");
            }
            let l = evaluate(left, image)?.0;
            let r = evaluate(right, image)?.0;
            Ok((Value::cons(l, r), env))
        }

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
            let mut vals = Vec::new();
            let mut i = 1;
            loop {
                let arg = sexp.nth(i);
                if arg.is_nil() {
                    break;
                }
                vals.push(evaluate(arg, image)?.0);
                i += 1;
            }
            let mut result = Value::Nil;
            for val in vals.into_iter().rev() {
                result = Value::cons(val, result);
            }
            Ok((result, env))
        }

        // --- binding / closures ---

        // `lambda`: create a closure that captures the current environment.
        //   (lambda (params...) body)
        // params is a cons list of symbols (may be empty / nil).
        // body is a single sexp (not evaluated until the closure is called).
        Special::Lambda => {
            let param_list = sexp.nth(1);
            let body = sexp.nth(2);
            if !sexp.nth_exists(2) {
                return Err("lambda: missing body.");
            }
            if sexp.nth_exists(3) {
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
            if !sexp.nth_exists(2) {
                return Err("set: expected name and value.");
            }
            if sexp.nth_exists(3) {
                return Err("set: too many arguments.");
            }

            let name = match &*name_val {
                Value::Symbol(s) => (**s).clone(),
                _ => return Err("set: first argument must be a symbol."),
            };

            // pre-create binding so that the value expression (e.g. a lambda)
            // captures the slot — enabling self-recursion via shared RefCell
            if image.binding(&name).is_none() {
                image.insert(name.clone(), val_expr.clone());
            }

            let val = evaluate(val_expr, image)?.0;

            // mutate in-place — visible to any closure that captured this binding
            *image.binding(&name).unwrap().borrow_mut() = Rc::new(val.clone());

            let env = image.e.clone();
            Ok((val, env))
        }

        // `defun`: desugars (defun name (params) body) into
        // (set name (lambda (params) body)) and evaluates that.
        Special::Defun => {
            let name = sexp.nth(1);
            let params = sexp.nth(2);
            let body = sexp.nth(3);
            if !sexp.nth_exists(3) {
                return Err("defun: expected name, params, and body.");
            }
            if sexp.nth_exists(4) {
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
            if !sexp.nth_exists(2) {
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
            if !sexp.nth_exists(2) {
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
            if !sexp.nth_exists(2) {
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
            if !sexp.nth_exists(3) {
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
            if !sexp.nth_exists(2) {
                return Err("let: expected bindings and body.");
            }
            if sexp.nth_exists(3) {
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
            if !sexp.nth_exists(3) {
                return Err("defmacro: expected name, params, and body.");
            }
            if sexp.nth_exists(4) {
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
            if !sexp.nth_exists(1) {
                return Err("macroexpand: expected 1 argument.");
            }
            if sexp.nth_exists(2) {
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

        // --- arrays ---

        // `(array list)` — convert a lisp list of numbers to a Value::Array
        Special::Array => {
            let val = extract_unary(sexp, image)?;
            let mut v = Vec::new();
            let mut cur = val;
            loop {
                match &cur {
                    Value::Nil => break,
                    Value::Cons(head, tail) => {
                        if let Value::Number(n) = head.as_ref() {
                            let u = n.as_u32().map_err(|_| "array: elements must be u32.")?;
                            v.push(u);
                            cur = tail.as_ref().clone();
                        } else {
                            return Err("array: elements must be numbers.");
                        }
                    }
                    _ => return Err("array: argument must be a list."),
                }
            }
            Ok((Value::array(v), env))
        }

        // `(full n value)` — create a new array of n copies of value
        Special::Full => {
            let (l, r) = extract_numeric_binop(sexp.clone(), image)?;
            let n = l.as_i32().map_err(|_| "full: first arg must be an integer.")? as usize;
            let val = r.as_u32().map_err(|_| "full: second arg must be a u32.")?;
            Ok((Value::array_fill(n, val), env))
        }

        // `(unpack array)` — convert a Value::Array back to a lisp list
        Special::Unpack => {
            let val = extract_unary(sexp, image)?;
            if let Value::Array(a) = &val {
                let borrowed = a.borrow();
                let mut result = Value::Nil;
                for u in borrowed.iter().rev() {
                    result = Value::cons(
                        Value::Number(Number::Unsigned(*u)),
                        result,
                    );
                }
                Ok((result, env))
            } else {
                Err("unpack: argument must be an array.")
            }
        }

        // `(getidx target n)` — get u32 at index n.
        // target: Array (bounds-checked) or Addr (raw, reads at addr+n*4).
        Special::GetIdx => {
            let target = evaluate(sexp.nth(1), image)?.0;
            let idx_val = evaluate(sexp.nth(2), image)?.0;
            let i = extract_usize(&idx_val, "getidx: index")?;
            let val = match &target {
                Value::Array(a) => {
                    let b = a.borrow();
                    if i >= b.len() { return Err("getidx: index out of bounds."); }
                    b[i]
                }
                Value::Number(n) => {
                    let base = extract_addr(n, "getidx: first arg")?;
                    unsafe { *(base.wrapping_add(i) as *const u32) }
                }
                _ => return Err("getidx: first arg must be an array or address."),
            };
            Ok((Value::Number(Number::Unsigned(val)), env))
        }

        // `(putidx target n val)` — set u32 at index n.
        // target: Array (bounds-checked, mutates in place) or Addr (raw write at addr+n*4).
        Special::PutIdx => {
            let target = evaluate(sexp.nth(1), image)?.0;
            let idx_val = evaluate(sexp.nth(2), image)?.0;
            let val_val = evaluate(sexp.nth(3), image)?.0;
            let i = extract_usize(&idx_val, "putidx: index")?;
            let val = extract_u32(&val_val, "putidx: value")?;
            match &target {
                Value::Array(a) => {
                    let mut b = a.borrow_mut();
                    if i >= b.len() { return Err("putidx: index out of bounds."); }
                    b[i] = val;
                }
                Value::Number(n) => {
                    let base = extract_addr(n, "putidx: first arg")?;
                    unsafe { *(base.wrapping_add(i) as *mut u32) = val; }
                }
                _ => return Err("putidx: first arg must be an array or address."),
            }
            Ok((Value::Nil, env))
        }

        // `(readidx target offset n)` — read n u32s starting at offset into a list.
        // target: Array (bounds-checked) or Addr (raw, reads at addr+(offset+i)*4).
        Special::ReadIdx => {
            let target = evaluate(sexp.nth(1), image)?.0;
            let off_val = evaluate(sexp.nth(2), image)?.0;
            let n_val = evaluate(sexp.nth(3), image)?.0;
            let offset = extract_usize(&off_val, "readidx: offset")?;
            let count = extract_usize(&n_val, "readidx: count")?;
            let mut result = Value::Nil;
            match &target {
                Value::Array(a) => {
                    let b = a.borrow();
                    if offset + count > b.len() { return Err("readidx: range out of bounds."); }
                    for i in (0..count).rev() {
                        result = Value::cons(
                            Value::Number(Number::Unsigned(b[offset + i])),
                            result,
                        );
                    }
                }
                Value::Number(n) => {
                    let base = extract_addr(n, "readidx: first arg")?;
                    for i in (0..count).rev() {
                        let val = unsafe { *(base.wrapping_add(offset + i) as *const u32) };
                        result = Value::cons(
                            Value::Number(Number::Unsigned(val)),
                            result,
                        );
                    }
                }
                _ => return Err("readidx: first arg must be an array or address."),
            }
            Ok((result, env))
        }

        // `(fillidx target offset list)` — write list values starting at offset.
        // target: Array (bounds-checked) or Addr (raw write at addr+(offset+i)*4).
        Special::FillIdx => {
            let target = evaluate(sexp.nth(1), image)?.0;
            let off_val = evaluate(sexp.nth(2), image)?.0;
            let list_val = evaluate(sexp.nth(3), image)?.0;
            let offset = extract_usize(&off_val, "fillidx: offset")?;
            match &target {
                Value::Array(a) => {
                    let mut b = a.borrow_mut();
                    let mut cur = list_val;
                    let mut i = 0;
                    loop {
                        match &cur {
                            Value::Nil => break,
                            Value::Cons(head, tail) => {
                                let val = extract_u32(head, "fillidx: list element")?;
                                if offset + i >= b.len() { return Err("fillidx: write out of bounds."); }
                                b[offset + i] = val;
                                i += 1;
                                cur = tail.as_ref().clone();
                            }
                            _ => return Err("fillidx: third arg must be a list."),
                        }
                    }
                }
                Value::Number(n) => {
                    let base = extract_addr(n, "fillidx: first arg")?;
                    let mut cur = list_val;
                    let mut i = 0;
                    loop {
                        match &cur {
                            Value::Nil => break,
                            Value::Cons(head, tail) => {
                                let val = extract_u32(head, "fillidx: list element")?;
                                unsafe { *(base.wrapping_add(offset + i) as *mut u32) = val; }
                                i += 1;
                                cur = tail.as_ref().clone();
                            }
                            _ => return Err("fillidx: third arg must be a list."),
                        }
                    }
                }
                _ => return Err("fillidx: first arg must be an array or address."),
            }
            Ok((Value::Nil, env))
        }

        // `(fullidx target offset n val)` — fill n slots starting at offset with val.
        // target: Array (bounds-checked) or Addr (raw write at addr+(offset+i)*4).
        Special::FullIdx => {
            let target = evaluate(sexp.nth(1), image)?.0;
            let off_val = evaluate(sexp.nth(2), image)?.0;
            let n_val = evaluate(sexp.nth(3), image)?.0;
            let val_val = evaluate(sexp.nth(4), image)?.0;
            let offset = extract_usize(&off_val, "fullidx: offset")?;
            let count = extract_usize(&n_val, "fullidx: count")?;
            let val = extract_u32(&val_val, "fullidx: value")?;
            match &target {
                Value::Array(a) => {
                    let mut b = a.borrow_mut();
                    if offset + count > b.len() { return Err("fullidx: range out of bounds."); }
                    b[offset..offset + count].fill(val);
                }
                Value::Number(n) => {
                    let base = extract_addr(n, "fullidx: first arg")?;
                    let dst = unsafe {
                        core::slice::from_raw_parts_mut(base.wrapping_add(offset), count)
                    };
                    dst.fill(val);
                }
                _ => return Err("fullidx: first arg must be an array or address."),
            }
            Ok((Value::Nil, env))
        }
    }
}
