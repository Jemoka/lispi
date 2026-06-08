//! Type-specialized, JIT-compiled closures.
//!
//! `(jit <closure-expr> <dummy1> <dummy2> ...)` evaluates the closure
//! and dummy inputs, derives a type signature from the dummies, binds
//! each parameter to its dummy, then runs the JIT pipeline on the
//! closure's body. The result is a `Value::JittedClosure` wrapping
//! the compiled artifact.
//!
//! Calling a `JittedClosure` from the interpreter:
//!   1. Evaluate the actual args.
//!   2. Verify each arg's `InputType` matches the one stored at compile
//!      time (else bail — the body was specialized).
//!   3. Overwrite the pinned param `Binding` cells with the actual args.
//!      This is the load-bearing trick: the JIT-emitted code holds raw
//!      pointers into those Bindings (via `LoadCapture`); writing
//!      through `RefCell::borrow_mut()` keeps the pointers stable and
//!      makes the new values immediately visible.
//!   4. Run the executor.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

use crate::language::ast::{Symbol, Value};
use crate::language::environment::{Binding, Environment};
use crate::language::number::Number;

use super::executor::JitExecutor;

/// Input type tag for a jitted closure's parameter slot.
///
/// We discriminate the four numeric / scalar shapes the JIT actually
/// specializes on, plus a catch-all `Heap` bucket for cons/array/
/// closure/etc. (the JIT treats those as opaque shadow-slot refs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputType {
    Integer,
    Unsigned,
    Addr,
    Bool,
    Nil,
    Heap,
}

impl InputType {
    /// Derive an `InputType` from a runtime `Value`.
    pub fn of(v: &Value) -> Self {
        match v {
            Value::Number(Number::Integer(_)) => InputType::Integer,
            Value::Number(Number::Unsigned(_)) => InputType::Unsigned,
            Value::Number(Number::Addr(_)) => InputType::Addr,
            Value::Bool(_) => InputType::Bool,
            Value::Nil => InputType::Nil,
            _ => InputType::Heap,
        }
    }

    /// True iff `v` is acceptable as input for a slot of this declared
    /// type. Exact match on the discriminant; widening / coercion would
    /// invalidate the JIT's type-specialized body.
    pub fn accepts(&self, v: &Value) -> bool {
        Self::of(v) == *self
    }
}

/// A closure whose body has been compiled to LIR + machine code.
///
/// The `executor` is wrapped in `RefCell` so it can be `run()`
/// through the immutable `&JittedClosure` we get out of an `Rc`.
pub struct JittedClosure {
    pub params: Vec<Rc<Symbol>>,
    /// The same `Binding` cells the JIT's emitted code holds raw
    /// pointers to (via `LoadCapture` literal-pool entries). Mutating
    /// them in place at call time is how we feed new args to the
    /// compiled body without recompiling.
    pub param_bindings: Vec<Binding>,
    /// Per-param type guard derived from the `(jit ...)` dummies.
    pub input_types: Vec<InputType>,
    /// The closure's captured environment at JIT-compile time. We
    /// re-push this onto the runtime `Image` at each call so that any
    /// interpreter dispatch invoked via `Escape` from inside the
    /// compiled body resolves names the same way the JIT did.
    pub env: Rc<Environment>,
    pub executor: RefCell<JitExecutor>,
}

impl fmt::Debug for JittedClosure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<jitted-closure: arity={}>", self.params.len())
    }
}

impl PartialEq for JittedClosure {
    /// Two `JittedClosure`s are never structurally equal — the only
    /// meaningful notion is identity, which the enclosing
    /// `Value::JittedClosure(Rc<_>)` handles via `Rc::ptr_eq` if a
    /// caller needs it.
    fn eq(&self, _: &Self) -> bool { false }
}
