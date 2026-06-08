//! Type-specialized, JIT-compiled closures and the auto-JIT cache key.
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
//!
//! ## Auto-JIT (descending JIT through call graphs)
//!
//! When a JIT'd body Escapes a call whose head resolves to a
//! `Value::Closure`, we transparently compile that callee, cache the
//! resulting `JittedClosure` on **the currently-running `JitExecutor`**,
//! and dispatch through it instead of the interpreter. The cache lives
//! per-executor so dropping the parent JIT'd artifact reclaims every
//! specialization it accumulated.
//!
//! Cache key (see `CalleeKey`):
//!   `(Rc::as_ptr(&closure.body), Rc::as_ptr(&closure.env), input_types)`
//! Both pointer fields are stable across `Closure::clone()` (which is
//! field-wise `Rc::clone`).

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

use crate::language::ast::{Closure, Symbol, Value};
use crate::language::environment::{Binding, Environment, Image};
use crate::language::number::Number;

use super::executor::JitExecutor;

/// Input type tag for a jitted closure's parameter slot.
///
/// We discriminate the four numeric / scalar shapes the JIT actually
/// specializes on, plus a catch-all `Heap` bucket for cons/array/
/// closure/etc. (the JIT treats those as opaque shadow-slot refs).
#[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
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

/// Cache key for auto-JIT specializations. See module-level docs.
///
/// `body_ptr`/`env_ptr` are stable across `Closure::clone()` because
/// the clone is field-wise `Rc::clone`. Comparing them by `usize`
/// gives Eq+Ord without touching the underlying `Rc<T>`s.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct CalleeKey {
    pub body_ptr: usize,
    pub env_ptr: usize,
    pub types: Vec<InputType>,
}

impl CalleeKey {
    pub(crate) fn from(closure: &Closure, types: Vec<InputType>) -> Self {
        CalleeKey {
            body_ptr: Rc::as_ptr(&closure.body) as usize,
            env_ptr: Rc::as_ptr(&closure.env) as usize,
            types,
        }
    }
}

/// Cache entry — either a compiled JIT artifact or a memoized failure
/// so we don't repeatedly re-attempt compilation of a known-bad shape.
pub(crate) enum CalleeEntry {
    Ready(Rc<JittedClosure>),
    Failed,
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

impl JittedClosure {
    /// Compile a closure's body specialized to the given input types.
    /// Caller must supply `dummies` of the appropriate types (used to
    /// seed the param `Binding`s during cgen). Pushes the closure's
    /// captured env + a fresh owned frame on `image`, mints Bindings,
    /// runs the full pipeline, then pops the frames; the snapshotted
    /// `Binding`s stay alive via the returned struct's `param_bindings`.
    pub(crate) fn compile(
        closure: &Closure,
        dummies: &[Value],
        input_types: Vec<InputType>,
        image: &mut Image,
    ) -> Result<JittedClosure, &'static str> {
        if closure.params.len() != dummies.len() || closure.params.len() != input_types.len() {
            return Err("jit::compile: arity mismatch between closure params and dummies/types.");
        }

        // Push closure's captured env (shared, O(1)) + a new owned frame
        // for the param Bindings.
        image.push_env(&closure.env);
        image.push_frame();
        for (param, dummy) in closure.params.iter().zip(dummies.iter()) {
            image.insert((**param).clone(), Rc::new(dummy.clone()));
        }

        // Snapshot the freshly-created param Bindings so they survive
        // the pop_frame() below; the JIT's emitted code will hold raw
        // pointers into these via `LoadCapture`.
        let param_bindings: Vec<Binding> = closure
            .params
            .iter()
            .map(|p| image.binding(p).expect("just inserted").clone())
            .collect();

        // Compile the body through the full JIT pipeline.
        let mut seg = super::ir::IRSegment::new();
        let mut jit_scope = super::scope::JitImage::new(image);
        let cgen_result = seg.cgen(closure.body.clone(), &mut jit_scope);

        // Pop frames regardless of success — `param_bindings` keeps
        // each RefCell alive past this point.
        image.pop_frame();
        image.pop_frame();

        cgen_result?;
        let optimized = super::optimize::optimize(seg);
        let folded: super::ir::IRSegment = optimized.into();
        let mir: super::ir2::MIRSegment = folded.into();
        let mir2 = super::optimize2::optimize2(mir);
        let rir: super::ir3::RIRSegment = mir2.into();
        let lir = super::regalloc::regalloc(rir);
        let executor = JitExecutor::new(lir);

        Ok(JittedClosure {
            params: closure.params.clone(),
            param_bindings,
            input_types,
            env: Rc::clone(&closure.env),
            executor: RefCell::new(executor),
        })
    }
}

/// Pre-scan helper: does `closure.body` contain a *tail-position*
/// self-call (a cons form `(name args...)` where `name` matches one
/// of the closure's params bound to the closure itself, or — more
/// practically — appears as a free symbol whose lexical occurrence
/// references this closure)?
///
/// We use a conservative structural pre-scan: walk `body` looking
/// for cons forms in tail position, and refuse to JIT if *any*
/// tail-position call's head is a `Symbol`. Reason: we don't have
/// access to a `JitImage` here, so we can't tell whether the symbol
/// resolves to `closure` itself — but in the absence of a smarter
/// analysis, any tail-position call risks turning a TCO'd loop into
/// stack recursion. The user can still force-compile via `(jit ...)`.
///
/// The walk recognizes the tail positions exposed by the existing
/// interpreter (`begin`'s last element, `if`'s then/else branches,
/// `let`'s body) and ignores non-tail subexpressions.
pub(crate) fn has_tail_self_call(closure: &Closure) -> bool {
    use crate::language::special::Special;
    fn is_call(v: &Value) -> bool {
        matches!(v, Value::Cons(head, _) if matches!(&**head, Value::Symbol(_)))
    }
    fn walk(v: &Value) -> bool {
        match v {
            Value::Cons(head, rest) => {
                match &**head {
                    Value::Special(Special::If) => {
                        // (if cond then else) — then/else are tail.
                        let then_v = rest.nth(1);
                        let else_v = rest.nth(2);
                        walk(&then_v) || walk(&else_v)
                    }
                    Value::Special(Special::Begin) => {
                        // (begin e1 ... eN) — only eN is tail.
                        let mut last = v.clone();
                        loop {
                            let next = last.cdr();
                            if matches!(&*next, Value::Nil) {
                                break;
                            }
                            let nn = (*next).cdr();
                            if matches!(&*nn, Value::Nil) {
                                let final_arg = (*next).car();
                                return walk(&final_arg);
                            }
                            last = (*next).clone();
                        }
                        false
                    }
                    Value::Special(Special::Let) => {
                        // (let bindings body) — body is tail.
                        let body = rest.nth(1);
                        walk(&body)
                    }
                    _ => is_call(v),
                }
            }
            _ => false,
        }
    }
    walk(&closure.body)
}
