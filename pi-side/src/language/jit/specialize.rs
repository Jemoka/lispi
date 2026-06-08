//! Specialization layer for the JIT IR.
//!
//! Pattern-matches against `Special` heads on cons forms and emits
//! dedicated `IRStatement`s. Anything not specialized — closure calls,
//! control flow not yet supported, scope-introducing forms,
//! closure/macro construction, quasiquote — falls through to `escape`,
//! which hands the whole sexp back to the interpreter at run time.

use alloc::rc::Rc;
use alloc::vec::Vec;
use crate::language::ast::Value;
use crate::language::special::Special;
use crate::language::syscalls::Syscall;
use super::ir::{IRSegment, IRStatement, VReg};
use super::scope::{JitImage, Resolution};

/// Extract the single argument of `(op arg)` without evaluating it.
/// Returns Err on wrong arity.
fn arg1(sexp: &Rc<Value>) -> Result<Rc<Value>, &'static str> {
    if !sexp.nth_exists(1) { return Err("expected 1 argument, got fewer."); }
    if sexp.nth_exists(2)  { return Err("expected 1 argument, got more."); }
    Ok(sexp.nth(1))
}

/// Extract the two arguments of `(op a b)` without evaluating them.
fn arg2(sexp: &Rc<Value>) -> Result<(Rc<Value>, Rc<Value>), &'static str> {
    if !sexp.nth_exists(2) { return Err("expected 2 arguments, got fewer."); }
    if sexp.nth_exists(3)  { return Err("expected 2 arguments, got more."); }
    Ok((sexp.nth(1), sexp.nth(2)))
}

/// Extract the three arguments of `(op a b c)` without evaluating them.
fn arg3(sexp: &Rc<Value>) -> Result<(Rc<Value>, Rc<Value>, Rc<Value>), &'static str> {
    if !sexp.nth_exists(3) { return Err("expected 3 arguments, got fewer."); }
    if sexp.nth_exists(4)  { return Err("expected 3 arguments, got more."); }
    Ok((sexp.nth(1), sexp.nth(2), sexp.nth(3)))
}

/// Extract the four arguments of `(op a b c d)` without evaluating them.
fn arg4(sexp: &Rc<Value>) -> Result<(Rc<Value>, Rc<Value>, Rc<Value>, Rc<Value>), &'static str> {
    if !sexp.nth_exists(4) { return Err("expected 4 arguments, got fewer."); }
    if sexp.nth_exists(5)  { return Err("expected 4 arguments, got more."); }
    Ok((sexp.nth(1), sexp.nth(2), sexp.nth(3), sexp.nth(4)))
}

/// Assert no arguments (`(op)`). Used by 0-arg syscalls.
fn arg0(sexp: &Rc<Value>) -> Result<(), &'static str> {
    if sexp.nth_exists(1) { return Err("expected 0 arguments, got more."); }
    Ok(())
}

impl IRSegment {
    /// Helper: cgen a unary form `(op x)` and emit `build(dst, x)`.
    fn unop<F>(&mut self, sexp: &Rc<Value>, scope: &mut JitImage<'_>, build: F)
        -> Result<VReg, &'static str>
    where F: FnOnce(VReg, VReg) -> IRStatement {
        let a = arg1(sexp)?;
        let ra = self.cgen_inner(a, scope)?;
        let r = self.reg();
        self.emit(build(r.clone(), ra));
        Ok(r)
    }

    /// Helper: cgen a binary form `(op a b)` and emit `build(dst, a, b)`.
    fn binop<F>(&mut self, sexp: &Rc<Value>, scope: &mut JitImage<'_>, build: F)
        -> Result<VReg, &'static str>
    where F: FnOnce(VReg, VReg, VReg) -> IRStatement {
        let (a, b) = arg2(sexp)?;
        let ra = self.cgen_inner(a, scope)?;
        let rb = self.cgen_inner(b, scope)?;
        let r = self.reg();
        self.emit(build(r.clone(), ra, rb));
        Ok(r)
    }

    /// Helper: cgen a ternary form `(op a b c)` and emit `build(dst, a, b, c)`.
    fn ternop<F>(&mut self, sexp: &Rc<Value>, scope: &mut JitImage<'_>, build: F)
        -> Result<VReg, &'static str>
    where F: FnOnce(VReg, VReg, VReg, VReg) -> IRStatement {
        let (a, b, c) = arg3(sexp)?;
        let ra = self.cgen_inner(a, scope)?;
        let rb = self.cgen_inner(b, scope)?;
        let rc = self.cgen_inner(c, scope)?;
        let r = self.reg();
        self.emit(build(r.clone(), ra, rb, rc));
        Ok(r)
    }

    /// Helper: cgen a quaternary form `(op a b c d)` and emit
    /// `build(dst, a, b, c, d)`.
    fn quadop<F>(&mut self, sexp: &Rc<Value>, scope: &mut JitImage<'_>, build: F)
        -> Result<VReg, &'static str>
    where F: FnOnce(VReg, VReg, VReg, VReg, VReg) -> IRStatement {
        let (a, b, c, d) = arg4(sexp)?;
        let ra = self.cgen_inner(a, scope)?;
        let rb = self.cgen_inner(b, scope)?;
        let rc = self.cgen_inner(c, scope)?;
        let rd = self.cgen_inner(d, scope)?;
        let r = self.reg();
        self.emit(build(r.clone(), ra, rb, rc, rd));
        Ok(r)
    }

    /// Helper for 0-arg syscalls: check arity, mint a dst VReg, emit
    /// `build(dst)`.
    fn nop_op<F>(&mut self, sexp: &Rc<Value>, build: F)
        -> Result<VReg, &'static str>
    where F: FnOnce(VReg) -> IRStatement {
        arg0(sexp)?;
        let r = self.reg();
        self.emit(build(r.clone()));
        Ok(r)
    }

    /// Emit a full-sexp Escape: load `sexp` as a literal, then hand it
    /// to the interpreter. Used both as the fallback in `specialize` and
    /// for special forms we don't yet (or won't) JIT.
    pub(super) fn escape(&mut self, sexp: &Rc<Value>) -> VReg {
        let r = self.reg();
        self.emit(IRStatement::Load(r.clone(), (**sexp).clone()));
        let ret = self.reg();
        self.emit(IRStatement::Escape(ret.clone(), r));
        ret
    }

    /// Cgen helper for `let`: walks the flat binding list
    /// `(name1 val1 name2 val2 ...)`, evaluating each value in the
    /// current (evolving) scope so earlier bindings are visible to
    /// later values. Assumes the caller has already pushed a frame.
    fn cgen_let_bindings(
        &mut self,
        bindings_list: Rc<Value>,
        body: Rc<Value>,
        scope: &mut JitImage<'_>,
    ) -> Result<VReg, &'static str> {
        let mut current = bindings_list;
        loop {
            // pull (name, val_expr, tail) out of the match so the borrow
            // on `current` ends before we reassign it
            let (name, val_expr, tail) = match &*current {
                Value::Nil => break,
                Value::Cons(name_rc, rest) => {
                    let name = match &**name_rc {
                        Value::Symbol(s) => (**s).clone(),
                        _ => return Err("let: binding name must be a symbol."),
                    };
                    let (val_expr, tail) = match &**rest {
                        Value::Cons(val, t) => (Rc::clone(val), Rc::clone(t)),
                        _ => return Err("let: odd number of elements in binding list."),
                    };
                    (name, val_expr, tail)
                }
                _ => return Err("let: malformed binding list."),
            };
            let rv = self.cgen_inner(val_expr, scope)?;
            let id = scope.insert(name.clone());
            self.emit(IRStatement::BindLocal { name, id, src: rv });
            current = tail;
        }
        self.cgen_inner(body, scope)
    }

    /// some specialization for special forms and syscalls, everything else we don't know
    /// about is handed off to the intepreter
    pub(super) fn specialize(&mut self, value: Rc<Value>, scope: &mut JitImage<'_>) -> Result<VReg, &'static str> {
        let car = match &*value {
            Value::Cons(a, _) => Rc::clone(a),
            _ => unreachable!("specialize should only be called on cons lists!"),
        };

        match &*car {
            // --- arithmetic ---
            Value::Special(Special::Add) => self.binop(&value, scope, IRStatement::Add),
            Value::Special(Special::Sub) => self.binop(&value, scope, IRStatement::Sub),
            Value::Special(Special::Mul) => self.binop(&value, scope, IRStatement::Mul),
            Value::Special(Special::Div) => self.binop(&value, scope, IRStatement::Div),
            Value::Special(Special::Mod) => self.binop(&value, scope, IRStatement::Mod),
            Value::Special(Special::Lshift) => self.binop(&value, scope, IRStatement::Lshift),
            Value::Special(Special::Rshift) => self.binop(&value, scope, IRStatement::Rshift),

            // --- bitwise ---
            Value::Special(Special::BinNot) => self.unop(&value, scope, IRStatement::BinNot),
            Value::Special(Special::BinOr)  => self.binop(&value, scope, IRStatement::BinOr),
            Value::Special(Special::BinAnd) => self.binop(&value, scope, IRStatement::BinAnd),

            // --- logical (eager) ---
            Value::Special(Special::Not) => self.unop(&value, scope, IRStatement::LogNot),
            Value::Special(Special::Xor) => self.binop(&value, scope, IRStatement::Xor),

            // --- comparisons ---
            Value::Special(Special::Eq)  => self.binop(&value, scope, IRStatement::Eq),
            Value::Special(Special::Gt)  => self.binop(&value, scope, IRStatement::Gt),
            Value::Special(Special::Lt)  => self.binop(&value, scope, IRStatement::Lt),
            Value::Special(Special::Gte) => self.binop(&value, scope, IRStatement::Gte),
            Value::Special(Special::Lte) => self.binop(&value, scope, IRStatement::Lte),

            // --- type coercion ---
            Value::Special(Special::Addr)     => self.unop(&value, scope, IRStatement::AsAddr),
            Value::Special(Special::Signed)   => self.unop(&value, scope, IRStatement::AsSigned),
            Value::Special(Special::Unsigned) => self.unop(&value, scope, IRStatement::AsUnsigned),

            // --- cons / list primitives ---
            Value::Special(Special::Cons)  => self.binop(&value, scope, IRStatement::Cons),
            Value::Special(Special::Car)   => self.unop(&value, scope, IRStatement::Car),
            Value::Special(Special::Cdr)   => self.unop(&value, scope, IRStatement::Cdr),
            Value::Special(Special::Nullp) => self.unop(&value, scope, IRStatement::Nullp),

            // (list a b c) → (cons a (cons b (cons c nil)))
            // built right-to-left
            Value::Special(Special::List) => {
                let mut args = Vec::new();
                let mut i = 1;
                loop {
                    let arg = value.nth(i);
                    if arg.is_nil() { break; }
                    args.push(arg);
                    i += 1;
                }
                let mut cur = self.reg();
                self.emit(IRStatement::Load(cur.clone(), Value::Nil));
                for arg in args.into_iter().rev() {
                    let ra = self.cgen_inner(arg, scope)?;
                    let next = self.reg();
                    self.emit(IRStatement::Cons(next.clone(), ra, cur));
                    cur = next;
                }
                Ok(cur)
            }

            // --- arrays ---
            Value::Special(Special::Array)   => self.unop(&value, scope, IRStatement::Array),
            Value::Special(Special::Full)    => self.binop(&value, scope, IRStatement::Full),
            Value::Special(Special::Unpack)  => self.unop(&value, scope, IRStatement::Unpack),
            Value::Special(Special::GetIdx)  => self.binop(&value, scope, IRStatement::GetIdx),
            Value::Special(Special::PutIdx)  => self.ternop(&value, scope, IRStatement::PutIdx),
            Value::Special(Special::ReadIdx) => self.ternop(&value, scope, IRStatement::ReadIdx),
            Value::Special(Special::FillIdx) => self.ternop(&value, scope, IRStatement::FillIdx),
            Value::Special(Special::FullIdx) => self.quadop(&value, scope, IRStatement::FullIdx),

            // --- introspection ---
            Value::Special(Special::Hits) => self.unop(&value, scope, IRStatement::Hits),

            // (begin e1 e2 ... en) — sequence, return last
            Value::Special(Special::Begin) => {
                let mut last: Option<VReg> = None;
                let mut i = 1;
                loop {
                    let arg = value.nth(i);
                    if arg.is_nil() { break; }
                    last = Some(self.cgen_inner(arg, scope)?);
                    i += 1;
                }
                match last {
                    Some(r) => Ok(r),
                    None => {
                        // (begin) — empty, return nil
                        let r = self.reg();
                        self.emit(IRStatement::Load(r.clone(), Value::Nil));
                        Ok(r)
                    }
                }
            }

            // (set name value) — name is unevaluated, must be a symbol.
            // If the name resolves to a local, emit StoreLocal; if it
            // resolves to a capture/global, emit StoreCapture. If the
            // name is unbound, escape to the interpreter (which has the
            // authority to allocate a new top-level binding — the JIT
            // borrows the runtime Image immutably and cannot).
            Value::Special(Special::Set) => {
                let (name_val, val_expr) = arg2(&value)?;
                let name = match &*name_val {
                    Value::Symbol(s) => (**s).clone(),
                    _ => return Err("set: first argument must be a symbol."),
                };
                match scope.binding(&name) {
                    Resolution::Local(id) => {
                        let rv = self.cgen_inner(val_expr, scope)?;
                        self.emit(IRStatement::StoreLocal(id, rv.clone()));
                        Ok(rv)
                    }
                    Resolution::Capture(b) => {
                        let rv = self.cgen_inner(val_expr, scope)?;
                        self.emit(IRStatement::StoreCapture(b, rv.clone()));
                        Ok(rv)
                    }
                    Resolution::Unbound => Ok(self.escape(&value)),
                }
            }

            // (let (a 1 b 2 ...) body) — emit a runtime scope boundary
            // (PushFrame ... PopFrame) around a sequence of BindLocals,
            // mirroring the interpreter's per-`let` frame push/pop.
            // The compile-time JitImage push/pop mirrors the runtime
            // structure so lookups during cgen resolve correctly.
            Value::Special(Special::Let) => {
                let (bindings_list, body) = arg2(&value)?;
                scope.push_frame();
                self.emit(IRStatement::PushFrame);
                let result = self.cgen_let_bindings(bindings_list, body, scope);
                self.emit(IRStatement::PopFrame);
                scope.pop_frame();
                result
            }

            // --- control flow ---
            //
            // Pattern for all three:
            //   1. cgen the condition (or LHS for and/or) in current block
            //   2. push then/else blocks (or rhs/short for and/or),
            //      cgen each arm, remember the END block of each arm
            //   3. push the merge block
            //   4. back-fill: Br from each arm's end to merge,
            //      CondBr in the cond block to then/else
            //   5. phi at top of merge with (arm_end, arm_vreg) pairs
            //
            // The "back-fill after all blocks are known" pattern avoids
            // forward-reference acrobatics.

            // (if cond then else)
            Value::Special(Special::If) => {
                let (cond, then_e, else_e) = arg3(&value)?;
                let r_c = self.cgen_inner(cond, scope)?;
                let cond_end = self.btop();

                let then_blk = self.bpush();
                let r_t = self.cgen_inner(then_e, scope)?;
                let then_end = self.btop();

                let else_blk = self.bpush();
                let r_e = self.cgen_inner(else_e, scope)?;
                let else_end = self.btop();

                let merge_blk = self.bpush();

                self.emit_at(then_end, IRStatement::Br(merge_blk));
                self.emit_at(else_end, IRStatement::Br(merge_blk));
                self.emit_at(cond_end, IRStatement::CondBr {
                    cond: r_c, then_blk, else_blk,
                });

                Ok(self.phi(merge_blk, (then_end, r_t), (else_end, r_e)))
            }

            // (and a b) — eval a; if truthy eval and return b; else
            // short-circuit on the value of a.
            Value::Special(Special::And) => {
                let (a, b) = arg2(&value)?;
                let r_a = self.cgen_inner(a, scope)?;
                let cond_end = self.btop();

                let rhs_blk = self.bpush();
                let r_b = self.cgen_inner(b, scope)?;
                let rhs_end = self.btop();

                let short_blk = self.bpush();
                // short arm has no value computation — falls straight to merge

                let merge_blk = self.bpush();

                self.emit_at(rhs_end, IRStatement::Br(merge_blk));
                self.emit_at(short_blk, IRStatement::Br(merge_blk));
                self.emit_at(cond_end, IRStatement::CondBr {
                    cond: r_a.clone(), then_blk: rhs_blk, else_blk: short_blk,
                });

                Ok(self.phi(merge_blk, (rhs_end, r_b), (short_blk, r_a)))
            }

            // (or a b) — eval a; if truthy short-circuit on a; else
            // eval and return b. Mirror of `and` with arms swapped.
            Value::Special(Special::Or) => {
                let (a, b) = arg2(&value)?;
                let r_a = self.cgen_inner(a, scope)?;
                let cond_end = self.btop();

                let short_blk = self.bpush();
                // short arm has no value computation

                let rhs_blk = self.bpush();
                let r_b = self.cgen_inner(b, scope)?;
                let rhs_end = self.btop();

                let merge_blk = self.bpush();

                self.emit_at(short_blk, IRStatement::Br(merge_blk));
                self.emit_at(rhs_end, IRStatement::Br(merge_blk));
                self.emit_at(cond_end, IRStatement::CondBr {
                    cond: r_a.clone(), then_blk: short_blk, else_blk: rhs_blk,
                });

                Ok(self.phi(merge_blk, (short_blk, r_a), (rhs_end, r_b)))
            }

            // --- syscalls ---
            // Specialized: ones that lower to a single asm instruction
            // sequence, MMIO read/write, or memset-shaped fill.
            // Escaped (caught by the `_` arm below): everything that
            // hits the Rust heap allocator, walks a list, or borrows
            // an Array — `alloc32`, `free32`, `read32`, `fill32`,
            // `ldr`, `str`, `unpack1to16`.

            // 0-arg (pure asm / MMIO)
            Value::Syscall(Syscall::DSB)             => self.nop_op(&value, IRStatement::SysDsb),
            Value::Syscall(Syscall::PrefetchFlush)   => self.nop_op(&value, IRStatement::SysPrefetchFlush),
            Value::Syscall(Syscall::UartInit)        => self.nop_op(&value, IRStatement::SysUartInit),
            Value::Syscall(Syscall::UartGet8)        => self.nop_op(&value, IRStatement::SysUartGet8),
            Value::Syscall(Syscall::ClearSetMonitor) => self.nop_op(&value, IRStatement::SysClearMonitor),
            Value::Syscall(Syscall::GetMonitor)      => self.nop_op(&value, IRStatement::SysGetMonitor),
            Value::Syscall(Syscall::StopMonitor)     => self.nop_op(&value, IRStatement::SysStopMonitor),

            // 1-arg
            Value::Syscall(Syscall::Get32)    => self.unop(&value, scope, IRStatement::SysGet32),
            Value::Syscall(Syscall::UartPut8) => self.unop(&value, scope, IRStatement::SysUartPut8),
            Value::Syscall(Syscall::Delay)    => self.unop(&value, scope, IRStatement::SysDelay),

            // 2-arg
            Value::Syscall(Syscall::Put32)    => self.binop(&value, scope, IRStatement::SysPut32),

            // 3-arg (memset-shaped)
            Value::Syscall(Syscall::Zero32)   => self.ternop(&value, scope, IRStatement::SysZero32),

            // 4-arg (memset-shaped)
            Value::Syscall(Syscall::Full32)   => self.quadop(&value, scope, IRStatement::SysFull32),

            // --- closure / definition construction ---
            // These all build a `Value::Closure` (or `Value::Macro`)
            // whose `env` is `image.snapshot()` — a flatten of every
            // live frame at the moment the form runs, including the
            // local frames our `BindLocal` opcodes pushed. The
            // interpreter already does this against the same runtime
            // Image we'd be working with; reimplementing snapshot
            // inside the JIT buys nothing.
            //   `Defun` desugars at runtime to `(set name (lambda ...))`
            //   so it inherits Lambda's escape reason transitively.
            Value::Special(Special::Lambda)
            | Value::Special(Special::Defun)
            | Value::Special(Special::Defmacro)

            // --- quote / quasiquote (literal-data construction) ---
            // `quote` returns its argument as data, unchanged — trivial
            // but rarely hot. `quasiquote` walks the structure
            // recursively, evaluating `(unquote x)` and splicing
            // `(unquote-splicing x)`; a complex tree walk the
            // interpreter already implements. `unquote` /
            // `unquote-splicing` are only meaningful inside a
            // quasiquote walker — by themselves they raise errors.
            | Value::Special(Special::Quote)
            | Value::Special(Special::Quasiquote)
            | Value::Special(Special::Unquote)
            | Value::Special(Special::UnquoteSplicing)

            // --- meta / macro expansion ---
            // `macroexpand` looks up a macro, invokes it on the
            // *unevaluated* args, and returns the expanded sexp
            // without executing it. Inherently a meta-level operation
            // that runs at interpret time, not run time.
            | Value::Special(Special::Macroexpand)
                => Ok(self.escape(&value)),

            // --- catch-all ---
            // Anything we haven't matched: regular function calls
            // (closure invocations), syscalls, computed callees, etc.
            // The interpreter handles them. If the callee body is
            // hot enough to JIT, that's decided on its own later run,
            // not at this escape site.
            _ => Ok(self.escape(&value)),
        }
    }
}
