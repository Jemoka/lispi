//! Linear IR for JIT compilation of closure bodies.
//!
//! # Model
//!
//! Each `IRSegment` is the compiled form of one closure body, specialized
//! against the env present at the moment of capture. Compilation happens
//! once per body; the resulting IR is reused across every invocation of
//! that closure.
//!
//! # Symbol resolution: locals vs captures
//!
//! Symbol references split into two categories, distinguished at IR-gen
//! time by a lexical scope walk:
//!
//! * **Locals** — names introduced by the call frame itself: function
//!   params and `let` bindings inside the body. Each invocation gets a
//!   fresh `Binding` allocation for these, so we cannot bake any specific
//!   `Binding` pointer at IR-gen time. Instead we tag locals with a
//!   stable `LocalId` (assigned by the lexical scope pass); the JIT
//!   wrapper resolves each `LocalId` to the call's actual register /
//!   `Binding` at invocation entry.
//!
//! * **Captures** — names that escape the lexical scope to the closure's
//!   captured env (or globals). Their `Binding`s live in the closure's
//!   `Rc<Environment>` snapshot, which is stable across invocations.
//!   We bake the `Binding` directly into the IR opcode; lowering reads
//!   the slot via `Rc::as_ptr` on the inner `RefCell`.
//!
//! See `LoadLocal` / `LoadCapture` / `StoreLocal` / `StoreCapture` below.
//! Lexical scope is tracked via `JitImage` in `super::scope`.

use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use heapless::String;
use crate::language::ast::Value;
use crate::language::constants::SYMB_NAME_LEN;
use crate::language::environment::Binding;
use super::scope::{JitImage, LocalId, Resolution};

/// Symbol name used inside `BindLocal` so the runtime can register the
/// binding under its source name in the Image's top frame.
pub(crate) type Name = String<SYMB_NAME_LEN>;

/// SSA virtual register. Each emitted statement assigns its destination
/// exactly once; later analysis (liveness, regalloc) depends on this.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct VReg(pub(super) u32);

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum IRStatement {
    /// Load a literal value into `dst`. Used for self-evaluating forms
    /// (numbers, strings, nil, bool, quoted data, special-form tokens).
    Load(VReg, Value),

    // --- symbol resolution ---

    // --- runtime scope management ---
    // The runtime keeps `locals: Vec<Binding>` (length = num_locals,
    // pre-allocated by the call-entry trampoline) and a parallel
    // Image frame stack. The opcodes below tie them together.

    /// Push an empty Owned frame onto the runtime Image. Marks entry
    /// of a `let` scope.
    PushFrame,

    /// Pop the top frame from the runtime Image. Marks exit of a
    /// `let` scope. The `Binding`s themselves stay alive (they're
    /// owned by `locals`); only the name → binding entries leave scope.
    PopFrame,

    /// Initialize `locals[id]` from `src`, then insert
    /// `(name → locals[id])` into the Image's top frame so escaping
    /// code can look up `name` and find this binding. One per
    /// `let`-binding; params are bound by the call-entry trampoline
    /// outside the IR.
    BindLocal { name: Name, id: LocalId, src: VReg },

    /// Constant-folded `BindLocal`: the source has been collapsed from
    /// a (`Load r, #v`) + (`BindLocal $id, r`) pair into a single
    /// inline-value op. Same runtime semantics as `BindLocal` —
    /// writes `locals[id]`'s inner value and registers
    /// `(name → locals[id])` in the top frame — just without the
    /// scaffolding Load. Emitted by the SCCP fold pass when the
    /// BindLocal's source VReg is `Constant`.
    BindImmediate { name: Name, id: LocalId, src: Value },

    /// Load the current value of a local (`param` / `let`) into `dst`.
    /// Lowering bakes the slot address `locals[id].as_ref().as_ptr()`
    /// as an immediate.
    LoadLocal(VReg, LocalId),

    /// Load the current value of a captured / global slot into `dst`.
    /// `Binding` is baked at IR-gen time; lowering emits a slot load.
    LoadCapture(VReg, Binding),

    /// Write the value in `src` into a local slot.
    StoreLocal(LocalId, VReg),

    /// Write the value in `src` into a captured / global slot.
    /// The store mutates the slot in place; observable by any closure
    /// that captured the same `Binding`.
    StoreCapture(Binding, VReg),

    // --- arithmetic (assumes numeric inputs; numeric tagging is a
    //     second-pass concern, see header comment) ---

    Add(VReg, VReg, VReg),
    Sub(VReg, VReg, VReg),
    Mul(VReg, VReg, VReg),
    Div(VReg, VReg, VReg),
    Mod(VReg, VReg, VReg),
    Lshift(VReg, VReg, VReg),
    Rshift(VReg, VReg, VReg),

    // --- bitwise ---

    BinNot(VReg, VReg),
    BinOr(VReg, VReg, VReg),
    BinAnd(VReg, VReg, VReg),

    // --- logical (eager — short-circuit forms are control flow) ---

    /// Logical NOT, per `is_falsy` semantics: falsy → true, else → false.
    LogNot(VReg, VReg),
    /// Eager XOR — both sides always evaluated.
    Xor(VReg, VReg, VReg),

    // --- comparisons (produce Bool) ---

    Eq(VReg, VReg, VReg),
    Gt(VReg, VReg, VReg),
    Lt(VReg, VReg, VReg),
    Gte(VReg, VReg, VReg),
    Lte(VReg, VReg, VReg),

    // --- numeric type coercion ---

    AsAddr(VReg, VReg),
    AsSigned(VReg, VReg),
    AsUnsigned(VReg, VReg),

    // --- cons / list primitives ---

    Cons(VReg, VReg, VReg),
    Car(VReg, VReg),
    Cdr(VReg, VReg),
    Nullp(VReg, VReg),

    // --- array primitives ---

    /// `(array list)` — pack a list of u32s into a new Array.
    Array(VReg, VReg),
    /// `(full n val)` — allocate an array of `n` copies of `val`.
    Full(VReg, VReg, VReg),
    /// `(unpack array)` — array → list.
    Unpack(VReg, VReg),
    /// `(getidx target idx)` — read u32 at `idx`.
    GetIdx(VReg, VReg, VReg),
    /// `(putidx target idx val)` — write u32 at `idx`. Result is nil.
    PutIdx(VReg, VReg, VReg, VReg),
    /// `(readidx target offset n)` — read `n` u32s into a list.
    ReadIdx(VReg, VReg, VReg, VReg),
    /// `(fillidx target offset list)` — write list values from `offset`. Result is nil.
    FillIdx(VReg, VReg, VReg, VReg),
    /// `(fullidx target offset n val)` — fill `n` slots from `offset`. Result is nil.
    FullIdx(VReg, VReg, VReg, VReg, VReg),

    // --- introspection ---

    /// `(hits f)` — read the closure/macro hit counter as an Unsigned.
    Hits(VReg, VReg),

    // --- control flow ---
    // Blocks have no fall-through. A block is "closed" by emitting a
    // `Br` or `CondBr`; the function returns from whichever open block
    // execution reaches at the end (its final result VReg = whatever
    // cgen returned for the top-level body).

    /// Unconditional jump to `target`.
    Br(usize),

    /// Conditional jump: if `cond`'s value is *not* falsy (per
    /// `is_falsy`: nil, false, integer 0, unsigned 0), go to `then_blk`;
    /// otherwise go to `else_blk`.
    CondBr { cond: VReg, then_blk: usize, else_blk: usize },

    /// SSA phi: `dst` takes the value of `src_a` if the merge block was
    /// reached from `pred_a`, or `src_b` if reached from `pred_b`.
    /// MUST be the first statement in its block (enforced by
    /// `IRSegment::phi`). All merges in this language are binary
    /// (`if` / `and` / `or`), so the source list is fixed at two.
    ///
    /// After `optimize`, one of the predecessor block IDs may refer to
    /// a block whose `dead == true` (e.g. when a `CondBr` folded to a
    /// `Br` and the not-taken arm became unreachable). The syntactic
    /// reference is intentionally preserved so block IDs stay stable;
    /// lowering should treat phi sources from dead predecessors as
    /// "edge never taken" and not emit anything for them.
    PhiOp(VReg, (usize, VReg), (usize, VReg)),

    /// Return from the function with `src` as the result. The unique
    /// terminator for any block that ends function execution; emitted
    /// by `IRSegment::finalize` on the block left open after top-level
    /// cgen. Multiple `Ret`s are legal (one per open block).
    Ret(VReg),

    // --- syscalls ---
    // Only the asm / MMIO / memset-shaped syscalls get specialized.
    // Anything that allocates on the Rust heap, walks a list, or
    // borrows an Array (alloc32/free32/read32/fill32/ldr/str/
    // unpack1to16) escapes — the interpreter handles those just fine
    // and the JIT win wouldn't pay for the complexity.

    // 0-arg
    SysDsb(VReg),
    SysPrefetchFlush(VReg),
    SysUartInit(VReg),
    SysUartGet8(VReg),
    SysClearMonitor(VReg),
    SysGetMonitor(VReg),
    SysStopMonitor(VReg),

    // 1-arg
    SysGet32(VReg, VReg),       // dst (u32),  addr
    SysUartPut8(VReg, VReg),    // dst (nil),  byte
    SysDelay(VReg, VReg),       // dst (nil),  count

    // 2-arg
    SysPut32(VReg, VReg, VReg),         // dst (nil),  addr, val

    // 3-arg
    SysZero32(VReg, VReg, VReg, VReg),  // dst (nil),  addr, offset, n

    // 4-arg
    SysFull32(VReg, VReg, VReg, VReg, VReg),  // dst (nil), addr, offset, n, val

    // --- direct closure call ---
    //
    // When the head of a cons form resolves to a captured (non-local)
    // Binding at IR-gen time, lower the call directly instead of
    // escaping the whole sexp. Args are evaluated in the caller's IR;
    // the runtime helper resolves the binding to a Closure, dispatches
    // through the executor's JC cache, and returns the result slot.
    /// Direct dispatch: `dst = call(callee.binding, args...)`.
    /// Supported arity is 1-3 (lower bound: must have ≥1 arg; upper
    /// bound: AAPCS gives us r1..r3 after r0 = binding_ptr). Higher-
    /// arity forms fall back to `Escape` at IR-gen time.
    Call { dst: VReg, callee: Binding, args: Vec<VReg> },

    // --- escape hatch ---

    /// Evaluate the value in `src` through the interpreter and put the
    /// result in `dst`. Used for anything the JIT can't (or won't)
    /// specialize: function calls, quasiquote, macro expansion, lambda
    /// construction, etc.
    Escape(VReg, VReg),
}

#[derive(Clone, Debug)]
pub(crate) struct IRBasicBlock {
    pub statements: Vec<IRStatement>,
    /// Marked true by `optimize::dead_blocks` for blocks unreachable
    /// from the entry. **Dead blocks are kept in-place** so block IDs
    /// stay stable for phi predecessors. The pretty-printer renders
    /// them as `.LN (Dead):` stubs; lowering must do the same and
    /// skip the body entirely — surviving statements inside a dead
    /// block may include side-effecting ops that DCE couldn't kill
    /// (e.g. their VReg was still syntactically referenced by a phi
    /// in a live block). See `optimize.rs` header for details.
    pub dead: bool,
}

#[derive(Clone)]
pub(crate) struct IRSegment {
    /// next register being used
    pub regs: VReg,
    /// current basic block emitting
    pub blocks: Vec<IRBasicBlock>,
}

impl IRSegment {
    pub(crate) fn new() -> Self {
        IRSegment {
            regs: VReg(0),
            blocks: vec![IRBasicBlock { statements: Vec::new(), dead: false }],
        }
    }

    /// ID of the current (last) block — where `emit` writes. By the
    /// invariant "no fall-through and `emit` always targets the last
    /// block", current = `blocks.len() - 1`.
    pub(super) fn btop(&self) -> usize {
        self.blocks.len() - 1
    }

    /// Push a fresh empty block and return its id. The new block
    /// becomes "current" — subsequent `emit` calls write into it.
    pub(super) fn bpush(&mut self) -> usize {
        self.blocks.push(IRBasicBlock { statements: Vec::new(), dead: false });
        self.blocks.len() - 1
    }

    /// Emit into the current (last) block.
    pub(super) fn emit(&mut self, stmt: IRStatement) {
        self.blocks.last_mut().unwrap().statements.push(stmt);
    }

    /// Emit into an arbitrary block. Used by control-flow specializations
    /// that back-fill terminators after all relevant block ids are known.
    pub(super) fn emit_at(&mut self, blk: usize, stmt: IRStatement) {
        self.blocks[blk].statements.push(stmt);
    }

    /// Emit a phi at the top of `blk`, mint and return its destination
    /// VReg. Asserts the block is empty — phi must be the first
    /// statement in its block (SSA invariant).
    pub(super) fn phi(
        &mut self,
        blk: usize,
        src_a: (usize, VReg),
        src_b: (usize, VReg),
    ) -> VReg {
        debug_assert!(
            self.blocks[blk].statements.is_empty(),
            "phi must be the first statement in its block"
        );
        let dst = self.reg();
        self.emit_at(blk, IRStatement::PhiOp(dst.clone(), src_a, src_b));
        dst
    }

    pub(super) fn reg(&mut self) -> VReg {
        self.regs.0 += 1;
        VReg(self.regs.0)
    }

    /// Top-level compile entry. Walks the body sexp, emits IR, and
    /// terminates the open block with `Ret(result)`. After this every
    /// block in the segment ends in exactly one terminator
    /// (`Br` / `CondBr` / `Ret`) — no implicit "fall through to return"
    /// convention. Use this from outside the JIT module; use
    /// `cgen_inner` from within specializations that need the result
    /// VReg of a subexpression.
    pub(crate) fn cgen(&mut self, body: Rc<Value>, scope: &mut JitImage<'_>) -> Result<(), &'static str> {
        let result = self.cgen_inner(body, scope)?;
        self.emit(IRStatement::Ret(result));
        Ok(())
    }

    /// Recursive expression codegen — returns the VReg holding the
    /// result of `sexp`. Used internally by `cgen` and by
    /// `specialize` for subexpressions. Does NOT emit a terminator on
    /// the open block; callers compose.
    pub(super) fn cgen_inner(&mut self, sexp: Rc<Value>, scope: &mut JitImage<'_>) -> Result<VReg, &'static str> {
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
                | Value::JittedClosure(_) => {
                    let r = self.reg();
                    self.emit(IRStatement::Load(r.clone(), (*sexp).clone()));
                    Ok(r)
                },
            // symbols — resolve through the compile scope. Locals get
            // a LoadLocal; captures/globals get a LoadCapture with the
            // baked Binding; anything unbound is a compile error.
            Value::Symbol(sym) => {
                let r = self.reg();
                match scope.binding(sym) {
                    Resolution::Local(id) => {
                        self.emit(IRStatement::LoadLocal(r.clone(), id));
                    }
                    Resolution::Capture(b) => {
                        self.emit(IRStatement::LoadCapture(r.clone(), b));
                    }
                    Resolution::Unbound => {
                        // Symbol unbound at IR-gen time — could be a
                        // top-level set! creating it later in the same
                        // expression. Escape to the interpreter, which
                        // resolves it at runtime.
                        return Ok(self.escape(&sexp));
                    }
                }
                Ok(r)
            },
            // cons lists: if the caller is a Special, specialize it;
            // otherwise (computed callee, regular function call, etc.)
            // escape the entire sexp to the interpreter.
            //
            // i.e. we handle (print 3) but we don't handle
            // ((thing_that_returns_print) 3) — the latter is just
            // an Escape on the whole sexp.
            Value::Cons(car, _) => {
                match &**car {
                    // computed callee — interpreter has to do the call
                    Value::Cons(..) => Ok(self.escape(&sexp)),
                    _ => self.specialize(sexp, scope),
                }
            },

            // tail calls: we should never see this mixed in between
            // intepreted code as long as CONTRAST never call `exec`
            // during jitting
            Value::TailCall(..) => Err("unexpected tail call in IR generation"),
        }
    }
}

// ===================== pretty-printer =====================
//
// `Debug` for `IRSegment` prints the IR in an ARM-asm-ish layout:
// blocks labeled `.Ln:`, opcodes indented and padded to a fixed
// mnemonic column, VRegs as `rN`, locals as `LN`, capture slots
// as their inner-Rc pointer, block ids as `.LN`.

fn fmt_vreg(f: &mut fmt::Formatter<'_>, r: &VReg) -> fmt::Result {
    write!(f, "r{}", r.0)
}

fn fmt_local(f: &mut fmt::Formatter<'_>, id: &LocalId) -> fmt::Result {
    write!(f, "${}", id.0)
}

fn fmt_binding(f: &mut fmt::Formatter<'_>, b: &Binding) -> fmt::Result {
    // Bracket the address so `ldc rN, [#0xADDR]` reads as a load *from*
    // a slot rather than a load of an immediate constant. Same for
    // `stc [#0xADDR], rN` — store *into* the slot.
    write!(f, "[#{:p}]", b.as_ref().as_ptr())
}

fn fmt_blk(f: &mut fmt::Formatter<'_>, b: usize) -> fmt::Result {
    write!(f, ".L{}", b)
}

/// Write one statement. Indented and padded by the caller.
fn fmt_stmt(f: &mut fmt::Formatter<'_>, s: &IRStatement) -> fmt::Result {
    // Mnemonic width — keep operands aligned.
    const W: usize = 7;
    macro_rules! mn { ($m:expr) => { write!(f, "{:<width$}", $m, width = W) }; }

    match s {
        IRStatement::Load(r, v) => {
            mn!("ldv")?; fmt_vreg(f, r)?; write!(f, ", #{}", v)
        }

        // --- symbol resolution ---
        IRStatement::LoadLocal(r, id) => {
            mn!("ldl")?; fmt_vreg(f, r)?; write!(f, ", ")?; fmt_local(f, id)
        }
        IRStatement::LoadCapture(r, b) => {
            mn!("ldc")?; fmt_vreg(f, r)?; write!(f, ", ")?; fmt_binding(f, b)
        }
        IRStatement::StoreLocal(id, r) => {
            mn!("stl")?; fmt_local(f, id)?; write!(f, ", ")?; fmt_vreg(f, r)
        }
        IRStatement::StoreCapture(b, r) => {
            mn!("stc")?; fmt_binding(f, b)?; write!(f, ", ")?; fmt_vreg(f, r)
        }

        // --- runtime scope management ---
        IRStatement::PushFrame => mn!("pushf"),
        IRStatement::PopFrame  => mn!("popf"),
        IRStatement::BindLocal { name, id, src } => {
            mn!("bnd")?;
            fmt_local(f, id)?; write!(f, ", ")?; fmt_vreg(f, src)?;
            write!(f, "    ; {:?}", name.as_str())
        }
        IRStatement::BindImmediate { name, id, src } => {
            mn!("bim")?;
            fmt_local(f, id)?; write!(f, ", #{}", src)?;
            write!(f, "    ; {:?}", name.as_str())
        }

        // --- arithmetic / bitwise / logical / compare (3-op binops) ---
        IRStatement::Add(r,a,b)    => bin(f, "add",  r, a, b),
        IRStatement::Sub(r,a,b)    => bin(f, "sub",  r, a, b),
        IRStatement::Mul(r,a,b)    => bin(f, "mul",  r, a, b),
        IRStatement::Div(r,a,b)    => bin(f, "div",  r, a, b),
        IRStatement::Mod(r,a,b)    => bin(f, "mod",  r, a, b),
        IRStatement::Lshift(r,a,b) => bin(f, "lsl",  r, a, b),
        IRStatement::Rshift(r,a,b) => bin(f, "lsr",  r, a, b),
        IRStatement::BinOr(r,a,b)  => bin(f, "bor",  r, a, b),
        IRStatement::BinAnd(r,a,b) => bin(f, "band", r, a, b),
        IRStatement::Xor(r,a,b)    => bin(f, "xor",  r, a, b),
        IRStatement::Eq(r,a,b)     => bin(f, "eq",   r, a, b),
        IRStatement::Gt(r,a,b)     => bin(f, "gt",   r, a, b),
        IRStatement::Lt(r,a,b)     => bin(f, "lt",   r, a, b),
        IRStatement::Gte(r,a,b)    => bin(f, "gte",  r, a, b),
        IRStatement::Lte(r,a,b)    => bin(f, "lte",  r, a, b),

        // --- 2-op unops ---
        IRStatement::BinNot(r,a)     => uno(f, "bnot", r, a),
        IRStatement::LogNot(r,a)     => uno(f, "lnot", r, a),
        IRStatement::AsAddr(r,a)     => uno(f, "tadr", r, a),
        IRStatement::AsSigned(r,a)   => uno(f, "tsig", r, a),
        IRStatement::AsUnsigned(r,a) => uno(f, "tuns", r, a),
        IRStatement::Car(r,a)        => uno(f, "car",  r, a),
        IRStatement::Cdr(r,a)        => uno(f, "cdr",  r, a),
        IRStatement::Nullp(r,a)      => uno(f, "null", r, a),
        IRStatement::Array(r,a)      => uno(f, "arr",  r, a),
        IRStatement::Unpack(r,a)     => uno(f, "unpk", r, a),
        IRStatement::Hits(r,a)       => uno(f, "hits", r, a),

        // --- cons / array ops with more operands ---
        IRStatement::Cons(r,a,b)   => bin(f, "cons", r, a, b),
        IRStatement::Full(r,a,b)   => bin(f, "full", r, a, b),
        IRStatement::GetIdx(r,a,b) => bin(f, "gidx", r, a, b),

        IRStatement::PutIdx(r,t,i,v)   => tri(f, "pidx", r, t, i, v),
        IRStatement::ReadIdx(r,t,o,n)  => tri(f, "ridx", r, t, o, n),
        IRStatement::FillIdx(r,t,o,l)  => tri(f, "fidx", r, t, o, l),
        IRStatement::FullIdx(r,t,o,n,v) => qua(f, "Fidx", r, t, o, n, v),

        // --- syscalls ---
        IRStatement::SysDsb(r)            => uno0(f, "@dsb",    r),
        IRStatement::SysPrefetchFlush(r)  => uno0(f, "@pfflsh", r),
        IRStatement::SysUartInit(r)       => uno0(f, "@uinit",  r),
        IRStatement::SysUartGet8(r)       => uno0(f, "@uget8",  r),
        IRStatement::SysClearMonitor(r)   => uno0(f, "@mclr",   r),
        IRStatement::SysGetMonitor(r)     => uno0(f, "@mget",   r),
        IRStatement::SysStopMonitor(r)    => uno0(f, "@mstp",   r),

        IRStatement::SysGet32(r,a)        => uno(f, "@get32",  r, a),
        IRStatement::SysUartPut8(r,a)     => uno(f, "@uput8",  r, a),
        IRStatement::SysDelay(r,a)        => uno(f, "@delay",  r, a),

        IRStatement::SysPut32(r,a,b)      => bin(f, "@put32",  r, a, b),

        IRStatement::SysZero32(r,a,b,c)   => tri(f, "@zero32", r, a, b, c),

        IRStatement::SysFull32(r,a,b,c,d) => qua(f, "@full32", r, a, b, c, d),

        // --- control flow ---
        IRStatement::Br(b) => { mn!("b")?; fmt_blk(f, *b) }
        IRStatement::CondBr { cond, then_blk, else_blk } => {
            mn!("cbr")?; fmt_vreg(f, cond)?;
            write!(f, ", ")?; fmt_blk(f, *then_blk)?;
            write!(f, ", ")?; fmt_blk(f, *else_blk)
        }
        IRStatement::PhiOp(r, (ab, av), (bb, bv)) => {
            mn!("phi")?; fmt_vreg(f, r)?;
            write!(f, ", [")?; fmt_blk(f, *ab)?; write!(f, ": ")?; fmt_vreg(f, av)?;
            write!(f, "], [")?; fmt_blk(f, *bb)?; write!(f, ": ")?; fmt_vreg(f, bv)?;
            write!(f, "]")
        }
        IRStatement::Ret(r) => { mn!("ret")?; fmt_vreg(f, r) }

        // --- call ---
        IRStatement::Call { dst, callee, args } => {
            mn!("call")?;
            fmt_vreg(f, dst)?;
            write!(f, ", ")?;
            fmt_binding(f, callee)?;
            for a in args {
                write!(f, ", ")?;
                fmt_vreg(f, a)?;
            }
            Ok(())
        }

        // --- escape ---
        IRStatement::Escape(r, src) => uno(f, "esc", r, src),
    }
}

fn uno0(f: &mut fmt::Formatter<'_>, m: &str, r: &VReg) -> fmt::Result {
    write!(f, "{:<7}", m)?;
    fmt_vreg(f, r)
}

fn uno(f: &mut fmt::Formatter<'_>, m: &str, r: &VReg, a: &VReg) -> fmt::Result {
    write!(f, "{:<7}", m)?;
    fmt_vreg(f, r)?; write!(f, ", ")?;
    fmt_vreg(f, a)
}

fn bin(f: &mut fmt::Formatter<'_>, m: &str, r: &VReg, a: &VReg, b: &VReg) -> fmt::Result {
    write!(f, "{:<7}", m)?;
    fmt_vreg(f, r)?; write!(f, ", ")?;
    fmt_vreg(f, a)?; write!(f, ", ")?;
    fmt_vreg(f, b)
}

fn tri(f: &mut fmt::Formatter<'_>, m: &str, r: &VReg, a: &VReg, b: &VReg, c: &VReg) -> fmt::Result {
    write!(f, "{:<7}", m)?;
    fmt_vreg(f, r)?; write!(f, ", ")?;
    fmt_vreg(f, a)?; write!(f, ", ")?;
    fmt_vreg(f, b)?; write!(f, ", ")?;
    fmt_vreg(f, c)
}

fn qua(f: &mut fmt::Formatter<'_>, m: &str, r: &VReg, a: &VReg, b: &VReg, c: &VReg, d: &VReg) -> fmt::Result {
    write!(f, "{:<7}", m)?;
    fmt_vreg(f, r)?; write!(f, ", ")?;
    fmt_vreg(f, a)?; write!(f, ", ")?;
    fmt_vreg(f, b)?; write!(f, ", ")?;
    fmt_vreg(f, c)?; write!(f, ", ")?;
    fmt_vreg(f, d)
}

impl fmt::Debug for IRSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, block) in self.blocks.iter().enumerate() {
            if block.dead {
                // Stub: preserve the label so block IDs stay aligned
                // with whatever produced the segment, but emit no code.
                writeln!(f, ".L{} (Dead):", i)?;
                continue;
            }
            writeln!(f, ".L{}:", i)?;
            for stmt in &block.statements {
                write!(f, "    ")?;
                fmt_stmt(f, stmt)?;
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
