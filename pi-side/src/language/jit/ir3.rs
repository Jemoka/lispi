//! Third-tier IR ("RIR" — Register IR). Final IR before asm emission.
//!
//! # What this layer adds
//!
//! * **Single register namespace**: ImmReg/HeapReg distinction is gone.
//!   Every operand is a `VReg`. Each opcode's *shape* already tells you
//!   whether the value-in-register is a 32-bit payload or a heap
//!   pointer, so the type tag was redundant.
//! * **No more `Value` literals for numeric loads**: `LoadValueImm`
//!   becomes `MovImm(VReg, ImmNumber)` — a true single-instruction
//!   immediate. Non-numeric constants (Cons, closures, Nil, …) lower
//!   to `MovImmAddr(VReg, Value)` which emits an `ldr r, =LITERAL`
//!   against the literal pool.
//! * **`BindImmediate` is narrowed**: it only handles `ImmNumber` srcs.
//!   `MIRStatement::BindImmediate` with a non-Number src demotes
//!   during conversion to `MovImmAddr` + `BindLocal`.
//! * **Phi is unified** (no PhiOpImm/PhiOpHeap split). The copy-insertion
//!   pass that follows lowers each phi to register moves at predecessor
//!   tails — by this layer the kind sigil isn't load-bearing.
//!
//! # Asm sketch notation
//!
//! Each variant's doc comment includes:
//!   * The moral asm sequence (registers as `r[N]` ↔ `VReg(N)`).
//!   * Clobbers — what state the lowering touches beyond writing dst.
//!     "call-clobbered" means the AAPCS volatile set (r0–r3, r12, lr,
//!     plus condition flags). Helper-call lowerings are tagged this way.
//!
//! # Register layout assumptions
//!
//! The Pi-side runtime keeps boxed values as `Rc<Value>` — under the
//! hood a pointer to a Value cell. For a `Value::Number(n)` cell, the
//! payload word lives at a known offset; `Unbox` lowers to a single
//! `ldr` against that offset. `Box` calls a helper that allocates a
//! Number cell and stores the payload. Capture `Binding`s are
//! pointer-stable across calls (the closure's env snapshot owns
//! them), so we bake the slot address as an `ldr =LIT` literal-pool
//! load.

use alloc::vec::Vec;
use core::fmt;

use super::ir::Name;
use super::ir2::{HeapReg, ImmReg, MIRSegment, MIRStatement};
use super::scope::LocalId;
use crate::language::ast::Value;
use crate::language::environment::Binding;
use crate::language::number::Number;

/// Single-namespace register, same shape as `VReg` in the SSA IR but
/// living in MIR's unified pool (ImmReg/HeapReg IDs after optimize2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct VReg(pub(super) u32);

impl From<ImmReg> for VReg {
    fn from(r: ImmReg) -> Self {
        VReg(r.0)
    }
}
impl From<HeapReg> for VReg {
    fn from(r: HeapReg) -> Self {
        VReg(r.0)
    }
}

/// 32-bit immediate payload, tagged by source numeric type for the
/// pretty-printer and any downstream coercion checks. At asm-emit time
/// the bit pattern is what lands in the `mov rN, #imm` (or in a
/// literal-pool entry if the value doesn't fit in a single instruction
/// encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImmNumber {
    Integer(i32),
    Unsigned(u32),
    Addr(usize),
}

impl ImmNumber {
    /// Try to lift a `Value` into an `ImmNumber`. Returns `None` for
    /// non-numeric constants (Cons, Closure, Nil, …) — caller demotes
    /// to a `MovImmAddr` (heap-pointer load) in that case.
    fn from_value(v: &Value) -> Option<Self> {
        match v {
            Value::Number(Number::Integer(i)) => Some(ImmNumber::Integer(*i)),
            Value::Number(Number::Unsigned(u)) => Some(ImmNumber::Unsigned(*u)),
            Value::Number(Number::Addr(a)) => Some(ImmNumber::Addr(*a)),
            // Bool collapses to Integer 0/1 — fits an immediate slot.
            Value::Bool(b) => Some(ImmNumber::Integer(if *b { 1 } else { 0 })),
            _ => None,
        }
    }
}

impl fmt::Display for ImmNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImmNumber::Integer(i) => write!(f, "{}i", i),
            ImmNumber::Unsigned(u) => write!(f, "{}u", u),
            ImmNumber::Addr(a) => write!(f, "{:#x}a", a),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RIRStatement {
    // ===================== loads =====================
    /// `mov r[dst], #imm` (or a literal-pool load if `imm` doesn't fit
    /// the encoding). Replaces ir2's `LoadValueImm` for `Number`/`Bool`.
    /// **Clobbers:** `r[dst]`.
    MovImm(VReg, ImmNumber),

    /// `ldr r[dst], =VALUE_RC` — load the address of a static `Value`
    /// cell from the literal pool. Used for non-numeric constants
    /// (`Nil`, `Cons`, closures, syscalls, special tokens, …).
    /// **Clobbers:** `r[dst]`.
    MovImmAddr(VReg, Value),

    // ===================== runtime scope management =====================
    /// `bl push_frame_trampoline` — pushes an empty owned frame onto
    /// the runtime Image. Marks entry to a `let` scope.
    /// **Clobbers:** call-clobbered (r0–r3, r12, lr, flags).
    PushFrame,
    /// `bl pop_frame_trampoline` — pops the top frame from the runtime
    /// Image. Marks exit from a `let` scope.
    /// **Clobbers:** call-clobbered.
    PopFrame,

    /// `mov r0, r[src]; mov r1, #<id>; ldr r2, =NAME; bl bind_local`
    /// — write `locals[id]`'s inner value from src and register
    /// `(name → locals[id])` in the top runtime frame.
    /// **Clobbers:** call-clobbered.
    BindLocal {
        name: Name,
        id: LocalId,
        src: VReg,
    },

    /// `mov r0, #<src>; mov r1, #<id>; ldr r2, =NAME; bl bind_local`
    /// — same as `BindLocal` but the value is inlined as an immediate.
    /// MIR `BindImmediate` with non-Number/non-Bool src demotes during
    /// conversion to `MovImmAddr` + `BindLocal`.
    /// **Clobbers:** call-clobbered.
    BindImmediate {
        name: Name,
        id: LocalId,
        src: ImmNumber,
    },

    /// `mov r0, #<id>; bl load_local; mov r[dst], r0` — fetch the
    /// current heap-boxed value out of `locals[id]`.
    /// **Clobbers:** call-clobbered.
    LoadLocal(VReg, LocalId),

    /// `ldr r12, =BINDING; ldr r[dst], [r12]` — the `Binding` literal
    /// holds a stable pointer to the captured slot's `Rc<Value>` cell.
    /// **Clobbers:** `r[dst]`, scratch (r12).
    LoadCapture(VReg, Binding),

    /// `mov r0, #<id>; mov r1, r[src]; bl store_local` — overwrite
    /// `locals[id]`'s inner value.
    /// **Clobbers:** call-clobbered.
    StoreLocal(LocalId, VReg),

    /// `ldr r12, =BINDING; str r[src], [r12]` — store through the
    /// captured slot. Observable by every closure sharing this Binding.
    /// **Clobbers:** scratch (r12).
    StoreCapture(Binding, VReg),

    // ===================== casts =====================
    /// `ldr r[dst], [r[src], #PAYLOAD_OFFSET]` — read the 32-bit
    /// payload word out of a boxed `Value::Number` cell.
    /// **Clobbers:** `r[dst]`.
    Unbox(VReg, VReg),

    /// **Operand order: (src_imm, dst_heap).** Calls a helper that
    /// allocates a fresh `Value::Number` cell and stores the payload.
    ///   `mov r0, r[src]; bl box_number; mov r[dst], r0`
    /// **Clobbers:** call-clobbered.
    Box(VReg, VReg),

    /// Collapse any boxed value to its 0/1 falsy test.
    ///   `mov r0, r[src]; bl is_truthy; mov r[dst], r0`
    /// (or an inline tag-check + cset-style materialization for fast
    /// paths). Used for `CondBr` conditions.
    /// **Clobbers:** call-clobbered.
    Truthy(VReg, VReg),

    /// `mov r0, #<id>; bl unbox_local; mov r[dst], r0` — fused
    /// `LoadLocal + Unbox` introduced by `optimize2`'s peephole.
    /// **Clobbers:** call-clobbered.
    UnboxLocal(VReg, LocalId),

    // ===================== arithmetic =====================
    //
    // All operands are 32-bit imm-shaped (i.e., live in a general
    // register without indirection). Three-address ARM forms map
    // directly. **Clobbers:** `r[dst]`, condition flags.
    /// `add r[dst], r[a], r[b]`
    Add(VReg, VReg, VReg),
    /// `sub r[dst], r[a], r[b]`
    Sub(VReg, VReg, VReg),
    /// `mul r[dst], r[a], r[b]` (ARM `mul` allows same operands).
    Mul(VReg, VReg, VReg),
    /// `bl __divsi3` (or arch-native `sdiv` on ARMv7+):
    /// `mov r0, r[a]; mov r1, r[b]; bl __divsi3; mov r[dst], r0`.
    /// **Clobbers:** call-clobbered.
    Div(VReg, VReg, VReg),
    /// `bl __modsi3` analogous to `Div`.
    /// **Clobbers:** call-clobbered.
    Mod(VReg, VReg, VReg),
    /// `lsl r[dst], r[a], r[b]`
    Lshift(VReg, VReg, VReg),
    /// `lsr r[dst], r[a], r[b]`
    Rshift(VReg, VReg, VReg),

    // ===================== bitwise =====================
    /// `mvn r[dst], r[src]`
    /// **Clobbers:** `r[dst]`.
    BinNot(VReg, VReg),
    /// `orr r[dst], r[a], r[b]`
    /// **Clobbers:** `r[dst]`, flags.
    BinOr(VReg, VReg, VReg),
    /// `and r[dst], r[a], r[b]`
    /// **Clobbers:** `r[dst]`, flags.
    BinAnd(VReg, VReg, VReg),

    // ===================== logical (eager) =====================
    /// Per-tag-type `not`, matching the interpreter's rules. Easiest
    /// path: `mov r0, r[src]; bl lognot; mov r[dst], r0`.
    /// **Clobbers:** call-clobbered.
    LogNot(VReg, VReg),
    /// Eager xor on truthiness: `r[dst] = is_truthy(a) ^ is_truthy(b)`.
    /// `mov r0, r[a]; mov r1, r[b]; bl xor_truthy; mov r[dst], r0`.
    /// **Clobbers:** call-clobbered.
    Xor(VReg, VReg, VReg),

    // ===================== comparisons (produce Bool) =====================
    //
    // Pattern: `cmp r[a], r[b]; movXX r[dst], #1; movYY r[dst], #0`
    // (XX/YY chosen per relation). **Clobbers:** `r[dst]`, flags.
    Eq(VReg, VReg, VReg),
    Gt(VReg, VReg, VReg),
    Lt(VReg, VReg, VReg),
    Gte(VReg, VReg, VReg),
    Lte(VReg, VReg, VReg),

    // ===================== numeric type coercion =====================
    //
    // In a 32-bit register-only world these are bit-equivalent moves —
    // the type tag exists only in `ImmNumber`'s sigil for the printer.
    // The Box that wraps the final result picks the right Number variant.
    //   `mov r[dst], r[src]`
    // **Clobbers:** `r[dst]`.
    AsAddr(VReg, VReg),
    AsSigned(VReg, VReg),
    AsUnsigned(VReg, VReg),

    // ===================== cons / list =====================
    /// `mov r0, r[a]; mov r1, r[b]; bl cons_alloc; mov r[dst], r0`
    /// **Clobbers:** call-clobbered.
    Cons(VReg, VReg, VReg),
    /// `ldr r[dst], [r[src], #CAR_OFFSET]`
    /// **Clobbers:** `r[dst]`.
    Car(VReg, VReg),
    /// `ldr r[dst], [r[src], #CDR_OFFSET]`
    /// **Clobbers:** `r[dst]`.
    Cdr(VReg, VReg),
    /// `ldr r12, [r[src]]; cmp r12, =NIL_TAG; movXX r[dst], #1; movYY r[dst], #0`
    /// (or a `bl is_nil` helper for clarity).
    /// **Clobbers:** `r[dst]`, scratch (r12), flags.
    Nullp(VReg, VReg),

    // ===================== arrays =====================
    /// `mov r0, r[src]; bl array_pack; mov r[dst], r0`
    /// **Clobbers:** call-clobbered.
    Array(VReg, VReg),
    /// `mov r0, r[n]; mov r1, r[val]; bl array_full; mov r[dst], r0`
    /// **Clobbers:** call-clobbered.
    Full(VReg, VReg, VReg),
    /// `mov r0, r[src]; bl array_unpack; mov r[dst], r0`
    /// **Clobbers:** call-clobbered.
    Unpack(VReg, VReg),
    /// `mov r0, r[arr]; mov r1, r[idx]; bl array_getidx; mov r[dst], r0`
    /// (helper does the bounds check + tag dispatch for Array-vs-Addr).
    /// **Clobbers:** call-clobbered.
    GetIdx(VReg, VReg, VReg),
    /// `mov r0, …; … bl array_putidx; …`
    /// **Clobbers:** call-clobbered.
    PutIdx(VReg, VReg, VReg, VReg),
    /// `bl array_readidx` — reads n u32s into a fresh list.
    /// **Clobbers:** call-clobbered.
    ReadIdx(VReg, VReg, VReg, VReg),
    /// `bl array_fillidx` — writes list values from offset.
    /// **Clobbers:** call-clobbered.
    FillIdx(VReg, VReg, VReg, VReg),
    /// `bl array_fullidx` — fills n slots with val from offset.
    /// **Clobbers:** call-clobbered.
    FullIdx(VReg, VReg, VReg, VReg, VReg),

    // ===================== introspection =====================
    /// `ldr r[dst], [r[src], #HITS_OFFSET]` — read closure/macro hit
    /// counter as Unsigned. (Assumes Closure layout exposes `hits` at
    /// a known offset within the Rc<Closure>.)
    /// **Clobbers:** `r[dst]`.
    Hits(VReg, VReg),

    // ===================== control flow =====================
    /// `b .L<target>` — unconditional branch.
    /// **Clobbers:** none.
    Br(usize),

    /// `cmp r[cond], #0; beq .L<else>; b .L<then>` — cond is a 0/1
    /// produced by `Truthy` or an `eq/gt/…` comparison.
    /// **Clobbers:** flags.
    CondBr {
        cond: VReg,
        then_blk: usize,
        else_blk: usize,
    },

    /// SSA phi. Lowered by a post-pass copy-insertion to register moves
    /// at each predecessor's tail (just before its terminator). After
    /// regalloc this turns into `mov r[dst], r[src_for_this_edge]`
    /// emitted on the appropriate fall-through path.
    /// **Clobbers:** at lowering time, the dst reg (via the inserted
    /// `mov`). Dead predecessors (`block.dead`) get no emit.
    PhiOp(VReg, (usize, VReg), (usize, VReg)),

    /// `mov r0, r[src]; b .Lepilog` (or epilog inlined: pop callee-
    /// saves + `bx lr`). The result lives in r0 by AAPCS.
    /// **Clobbers:** call-clobbered.
    Ret(VReg),

    // ===================== syscalls =====================
    //
    // The JIT specializes these so they can lower to inline asm rather
    // than going through the SWI trampoline. Each comment lists the
    // moral instruction sequence; helper calls are flagged.
    /// `dsb sy`
    /// **Clobbers:** memory-ordering effect only.
    SysDsb(VReg),
    /// `mcr p15, …` prefetch-flush sequence.
    /// **Clobbers:** call-clobbered (helper) or scratch (inline).
    SysPrefetchFlush(VReg),
    /// `bl uart_init`
    /// **Clobbers:** call-clobbered.
    SysUartInit(VReg),
    /// `bl uart_get8; mov r[dst], r0`
    /// **Clobbers:** call-clobbered.
    SysUartGet8(VReg),
    /// `bl monitor_clear`
    /// **Clobbers:** call-clobbered.
    SysClearMonitor(VReg),
    /// `bl monitor_get; mov r[dst], r0`
    /// **Clobbers:** call-clobbered.
    SysGetMonitor(VReg),
    /// `bl monitor_stop`
    /// **Clobbers:** call-clobbered.
    SysStopMonitor(VReg),

    /// `ldr r[dst], [r[addr]]` — raw MMIO 32-bit load.
    /// **Clobbers:** `r[dst]`.
    SysGet32(VReg, VReg),
    /// `mov r0, r[byte]; bl uart_put8`
    /// **Clobbers:** call-clobbered.
    SysUartPut8(VReg, VReg),
    /// `mov r0, r[count]; bl delay`
    /// **Clobbers:** call-clobbered.
    SysDelay(VReg, VReg),

    /// `str r[val], [r[addr]]` — raw MMIO 32-bit store.
    /// **Clobbers:** none.
    SysPut32(VReg, VReg, VReg),

    /// `bl zero32` (memset-shaped fill of 0).
    /// **Clobbers:** call-clobbered.
    SysZero32(VReg, VReg, VReg, VReg),
    /// `bl/inline str` (copy array contents to destination).
    /// **Clobbers:** call-clobbered/scratch.
    SysStr(VReg, VReg, VReg, VReg),

    /// `bl full32` (memset-shaped fill of `val`).
    /// **Clobbers:** call-clobbered.
    SysFull32(VReg, VReg, VReg, VReg, VReg),

    // ===================== escape =====================
    /// `mov r0, r[src]; bl escape_to_interp; mov r[dst], r0` — bail
    /// out into the interpreter for an arbitrary sexp evaluation.
    /// **Clobbers:** call-clobbered — and conceptually any local slot
    /// the interpreter can reach via `set!` (already taken into account
    /// by the optimizer's escape-taint).
    Escape(VReg, VReg),

    // ===================== direct call =====================
    /// `mov r0, =BINDING; mov r1, r[args[0]]; … ; bl call_N; mov r[dst], r0`
    /// where N ∈ {1,2,3}. The runtime helper resolves the binding to a
    /// `Value::Closure`, dispatches through the executor's JC cache,
    /// and returns the result slot id. Falls back to `evaluate()` if
    /// the binding doesn't currently hold a Closure or types mismatch.
    /// **Clobbers:** call-clobbered.
    Call {
        dst: VReg,
        callee: Binding,
        args: alloc::vec::Vec<VReg>,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct RIRBasicBlock {
    pub statements: Vec<RIRStatement>,
    /// Same lowering-contract semantics as `IRBasicBlock::dead` —
    /// asm-emit should skip dead blocks entirely.
    pub dead: bool,
}

#[derive(Clone)]
pub(crate) struct RIRSegment {
    pub blocks: Vec<RIRBasicBlock>,
}

// ===================== conversion =====================

impl From<MIRSegment> for RIRSegment {
    fn from(seg: MIRSegment) -> Self {
        // Demotion of non-Number `BindImmediate` may need a fresh VReg
        // (a MovImmAddr followed by a normal BindLocal). Sourcing fresh
        // IDs from the MIR's unified counter keeps them collision-free
        // with everything we inherit; we don't carry the counter past
        // this conversion — IR4 (regalloc) owns its own state.
        let mut next_reg = seg.next_reg;
        let mut blocks: Vec<RIRBasicBlock> = Vec::with_capacity(seg.blocks.len());

        for b in seg.blocks {
            let mut out: Vec<RIRStatement> = Vec::with_capacity(b.statements.len());
            for stmt in b.statements {
                lower(stmt, &mut out, &mut next_reg);
            }
            blocks.push(RIRBasicBlock {
                statements: out,
                dead: b.dead,
            });
        }

        RIRSegment { blocks }
    }
}

/// Translate one MIR statement into RIR. Most opcodes are 1:1 with a
/// type-narrowing on registers. The two exceptions:
///   * `LoadValueImm(d, v)` with non-numeric `v` → `MovImmAddr` (rare;
///     should be folded into LoadValuePtr by ir2 in practice, but we
///     handle it defensively here).
///   * `BindImmediate { src: v }` with non-numeric `v` → demote to
///     `MovImmAddr(fresh, v)` + `BindLocal { src: fresh }`.
fn lower(stmt: MIRStatement, out: &mut Vec<RIRStatement>, next_reg: &mut u32) {
    use MIRStatement as M;
    use RIRStatement as R;
    match stmt {
        M::LoadValueImm(d, v) => match ImmNumber::from_value(&v) {
            Some(imm) => out.push(R::MovImm(d.into(), imm)),
            None => out.push(R::MovImmAddr(d.into(), v)),
        },
        M::LoadValuePtr(d, v) => out.push(R::MovImmAddr(d.into(), v)),

        M::PushFrame => out.push(R::PushFrame),
        M::PopFrame => out.push(R::PopFrame),

        M::BindLocal { name, id, src } => out.push(R::BindLocal {
            name,
            id,
            src: src.into(),
        }),
        M::BindImmediate { name, id, src } => {
            match ImmNumber::from_value(&src) {
                Some(imm) => out.push(R::BindImmediate { name, id, src: imm }),
                None => {
                    // Demote: materialize the heap pointer, then bind it.
                    let fresh = VReg(*next_reg);
                    *next_reg += 1;
                    out.push(R::MovImmAddr(fresh, src));
                    out.push(R::BindLocal {
                        name,
                        id,
                        src: fresh,
                    });
                }
            }
        }

        M::LoadLocal(d, id) => out.push(R::LoadLocal(d.into(), id)),
        M::LoadCapture(d, b) => out.push(R::LoadCapture(d.into(), b)),
        M::StoreLocal(id, r) => out.push(R::StoreLocal(id, r.into())),
        M::StoreCapture(b, r) => out.push(R::StoreCapture(b, r.into())),

        M::Unbox(i, h) => out.push(R::Unbox(i.into(), h.into())),
        M::Box(i, h) => out.push(R::Box(i.into(), h.into())),
        M::Truthy(i, h) => out.push(R::Truthy(i.into(), h.into())),
        M::UnboxLocal(i, id) => out.push(R::UnboxLocal(i.into(), id)),

        M::Add(d, a, b) => out.push(R::Add(d.into(), a.into(), b.into())),
        M::Sub(d, a, b) => out.push(R::Sub(d.into(), a.into(), b.into())),
        M::Mul(d, a, b) => out.push(R::Mul(d.into(), a.into(), b.into())),
        M::Div(d, a, b) => out.push(R::Div(d.into(), a.into(), b.into())),
        M::Mod(d, a, b) => out.push(R::Mod(d.into(), a.into(), b.into())),
        M::Lshift(d, a, b) => out.push(R::Lshift(d.into(), a.into(), b.into())),
        M::Rshift(d, a, b) => out.push(R::Rshift(d.into(), a.into(), b.into())),

        M::BinNot(d, a) => out.push(R::BinNot(d.into(), a.into())),
        M::BinOr(d, a, b) => out.push(R::BinOr(d.into(), a.into(), b.into())),
        M::BinAnd(d, a, b) => out.push(R::BinAnd(d.into(), a.into(), b.into())),

        M::LogNot(d, a) => out.push(R::LogNot(d.into(), a.into())),
        M::Xor(d, a, b) => out.push(R::Xor(d.into(), a.into(), b.into())),

        M::Eq(d, a, b) => out.push(R::Eq(d.into(), a.into(), b.into())),
        M::Gt(d, a, b) => out.push(R::Gt(d.into(), a.into(), b.into())),
        M::Lt(d, a, b) => out.push(R::Lt(d.into(), a.into(), b.into())),
        M::Gte(d, a, b) => out.push(R::Gte(d.into(), a.into(), b.into())),
        M::Lte(d, a, b) => out.push(R::Lte(d.into(), a.into(), b.into())),

        M::AsAddr(d, a) => out.push(R::AsAddr(d.into(), a.into())),
        M::AsSigned(d, a) => out.push(R::AsSigned(d.into(), a.into())),
        M::AsUnsigned(d, a) => out.push(R::AsUnsigned(d.into(), a.into())),

        M::Cons(d, a, b) => out.push(R::Cons(d.into(), a.into(), b.into())),
        M::Car(d, a) => out.push(R::Car(d.into(), a.into())),
        M::Cdr(d, a) => out.push(R::Cdr(d.into(), a.into())),
        M::Nullp(d, a) => out.push(R::Nullp(d.into(), a.into())),

        M::Array(d, a) => out.push(R::Array(d.into(), a.into())),
        M::Full(d, a, b) => out.push(R::Full(d.into(), a.into(), b.into())),
        M::Unpack(d, a) => out.push(R::Unpack(d.into(), a.into())),
        M::GetIdx(d, a, b) => out.push(R::GetIdx(d.into(), a.into(), b.into())),
        M::PutIdx(d, t, i, v) => out.push(R::PutIdx(d.into(), t.into(), i.into(), v.into())),
        M::ReadIdx(d, t, o, n) => out.push(R::ReadIdx(d.into(), t.into(), o.into(), n.into())),
        M::FillIdx(d, t, o, l) => out.push(R::FillIdx(d.into(), t.into(), o.into(), l.into())),
        M::FullIdx(d, t, o, n, v) => {
            out.push(R::FullIdx(d.into(), t.into(), o.into(), n.into(), v.into()))
        }

        M::Hits(d, a) => out.push(R::Hits(d.into(), a.into())),

        M::Br(t) => out.push(R::Br(t)),
        M::CondBr {
            cond,
            then_blk,
            else_blk,
        } => out.push(R::CondBr {
            cond: cond.into(),
            then_blk,
            else_blk,
        }),

        M::PhiOpImm(d, (pa, va), (pb, vb)) => {
            out.push(R::PhiOp(d.into(), (pa, va.into()), (pb, vb.into())))
        }
        M::PhiOpHeap(d, (pa, va), (pb, vb)) => {
            out.push(R::PhiOp(d.into(), (pa, va.into()), (pb, vb.into())))
        }

        M::Ret(r) => out.push(R::Ret(r.into())),

        M::SysDsb(d) => out.push(R::SysDsb(d.into())),
        M::SysPrefetchFlush(d) => out.push(R::SysPrefetchFlush(d.into())),
        M::SysUartInit(d) => out.push(R::SysUartInit(d.into())),
        M::SysUartGet8(d) => out.push(R::SysUartGet8(d.into())),
        M::SysClearMonitor(d) => out.push(R::SysClearMonitor(d.into())),
        M::SysGetMonitor(d) => out.push(R::SysGetMonitor(d.into())),
        M::SysStopMonitor(d) => out.push(R::SysStopMonitor(d.into())),

        M::SysGet32(d, a) => out.push(R::SysGet32(d.into(), a.into())),
        M::SysUartPut8(d, a) => out.push(R::SysUartPut8(d.into(), a.into())),
        M::SysDelay(d, a) => out.push(R::SysDelay(d.into(), a.into())),

        M::SysPut32(d, a, b) => out.push(R::SysPut32(d.into(), a.into(), b.into())),

        M::SysZero32(d, a, b, c) => out.push(R::SysZero32(d.into(), a.into(), b.into(), c.into())),

        M::SysStr(d, a, b, c) => out.push(R::SysStr(d.into(), a.into(), b.into(), c.into())),

        M::SysFull32(d, a, b, c, e) => out.push(R::SysFull32(
            d.into(),
            a.into(),
            b.into(),
            c.into(),
            e.into(),
        )),

        M::Escape(d, src) => out.push(R::Escape(d.into(), src.into())),

        M::Call { dst, callee, args } => {
            let rargs: alloc::vec::Vec<VReg> = args.into_iter().map(Into::into).collect();
            out.push(R::Call {
                dst: dst.into(),
                callee,
                args: rargs,
            });
        }
    }
}

// ===================== pretty-printer =====================

fn fmt_reg(f: &mut fmt::Formatter<'_>, r: &VReg) -> fmt::Result {
    write!(f, "r{}", r.0)
}
fn fmt_local(f: &mut fmt::Formatter<'_>, id: &LocalId) -> fmt::Result {
    write!(f, "${}", id.0)
}
fn fmt_binding(f: &mut fmt::Formatter<'_>, b: &Binding) -> fmt::Result {
    write!(f, "[#{:p}]", b.as_ref().as_ptr())
}
fn fmt_blk(f: &mut fmt::Formatter<'_>, b: usize) -> fmt::Result {
    write!(f, ".L{}", b)
}

fn fmt_stmt(f: &mut fmt::Formatter<'_>, s: &RIRStatement) -> fmt::Result {
    const W: usize = 7;
    macro_rules! mn {
        ($m:expr) => {
            write!(f, "{:<width$}", $m, width = W)
        };
    }

    match s {
        RIRStatement::MovImm(r, n) => {
            mn!("mov")?;
            fmt_reg(f, r)?;
            write!(f, ", #{}", n)
        }
        RIRStatement::MovImmAddr(r, v) => {
            mn!("ldr=")?;
            fmt_reg(f, r)?;
            write!(f, ", #{}", v)
        }

        RIRStatement::PushFrame => mn!("pushf"),
        RIRStatement::PopFrame => mn!("popf"),

        RIRStatement::BindLocal { name, id, src } => {
            mn!("bnd")?;
            fmt_local(f, id)?;
            write!(f, ", ")?;
            fmt_reg(f, src)?;
            write!(f, "    ; {:?}", name.as_str())
        }
        RIRStatement::BindImmediate { name, id, src } => {
            mn!("bim")?;
            fmt_local(f, id)?;
            write!(f, ", #{}", src)?;
            write!(f, "    ; {:?}", name.as_str())
        }

        RIRStatement::LoadLocal(r, id) => {
            mn!("ldl")?;
            fmt_reg(f, r)?;
            write!(f, ", ")?;
            fmt_local(f, id)
        }
        RIRStatement::LoadCapture(r, b) => {
            mn!("ldc")?;
            fmt_reg(f, r)?;
            write!(f, ", ")?;
            fmt_binding(f, b)
        }
        RIRStatement::StoreLocal(id, r) => {
            mn!("stl")?;
            fmt_local(f, id)?;
            write!(f, ", ")?;
            fmt_reg(f, r)
        }
        RIRStatement::StoreCapture(b, r) => {
            mn!("stc")?;
            fmt_binding(f, b)?;
            write!(f, ", ")?;
            fmt_reg(f, r)
        }

        RIRStatement::Unbox(d, s) => uno(f, "ubox", d, s),
        RIRStatement::Box(s, d) => uno(f, "box", s, d),
        RIRStatement::Truthy(d, s) => uno(f, "trth", d, s),
        RIRStatement::UnboxLocal(d, id) => {
            mn!("uboxl")?;
            fmt_reg(f, d)?;
            write!(f, ", ")?;
            fmt_local(f, id)
        }

        RIRStatement::Add(r, a, b) => bin(f, "add", r, a, b),
        RIRStatement::Sub(r, a, b) => bin(f, "sub", r, a, b),
        RIRStatement::Mul(r, a, b) => bin(f, "mul", r, a, b),
        RIRStatement::Div(r, a, b) => bin(f, "div", r, a, b),
        RIRStatement::Mod(r, a, b) => bin(f, "mod", r, a, b),
        RIRStatement::Lshift(r, a, b) => bin(f, "lsl", r, a, b),
        RIRStatement::Rshift(r, a, b) => bin(f, "lsr", r, a, b),

        RIRStatement::BinNot(r, a) => uno(f, "bnot", r, a),
        RIRStatement::BinOr(r, a, b) => bin(f, "bor", r, a, b),
        RIRStatement::BinAnd(r, a, b) => bin(f, "band", r, a, b),

        RIRStatement::LogNot(r, a) => uno(f, "lnot", r, a),
        RIRStatement::Xor(r, a, b) => bin(f, "xor", r, a, b),

        RIRStatement::Eq(r, a, b) => bin(f, "eq", r, a, b),
        RIRStatement::Gt(r, a, b) => bin(f, "gt", r, a, b),
        RIRStatement::Lt(r, a, b) => bin(f, "lt", r, a, b),
        RIRStatement::Gte(r, a, b) => bin(f, "gte", r, a, b),
        RIRStatement::Lte(r, a, b) => bin(f, "lte", r, a, b),

        RIRStatement::AsAddr(r, a) => uno(f, "tadr", r, a),
        RIRStatement::AsSigned(r, a) => uno(f, "tsig", r, a),
        RIRStatement::AsUnsigned(r, a) => uno(f, "tuns", r, a),

        RIRStatement::Cons(r, a, b) => bin(f, "cons", r, a, b),
        RIRStatement::Car(r, a) => uno(f, "car", r, a),
        RIRStatement::Cdr(r, a) => uno(f, "cdr", r, a),
        RIRStatement::Nullp(r, a) => uno(f, "null", r, a),

        RIRStatement::Array(r, a) => uno(f, "arr", r, a),
        RIRStatement::Full(r, a, b) => bin(f, "full", r, a, b),
        RIRStatement::Unpack(r, a) => uno(f, "unpk", r, a),
        RIRStatement::GetIdx(r, a, b) => bin(f, "gidx", r, a, b),
        RIRStatement::PutIdx(r, t, i, v) => tri(f, "pidx", r, t, i, v),
        RIRStatement::ReadIdx(r, t, o, n) => tri(f, "ridx", r, t, o, n),
        RIRStatement::FillIdx(r, t, o, l) => tri(f, "fidx", r, t, o, l),
        RIRStatement::FullIdx(r, t, o, n, v) => qua(f, "Fidx", r, t, o, n, v),

        RIRStatement::Hits(r, a) => uno(f, "hits", r, a),

        RIRStatement::Br(b) => {
            mn!("b")?;
            fmt_blk(f, *b)
        }
        RIRStatement::CondBr {
            cond,
            then_blk,
            else_blk,
        } => {
            mn!("cbr")?;
            fmt_reg(f, cond)?;
            write!(f, ", ")?;
            fmt_blk(f, *then_blk)?;
            write!(f, ", ")?;
            fmt_blk(f, *else_blk)
        }
        RIRStatement::PhiOp(r, (ab, av), (bb, bv)) => {
            mn!("phi")?;
            fmt_reg(f, r)?;
            write!(f, ", [")?;
            fmt_blk(f, *ab)?;
            write!(f, ": ")?;
            fmt_reg(f, av)?;
            write!(f, "], [")?;
            fmt_blk(f, *bb)?;
            write!(f, ": ")?;
            fmt_reg(f, bv)?;
            write!(f, "]")
        }
        RIRStatement::Ret(r) => {
            mn!("ret")?;
            fmt_reg(f, r)
        }

        RIRStatement::SysDsb(r) => uno0(f, "@dsb", r),
        RIRStatement::SysPrefetchFlush(r) => uno0(f, "@pfflsh", r),
        RIRStatement::SysUartInit(r) => uno0(f, "@uinit", r),
        RIRStatement::SysUartGet8(r) => uno0(f, "@uget8", r),
        RIRStatement::SysClearMonitor(r) => uno0(f, "@mclr", r),
        RIRStatement::SysGetMonitor(r) => uno0(f, "@mget", r),
        RIRStatement::SysStopMonitor(r) => uno0(f, "@mstp", r),

        RIRStatement::SysGet32(r, a) => uno(f, "@get32", r, a),
        RIRStatement::SysUartPut8(r, a) => uno(f, "@uput8", r, a),
        RIRStatement::SysDelay(r, a) => uno(f, "@delay", r, a),
        RIRStatement::SysPut32(r, a, b) => bin(f, "@put32", r, a, b),
        RIRStatement::SysZero32(r, a, b, c) => tri(f, "@zero32", r, a, b, c),
        RIRStatement::SysStr(r, a, b, c) => tri(f, "@str", r, a, b, c),
        RIRStatement::SysFull32(r, a, b, c, d) => qua(f, "@full32", r, a, b, c, d),

        RIRStatement::Escape(r, src) => uno(f, "esc", r, src),

        RIRStatement::Call { dst, callee, args } => {
            mn!("call")?;
            fmt_reg(f, dst)?;
            write!(f, ", [#{:p}]", callee.as_ref().as_ptr())?;
            for a in args {
                write!(f, ", ")?;
                fmt_reg(f, a)?;
            }
            Ok(())
        }
    }
}

fn uno0(f: &mut fmt::Formatter<'_>, m: &str, r: &VReg) -> fmt::Result {
    write!(f, "{:<7}", m)?;
    fmt_reg(f, r)
}
fn uno(f: &mut fmt::Formatter<'_>, m: &str, r: &VReg, a: &VReg) -> fmt::Result {
    write!(f, "{:<7}", m)?;
    fmt_reg(f, r)?;
    write!(f, ", ")?;
    fmt_reg(f, a)
}
fn bin(f: &mut fmt::Formatter<'_>, m: &str, r: &VReg, a: &VReg, b: &VReg) -> fmt::Result {
    write!(f, "{:<7}", m)?;
    fmt_reg(f, r)?;
    write!(f, ", ")?;
    fmt_reg(f, a)?;
    write!(f, ", ")?;
    fmt_reg(f, b)
}
fn tri(f: &mut fmt::Formatter<'_>, m: &str, r: &VReg, a: &VReg, b: &VReg, c: &VReg) -> fmt::Result {
    write!(f, "{:<7}", m)?;
    fmt_reg(f, r)?;
    write!(f, ", ")?;
    fmt_reg(f, a)?;
    write!(f, ", ")?;
    fmt_reg(f, b)?;
    write!(f, ", ")?;
    fmt_reg(f, c)
}
fn qua(
    f: &mut fmt::Formatter<'_>,
    m: &str,
    r: &VReg,
    a: &VReg,
    b: &VReg,
    c: &VReg,
    d: &VReg,
) -> fmt::Result {
    write!(f, "{:<7}", m)?;
    fmt_reg(f, r)?;
    write!(f, ", ")?;
    fmt_reg(f, a)?;
    write!(f, ", ")?;
    fmt_reg(f, b)?;
    write!(f, ", ")?;
    fmt_reg(f, c)?;
    write!(f, ", ")?;
    fmt_reg(f, d)
}

impl fmt::Debug for RIRSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, block) in self.blocks.iter().enumerate() {
            if block.dead {
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
