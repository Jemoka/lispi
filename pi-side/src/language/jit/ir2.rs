//! Second-tier IR: registers are typed as either `ImmReg` (anything that
//! fits in a single 32-bit slot — `Number`, `Bool`) or `HeapReg` (boxed
//! heap values: `Cons`, `Closure`, `Array`, ...).
//!
//! # Lowering from `IRSegment`
//!
//! Each `VReg` is classified into exactly one kind based on its defining
//! statement, then re-numbered into the matching namespace **with its
//! original ID preserved**: `VReg(5)` classified Imm becomes `ImmReg(5)`;
//! classified Heap becomes `HeapReg(5)`. Because `ImmReg` and `HeapReg`
//! live in separate namespaces, the kind sigil (`i`/`h`) disambiguates.
//! IDs are sparse within each namespace.
//!
//! Casts are emitted on the fly when a use-site's expected kind differs
//! from the operand's natural kind:
//!   * `Unbox(ImmReg, HeapReg)` — read a 32-bit value out of a boxed slot
//!     (arith / cmp / coerce / bitwise consumers)
//!   * `Box(ImmReg, HeapReg)`   — wrap a 32-bit value into a fresh box
//!     (consumers that take HeapReg: BindLocal, Cons, Ret, etc.)
//!   * `Truthy(ImmReg, HeapReg)` — collapse any heap value to its
//!     `is_falsy` 0/1 reading. Used for `CondBr` conditions whose source
//!     is heap-typed (we only need a branch decision, not the value).
//!
//! Phi gets two variants: `PhiOpImm` (both arms Imm) and `PhiOpHeap`
//! (any heap arm). When `PhiOpHeap` has an Imm-natural arm, a `Box` is
//! injected at the **predecessor's tail, just before its terminator** —
//! that way the boxed value dominates the phi's join.
//!
//! Shim registers (cast dsts, phi-box dsts) get freshly minted IDs
//! starting at `original_max + 1` so they're visibly distinguishable
//! from preserved-VReg IDs.
//!
//! # Unified ID namespace
//!
//! `ImmReg` and `HeapReg` are *typed wrappers* over a shared `u32` ID
//! pool. Each VReg lives in exactly one namespace at lowering time, so
//! preserved IDs already don't collide. Freshly-minted shim regs draw
//! from a single `next_reg` counter so an `ImmReg(N)` and a `HeapReg(N)`
//! never coexist for the same N. After this layer, downstream passes
//! (coloring, regalloc) can use the bare integer ID without caring
//! about the kind sigil.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;

use crate::language::ast::Value;
use crate::language::environment::Binding;
use super::ir::{IRBasicBlock, IRSegment, IRStatement, Name, VReg};
use super::scope::LocalId;

/// Register holding a value that fits in a single 32-bit slot
/// (`Number` variants, `Bool`, the falsy-test from `Truthy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ImmReg(pub(super) u32);

/// Register holding a boxed heap value (`Cons`, `Closure`, `Array`,
/// `String`, `Nil`, `Special`, `Syscall`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct HeapReg(pub(super) u32);

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MIRStatement {
    // --- loads ---
    /// Load an inline 32-bit-fitting value into an `ImmReg`. Per the
    /// classifier this is emitted when the source IR was a `Load` of a
    /// `Number` or `Bool`.
    LoadValueImm(ImmReg, Value),
    /// Load a heap-resident value into a `HeapReg`.
    LoadValuePtr(HeapReg, Value),

    // --- runtime scope management (mirrors IR) ---
    PushFrame,
    PopFrame,
    BindLocal { name: Name, id: LocalId, src: HeapReg },
    BindImmediate { name: Name, id: LocalId, src: Value },

    LoadLocal(HeapReg, LocalId),
    LoadCapture(HeapReg, Binding),
    StoreLocal(LocalId, HeapReg),
    StoreCapture(Binding, HeapReg),

    // --- casts (analogues of ldr/str/test on tagged box layout) ---
    /// `ldr i, [h]` — read 32-bit payload out of a boxed slot.
    Unbox(ImmReg, HeapReg),
    /// `str i, [h]` — allocate a fresh heap slot, store the 32-bit payload
    /// into it. The HeapReg is the freshly-minted destination.
    Box(ImmReg, HeapReg),
    /// `i = !is_falsy(h)` — collapse any heap value to 0/1. Used for
    /// `CondBr` conditions; never carries a value through a phi.
    Truthy(ImmReg, HeapReg),

    // --- intetrmediate version of casks ---
    UnboxLocal(ImmReg, LocalId),

    // --- arithmetic / bitwise / cmp / coerce (all 32-bit) ---
    Add(ImmReg, ImmReg, ImmReg),
    Sub(ImmReg, ImmReg, ImmReg),
    Mul(ImmReg, ImmReg, ImmReg),
    Div(ImmReg, ImmReg, ImmReg),
    Mod(ImmReg, ImmReg, ImmReg),
    Lshift(ImmReg, ImmReg, ImmReg),
    Rshift(ImmReg, ImmReg, ImmReg),

    BinNot(ImmReg, ImmReg),
    BinOr(ImmReg, ImmReg, ImmReg),
    BinAnd(ImmReg, ImmReg, ImmReg),

    LogNot(ImmReg, ImmReg),
    Xor(ImmReg, ImmReg, ImmReg),

    Eq(ImmReg, ImmReg, ImmReg),
    Gt(ImmReg, ImmReg, ImmReg),
    Lt(ImmReg, ImmReg, ImmReg),
    Gte(ImmReg, ImmReg, ImmReg),
    Lte(ImmReg, ImmReg, ImmReg),

    AsAddr(ImmReg, ImmReg),
    AsSigned(ImmReg, ImmReg),
    AsUnsigned(ImmReg, ImmReg),

    // --- cons / list / array primitives ---
    Cons(HeapReg, HeapReg, HeapReg),
    Car(HeapReg, HeapReg),
    Cdr(HeapReg, HeapReg),
    /// Bool result fits in `ImmReg` (harmonized vs raw IR).
    Nullp(ImmReg, HeapReg),

    Array(HeapReg, HeapReg),
    Full(HeapReg, HeapReg, HeapReg),
    Unpack(HeapReg, HeapReg),
    /// u32 result fits in `ImmReg` (harmonized vs raw IR).
    GetIdx(HeapReg, HeapReg, HeapReg),
    PutIdx(HeapReg, HeapReg, HeapReg, HeapReg),
    ReadIdx(HeapReg, HeapReg, HeapReg, HeapReg),
    FillIdx(HeapReg, HeapReg, HeapReg, HeapReg),
    FullIdx(HeapReg, HeapReg, HeapReg, HeapReg, HeapReg),

    Hits(ImmReg, HeapReg),

    // --- control flow ---
    Br(usize),
    CondBr { cond: ImmReg, then_blk: usize, else_blk: usize },

    /// Phi over imm-typed arms (both predecessors deliver an `ImmReg`).
    PhiOpImm(ImmReg, (usize, ImmReg), (usize, ImmReg)),
    /// Phi over heap-typed arms. If an arm is naturally imm, a `Box`
    /// gets injected at the predecessor's tail and the boxed `HeapReg`
    /// is what appears here.
    PhiOpHeap(HeapReg, (usize, HeapReg), (usize, HeapReg)),

    /// Functions return a heap-shaped Value (the trampoline reads it
    /// back into the interpreter's `Value` type). Imm-typed results
    /// get boxed before `Ret`.
    Ret(HeapReg),

    // --- syscalls ---
    SysDsb(ImmReg),
    SysPrefetchFlush(ImmReg),
    SysUartInit(ImmReg),
    SysUartGet8(ImmReg),
    SysClearMonitor(ImmReg),
    SysGetMonitor(ImmReg),
    SysStopMonitor(ImmReg),

    SysGet32(ImmReg, ImmReg),
    SysUartPut8(ImmReg, ImmReg),
    SysDelay(ImmReg, ImmReg),

    SysPut32(ImmReg, ImmReg, ImmReg),

    SysZero32(ImmReg, ImmReg, ImmReg, ImmReg),

    SysFull32(ImmReg, ImmReg, ImmReg, ImmReg, ImmReg),

    // --- direct closure call ---
    /// Direct closure dispatch via captured Binding. Result is always
    /// a HeapReg (the callee's return slot). All args boxed up.
    Call { dst: HeapReg, callee: Binding, args: Vec<HeapReg> },

    // --- escape (interpreter bailout) ---
    Escape(HeapReg, HeapReg),
}

#[derive(Clone, Debug)]
pub(crate) struct MIRBasicBlock {
    pub statements: Vec<MIRStatement>,
    /// Same semantics as `IRBasicBlock::dead` — preserved across the
    /// lowering so the printer / asm-emit can render stubs.
    pub dead: bool,
}

#[derive(Clone)]
pub(crate) struct MIRSegment {
    /// Highest register id used across BOTH namespaces (one past —
    /// first free). Imm and Heap IDs never collide thanks to the
    /// shared counter; downstream passes can ignore the kind sigil.
    pub next_reg: u32,
    pub blocks: Vec<MIRBasicBlock>,
}

// ===================== conversion =====================

/// Natural kind of a VReg, determined by what produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind { Imm, Heap }

/// Classify the destination of one IR statement into Imm or Heap.
/// PhiOp's kind is the meet of its arms (both Imm → Imm; otherwise Heap).
/// Predecessors precede merges in the block ordering (DAG IR, no loops),
/// so arm lookups always hit a previously-classified entry.
fn classify_dst(stmt: &IRStatement, kinds: &mut BTreeMap<VReg, Kind>) {
    use IRStatement::*;
    match stmt {
        Load(d, v) => {
            // Numbers and Bools fit in a single 32-bit slot. Everything
            // else (Nil, Cons, Closure, Macro, Special, Syscall, Array,
            // String) goes through the heap path.
            let k = match v {
                Value::Number(_) | Value::Bool(_) => Kind::Imm,
                _ => Kind::Heap,
            };
            kinds.insert(d.clone(), k);
        }

        // arith / bitwise / cmp / logical / coercion / nullp / getidx /
        // hits / sys* — all produce 32-bit-fitting values.
        Add(d, _, _) | Sub(d, _, _) | Mul(d, _, _) | Div(d, _, _)
        | Mod(d, _, _) | Lshift(d, _, _) | Rshift(d, _, _)
        | BinNot(d, _) | BinOr(d, _, _) | BinAnd(d, _, _)
        | LogNot(d, _) | Xor(d, _, _)
        | Eq(d, _, _) | Gt(d, _, _) | Lt(d, _, _) | Gte(d, _, _) | Lte(d, _, _)
        | AsAddr(d, _) | AsSigned(d, _) | AsUnsigned(d, _)
        | Nullp(d, _) | Hits(d, _)
        | SysDsb(d) | SysPrefetchFlush(d) | SysUartInit(d) | SysUartGet8(d)
        | SysClearMonitor(d) | SysGetMonitor(d) | SysStopMonitor(d)
        | SysGet32(d, _) | SysUartPut8(d, _) | SysDelay(d, _)
        | SysPut32(d, _, _)
        | SysZero32(d, _, _, _)
        | SysFull32(d, _, _, _, _)
            => { kinds.insert(d.clone(), Kind::Imm); }

        // LoadLocal/LoadCapture/Cons/Car/Cdr/Array/Full/Unpack/
        // Put/Read/Fill/FullIdx/Escape — all heap-typed.
        LoadLocal(d, _) | LoadCapture(d, _)
        | Cons(d, _, _) | Car(d, _) | Cdr(d, _)
        | Array(d, _) | Full(d, _, _) | Unpack(d, _) | GetIdx(d, _, _)
        | PutIdx(d, _, _, _) | ReadIdx(d, _, _, _) | FillIdx(d, _, _, _)
        | FullIdx(d, _, _, _, _)
        | Escape(d, _)
            => { kinds.insert(d.clone(), Kind::Heap); }

        // Call returns a heap slot id from the runtime helper.
        Call { dst, .. } => { kinds.insert(dst.clone(), Kind::Heap); }

        PhiOp(d, (_, a), (_, b)) => {
            // Default unclassified arms (e.g. arm in a dead block whose
            // defining statement got DCE-dropped) to Heap — that's the
            // safer default since heap is the more general carrier.
            let ka = kinds.get(a).copied().unwrap_or(Kind::Heap);
            let kb = kinds.get(b).copied().unwrap_or(Kind::Heap);
            let k = if ka == Kind::Imm && kb == Kind::Imm { Kind::Imm } else { Kind::Heap };
            kinds.insert(d.clone(), k);
        }

        // No-dst statements.
        PushFrame | PopFrame
        | BindLocal { .. } | BindImmediate { .. }
        | StoreLocal(..) | StoreCapture(..)
        | Br(_) | CondBr { .. } | Ret(_) => {}
    }
}

/// Lowering context — owns the kind map, the per-namespace fresh-id
/// counters, and helpers that emit `Unbox`/`Box`/`Truthy` shims on
/// demand at use sites.
struct LowerCtx {
    kinds: BTreeMap<VReg, Kind>,
    /// Shared shim-id counter across both namespaces.
    next_reg: u32,
}

impl LowerCtx {
    fn new(kinds: BTreeMap<VReg, Kind>, fresh_base: u32) -> Self {
        LowerCtx { kinds, next_reg: fresh_base }
    }

    fn fresh_imm(&mut self) -> ImmReg {
        let r = ImmReg(self.next_reg); self.next_reg += 1; r
    }
    fn fresh_heap(&mut self) -> HeapReg {
        let r = HeapReg(self.next_reg); self.next_reg += 1; r
    }

    fn kind(&self, v: &VReg) -> Kind {
        // Default to Heap if missing — matches `classify_dst`'s phi
        // fallback. In practice every VReg referenced here was a dst
        // somewhere, so this is rarely hit.
        self.kinds.get(v).copied().unwrap_or(Kind::Heap)
    }

    /// Get an `ImmReg` view of `v` for an arithmetic/cmp/etc. use site.
    /// If `v` is heap-natural, emit `Unbox` into a fresh ImmReg.
    fn use_imm(&mut self, v: &VReg, out: &mut Vec<MIRStatement>) -> ImmReg {
        match self.kind(v) {
            Kind::Imm => ImmReg(v.0),
            Kind::Heap => {
                let i = self.fresh_imm();
                out.push(MIRStatement::Unbox(i, HeapReg(v.0)));
                i
            }
        }
    }

    /// Get an `ImmReg` for a `CondBr` cond. Heap-natural sources go
    /// through `Truthy` (a 0/1 falsy test) — not `Unbox`, because we
    /// only need the branch decision, not the inner payload.
    fn use_truthy(&mut self, v: &VReg, out: &mut Vec<MIRStatement>) -> ImmReg {
        match self.kind(v) {
            Kind::Imm => ImmReg(v.0),
            Kind::Heap => {
                let i = self.fresh_imm();
                out.push(MIRStatement::Truthy(i, HeapReg(v.0)));
                i
            }
        }
    }

    /// Get a `HeapReg` view of `v`. Imm-natural sources are boxed.
    fn use_heap(&mut self, v: &VReg, out: &mut Vec<MIRStatement>) -> HeapReg {
        match self.kind(v) {
            Kind::Heap => HeapReg(v.0),
            Kind::Imm => {
                let h = self.fresh_heap();
                out.push(MIRStatement::Box(ImmReg(v.0), h));
                h
            }
        }
    }
}

impl From<IRSegment> for MIRSegment {
    fn from(seg: IRSegment) -> Self {
        // Pass 1 — classify every VReg's natural kind.
        let mut kinds: BTreeMap<VReg, Kind> = BTreeMap::new();
        for block in &seg.blocks {
            for stmt in &block.statements {
                classify_dst(stmt, &mut kinds);
            }
        }

        // Shim IDs start one past the highest VReg id used. `IRSegment::reg`
        // pre-increments, so `seg.regs.0` IS the last assigned id; the
        // first free id is `seg.regs.0 + 1`.
        let fresh_base = seg.regs.0 + 1;
        let mut ctx = LowerCtx::new(kinds, fresh_base);

        // Output blocks pre-sized; we fill statements in order.
        let mut blocks: Vec<MIRBasicBlock> = seg.blocks.iter()
            .map(|b| MIRBasicBlock { statements: Vec::new(), dead: b.dead })
            .collect();

        // Phi predecessor-tail boxes. We can't insert directly while
        // lowering merge blocks, because the predecessor block has
        // already been emitted (with its terminator at the end). We
        // queue here and inject before that terminator in a final pass.
        let mut pending_boxes: Vec<(usize, ImmReg, HeapReg)> = Vec::new();

        for (bi, block) in seg.blocks.iter().enumerate() {
            for stmt in &block.statements {
                lower_stmt(stmt, &mut blocks[bi].statements, &mut ctx, &mut pending_boxes);
            }
        }

        // Inject the queued phi-arm boxes at each predecessor's tail
        // (immediately before its terminator). The terminator is always
        // the last statement of a non-empty block; insert at len-1.
        for (pred, src_imm, dst_heap) in pending_boxes {
            let stmts = &mut blocks[pred].statements;
            let term_idx = stmts.len().saturating_sub(1);
            stmts.insert(term_idx, MIRStatement::Box(src_imm, dst_heap));
        }

        MIRSegment {
            next_reg: ctx.next_reg,
            blocks,
        }
    }
}

/// Translate one IR statement, emitting any required cast shims first.
fn lower_stmt(
    stmt: &IRStatement,
    out: &mut Vec<MIRStatement>,
    ctx: &mut LowerCtx,
    pending_boxes: &mut Vec<(usize, ImmReg, HeapReg)>,
) {
    use IRStatement as I;
    use MIRStatement as M;

    // Imm-dst three-operand op (arith / cmp / bitwise / logical / etc.)
    macro_rules! imm3 {
        ($d:expr, $a:expr, $b:expr, $variant:ident) => {{
            let ra = ctx.use_imm($a, out);
            let rb = ctx.use_imm($b, out);
            out.push(M::$variant(ImmReg($d.0), ra, rb));
        }};
    }
    macro_rules! imm2 {
        ($d:expr, $a:expr, $variant:ident) => {{
            let ra = ctx.use_imm($a, out);
            out.push(M::$variant(ImmReg($d.0), ra));
        }};
    }

    match stmt {
        I::Load(d, v) => {
            // Classifier already decided which namespace `d` lives in;
            // pick the matching emit.
            match ctx.kind(d) {
                Kind::Imm  => out.push(M::LoadValueImm(ImmReg(d.0), v.clone())),
                Kind::Heap => out.push(M::LoadValuePtr(HeapReg(d.0), v.clone())),
            }
        }

        I::PushFrame => out.push(M::PushFrame),
        I::PopFrame  => out.push(M::PopFrame),

        I::BindLocal { name, id, src } => {
            let s = ctx.use_heap(src, out);
            out.push(M::BindLocal { name: name.clone(), id: *id, src: s });
        }
        I::BindImmediate { name, id, src } => {
            out.push(M::BindImmediate { name: name.clone(), id: *id, src: src.clone() });
        }

        I::LoadLocal(d, id)   => out.push(M::LoadLocal(HeapReg(d.0), *id)),
        I::LoadCapture(d, b)  => out.push(M::LoadCapture(HeapReg(d.0), b.clone())),
        I::StoreLocal(id, r)  => {
            let s = ctx.use_heap(r, out);
            out.push(M::StoreLocal(*id, s));
        }
        I::StoreCapture(b, r) => {
            let s = ctx.use_heap(r, out);
            out.push(M::StoreCapture(b.clone(), s));
        }

        // arithmetic
        I::Add(d, a, b)    => imm3!(d, a, b, Add),
        I::Sub(d, a, b)    => imm3!(d, a, b, Sub),
        I::Mul(d, a, b)    => imm3!(d, a, b, Mul),
        I::Div(d, a, b)    => imm3!(d, a, b, Div),
        I::Mod(d, a, b)    => imm3!(d, a, b, Mod),
        I::Lshift(d, a, b) => imm3!(d, a, b, Lshift),
        I::Rshift(d, a, b) => imm3!(d, a, b, Rshift),

        // bitwise
        I::BinNot(d, a)    => imm2!(d, a, BinNot),
        I::BinOr(d, a, b)  => imm3!(d, a, b, BinOr),
        I::BinAnd(d, a, b) => imm3!(d, a, b, BinAnd),

        // logical
        I::LogNot(d, a) => imm2!(d, a, LogNot),
        I::Xor(d, a, b) => imm3!(d, a, b, Xor),

        // comparison
        I::Eq(d, a, b)  => imm3!(d, a, b, Eq),
        I::Gt(d, a, b)  => imm3!(d, a, b, Gt),
        I::Lt(d, a, b)  => imm3!(d, a, b, Lt),
        I::Gte(d, a, b) => imm3!(d, a, b, Gte),
        I::Lte(d, a, b) => imm3!(d, a, b, Lte),

        // coercion
        I::AsAddr(d, a)     => imm2!(d, a, AsAddr),
        I::AsSigned(d, a)   => imm2!(d, a, AsSigned),
        I::AsUnsigned(d, a) => imm2!(d, a, AsUnsigned),

        // cons / list
        I::Cons(d, a, b) => {
            let ra = ctx.use_heap(a, out);
            let rb = ctx.use_heap(b, out);
            out.push(M::Cons(HeapReg(d.0), ra, rb));
        }
        I::Car(d, a) => { let ra = ctx.use_heap(a, out); out.push(M::Car(HeapReg(d.0), ra)); }
        I::Cdr(d, a) => { let ra = ctx.use_heap(a, out); out.push(M::Cdr(HeapReg(d.0), ra)); }
        I::Nullp(d, a) => { let ra = ctx.use_heap(a, out); out.push(M::Nullp(ImmReg(d.0), ra)); }

        // arrays
        I::Array(d, a)   => { let ra = ctx.use_heap(a, out); out.push(M::Array(HeapReg(d.0), ra)); }
        I::Full(d, a, b) => {
            let ra = ctx.use_heap(a, out);
            let rb = ctx.use_heap(b, out);
            out.push(M::Full(HeapReg(d.0), ra, rb));
        }
        I::Unpack(d, a)  => { let ra = ctx.use_heap(a, out); out.push(M::Unpack(HeapReg(d.0), ra)); }
        I::GetIdx(d, a, b) => {
            let ra = ctx.use_heap(a, out);
            let rb = ctx.use_heap(b, out);
            out.push(M::GetIdx(HeapReg(d.0), ra, rb));
        }
        I::PutIdx(d, t, i, v) => {
            let rt = ctx.use_heap(t, out);
            let ri = ctx.use_heap(i, out);
            let rv = ctx.use_heap(v, out);
            out.push(M::PutIdx(HeapReg(d.0), rt, ri, rv));
        }
        I::ReadIdx(d, t, o, n) => {
            let rt = ctx.use_heap(t, out);
            let ro = ctx.use_heap(o, out);
            let rn = ctx.use_heap(n, out);
            out.push(M::ReadIdx(HeapReg(d.0), rt, ro, rn));
        }
        I::FillIdx(d, t, o, l) => {
            let rt = ctx.use_heap(t, out);
            let ro = ctx.use_heap(o, out);
            let rl = ctx.use_heap(l, out);
            out.push(M::FillIdx(HeapReg(d.0), rt, ro, rl));
        }
        I::FullIdx(d, t, o, n, v) => {
            let rt = ctx.use_heap(t, out);
            let ro = ctx.use_heap(o, out);
            let rn = ctx.use_heap(n, out);
            let rv = ctx.use_heap(v, out);
            out.push(M::FullIdx(HeapReg(d.0), rt, ro, rn, rv));
        }

        // introspection
        I::Hits(d, a) => { let ra = ctx.use_heap(a, out); out.push(M::Hits(ImmReg(d.0), ra)); }

        // control flow
        I::Br(t) => out.push(M::Br(*t)),
        I::CondBr { cond, then_blk, else_blk } => {
            let c = ctx.use_truthy(cond, out);
            out.push(M::CondBr { cond: c, then_blk: *then_blk, else_blk: *else_blk });
        }

        I::PhiOp(d, (pa, va), (pb, vb)) => {
            match ctx.kind(d) {
                Kind::Imm => {
                    // Both arms classified Imm (else d would be Heap).
                    out.push(M::PhiOpImm(
                        ImmReg(d.0),
                        (*pa, ImmReg(va.0)),
                        (*pb, ImmReg(vb.0)),
                    ));
                }
                Kind::Heap => {
                    // Heap phi: any Imm-natural arm needs a Box at its
                    // pred-tail. The boxed dst is a fresh HeapReg.
                    let ha = match ctx.kind(va) {
                        Kind::Heap => HeapReg(va.0),
                        Kind::Imm  => {
                            let h = ctx.fresh_heap();
                            pending_boxes.push((*pa, ImmReg(va.0), h));
                            h
                        }
                    };
                    let hb = match ctx.kind(vb) {
                        Kind::Heap => HeapReg(vb.0),
                        Kind::Imm  => {
                            let h = ctx.fresh_heap();
                            pending_boxes.push((*pb, ImmReg(vb.0), h));
                            h
                        }
                    };
                    out.push(M::PhiOpHeap(HeapReg(d.0), (*pa, ha), (*pb, hb)));
                }
            }
        }

        I::Ret(r) => {
            let h = ctx.use_heap(r, out);
            out.push(M::Ret(h));
        }

        // syscalls
        I::SysDsb(d)           => out.push(M::SysDsb(ImmReg(d.0))),
        I::SysPrefetchFlush(d) => out.push(M::SysPrefetchFlush(ImmReg(d.0))),
        I::SysUartInit(d)      => out.push(M::SysUartInit(ImmReg(d.0))),
        I::SysUartGet8(d)      => out.push(M::SysUartGet8(ImmReg(d.0))),
        I::SysClearMonitor(d)  => out.push(M::SysClearMonitor(ImmReg(d.0))),
        I::SysGetMonitor(d)    => out.push(M::SysGetMonitor(ImmReg(d.0))),
        I::SysStopMonitor(d)   => out.push(M::SysStopMonitor(ImmReg(d.0))),

        I::SysGet32(d, a)    => imm2!(d, a, SysGet32),
        I::SysUartPut8(d, a) => imm2!(d, a, SysUartPut8),
        I::SysDelay(d, a)    => imm2!(d, a, SysDelay),

        I::SysPut32(d, a, b) => imm3!(d, a, b, SysPut32),

        I::SysZero32(d, a, b, c) => {
            let ra = ctx.use_imm(a, out);
            let rb = ctx.use_imm(b, out);
            let rc = ctx.use_imm(c, out);
            out.push(M::SysZero32(ImmReg(d.0), ra, rb, rc));
        }

        I::SysFull32(d, a, b, c, e) => {
            let ra = ctx.use_imm(a, out);
            let rb = ctx.use_imm(b, out);
            let rc = ctx.use_imm(c, out);
            let re = ctx.use_imm(e, out);
            out.push(M::SysFull32(ImmReg(d.0), ra, rb, rc, re));
        }

        // escape — pure heap on both sides
        I::Escape(d, src) => {
            let s = ctx.use_heap(src, out);
            out.push(M::Escape(HeapReg(d.0), s));
        }

        // direct call — box each arg, push Call.
        I::Call { dst, callee, args } => {
            let mut harg: alloc::vec::Vec<HeapReg> = alloc::vec::Vec::with_capacity(args.len());
            for a in args {
                harg.push(ctx.use_heap(a, out));
            }
            out.push(M::Call { dst: HeapReg(dst.0), callee: callee.clone(), args: harg });
        }
    }
}

// Suppress unused-import warnings on IRBasicBlock when this file is
// imported but no caller touches the IR struct directly via this module.
#[allow(dead_code)]
fn _keep_irbb_alive(_: IRBasicBlock) {}

// ===================== pretty-printer =====================
//
// Same ARM-asm-ish layout as IR: blocks labeled `.Ln:`, opcodes indented
// and padded. Imm registers print as `iN`, heap registers as `hN` — the
// sigil tells you which namespace.

fn fmt_imm(f: &mut fmt::Formatter<'_>, r: &ImmReg) -> fmt::Result {
    write!(f, "i{}", r.0)
}
fn fmt_heap(f: &mut fmt::Formatter<'_>, r: &HeapReg) -> fmt::Result {
    write!(f, "h{}", r.0)
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

fn fmt_stmt(f: &mut fmt::Formatter<'_>, s: &MIRStatement) -> fmt::Result {
    const W: usize = 7;
    macro_rules! mn { ($m:expr) => { write!(f, "{:<width$}", $m, width = W) }; }

    match s {
        MIRStatement::LoadValueImm(r, v) => {
            mn!("ldvi")?; fmt_imm(f, r)?; write!(f, ", #{}", v)
        }
        MIRStatement::LoadValuePtr(r, v) => {
            mn!("ldvp")?; fmt_heap(f, r)?; write!(f, ", #{}", v)
        }

        MIRStatement::PushFrame => mn!("pushf"),
        MIRStatement::PopFrame  => mn!("popf"),

        MIRStatement::BindLocal { name, id, src } => {
            mn!("bnd")?;
            fmt_local(f, id)?; write!(f, ", ")?; fmt_heap(f, src)?;
            write!(f, "    ; {:?}", name.as_str())
        }
        MIRStatement::BindImmediate { name, id, src } => {
            mn!("bim")?;
            fmt_local(f, id)?; write!(f, ", #{}", src)?;
            write!(f, "    ; {:?}", name.as_str())
        }

        MIRStatement::LoadLocal(r, id)   => { mn!("ldl")?; fmt_heap(f, r)?; write!(f, ", ")?; fmt_local(f, id) }
        MIRStatement::LoadCapture(r, b)  => { mn!("ldc")?; fmt_heap(f, r)?; write!(f, ", ")?; fmt_binding(f, b) }
        MIRStatement::StoreLocal(id, r)  => { mn!("stl")?; fmt_local(f, id)?; write!(f, ", ")?; fmt_heap(f, r) }
        MIRStatement::StoreCapture(b, r) => { mn!("stc")?; fmt_binding(f, b)?; write!(f, ", ")?; fmt_heap(f, r) }

        // casts
        MIRStatement::Unbox(i, h)  => { mn!("ubox")?; fmt_imm(f, i)?; write!(f, ", ")?; fmt_heap(f, h) }
        MIRStatement::Box(i, h)    => { mn!("box")?;  fmt_imm(f, i)?; write!(f, ", ")?; fmt_heap(f, h) }
        MIRStatement::Truthy(i, h) => { mn!("trth")?; fmt_imm(f, i)?; write!(f, ", ")?; fmt_heap(f, h) }
        MIRStatement::UnboxLocal(i, id) => {
            mn!("uboxl")?; fmt_imm(f, i)?; write!(f, ", ")?; fmt_local(f, id)
        }

        // arith / bitwise / cmp / logical — all imm-imm-imm or imm-imm
        MIRStatement::Add(r,a,b)    => bin_i(f, "add",  r, a, b),
        MIRStatement::Sub(r,a,b)    => bin_i(f, "sub",  r, a, b),
        MIRStatement::Mul(r,a,b)    => bin_i(f, "mul",  r, a, b),
        MIRStatement::Div(r,a,b)    => bin_i(f, "div",  r, a, b),
        MIRStatement::Mod(r,a,b)    => bin_i(f, "mod",  r, a, b),
        MIRStatement::Lshift(r,a,b) => bin_i(f, "lsl",  r, a, b),
        MIRStatement::Rshift(r,a,b) => bin_i(f, "lsr",  r, a, b),
        MIRStatement::BinOr(r,a,b)  => bin_i(f, "bor",  r, a, b),
        MIRStatement::BinAnd(r,a,b) => bin_i(f, "band", r, a, b),
        MIRStatement::Xor(r,a,b)    => bin_i(f, "xor",  r, a, b),
        MIRStatement::Eq(r,a,b)     => bin_i(f, "eq",   r, a, b),
        MIRStatement::Gt(r,a,b)     => bin_i(f, "gt",   r, a, b),
        MIRStatement::Lt(r,a,b)     => bin_i(f, "lt",   r, a, b),
        MIRStatement::Gte(r,a,b)    => bin_i(f, "gte",  r, a, b),
        MIRStatement::Lte(r,a,b)    => bin_i(f, "lte",  r, a, b),

        MIRStatement::BinNot(r,a)     => uno_i(f, "bnot", r, a),
        MIRStatement::LogNot(r,a)     => uno_i(f, "lnot", r, a),
        MIRStatement::AsAddr(r,a)     => uno_i(f, "tadr", r, a),
        MIRStatement::AsSigned(r,a)   => uno_i(f, "tsig", r, a),
        MIRStatement::AsUnsigned(r,a) => uno_i(f, "tuns", r, a),

        // heap-shaped ops
        MIRStatement::Cons(r,a,b) => bin_h(f, "cons", r, a, b),
        MIRStatement::Car(r,a)    => uno_h(f, "car",  r, a),
        MIRStatement::Cdr(r,a)    => uno_h(f, "cdr",  r, a),
        MIRStatement::Nullp(r,a)  => {
            mn!("null")?; fmt_imm(f, r)?; write!(f, ", ")?; fmt_heap(f, a)
        }

        MIRStatement::Array(r,a)  => uno_h(f, "arr",  r, a),
        MIRStatement::Full(r,a,b) => bin_h(f, "full", r, a, b),
        MIRStatement::Unpack(r,a) => uno_h(f, "unpk", r, a),
        MIRStatement::GetIdx(r,a,b) => {
            mn!("gidx")?; fmt_heap(f, r)?; write!(f, ", ")?;
            fmt_heap(f, a)?; write!(f, ", ")?; fmt_heap(f, b)
        }
        MIRStatement::PutIdx(r,t,i,v) => tri_h(f, "pidx", r, t, i, v),
        MIRStatement::ReadIdx(r,t,o,n) => tri_h(f, "ridx", r, t, o, n),
        MIRStatement::FillIdx(r,t,o,l) => tri_h(f, "fidx", r, t, o, l),
        MIRStatement::FullIdx(r,t,o,n,v) => qua_h(f, "Fidx", r, t, o, n, v),

        MIRStatement::Hits(r,a) => {
            mn!("hits")?; fmt_imm(f, r)?; write!(f, ", ")?; fmt_heap(f, a)
        }

        // syscalls — all imm
        MIRStatement::SysDsb(r)            => uno0_i(f, "@dsb",    r),
        MIRStatement::SysPrefetchFlush(r)  => uno0_i(f, "@pfflsh", r),
        MIRStatement::SysUartInit(r)       => uno0_i(f, "@uinit",  r),
        MIRStatement::SysUartGet8(r)       => uno0_i(f, "@uget8",  r),
        MIRStatement::SysClearMonitor(r)   => uno0_i(f, "@mclr",   r),
        MIRStatement::SysGetMonitor(r)     => uno0_i(f, "@mget",   r),
        MIRStatement::SysStopMonitor(r)    => uno0_i(f, "@mstp",   r),

        MIRStatement::SysGet32(r,a)        => uno_i(f, "@get32",  r, a),
        MIRStatement::SysUartPut8(r,a)     => uno_i(f, "@uput8",  r, a),
        MIRStatement::SysDelay(r,a)        => uno_i(f, "@delay",  r, a),
        MIRStatement::SysPut32(r,a,b)      => bin_i(f, "@put32",  r, a, b),
        MIRStatement::SysZero32(r,a,b,c)   => tri_i(f, "@zero32", r, a, b, c),
        MIRStatement::SysFull32(r,a,b,c,d) => qua_i(f, "@full32", r, a, b, c, d),

        // control flow
        MIRStatement::Br(b) => { mn!("b")?; fmt_blk(f, *b) }
        MIRStatement::CondBr { cond, then_blk, else_blk } => {
            mn!("cbr")?; fmt_imm(f, cond)?;
            write!(f, ", ")?; fmt_blk(f, *then_blk)?;
            write!(f, ", ")?; fmt_blk(f, *else_blk)
        }
        MIRStatement::PhiOpImm(r, (ab, av), (bb, bv)) => {
            mn!("phii")?; fmt_imm(f, r)?;
            write!(f, ", [")?; fmt_blk(f, *ab)?; write!(f, ": ")?; fmt_imm(f, av)?;
            write!(f, "], [")?; fmt_blk(f, *bb)?; write!(f, ": ")?; fmt_imm(f, bv)?;
            write!(f, "]")
        }
        MIRStatement::PhiOpHeap(r, (ab, av), (bb, bv)) => {
            mn!("phih")?; fmt_heap(f, r)?;
            write!(f, ", [")?; fmt_blk(f, *ab)?; write!(f, ": ")?; fmt_heap(f, av)?;
            write!(f, "], [")?; fmt_blk(f, *bb)?; write!(f, ": ")?; fmt_heap(f, bv)?;
            write!(f, "]")
        }
        MIRStatement::Ret(r) => { mn!("ret")?; fmt_heap(f, r) }

        MIRStatement::Escape(r, src) => uno_h(f, "esc", r, src),

        MIRStatement::Call { dst, callee, args } => {
            mn!("call")?;
            fmt_heap(f, dst)?;
            write!(f, ", [#{:p}]", callee.as_ref().as_ptr())?;
            for a in args {
                write!(f, ", ")?;
                fmt_heap(f, a)?;
            }
            Ok(())
        }
    }
}

// Imm-typed shape helpers.
fn uno0_i(f: &mut fmt::Formatter<'_>, m: &str, r: &ImmReg) -> fmt::Result {
    write!(f, "{:<7}", m)?; fmt_imm(f, r)
}
fn uno_i(f: &mut fmt::Formatter<'_>, m: &str, r: &ImmReg, a: &ImmReg) -> fmt::Result {
    write!(f, "{:<7}", m)?; fmt_imm(f, r)?; write!(f, ", ")?; fmt_imm(f, a)
}
fn bin_i(f: &mut fmt::Formatter<'_>, m: &str, r: &ImmReg, a: &ImmReg, b: &ImmReg) -> fmt::Result {
    write!(f, "{:<7}", m)?; fmt_imm(f, r)?; write!(f, ", ")?;
    fmt_imm(f, a)?; write!(f, ", ")?; fmt_imm(f, b)
}
fn tri_i(f: &mut fmt::Formatter<'_>, m: &str, r: &ImmReg, a: &ImmReg, b: &ImmReg, c: &ImmReg) -> fmt::Result {
    write!(f, "{:<7}", m)?; fmt_imm(f, r)?; write!(f, ", ")?;
    fmt_imm(f, a)?; write!(f, ", ")?; fmt_imm(f, b)?; write!(f, ", ")?; fmt_imm(f, c)
}
fn qua_i(f: &mut fmt::Formatter<'_>, m: &str, r: &ImmReg, a: &ImmReg, b: &ImmReg, c: &ImmReg, d: &ImmReg) -> fmt::Result {
    write!(f, "{:<7}", m)?; fmt_imm(f, r)?; write!(f, ", ")?;
    fmt_imm(f, a)?; write!(f, ", ")?; fmt_imm(f, b)?; write!(f, ", ")?;
    fmt_imm(f, c)?; write!(f, ", ")?; fmt_imm(f, d)
}

// Heap-typed shape helpers.
fn uno_h(f: &mut fmt::Formatter<'_>, m: &str, r: &HeapReg, a: &HeapReg) -> fmt::Result {
    write!(f, "{:<7}", m)?; fmt_heap(f, r)?; write!(f, ", ")?; fmt_heap(f, a)
}
fn bin_h(f: &mut fmt::Formatter<'_>, m: &str, r: &HeapReg, a: &HeapReg, b: &HeapReg) -> fmt::Result {
    write!(f, "{:<7}", m)?; fmt_heap(f, r)?; write!(f, ", ")?;
    fmt_heap(f, a)?; write!(f, ", ")?; fmt_heap(f, b)
}
fn tri_h(f: &mut fmt::Formatter<'_>, m: &str, r: &HeapReg, a: &HeapReg, b: &HeapReg, c: &HeapReg) -> fmt::Result {
    write!(f, "{:<7}", m)?; fmt_heap(f, r)?; write!(f, ", ")?;
    fmt_heap(f, a)?; write!(f, ", ")?; fmt_heap(f, b)?; write!(f, ", ")?; fmt_heap(f, c)
}
fn qua_h(f: &mut fmt::Formatter<'_>, m: &str, r: &HeapReg, a: &HeapReg, b: &HeapReg, c: &HeapReg, d: &HeapReg) -> fmt::Result {
    write!(f, "{:<7}", m)?; fmt_heap(f, r)?; write!(f, ", ")?;
    fmt_heap(f, a)?; write!(f, ", ")?; fmt_heap(f, b)?; write!(f, ", ")?;
    fmt_heap(f, c)?; write!(f, ", ")?; fmt_heap(f, d)
}

impl fmt::Debug for MIRSegment {
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
