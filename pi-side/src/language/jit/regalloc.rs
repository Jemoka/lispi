//! IR3 → LIR: register allocation via Chaitin-Briggs.
//!
//! Internally operates on `Pseudo = Instr<Operand>` — the same enum
//! the post-regalloc LIR uses, just parameterized over `Operand`
//! (virtual or pre-colored physical) instead of `Register`. The final
//! `apply_colors` pass walks the Pseudo and produces `Instr<Register>`.
//!
//! # Pipeline
//!
//!   1. **Sequencer** — lower each IR3 statement to a sequence of
//!      `Pseudo` instructions. Helper-call ops expand to
//!      `Mov(P(R0), V(arg))` + bare opcode + `Mov(V(dst), P(R0))`.
//!      Direct ARM ops pass through with `V(_)` operands.
//!   2. **Phi destruction** — for each phi, insert `Mov` copies at
//!      every predecessor's tail (before its terminator). Resolve
//!      parallel-move cycles with a fresh temp.
//!   3. **Regalloc loop**: liveness → interference → Briggs coalesce
//!      → simplify+select → spill rewrite if needed → restart.
//!   4. **Apply colors** — `map_operands` rewrites each `V(_)` to
//!      `P(_)` and identity `Mov`s are dropped. `CondBr` is expanded
//!      to `CmpImm + Beq + B`.
//!   5. **Mul fixup** — ARMv6 requires `rd ≠ rm` in `mul`; rewrite
//!      violators via R12 scratch.
//!
//! # Calling-convention modelling
//!
//! Helper opcodes are modeled with `uses = {R0..R3}` and `defs/clobbers
//! = {R0..R3, R12}`. Since the regalloc pool is `Register::POOL`
//! (r4–r11), no virtual register ever ends up in a caller-saved color
//! — call-clobber interference is automatically satisfied. Pre-colored
//! physical nodes (R0–R3, R12) sit in the interference graph so
//! Briggs's safe-coalesce check correctly rejects attempts to merge a
//! long-lived vreg into a clobbered register.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use alloc::vec;
use core::fmt;

use crate::language::ast::Value;
use super::ir3::{ImmNumber, RIRSegment, RIRStatement, VReg};
use super::ir4::{
    Cond, Instr, Instruction, LIRBasicBlock, LIRSegment, Register, SpillSlot,
};

// ===================== field offsets =====================
//
// Placeholder values for direct-ARM `ldr/str` lowerings. The actual
// memory layout of `Value::Number` / `Cons` / `Closure` cells is
// determined by Rust's enum representation; the asm-emit layer will
// fix these up to match.

const PAYLOAD_OFFSET: i32 = 4;
const CAR_OFFSET: i32 = 4;
const CDR_OFFSET: i32 = 8;
const HITS_OFFSET: i32 = 4;

// ===================== Operand =====================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Operand {
    /// Virtual register (matches `VReg(N)` from IR3, plus regalloc-
    /// minted ones for phi-cycle temps and spill reloads).
    V(u32),
    /// Pre-colored physical register.
    P(Register),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::V(n) => write!(f, "v{}", n),
            Operand::P(r) => write!(f, "{}", r),
        }
    }
}

type Pseudo = Instr<Operand>;

// ===================== Operand walkers (generic over R) =====================

/// Apply `f` to every R-typed operand inside `instr`, producing an
/// `Instr<S>`. All non-R fields are moved through unchanged.
fn map_operands<R, S, F>(instr: Instr<R>, mut f: F) -> Instr<S>
where F: FnMut(R) -> S {
    use Instr::*;
    match instr {
        Mov(a, b)              => Mov(f(a), f(b)),
        MovImm(d, n)           => MovImm(f(d), n),
        MovId(d, id)           => MovId(f(d), id),
        LdrValuePtr(d, v)      => LdrValuePtr(f(d), v),
        LdrNamePtr(d, n)       => LdrNamePtr(f(d), n),
        LdrCapture(d, b)       => LdrCapture(f(d), b),
        StrCapture(b, s)       => StrCapture(b, f(s)),
        LdrOffset(d, b, o)     => LdrOffset(f(d), f(b), o),
        StrOffset(s, b, o)     => StrOffset(f(s), f(b), o),
        LoadSpill(d, s)        => LoadSpill(f(d), s),
        StoreSpill(s, r)       => StoreSpill(s, f(r)),
        Add(d, a, b)           => Add(f(d), f(a), f(b)),
        Sub(d, a, b)           => Sub(f(d), f(a), f(b)),
        Mul(d, a, b)           => Mul(f(d), f(a), f(b)),
        Lshift(d, a, b)        => Lshift(f(d), f(a), f(b)),
        Rshift(d, a, b)        => Rshift(f(d), f(a), f(b)),
        BinOr(d, a, b)         => BinOr(f(d), f(a), f(b)),
        BinAnd(d, a, b)        => BinAnd(f(d), f(a), f(b)),
        Mvn(d, a)              => Mvn(f(d), f(a)),
        Cmp(a, b)              => Cmp(f(a), f(b)),
        CmpImm(a, n)           => CmpImm(f(a), n),
        Cset(d, c)             => Cset(f(d), c),
        BindLocal              => BindLocal,
        LoadLocal              => LoadLocal,
        StoreLocal             => StoreLocal,
        UnboxLocal             => UnboxLocal,
        PushFrame              => PushFrame,
        PopFrame               => PopFrame,
        Box                    => Box,
        Truthy                 => Truthy,
        LogNot                 => LogNot,
        Xor                    => Xor,
        Div                    => Div,
        Mod                    => Mod,
        Cons                   => Cons,
        Nullp                  => Nullp,
        Array                  => Array,
        Full                   => Full,
        Unpack                 => Unpack,
        GetIdx                 => GetIdx,
        PutIdx                 => PutIdx,
        ReadIdx                => ReadIdx,
        FillIdx                => FillIdx,
        FullIdx                => FullIdx,
        Hits                   => Hits,
        Escape                 => Escape,
        UartInit               => UartInit,
        UartGet8               => UartGet8,
        UartPut8               => UartPut8,
        Delay                  => Delay,
        ClearMonitor           => ClearMonitor,
        GetMonitor             => GetMonitor,
        StopMonitor            => StopMonitor,
        Zero32                 => Zero32,
        Full32                 => Full32,
        Dsb                    => Dsb,
        PrefetchFlush          => PrefetchFlush,
        StackPush(m)           => StackPush(m),
        StackPop(m)            => StackPop(m),
        B(t)                   => B(t),
        Beq(t)                 => Beq(t),
        Bne(t)                 => Bne(t),
        Ret                    => Ret,
        CondBr { cond, then_blk, else_blk } => CondBr { cond: f(cond), then_blk, else_blk },
        Phi(d, (pa, a), (pb, b)) => Phi(f(d), (pa, f(a)), (pb, f(b))),
    }
}

/// Mutate `instr` in place, replacing every occurrence of `from` with `to`.
fn rename<R: Eq + Copy>(instr: &mut Instr<R>, from: R, to: R) {
    let r = |o: &mut R| if *o == from { *o = to; };
    use Instr::*;
    match instr {
        Mov(a, b)              => { r(a); r(b); }
        MovImm(d, _) | MovId(d, _)
        | LdrValuePtr(d, _) | LdrNamePtr(d, _) | LdrCapture(d, _)
        | LoadSpill(d, _) | Mvn(d, _) | Cset(d, _) => r(d),
        StrCapture(_, s) | StoreSpill(_, s) | CmpImm(s, _) => r(s),
        LdrOffset(d, b, _) => { r(d); r(b); }
        StrOffset(s, b, _) => { r(s); r(b); }
        Add(d, a, b) | Sub(d, a, b) | Mul(d, a, b)
        | Lshift(d, a, b) | Rshift(d, a, b)
        | BinOr(d, a, b) | BinAnd(d, a, b) => { r(d); r(a); r(b); }
        Cmp(a, b) => { r(a); r(b); }
        CondBr { cond, .. } => r(cond),
        Phi(d, (_, a), (_, b)) => { r(d); r(a); r(b); }
        _ => {}
    }
}

// ===================== uses / defs (Pseudo-specific) =====================
//
// Helper-call calling-convention is baked in here: helper opcodes
// implicitly use r0..r3 and clobber r0..r3 + r12.

fn uses(p: &Pseudo) -> Vec<Operand> {
    use Instr::*;
    match p {
        Mov(_, s) => vec![*s],
        MovImm(_, _) | MovId(_, _)
        | LdrValuePtr(_, _) | LdrNamePtr(_, _) | LdrCapture(_, _) => vec![],
        StrCapture(_, s) => vec![*s],
        LdrOffset(_, base, _) => vec![*base],
        StrOffset(s, base, _) => vec![*s, *base],
        LoadSpill(_, _) => vec![],
        StoreSpill(_, s) => vec![*s],
        Add(_, a, b) | Sub(_, a, b) | Mul(_, a, b)
        | Lshift(_, a, b) | Rshift(_, a, b)
        | BinOr(_, a, b) | BinAnd(_, a, b) => vec![*a, *b],
        Mvn(_, a) => vec![*a],
        Cmp(a, b) => vec![*a, *b],
        CmpImm(a, _) => vec![*a],
        Cset(_, _) => vec![],

        // Helper opcodes — conservatively read all four arg regs. The
        // preceding `Mov P(Ri), V(arg)` instructions kill their V(arg)
        // operand at the read site; this just makes the in-r0..r3
        // values live until the call.
        BindLocal | LoadLocal | StoreLocal | UnboxLocal
        | PushFrame | PopFrame
        | Box | Truthy | LogNot | Xor | Div | Mod | Cons | Nullp
        | Array | Full | Unpack
        | GetIdx | PutIdx | ReadIdx | FillIdx | FullIdx
        | Hits | Escape
        | UartInit | UartGet8 | UartPut8 | Delay
        | ClearMonitor | GetMonitor | StopMonitor
        | Zero32 | Full32 => vec![
            Operand::P(Register::R0), Operand::P(Register::R1),
            Operand::P(Register::R2), Operand::P(Register::R3),
        ],

        Dsb | PrefetchFlush => vec![],
        StackPush(_) | StackPop(_) => vec![],
        B(_) | Beq(_) | Bne(_) => vec![],
        CondBr { cond, .. } => vec![*cond],
        Ret => vec![Operand::P(Register::R0)],
        Phi(_, _, _) => vec![],  // phi sources handled per-edge in liveness
    }
}

fn defs(p: &Pseudo) -> Vec<Operand> {
    use Instr::*;
    match p {
        Mov(d, _) | MovImm(d, _) | MovId(d, _)
        | LdrValuePtr(d, _) | LdrNamePtr(d, _) | LdrCapture(d, _)
        | LdrOffset(d, _, _) | LoadSpill(d, _)
        | Mvn(d, _) | Cset(d, _) => vec![*d],
        StrCapture(_, _) | StrOffset(_, _, _) | StoreSpill(_, _) => vec![],
        Add(d, _, _) | Sub(d, _, _) | Mul(d, _, _)
        | Lshift(d, _, _) | Rshift(d, _, _)
        | BinOr(d, _, _) | BinAnd(d, _, _) => vec![*d],
        Cmp(_, _) | CmpImm(_, _) => vec![],

        // Helper opcodes clobber the AAPCS caller-saved set.
        BindLocal | LoadLocal | StoreLocal | UnboxLocal
        | PushFrame | PopFrame
        | Box | Truthy | LogNot | Xor | Div | Mod | Cons | Nullp
        | Array | Full | Unpack
        | GetIdx | PutIdx | ReadIdx | FillIdx | FullIdx
        | Hits | Escape
        | UartInit | UartGet8 | UartPut8 | Delay
        | ClearMonitor | GetMonitor | StopMonitor
        | Zero32 | Full32 => vec![
            Operand::P(Register::R0), Operand::P(Register::R1),
            Operand::P(Register::R2), Operand::P(Register::R3),
            Operand::P(Register::R12),
        ],

        Dsb | PrefetchFlush => vec![],
        StackPush(_) | StackPop(_) => vec![],
        B(_) | Beq(_) | Bne(_) | Ret | CondBr { .. } => vec![],
        Phi(d, _, _) => vec![*d],
    }
}

fn is_terminator(p: &Pseudo) -> bool {
    matches!(p, Instr::B(_) | Instr::CondBr { .. } | Instr::Ret)
}

// ===================== sequencer =====================

fn lower_seg(seg: RIRSegment) -> (Vec<Vec<Pseudo>>, Vec<bool>, u32) {
    let mut max_vreg = 0u32;
    let mut blocks: Vec<Vec<Pseudo>> = Vec::with_capacity(seg.blocks.len());
    let mut dead: Vec<bool> = Vec::with_capacity(seg.blocks.len());

    for blk in seg.blocks {
        let mut out: Vec<Pseudo> = Vec::new();
        for stmt in &blk.statements {
            walk_vregs(stmt, &mut |v| if v.0 > max_vreg { max_vreg = v.0 });
            lower_stmt(stmt, &mut out);
        }
        blocks.push(out);
        dead.push(blk.dead);
    }
    (blocks, dead, max_vreg + 1)
}

fn walk_vregs(stmt: &RIRStatement, f: &mut impl FnMut(&VReg)) {
    use RIRStatement as R;
    match stmt {
        R::MovImm(d, _) | R::MovImmAddr(d, _) => f(d),
        R::PushFrame | R::PopFrame => {}
        R::BindLocal { src, .. } => f(src),
        R::BindImmediate { .. } => {}
        R::LoadLocal(d, _) | R::LoadCapture(d, _) | R::UnboxLocal(d, _) => f(d),
        R::StoreLocal(_, s) | R::StoreCapture(_, s) => f(s),
        R::Unbox(d, s) | R::Box(d, s) | R::Truthy(d, s) => { f(d); f(s); }
        R::Add(d, a, b) | R::Sub(d, a, b) | R::Mul(d, a, b)
        | R::Div(d, a, b) | R::Mod(d, a, b)
        | R::Lshift(d, a, b) | R::Rshift(d, a, b) => { f(d); f(a); f(b); }
        R::BinNot(d, a) => { f(d); f(a); }
        R::BinOr(d, a, b) | R::BinAnd(d, a, b) => { f(d); f(a); f(b); }
        R::LogNot(d, a) => { f(d); f(a); }
        R::Xor(d, a, b) => { f(d); f(a); f(b); }
        R::Eq(d, a, b) | R::Gt(d, a, b) | R::Lt(d, a, b)
        | R::Gte(d, a, b) | R::Lte(d, a, b) => { f(d); f(a); f(b); }
        R::AsAddr(d, a) | R::AsSigned(d, a) | R::AsUnsigned(d, a) => { f(d); f(a); }
        R::Cons(d, a, b) => { f(d); f(a); f(b); }
        R::Car(d, a) | R::Cdr(d, a) | R::Nullp(d, a) => { f(d); f(a); }
        R::Array(d, a) | R::Unpack(d, a) | R::Hits(d, a) => { f(d); f(a); }
        R::Full(d, a, b) | R::GetIdx(d, a, b) => { f(d); f(a); f(b); }
        R::PutIdx(d, t, i, v) => { f(d); f(t); f(i); f(v); }
        R::ReadIdx(d, t, o, n) | R::FillIdx(d, t, o, n) => { f(d); f(t); f(o); f(n); }
        R::FullIdx(d, t, o, n, v) => { f(d); f(t); f(o); f(n); f(v); }
        R::Br(_) => {}
        R::CondBr { cond, .. } => f(cond),
        R::PhiOp(d, (_, a), (_, b)) => { f(d); f(a); f(b); }
        R::Ret(s) => f(s),
        R::SysDsb(d) | R::SysPrefetchFlush(d)
        | R::SysUartInit(d) | R::SysUartGet8(d)
        | R::SysClearMonitor(d) | R::SysGetMonitor(d) | R::SysStopMonitor(d) => f(d),
        R::SysGet32(d, a) | R::SysUartPut8(d, a) | R::SysDelay(d, a) => { f(d); f(a); }
        R::SysPut32(d, a, b) => { f(d); f(a); f(b); }
        R::SysZero32(d, a, b, c) => { f(d); f(a); f(b); f(c); }
        R::SysFull32(d, a, b, c, e) => { f(d); f(a); f(b); f(c); f(e); }
        R::Escape(d, s) => { f(d); f(s); }
    }
}

fn v(x: &VReg) -> Operand { Operand::V(x.0) }
fn p(r: Register) -> Operand { Operand::P(r) }

fn lower_stmt(stmt: &RIRStatement, out: &mut Vec<Pseudo>) {
    use Instr as I;
    use RIRStatement as R;

    macro_rules! call {
        ($op:ident; args=[$($arg:expr),*]; dst=$dst:expr) => {{
            let arg_regs = [Register::R0, Register::R1, Register::R2, Register::R3];
            let mut i = 0;
            $(
                out.push(I::Mov(p(arg_regs[i]), v($arg)));
                i += 1;
            )*
            let _ = i;
            out.push(I::$op);
            out.push(I::Mov(v($dst), p(Register::R0)));
        }};
    }

    match stmt {
        R::MovImm(d, n)       => out.push(I::MovImm(v(d), *n)),
        R::MovImmAddr(d, val) => out.push(I::LdrValuePtr(v(d), val.clone())),

        R::PushFrame => out.push(I::PushFrame),
        R::PopFrame  => out.push(I::PopFrame),

        R::BindLocal { name, id, src } => {
            out.push(I::Mov(p(Register::R0), v(src)));
            out.push(I::MovId(p(Register::R1), *id));
            out.push(I::LdrNamePtr(p(Register::R2), name.clone()));
            out.push(I::BindLocal);
        }
        R::BindImmediate { name, id, src } => {
            // Lower as: r0=imm payload → bl box_number (r0 ← slot_id)
            //        → r1=local_id, r2=&name → bl bind_local.
            // The Box helper turns the raw immediate into a fresh
            // shadow slot so bind_local receives a slot id like
            // every other BindLocal site.
            out.push(I::MovImm(p(Register::R0), *src));
            out.push(I::Box);
            out.push(I::MovId(p(Register::R1), *id));
            out.push(I::LdrNamePtr(p(Register::R2), name.clone()));
            out.push(I::BindLocal);
        }

        R::LoadLocal(d, id) => {
            out.push(I::MovId(p(Register::R0), *id));
            out.push(I::LoadLocal);
            out.push(I::Mov(v(d), p(Register::R0)));
        }
        R::LoadCapture(d, b) => out.push(I::LdrCapture(v(d), b.clone())),
        R::StoreLocal(id, src) => {
            out.push(I::MovId(p(Register::R0), *id));
            out.push(I::Mov(p(Register::R1), v(src)));
            out.push(I::StoreLocal);
        }
        R::StoreCapture(b, src) => out.push(I::StrCapture(b.clone(), v(src))),

        R::Unbox(d, s)       => out.push(I::LdrOffset(v(d), v(s), PAYLOAD_OFFSET)),
        R::Box(src, dst)     => call!(Box; args=[src]; dst=dst),
        R::Truthy(d, s)      => call!(Truthy; args=[s]; dst=d),
        R::UnboxLocal(d, id) => {
            out.push(I::MovId(p(Register::R0), *id));
            out.push(I::UnboxLocal);
            out.push(I::Mov(v(d), p(Register::R0)));
        }

        R::Add(d, a, b)    => out.push(I::Add(v(d), v(a), v(b))),
        R::Sub(d, a, b)    => out.push(I::Sub(v(d), v(a), v(b))),
        R::Mul(d, a, b)    => out.push(I::Mul(v(d), v(a), v(b))),
        R::Lshift(d, a, b) => out.push(I::Lshift(v(d), v(a), v(b))),
        R::Rshift(d, a, b) => out.push(I::Rshift(v(d), v(a), v(b))),
        R::BinOr(d, a, b)  => out.push(I::BinOr(v(d), v(a), v(b))),
        R::BinAnd(d, a, b) => out.push(I::BinAnd(v(d), v(a), v(b))),
        R::BinNot(d, a)    => out.push(I::Mvn(v(d), v(a))),

        R::LogNot(d, a) => call!(LogNot; args=[a]; dst=d),
        R::Xor(d, a, b) => call!(Xor; args=[a, b]; dst=d),
        R::Div(d, a, b) => call!(Div; args=[a, b]; dst=d),
        R::Mod(d, a, b) => call!(Mod; args=[a, b]; dst=d),

        R::Eq(d, a, b)  => { out.push(I::Cmp(v(a), v(b))); out.push(I::Cset(v(d), Cond::Eq)); }
        R::Gt(d, a, b)  => { out.push(I::Cmp(v(a), v(b))); out.push(I::Cset(v(d), Cond::Gt)); }
        R::Lt(d, a, b)  => { out.push(I::Cmp(v(a), v(b))); out.push(I::Cset(v(d), Cond::Lt)); }
        R::Gte(d, a, b) => { out.push(I::Cmp(v(a), v(b))); out.push(I::Cset(v(d), Cond::Ge)); }
        R::Lte(d, a, b) => { out.push(I::Cmp(v(a), v(b))); out.push(I::Cset(v(d), Cond::Le)); }

        R::AsAddr(d, s) | R::AsSigned(d, s) | R::AsUnsigned(d, s)
            => out.push(I::Mov(v(d), v(s))),

        R::Cons(d, a, b) => call!(Cons; args=[a, b]; dst=d),
        R::Car(d, s)     => out.push(I::LdrOffset(v(d), v(s), CAR_OFFSET)),
        R::Cdr(d, s)     => out.push(I::LdrOffset(v(d), v(s), CDR_OFFSET)),
        R::Nullp(d, s)   => call!(Nullp; args=[s]; dst=d),

        R::Array(d, s)            => call!(Array; args=[s]; dst=d),
        R::Full(d, a, b)          => call!(Full; args=[a, b]; dst=d),
        R::Unpack(d, s)           => call!(Unpack; args=[s]; dst=d),
        R::GetIdx(d, a, b)        => call!(GetIdx; args=[a, b]; dst=d),
        R::PutIdx(d, t, i, val)   => call!(PutIdx; args=[t, i, val]; dst=d),
        R::ReadIdx(d, t, o, n)    => call!(ReadIdx; args=[t, o, n]; dst=d),
        R::FillIdx(d, t, o, l)    => call!(FillIdx; args=[t, o, l]; dst=d),
        R::FullIdx(d, t, o, n, val) => call!(FullIdx; args=[t, o, n, val]; dst=d),

        R::Hits(d, s) => out.push(I::LdrOffset(v(d), v(s), HITS_OFFSET)),

        R::Br(t) => out.push(I::B(*t)),
        R::CondBr { cond, then_blk, else_blk } => out.push(I::CondBr {
            cond: v(cond), then_blk: *then_blk, else_blk: *else_blk,
        }),
        R::PhiOp(d, (pa, va), (pb, vb)) => out.push(I::Phi(
            v(d), (*pa, v(va)), (*pb, v(vb)),
        )),
        R::Ret(s) => {
            out.push(I::Mov(p(Register::R0), v(s)));
            out.push(I::Ret);
        }

        R::SysDsb(d) => { out.push(I::Dsb); out.push(I::LdrValuePtr(v(d), Value::Nil)); }
        R::SysPrefetchFlush(d) => { out.push(I::PrefetchFlush); out.push(I::LdrValuePtr(v(d), Value::Nil)); }
        R::SysUartInit(d)     => { out.push(I::UartInit);     out.push(I::Mov(v(d), p(Register::R0))); }
        R::SysUartGet8(d)     => { out.push(I::UartGet8);     out.push(I::Mov(v(d), p(Register::R0))); }
        R::SysClearMonitor(d) => { out.push(I::ClearMonitor); out.push(I::Mov(v(d), p(Register::R0))); }
        R::SysGetMonitor(d)   => { out.push(I::GetMonitor);   out.push(I::Mov(v(d), p(Register::R0))); }
        R::SysStopMonitor(d)  => { out.push(I::StopMonitor);  out.push(I::Mov(v(d), p(Register::R0))); }

        R::SysGet32(d, addr)     => out.push(I::LdrOffset(v(d), v(addr), 0)),
        R::SysUartPut8(d, byte_) => {
            out.push(I::Mov(p(Register::R0), v(byte_)));
            out.push(I::UartPut8);
            out.push(I::LdrValuePtr(v(d), Value::Nil));
        }
        R::SysDelay(d, cnt) => {
            out.push(I::Mov(p(Register::R0), v(cnt)));
            out.push(I::Delay);
            out.push(I::LdrValuePtr(v(d), Value::Nil));
        }
        R::SysPut32(d, addr, val) => {
            out.push(I::StrOffset(v(val), v(addr), 0));
            out.push(I::LdrValuePtr(v(d), Value::Nil));
        }
        R::SysZero32(d, a, b, c) => {
            out.push(I::Mov(p(Register::R0), v(a)));
            out.push(I::Mov(p(Register::R1), v(b)));
            out.push(I::Mov(p(Register::R2), v(c)));
            out.push(I::Zero32);
            out.push(I::LdrValuePtr(v(d), Value::Nil));
        }
        R::SysFull32(d, a, b, c, e) => {
            out.push(I::Mov(p(Register::R0), v(a)));
            out.push(I::Mov(p(Register::R1), v(b)));
            out.push(I::Mov(p(Register::R2), v(c)));
            out.push(I::Mov(p(Register::R3), v(e)));
            out.push(I::Full32);
            out.push(I::LdrValuePtr(v(d), Value::Nil));
        }

        R::Escape(d, s) => call!(Escape; args=[s]; dst=d),
    }
}

// ===================== phi destruction =====================

fn phi_destruct(blocks: &mut [Vec<Pseudo>], dead: &[bool], next_vreg: &mut u32) {
    let mut edge_moves: BTreeMap<(usize, usize), Vec<(Operand, Operand)>> = BTreeMap::new();

    for (bi, blk) in blocks.iter().enumerate() {
        if dead[bi] { continue; }
        for stmt in blk {
            if let Instr::Phi(dst, (pa, va), (pb, vb)) = stmt {
                if !dead[*pa] {
                    edge_moves.entry((*pa, bi)).or_default().push((*dst, *va));
                }
                if !dead[*pb] {
                    edge_moves.entry((*pb, bi)).or_default().push((*dst, *vb));
                }
            }
        }
    }

    for ((pred, _succ), moves) in edge_moves {
        let resolved = resolve_parallel_moves(moves, next_vreg);
        let blk = &mut blocks[pred];
        let term_idx = blk.iter()
            .rposition(is_terminator)
            .expect("predecessor block has no terminator");
        for mv in resolved.into_iter().rev() {
            blk.insert(term_idx, mv);
        }
    }

    for blk in blocks.iter_mut() {
        blk.retain(|s| !matches!(s, Instr::Phi(..)));
    }
}

fn resolve_parallel_moves(
    moves: Vec<(Operand, Operand)>,
    next_vreg: &mut u32,
) -> Vec<Pseudo> {
    let mut moves: Vec<(Operand, Operand)> =
        moves.into_iter().filter(|(d, s)| d != s).collect();

    let mut out: Vec<Pseudo> = Vec::new();

    while !moves.is_empty() {
        let leaf = moves.iter().position(|(d, _)| {
            moves.iter().all(|(_, s)| s != d)
        });
        if let Some(idx) = leaf {
            let (d, s) = moves.swap_remove(idx);
            out.push(Instr::Mov(d, s));
        } else {
            // Cycle remains. Break with a fresh temp.
            let (_d0, s0) = moves[0];
            let tmp = Operand::V(*next_vreg);
            *next_vreg += 1;
            out.push(Instr::Mov(tmp, s0));
            for m in moves.iter_mut() {
                if m.1 == s0 { m.1 = tmp; }
            }
        }
    }
    out
}

// ===================== CFG / liveness =====================

fn successors(block: &[Pseudo]) -> Vec<usize> {
    if let Some(term) = block.last() {
        match term {
            Instr::B(t) => vec![*t],
            Instr::CondBr { then_blk, else_blk, .. } => vec![*then_blk, *else_blk],
            Instr::Beq(t) => vec![*t],  // shouldn't occur pre-apply, but tolerated
            Instr::Bne(t) => vec![*t],
            Instr::Ret => vec![],
            _ => vec![],
        }
    } else {
        vec![]
    }
}

fn liveness(
    blocks: &[Vec<Pseudo>],
    dead: &[bool],
) -> BTreeMap<(usize, usize), BTreeSet<Operand>> {
    let n = blocks.len();
    let mut block_in: Vec<BTreeSet<Operand>> = vec![BTreeSet::new(); n];
    let mut block_out: Vec<BTreeSet<Operand>> = vec![BTreeSet::new(); n];

    loop {
        let mut changed = false;
        for bi in 0..n {
            if dead[bi] { continue; }
            let mut out_set = BTreeSet::new();
            for s in successors(&blocks[bi]) {
                if s < n && !dead[s] {
                    for v in &block_in[s] { out_set.insert(*v); }
                }
            }
            let mut live = out_set.clone();
            for stmt in blocks[bi].iter().rev() {
                for d in defs(stmt) { live.remove(&d); }
                for u in uses(stmt) { live.insert(u); }
            }
            if live != block_in[bi] || out_set != block_out[bi] {
                changed = true;
                block_in[bi] = live;
                block_out[bi] = out_set;
            }
        }
        if !changed { break; }
    }

    let mut per_stmt: BTreeMap<(usize, usize), BTreeSet<Operand>> = BTreeMap::new();
    for bi in 0..n {
        if dead[bi] { continue; }
        let mut live = block_out[bi].clone();
        for (si, stmt) in blocks[bi].iter().enumerate().rev() {
            per_stmt.insert((bi, si + 1), live.clone());
            for d in defs(stmt) { live.remove(&d); }
            for u in uses(stmt) { live.insert(u); }
            per_stmt.insert((bi, si), live.clone());
        }
    }
    per_stmt
}

// ===================== interference graph =====================

struct InterferenceGraph {
    nodes: BTreeSet<Operand>,
    edges: BTreeMap<Operand, BTreeSet<Operand>>,
    moves: Vec<(Operand, Operand)>,
}

impl InterferenceGraph {
    fn new() -> Self {
        InterferenceGraph {
            nodes: BTreeSet::new(),
            edges: BTreeMap::new(),
            moves: Vec::new(),
        }
    }

    fn add_node(&mut self, n: Operand) {
        self.nodes.insert(n);
        self.edges.entry(n).or_default();
    }

    fn add_edge(&mut self, a: Operand, b: Operand) {
        if a == b { return; }
        self.add_node(a);
        self.add_node(b);
        self.edges.get_mut(&a).unwrap().insert(b);
        self.edges.get_mut(&b).unwrap().insert(a);
    }

    fn degree(&self, n: &Operand) -> usize {
        self.edges.get(n).map_or(0, |e| e.len())
    }

    fn neighbors(&self, n: &Operand) -> BTreeSet<Operand> {
        self.edges.get(n).cloned().unwrap_or_default()
    }
}

fn build_interference(
    blocks: &[Vec<Pseudo>],
    dead: &[bool],
    per_stmt: &BTreeMap<(usize, usize), BTreeSet<Operand>>,
) -> InterferenceGraph {
    let mut g = InterferenceGraph::new();

    for r in [Register::R0, Register::R1, Register::R2, Register::R3,
              Register::R4, Register::R5, Register::R6, Register::R7,
              Register::R8, Register::R9, Register::R10, Register::R11,
              Register::R12] {
        g.add_node(Operand::P(r));
    }

    for bi in 0..blocks.len() {
        if dead[bi] { continue; }
        for (si, stmt) in blocks[bi].iter().enumerate() {
            let live_out = per_stmt.get(&(bi, si + 1)).cloned().unwrap_or_default();
            let defs_set = defs(stmt);

            let move_src: Option<Operand> = if let Instr::Mov(_, s) = stmt {
                Some(*s)
            } else { None };

            for d in &defs_set {
                g.add_node(*d);
                for v_op in &live_out {
                    if v_op == d { continue; }
                    if Some(*v_op) == move_src { continue; }
                    g.add_edge(*d, *v_op);
                }
            }
            // Defs interfere with each other (helper-call clobber set).
            for i in 0..defs_set.len() {
                for j in (i+1)..defs_set.len() {
                    g.add_edge(defs_set[i], defs_set[j]);
                }
            }

            if let Instr::Mov(d, s) = stmt {
                if d != s { g.moves.push((*d, *s)); }
            }

            for u in uses(stmt) { g.add_node(u); }
            for d in &defs_set { g.add_node(*d); }
        }
    }

    g
}

// ===================== Briggs coalesce =====================

struct Coalescer<'a> {
    graph: InterferenceGraph,
    color: BTreeMap<Operand, Register>,
    blocks: &'a mut Vec<Vec<Pseudo>>,
}

impl<'a> Coalescer<'a> {
    fn new(graph: InterferenceGraph, blocks: &'a mut Vec<Vec<Pseudo>>) -> Self {
        let mut color = BTreeMap::new();
        for r in [Register::R0, Register::R1, Register::R2, Register::R3,
                  Register::R4, Register::R5, Register::R6, Register::R7,
                  Register::R8, Register::R9, Register::R10, Register::R11,
                  Register::R12] {
            color.insert(Operand::P(r), r);
        }
        Coalescer { graph, color, blocks }
    }

    fn briggs_safe(&self, a: Operand, b: Operand) -> bool {
        let neighbors: BTreeSet<Operand> = self.graph.neighbors(&a)
            .union(&self.graph.neighbors(&b))
            .copied()
            .filter(|n| *n != a && *n != b)
            .collect();
        let k = Register::POOL.len();
        let count = neighbors.iter()
            .filter(|n| self.graph.degree(n) >= k)
            .count();
        count < k
    }

    fn coalesce_pass(&mut self) -> bool {
        let moves: Vec<(Operand, Operand)> = self.graph.moves.clone();
        let mut any = false;
        for (a, b) in moves {
            if a == b { continue; }
            if !self.graph.nodes.contains(&a) || !self.graph.nodes.contains(&b) { continue; }
            if self.graph.edges.get(&a).map_or(false, |e| e.contains(&b)) { continue; }
            match (a, b) {
                (Operand::P(ra), Operand::P(rb)) if ra != rb => continue,
                _ => {}
            }
            let preferred_color = match (a, b) {
                (Operand::P(r), _) | (_, Operand::P(r)) => {
                    if !Register::POOL.contains(&r) { continue; }
                    Some(r)
                }
                _ => None,
            };
            if !self.briggs_safe(a, b) { continue; }

            // Merge: survivor = a, victim = b.
            self.merge(a, b, preferred_color);
            any = true;
        }
        any
    }

    fn merge(&mut self, survivor: Operand, victim: Operand, color: Option<Register>) {
        let v_neighbors = self.graph.neighbors(&victim);
        for n in v_neighbors {
            if n == survivor { continue; }
            self.graph.add_edge(survivor, n);
        }
        self.graph.edges.remove(&victim);
        for (_, edges) in self.graph.edges.iter_mut() {
            edges.remove(&victim);
        }
        self.graph.nodes.remove(&victim);
        if let Some(c) = color {
            self.color.insert(survivor, c);
        }
        for blk in self.blocks.iter_mut() {
            for stmt in blk.iter_mut() {
                rename(stmt, victim, survivor);
            }
        }
    }

    fn coalesce_all(&mut self) {
        while self.coalesce_pass() {}
        // Drop now-identity Movs.
        for blk in self.blocks.iter_mut() {
            blk.retain(|s| !matches!(s, Instr::Mov(d, sr) if d == sr));
        }
    }
}

// ===================== simplify + select =====================

fn simplify_select_spill(
    graph: &InterferenceGraph,
    color_in: &BTreeMap<Operand, Register>,
) -> (BTreeMap<Operand, Register>, BTreeSet<Operand>) {
    let k = Register::POOL.len();
    let mut color = color_in.clone();
    let mut stack: Vec<Operand> = Vec::new();
    let mut adj: BTreeMap<Operand, BTreeSet<Operand>> = graph.edges.clone();
    let mut removed: BTreeSet<Operand> = BTreeSet::new();

    fn remove(
        adj: &mut BTreeMap<Operand, BTreeSet<Operand>>,
        removed: &mut BTreeSet<Operand>,
        n: Operand,
    ) {
        if removed.contains(&n) { return; }
        let nbrs: Vec<Operand> = adj.get(&n).cloned().unwrap_or_default().into_iter().collect();
        for nb in nbrs {
            if let Some(s) = adj.get_mut(&nb) { s.remove(&n); }
        }
        adj.remove(&n);
        removed.insert(n);
    }

    loop {
        let pick = adj.iter()
            .filter(|(n, _)| !matches!(n, Operand::P(_)))
            .find(|(_, e)| e.len() < k)
            .map(|(n, _)| *n);
        if let Some(n) = pick {
            remove(&mut adj, &mut removed, n);
            stack.push(n);
            continue;
        }
        let spill = adj.iter()
            .filter(|(n, _)| !matches!(n, Operand::P(_)))
            .max_by_key(|(_, e)| e.len())
            .map(|(n, _)| *n);
        if let Some(n) = spill {
            remove(&mut adj, &mut removed, n);
            stack.push(n);
            continue;
        }
        break;
    }

    let mut actual_spill: BTreeSet<Operand> = BTreeSet::new();
    while let Some(n) = stack.pop() {
        let mut taken: BTreeSet<Register> = BTreeSet::new();
        for nb in graph.neighbors(&n) {
            if actual_spill.contains(&nb) { continue; }
            if let Some(c) = color.get(&nb) { taken.insert(*c); }
        }
        let chosen = Register::POOL.iter().find(|r| !taken.contains(r)).copied();
        match chosen {
            Some(c) => { color.insert(n, c); }
            None => { actual_spill.insert(n); }
        }
    }

    (color, actual_spill)
}

// ===================== spill rewrite =====================

fn spill_rewrite(
    blocks: &mut Vec<Vec<Pseudo>>,
    spill_set: &BTreeSet<Operand>,
    next_vreg: &mut u32,
    next_spill: &mut u32,
) {
    let mut slot_for: BTreeMap<Operand, SpillSlot> = BTreeMap::new();
    for s in spill_set {
        slot_for.insert(*s, SpillSlot(*next_spill));
        *next_spill += 1;
    }

    for blk in blocks.iter_mut() {
        let old = core::mem::take(blk);
        for mut stmt in old {
            let used: Vec<Operand> = uses(&stmt).into_iter()
                .filter(|o| spill_set.contains(o)).collect();
            let defd: Vec<Operand> = defs(&stmt).into_iter()
                .filter(|o| spill_set.contains(o)).collect();

            for u in &used {
                let fresh = Operand::V(*next_vreg);
                *next_vreg += 1;
                blk.push(Instr::LoadSpill(fresh, slot_for[u]));
                rename(&mut stmt, *u, fresh);
            }
            blk.push(stmt);
            for d in &defd {
                let fresh = Operand::V(*next_vreg);
                *next_vreg += 1;
                let last = blk.last_mut().unwrap();
                rename(last, *d, fresh);
                blk.push(Instr::StoreSpill(slot_for[d], fresh));
            }
        }
    }
}

// ===================== apply colors → LIR =====================

fn apply_colors(
    blocks: Vec<Vec<Pseudo>>,
    dead: Vec<bool>,
    color: BTreeMap<Operand, Register>,
    spill_slots: u32,
) -> LIRSegment {
    let resolve = |o: Operand| -> Register {
        match o {
            Operand::P(r) => r,
            Operand::V(_) => *color.get(&o)
                .unwrap_or_else(|| panic!("uncolored vreg in apply_colors: {:?}", o)),
        }
    };

    let mut out_blocks: Vec<LIRBasicBlock> = Vec::with_capacity(blocks.len());
    let mut callee_saves_used: u32 = 0;

    for (bi, blk) in blocks.into_iter().enumerate() {
        let mut out: Vec<Instruction> = Vec::new();
        for stmt in blk {
            // Expand CondBr into the 3-instruction asm form.
            if let Instr::CondBr { cond, then_blk, else_blk } = stmt {
                let c = resolve(cond);
                out.push(Instr::CmpImm(c, ImmNumber::Integer(0)));
                out.push(Instr::Beq(else_blk));
                out.push(Instr::B(then_blk));
                continue;
            }
            // Skip identity movs (should already be filtered, but
            // double-check after coloring).
            if let Instr::Mov(a, b) = stmt {
                let ra = resolve(a);
                let rb = resolve(b);
                if ra == rb { continue; }
            }
            let lowered: Instruction = map_operands(stmt, |op| resolve(op));
            out.push(lowered);
        }

        for instr in &out {
            // Track callee-saves used by walking destination registers.
            let mut check = |r: Register| {
                if let Some(i) = r.pool_index() {
                    callee_saves_used |= 1u32 << i;
                }
            };
            walk_dst_reg(instr, &mut check);
        }

        out_blocks.push(LIRBasicBlock { instructions: out, dead: dead[bi] });
    }

    LIRSegment {
        blocks: out_blocks,
        spill_slots,
        callee_saves_used,
    }
}

fn walk_dst_reg<F: FnMut(Register)>(i: &Instruction, f: &mut F) {
    use Instr::*;
    match i {
        Mov(d, _) | MovImm(d, _) | MovId(d, _)
        | LdrValuePtr(d, _) | LdrNamePtr(d, _) | LdrCapture(d, _)
        | LdrOffset(d, _, _) | LoadSpill(d, _)
        | Add(d, _, _) | Sub(d, _, _) | Mul(d, _, _)
        | Lshift(d, _, _) | Rshift(d, _, _)
        | BinOr(d, _, _) | BinAnd(d, _, _)
        | Mvn(d, _) | Cset(d, _) => f(*d),
        _ => {}
    }
}

// ===================== mul fixup =====================

fn mul_fixup(seg: &mut LIRSegment) {
    for blk in seg.blocks.iter_mut() {
        let old = core::mem::take(&mut blk.instructions);
        for instr in old {
            if let Instr::Mul(rd, rm, rs) = instr {
                if rd == rm {
                    blk.instructions.push(Instr::Mov(Register::R12, rm));
                    blk.instructions.push(Instr::Mul(rd, Register::R12, rs));
                    continue;
                }
            }
            blk.instructions.push(instr);
        }
    }
}

// ===================== public entry =====================

pub(crate) fn regalloc(seg: RIRSegment) -> LIRSegment {
    let (mut blocks, dead, mut next_vreg) = lower_seg(seg);

    phi_destruct(&mut blocks, &dead, &mut next_vreg);

    let mut next_spill: u32 = 0;

    loop {
        let per_stmt = liveness(&blocks, &dead);
        let graph = build_interference(&blocks, &dead, &per_stmt);

        let mut coalescer = Coalescer::new(graph, &mut blocks);
        coalescer.coalesce_all();
        let post_color_seed = coalescer.color.clone();

        let per_stmt = liveness(&blocks, &dead);
        let graph = build_interference(&blocks, &dead, &per_stmt);

        let (color, spill_set) = simplify_select_spill(&graph, &post_color_seed);

        if spill_set.is_empty() {
            let mut seg = apply_colors(blocks, dead, color, next_spill);
            mul_fixup(&mut seg);
            return seg;
        }

        spill_rewrite(&mut blocks, &spill_set, &mut next_vreg, &mut next_spill);
    }
}
