//! LIR-level optimization over post-regalloc IR4.
//!
//! Unlike MIR, LIR no longer has SSA virtual registers: every operand is
//! a physical ARM register and registers are reassigned throughout the
//! block. This pass therefore uses a forward dataflow lattice over
//! physical-register contents and local bindings rather than the sparse
//! per-VReg SCCP map used by `optimize2`.
//!
//! The pass is intentionally conservative. It folds raw immediate ALU
//! instructions, folds compare/branch and compare/cset pairs, propagates
//! constants through the helper-call ABI where the result is semantic
//! and side-effect-free, tracks local constants through Bind/Store/Load,
//! and then removes dead pure instructions.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use super::ir3::ImmNumber;
use super::ir4::{Cond, Instr, Instruction, LIRSegment, Register};
use super::scope::LocalId;
use crate::language::ast::Value;
use crate::language::number::Number;

const PAYLOAD_OFFSET: i32 = 4;

#[derive(Clone, Debug, PartialEq)]
enum AbsValue {
    Unknown,
    Imm(ImmNumber),
    ValuePtr(Value),
    SlotConst(Value),
    LocalId(LocalId),
    StackPtr,
}

#[derive(Clone, Debug, PartialEq)]
struct State {
    regs: BTreeMap<Register, AbsValue>,
    locals: BTreeMap<LocalId, Value>,
}

impl State {
    fn new() -> Self {
        State {
            regs: BTreeMap::new(),
            locals: BTreeMap::new(),
        }
    }

    fn get(&self, r: Register) -> AbsValue {
        if r == Register::SP {
            return AbsValue::StackPtr;
        }
        self.regs.get(&r).cloned().unwrap_or(AbsValue::Unknown)
    }

    fn set(&mut self, r: Register, v: AbsValue) {
        if r == Register::SP {
            return;
        }
        match v {
            AbsValue::Unknown => {
                self.regs.remove(&r);
            }
            _ => {
                self.regs.insert(r, v);
            }
        }
    }

    fn clobber(&mut self, regs: &[Register]) {
        for r in regs {
            self.set(*r, AbsValue::Unknown);
        }
    }

    fn meet(&self, other: &State) -> State {
        let mut regs = BTreeMap::new();
        let keys: BTreeSet<Register> = self.regs.keys().chain(other.regs.keys()).copied().collect();
        for r in keys {
            let a = self.get(r);
            let b = other.get(r);
            if a == b && !matches!(a, AbsValue::Unknown) {
                regs.insert(r, a);
            }
        }

        let mut locals = BTreeMap::new();
        for (id, v) in &self.locals {
            if other.locals.get(id) == Some(v) {
                locals.insert(*id, v.clone());
            }
        }

        State { regs, locals }
    }
}

pub(crate) fn optimize4(mut seg: LIRSegment) -> LIRSegment {
    for _ in 0..4 {
        let entry = analyze_entries(&seg);
        let mut changed = fold(&mut seg, &entry);
        let entry = analyze_entries(&seg);
        changed |= virtualize_slots(&mut seg, &entry);
        changed |= dce(&mut seg);
        changed |= dead_blocks(&mut seg);
        recompute_callee_saves(&mut seg);
        if !changed {
            break;
        }
    }
    seg
}

fn analyze_entries(seg: &LIRSegment) -> Vec<Option<State>> {
    let n = seg.blocks.len();
    let mut entry: Vec<Option<State>> = vec![None; n];
    let mut worklist = vec![0usize];
    if n > 0 {
        entry[0] = Some(State::new());
    }

    while let Some(bi) = worklist.pop() {
        if bi >= n || seg.blocks[bi].dead {
            continue;
        }
        let mut state = match entry[bi].clone() {
            Some(s) => s,
            None => continue,
        };
        for instr in &seg.blocks[bi].instructions {
            transfer(instr, &mut state);
        }
        for succ in successors(&seg.blocks[bi].instructions) {
            if succ >= n || seg.blocks[succ].dead {
                continue;
            }
            let next = match &entry[succ] {
                Some(old) => old.meet(&state),
                None => state.clone(),
            };
            if entry[succ].as_ref() != Some(&next) {
                entry[succ] = Some(next);
                worklist.push(succ);
            }
        }
    }

    entry
}

fn fold(seg: &mut LIRSegment, entries: &[Option<State>]) -> bool {
    let mut changed = false;
    for (bi, block) in seg.blocks.iter_mut().enumerate() {
        if block.dead {
            continue;
        }
        let mut state = entries
            .get(bi)
            .and_then(|s| s.clone())
            .unwrap_or_else(State::new);
        let old = core::mem::take(&mut block.instructions);
        let mut out = Vec::with_capacity(old.len());
        let mut i = 0;
        while i < old.len() {
            if let Some((replacement, consumed)) = fold_window(&old, i, &state) {
                transfer(&replacement, &mut state);
                out.push(replacement);
                changed = true;
                i += consumed;
                continue;
            }

            let mut instr = old[i].clone();
            if let Some(replacement) = fold_instr(&instr, &state) {
                instr = replacement;
                changed = true;
            }

            if matches!(instr, Instr::Mov(d, s) if d == s) {
                changed = true;
                i += 1;
                continue;
            }

            transfer(&instr, &mut state);
            out.push(instr);
            i += 1;
        }
        block.instructions = out;
    }
    changed
}

#[derive(Clone, Debug)]
struct VirtualSlot {
    producer: usize,
    required: bool,
}

fn virtualize_slots(seg: &mut LIRSegment, entries: &[Option<State>]) -> bool {
    let mut changed = false;
    for (bi, block) in seg.blocks.iter_mut().enumerate() {
        if block.dead {
            continue;
        }

        let mut state = entries
            .get(bi)
            .and_then(|s| s.clone())
            .unwrap_or_else(State::new);
        let mut slots: Vec<VirtualSlot> = Vec::new();
        let mut in_reg: BTreeMap<Register, usize> = BTreeMap::new();

        for (i, instr) in block.instructions.iter().enumerate() {
            match instr {
                Instr::Mov(d, s) => {
                    match in_reg.get(s).copied() {
                        Some(tok) => {
                            in_reg.insert(*d, tok);
                        }
                        None => {
                            in_reg.remove(d);
                        }
                    }
                    transfer(instr, &mut state);
                    continue;
                }
                Instr::Box => {
                    let virtualized = value_from_imm_abs(&state.get(Register::R0)).is_some();
                    transfer(instr, &mut state);
                    clear_defs(&mut in_reg, instr);
                    if virtualized {
                        let tok = slots.len();
                        slots.push(VirtualSlot {
                            producer: i,
                            required: false,
                        });
                        in_reg.insert(Register::R0, tok);
                    } else {
                        in_reg.remove(&Register::R0);
                    }
                    continue;
                }
                Instr::LoadLocal => {
                    let virtualized = match state.get(Register::R0) {
                        AbsValue::LocalId(id) => state.locals.contains_key(&id),
                        _ => false,
                    };
                    transfer(instr, &mut state);
                    clear_defs(&mut in_reg, instr);
                    if virtualized {
                        let tok = slots.len();
                        slots.push(VirtualSlot {
                            producer: i,
                            required: false,
                        });
                        in_reg.insert(Register::R0, tok);
                    } else {
                        for r in defs(instr) {
                            in_reg.remove(&r);
                        }
                    }
                    continue;
                }
                Instr::LdrValuePtr(d, _) => {
                    transfer(instr, &mut state);
                    in_reg.remove(d);
                    let tok = slots.len();
                    slots.push(VirtualSlot {
                        producer: i,
                        required: false,
                    });
                    in_reg.insert(*d, tok);
                    continue;
                }
                _ => {}
            }

            for r in materialized_slot_uses(instr) {
                if let Some(tok) = in_reg.get(&r).copied() {
                    slots[tok].required = true;
                }
            }
            if materializes_virtual_region(instr) {
                for tok in in_reg.values().copied() {
                    slots[tok].required = true;
                }
            }

            transfer(instr, &mut state);

            for r in defs(instr) {
                in_reg.remove(&r);
            }

            if is_slot_boundary(instr) {
                in_reg.clear();
            }
        }

        let removable: BTreeSet<usize> = slots
            .iter()
            .filter(|s| !s.required)
            .map(|s| s.producer)
            .collect();
        if !removable.is_empty() {
            let old = core::mem::take(&mut block.instructions);
            block.instructions = old
                .into_iter()
                .enumerate()
                .filter_map(|(i, instr)| {
                    if removable.contains(&i) {
                        changed = true;
                        None
                    } else {
                        Some(instr)
                    }
                })
                .collect();
        }
    }
    changed
}

fn clear_defs(in_reg: &mut BTreeMap<Register, usize>, instr: &Instruction) {
    for r in defs(instr) {
        in_reg.remove(&r);
    }
}

fn materialized_slot_uses(instr: &Instruction) -> Vec<Register> {
    use Instr::*;
    match instr {
        LdrOffset(_, base, _) => vec![*base],
        StrCapture(_, src) | StoreSpill(_, src) | CmpImm(src, _) => vec![*src],
        StrOffset(src, base, _) => vec![*src, *base],
        BindLocal
        | StoreLocal
        | Cons
        | Array
        | Full
        | Unpack
        | GetIdx
        | PutIdx
        | ReadIdx
        | FillIdx
        | FullIdx
        | Hits
        | Escape
        | Call => vec![Register::R0, Register::R1, Register::R2, Register::R3],
        Ret => vec![Register::R0],
        // These consumers should already have folded when their input is
        // a virtual slot with a known value. If one remains, it needs a
        // real slot id.
        Truthy | LogNot | Xor | Nullp => vec![Register::R0, Register::R1],
        LoadLocal
        | UnboxLocal
        | PushFrame
        | PopFrame
        | Box
        | Div
        | Mod
        | UartInit
        | UartGet8
        | UartPut8
        | Delay
        | ClearMonitor
        | GetMonitor
        | StopMonitor
        | Zero32
        | StrMem
        | Full32
        | Dsb
        | PrefetchFlush
        | StackPush(_)
        | StackPop(_)
        | B(_)
        | Beq(_)
        | Bne(_)
        | CondBr { .. }
        | Phi(..)
        | Mov(..)
        | MovImm(..)
        | MovId(..)
        | LdrValuePtr(..)
        | LdrNamePtr(..)
        | LdrCapture(..)
        | LdrCapturePtr(..)
        | LdrCallCachePtr(..)
        | LoadSpill(..)
        | Add(..)
        | Sub(..)
        | Mul(..)
        | Lshift(..)
        | Rshift(..)
        | BinOr(..)
        | BinAnd(..)
        | Mvn(..)
        | Cmp(..)
        | Cset(..) => vec![],
    }
}

fn is_slot_boundary(instr: &Instruction) -> bool {
    matches!(
        instr,
        Instr::Escape
            | Instr::Call
            | Instr::Ret
            | Instr::B(_)
            | Instr::Beq(_)
            | Instr::Bne(_)
            | Instr::CondBr { .. }
    )
}

fn materializes_virtual_region(instr: &Instruction) -> bool {
    matches!(
        instr,
        Instr::BindLocal
            | Instr::StoreLocal
            | Instr::StrCapture(..)
            | Instr::Cons
            | Instr::Array
            | Instr::Full
            | Instr::Unpack
            | Instr::GetIdx
            | Instr::PutIdx
            | Instr::ReadIdx
            | Instr::FillIdx
            | Instr::FullIdx
            | Instr::Hits
            | Instr::Escape
            | Instr::Call
            | Instr::Ret
    )
}

fn fold_window(instrs: &[Instruction], i: usize, state: &State) -> Option<(Instruction, usize)> {
    match instrs.get(i)? {
        Instr::Cmp(a, b) => {
            let cset = instrs.get(i + 1)?;
            if let Instr::Cset(dst, cond) = cset {
                let result = eval_cmp(state.get(*a), state.get(*b), *cond)?;
                return Some((Instr::MovImm(*dst, bool_imm(result)), 2));
            }
        }
        Instr::CmpImm(r, imm) => {
            let branch = instrs.get(i + 1)?;
            let fallthrough = instrs.get(i + 2)?;
            let known = imm_eq(state.get(*r), *imm)?;
            match (branch, fallthrough) {
                (Instr::Beq(eq_target), Instr::B(ne_target)) => {
                    let target = if known { *eq_target } else { *ne_target };
                    return Some((Instr::B(target), 3));
                }
                (Instr::Bne(ne_target), Instr::B(eq_target)) => {
                    let target = if known { *eq_target } else { *ne_target };
                    return Some((Instr::B(target), 3));
                }
                _ => {}
            }
        }
        _ => {}
    }
    None
}

fn fold_instr(instr: &Instruction, state: &State) -> Option<Instruction> {
    use Instr::*;
    match instr {
        Mov(_, _) => None,
        Add(d, a, b) => {
            fold_arith(state.get(*a), state.get(*b), Number::add).map(|n| MovImm(*d, n))
        }
        Sub(d, a, b) => {
            fold_arith(state.get(*a), state.get(*b), Number::sub).map(|n| MovImm(*d, n))
        }
        Mul(d, a, b) => {
            fold_arith(state.get(*a), state.get(*b), Number::mul).map(|n| MovImm(*d, n))
        }
        Lshift(d, a, b) => {
            fold_arith(state.get(*a), state.get(*b), Number::lshift).map(|n| MovImm(*d, n))
        }
        Rshift(d, a, b) => {
            fold_arith(state.get(*a), state.get(*b), Number::rshift).map(|n| MovImm(*d, n))
        }
        BinOr(d, a, b) => fold_bit(state.get(*a), state.get(*b), |x, y| x | y, |x, y| x | y)
            .map(|n| MovImm(*d, n)),
        BinAnd(d, a, b) => fold_bit(state.get(*a), state.get(*b), |x, y| x & y, |x, y| x & y)
            .map(|n| MovImm(*d, n)),
        Mvn(d, a) => fold_not(state.get(*a)).map(|n| MovImm(*d, n)),
        LdrOffset(d, base, off) if *off == PAYLOAD_OFFSET => value_from_heapish(&state.get(*base))
            .and_then(|v| imm_from_value(&v))
            .map(|n| MovImm(*d, n)),
        Box => None,
        Truthy => value_from_heapish(&state.get(Register::R0))
            .map(|v| MovImm(Register::R0, bool_imm(!is_falsy(&v)))),
        LogNot => fold_lognot(state.get(Register::R0)).map(|n| MovImm(Register::R0, n)),
        Xor => match (
            truthy_known(state.get(Register::R0)),
            truthy_known(state.get(Register::R1)),
        ) {
            (Some(a), Some(b)) => Some(MovImm(Register::R0, bool_imm(a != b))),
            _ => None,
        },
        Div => fold_arith(
            state.get(Register::R0),
            state.get(Register::R1),
            Number::div,
        )
        .map(|n| MovImm(Register::R0, n)),
        Mod => fold_arith(
            state.get(Register::R0),
            state.get(Register::R1),
            Number::modulo,
        )
        .map(|n| MovImm(Register::R0, n)),
        Nullp => value_from_heapish(&state.get(Register::R0))
            .map(|v| MovImm(Register::R0, bool_imm(matches!(v, Value::Nil)))),
        UnboxLocal => match state.get(Register::R0) {
            AbsValue::LocalId(id) => state
                .locals
                .get(&id)
                .and_then(imm_from_value)
                .map(|n| MovImm(Register::R0, n)),
            _ => None,
        },
        _ => None,
    }
}

fn transfer(instr: &Instruction, state: &mut State) {
    use Instr::*;
    match instr {
        Mov(d, s) => state.set(*d, state.get(*s)),
        MovImm(d, n) => state.set(*d, AbsValue::Imm(*n)),
        MovId(d, id) => state.set(*d, AbsValue::LocalId(*id)),
        LdrValuePtr(d, v) => state.set(*d, AbsValue::ValuePtr(v.clone())),
        LdrNamePtr(d, _)
        | LdrCapture(d, _)
        | LdrCapturePtr(d, _)
        | LdrCallCachePtr(d, _)
        | LoadSpill(d, _) => state.set(*d, AbsValue::Unknown),
        LdrOffset(d, base, off) if *off == PAYLOAD_OFFSET => {
            let v = value_from_heapish(&state.get(*base))
                .and_then(|v| imm_from_value(&v))
                .map(AbsValue::Imm)
                .unwrap_or(AbsValue::Unknown);
            state.set(*d, v);
        }
        LdrOffset(d, _, _) => state.set(*d, AbsValue::Unknown),
        Add(d, a, b) => set_folded(
            state,
            *d,
            fold_arith(state.get(*a), state.get(*b), Number::add),
        ),
        Sub(d, a, b) => set_folded(
            state,
            *d,
            fold_arith(state.get(*a), state.get(*b), Number::sub),
        ),
        Mul(d, a, b) => set_folded(
            state,
            *d,
            fold_arith(state.get(*a), state.get(*b), Number::mul),
        ),
        Lshift(d, a, b) => set_folded(
            state,
            *d,
            fold_arith(state.get(*a), state.get(*b), Number::lshift),
        ),
        Rshift(d, a, b) => set_folded(
            state,
            *d,
            fold_arith(state.get(*a), state.get(*b), Number::rshift),
        ),
        BinOr(d, a, b) => set_folded(
            state,
            *d,
            fold_bit(state.get(*a), state.get(*b), |x, y| x | y, |x, y| x | y),
        ),
        BinAnd(d, a, b) => set_folded(
            state,
            *d,
            fold_bit(state.get(*a), state.get(*b), |x, y| x & y, |x, y| x & y),
        ),
        Mvn(d, a) => set_folded(state, *d, fold_not(state.get(*a))),
        Cset(d, _) => state.set(*d, AbsValue::Unknown),
        StrCapture(_, _) | StrOffset(_, _, _) | StoreSpill(_, _) | Cmp(_, _) | CmpImm(_, _) => {}

        BindLocal => {
            let id = match state.get(Register::R1) {
                AbsValue::LocalId(id) => Some(id),
                _ => None,
            };
            if let Some(id) = id {
                match value_from_heapish(&state.get(Register::R0)) {
                    Some(v) => {
                        state.locals.insert(id, v);
                    }
                    None => {
                        state.locals.remove(&id);
                    }
                }
            } else {
                state.locals.clear();
            }
            clobber_helper(state);
        }
        StoreLocal => {
            let id = match state.get(Register::R0) {
                AbsValue::LocalId(id) => Some(id),
                _ => None,
            };
            if let Some(id) = id {
                match value_from_heapish(&state.get(Register::R1)) {
                    Some(v) => {
                        state.locals.insert(id, v);
                    }
                    None => {
                        state.locals.remove(&id);
                    }
                }
            } else {
                state.locals.clear();
            }
            clobber_helper(state);
        }
        LoadLocal => {
            let result = match state.get(Register::R0) {
                AbsValue::LocalId(id) => state
                    .locals
                    .get(&id)
                    .cloned()
                    .map(AbsValue::SlotConst)
                    .unwrap_or(AbsValue::Unknown),
                _ => AbsValue::Unknown,
            };
            clobber_helper(state);
            state.set(Register::R0, result);
        }
        UnboxLocal => {
            let result = match state.get(Register::R0) {
                AbsValue::LocalId(id) => state
                    .locals
                    .get(&id)
                    .and_then(imm_from_value)
                    .map(AbsValue::Imm)
                    .unwrap_or(AbsValue::Unknown),
                _ => AbsValue::Unknown,
            };
            clobber_helper(state);
            state.set(Register::R0, result);
        }
        Box => {
            let result = value_from_imm_abs(&state.get(Register::R0))
                .map(AbsValue::SlotConst)
                .unwrap_or(AbsValue::Unknown);
            clobber_helper(state);
            state.set(Register::R0, result);
        }
        Truthy => {
            let result = truthy_known(state.get(Register::R0))
                .map(bool_imm)
                .map(AbsValue::Imm)
                .unwrap_or(AbsValue::Unknown);
            clobber_helper(state);
            state.set(Register::R0, result);
        }
        LogNot => {
            let result = fold_lognot(state.get(Register::R0))
                .map(AbsValue::Imm)
                .unwrap_or(AbsValue::Unknown);
            clobber_helper(state);
            state.set(Register::R0, result);
        }
        Xor => {
            let result = match (
                truthy_known(state.get(Register::R0)),
                truthy_known(state.get(Register::R1)),
            ) {
                (Some(a), Some(b)) => AbsValue::Imm(bool_imm(a != b)),
                _ => AbsValue::Unknown,
            };
            clobber_helper(state);
            state.set(Register::R0, result);
        }
        Div => {
            let result = fold_arith(
                state.get(Register::R0),
                state.get(Register::R1),
                Number::div,
            )
            .map(AbsValue::Imm)
            .unwrap_or(AbsValue::Unknown);
            clobber_helper(state);
            state.set(Register::R0, result);
        }
        Mod => {
            let result = fold_arith(
                state.get(Register::R0),
                state.get(Register::R1),
                Number::modulo,
            )
            .map(AbsValue::Imm)
            .unwrap_or(AbsValue::Unknown);
            clobber_helper(state);
            state.set(Register::R0, result);
        }
        Nullp => {
            let result = value_from_heapish(&state.get(Register::R0))
                .map(|v| AbsValue::Imm(bool_imm(matches!(v, Value::Nil))))
                .unwrap_or(AbsValue::Unknown);
            clobber_helper(state);
            state.set(Register::R0, result);
        }
        Escape | Call => {
            state.locals.clear();
            clobber_helper(state);
        }
        PushFrame | PopFrame | Cons | Array | Full | Unpack | GetIdx | PutIdx | ReadIdx
        | FillIdx | FullIdx | Hits | UartInit | UartGet8 | UartPut8 | Delay | ClearMonitor
        | GetMonitor | StopMonitor | Zero32 | StrMem | Full32 => {
            clobber_helper(state);
        }
        Dsb | PrefetchFlush | StackPush(_) | StackPop(_) | B(_) | Beq(_) | Bne(_) | Ret => {}
        CondBr { .. } | Phi(..) => {}
    }
}

fn clobber_helper(state: &mut State) {
    state.clobber(&[
        Register::R0,
        Register::R1,
        Register::R2,
        Register::R3,
        Register::R12,
    ]);
}

fn set_folded(state: &mut State, dst: Register, val: Option<ImmNumber>) {
    state.set(dst, val.map(AbsValue::Imm).unwrap_or(AbsValue::Unknown));
}

fn successors(instrs: &[Instruction]) -> Vec<usize> {
    let mut out = Vec::new();
    for instr in instrs {
        match instr {
            Instr::B(t) => {
                out.push(*t);
                break;
            }
            Instr::Beq(t) | Instr::Bne(t) => out.push(*t),
            Instr::Ret => break,
            _ => {}
        }
    }
    out
}

fn eval_cmp(a: AbsValue, b: AbsValue, cond: Cond) -> Option<bool> {
    let na = number_from_abs(&a)?;
    let nb = number_from_abs(&b)?;
    let ord = na.partial_cmp(&nb)?;
    Some(match cond {
        Cond::Eq => na == nb,
        Cond::Ne => na != nb,
        Cond::Lt => ord == Ordering::Less,
        Cond::Le => ord != Ordering::Greater,
        Cond::Gt => ord == Ordering::Greater,
        Cond::Ge => ord != Ordering::Less,
    })
}

fn imm_eq(a: AbsValue, b: ImmNumber) -> Option<bool> {
    Some(number_from_abs(&a)? == number_from_imm(b))
}

fn fold_arith(
    a: AbsValue,
    b: AbsValue,
    op: fn(Number, Number) -> Result<Number, &'static str>,
) -> Option<ImmNumber> {
    let na = number_from_abs(&a)?;
    let nb = number_from_abs(&b)?;
    op(na, nb).ok().map(imm_from_number)
}

fn fold_bit(
    a: AbsValue,
    b: AbsValue,
    oi: fn(i32, i32) -> i32,
    ou: fn(u32, u32) -> u32,
) -> Option<ImmNumber> {
    match (number_from_abs(&a)?, number_from_abs(&b)?) {
        (Number::Integer(x), Number::Integer(y)) => Some(ImmNumber::Integer(oi(x, y))),
        (Number::Unsigned(x), Number::Unsigned(y)) => Some(ImmNumber::Unsigned(ou(x, y))),
        (Number::Unsigned(x), Number::Integer(y)) => Some(ImmNumber::Unsigned(ou(x, y as u32))),
        (Number::Integer(x), Number::Unsigned(y)) => Some(ImmNumber::Unsigned(ou(x as u32, y))),
        _ => None,
    }
}

fn fold_not(a: AbsValue) -> Option<ImmNumber> {
    match number_from_abs(&a)? {
        Number::Integer(i) => Some(ImmNumber::Integer(!i)),
        Number::Unsigned(u) => Some(ImmNumber::Unsigned(!u)),
        Number::Addr(_) => None,
    }
}

fn fold_lognot(a: AbsValue) -> Option<ImmNumber> {
    match a {
        AbsValue::Imm(ImmNumber::Integer(0)) | AbsValue::Imm(ImmNumber::Unsigned(0)) => {
            Some(ImmNumber::Integer(1))
        }
        AbsValue::Imm(ImmNumber::Integer(_))
        | AbsValue::Imm(ImmNumber::Unsigned(_))
        | AbsValue::Imm(ImmNumber::Addr(_)) => Some(ImmNumber::Integer(0)),
        _ => None,
    }
}

fn truthy_known(a: AbsValue) -> Option<bool> {
    match a {
        AbsValue::Imm(n) => Some(number_from_imm(n) != Number::Integer(0)),
        AbsValue::ValuePtr(v) | AbsValue::SlotConst(v) => Some(!is_falsy(&v)),
        _ => None,
    }
}

fn value_from_heapish(a: &AbsValue) -> Option<Value> {
    match a {
        AbsValue::ValuePtr(v) | AbsValue::SlotConst(v) => Some(v.clone()),
        _ => None,
    }
}

fn value_from_imm_abs(a: &AbsValue) -> Option<Value> {
    match a {
        AbsValue::Imm(n) => Some(Value::Number(number_from_imm(*n))),
        _ => None,
    }
}

fn number_from_abs(a: &AbsValue) -> Option<Number> {
    match a {
        AbsValue::Imm(n) => Some(number_from_imm(*n)),
        AbsValue::ValuePtr(Value::Number(n)) | AbsValue::SlotConst(Value::Number(n)) => Some(*n),
        AbsValue::ValuePtr(Value::Bool(b)) | AbsValue::SlotConst(Value::Bool(b)) => {
            Some(Number::Integer(if *b { 1 } else { 0 }))
        }
        _ => None,
    }
}

fn number_from_imm(n: ImmNumber) -> Number {
    match n {
        ImmNumber::Integer(i) => Number::Integer(i),
        ImmNumber::Unsigned(u) => Number::Unsigned(u),
        ImmNumber::Addr(a) => Number::Addr(a),
    }
}

fn imm_from_number(n: Number) -> ImmNumber {
    match n {
        Number::Integer(i) => ImmNumber::Integer(i),
        Number::Unsigned(u) => ImmNumber::Unsigned(u),
        Number::Addr(a) => ImmNumber::Addr(a),
    }
}

fn imm_from_value(v: &Value) -> Option<ImmNumber> {
    match v {
        Value::Number(n) => Some(imm_from_number(*n)),
        Value::Bool(b) => Some(bool_imm(*b)),
        _ => None,
    }
}

fn bool_imm(b: bool) -> ImmNumber {
    ImmNumber::Integer(if b { 1 } else { 0 })
}

fn is_falsy(v: &Value) -> bool {
    matches!(v, Value::Nil | Value::Bool(false))
}

fn dce(seg: &mut LIRSegment) -> bool {
    let n = seg.blocks.len();
    let mut live_in: Vec<BTreeSet<Register>> = vec![BTreeSet::new(); n];
    let mut live_out: Vec<BTreeSet<Register>> = vec![BTreeSet::new(); n];

    loop {
        let mut changed = false;
        for bi in (0..n).rev() {
            if seg.blocks[bi].dead {
                continue;
            }
            let mut out = BTreeSet::new();
            for succ in successors(&seg.blocks[bi].instructions) {
                if succ < n && !seg.blocks[succ].dead {
                    out.extend(live_in[succ].iter().copied());
                }
            }

            let mut live = out.clone();
            for instr in seg.blocks[bi].instructions.iter().rev() {
                for d in defs(instr) {
                    live.remove(&d);
                }
                for u in uses(instr) {
                    live.insert(u);
                }
            }
            if live != live_in[bi] || out != live_out[bi] {
                live_in[bi] = live;
                live_out[bi] = out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut changed = false;
    for bi in 0..n {
        if seg.blocks[bi].dead {
            continue;
        }
        let old = core::mem::take(&mut seg.blocks[bi].instructions);
        let mut live = live_out[bi].clone();
        let mut kept_rev = Vec::with_capacity(old.len());
        for instr in old.into_iter().rev() {
            let ds = defs(&instr);
            if is_pure(&instr) && !ds.is_empty() && ds.iter().all(|d| !live.contains(d)) {
                changed = true;
                continue;
            }
            for d in &ds {
                live.remove(d);
            }
            for u in uses(&instr) {
                live.insert(u);
            }
            kept_rev.push(instr);
        }
        kept_rev.reverse();
        seg.blocks[bi].instructions = kept_rev;
    }
    changed
}

fn uses(i: &Instruction) -> Vec<Register> {
    use Instr::*;
    match i {
        Mov(_, s) => vec![*s],
        MovImm(_, _)
        | MovId(_, _)
        | LdrValuePtr(_, _)
        | LdrNamePtr(_, _)
        | LdrCapture(_, _)
        | LdrCapturePtr(_, _)
        | LdrCallCachePtr(_, _)
        | LoadSpill(_, _) => vec![],
        StrCapture(_, s) | StoreSpill(_, s) | CmpImm(s, _) => vec![*s],
        LdrOffset(_, base, _) => vec![*base],
        StrOffset(s, base, _) => vec![*s, *base],
        Add(_, a, b)
        | Sub(_, a, b)
        | Mul(_, a, b)
        | Lshift(_, a, b)
        | Rshift(_, a, b)
        | BinOr(_, a, b)
        | BinAnd(_, a, b)
        | Cmp(a, b) => vec![*a, *b],
        Mvn(_, a) => vec![*a],
        Cset(_, _) => vec![],
        BindLocal | LoadLocal | StoreLocal | UnboxLocal | PushFrame | PopFrame | Box | Truthy
        | LogNot | Xor | Div | Mod | Cons | Nullp | Array | Full | Unpack | GetIdx | PutIdx
        | ReadIdx | FillIdx | FullIdx | Hits | Escape | Call | UartInit | UartGet8 | UartPut8
        | Delay | ClearMonitor | GetMonitor | StopMonitor | Zero32 | StrMem | Full32 => {
            vec![Register::R0, Register::R1, Register::R2, Register::R3]
        }
        Dsb | PrefetchFlush | StackPush(_) | StackPop(_) | B(_) | Beq(_) | Bne(_) => vec![],
        Ret => vec![Register::R0],
        CondBr { cond, .. } => vec![*cond],
        Phi(_, _, _) => vec![],
    }
}

fn defs(i: &Instruction) -> Vec<Register> {
    use Instr::*;
    match i {
        Mov(d, _)
        | MovImm(d, _)
        | MovId(d, _)
        | LdrValuePtr(d, _)
        | LdrNamePtr(d, _)
        | LdrCapture(d, _)
        | LdrCapturePtr(d, _)
        | LdrCallCachePtr(d, _)
        | LdrOffset(d, _, _)
        | LoadSpill(d, _)
        | Add(d, _, _)
        | Sub(d, _, _)
        | Mul(d, _, _)
        | Lshift(d, _, _)
        | Rshift(d, _, _)
        | BinOr(d, _, _)
        | BinAnd(d, _, _)
        | Mvn(d, _)
        | Cset(d, _) => vec![*d],
        StrCapture(_, _) | StrOffset(_, _, _) | StoreSpill(_, _) | Cmp(_, _) | CmpImm(_, _) => {
            vec![]
        }
        BindLocal | LoadLocal | StoreLocal | UnboxLocal | PushFrame | PopFrame | Box | Truthy
        | LogNot | Xor | Div | Mod | Cons | Nullp | Array | Full | Unpack | GetIdx | PutIdx
        | ReadIdx | FillIdx | FullIdx | Hits | Escape | Call | UartInit | UartGet8 | UartPut8
        | Delay | ClearMonitor | GetMonitor | StopMonitor | Zero32 | StrMem | Full32 => {
            vec![
                Register::R0,
                Register::R1,
                Register::R2,
                Register::R3,
                Register::R12,
            ]
        }
        Dsb | PrefetchFlush | StackPush(_) | StackPop(_) | B(_) | Beq(_) | Bne(_) | Ret => vec![],
        CondBr { .. } => vec![],
        Phi(d, _, _) => vec![*d],
    }
}

fn is_pure(i: &Instruction) -> bool {
    use Instr::*;
    matches!(
        i,
        Mov(..)
            | MovImm(..)
            | MovId(..)
            | LdrValuePtr(..)
            | LdrNamePtr(..)
            | LdrCapturePtr(..)
            | LdrCallCachePtr(..)
            | LoadSpill(..)
            | Add(..)
            | Sub(..)
            | Mul(..)
            | Lshift(..)
            | Rshift(..)
            | BinOr(..)
            | BinAnd(..)
            | Mvn(..)
            | Cset(..)
    )
}

fn dead_blocks(seg: &mut LIRSegment) -> bool {
    let n = seg.blocks.len();
    if n == 0 {
        return false;
    }
    let mut seen = BTreeSet::new();
    let mut wl = vec![0usize];
    while let Some(bi) = wl.pop() {
        if bi >= n || !seen.insert(bi) {
            continue;
        }
        for succ in successors(&seg.blocks[bi].instructions) {
            wl.push(succ);
        }
    }

    let mut changed = false;
    for i in 0..n {
        let dead = !seen.contains(&i);
        if seg.blocks[i].dead != dead {
            seg.blocks[i].dead = dead;
            changed = true;
        }
    }
    changed
}

fn recompute_callee_saves(seg: &mut LIRSegment) {
    let mut used = 0u32;
    for block in &seg.blocks {
        if block.dead {
            continue;
        }
        for instr in &block.instructions {
            for r in defs(instr).into_iter().chain(uses(instr)) {
                if let Some(i) = r.pool_index() {
                    used |= 1u32 << i;
                }
            }
        }
    }
    seg.callee_saves_used = used;
}
