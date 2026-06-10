//! MIR-level optimization: a second SCCP pass that operates on the
//! typed `MIRSegment` plus a peephole that fuses `LoadLocal` + `Unbox`
//! pairs into the `UnboxLocal` opcode.
//!
//! Same shape as `optimize.rs`: lift `MIRSegment` into an enriched
//! representation, run SCCP / fold / DCE / dead-block tagging, then
//! strip back to a plain `MIRSegment`. **Public signature is
//! `MIRSegment -> MIRSegment`** — fully in-place from the caller's POV.
//!
//! # SCCP lattice key
//!
//! `ImmReg` and `HeapReg` now share a unified `u32` ID pool (no
//! collisions even between fresh shims), so a single lattice map
//! keyed by raw register ID is sufficient. Heap-typed values can hold
//! `Constant(Value::Cons(..))` or boxed numeric constants — the
//! lattice doesn't care about the wrapper.
//!
//! # Peephole
//!
//! After SCCP, walk each block forward maintaining a
//! `recent: BTreeMap<HeapReg, LocalId>` of "this heap reg is currently
//! a fresh `LoadLocal` snapshot of this LocalId." Invalidate entries
//! when:
//!   * `BindLocal` / `BindImmediate` / `StoreLocal` touches that
//!     LocalId — the snapshot diverges from `locals[id]`.
//!   * `Escape` runs — the interpreter could `set!` any in-scope
//!     local (and we don't try to be more precise here than the IR1
//!     pass already was).
//!
//! When we see `Unbox(i, h)` and `h` is in `recent`, rewrite to
//! `UnboxLocal(i, id)`. Don't kill the original `LoadLocal` — DCE
//! does that if no other consumer keeps the heap reg alive.
//!
//! # Correctness notes
//!
//! * SSA preserved: every MIR dst is written exactly once. The peephole
//!   only ever rewrites in-place (no fresh defs), so the SSA property
//!   survives.
//! * Per-block analysis is intentional. Crossing a block boundary
//!   would require dataflow on `recent`; the common case after IR1
//!   lowering is `ldl … ubox …` inside the same block.
//! * `PushFrame` / `PopFrame` do NOT invalidate `recent` — they touch
//!   the runtime image's *name-binding* stack, not `locals[id]`. A
//!   popped local's slot is still readable via UnboxLocal.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use super::ir2::{HeapReg, ImmReg, MIRBasicBlock, MIRSegment, MIRStatement};
use super::scope::LocalId;
use crate::language::ast::Value;
use crate::language::number::Number;

// ===================== enriched types =====================

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EnrichedMIRStatement {
    pub seg: MIRStatement,
    pub dead: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockState {
    Top,
    Bottom,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EnrichedMIRBasicBlock {
    pub statements: Vec<EnrichedMIRStatement>,
    pub state: BlockState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FlowWLEntry {
    from: Option<usize>,
    to: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct UseEntry {
    block: usize,
    stmt: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum SCCPState {
    Top,
    Constant(Value),
    Bottom,
}

impl SCCPState {
    fn meet(self, other: SCCPState) -> SCCPState {
        use SCCPState::*;
        match (self, other) {
            (Top, x) | (x, Top) => x,
            (Bottom, _) | (_, Bottom) => Bottom,
            (Constant(a), Constant(b)) => {
                if a == b {
                    Constant(a)
                } else {
                    Bottom
                }
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct EnrichedMIRSegment {
    pub next_reg: u32,
    pub blocks: Vec<EnrichedMIRBasicBlock>,

    /// Per-register lattice state (single map, since Imm/Heap IDs share).
    state: BTreeMap<u32, SCCPState>,
    /// Per-LocalId lattice state.
    lstate: BTreeMap<LocalId, SCCPState>,
    flow_wl: Vec<FlowWLEntry>,
    use_wl: Vec<UseEntry>,
}

// ===================== conversion =====================

impl From<MIRSegment> for EnrichedMIRSegment {
    fn from(seg: MIRSegment) -> Self {
        let blocks = seg
            .blocks
            .into_iter()
            .map(|b| EnrichedMIRBasicBlock {
                statements: b
                    .statements
                    .into_iter()
                    .map(|s| EnrichedMIRStatement {
                        seg: s,
                        dead: false,
                    })
                    .collect(),
                // Preserve dead-block tagging from IR1's optimize → ir2 lower
                // — those blocks should stay dead post-MIR-SCCP too.
                state: if b.dead {
                    BlockState::Bottom
                } else {
                    BlockState::Top
                },
            })
            .collect();
        EnrichedMIRSegment {
            next_reg: seg.next_reg,
            blocks,
            state: BTreeMap::new(),
            lstate: BTreeMap::new(),
            flow_wl: vec![FlowWLEntry { from: None, to: 0 }],
            use_wl: Vec::new(),
        }
    }
}

impl From<EnrichedMIRSegment> for MIRSegment {
    fn from(e: EnrichedMIRSegment) -> Self {
        MIRSegment {
            next_reg: e.next_reg,
            blocks: e
                .blocks
                .into_iter()
                .map(|b| MIRBasicBlock {
                    dead: matches!(b.state, BlockState::Bottom),
                    statements: b
                        .statements
                        .into_iter()
                        .filter(|s| !s.dead)
                        .map(|s| s.seg)
                        .collect(),
                })
                .collect(),
        }
    }
}

// ===================== public entry =====================

/// Full MIR-level optimization pipeline. Consumes a `MIRSegment` and
/// returns an optimized one.
pub(crate) fn optimize2(seg: MIRSegment) -> MIRSegment {
    let mut e: EnrichedMIRSegment = seg.into();
    e.sccp();
    e.fold();
    e.peephole();
    e.dce();
    e.dead_blocks();
    e.into()
}

// ===================== SCCP =====================

impl EnrichedMIRSegment {
    fn get_id(&self, id: u32) -> SCCPState {
        self.state.get(&id).cloned().unwrap_or(SCCPState::Top)
    }
    fn get_imm(&self, r: &ImmReg) -> SCCPState {
        self.get_id(r.0)
    }
    fn get_heap(&self, r: &HeapReg) -> SCCPState {
        self.get_id(r.0)
    }
    fn get_local(&self, id: &LocalId) -> SCCPState {
        self.lstate.get(id).cloned().unwrap_or(SCCPState::Top)
    }

    fn update(&mut self, id: u32, new: SCCPState, uses: &BTreeMap<u32, Vec<UseEntry>>) {
        let old = self.get_id(id);
        let merged = old.clone().meet(new);
        if merged != old {
            self.state.insert(id, merged);
            if let Some(u) = uses.get(&id) {
                self.use_wl.extend(u.iter().copied());
            }
        }
    }

    fn update_local(
        &mut self,
        id: LocalId,
        new: SCCPState,
        local_uses: &BTreeMap<LocalId, Vec<UseEntry>>,
    ) {
        let old = self.get_local(&id);
        let merged = old.clone().meet(new);
        if merged != old {
            self.lstate.insert(id, merged);
            if let Some(u) = local_uses.get(&id) {
                self.use_wl.extend(u.iter().copied());
            }
        }
    }

    fn push_flow(&mut self, exec_edges: &BTreeSet<FlowWLEntry>, e: FlowWLEntry) {
        if !exec_edges.contains(&e) {
            self.flow_wl.push(e);
        }
    }

    fn sccp(&mut self) {
        let uses = compute_uses(&self.blocks);
        let local_uses = compute_local_uses(&self.blocks);
        let reach_out = compute_reach(&self.blocks);
        let mut exec_edges: BTreeSet<FlowWLEntry> = BTreeSet::new();
        let mut exec_blocks: BTreeSet<usize> = BTreeSet::new();

        loop {
            if let Some(edge) = self.flow_wl.pop() {
                exec_edges.insert(edge);
                let first_time = exec_blocks.insert(edge.to);
                let n = self.blocks[edge.to].statements.len();
                for i in 0..n {
                    let stmt = self.blocks[edge.to].statements[i].seg.clone();
                    if first_time {
                        self.visit(edge.to, &stmt, &uses, &local_uses, &reach_out, &exec_edges);
                    } else if matches!(
                        stmt,
                        MIRStatement::PhiOpImm(..) | MIRStatement::PhiOpHeap(..)
                    ) {
                        self.visit(edge.to, &stmt, &uses, &local_uses, &reach_out, &exec_edges);
                    } else {
                        break;
                    }
                }
            } else if let Some(u) = self.use_wl.pop() {
                if !exec_blocks.contains(&u.block) {
                    continue;
                }
                let stmt = self.blocks[u.block].statements[u.stmt].seg.clone();
                self.visit(u.block, &stmt, &uses, &local_uses, &reach_out, &exec_edges);
            } else {
                return;
            }
        }
    }

    fn visit(
        &mut self,
        block: usize,
        s: &MIRStatement,
        uses: &BTreeMap<u32, Vec<UseEntry>>,
        local_uses: &BTreeMap<LocalId, Vec<UseEntry>>,
        reach_out: &[BTreeSet<LocalId>],
        exec_edges: &BTreeSet<FlowWLEntry>,
    ) {
        use MIRStatement::*;
        match s {
            Br(t) => self.push_flow(
                exec_edges,
                FlowWLEntry {
                    from: Some(block),
                    to: *t,
                },
            ),

            CondBr {
                cond,
                then_blk,
                else_blk,
            } => match self.get_imm(cond) {
                SCCPState::Constant(v) => {
                    let to = if is_falsy(&v) { *else_blk } else { *then_blk };
                    self.push_flow(
                        exec_edges,
                        FlowWLEntry {
                            from: Some(block),
                            to,
                        },
                    );
                }
                SCCPState::Bottom => {
                    self.push_flow(
                        exec_edges,
                        FlowWLEntry {
                            from: Some(block),
                            to: *then_blk,
                        },
                    );
                    self.push_flow(
                        exec_edges,
                        FlowWLEntry {
                            from: Some(block),
                            to: *else_blk,
                        },
                    );
                }
                SCCPState::Top => {}
            },

            Ret(_) => {}

            PhiOpImm(dst, (pa, va), (pb, vb)) => {
                let ra = exec_edges.contains(&FlowWLEntry {
                    from: Some(*pa),
                    to: block,
                });
                let rb = exec_edges.contains(&FlowWLEntry {
                    from: Some(*pb),
                    to: block,
                });
                let st = match (ra, rb) {
                    (true, true) => self.get_imm(va).meet(self.get_imm(vb)),
                    (true, false) => self.get_imm(va),
                    (false, true) => self.get_imm(vb),
                    (false, false) => SCCPState::Top,
                };
                self.update(dst.0, st, uses);
            }
            PhiOpHeap(dst, (pa, va), (pb, vb)) => {
                let ra = exec_edges.contains(&FlowWLEntry {
                    from: Some(*pa),
                    to: block,
                });
                let rb = exec_edges.contains(&FlowWLEntry {
                    from: Some(*pb),
                    to: block,
                });
                let st = match (ra, rb) {
                    (true, true) => self.get_heap(va).meet(self.get_heap(vb)),
                    (true, false) => self.get_heap(va),
                    (false, true) => self.get_heap(vb),
                    (false, false) => SCCPState::Top,
                };
                self.update(dst.0, st, uses);
            }

            // --- local writes ---
            BindLocal { id, src, .. } => {
                let st = self.get_heap(src);
                self.update_local(*id, st, local_uses);
            }
            BindImmediate { id, src, .. } => {
                let st = match src {
                    Value::Closure(_) | Value::Macro(_) | Value::JittedClosure(_) => {
                        SCCPState::Bottom
                    }
                    _ => SCCPState::Constant(src.clone()),
                };
                self.update_local(*id, st, local_uses);
            }
            StoreLocal(id, src) => {
                let st = self.get_heap(src);
                self.update_local(*id, st, local_uses);
            }

            // --- local reads ---
            LoadLocal(d, id) => {
                let st = self.get_local(id);
                self.update(d.0, st, uses);
            }
            UnboxLocal(d, id) => {
                let st = self.get_local(id);
                self.update(d.0, st, uses);
            }

            // --- escape: taint locals in scope at this point ---
            Escape(d, _) => {
                let to_taint: Vec<LocalId> = reach_out[block].iter().copied().collect();
                for id in to_taint {
                    self.update_local(id, SCCPState::Bottom, local_uses);
                }
                self.update(d.0, SCCPState::Bottom, uses);
            }

            // --- direct call: same conservative taint as escape (the
            // callee can recursively bounce back into the interpreter
            // and set! any captured local on this path).
            Call { dst, .. } => {
                let to_taint: Vec<LocalId> = reach_out[block].iter().copied().collect();
                for id in to_taint {
                    self.update_local(id, SCCPState::Bottom, local_uses);
                }
                self.update(dst.0, SCCPState::Bottom, uses);
            }

            _ => {
                if let Some((d, st)) = self.eval(s) {
                    self.update(d, st, uses);
                }
            }
        }
    }

    /// Compute a new lattice value for the destination of `s`. Returns
    /// `(dst_id, state)` for value-producing ops, `None` for ops with
    /// no dst or ops handled in `visit`.
    fn eval(&self, s: &MIRStatement) -> Option<(u32, SCCPState)> {
        use MIRStatement::*;
        use SCCPState::*;

        // Direct loads — propagate the literal.
        if let LoadValueImm(d, v) = s {
            return Some((d.0, Constant(v.clone())));
        }
        if let LoadValuePtr(d, v) = s {
            let st = match v {
                Value::Closure(_) | Value::Macro(_) | Value::JittedClosure(_) => Bottom,
                _ => Constant(v.clone()),
            };
            return Some((d.0, st));
        }

        // Cast opcodes — identity on the value lattice.
        // Note Box layout: Box(src_imm, dst_heap) — dst is the second arg.
        if let Unbox(d, h) = s {
            return Some((d.0, self.get_heap(h)));
        }
        if let Box(src, dst) = s {
            return Some((dst.0, self.get_imm(src)));
        }
        if let Truthy(d, h) = s {
            let st = match self.get_heap(h) {
                Constant(v) => Constant(Value::Bool(!is_falsy(&v))),
                Bottom => Bottom,
                Top => Top,
            };
            return Some((d.0, st));
        }

        // Numeric folding macros.
        macro_rules! fold_arith {
            ($d:expr, $a:expr, $b:expr, $m:ident) => {{
                let st = match (self.get_imm($a), self.get_imm($b)) {
                    (Constant(Value::Number(na)), Constant(Value::Number(nb))) => match na.$m(nb) {
                        Ok(n) => Constant(Value::Number(n)),
                        Err(_) => Bottom,
                    },
                    (Bottom, _) | (_, Bottom) => Bottom,
                    (Top, _) | (_, Top) => Top,
                    _ => Bottom,
                };
                Some(($d.0, st))
            }};
        }
        macro_rules! fold_cmp {
            ($d:expr, $a:expr, $b:expr, $op:tt) => {{
                let st = match (self.get_imm($a), self.get_imm($b)) {
                    (Constant(Value::Number(na)), Constant(Value::Number(nb))) => {
                        Constant(Value::Bool(na $op nb))
                    }
                    (Bottom, _) | (_, Bottom) => Bottom,
                    (Top, _) | (_, Top) => Top,
                    _ => Bottom,
                };
                Some(($d.0, st))
            }};
        }

        match s {
            // arithmetic
            Add(d, a, b) => fold_arith!(d, a, b, add),
            Sub(d, a, b) => fold_arith!(d, a, b, sub),
            Mul(d, a, b) => fold_arith!(d, a, b, mul),
            Div(d, a, b) => fold_arith!(d, a, b, div),
            Mod(d, a, b) => fold_arith!(d, a, b, modulo),
            Lshift(d, a, b) => fold_arith!(d, a, b, lshift),
            Rshift(d, a, b) => fold_arith!(d, a, b, rshift),

            // comparisons
            Eq(d, a, b) => fold_cmp!(d, a, b, ==),
            Gt(d, a, b) => fold_cmp!(d, a, b, >),
            Lt(d, a, b) => fold_cmp!(d, a, b, <),
            Gte(d, a, b) => fold_cmp!(d, a, b, >=),
            Lte(d, a, b) => fold_cmp!(d, a, b, <=),

            // logical / bitwise
            LogNot(d, a) => {
                let st = match self.get_imm(a) {
                    Constant(Value::Bool(b)) => Constant(Value::Bool(!b)),
                    Constant(Value::Number(Number::Integer(0))) => {
                        Constant(Value::Number(Number::Integer(1)))
                    }
                    Constant(Value::Number(Number::Integer(_))) => {
                        Constant(Value::Number(Number::Integer(0)))
                    }
                    Constant(Value::Number(Number::Unsigned(0))) => {
                        Constant(Value::Number(Number::Unsigned(1)))
                    }
                    Constant(Value::Number(Number::Unsigned(_))) => {
                        Constant(Value::Number(Number::Unsigned(0)))
                    }
                    Constant(_) => Bottom,
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.0, st))
            }
            Xor(d, a, b) => {
                let st = match (self.get_imm(a), self.get_imm(b)) {
                    (Constant(va), Constant(vb)) => {
                        Constant(Value::Bool(is_falsy(&va) != is_falsy(&vb)))
                    }
                    (Bottom, _) | (_, Bottom) => Bottom,
                    _ => Top,
                };
                Some((d.0, st))
            }
            BinNot(d, a) => {
                let st = match self.get_imm(a) {
                    Constant(Value::Number(Number::Integer(i))) => {
                        Constant(Value::Number(Number::Integer(!i)))
                    }
                    Constant(Value::Number(Number::Unsigned(u))) => {
                        Constant(Value::Number(Number::Unsigned(!u)))
                    }
                    Constant(_) => Bottom,
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.0, st))
            }
            BinOr(d, a, b) => Some((
                d.0,
                fold_bin(self.get_imm(a), self.get_imm(b), |x, y| x | y, |x, y| x | y),
            )),
            BinAnd(d, a, b) => Some((
                d.0,
                fold_bin(self.get_imm(a), self.get_imm(b), |x, y| x & y, |x, y| x & y),
            )),

            // cons/list on constants
            Cons(d, a, b) => {
                let st = match (self.get_heap(a), self.get_heap(b)) {
                    (Constant(va), Constant(vb)) => Constant(Value::cons(va, vb)),
                    (Bottom, _) | (_, Bottom) => Bottom,
                    _ => Top,
                };
                Some((d.0, st))
            }
            Car(d, a) => {
                let st = match self.get_heap(a) {
                    Constant(Value::Cons(car, _)) => Constant((*car).clone()),
                    Constant(_) => Constant(Value::Nil),
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.0, st))
            }
            Cdr(d, a) => {
                let st = match self.get_heap(a) {
                    Constant(Value::Cons(_, cdr)) => Constant((*cdr).clone()),
                    Constant(_) => Constant(Value::Nil),
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.0, st))
            }
            Nullp(d, a) => {
                let st = match self.get_heap(a) {
                    Constant(Value::Nil) => Constant(Value::Bool(true)),
                    Constant(_) => Constant(Value::Bool(false)),
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.0, st))
            }

            // coercion
            AsAddr(d, a) => {
                let st = match self.get_imm(a) {
                    Constant(Value::Number(n)) => match n.as_addr() {
                        Ok(nn) => Constant(Value::Number(nn)),
                        Err(_) => Bottom,
                    },
                    Constant(_) => Bottom,
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.0, st))
            }
            AsSigned(d, a) => {
                let st = match self.get_imm(a) {
                    Constant(Value::Number(n)) => match n.as_i32() {
                        Ok(i) => Constant(Value::Number(Number::Integer(i))),
                        Err(_) => Bottom,
                    },
                    Constant(_) => Bottom,
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.0, st))
            }
            AsUnsigned(d, a) => {
                let st = match self.get_imm(a) {
                    Constant(Value::Number(n)) => match n.as_u32() {
                        Ok(u) => Constant(Value::Number(Number::Unsigned(u))),
                        Err(_) => Bottom,
                    },
                    Constant(_) => Bottom,
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.0, st))
            }

            // Opaque value-producers — Bottom. Split into imm-dst and
            // heap-dst groups so the bound `d`'s type stays consistent
            // within each or-pattern.

            // heap-dst opaque
            LoadCapture(d, _) | Array(d, _) | Unpack(d, _) | GetIdx(d, _, _) => Some((d.0, Bottom)),
            Full(d, _, _) => Some((d.0, Bottom)),
            PutIdx(d, _, _, _) | ReadIdx(d, _, _, _) | FillIdx(d, _, _, _) => Some((d.0, Bottom)),
            FullIdx(d, _, _, _, _) => Some((d.0, Bottom)),

            // imm-dst opaque
            Hits(d, _) => Some((d.0, Bottom)),

            SysDsb(d) | SysPrefetchFlush(d) | SysUartInit(d) | SysUartGet8(d)
            | SysClearMonitor(d) | SysGetMonitor(d) | SysStopMonitor(d) => Some((d.0, Bottom)),

            SysGet32(d, _) | SysUartPut8(d, _) | SysDelay(d, _) => Some((d.0, Bottom)),
            SysPut32(d, _, _) => Some((d.0, Bottom)),
            SysZero32(d, _, _, _) => Some((d.0, Bottom)),
            SysStr(d, _, _, _) => Some((d.0, Bottom)),
            SysFull32(d, _, _, _, _) => Some((d.0, Bottom)),

            // Call result is opaque (Bottom).
            Call { dst, .. } => Some((dst.0, Bottom)),

            // handled in visit, or no dst
            LoadValueImm(_, _)
            | LoadValuePtr(_, _)
            | Unbox(_, _)
            | Box(_, _)
            | Truthy(_, _)
            | UnboxLocal(_, _)
            | LoadLocal(_, _)
            | PushFrame
            | PopFrame
            | BindLocal { .. }
            | BindImmediate { .. }
            | StoreLocal(_, _)
            | StoreCapture(_, _)
            | Br(_)
            | CondBr { .. }
            | Ret(_)
            | PhiOpImm(..)
            | PhiOpHeap(..)
            | Escape(_, _) => None,
        }
    }

    /// Rewrite passes:
    /// * `BindLocal` with constant src → `BindImmediate`.
    /// * `CondBr` with constant cond → `Br`.
    /// * Any value-producing stmt whose dst became `Constant` →
    ///   replace with the appropriate `LoadValue*` for its register kind.
    fn fold(&mut self) {
        for b in &mut self.blocks {
            for s in &mut b.statements {
                if let MIRStatement::BindLocal { name, id, src } = &s.seg {
                    if let SCCPState::Constant(v) =
                        self.state.get(&src.0).cloned().unwrap_or(SCCPState::Top)
                    {
                        s.seg = MIRStatement::BindImmediate {
                            name: name.clone(),
                            id: *id,
                            src: v,
                        };
                        continue;
                    }
                }
                if let MIRStatement::CondBr {
                    cond,
                    then_blk,
                    else_blk,
                } = &s.seg
                {
                    if let SCCPState::Constant(v) =
                        self.state.get(&cond.0).cloned().unwrap_or(SCCPState::Top)
                    {
                        let to = if is_falsy(&v) { *else_blk } else { *then_blk };
                        s.seg = MIRStatement::Br(to);
                        continue;
                    }
                }

                if let Some((id, is_imm)) = dst_reg(&s.seg) {
                    if let SCCPState::Constant(v) =
                        self.state.get(&id).cloned().unwrap_or(SCCPState::Top)
                    {
                        // Don't rewrite ops that already are the canonical
                        // LoadValue for their kind — avoids churn.
                        let already_canonical = match (&s.seg, is_imm) {
                            (MIRStatement::LoadValueImm(_, _), true) => true,
                            (MIRStatement::LoadValuePtr(_, _), false) => true,
                            _ => false,
                        };
                        if !already_canonical {
                            s.seg = if is_imm {
                                MIRStatement::LoadValueImm(ImmReg(id), v)
                            } else {
                                MIRStatement::LoadValuePtr(HeapReg(id), v)
                            };
                        }
                    }
                }
            }
        }
    }

    /// Walk each block forward and apply two fusions:
    ///
    /// 1. `LoadLocal(h, id)` then `Unbox(i, h)` → `UnboxLocal(i, id)`.
    ///    The original LoadLocal is left in place; DCE drops it if `h`
    ///    has no other consumer.
    ///
    /// 2. Redundant `UnboxLocal(i, id)` when an earlier `UnboxLocal(p, id)`
    ///    already produced the same value AND no intervening
    ///    `StoreLocal`/`BindLocal*`/`Escape` invalidated it. The second
    ///    statement's dst `i` becomes an alias for `p`; every operand
    ///    use of `i` gets rewritten to `p`, and DCE drops the now-dead
    ///    `UnboxLocal(i, id)`.
    ///
    /// Per-block only — crossing block boundaries would require dataflow.
    /// Same invalidation rules for both fusions: store-to-id, rebind-id,
    /// or any escape kills the in-flight snapshot for that local.
    fn peephole(&mut self) {
        // imm-reg dst alias: redundant_dst → canonical_dst.
        let mut alias: BTreeMap<u32, u32> = BTreeMap::new();

        for b in &mut self.blocks {
            // ldl-h-to-id: heap regs currently snapshotting locals[id].
            let mut recent: BTreeMap<HeapReg, LocalId> = BTreeMap::new();
            // uboxl-canonical: the first ImmReg seen for this id, until
            // invalidated. Subsequent uboxls of the same id alias to this.
            let mut current: BTreeMap<LocalId, ImmReg> = BTreeMap::new();

            for s in &mut b.statements {
                if s.dead {
                    continue;
                }
                match &mut s.seg {
                    MIRStatement::LoadLocal(h, id) => {
                        recent.insert(*h, *id);
                    }
                    MIRStatement::StoreLocal(id, _)
                    | MIRStatement::BindLocal { id, .. }
                    | MIRStatement::BindImmediate { id, .. } => {
                        let id = *id;
                        recent.retain(|_, v| *v != id);
                        current.remove(&id);
                    }
                    MIRStatement::Escape(..) => {
                        recent.clear();
                        current.clear();
                    }
                    MIRStatement::Unbox(i, h) => {
                        // Fusion 1: ldl + ubox → uboxl.
                        if let Some(&id) = recent.get(h) {
                            let dst = *i;
                            s.seg = MIRStatement::UnboxLocal(dst, id);
                            // The rewritten uboxl now participates in
                            // Fusion 2's bookkeeping.
                            if let Some(&prev) = current.get(&id) {
                                alias.insert(dst.0, prev.0);
                            } else {
                                current.insert(id, dst);
                            }
                        }
                    }
                    MIRStatement::UnboxLocal(i, id) => {
                        // Fusion 2: redundant uboxl elimination.
                        if let Some(&prev) = current.get(id) {
                            alias.insert(i.0, prev.0);
                        } else {
                            current.insert(*id, *i);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Substitute aliased imm operands across all statements.
        // Aliases only arise from `UnboxLocal` rewrites (imm-dst), so
        // the only operand kind that needs rewriting is `ImmReg`.
        // Dominance is automatic: the canonical def is in the same
        // block as (and lexically before) the aliased def, so it
        // dominates every use of the aliased def too.
        if !alias.is_empty() {
            for b in &mut self.blocks {
                for s in &mut b.statements {
                    substitute_imm_operands(&mut s.seg, &alias);
                }
            }
        }
    }

    fn dce(&mut self) {
        let mut uses = compute_uses(&self.blocks);
        let defs = compute_defs(&self.blocks);

        let mut wl: Vec<UseEntry> = Vec::new();
        for (bi, b) in self.blocks.iter().enumerate() {
            for si in 0..b.statements.len() {
                wl.push(UseEntry {
                    block: bi,
                    stmt: si,
                });
            }
        }

        while let Some(u) = wl.pop() {
            let s = &self.blocks[u.block].statements[u.stmt];
            if s.dead {
                continue;
            }
            if has_side_effect(&s.seg) {
                continue;
            }
            let dst = match dst_reg(&s.seg) {
                Some((d, _)) => d,
                None => continue,
            };
            if !uses.get(&dst).map_or(true, |v| v.is_empty()) {
                continue;
            }

            self.blocks[u.block].statements[u.stmt].dead = true;
            let operands = stmt_operands(&self.blocks[u.block].statements[u.stmt].seg);
            for r in operands {
                if let Some(v) = uses.get_mut(&r) {
                    v.retain(|&e| e != u);
                }
                if let Some(def) = defs.get(&r).copied() {
                    wl.push(def);
                }
            }
        }
    }

    fn dead_blocks(&mut self) {
        let n = self.blocks.len();
        let mut wl: Vec<usize> = vec![0];
        let mut seen: BTreeSet<usize> = BTreeSet::new();
        while let Some(b) = wl.pop() {
            if !seen.insert(b) {
                continue;
            }
            for s in &self.blocks[b].statements {
                if s.dead {
                    continue;
                }
                match &s.seg {
                    MIRStatement::Br(t) => wl.push(*t),
                    MIRStatement::CondBr {
                        then_blk, else_blk, ..
                    } => {
                        wl.push(*then_blk);
                        wl.push(*else_blk);
                    }
                    _ => {}
                }
            }
        }
        for i in 0..n {
            self.blocks[i].state = if seen.contains(&i) {
                BlockState::Top
            } else {
                BlockState::Bottom
            };
        }
    }
}

// ===================== helpers =====================

fn compute_defs(blocks: &[EnrichedMIRBasicBlock]) -> BTreeMap<u32, UseEntry> {
    let mut defs = BTreeMap::new();
    for (bi, b) in blocks.iter().enumerate() {
        for (si, s) in b.statements.iter().enumerate() {
            if let Some((d, _)) = dst_reg(&s.seg) {
                defs.insert(
                    d,
                    UseEntry {
                        block: bi,
                        stmt: si,
                    },
                );
            }
        }
    }
    defs
}

fn compute_uses(blocks: &[EnrichedMIRBasicBlock]) -> BTreeMap<u32, Vec<UseEntry>> {
    let mut uses: BTreeMap<u32, Vec<UseEntry>> = BTreeMap::new();
    for (bi, b) in blocks.iter().enumerate() {
        for (si, s) in b.statements.iter().enumerate() {
            let e = UseEntry {
                block: bi,
                stmt: si,
            };
            for r in stmt_operands(&s.seg) {
                uses.entry(r).or_default().push(e);
            }
        }
    }
    uses
}

fn compute_local_uses(blocks: &[EnrichedMIRBasicBlock]) -> BTreeMap<LocalId, Vec<UseEntry>> {
    let mut uses: BTreeMap<LocalId, Vec<UseEntry>> = BTreeMap::new();
    for (bi, b) in blocks.iter().enumerate() {
        for (si, s) in b.statements.iter().enumerate() {
            let e = UseEntry {
                block: bi,
                stmt: si,
            };
            match &s.seg {
                MIRStatement::LoadLocal(_, id) | MIRStatement::UnboxLocal(_, id) => {
                    uses.entry(*id).or_default().push(e);
                }
                _ => {}
            }
        }
    }
    uses
}

fn compute_reach(blocks: &[EnrichedMIRBasicBlock]) -> Vec<BTreeSet<LocalId>> {
    let n = blocks.len();
    let mut binds: Vec<BTreeSet<LocalId>> = vec![BTreeSet::new(); n];
    for (bi, b) in blocks.iter().enumerate() {
        for s in &b.statements {
            match &s.seg {
                MIRStatement::BindLocal { id, .. } | MIRStatement::BindImmediate { id, .. } => {
                    binds[bi].insert(*id);
                }
                _ => {}
            }
        }
    }

    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (bi, b) in blocks.iter().enumerate() {
        for s in &b.statements {
            match &s.seg {
                MIRStatement::Br(t) => preds[*t].push(bi),
                MIRStatement::CondBr {
                    then_blk, else_blk, ..
                } => {
                    preds[*then_blk].push(bi);
                    preds[*else_blk].push(bi);
                }
                _ => {}
            }
        }
    }

    let mut reach_out: Vec<BTreeSet<LocalId>> = vec![BTreeSet::new(); n];
    loop {
        let mut changed = false;
        for bi in 0..n {
            let mut new_set = binds[bi].clone();
            for &p in &preds[bi] {
                for &id in &reach_out[p] {
                    new_set.insert(id);
                }
            }
            if new_set != reach_out[bi] {
                reach_out[bi] = new_set;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    reach_out
}

/// Side-effect classification for DCE. Side-effect-free ops can be
/// dropped if their dst has no uses.
fn has_side_effect(s: &MIRStatement) -> bool {
    use MIRStatement::*;
    !matches!(
        s,
        LoadValueImm(..)
            | LoadValuePtr(..)
            | LoadLocal(..)
            | LoadCapture(..)
            | Unbox(..)
            | Box(..)
            | Truthy(..)
            | UnboxLocal(..)
            | Add(..)
            | Sub(..)
            | Mul(..)
            | Div(..)
            | Mod(..)
            | Lshift(..)
            | Rshift(..)
            | BinNot(..)
            | BinOr(..)
            | BinAnd(..)
            | LogNot(..)
            | Xor(..)
            | Eq(..)
            | Gt(..)
            | Lt(..)
            | Gte(..)
            | Lte(..)
            | AsAddr(..)
            | AsSigned(..)
            | AsUnsigned(..)
            | Cons(..)
            | Car(..)
            | Cdr(..)
            | Nullp(..)
            | PhiOpImm(..)
            | PhiOpHeap(..)
            | Hits(..)
    )
}

/// Returns (dst_id, is_imm) for value-producing ops.
fn dst_reg(s: &MIRStatement) -> Option<(u32, bool)> {
    use MIRStatement::*;
    match s {
        // imm dsts
        LoadValueImm(d, _)
        | Unbox(d, _)
        | Truthy(d, _)
        | UnboxLocal(d, _)
        | BinNot(d, _)
        | LogNot(d, _)
        | AsAddr(d, _)
        | AsSigned(d, _)
        | AsUnsigned(d, _)
        | Nullp(d, _)
        | Hits(d, _)
        | SysDsb(d)
        | SysPrefetchFlush(d)
        | SysUartInit(d)
        | SysUartGet8(d)
        | SysClearMonitor(d)
        | SysGetMonitor(d)
        | SysStopMonitor(d)
        | SysGet32(d, _)
        | SysUartPut8(d, _)
        | SysDelay(d, _) => Some((d.0, true)),

        Add(d, _, _)
        | Sub(d, _, _)
        | Mul(d, _, _)
        | Div(d, _, _)
        | Mod(d, _, _)
        | Lshift(d, _, _)
        | Rshift(d, _, _)
        | BinOr(d, _, _)
        | BinAnd(d, _, _)
        | Xor(d, _, _)
        | Eq(d, _, _)
        | Gt(d, _, _)
        | Lt(d, _, _)
        | Gte(d, _, _)
        | Lte(d, _, _)
        | SysPut32(d, _, _) => Some((d.0, true)),

        GetIdx(d, _, _) => Some((d.0, true)),

        SysZero32(d, _, _, _) | SysStr(d, _, _, _) | SysFull32(d, _, _, _, _) => Some((d.0, true)),

        PhiOpImm(d, _, _) => Some((d.0, true)),

        // heap dsts
        LoadValuePtr(d, _)
        | LoadLocal(d, _)
        | LoadCapture(d, _)
        | Car(d, _)
        | Cdr(d, _)
        | Array(d, _)
        | Unpack(d, _)
        | Escape(d, _) => Some((d.0, false)),

        // Box has reversed layout: Box(src_imm, dst_heap).
        Box(_, d) => Some((d.0, false)),

        Cons(d, _, _) | Full(d, _, _) => Some((d.0, false)),

        PutIdx(d, _, _, _) | ReadIdx(d, _, _, _) | FillIdx(d, _, _, _) => Some((d.0, false)),

        FullIdx(d, _, _, _, _) => Some((d.0, false)),

        PhiOpHeap(d, _, _) => Some((d.0, false)),

        // Call dst is heap.
        Call { dst, .. } => Some((dst.0, false)),

        // no dst
        PushFrame
        | PopFrame
        | BindLocal { .. }
        | BindImmediate { .. }
        | StoreLocal(_, _)
        | StoreCapture(_, _)
        | Br(_)
        | CondBr { .. }
        | Ret(_) => None,
    }
}

fn stmt_operands(s: &MIRStatement) -> Vec<u32> {
    use MIRStatement::*;
    match s {
        LoadValueImm(_, _)
        | LoadValuePtr(_, _)
        | LoadLocal(_, _)
        | LoadCapture(_, _)
        | UnboxLocal(_, _)
        | PushFrame
        | PopFrame
        | Br(_)
        | BindImmediate { .. }
        | SysDsb(_)
        | SysPrefetchFlush(_)
        | SysUartInit(_)
        | SysUartGet8(_)
        | SysClearMonitor(_)
        | SysGetMonitor(_)
        | SysStopMonitor(_) => vec![],

        StoreLocal(_, h)
        | StoreCapture(_, h)
        | Ret(h)
        | BindLocal { src: h, .. }
        | Car(_, h)
        | Cdr(_, h)
        | Nullp(_, h)
        | Array(_, h)
        | Unpack(_, h)
        | Hits(_, h)
        | Unbox(_, h)
        | Truthy(_, h)
        | Escape(_, h) => vec![h.0],

        BinNot(_, i)
        | LogNot(_, i)
        | AsAddr(_, i)
        | AsSigned(_, i)
        | AsUnsigned(_, i)
        | SysGet32(_, i)
        | SysUartPut8(_, i)
        | SysDelay(_, i) => vec![i.0],

        // Box reads its src ImmReg (first arg).
        Box(src, _) => vec![src.0],

        CondBr { cond, .. } => vec![cond.0],

        Add(_, a, b)
        | Sub(_, a, b)
        | Mul(_, a, b)
        | Div(_, a, b)
        | Mod(_, a, b)
        | Lshift(_, a, b)
        | Rshift(_, a, b)
        | BinOr(_, a, b)
        | BinAnd(_, a, b)
        | Xor(_, a, b)
        | Eq(_, a, b)
        | Gt(_, a, b)
        | Lt(_, a, b)
        | Gte(_, a, b)
        | Lte(_, a, b)
        | SysPut32(_, a, b) => vec![a.0, b.0],

        Cons(_, a, b) | Full(_, a, b) => vec![a.0, b.0],
        GetIdx(_, a, b) => vec![a.0, b.0],

        PutIdx(_, t, i, v) => vec![t.0, i.0, v.0],
        ReadIdx(_, t, o, n) => vec![t.0, o.0, n.0],
        FillIdx(_, t, o, l) => vec![t.0, o.0, l.0],
        SysZero32(_, a, b, c) => vec![a.0, b.0, c.0],
        SysStr(_, a, b, c) => vec![a.0, b.0, c.0],

        FullIdx(_, t, o, n, v) => vec![t.0, o.0, n.0, v.0],
        SysFull32(_, a, b, c, d) => vec![a.0, b.0, c.0, d.0],

        PhiOpImm(_, (_, a), (_, b)) => vec![a.0, b.0],
        PhiOpHeap(_, (_, a), (_, b)) => vec![a.0, b.0],

        Call { args, .. } => args.iter().map(|a| a.0).collect(),
    }
}

fn fold_bin(
    a: SCCPState,
    b: SCCPState,
    oi: fn(i32, i32) -> i32,
    ou: fn(u32, u32) -> u32,
) -> SCCPState {
    use SCCPState::*;
    match (a, b) {
        (Constant(Value::Number(na)), Constant(Value::Number(nb))) => match (na, nb) {
            (Number::Integer(x), Number::Integer(y)) => {
                Constant(Value::Number(Number::Integer(oi(x, y))))
            }
            (Number::Unsigned(x), Number::Unsigned(y)) => {
                Constant(Value::Number(Number::Unsigned(ou(x, y))))
            }
            (Number::Unsigned(x), Number::Integer(y)) => {
                Constant(Value::Number(Number::Unsigned(ou(x, y as u32))))
            }
            (Number::Integer(x), Number::Unsigned(y)) => {
                Constant(Value::Number(Number::Unsigned(ou(x as u32, y))))
            }
            _ => Bottom,
        },
        (Bottom, _) | (_, Bottom) => Bottom,
        (Top, _) | (_, Top) => Top,
        _ => Bottom,
    }
}

/// Rewrite every `ImmReg` operand `i` of `s` whose `i.0` appears as a
/// key in `alias` to its canonical replacement. Used by the redundant-
/// uboxl elimination — aliases are always imm→imm so we only touch
/// imm operands. `HeapReg` operands and `dst` registers are untouched.
fn substitute_imm_operands(s: &mut MIRStatement, alias: &BTreeMap<u32, u32>) {
    use MIRStatement::*;
    macro_rules! sub {
        ($x:expr) => {
            if let Some(&n) = alias.get(&$x.0) {
                $x.0 = n;
            }
        };
    }
    match s {
        // single imm operand
        Box(i, _) => {
            sub!(i);
        }
        BinNot(_, i)
        | LogNot(_, i)
        | AsAddr(_, i)
        | AsSigned(_, i)
        | AsUnsigned(_, i)
        | SysGet32(_, i)
        | SysUartPut8(_, i)
        | SysDelay(_, i) => {
            sub!(i);
        }

        // condbr cond
        CondBr { cond, .. } => {
            sub!(cond);
        }

        // two-operand arith / cmp / bitwise / logical
        Add(_, a, b)
        | Sub(_, a, b)
        | Mul(_, a, b)
        | Div(_, a, b)
        | Mod(_, a, b)
        | Lshift(_, a, b)
        | Rshift(_, a, b)
        | BinOr(_, a, b)
        | BinAnd(_, a, b)
        | Xor(_, a, b)
        | Eq(_, a, b)
        | Gt(_, a, b)
        | Lt(_, a, b)
        | Gte(_, a, b)
        | Lte(_, a, b)
        | SysPut32(_, a, b) => {
            sub!(a);
            sub!(b);
        }

        SysZero32(_, a, b, c) => {
            sub!(a);
            sub!(b);
            sub!(c);
        }
        SysStr(_, _, b, _) => {
            sub!(b);
        }
        SysFull32(_, a, b, c, d) => {
            sub!(a);
            sub!(b);
            sub!(c);
            sub!(d);
        }

        PhiOpImm(_, (_, a), (_, b)) => {
            sub!(a);
            sub!(b);
        }

        _ => {}
    }
}

fn is_falsy(v: &Value) -> bool {
    matches!(
        v,
        Value::Nil
            | Value::Bool(false)
            | Value::Number(Number::Integer(0))
            | Value::Number(Number::Unsigned(0))
    )
}
