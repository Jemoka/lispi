//! # Lowering contract (read this before writing asm-emit)
//!
//! Two things the lowering layer MUST handle, both stemming from "we
//! preserve block IDs for phi-predecessor stability instead of compacting":
//!
//! * **Dead blocks may still contain statements.** `From<EnrichedIRSegment>`
//!   filters per-statement `dead` flags, but does NOT empty out blocks
//!   whose `block.dead == true`. Some side-effecting opcode in a dead
//!   block could survive DCE (e.g. its VReg's only use was a phi in a
//!   live block, keeping the def syntactically alive). The pretty-
//!   printer renders `.LN (Dead):` and skips the body. **Lowering must
//!   do the same** — iterate `blocks.iter().filter(|b| !b.dead)` (or
//!   the index-aware equivalent). Code-space waste, not a cycle-time
//!   bug, but a real correctness bug if you emit asm for a dead block.
//!
//! * **Phi predecessors may name dead blocks.** When a `CondBr` folds
//!   to `Br`, the not-taken arm becomes a dead block, but the merge
//!   block's `PhiOp(dst, (dead_blk, vreg_in_dead), (live_blk, vreg_in_live))`
//!   keeps the syntactic reference. At runtime that edge is never
//!   taken. Lowering should treat phi sources whose predecessor block
//!   has `dead == true` as "this edge doesn't exist" — don't emit a
//!   case for it in whatever the phi lowers to.
//!
//! Today this saves us from invalidating block IDs across passes. If
//! that constraint ever relaxes (e.g. a renumber pass), revisit.
//!
//! # SCCP (Sparse Conditional Constant Propagation) over the JIT IR.
//!
//! Lifts an `IRSegment` into an `EnrichedIRSegment`, runs SCCP to
//! determine each `VReg`'s lattice state, then rewrites any statement
//! whose destination became a `Constant` into a plain `Load`.
//! **No DCE** — unreachable blocks and now-dead stores stay in place.
//!
//! Folding rules (kept deliberately narrow):
//! * Only arithmetic / comparison on `Value::Number` operands is
//!   *evaluated* — uses the existing `Number` methods + `PartialOrd`.
//! * Other operations don't fold; their destinations become `Bottom`.
//! * `Value::Closure` / `Value::Macro` are always `Bottom` — env
//!   snapshots are runtime-defined and we treat them as opaque.
//! * Escapes are `Bottom` (interpreter-side).
//! * Captures are `Bottom` — the binding's runtime value is unknown at
//!   compile time and a flow-insensitive `cstate` is unsound (a later
//!   `StoreCapture` would leak its constant backward to earlier
//!   `LoadCapture`s).
//! * **Locals** track a flow-insensitive `lstate: BTreeMap<LocalId, …>`.
//!   Sound because `BindLocal*` is the unique entry point for a LocalId
//!   — no `LoadLocal(id)` is reachable before its `BindLocal*(id)`, so
//!   the "leak backward" failure mode doesn't apply. Writes are
//!   meet-merged across all reachable `Bind*`/`StoreLocal` sites.
//!
//! # Escape and the in-scope set
//!
//! An `Escape` calls the interpreter, which can `set!` any LocalId
//! whose binding is currently in the runtime Image's frame stack —
//! i.e., any LocalId whose `BindLocal*` has already executed on the
//! path leading to the Escape. We approximate this with a forward
//! dataflow `reach_out[block]` (over the syntactic CFG): for each
//! block B, the set of LocalIds whose `BindLocal*` appears on some
//! path from entry through B. At every `Escape` in block B, we Bottom
//! every LocalId in `reach_out[B]`. Slightly over-tainting (a bind
//! later in B contributes even if it follows the escape), but sound
//! and far simpler than full dominator analysis.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use crate::language::ast::Value;
use crate::language::number::Number;
use super::ir::{IRBasicBlock, IRSegment, IRStatement, VReg};
use super::scope::LocalId;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EnrichedIRStatement {
    pub seg: IRStatement,
    /// Set true by the DCE pass when this statement's destination is
    /// dead and the statement itself has no side effects. Marked
    /// statements are dropped at IRSegment-conversion time.
    pub dead: bool,
}

/// Reachability tag for an enriched block.
///   `Bottom` — initial; not yet proved reachable.
///   `Top`    — proved reachable by tracing from `.L0` through `Br` /
///              `CondBr` terminators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockState { Top, Bottom }

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EnrichedIRBasicBlock {
    pub statements: Vec<EnrichedIRStatement>,
    pub state: BlockState,
}

/// CFG worklist entry: an edge to (re-)process. `from = None` is the
/// synthetic entry edge into block 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FlowWLEntry { from: Option<usize>, to: usize }

/// SSA-use worklist entry: a specific (block, stmt) to re-evaluate
/// because one of its operands' lattice values changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct UseEntry { block: usize, stmt: usize }

/// Lattice for SSA values:
///   `Top`      — never observed (initial; absence from map == Top).
///   `Constant` — provably has exactly this Value.
///   `Bottom`   — varies at runtime / opaque.
#[derive(Clone, Debug, PartialEq)]
enum SCCPState {
    Top,
    Constant(Value),
    Bottom,
}

impl SCCPState {
    /// Lattice meet (greatest lower bound).
    /// `(Top, x) → x`, `(Bottom, _) → Bottom`,
    /// `(C(a), C(b)) → C(a) if a==b else Bottom`.
    fn meet(self, other: SCCPState) -> SCCPState {
        use SCCPState::*;
        match (self, other) {
            (Top, x) | (x, Top) => x,
            (Bottom, _) | (_, Bottom) => Bottom,
            (Constant(a), Constant(b)) => if a == b { Constant(a) } else { Bottom },
        }
    }
}

#[derive(Clone)]
pub(crate) struct EnrichedIRSegment {
    pub vregs: VReg,
    pub blocks: Vec<EnrichedIRBasicBlock>,

    /// Per-VReg lattice state (SSA values).
    state: BTreeMap<VReg, SCCPState>,
    /// Per-LocalId lattice state (flow-insensitive; see header).
    /// Meet across all reachable `BindLocal`/`BindImmediate`/`StoreLocal`
    /// sites for the same id, and explicitly Bottom'd by Escape taint.
    lstate: BTreeMap<LocalId, SCCPState>,
    flow_wl: Vec<FlowWLEntry>,
    use_wl: Vec<UseEntry>,
}

/// Strip the SCCP bookkeeping back into a plain `IRSegment`:
///   * drop statements that DCE marked `dead`
///   * propagate per-block reachability into `IRBasicBlock::dead`
///     (so the pretty-printer can render dead-block stubs).
impl From<EnrichedIRSegment> for IRSegment {
    fn from(e: EnrichedIRSegment) -> Self {
        IRSegment {
            regs: e.vregs,
            blocks: e.blocks.into_iter().map(|b| IRBasicBlock {
                dead: matches!(b.state, BlockState::Bottom),
                statements: b.statements.into_iter()
                    .filter(|s| !s.dead)
                    .map(|s| s.seg)
                    .collect(),
            }).collect(),
        }
    }
}

impl From<IRSegment> for EnrichedIRSegment {
    fn from(seg: IRSegment) -> Self {
        let blocks = seg.blocks.into_iter().map(|b| EnrichedIRBasicBlock {
            statements: b.statements.into_iter()
                .map(|s| EnrichedIRStatement { seg: s, dead: false })
                .collect(),
            state: BlockState::Bottom,
        }).collect();
        EnrichedIRSegment {
            vregs: seg.regs,
            blocks,
            // All VRegs/LocalIds implicitly Top — missing == Top.
            state: BTreeMap::new(),
            lstate: BTreeMap::new(),
            // Block 0 is reachable via a synthetic entry edge.
            flow_wl: vec![FlowWLEntry { from: None, to: 0 }],
            use_wl: Vec::new(),
        }
    }
}

/// Full optimization pipeline:
///   1. SCCP lattice fixpoint.
///   2. Fold statements / `CondBr`s whose values became constant.
///   3. DCE — mark side-effect-free, no-use statements as dead.
///   4. Dead-block tagging — mark blocks unreachable from `.L0`.
/// Dead statements are dropped at IRSegment-conversion time; dead
/// blocks are kept in place (with the flag set) so block IDs stay
/// stable for phi predecessors.
pub(crate) fn optimize(seg: IRSegment) -> EnrichedIRSegment {
    let mut e: EnrichedIRSegment = seg.into();
    e.sccp();
    e.fold();
    e.dce();
    e.dead_blocks();
    e
}

impl EnrichedIRSegment {
    fn get(&self, r: &VReg) -> SCCPState {
        self.state.get(r).cloned().unwrap_or(SCCPState::Top)
    }

    fn get_local(&self, id: &LocalId) -> SCCPState {
        self.lstate.get(id).cloned().unwrap_or(SCCPState::Top)
    }

    /// Lower `r` via meet with `new`. If the value moved, push every use
    /// of `r` onto the SSA worklist for re-evaluation.
    fn update(
        &mut self,
        r: &VReg,
        new: SCCPState,
        uses: &BTreeMap<VReg, Vec<UseEntry>>,
    ) {
        let old = self.get(r);
        let merged = old.clone().meet(new);
        if merged != old {
            self.state.insert(r.clone(), merged);
            if let Some(u) = uses.get(r) {
                self.use_wl.extend(u.iter().copied());
            }
        }
    }

    /// Lower `lstate[id]` via meet. If the value moved, push every
    /// `LoadLocal(_, id)` site onto the use worklist so consumers
    /// re-evaluate against the new state.
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

    /// Push a flow-worklist edge — but only if `exec_edges` hasn't
    /// already marked it live. Keeps `flow_wl` free of duplicates so
    /// `exec_edges` stays the sole source of truth for liveness.
    fn push_flow(&mut self, exec_edges: &BTreeSet<FlowWLEntry>, e: FlowWLEntry) {
        if !exec_edges.contains(&e) {
            self.flow_wl.push(e);
        }
    }

    /// SCCP fixpoint loop. Alternates between CFG-edge and SSA-use
    /// worklists until both are empty.
    fn sccp(&mut self) {
        let uses = compute_uses(&self.blocks);
        let local_uses = compute_local_uses(&self.blocks);
        let reach_out = compute_reach(&self.blocks);
        let mut exec_edges: BTreeSet<FlowWLEntry> = BTreeSet::new();
        let mut exec_blocks: BTreeSet<usize> = BTreeSet::new();

        loop {
            if let Some(edge) = self.flow_wl.pop() {
                // Invariant: `push_flow` filtered out edges already in
                // `exec_edges`, so every pop is genuinely new work.
                exec_edges.insert(edge);
                let first_time = exec_blocks.insert(edge.to);
                let n = self.blocks[edge.to].statements.len();
                for i in 0..n {
                    let stmt = self.blocks[edge.to].statements[i].seg.clone();
                    if first_time {
                        self.visit(edge.to, &stmt, &uses, &local_uses, &reach_out, &exec_edges);
                    } else if matches!(stmt, IRStatement::PhiOp(..)) {
                        // New edge into an already-live block: only the
                        // phis need to refresh; their inputs may now
                        // include the new predecessor.
                        self.visit(edge.to, &stmt, &uses, &local_uses, &reach_out, &exec_edges);
                    } else {
                        // Phis live at the top of the block — first
                        // non-phi means we're done.
                        break;
                    }
                }
            } else if let Some(u) = self.use_wl.pop() {
                // Ignore uses inside not-yet-reachable blocks; we'll see
                // them again when the block does become live.
                if !exec_blocks.contains(&u.block) { continue; }
                let stmt = self.blocks[u.block].statements[u.stmt].seg.clone();
                self.visit(u.block, &stmt, &uses, &local_uses, &reach_out, &exec_edges);
            } else {
                return;
            }
        }
    }

    /// Dispatch on statement kind. Terminators feed the flow worklist;
    /// phis meet over reachable predecessors only; locals propagate to
    /// `lstate`; Escape taints in-scope locals; everything else goes
    /// through `eval`.
    fn visit(
        &mut self,
        block: usize,
        s: &IRStatement,
        uses: &BTreeMap<VReg, Vec<UseEntry>>,
        local_uses: &BTreeMap<LocalId, Vec<UseEntry>>,
        reach_out: &[BTreeSet<LocalId>],
        exec_edges: &BTreeSet<FlowWLEntry>,
    ) {
        use IRStatement::*;
        match s {
            Br(t) => self.push_flow(exec_edges, FlowWLEntry { from: Some(block), to: *t }),

            CondBr { cond, then_blk, else_blk } => match self.get(cond) {
                // Known condition → only one successor lives.
                SCCPState::Constant(v) => {
                    let to = if is_falsy(&v) { *else_blk } else { *then_blk };
                    self.push_flow(exec_edges, FlowWLEntry { from: Some(block), to });
                }
                // Unknowable → both successors live.
                SCCPState::Bottom => {
                    self.push_flow(exec_edges, FlowWLEntry { from: Some(block), to: *then_blk });
                    self.push_flow(exec_edges, FlowWLEntry { from: Some(block), to: *else_blk });
                }
                // Not yet observed — we'll be revisited when `cond` drops.
                SCCPState::Top => {}
            }

            Ret(_) => {}

            PhiOp(dst, (pa, va), (pb, vb)) => {
                // Meet only over predecessor edges that are actually live.
                let ra = exec_edges.contains(&FlowWLEntry { from: Some(*pa), to: block });
                let rb = exec_edges.contains(&FlowWLEntry { from: Some(*pb), to: block });
                let st = match (ra, rb) {
                    (true, true) => self.get(va).meet(self.get(vb)),
                    (true, false) => self.get(va),
                    (false, true) => self.get(vb),
                    (false, false) => SCCPState::Top,
                };
                self.update(dst, st, uses);
            }

            // --- locals: lstate writes ---
            BindLocal { id, src, .. } => {
                let st = self.get(src);
                self.update_local(*id, st, local_uses);
            }
            BindImmediate { id, src, .. } => {
                // Inline-Value bind — Closures/Macros stay opaque, mirror
                // `eval`'s rule for `Load`.
                let st = match src {
                    Value::Closure(_) | Value::Macro(_) | Value::JittedClosure(_) => SCCPState::Bottom,
                    _ => SCCPState::Constant(src.clone()),
                };
                self.update_local(*id, st, local_uses);
            }
            StoreLocal(id, src) => {
                let st = self.get(src);
                self.update_local(*id, st, local_uses);
            }

            // --- locals: lstate read ---
            LoadLocal(d, id) => {
                let st = self.get_local(id);
                self.update(d, st, uses);
            }

            // --- escape: taint everything possibly bound at this point ---
            Escape(d, _) => {
                // Snapshot the set first to avoid the borrow conflict with
                // `update_local`'s &mut self.
                let to_taint: Vec<LocalId> = reach_out[block].iter().copied().collect();
                for id in to_taint {
                    self.update_local(id, SCCPState::Bottom, local_uses);
                }
                self.update(d, SCCPState::Bottom, uses);
            }

            // --- call: same taint rules as escape, since the callee may
            // recursively bounce back into the interpreter and set! any
            // in-scope local on this path.
            Call { dst, .. } => {
                let to_taint: Vec<LocalId> = reach_out[block].iter().copied().collect();
                for id in to_taint {
                    self.update_local(id, SCCPState::Bottom, local_uses);
                }
                self.update(dst, SCCPState::Bottom, uses);
            }

            _ => {
                if let Some((d, st)) = self.eval(s) {
                    self.update(&d, st, uses);
                }
            }
        }
    }

    /// Compute a new lattice value for the destination of `s`.
    /// Returns `None` for statements that don't write a VReg.
    fn eval(&self, s: &IRStatement) -> Option<(VReg, SCCPState)> {
        use IRStatement::*;
        use SCCPState::*;

        // `Load`: literal propagation. Closures/Macros opaque → Bottom.
        if let Load(d, v) = s {
            let st = match v {
                Value::Closure(_) | Value::Macro(_) | Value::JittedClosure(_) => Bottom,
                _ => Constant(v.clone()),
            };
            return Some((d.clone(), st));
        }

        // Numeric arithmetic — evaluate via existing Number methods.
        // If both operands are constant non-numbers, the runtime would
        // error — we collapse to Bottom (no useful info to propagate).
        macro_rules! fold_arith {
            ($d:expr, $a:expr, $b:expr, $m:ident) => {{
                let st = match (self.get($a), self.get($b)) {
                    (Constant(Value::Number(na)), Constant(Value::Number(nb))) => {
                        match na.$m(nb) {
                            Ok(n) => Constant(Value::Number(n)),
                            Err(_) => Bottom,
                        }
                    }
                    (Bottom, _) | (_, Bottom) => Bottom,
                    (Top, _) | (_, Top) => Top,
                    _ => Bottom, // both Constant but non-numeric — type error at runtime
                };
                Some(($d.clone(), st))
            }};
        }

        // Numeric comparisons via `PartialOrd`/`PartialEq`.
        macro_rules! fold_cmp {
            ($d:expr, $a:expr, $b:expr, $op:tt) => {{
                let st = match (self.get($a), self.get($b)) {
                    (Constant(Value::Number(na)), Constant(Value::Number(nb))) => {
                        Constant(Value::Bool(na $op nb))
                    }
                    (Bottom, _) | (_, Bottom) => Bottom,
                    (Top, _) | (_, Top) => Top,
                    _ => Bottom,
                };
                Some(($d.clone(), st))
            }};
        }

        match s {
            // Arithmetic
            Add(d, a, b)    => fold_arith!(d, a, b, add),
            Sub(d, a, b)    => fold_arith!(d, a, b, sub),
            Mul(d, a, b)    => fold_arith!(d, a, b, mul),
            Div(d, a, b)    => fold_arith!(d, a, b, div),
            Mod(d, a, b)    => fold_arith!(d, a, b, modulo),
            Lshift(d, a, b) => fold_arith!(d, a, b, lshift),
            Rshift(d, a, b) => fold_arith!(d, a, b, rshift),

            // Comparisons
            Eq(d, a, b)  => fold_cmp!(d, a, b, ==),
            Gt(d, a, b)  => fold_cmp!(d, a, b, >),
            Lt(d, a, b)  => fold_cmp!(d, a, b, <),
            Gte(d, a, b) => fold_cmp!(d, a, b, >=),
            Lte(d, a, b) => fold_cmp!(d, a, b, <=),

            // Logical NOT — mirrors the interpreter's per-variant rules.
            LogNot(d, a) => {
                let st = match self.get(a) {
                    Constant(Value::Bool(b)) => Constant(Value::Bool(!b)),
                    Constant(Value::Number(Number::Integer(0))) =>
                        Constant(Value::Number(Number::Integer(1))),
                    Constant(Value::Number(Number::Integer(_))) =>
                        Constant(Value::Number(Number::Integer(0))),
                    Constant(Value::Number(Number::Unsigned(0))) =>
                        Constant(Value::Number(Number::Unsigned(1))),
                    Constant(Value::Number(Number::Unsigned(_))) =>
                        Constant(Value::Number(Number::Unsigned(0))),
                    Constant(_) => Bottom, // runtime error
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.clone(), st))
            }

            // Eager XOR — Bool(truthy(a) != truthy(b)).
            Xor(d, a, b) => {
                let st = match (self.get(a), self.get(b)) {
                    (Constant(va), Constant(vb)) =>
                        Constant(Value::Bool(is_falsy(&va) != is_falsy(&vb))),
                    (Bottom, _) | (_, Bottom) => Bottom,
                    _ => Top,
                };
                Some((d.clone(), st))
            }

            // Bitwise — Integer/Unsigned only, mixing matches the
            // interpreter's coercion (Unsigned dominates).
            BinNot(d, a) => {
                let st = match self.get(a) {
                    Constant(Value::Number(Number::Integer(i))) =>
                        Constant(Value::Number(Number::Integer(!i))),
                    Constant(Value::Number(Number::Unsigned(u))) =>
                        Constant(Value::Number(Number::Unsigned(!u))),
                    Constant(_) => Bottom,
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.clone(), st))
            }
            BinOr(d, a, b)  => Some((d.clone(), fold_bin(self.get(a), self.get(b), |x, y| x | y, |x, y| x | y))),
            BinAnd(d, a, b) => Some((d.clone(), fold_bin(self.get(a), self.get(b), |x, y| x & y, |x, y| x & y))),

            // Cons / car / cdr / nullp on constant data.
            Cons(d, a, b) => {
                let st = match (self.get(a), self.get(b)) {
                    (Constant(va), Constant(vb)) => Constant(Value::cons(va, vb)),
                    (Bottom, _) | (_, Bottom) => Bottom,
                    _ => Top,
                };
                Some((d.clone(), st))
            }
            Car(d, a) => {
                let st = match self.get(a) {
                    Constant(Value::Cons(car, _)) => Constant((*car).clone()),
                    Constant(_) => Constant(Value::Nil), // interpreter: non-cons → nil
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.clone(), st))
            }
            Cdr(d, a) => {
                let st = match self.get(a) {
                    Constant(Value::Cons(_, cdr)) => Constant((*cdr).clone()),
                    Constant(_) => Constant(Value::Nil),
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.clone(), st))
            }
            Nullp(d, a) => {
                let st = match self.get(a) {
                    Constant(Value::Nil) => Constant(Value::Bool(true)),
                    Constant(_) => Constant(Value::Bool(false)),
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.clone(), st))
            }

            // Type coercion on a constant Number — fold via the
            // Number methods. Non-Number constants or failed coercion
            // would error at runtime → Bottom.
            AsAddr(d, a) => {
                let st = match self.get(a) {
                    Constant(Value::Number(n)) => match n.as_addr() {
                        Ok(nn) => Constant(Value::Number(nn)),
                        Err(_) => Bottom,
                    },
                    Constant(_) => Bottom,
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.clone(), st))
            }
            AsSigned(d, a) => {
                let st = match self.get(a) {
                    Constant(Value::Number(n)) => match n.as_i32() {
                        Ok(i) => Constant(Value::Number(Number::Integer(i))),
                        Err(_) => Bottom,
                    },
                    Constant(_) => Bottom,
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.clone(), st))
            }
            AsUnsigned(d, a) => {
                let st = match self.get(a) {
                    Constant(Value::Number(n)) => match n.as_u32() {
                        Ok(u) => Constant(Value::Number(Number::Unsigned(u))),
                        Err(_) => Bottom,
                    },
                    Constant(_) => Bottom,
                    Top => Top,
                    Bottom => Bottom,
                };
                Some((d.clone(), st))
            }

            // Everything else producing a value: Bottom. Captures
            // (mutable, no flow-insensitive tracking), arrays/cons-array
            // ops (heap), syscalls.
            // LoadLocal and Escape are handled in `visit` (they touch
            // lstate / trigger taint) — they never reach this `eval`.
            LoadCapture(d, _)
            | Array(d, _) | Full(d, _, _) | Unpack(d, _)
            | GetIdx(d, _, _)
            | Hits(d, _)
            | SysDsb(d) | SysPrefetchFlush(d) | SysUartInit(d) | SysUartGet8(d)
            | SysClearMonitor(d) | SysGetMonitor(d) | SysStopMonitor(d)
            | SysGet32(d, _) | SysUartPut8(d, _) | SysDelay(d, _)
            | SysPut32(d, _, _)
                => Some((d.clone(), Bottom)),

            PutIdx(d, _, _, _) | ReadIdx(d, _, _, _) | FillIdx(d, _, _, _)
            | SysZero32(d, _, _, _)
                => Some((d.clone(), Bottom)),

            FullIdx(d, _, _, _, _) | SysFull32(d, _, _, _, _)
                => Some((d.clone(), Bottom)),

            // Call result is opaque — like Escape, treat as Bottom.
            // Handled here (not via `visit`) so the dst gets a value
            // assignment in the lattice.
            Call { dst, .. } => Some((dst.clone(), Bottom)),

            // No-dst stmts or handled in `visit`.
            Load(_, _)
            | LoadLocal(_, _) | Escape(_, _)
            | PushFrame | PopFrame
            | BindLocal { .. } | BindImmediate { .. }
            | StoreLocal(_, _) | StoreCapture(_, _)
            | Br(_) | CondBr { .. } | Ret(_) | PhiOp(_, _, _)
                => None,
        }
    }

    /// Replace every statement whose destination is `Constant(v)` with a
    /// `Load(dst, v)`. Also rewrite any `CondBr` whose condition became
    /// constant into a plain `Br` to the taken successor. Statements
    /// still at Top or Bottom keep their original form.
    fn fold(&mut self) {
        for b in &mut self.blocks {
            for s in &mut b.statements {
                // Collapse `Load + BindLocal` into a single
                // `BindImmediate` when the BindLocal's source VReg has
                // been proved constant. The Load that defined that VReg
                // becomes dead and gets cleaned up by the DCE pass.
                if let IRStatement::BindLocal { name, id, src } = &s.seg {
                    if let SCCPState::Constant(v) =
                        self.state.get(src).cloned().unwrap_or(SCCPState::Top)
                    {
                        s.seg = IRStatement::BindImmediate {
                            name: name.clone(),
                            id: *id,
                            src: v,
                        };
                        continue;
                    }
                }

                // Constant-folded CondBr → unconditional Br to the
                // statically-known successor.
                if let IRStatement::CondBr { cond, then_blk, else_blk } = &s.seg {
                    if let SCCPState::Constant(v) =
                        self.state.get(cond).cloned().unwrap_or(SCCPState::Top)
                    {
                        let to = if is_falsy(&v) { *else_blk } else { *then_blk };
                        s.seg = IRStatement::Br(to);
                        continue;
                    }
                }
                // Value-producing statement whose result is constant →
                // plain Load.
                if let Some(d) = dst_reg(&s.seg) {
                    if let SCCPState::Constant(v) =
                        self.state.get(&d).cloned().unwrap_or(SCCPState::Top)
                    {
                        if !matches!(&s.seg, IRStatement::Load(_, _)) {
                            s.seg = IRStatement::Load(d, v);
                        }
                    }
                }
            }
        }
    }
}

impl EnrichedIRSegment {
    /// Dead-code elimination at the statement level. Only side-effect-free
    /// statements with empty use-lists become dead; the kill propagates
    /// because removing a stmt shrinks its operands' use-lists, possibly
    /// enabling more kills.
    ///
    /// SSA matters here: each VReg has exactly one definition, so a
    /// single per-VReg use-list is enough to track "is this value still
    /// used?". When we mark a stmt dead, we drop it from every operand's
    /// use-list and re-queue each operand's defining statement; the
    /// worklist runs to fixpoint.
    fn dce(&mut self) {
        let mut uses = compute_uses(&self.blocks);
        let defs = compute_defs(&self.blocks);

        // Seed the worklist with every statement.
        let mut wl: Vec<UseEntry> = Vec::new();
        for (bi, b) in self.blocks.iter().enumerate() {
            for si in 0..b.statements.len() {
                wl.push(UseEntry { block: bi, stmt: si });
            }
        }

        while let Some(u) = wl.pop() {
            let s = &self.blocks[u.block].statements[u.stmt];
            if s.dead { continue; }
            if has_side_effect(&s.seg) { continue; }
            let dst = match dst_reg(&s.seg) {
                Some(d) => d,
                None => continue,
            };
            // Live iff something still reads `dst`.
            if !uses.get(&dst).map_or(true, |v| v.is_empty()) { continue; }

            // Kill this statement. Pull `u` out of every operand's
            // use-list, then re-queue each operand's defining site —
            // they may have just become DCE-eligible too.
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

    /// Trace reachability from `.L0` through live `Br` / `CondBr`
    /// terminators. Blocks we reach are tagged `Top`; anything left at
    /// `Bottom` is dead. We never delete dead blocks — IDs are kept
    /// stable so phi predecessors keep pointing at the right slot.
    fn dead_blocks(&mut self) {
        let mut wl: Vec<usize> = alloc::vec![0];
        let mut seen: BTreeSet<usize> = BTreeSet::new();
        while let Some(b) = wl.pop() {
            if !seen.insert(b) { continue; }
            self.blocks[b].state = BlockState::Top;
            for s in &self.blocks[b].statements {
                if s.dead { continue; }
                match &s.seg {
                    IRStatement::Br(t) => wl.push(*t),
                    IRStatement::CondBr { then_blk, else_blk, .. } => {
                        wl.push(*then_blk);
                        wl.push(*else_blk);
                    }
                    _ => {}
                }
            }
        }
    }
}

/// For each VReg, the (block, stmt) site that defines it. Built once
/// per DCE invocation; we rely on SSA (one def per VReg) so a single
/// entry per key is sufficient.
fn compute_defs(blocks: &[EnrichedIRBasicBlock]) -> BTreeMap<VReg, UseEntry> {
    let mut defs = BTreeMap::new();
    for (bi, b) in blocks.iter().enumerate() {
        for (si, s) in b.statements.iter().enumerate() {
            if let Some(d) = dst_reg(&s.seg) {
                defs.insert(d, UseEntry { block: bi, stmt: si });
            }
        }
    }
    defs
}

/// True iff removing the statement could change observable program
/// behavior (mutating state, allocating, branching, calling out to the
/// interpreter, touching MMIO, etc.). DCE only kills statements that
/// are side-effect-*free*.
fn has_side_effect(s: &IRStatement) -> bool {
    use IRStatement::*;
    // BindLocal / BindImmediate / StoreLocal / StoreCapture all mutate
    // runtime state (slot contents and/or frame entries) — they're
    // implicitly side-effecting via the catch-all below.
    !matches!(s,
        Load(..) | LoadLocal(..) | LoadCapture(..)
        | Add(..) | Sub(..) | Mul(..) | Div(..) | Mod(..) | Lshift(..) | Rshift(..)
        | BinNot(..) | BinOr(..) | BinAnd(..) | LogNot(..) | Xor(..)
        | Eq(..) | Gt(..) | Lt(..) | Gte(..) | Lte(..)
        | AsAddr(..) | AsSigned(..) | AsUnsigned(..)
        | Cons(..) | Car(..) | Cdr(..) | Nullp(..)
        | PhiOp(..) | Hits(..)
    )
}

/// For each LocalId, the (block, stmt) sites that read it via
/// `LoadLocal`. Used to drive re-evaluation when `lstate[id]` lowers.
fn compute_local_uses(
    blocks: &[EnrichedIRBasicBlock],
) -> BTreeMap<LocalId, Vec<UseEntry>> {
    let mut uses: BTreeMap<LocalId, Vec<UseEntry>> = BTreeMap::new();
    for (bi, b) in blocks.iter().enumerate() {
        for (si, s) in b.statements.iter().enumerate() {
            if let IRStatement::LoadLocal(_, id) = &s.seg {
                uses.entry(*id).or_default().push(UseEntry { block: bi, stmt: si });
            }
        }
    }
    uses
}

/// Forward dataflow (syntactic CFG, all blocks): for each block B,
/// `reach_out[B]` is the set of LocalIds whose `BindLocal*` appears on
/// some path from entry through B. Used by `visit`'s Escape arm to
/// taint locals possibly in-scope at the escape.
///
/// Uses the syntactic CFG (every Br/CondBr edge, dead or not). Over-
/// tainting due to a dead block contributing binds is sound — the
/// tainted local would have been Bottom anyway since SCCP wouldn't
/// have lowered its lstate from those dead Bind*s.
fn compute_reach(blocks: &[EnrichedIRBasicBlock]) -> Vec<BTreeSet<LocalId>> {
    let n = blocks.len();

    // binds[B] = LocalIds whose initial bind lives in B. Only `BindLocal`/
    // `BindImmediate` introduce a LocalId into the runtime image's
    // frame stack — `StoreLocal` mutates an already-introduced binding
    // and doesn't widen the in-scope set.
    let mut binds: Vec<BTreeSet<LocalId>> = vec![BTreeSet::new(); n];
    for (bi, b) in blocks.iter().enumerate() {
        for s in &b.statements {
            match &s.seg {
                IRStatement::BindLocal { id, .. }
                | IRStatement::BindImmediate { id, .. } => {
                    binds[bi].insert(*id);
                }
                _ => {}
            }
        }
    }

    // Build predecessor list from the syntactic CFG.
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (bi, b) in blocks.iter().enumerate() {
        for s in &b.statements {
            match &s.seg {
                IRStatement::Br(t) => preds[*t].push(bi),
                IRStatement::CondBr { then_blk, else_blk, .. } => {
                    preds[*then_blk].push(bi);
                    preds[*else_blk].push(bi);
                }
                _ => {}
            }
        }
    }

    // Naive fixpoint: reach_out[B] = (∪ reach_out[p] for p ∈ preds) ∪ binds[B].
    // Segments are small (a handful of blocks); the cost is negligible.
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
        if !changed { break; }
    }
    reach_out
}

/// For each VReg, the (block, stmt) sites that read it. Used to drive
/// re-evaluation when a value's lattice state lowers.
fn compute_uses(blocks: &[EnrichedIRBasicBlock]) -> BTreeMap<VReg, Vec<UseEntry>> {
    let mut uses: BTreeMap<VReg, Vec<UseEntry>> = BTreeMap::new();
    for (bi, b) in blocks.iter().enumerate() {
        for (si, s) in b.statements.iter().enumerate() {
            let e = UseEntry { block: bi, stmt: si };
            for r in stmt_operands(&s.seg) {
                uses.entry(r).or_default().push(e);
            }
        }
    }
    uses
}

/// Every VReg operand a statement reads.
fn stmt_operands(s: &IRStatement) -> Vec<VReg> {
    use IRStatement::*;
    match s {
        Load(_, _) | LoadLocal(_, _) | LoadCapture(_, _)
        | PushFrame | PopFrame | Br(_)
        | BindImmediate { .. }
        | SysDsb(_) | SysPrefetchFlush(_) | SysUartInit(_) | SysUartGet8(_)
        | SysClearMonitor(_) | SysGetMonitor(_) | SysStopMonitor(_)
            => vec![],

        StoreLocal(_, r) | StoreCapture(_, r) | Ret(r)
        | BindLocal { src: r, .. }
        | BinNot(_, r) | LogNot(_, r)
        | AsAddr(_, r) | AsSigned(_, r) | AsUnsigned(_, r)
        | Car(_, r) | Cdr(_, r) | Nullp(_, r)
        | Array(_, r) | Unpack(_, r) | Hits(_, r)
        | SysGet32(_, r) | SysUartPut8(_, r) | SysDelay(_, r)
        | Escape(_, r)
        | CondBr { cond: r, .. }
            => vec![r.clone()],

        Add(_, a, b) | Sub(_, a, b) | Mul(_, a, b) | Div(_, a, b) | Mod(_, a, b)
        | Lshift(_, a, b) | Rshift(_, a, b)
        | BinOr(_, a, b) | BinAnd(_, a, b) | Xor(_, a, b)
        | Eq(_, a, b) | Gt(_, a, b) | Lt(_, a, b) | Gte(_, a, b) | Lte(_, a, b)
        | Cons(_, a, b) | Full(_, a, b) | GetIdx(_, a, b)
        | SysPut32(_, a, b)
            => vec![a.clone(), b.clone()],

        PutIdx(_, t, i, v) => vec![t.clone(), i.clone(), v.clone()],
        ReadIdx(_, t, o, n) => vec![t.clone(), o.clone(), n.clone()],
        FillIdx(_, t, o, l) => vec![t.clone(), o.clone(), l.clone()],
        SysZero32(_, a, b, c) => vec![a.clone(), b.clone(), c.clone()],

        FullIdx(_, t, o, n, v) => vec![t.clone(), o.clone(), n.clone(), v.clone()],
        SysFull32(_, a, b, c, d) => vec![a.clone(), b.clone(), c.clone(), d.clone()],

        PhiOp(_, (_, a), (_, b)) => vec![a.clone(), b.clone()],

        Call { args, .. } => args.clone(),
    }
}

/// Destination VReg of a statement, if it writes one.
fn dst_reg(s: &IRStatement) -> Option<VReg> {
    use IRStatement::*;
    match s {
        Load(d, _) | LoadLocal(d, _) | LoadCapture(d, _)
        | BinNot(d, _) | LogNot(d, _)
        | AsAddr(d, _) | AsSigned(d, _) | AsUnsigned(d, _)
        | Car(d, _) | Cdr(d, _) | Nullp(d, _)
        | Array(d, _) | Unpack(d, _) | Hits(d, _)
        | SysDsb(d) | SysPrefetchFlush(d) | SysUartInit(d) | SysUartGet8(d)
        | SysClearMonitor(d) | SysGetMonitor(d) | SysStopMonitor(d)
        | SysGet32(d, _) | SysUartPut8(d, _) | SysDelay(d, _)
        | Escape(d, _)
            => Some(d.clone()),

        Add(d, _, _) | Sub(d, _, _) | Mul(d, _, _) | Div(d, _, _) | Mod(d, _, _)
        | Lshift(d, _, _) | Rshift(d, _, _)
        | BinOr(d, _, _) | BinAnd(d, _, _) | Xor(d, _, _)
        | Eq(d, _, _) | Gt(d, _, _) | Lt(d, _, _) | Gte(d, _, _) | Lte(d, _, _)
        | Cons(d, _, _) | Full(d, _, _) | GetIdx(d, _, _)
        | SysPut32(d, _, _)
            => Some(d.clone()),

        PutIdx(d, _, _, _) | ReadIdx(d, _, _, _) | FillIdx(d, _, _, _)
        | SysZero32(d, _, _, _)
            => Some(d.clone()),

        FullIdx(d, _, _, _, _) | SysFull32(d, _, _, _, _)
            => Some(d.clone()),

        PhiOp(d, _, _) => Some(d.clone()),

        Call { dst, .. } => Some(dst.clone()),

        PushFrame | PopFrame | BindLocal { .. } | BindImmediate { .. }
        | StoreLocal(_, _) | StoreCapture(_, _)
        | Br(_) | CondBr { .. } | Ret(_) => None,
    }
}

/// Fold a bitwise binop over Integer/Unsigned operands. Matches the
/// interpreter's coercion: mixed Integer+Unsigned promotes to Unsigned.
/// `oi`/`ou` are the operations on i32 and u32 respectively.
fn fold_bin(
    a: SCCPState,
    b: SCCPState,
    oi: fn(i32, i32) -> i32,
    ou: fn(u32, u32) -> u32,
) -> SCCPState {
    use SCCPState::*;
    match (a, b) {
        (Constant(Value::Number(na)), Constant(Value::Number(nb))) => match (na, nb) {
            (Number::Integer(x), Number::Integer(y))   => Constant(Value::Number(Number::Integer(oi(x, y)))),
            (Number::Unsigned(x), Number::Unsigned(y)) => Constant(Value::Number(Number::Unsigned(ou(x, y)))),
            (Number::Unsigned(x), Number::Integer(y))  => Constant(Value::Number(Number::Unsigned(ou(x, y as u32)))),
            (Number::Integer(x), Number::Unsigned(y))  => Constant(Value::Number(Number::Unsigned(ou(x as u32, y)))),
            _ => Bottom,
        },
        (Bottom, _) | (_, Bottom) => Bottom,
        (Top, _) | (_, Top) => Top,
        _ => Bottom,
    }
}

/// Truthiness matching the interpreter's `is_falsy` (used for the
/// CondBr-with-known-cond folding).
fn is_falsy(v: &Value) -> bool {
    matches!(v,
        Value::Nil
        | Value::Bool(false)
        | Value::Number(Number::Integer(0))
        | Value::Number(Number::Unsigned(0)))
}
