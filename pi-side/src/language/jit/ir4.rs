//! LIR — Low IR. Final form before machine-code emission.
//!
//! Each instruction maps 1:1 to one (or a tiny fixed sequence of) ARM
//! instruction(s). Operands are typed by the type parameter `R`:
//!   * `Instr<Register>` is the **post-regalloc** form — every operand
//!     is a physical register. This is what `LIRSegment.blocks` holds.
//!   * `Instr<Operand>` (defined in `regalloc.rs`) is the intermediate
//!     form used during regalloc, where operands can be virtual
//!     (`Operand::V(u32)`) or pre-colored physical (`Operand::P(Register)`).
//!
//! Sharing the variant set under `Instr<R>` means the operand walkers
//! (`uses` / `defs` / rename / printer) are written exactly once,
//! parameterized over `R`.
//!
//! # Register classification
//!
//! ARMv6 has 16 GPRs:
//!   * **r0–r3, r12**: caller-saved (AAPCS argument-and-scratch slots).
//!     NOT in the regalloc color pool. They appear in LIR only as
//!     targets of explicit `Mov` instructions emitted by the helper-
//!     call sequencer or as R12, the reserved spill-and-cycle-break
//!     scratch.
//!   * **r4–r11**: callee-saved. The regalloc color pool. The
//!     `LIRSegment` tracks which of these the function actually used
//!     so prologue/epilogue can save/restore exactly that subset.
//!   * **sp/lr/pc**: reserved.
//!
//! # Helper-call opcodes
//!
//! Each helper variant (e.g. `BindLocal`, `Cons`, `Escape`) is
//! conceptually `bl <symbol>`. Arguments arrive in r0–r3 (placed by
//! preceding `Mov` instructions); the return value is in r0 (captured
//! by a following `Mov`). Each helper opcode clobbers r0–r3, r12,
//! lr, and condition flags.
//!
//! # Pre-asm-emit forms
//!
//! `Phi` and `CondBr` are valid in `Instr<Operand>` (during regalloc)
//! but should NOT appear in `Instr<Register>` after `apply_colors`.
//! The latter expands `CondBr` to `CmpImm + Beq + B` and `Phi` is
//! dropped by the phi-destruction pass.

use alloc::vec::Vec;
use core::fmt;

use super::ir::Name;
use super::ir3::ImmNumber;
use super::scope::LocalId;
use crate::language::ast::Value;
use crate::language::environment::Binding;

/// ARMv6 GPR. Variants in canonical-number order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)]
pub(crate) enum Register {
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    R10,
    R11,
    R12,
    SP,
    LR,
    PC,
}

#[allow(dead_code)]
impl Register {
    /// Callee-saved color pool used by the regalloc.
    pub const POOL: &'static [Register] = &[
        Register::R4,
        Register::R5,
        Register::R6,
        Register::R7,
        Register::R8,
        Register::R9,
        Register::R10,
        Register::R11,
    ];

    /// Reserved spill / parallel-move-cycle-break scratch. Never colored.
    pub const SPILL_SCRATCH: Register = Register::R12;

    /// AAPCS argument registers in positional order.
    pub const ARG: [Register; 4] = [Register::R0, Register::R1, Register::R2, Register::R3];

    /// AAPCS return register.
    pub const RETURN: Register = Register::R0;

    /// Bit position of this register inside a `StackPush` / `StackPop`
    /// bitmask (R0=0, R1=1, …, LR=14, PC=15). Matches ARM's `ldm/stm`
    /// register-list encoding.
    pub fn bit(self) -> u16 {
        self as u16
    }

    /// Index of this register inside `POOL` (R4=0, R5=1, … R11=7).
    /// `None` for non-pool regs.
    pub fn pool_index(self) -> Option<u32> {
        match self {
            Register::R4 => Some(0),
            Register::R5 => Some(1),
            Register::R6 => Some(2),
            Register::R7 => Some(3),
            Register::R8 => Some(4),
            Register::R9 => Some(5),
            Register::R10 => Some(6),
            Register::R11 => Some(7),
            _ => None,
        }
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = match self {
            Register::R0 => "r0",
            Register::R1 => "r1",
            Register::R2 => "r2",
            Register::R3 => "r3",
            Register::R4 => "r4",
            Register::R5 => "r5",
            Register::R6 => "r6",
            Register::R7 => "r7",
            Register::R8 => "r8",
            Register::R9 => "r9",
            Register::R10 => "r10",
            Register::R11 => "r11",
            Register::R12 => "r12",
            Register::SP => "sp",
            Register::LR => "lr",
            Register::PC => "pc",
        };
        f.write_str(n)
    }
}

/// Frame slot for a spilled virtual register. The lowering layer maps
/// each slot to a concrete `[sp, #offset]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SpillSlot(pub(super) u32);

impl fmt::Display for SpillSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[sp+{}]", self.0)
    }
}

/// ARM condition codes used by `Cset` and conditional branches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum Cond {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cond {
    pub(super) fn suffix(self) -> &'static str {
        match self {
            Cond::Eq => "eq",
            Cond::Ne => "ne",
            Cond::Lt => "lt",
            Cond::Le => "le",
            Cond::Gt => "gt",
            Cond::Ge => "ge",
        }
    }
}

/// Generic LIR instruction over an operand type `R`. Specialized as
/// `Instr<Operand>` during regalloc and `Instr<Register>` after.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) enum Instr<R> {
    // ===== moves / loads =====
    /// `mov dst, src`. Sources: any IR3 op that produced a virtual-
    /// register move via the sequencer (helper arg/ret materialization,
    /// `AsAddr`/`AsSigned`/`AsUnsigned`, phi-destruction copies).
    Mov(R, R),
    /// `mov dst, #imm`. Source: IR3 `MovImm`, `BindImmediate` (when
    /// expanded — the `src` immediate is loaded into r0).
    MovImm(R, ImmNumber),
    /// `mov dst, #<localid>`. Source: every helper that takes a LocalId
    /// argument (`LoadLocal`, `StoreLocal`, `UnboxLocal`, `BindLocal*`).
    MovId(R, LocalId),
    /// `ldr dst, =VALUE_LITERAL` (literal-pool). Source: IR3
    /// `MovImmAddr`, plus the nil-return materialization for
    /// `SysDsb`/`SysPrefetchFlush`/`SysPut32`.
    LdrValuePtr(R, Value),
    /// `ldr dst, =NAME_LITERAL` (literal-pool). Source: IR3 `BindLocal*`
    /// — the name pointer arg.
    LdrNamePtr(R, Name),
    /// `ldr r12, =BINDING; ldr dst, [r12]` — emitted as a pair by the
    /// final asm. Source: IR3 `LoadCapture`.
    LdrCapture(R, Binding),
    /// `ldr dst, =BINDING` — load the raw Binding cell pointer (no
    /// dereference). Used by `Call` to hand the helper a stable
    /// pointer to the captured slot, where the helper does its own
    /// type/closure check on the inner value.
    LdrCapturePtr(R, Binding),
    /// `ldr dst, =CALL_CACHE_n` — load the address of the per-call-site
    /// `CallCache` instance owned by the JitExecutor. The integer is
    /// the index into the executor's `call_caches` Vec. Helper reads
    /// the cell to fast-path closure resolution.
    LdrCallCachePtr(R, u32),
    /// `ldr r12, =BINDING; str src, [r12]`. Source: IR3 `StoreCapture`.
    StrCapture(Binding, R),
    /// `ldr dst, [base, #off]`. Sources: IR3 `Unbox` (payload offset),
    /// `Car` / `Cdr` (cons cell offsets), `Hits` (closure hits-counter
    /// offset), `SysGet32` (raw MMIO load with off=0).
    LdrOffset(R, R, i32),
    /// `str src, [base, #off]`. Source: IR3 `SysPut32` (raw MMIO
    /// store with off=0).
    StrOffset(R, R, i32),

    // ===== spill (regalloc-emitted) =====
    /// `ldr dst, [sp, #spill_offset]`. Emitted by spill rewrite for
    /// each use of a spilled vreg.
    LoadSpill(R, SpillSlot),
    /// `str src, [sp, #spill_offset]`. Emitted by spill rewrite for
    /// each def of a spilled vreg.
    StoreSpill(SpillSlot, R),

    // ===== direct ARM ALU (3-addr / 2-addr) =====
    // Sources: identical-name IR3 ops, pass through unchanged with
    // virtual-register operands.
    Add(R, R, R),
    Sub(R, R, R),
    /// ARMv6: `rd` must differ from `rm`. Post-coloring fixup pass
    /// inserts a Mov via R12 if regalloc happens to assign rd == rm.
    Mul(R, R, R),
    Lshift(R, R, R),
    Rshift(R, R, R),
    BinOr(R, R, R),
    BinAnd(R, R, R),
    /// `mvn dst, src` — bitwise NOT. Source: IR3 `BinNot`.
    Mvn(R, R),

    // ===== comparison + materialization =====
    /// `cmp a, b`. Sources: IR3 `Eq`/`Gt`/`Lt`/`Gte`/`Lte` (paired
    /// with `Cset`).
    Cmp(R, R),
    /// `cmp a, #imm`. Source: `CondBr` expansion (`cmp cond, #0`).
    CmpImm(R, ImmNumber),
    /// `cset dst, <cond>` — pseudo-op. Asm-emit lowers to
    /// `mov dst, #0 ; mov<cond> dst, #1`. Source: IR3 comparisons.
    Cset(R, Cond),

    // ===== helper-call opcodes =====
    // Each is one `bl <symbol>`. Args in r0–r3, return in r0,
    // clobbers caller-saved + flags. Source IR3 op listed per variant.
    /// `bl bind_local`. Source: IR3 `BindLocal`, `BindImmediate`.
    BindLocal,
    /// `bl load_local`. Source: IR3 `LoadLocal`.
    LoadLocal,
    /// `bl store_local`. Source: IR3 `StoreLocal`.
    StoreLocal,
    /// `bl unbox_local`. Source: IR3 `UnboxLocal`.
    UnboxLocal,
    /// `bl push_frame_trampoline`. Source: IR3 `PushFrame`.
    PushFrame,
    /// `bl pop_frame_trampoline`. Source: IR3 `PopFrame`.
    PopFrame,
    /// `bl box_number`. Source: IR3 `Box`.
    Box,
    /// `bl truthy`. Source: IR3 `Truthy`.
    Truthy,
    /// `bl lognot`. Source: IR3 `LogNot`.
    LogNot,
    /// `bl xor`. Source: IR3 `Xor`.
    Xor,
    /// `bl __divsi3`. Source: IR3 `Div`. (ARMv6 has no native sdiv.)
    Div,
    /// `bl __modsi3`. Source: IR3 `Mod`.
    Mod,
    /// `bl cons_alloc`. Source: IR3 `Cons`.
    Cons,
    /// `bl is_nil`. Source: IR3 `Nullp`.
    Nullp,
    /// `bl array_pack`. Source: IR3 `Array`.
    Array,
    /// `bl array_full`. Source: IR3 `Full`.
    Full,
    /// `bl array_unpack`. Source: IR3 `Unpack`.
    Unpack,
    /// `bl array_getidx`. Source: IR3 `GetIdx`.
    GetIdx,
    /// `bl array_putidx`. Source: IR3 `PutIdx`.
    PutIdx,
    /// `bl array_readidx`. Source: IR3 `ReadIdx`.
    ReadIdx,
    /// `bl array_fillidx`. Source: IR3 `FillIdx`.
    FillIdx,
    /// `bl array_fullidx`. Source: IR3 `FullIdx`.
    FullIdx,
    /// `bl hits_counter`. Source: IR3 `Hits` (if not inlined as
    /// `LdrOffset`; currently inlined — variant kept for future).
    Hits,
    /// `bl escape_to_interp`. Source: IR3 `Escape`.
    Escape,
    /// `bl call`. Variable-arity dispatch through a runtime helper.
    /// Calling convention:
    ///   r0 = binding cell pointer (`*const RefCell<Rc<Value>>`)
    ///   r1 = argc
    ///   r2 = argv (pointer to `[argc; u32]` of slot ids — the caller
    ///        stashes them in the dedicated call-args stack region
    ///        reserved by the prologue at the bottom of SP)
    /// Helper returns the result slot id in r0.
    ///
    /// The opcode itself is *generic* — the calling convention is just
    /// "binding + argv". The current `h_call` helper happens to specialize
    /// for Closure-valued bindings (fast-path JC dispatch, fallback to
    /// interpreter); a future helper could dispatch other callable
    /// shapes (Macro, foreign fn, etc.) over the same ABI.
    /// Source: IR3 `Call`.
    Call,
    /// `bl uart_init`. Source: IR3 `SysUartInit`.
    UartInit,
    /// `bl uart_get8`. Source: IR3 `SysUartGet8`.
    UartGet8,
    /// `bl uart_put8`. Source: IR3 `SysUartPut8`.
    UartPut8,
    /// `bl delay`. Source: IR3 `SysDelay`.
    Delay,
    /// `bl monitor_clear`. Source: IR3 `SysClearMonitor`.
    ClearMonitor,
    /// `bl monitor_get`. Source: IR3 `SysGetMonitor`.
    GetMonitor,
    /// `bl monitor_stop`. Source: IR3 `SysStopMonitor`.
    StopMonitor,
    /// `bl zero32`. Source: IR3 `SysZero32`.
    Zero32,
    /// Inline word copy. Source: IR3 `SysStr`.
    StrMem,
    /// `bl full32`. Source: IR3 `SysFull32`.
    Full32,

    // ===== inline asm (no helper) =====
    /// `dsb sy`. Source: IR3 `SysDsb`.
    Dsb,
    /// MMU prefetch flush (CP15 sequence). Source: IR3 `SysPrefetchFlush`.
    PrefetchFlush,

    // ===== stack frame =====
    //
    // ARMv6 multi-push/pop. The bitmask follows ARM's `ldm/stm`
    // register-list encoding: bit N corresponds to register N
    // (R0=bit 0, R1=bit 1, …, LR=bit 14, PC=bit 15).
    // Use `Register::bit()` to set bits.
    /// `push {regs}` — equivalent to `stmdb sp!, {regs}`. Source:
    /// function prologue (callee-save save list).
    StackPush(u16),
    /// `pop {regs}` — equivalent to `ldmia sp!, {regs}`. Source:
    /// function epilogue (callee-save restore list, often including
    /// `pc` for combined ret).
    StackPop(u16),

    // ===== control flow =====
    /// `b .Ln`. Source: IR3 `Br`.
    B(usize),
    /// `beq .Ln`. Source: `CondBr` expansion (jump to else-arm).
    Beq(usize),
    /// `bne .Ln`. Source: `CondBr` expansion (alternate form).
    Bne(usize),
    /// Function return. Asm: `bx lr` (or `pop {…, pc}` if combined
    /// with epilogue). Source: IR3 `Ret` (after Mov-into-r0 of result).
    Ret,

    // ===== pre-final forms (Instr<Operand> only) =====
    //
    // These should never appear in `Instr<Register>`. They're variants
    // of the generic enum so the operand-walking code is shared.
    /// Conditional branch with virtual cond. Expanded during
    /// `apply_colors` to `CmpImm + Beq + B`. Source: IR3 `CondBr`.
    CondBr {
        cond: R,
        then_blk: usize,
        else_blk: usize,
    },
    /// SSA phi. Destructed (replaced with `Mov`s at predecessor
    /// tails) before regalloc proper. Source: IR3 `PhiOp`.
    Phi(R, (usize, R), (usize, R)),
}

/// Canonical post-regalloc instruction.
#[allow(dead_code)]
pub(crate) type Instruction = Instr<Register>;

#[derive(Clone, Debug)]
pub(crate) struct LIRBasicBlock {
    pub instructions: Vec<Instruction>,
    /// Same lowering-contract semantics as `IRBasicBlock::dead`.
    pub dead: bool,
}

#[derive(Clone)]
pub(crate) struct LIRSegment {
    pub blocks: Vec<LIRBasicBlock>,
    /// Number of spill slots allocated; prologue reserves `4 * spill_slots`
    /// bytes of stack.
    #[allow(dead_code)]
    pub spill_slots: u32,
    /// Bitmask of `Register::POOL` indices actually assigned. Prologue
    /// saves exactly the corresponding callee-saves.
    #[allow(dead_code)]
    pub callee_saves_used: u32,
    /// Maximum arg count across all `Call` instructions in this segment.
    /// Prologue reserves `4 * call_args_max` bytes at the bottom of SP
    /// (offsets `0 .. call_args_max*4`); spill slots sit above this
    /// region. Each `Call` stashes its args into this area before the
    /// helper call and passes `sp` as the argv pointer.
    #[allow(dead_code)]
    pub call_args_max: u32,
    /// Number of distinct `Call` sites in the segment. The executor
    /// allocates this many `CallCache` cells; each `LdrCallCachePtr(_, i)`
    /// references one by index.
    #[allow(dead_code)]
    pub call_cache_count: u32,
}

// ===================== pretty-printer =====================

/// Format a register-list bitmask in ARM's `{r4, r5, lr}` style.
fn fmt_reglist(f: &mut fmt::Formatter<'_>, mask: u16) -> fmt::Result {
    f.write_str("{")?;
    let names = [
        "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "sp",
        "lr", "pc",
    ];
    let mut first = true;
    for i in 0..16 {
        if (mask >> i) & 1 == 1 {
            if !first {
                f.write_str(", ")?;
            }
            f.write_str(names[i])?;
            first = false;
        }
    }
    f.write_str("}")
}

/// Print one `Instr<R>` where `R: Display`. Used for both post-regalloc
/// (`Instr<Register>`) and the regalloc-internal (`Instr<Operand>`)
/// debug dumps.
pub(super) fn fmt_instr<R: fmt::Display>(f: &mut fmt::Formatter<'_>, i: &Instr<R>) -> fmt::Result {
    const W: usize = 7;
    macro_rules! mn {
        ($m:expr) => {
            write!(f, "{:<width$}", $m, width = W)
        };
    }
    macro_rules! bl {
        ($sym:expr) => {
            write!(f, "{:<width$}{}", "bl", $sym, width = W)
        };
    }
    match i {
        Instr::Mov(d, s) => {
            mn!("mov")?;
            write!(f, "{}, {}", d, s)
        }
        Instr::MovImm(d, n) => {
            mn!("mov")?;
            write!(f, "{}, #{}", d, n)
        }
        Instr::MovId(d, id) => {
            mn!("mov")?;
            write!(f, "{}, #${}", d, id.0)
        }
        Instr::LdrValuePtr(d, v) => {
            mn!("ldr=v")?;
            write!(f, "{}, #{}", d, v)
        }
        Instr::LdrNamePtr(d, n) => {
            mn!("ldr=n")?;
            write!(f, "{}, {:?}", d, n.as_str())
        }
        Instr::LdrCapture(d, b) => {
            mn!("ldr=c")?;
            write!(f, "{}, [#{:p}]", d, b.as_ref().as_ptr())
        }
        Instr::LdrCapturePtr(d, b) => {
            mn!("ldr=cp")?;
            write!(f, "{}, #{:p}", d, b.as_ref().as_ptr())
        }
        Instr::LdrCallCachePtr(d, idx) => {
            mn!("ldr=cc")?;
            write!(f, "{}, #cache[{}]", d, idx)
        }
        Instr::StrCapture(b, s) => {
            mn!("str=c")?;
            write!(f, "[#{:p}], {}", b.as_ref().as_ptr(), s)
        }
        Instr::LdrOffset(d, b, o) => {
            mn!("ldr")?;
            write!(f, "{}, [{}, #{}]", d, b, o)
        }
        Instr::StrOffset(s, b, o) => {
            mn!("str")?;
            write!(f, "{}, [{}, #{}]", s, b, o)
        }

        Instr::LoadSpill(d, s) => {
            mn!("ldspl")?;
            write!(f, "{}, {}", d, s)
        }
        Instr::StoreSpill(s, r) => {
            mn!("stspl")?;
            write!(f, "{}, {}", s, r)
        }

        Instr::Add(d, a, b) => {
            mn!("add")?;
            write!(f, "{}, {}, {}", d, a, b)
        }
        Instr::Sub(d, a, b) => {
            mn!("sub")?;
            write!(f, "{}, {}, {}", d, a, b)
        }
        Instr::Mul(d, a, b) => {
            mn!("mul")?;
            write!(f, "{}, {}, {}", d, a, b)
        }
        Instr::Lshift(d, a, b) => {
            mn!("lsl")?;
            write!(f, "{}, {}, {}", d, a, b)
        }
        Instr::Rshift(d, a, b) => {
            mn!("lsr")?;
            write!(f, "{}, {}, {}", d, a, b)
        }
        Instr::BinOr(d, a, b) => {
            mn!("orr")?;
            write!(f, "{}, {}, {}", d, a, b)
        }
        Instr::BinAnd(d, a, b) => {
            mn!("and")?;
            write!(f, "{}, {}, {}", d, a, b)
        }
        Instr::Mvn(d, a) => {
            mn!("mvn")?;
            write!(f, "{}, {}", d, a)
        }

        Instr::Cmp(a, b) => {
            mn!("cmp")?;
            write!(f, "{}, {}", a, b)
        }
        Instr::CmpImm(a, n) => {
            mn!("cmp")?;
            write!(f, "{}, #{}", a, n)
        }
        Instr::Cset(d, c) => {
            mn!("cset")?;
            write!(f, "{}, {}", d, c.suffix())
        }

        Instr::BindLocal => bl!("bind_local"),
        Instr::LoadLocal => bl!("load_local"),
        Instr::StoreLocal => bl!("store_local"),
        Instr::UnboxLocal => bl!("unbox_local"),
        Instr::PushFrame => bl!("push_frame"),
        Instr::PopFrame => bl!("pop_frame"),
        Instr::Box => bl!("box_number"),
        Instr::Truthy => bl!("truthy"),
        Instr::LogNot => bl!("lognot"),
        Instr::Xor => bl!("xor"),
        Instr::Div => bl!("__divsi3"),
        Instr::Mod => bl!("__modsi3"),
        Instr::Cons => bl!("cons_alloc"),
        Instr::Nullp => bl!("is_nil"),
        Instr::Array => bl!("array_pack"),
        Instr::Full => bl!("array_full"),
        Instr::Unpack => bl!("array_unpack"),
        Instr::GetIdx => bl!("array_getidx"),
        Instr::PutIdx => bl!("array_putidx"),
        Instr::ReadIdx => bl!("array_readidx"),
        Instr::FillIdx => bl!("array_fillidx"),
        Instr::FullIdx => bl!("array_fullidx"),
        Instr::Hits => bl!("hits_counter"),
        Instr::Escape => bl!("escape_to_interp"),
        // `Call` is a single LIR opcode but the asm emit expands it to
        // 8 words: inline cache check + bl h_call_fast / bl h_call.
        // The printed form makes the expansion visible so the LIR dump
        // doesn't mislead readers into thinking it's a single `bl`.
        Instr::Call => write!(
            f,
            "{:<width$}cache[r3]?fast:slow  ; 8w: ldr r12,[r3]; cmp; beq Ls; \
             bl h_call_fast; b Ld; Ls: bl h_call; Ld:",
            "call*",
            width = W
        ),
        Instr::UartInit => bl!("uart_init"),
        Instr::UartGet8 => bl!("uart_get8"),
        Instr::UartPut8 => bl!("uart_put8"),
        Instr::Delay => bl!("delay"),
        Instr::ClearMonitor => bl!("monitor_clear"),
        Instr::GetMonitor => bl!("monitor_get"),
        Instr::StopMonitor => bl!("monitor_stop"),
        Instr::Zero32 => bl!("zero32"),
        Instr::StrMem => mn!("@str"),
        Instr::Full32 => bl!("full32"),

        Instr::Dsb => mn!("dsb sy"),
        Instr::PrefetchFlush => mn!("@pfflsh"),

        Instr::StackPush(m) => {
            mn!("push")?;
            fmt_reglist(f, *m)
        }
        Instr::StackPop(m) => {
            mn!("pop")?;
            fmt_reglist(f, *m)
        }

        Instr::B(t) => {
            mn!("b")?;
            write!(f, ".L{}", t)
        }
        Instr::Beq(t) => {
            mn!("beq")?;
            write!(f, ".L{}", t)
        }
        Instr::Bne(t) => {
            mn!("bne")?;
            write!(f, ".L{}", t)
        }
        Instr::Ret => write!(f, "{:<width$}lr", "bx", width = W),

        Instr::CondBr {
            cond,
            then_blk,
            else_blk,
        } => {
            mn!("cbr")?;
            write!(f, "{}, .L{}, .L{}", cond, then_blk, else_blk)
        }
        Instr::Phi(d, (ab, av), (bb, bv)) => {
            mn!("phi")?;
            write!(f, "{}, [.L{}: {}], [.L{}: {}]", d, ab, av, bb, bv)
        }
    }
}

impl fmt::Debug for LIRSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, block) in self.blocks.iter().enumerate() {
            if block.dead {
                writeln!(f, ".L{} (Dead):", i)?;
                continue;
            }
            writeln!(f, ".L{}:", i)?;
            for instr in &block.instructions {
                write!(f, "    ")?;
                fmt_instr(f, instr)?;
                writeln!(f)?;
            }
        }
        if self.spill_slots > 0 {
            writeln!(f, "; spill slots: {}", self.spill_slots)?;
        }
        if self.callee_saves_used != 0 {
            write!(f, "; callee-saves used:")?;
            for r in Register::POOL {
                if let Some(i) = r.pool_index() {
                    if (self.callee_saves_used >> i) & 1 == 1 {
                        write!(f, " {}", r)?;
                    }
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
