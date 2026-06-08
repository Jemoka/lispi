#![allow(dead_code)]
//! ARMv6 instruction encodings.
//!
//! Each `pub(crate) fn` returns the 32-bit ARM instruction word(s) for
//! one named operation. The encoding fields are laid out per the
//! "ARM Architecture Reference Manual, ARMv6 Edition" (the page
//! numbers in comments below refer to that document).
//!
//! Conventions:
//!   * All functions take typed `Register` operands (variants are
//!     discriminant-ordered so `as u32` yields the canonical
//!     register number 0..15).
//!   * Default condition for unconditional variants is `cond_AL`
//!     (always). Conditional variants take an explicit `Cond` arg
//!     (see [`COND_EQ`], etc.).
//!   * Immediate-form helpers validate that the 8-bit literal fits
//!     and that the rotate is even (ARM rotates are encoded as
//!     `rot4 = actual_rotate / 2`). Out-of-range arguments panic —
//!     this is JIT-internal, so panicking on a bug is the right
//!     failure mode.
//!   * Synthetic sequences (e.g. `load_imm32`) write multiple
//!     instructions into a `Vec<u32>`.
//!
//! # What's covered
//!
//! Data-processing (ADD/SUB/AND/EOR/ORR/MOV/MVN/CMP — both immediate
//! and register forms), shifts (LSL/LSR/ASR/ROR as MOV-with-shift),
//! multiply (MUL, MLA), load/store with 12-bit offset (signed via
//! the U flag), block load/store (LDM/STM family — pre/post,
//! up/down, writeback combinations), branch (B, BL, conditional B),
//! branch-and-exchange (BX), CP15 MCR (DSB barrier on ARMv6).
//!
//! Note: ARMv6 has no native `sdiv`/`udiv` and no `movw`/`movt`.
//! `load_imm32` builds a 32-bit constant via MOV+ORR rotations.

use alloc::vec::Vec;

use super::ir4::Register;

// ===================== conditions (bits 31:28) =====================

pub(crate) const COND_EQ: u32 = 0b0000;
pub(crate) const COND_NE: u32 = 0b0001;
pub(crate) const COND_CS: u32 = 0b0010;
pub(crate) const COND_CC: u32 = 0b0011;
pub(crate) const COND_MI: u32 = 0b0100;
pub(crate) const COND_PL: u32 = 0b0101;
pub(crate) const COND_VS: u32 = 0b0110;
pub(crate) const COND_VC: u32 = 0b0111;
pub(crate) const COND_HI: u32 = 0b1000;
pub(crate) const COND_LS: u32 = 0b1001;
pub(crate) const COND_GE: u32 = 0b1010;
pub(crate) const COND_LT: u32 = 0b1011;
pub(crate) const COND_GT: u32 = 0b1100;
pub(crate) const COND_LE: u32 = 0b1101;
pub(crate) const COND_AL: u32 = 0b1110;

// ===================== data-processing opcodes (bits 24:21) =====================

pub(crate) const OP_AND: u32 = 0b0000;
pub(crate) const OP_EOR: u32 = 0b0001;
pub(crate) const OP_SUB: u32 = 0b0010;
pub(crate) const OP_RSB: u32 = 0b0011;
pub(crate) const OP_ADD: u32 = 0b0100;
pub(crate) const OP_ADC: u32 = 0b0101;
pub(crate) const OP_SBC: u32 = 0b0110;
pub(crate) const OP_RSC: u32 = 0b0111;
pub(crate) const OP_TST: u32 = 0b1000;
pub(crate) const OP_TEQ: u32 = 0b1001;
pub(crate) const OP_CMP: u32 = 0b1010;
pub(crate) const OP_CMN: u32 = 0b1011;
pub(crate) const OP_ORR: u32 = 0b1100;
pub(crate) const OP_MOV: u32 = 0b1101;
pub(crate) const OP_BIC: u32 = 0b1110;
pub(crate) const OP_MVN: u32 = 0b1111;

// ===================== shift types (bits 6:5 of register-shift) =====================

pub(crate) const SHIFT_LSL: u32 = 0b00;
pub(crate) const SHIFT_LSR: u32 = 0b01;
pub(crate) const SHIFT_ASR: u32 = 0b10;
pub(crate) const SHIFT_ROR: u32 = 0b11;

// ===================== register helper =====================

/// Canonical ARM register number 0..15. `Register` is laid out so
/// `as u32` returns the right number.
#[inline]
fn rn(r: Register) -> u32 {
    let n = r as u32;
    debug_assert!(n < 16, "register out of range");
    n
}

// ===================== data-processing: register form =====================
//
// Format (A5-2): cond(31:28) 000(27:25) opcode(24:21) S(20)
//                 Rn(19:16) Rd(15:12) shift_imm(11:7) shift(6:5)
//                 0(4) Rm(3:0)
//
// "S" sets condition flags (required for CMP/CMN/TST/TEQ; optional
// for the other ALU ops).

#[inline]
fn dp_reg(cond: u32, opcode: u32, set_flags: bool, rd: Register, rn_reg: Register, rm: Register) -> u32 {
    (cond << 28)
        | (opcode << 21)
        | ((set_flags as u32) << 20)
        | (rn(rn_reg) << 16)
        | (rn(rd) << 12)
        | rn(rm)
}

pub(crate) fn mov(rd: Register, rm: Register) -> u32 {
    // Rn field unused for MOV — encoded as 0. Use R0 as a "don't care".
    dp_reg(COND_AL, OP_MOV, false, rd, Register::R0, rm)
}

pub(crate) fn mvn(rd: Register, rm: Register) -> u32 {
    dp_reg(COND_AL, OP_MVN, false, rd, Register::R0, rm)
}

pub(crate) fn add(rd: Register, rn: Register, rm: Register) -> u32 {
    dp_reg(COND_AL, OP_ADD, false, rd, rn, rm)
}

pub(crate) fn sub(rd: Register, rn: Register, rm: Register) -> u32 {
    dp_reg(COND_AL, OP_SUB, false, rd, rn, rm)
}

/// `add rd, rn, rm, lsl #shift` — register form with an immediate
/// shift in the operand2 field. Format (A5-8):
///   cond(31:28) 000(27:25) opcode(24:21) S(20) Rn(19:16) Rd(15:12)
///   shift_imm(11:7) shift_type(6:5) 0(4) Rm(3:0)
pub(crate) fn add_lsl_imm(rd: Register, rn_reg: Register, rm: Register, shift_imm5: u32) -> u32 {
    debug_assert!(shift_imm5 < 32);
    (COND_AL << 28)
        | (OP_ADD << 21)
        | (rn(rn_reg) << 16)
        | (rn(rd) << 12)
        | (shift_imm5 << 7)
        | (SHIFT_LSL << 5)
        | rn(rm)
}

/// `mov rd, rm, lsl #shift`.
pub(crate) fn mov_lsl_imm(rd: Register, rm: Register, shift_imm5: u32) -> u32 {
    debug_assert!(shift_imm5 < 32);
    (COND_AL << 28)
        | (OP_MOV << 21)
        | (rn(rd) << 12)
        | (shift_imm5 << 7)
        | (SHIFT_LSL << 5)
        | rn(rm)
}

pub(crate) fn orr(rd: Register, rn: Register, rm: Register) -> u32 {
    dp_reg(COND_AL, OP_ORR, false, rd, rn, rm)
}

pub(crate) fn and(rd: Register, rn: Register, rm: Register) -> u32 {
    dp_reg(COND_AL, OP_AND, false, rd, rn, rm)
}

pub(crate) fn eor(rd: Register, rn: Register, rm: Register) -> u32 {
    dp_reg(COND_AL, OP_EOR, false, rd, rn, rm)
}

/// `cmp rn, rm` — sets flags. Rd field is unused (encoded as 0).
pub(crate) fn cmp(rn_reg: Register, rm: Register) -> u32 {
    (COND_AL << 28)
        | (OP_CMP << 21)
        | (1u32 << 20)               // S=1 mandatory
        | (rn(rn_reg) << 16)
        | rn(rm)
}

// ===================== data-processing: register form with shift =====================
//
// MOV with a register-specified shift is how ARMv6 encodes LSL/LSR/
// ASR/ROR. Register-shift format (A5-9):
//   cond(31:28) 000(27:25) opcode(24:21) S(20) Rn(19:16) Rd(15:12)
//   Rs(11:8) 0(7) shift_type(6:5) 1(4) Rm(3:0)

#[inline]
fn dp_reg_shift_by_reg(opcode: u32, shift_type: u32, rd: Register, rm: Register, rs: Register) -> u32 {
    (COND_AL << 28)
        | (opcode << 21)
        | (rn(rd) << 12)
        | (rn(rs) << 8)
        | (shift_type << 5)
        | (1u32 << 4)                 // register-specified shift
        | rn(rm)
}

/// `mov rd, rm, lsl rs`.
pub(crate) fn lsl(rd: Register, rm: Register, rs: Register) -> u32 {
    dp_reg_shift_by_reg(OP_MOV, SHIFT_LSL, rd, rm, rs)
}

/// `mov rd, rm, lsr rs`.
pub(crate) fn lsr(rd: Register, rm: Register, rs: Register) -> u32 {
    dp_reg_shift_by_reg(OP_MOV, SHIFT_LSR, rd, rm, rs)
}

/// `mov rd, rm, asr rs`.
pub(crate) fn asr(rd: Register, rm: Register, rs: Register) -> u32 {
    dp_reg_shift_by_reg(OP_MOV, SHIFT_ASR, rd, rm, rs)
}

// ===================== data-processing: immediate form =====================
//
// Format (A5-3):
//   cond(31:28) 001(27:25) opcode(24:21) S(20) Rn(19:16) Rd(15:12)
//   rotate(11:8) imm8(7:0)
//
// The effective immediate is `imm8 ROR (2 * rotate)`. We expose
// `rot4` as the half (matching the C reference and the user's
// canonical names) — i.e. callers pass the *encoded* rotate value
// 0..15, and the actual rotation applied is `2 * rot4`.

fn check_imm8(imm8: u32) {
    assert!(imm8 < 0x100, "immediate {:#x} does not fit in 8 bits", imm8);
}

fn check_rot4(rot4: u32) {
    assert!(rot4 < 16, "rotate {} does not fit in 4 bits", rot4);
}

#[inline]
fn dp_imm(cond: u32, opcode: u32, set_flags: bool, rd: Register, rn_reg: Register, imm8: u32, rot4: u32) -> u32 {
    check_imm8(imm8);
    check_rot4(rot4);
    (cond << 28)
        | (1u32 << 25)                // I=1 (immediate)
        | (opcode << 21)
        | ((set_flags as u32) << 20)
        | (rn(rn_reg) << 16)
        | (rn(rd) << 12)
        | (rot4 << 8)
        | imm8
}

/// `mov rd, #imm8, ror (2*rot4)`. Conditional variant.
pub(crate) fn mov_imm8_cond_rot(cond: u32, rd: Register, imm8: u32, rot4: u32) -> u32 {
    dp_imm(cond, OP_MOV, false, rd, Register::R0, imm8, rot4)
}

pub(crate) fn mov_imm8_rot(rd: Register, imm8: u32, rot4: u32) -> u32 {
    mov_imm8_cond_rot(COND_AL, rd, imm8, rot4)
}

pub(crate) fn mov_imm8(rd: Register, imm8: u32) -> u32 {
    mov_imm8_rot(rd, imm8, 0)
}

pub(crate) fn mvn_imm8(rd: Register, imm8: u32) -> u32 {
    dp_imm(COND_AL, OP_MVN, false, rd, Register::R0, imm8, 0)
}

pub(crate) fn orr_imm8_rot(rd: Register, rn_reg: Register, imm8: u32, rot4: u32) -> u32 {
    dp_imm(COND_AL, OP_ORR, false, rd, rn_reg, imm8, rot4)
}

pub(crate) fn orr_imm8(rd: Register, rn_reg: Register, imm8: u32) -> u32 {
    orr_imm8_rot(rd, rn_reg, imm8, 0)
}

pub(crate) fn add_imm8(rd: Register, rn_reg: Register, imm8: u32) -> u32 {
    dp_imm(COND_AL, OP_ADD, false, rd, rn_reg, imm8, 0)
}

pub(crate) fn sub_imm8(rd: Register, rn_reg: Register, imm8: u32) -> u32 {
    dp_imm(COND_AL, OP_SUB, false, rd, rn_reg, imm8, 0)
}

pub(crate) fn and_imm8(rd: Register, rn_reg: Register, imm8: u32) -> u32 {
    dp_imm(COND_AL, OP_AND, false, rd, rn_reg, imm8, 0)
}

/// `cmp rn, #imm8` — sets flags.
pub(crate) fn cmp_imm8(rn_reg: Register, imm8: u32) -> u32 {
    check_imm8(imm8);
    (COND_AL << 28)
        | (1u32 << 25)
        | (OP_CMP << 21)
        | (1u32 << 20)                // S=1
        | (rn(rn_reg) << 16)
        | imm8
}

// ===================== multiply =====================
//
// MUL (A4-80): cond(31:28) 0000000(27:21) S(20) Rd(19:16) 0000(15:12)
//              Rs(11:8) 1001(7:4) Rm(3:0)
// Important constraint: Rd MUST NOT equal Rm.
//
// MLA (A4-66): cond(31:28) 0000001(27:21) S(20) Rd(19:16) Rn(15:12)
//              Rs(11:8) 1001(7:4) Rm(3:0)
//              rd = rm * rs + rn.

pub(crate) fn mul(rd: Register, rm: Register, rs: Register) -> u32 {
    debug_assert!(rd != rm, "ARMv6 mul requires Rd != Rm (got {:?})", rd);
    (COND_AL << 28)
        | (0b000_0000 << 21)
        | (rn(rd) << 16)
        | (rn(rs) << 8)
        | (0b1001 << 4)
        | rn(rm)
}

pub(crate) fn mla(rd: Register, rm: Register, rs: Register, rn_acc: Register) -> u32 {
    (COND_AL << 28)
        | (0b000_0001 << 21)
        | (rn(rd) << 16)
        | (rn(rn_acc) << 12)
        | (rn(rs) << 8)
        | (0b1001 << 4)
        | rn(rm)
}

// ===================== load / store with 12-bit offset =====================
//
// Format (A5-18):
//   cond(31:28) 01(27:26) I(25) P(24) U(23) B(22) W(21) L(20)
//   Rn(19:16) Rd(15:12) addr(11:0)
//
// For our use:
//   * I=0 (immediate offset, not register).
//   * P=1, W=0 (offset addressing — no writeback, no post-index).
//   * B=0 (word, not byte).
//   * L=1 for load, L=0 for store.
//   * U=1 for positive offset (added), U=0 for negative (subtracted);
//     the 12-bit field is always the unsigned magnitude.

fn ldr_str_off12(load: bool, rd_or_rs: Register, rn_base: Register, offset: i32) -> u32 {
    let (u_bit, abs) = if offset >= 0 {
        (1u32, offset as u32)
    } else {
        (0u32, (-offset) as u32)
    };
    assert!(abs < (1 << 12), "ldr/str offset {} doesn't fit in 12 bits", offset);
    (COND_AL << 28)
        | (0b01 << 26)
        | (0u32 << 25)                // I=0 (immediate)
        | (1u32 << 24)                // P=1 (offset addressing)
        | (u_bit << 23)
        | (0u32 << 22)                // B=0 (word)
        | (0u32 << 21)                // W=0 (no writeback)
        | ((load as u32) << 20)
        | (rn(rn_base) << 16)
        | (rn(rd_or_rs) << 12)
        | (abs & 0xFFF)
}

pub(crate) fn ldr_off12(rd: Register, rn_base: Register, offset: i32) -> u32 {
    ldr_str_off12(true, rd, rn_base, offset)
}

pub(crate) fn str_off12(rs: Register, rn_base: Register, offset: i32) -> u32 {
    ldr_str_off12(false, rs, rn_base, offset)
}

// ===================== block load / store (LDM/STM) =====================
//
// Format (A5-44):
//   cond(31:28) 100(27:25) P(24) U(23) S(22) W(21) L(20)
//   Rn(19:16) register_list(15:0)
//
// `register_list` bit N selects register N (R0..R15).
//
// Common forms:
//   * `push {regs}`  ≡ `stmdb sp!, {regs}` : P=1 U=0 S=0 W=1 L=0
//   * `pop  {regs}`  ≡ `ldmia sp!, {regs}` : P=0 U=1 S=0 W=1 L=1

fn ldm_stm(load: bool, p: bool, u: bool, w: bool, rn_base: Register, reglist: u16) -> u32 {
    (COND_AL << 28)
        | (0b100 << 25)
        | ((p as u32) << 24)
        | ((u as u32) << 23)
        | (0u32 << 22)                // S=0 (no exception-mode regs)
        | ((w as u32) << 21)
        | ((load as u32) << 20)
        | (rn(rn_base) << 16)
        | (reglist as u32)
}

/// `push {regs}` — stmdb sp!, {regs}.
pub(crate) fn push(reglist: u16) -> u32 {
    ldm_stm(false, true, false, true, Register::SP, reglist)
}

/// `pop {regs}` — ldmia sp!, {regs}.
pub(crate) fn pop(reglist: u16) -> u32 {
    ldm_stm(true, false, true, true, Register::SP, reglist)
}

// ===================== branches =====================
//
// B / BL (A4-10):
//   cond(31:28) 101(27:25) L(24) signed_imm24(23:0)
//
// The 24-bit immediate is a signed word offset from PC+8.
// `target = (current_addr + 8) + (sign_extend(imm24) << 2)`.

/// `b<cond> .Ltarget` where `byte_offset_from_pc8` is the byte
/// distance from the **PC-plus-8** (i.e. `branch_addr + 8`) to the
/// target instruction. Must be a multiple of 4 and fit in a signed
/// 26-bit range.
pub(crate) fn b_cond(cond: u32, byte_offset_from_pc8: i32) -> u32 {
    assert!(byte_offset_from_pc8 % 4 == 0, "branch offset must be word-aligned");
    let word_off = byte_offset_from_pc8 >> 2;
    let min = -(1 << 23);
    let max = (1 << 23) - 1;
    assert!(word_off >= min && word_off <= max, "branch offset out of range");
    let imm24 = (word_off as u32) & 0x00FF_FFFF;
    (cond << 28)
        | (0b101 << 25)
        | (0u32 << 24)                // L=0 (B, not BL)
        | imm24
}

pub(crate) fn b(byte_offset_from_pc8: i32) -> u32 {
    b_cond(COND_AL, byte_offset_from_pc8)
}

pub(crate) fn bl(byte_offset_from_pc8: i32) -> u32 {
    assert!(byte_offset_from_pc8 % 4 == 0, "branch offset must be word-aligned");
    let word_off = byte_offset_from_pc8 >> 2;
    let imm24 = (word_off as u32) & 0x00FF_FFFF;
    (COND_AL << 28)
        | (0b101 << 25)
        | (1u32 << 24)                // L=1 (BL)
        | imm24
}

// ===================== BX =====================
//
// BX (A4-20):
//   cond(31:28) 0001 0010(27:20) (1111 1111 1111)(19:8) 0001(7:4) Rm(3:0)

pub(crate) fn bx(rm: Register) -> u32 {
    (COND_AL << 28)
        | (0b0001_0010 << 20)
        | (0xFFF << 8)
        | (0b0001 << 4)
        | rn(rm)
}

// BLX <Rm>: like BX but also writes (next-instr-addr) to LR — a true
// register-indirect call. Encoding (A4-16):
//   cond(31:28) 0001 0010(27:20) (1111 1111 1111)(19:8) 0011(7:4) Rm(3:0)
pub(crate) fn blx(rm: Register) -> u32 {
    (COND_AL << 28)
        | (0b0001_0010 << 20)
        | (0xFFF << 8)
        | (0b0011 << 4)
        | rn(rm)
}

// ===================== conditional move-immediate (for Cset) =====================
//
// Materializing a bool from a condition flag is a two-instruction
// sequence on ARMv6 (no native `cset`):
//
//   mov   rd, #0                  (always)
//   mov<cond>  rd, #1             (conditional — set when <cond> holds)
//
// Returns both instructions. Caller pushes both into the code buffer.

pub(crate) fn cset(rd: Register, cond: u32) -> [u32; 2] {
    [
        mov_imm8(rd, 0),
        mov_imm8_cond_rot(cond, rd, 1, 0),
    ]
}

// ===================== synthetic: 32-bit immediate load =====================
//
// ARMv6 has no `movw`/`movt`. We materialize an arbitrary 32-bit
// constant in 4 instructions: a mov of the low byte, plus three
// ORR-immediates rotating the next byte into position. (Each ARM
// immediate has 8 bits of value rotated by an even amount; we use
// rotates 0, 24, 16, 8 — i.e. rot4 = 0, 12, 8, 4 in the encoded
// field — to plant bytes at offsets 0, 8, 16, 24.)
//
// This is the standard ARMv6 idiom for big constants; the encoder
// can pick a tighter sequence when the value happens to fit a single
// ARM rotated immediate, but this 4-instruction baseline is always
// safe.

pub(crate) fn load_imm32(buf: &mut Vec<u32>, rd: Register, imm32: u32) {
    let b0 = imm32 & 0xFF;
    let b1 = (imm32 >> 8) & 0xFF;
    let b2 = (imm32 >> 16) & 0xFF;
    let b3 = (imm32 >> 24) & 0xFF;
    buf.push(mov_imm8_rot(rd, b0, 0));
    buf.push(orr_imm8_rot(rd, rd, b1, 12));   // rotate-right 24 bits
    buf.push(orr_imm8_rot(rd, rd, b2, 8));    // rotate-right 16 bits
    buf.push(orr_imm8_rot(rd, rd, b3, 4));    // rotate-right  8 bits
}

// ===================== PC-relative load (literal pool) =====================
//
// `ldr rd, [pc, #offset]` is the standard way to load a 32-bit
// constant from the function-local literal pool. ARMv6 lacks
// `movw`/`movt`, so for any constant the JIT can't reach via
// `mov_imm8 + orr` rotations (or wants to avoid the 4-instr
// sequence), it parks the word in the literal pool and emits a
// PC-relative load.
//
// PC at execution time is `current_addr + 8`. Positive offsets are
// the common case (literal pool sits at the end of the function).

pub(crate) fn ldr_pc_rel(rd: Register, byte_offset_from_pc8: i32) -> u32 {
    ldr_str_off12(true, rd, Register::PC, byte_offset_from_pc8)
}

// ===================== CP15 — DSB on ARMv6 =====================
//
// ARMv6 has no dedicated `dsb` instruction; the canonical sync
// barrier is `mcr p15, 0, Rd, c7, c10, 4`. The Rd value is ignored
// by the coprocessor — we encode R0.
//
// MCR format:
//   cond(31:28) 1110(27:24) opcode_1(23:21) 0(20) CRn(19:16)
//   Rd(15:12) coproc(11:8) opcode_2(7:5) 1(4) CRm(3:0)
//
// For DSB: cond=AL, opcode_1=0, CRn=7, Rd=0, coproc=15, opcode_2=4,
// CRm=10 — bit-pattern `0xee07_0f9a`.

pub(crate) const DSB_SY: u32 = 0xee07_0f9a;
