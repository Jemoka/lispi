//! Banked Register Operations
//! Getting, moving, and messing with banked registers

use crate::{prefetch_flush};
use super::cpsr::{Mode};

////// user SP/LR get and set functions //////
#[unsafe(naked)]
pub extern "C" fn mode_get_lr_sp_asm(mode: Mode, out: &mut [usize; 2]) {
    core::arch::naked_asm!(
        "mrs r2, cpsr",
        "msr cpsr_c, r0",
        prefetch_flush!(r3),
        "stm r1, {{r13, r14}}", // lr, sp
        "msr cpsr_c, r2",
        prefetch_flush!(r3),
        "bx lr"
    );
}

/// switch to a given privileged mode to get banked lr,sp
pub fn mode_get_lr_sp(mode: Mode) -> BankedRegs {
    let mut out = [0; 2];
    mode_get_lr_sp_asm(mode, &mut out);
    BankedRegs {
        lr: out[1],
        sp: out[0],
    }
}

#[unsafe(naked)]
pub extern "C" fn mode_set_lr_sp_asm(mode: Mode, inp: &[usize; 2]) {
    core::arch::naked_asm!(
        "mrs r2, cpsr",
        "msr cpsr_c, r0",
        prefetch_flush!(r3),
        "ldm r1, {{r13, r14}}", // lr, sp
        "msr cpsr_c, r2",
        prefetch_flush!(r3),
        "bx lr"
    );
}

/// switch to a given privileged mode to set
pub fn mode_set_lr_sp(mode: Mode, regs: BankedRegs) {
    let in_arr = [regs.sp, regs.lr];
    mode_set_lr_sp_asm(mode, &in_arr);
}

#[derive(Clone, Copy)]
pub struct BankedRegs {
    pub lr: usize,
    pub sp: usize,
}

impl BankedRegs {
    pub fn new(lr: usize, sp: usize) -> Self {
        BankedRegs { lr, sp }
    }

    pub fn get(mode: Mode) -> Self {
        mode_get_lr_sp(mode)
    }

    pub fn set(&self, mode: Mode) {
        mode_set_lr_sp(mode, *self);
    }
}

/// stick all registers onto the stack and return a pointer to them
#[unsafe(naked)]
pub extern "C" fn get_registers_asm() -> Regs {
    core::arch::naked_asm!(
        "stmfd sp!, {{r0-r15}}",
        "mov r0, sp",
    );
}

#[unsafe(naked)]
pub extern "C" fn set_registers_asm(r: &Regs) {
    core::arch::naked_asm!(
        "ldmfd r0!, {{r0-r15}}"
    );
}

#[repr(C)]
pub struct Regs(pub [usize; 16]);

