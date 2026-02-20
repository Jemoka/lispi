#![no_std]
#![no_main]
#![feature(sync_unsafe_cell)]

// os stuff
mod boot;
mod comm;
mod regs;
mod threading;
mod utils;

// lisp stuff
// mod language; // UNCOMMENT WHEN BACK TO LISPI

extern crate alloc;

use regs::cpsr::Mode;
use regs::banked::{BankedRegs};


fn main() {
    // let regs = BankedRegisters::new(8,4);
    // regs.set(Mode::System);
    // let newregs = BankedRegisters::get(Mode::FIQ);
    let b = BankedRegs::get(Mode::FIQ);
    println!("h {:#x?} {:#x?}", b.lr, b.sp);

    BankedRegs::new(8,9).set(Mode::FIQ);

    let q = BankedRegs::get(Mode::FIQ);
    println!("h {:?} {:?}", q.lr, q.sp);


    // println!("stakc pointer: {:?} {:?}", newregs.lr, newregs.sp);
}
