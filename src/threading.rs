//! threading management system

use heapless::Vec;
use core::cell::SyncUnsafeCell;

use crate::println;
use crate::regs::banked::{get_registers_asm, set_registers_asm, Regs};
use crate::regs::cpsr::{PSR, Mode};

const MAX_THREADS: usize = 16;
const DEFAULT_STACK_SIZE: usize = 1024;

#[repr(align(8))]
struct Stack(pub Vec<u32, DEFAULT_STACK_SIZE>);

struct ThreadControlBlock {
    stack: Stack,
    regs: [usize; 16], // All Registers
    cpsr: u32,
    dead: bool,
}

impl ThreadControlBlock {
    fn default() -> Self {
        let mut m = Self {
            stack: Stack(Vec::new()),
            regs: [
                0, // r0
                0, // r1
                0, // r2
                0, // r3
                0, // r4
                0, // r5
                0, // r6
                0, // r7
                0, // r8
                0, // r9
                0, // r10
                0, // r11
                0, // r12
                0, // sp
                0, // lr
                (thread_start_execution) as *const () as usize, // pc
            ],
            dead: false,
            cpsr: PSR::from(0).with_mode(Mode::User).into()
        };
        m
    }
}

static THREAD_CONTROL_BLOCK: SyncUnsafeCell<Vec<ThreadControlBlock, MAX_THREADS>> = SyncUnsafeCell::new(Vec::new());
static CURRENT_THREAD: SyncUnsafeCell<Option<u32>> = SyncUnsafeCell::new(None);

/// create a new thread and push it to the thread control block
pub fn thread_push(entrypoint: extern "C" fn()) {
    // if no one is the curret thread, set us to it
    if unsafe { (*CURRENT_THREAD.get()).is_none() } {
        unsafe {
            *CURRENT_THREAD.get() = Some(0);
        }
    }
    
    let mut tcb = ThreadControlBlock::default();
    // because the thread will start executing at the entrypoint, we need to set the pc to the entrypoint
    // once it returns we want it to jump to the thread end trampoline
    tcb.regs[0] = entrypoint as usize;
    unsafe {
        let v = &mut *THREAD_CONTROL_BLOCK.get();
        v.push(tcb).unwrap();
        // TODO!!! get id, move into final storage FIRST, and then
        // set up registers + stack 



        tcb.regs[1] = (&*THREAD_CONTROL_BLOCK.get()).len();
    }
}

/// called when threads begin execution
extern "C" fn thread_start_execution(entrypoint: extern "C" fn(), id: usize) {
    println!("BEGIN THREAD");
    entrypoint();
    thread_end_trampoline(id);
}

/// called when threads finish execution
extern "C" fn thread_end_trampoline(id: usize) {
    println!("END THREAD");
    unsafe {
        (&mut *THREAD_CONTROL_BLOCK.get())[id].dead = true;
    }

    // if everything is dead, print and panic
    let all_dead = unsafe {
        let tcb = &*THREAD_CONTROL_BLOCK.get();
        tcb.iter().all(|t| t.dead)
    };
    if all_dead {
        println!("ALL THREADS DEAD");
        panic!("ALL THREADS DEAD");
    }

    // context switch to the next thread
    thread_context_switch();
}


/// context switch
extern "C" fn thread_context_switch() {
    let regs = get_registers_asm();
    // after this point we can't trust any of our pointers
    // because the rest of this is not a naked function and thus
    // nukes the fuck out of PC, LD, SP, etc.
    unsafe {
        let id = (*CURRENT_THREAD.get()).unwrap() as usize;
        let cur = &mut *THREAD_CONTROL_BLOCK.get();
        cur[id].regs = regs.0;
        cur[id].cpsr = PSR::get_spsr().into();
    }
    let (next_regs, cpsr) = unsafe {
        let tcb = &mut *THREAD_CONTROL_BLOCK.get();
        let mut next_id = (*CURRENT_THREAD.get()).unwrap() as usize;
        loop {
            next_id = (next_id + 1) % tcb.len();
            if !tcb[next_id].dead {
                break;
            }
        }
        *CURRENT_THREAD.get() = Some(next_id as u32);
        (tcb[next_id].regs, tcb[next_id].cpsr)
    };
    
    // and then finally RFE to LR!
    let rfe_tramp: [usize;2] = [next_regs[14], cpsr.try_into().unwrap()];

    // set up registers
    set_registers_asm(&Regs(next_regs));
    unsafe { core::arch::asm!("rfe {r}", r=in(reg) &rfe_tramp); }

}
