//! threading management system

use alloc::collections::VecDeque;
use alloc::vec;
use core::cell::SyncUnsafeCell;
use core::mem::ManuallyDrop;

use crate::println;
use crate::swi;
use crate::utils::psr::{Mode, PSR};

const STACK_SIZE: usize = 1024;

#[repr(C)]
#[derive(Debug, Clone)]
struct Context {
    r0: u32,
    r1: u32,
    r2: u32,
    r3: u32,
    r4: u32,
    r5: u32,
    r6: u32,
    r7: u32,
    r8: u32,
    r9: u32,
    r10: u32,
    r11: u32,
    r12: u32,
    sp: u32,   // user's sp; 13
    lr: u32,   // user's lr; 14
    pc: u32,   // exception's model's pc
    spsr: u32, // exception's model's cpsr, which we will restore into the user's cpsr when we rfe to the user
}
impl Default for Context {
    fn default() -> Self {
        Self {
            r0: 0,
            r1: 0,
            r2: 0,
            r3: 0,
            r4: 0,
            r5: 0,
            r6: 0,
            r7: 0,
            r8: 0,
            r9: 0,
            r10: 0,
            r11: 0,
            r12: 0,
            sp: 0,
            lr: 0,
            pc: 0,
            spsr: PSR::from(0).with_mode(Mode::User).into(),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
struct ThreadControlBlock {
    stack: *mut u32,
    regs: Context, // All Registers + CPSR
}

impl Default for ThreadControlBlock {
    fn default() -> Self {
        Self {
            stack: core::ptr::null_mut(),
            regs: Context::default(),
        }
    }
}

//  stack ptr is leaked so and each thread is effectively a critical section
unsafe impl Sync for ThreadControlBlock {}
unsafe impl Send for ThreadControlBlock {}

static THREAD_CONTROL_BLOCKS: SyncUnsafeCell<VecDeque<ThreadControlBlock>> =
    SyncUnsafeCell::new(VecDeque::new());
static THREADING_STACK_PTR: [u32; 1] = [0u32; 1]; //  pointer to stack of the system after returning

/// ASSUMES: SUPER mode
/// get a pointer into the regs array, and:
/// 1. restore all registers up to lr
/// 2. rfe to the pc and cpsr
#[unsafe(naked)]
extern "C" fn thread_dispatch_asm(regs: &Context) {
    core::arch::naked_asm!(
        "mov lr, r0", // load the address of the regs array into lr, which we can trash since we are in execption
        "ldm lr, {{r0-r14}}^", // restore all registers up to lr into user mode!
        "add lr, lr, #60", // set offset to the pc and cpsr, which are the last two elements of the regs array
        "rfe lr",          // and then rfe to the pc and cpsr, which will jump to the new thread
    );
}

/// ASSUMES: SUPER mode
/// dispatch to the next thread by loading its context and rfeing to it
extern "C" fn thread_dispatch() {
    unsafe {
        let tcbs = &mut *THREAD_CONTROL_BLOCKS.get();
        let next_thread: &ThreadControlBlock = tcbs.front().unwrap();
        thread_dispatch_asm(&next_thread.regs);
    }
}

/// ASSUMES: SUPER mode
/// create a new thread and push it to the thread control block
/// do brain surgery to set up the stack and registers so that when
/// the thread is dispatched it will start executing at the entrypoint
/// and then jump to the thread end trampoline when it returns
pub fn thread_push(entrypoint: extern "C" fn()) {
    let mut tcb = ThreadControlBlock::default();
    // allocate a stack for the thread
    let stack = vec![0u32; STACK_SIZE].into_boxed_slice();
    // TODO! we don't handle deallocation. cosmic rays or PG&E will
    let mut stack_ptr = ManuallyDrop::new(stack).as_mut_ptr();
    stack_ptr = stack_ptr.wrapping_add(STACK_SIZE); // move the stack pointer to the top of the stack

    // the start of a thread is actually a trampoline which sets up the thread
    // and then jumps to it, ensuring control returns to us instead of user at the end
    // it takes two arguments, function pointer and the ID of the thread
    unsafe {
        let tcbs = &mut *THREAD_CONTROL_BLOCKS.get();

        tcb.regs.r0 = entrypoint as u32;
        tcb.regs.sp = stack_ptr as u32;
        tcb.regs.pc = thread_start_trampoline as *const () as u32; // set the pc to the trampoline, which will jump to the entrypoint

        tcbs.push_back(tcb);
    }
}

/// ASSUMES: SUPER mode
/// context switch!
extern "C" fn thread_context_switch_do(current_context: &'static Context) {
    unsafe {
        // IMPORTANT: this has to live right here. we need to
        // eat up the data from stack pointer before anything happens
        let ctx = core::ptr::read(current_context);

        // and finally do the context switch by finding the next thread and dispatching to it
        let tcbs = &mut *THREAD_CONTROL_BLOCKS.get();
        let mut current_thread = tcbs.pop_front().unwrap(); // get the current thread, which is at the front of the queue

        // gather context
        current_thread.regs = ctx; // update the current thread's context with the gathered context

        // push the current thread to the back of the queue
        tcbs.push_back(current_thread);

        // and then dispatch to the next thread, which is now at the front of the queue
        thread_dispatch();
    }
}

/// ASSUMES: SUPER mode
/// capture state and then context switch!
#[unsafe(naked)]
pub extern "C" fn thread_context_switch() {
    core::arch::naked_asm!(
        // this move without cleanup is ok because syscalls reset the stack
        "sub sp, sp, #68", // make space for the context on the stack; 17 registers * 4 bytes each = 68 bytes
        "stmia sp, {{r0-r14}}^",
        "str lr, [sp, #60]", // store the USER pc at the end of the context; which is the lr in exception mode
        "mrs r1, spsr", // store the user's cpsr at the end of the context
        "str r1, [sp, #64]", // store the USER cpsr at the end of the context
        "mov r0, sp", // return value of sp
        "b {handler}", // call the context switch handler
        handler = sym thread_context_switch_do
    );
}

/// ASSUMES: SUPER mode
/// kill current thread and context switch to the next one
pub extern "C" fn thread_done() {
    unsafe {
        let tcbs = &mut *THREAD_CONTROL_BLOCKS.get();
        tcbs.pop_front(); // pop the current thread, which is at the front of the queue

        // TODO! handle deallocation of stack

        // if there are no more threads, we should return to the system thread, which is the thread that called join
        if tcbs.is_empty() {
            core::arch::asm!(
                "ldr r0, ={stack}", // restore the sp of the system thread, which we saved in join
                "ldr sp, [r0]", // restore the sp of the system thread, which we saved in join
                "pop {{r1, r4-r11, lr}}", // pop the callee saved registers, which are the registers of the system thread
                "msr cpsr, r1", // restore the cpsr of the system thread
                "bx lr", // restore the cpsr of the system thread
                stack = sym THREADING_STACK_PTR,
            );
        }

        // and then dispatch to the next thread, which is now at the front of the queue
        thread_dispatch();
    }
}

/// ASSUMES: SUPER mode
/// join and start threading system
/// importantly this isn't a naked function beacuse we want to appear as
/// a normal function to the caller, and we will eat up the stack and capture
/// registers in the handler
#[unsafe(naked)]
pub extern "C" fn thread_join() {
    core::arch::naked_asm!(
        "mrs r1, cpsr", // store the caller's cpsr in r1, which we will restore in the handler
        "push {{r1, r4-r11, lr}}", // push callee saved registers to the stack, which we will restore in the handler
        "ldr r0, ={stack}", // save the system thread's sp to the global static stack, which we will restore in the handler
        "str sp, [r0]", // save the system thread's sp to the global static stack, which we will restore in the handler
        "b {handler}", // call the join handler, which will save the system thread's context and dispatch to the first thread
        stack = sym THREADING_STACK_PTR,
        handler = sym thread_dispatch,
    );
}

/////// user mode helpers ///////

/// ASSUMES: USER mode
/// called when threads begin execution
extern "C" fn thread_start_trampoline(entrypoint: extern "C" fn()) {
    println!("START THREAD");
    unsafe {
        core::arch::asm!("mov r0, {entrypoint}", entrypoint=in(reg) entrypoint as u32);
        core::arch::asm!("blx r0");
    }
    thread_end_trampoline();
}

/// ASSUMES: USER mode
extern "C" fn thread_end_trampoline() {
    println!("END THREAD");

    // and then swi to the kernel to trigger a context switch, which will clean up the thread and jump to the next one
    unsafe {
        core::arch::asm!(swi!(1));
    } // syscall 1 is dead thread
}

/// ASSUMES: USER mode
/// voluntarily yield control
pub extern "C" fn thread_yield() {
    unsafe {
        core::arch::asm!(swi!(0));
    } // syscall 1 is context switch
}
