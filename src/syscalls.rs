//! handle syscalls

use crate::println;
use crate::threading::{thread_context_switch, thread_done};

/// util for user functions to trigger syscalls
#[macro_export]
macro_rules! swi {
    // base case: no more arguments to push into arguments
    // we stick the syscall number into r7, and trigger the software interrupt
    (@inner $nr:expr, $reg_idx:expr) => {
        concat!(
            "mov r7, #", stringify!($nr), "\n",
            "swi #0\n"
        )
    };

    // recursive case, eat one argument in a push, and move the rest into recursive registers
    (@inner $nr:expr, $reg_idx:expr, $head:expr $(, $tail:expr)*) => {
        concat!(
            "mov r", stringify!($reg_idx), ", ", stringify!($head), "\n",
            swi!(@inner $nr, $reg_idx + 1 $(, $tail)*)
        )
    };

    // entrypoint; trigger the inner loop
    ($nr:expr $(, $args:expr)*) => {
        swi!(@inner $nr, 0 $(, $args)*)
    };
}

/// unknown syscall handler
extern "C" fn unknown_syscall_handler() {
    println!("unknown syscall");
    panic!();
}

/// trampoline system-wide to handle syscalls
#[unsafe(naked)]
pub extern "C" fn trampoline_swi() {
    core::arch::naked_asm!(
        "cmp r7, #0",                        // Check the value in r0
        "beq {thread_context_switch}",       // If r7 == 0, branch to handler_zero
        "cmp r7, #1",
        "beq {thread_done}",
        "b {unknown_handler}",               // Default case: branch to unknown
        thread_context_switch = sym thread_context_switch,
        thread_done = sym thread_done,
        unknown_handler = sym unknown_syscall_handler
    );
}
