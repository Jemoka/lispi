use crate::syscalls::trampoline_swi;

pub extern "C" fn trampoline_unset() {
    panic!("Hehehe reached an unset trapoline!");
}

core::arch::global_asm!(
    r#"
.align 5; 
.globl exception_vector;
exception_vector:
    b {trampoline_unset} @ reset
    b {trampoline_unset} @ undefined instruction
    b {trampoline_swi}   @ software interrupt
    b {trampoline_unset} @ prefetch abort
    b {trampoline_unset} @ data abort
    b {trampoline_unset} @ nope
    b {trampoline_unset} @ IRQ
    b {trampoline_unset} @ FIQ

"#,
    trampoline_unset=sym trampoline_unset,
    trampoline_swi=sym trampoline_swi
);

#[unsafe(naked)]
pub extern "C" fn activate_exception_vector_hook() {
    core::arch::naked_asm!(
        "ldr r0, =exception_vector",
        "mcr p15, 0, r0, c12, c0, 0", // set the exception vector
        "mcr p15, 0, r0, c7, c10, 4", // DSB
        "mcr p15, 0, r0, c7, c5, 4",  // ISB
        "bx lr"
    );
}
