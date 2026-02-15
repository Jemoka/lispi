unsafe extern "C" {
    // We declare these as `[u32; 0]` so that they have an alignment of 4 but a size of zero. This
    // is to prevent aliasing, since otherwise producing mutable references to anything in the BSS
    // would be undefined behaviour.

    safe static __bss_start__: [u32; 0];
    safe static __bss_end__: [u32; 0];
    safe static __stack_init__: [u32; 0];
}

extern "C" fn __kernel_start() -> ! {
    crate::boot::entrypoint::entrypoint();
    crate::utils::watchdog::restart();
}

// This copies what staff-start.S does:
::core::arch::global_asm!(
    r#"
.pushsection ".text.boot"
.globl _start
_start:
    // switch to super mode, and disable FIQ and IRQ
    mrs r0, cpsr
    and r0, r0, {CLEAR_MODE_MASK}
    orr r0, r0, {SUPER_MODE}
    orr r0, r0, {CLEAR_MODE_IRQ_FIQ}
    msr cpsr, r0

    // Prefetch flush
    mov r0, #0
    mcr p15, 0, r0, c7, c5, 4

    // Clear the BSS (not very efficient; could be faster)
    mov r0, #0
    ldr r1, ={BSS_START}
    ldr r2, ={BSS_END}
    subs r2, r2, r1
    beq 3f
2:
    strb r0, [r1], #1
    subs r2, r2, #1
    bne 2b
3:

    // Initialize the stack pointer. `ldr <rN>, =<symbol>` is a pseudoinstruction; it tells the
    // assembler "somehow, put the address of <symbol> into the register <rN>".
    ldr sp, ={STACK_INIT}
    and sp, sp, -16

    // Clear the frame pointer
    mov fp, #0

    // Jump to __kernel_start
    bl {KERNEL_START}

    // If control returns from __kernel_start, then just restart. This shouldn't happen in the
    // first place.
    bl {RESTART}
.popsection
"#,
    CLEAR_MODE_MASK = const !0b11111u32,
    SUPER_MODE = const 0b10011u32,
    CLEAR_MODE_IRQ_FIQ = const (1u32 << 7) | (1u32 << 6),
    BSS_START = sym __bss_start__,
    BSS_END = sym __bss_end__,
    STACK_INIT = sym __stack_init__,
    KERNEL_START = sym self::__kernel_start,
    RESTART = sym crate::utils::watchdog::restart,
);
