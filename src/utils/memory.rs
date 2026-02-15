#[unsafe(no_mangle)]
pub extern "C" fn __aeabi_unwind_cpp_pr0() {}

#[inline(always)]
pub fn dsb() {
    unsafe { ::core::arch::asm!("mcr p15, 0, {t}, c7, c10, 4", t = in(reg) 0) }
}
