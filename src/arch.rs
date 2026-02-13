//! Support for the CPU that we're using.

#[inline(always)]
pub fn dsb() {
    unsafe { ::core::arch::asm!("mcr p15, 0, {t}, c7, c10, 4", t = in(reg) 0) }
}
