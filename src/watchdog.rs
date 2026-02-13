//! Minimal support for the watchdog timer in the SoC. Currently supports: restart
//ing.

const BCM_PWR_MAN_BASE: *mut u32 = ::core::ptr::with_exposed_provenance_mut(0x2010_0000);
const RSTC_OFFSET: isize = 0x1c;
const WDOG_OFFSET: isize = 0x24;

const COMMON_PASSWORD: u32 = 0x5a00_0000;
const RSTC_FULL_RESET: u32 = 0x0000_0020;

pub extern "C" fn restart() -> ! {
    crate::arch::dsb();

    unsafe {
        BCM_PWR_MAN_BASE
            .byte_offset(WDOG_OFFSET)
            .write_volatile(COMMON_PASSWORD | 0x0000f)
    }
    unsafe {
        BCM_PWR_MAN_BASE
            .byte_offset(RSTC_OFFSET)
            .write_volatile(COMMON_PASSWORD | RSTC_FULL_RESET)
    }

    loop {}
}
