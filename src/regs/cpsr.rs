//! CPSR / SPSR register support for ARMv6 rpi
use proc_bitfield::{ConvRaw, bitfield};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ConvRaw)]
pub enum Mode {
    User = 0b10000,
    FIQ = 0b10001,
    IRQ = 0b10010,
    Supervisor = 0b10011,
    Abort = 0b10111,
    Undefined = 0b11011,
    System = 0b11111,
}

bitfield! {
    #[derive(PartialEq, Eq, Copy, Clone)]
    pub struct PSR(pub u32): Debug, FromStorage, IntoStorage {
        pub mode: u8 [unwrap Mode] @ 0..=4,
        pub thumb: bool @ 5,
        pub fiq_disable: bool @ 6,
        pub irq_disable: bool @ 7,
        pub overflow: bool @ 28,
        pub carry: bool @ 29,
        pub zero: bool @ 30,
        pub negative: bool @ 31
    }
}

impl PSR {
    pub fn get_cpsr() -> Self {
        let value: u32;
        unsafe {
            core::arch::asm!(
                "mrs {value}, cpsr", value = out(reg) value, options(nomem, nostack, preserves_flags)
            );
        }
        Self::from(value)
    }
    pub fn set_cpsr(&self) {
        let value: u32 = (*self).into();
        unsafe {
            core::arch::asm!(
                "msr cpsr, {value}", value = in(reg) value, options(nomem, nostack, preserves_flags)
            );
        }
    }

    pub fn get_spsr() -> Self {
        let value: u32;
        unsafe {
            core::arch::asm!(
                "mrs {value}, spsr", value = out(reg) value, options(nomem, nostack, preserves_flags)
            );
        }
        Self::from(value)
    }
    pub fn set_spsr(&self) {
        let value: u32 = (*self).into();
        unsafe {
            core::arch::asm!(
                "msr spsr, {value}", value = in(reg) value, options(nomem, nostack, preserves_flags)
            );
        }
    }
}
