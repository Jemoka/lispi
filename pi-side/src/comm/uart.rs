use crate::comm::gpio::{self, GpioFunc};
use crate::utils::memory::dmb;
use crate::utils::memory;

const AUX_ENABLES: usize = 0x2021_5004;
const AUX_MU_IO_REG: usize = 0x2021_5040;
const AUX_MU_IER_REG: usize = 0x2021_5044;
const AUX_MU_IIR_REG: usize = 0x2021_5048;
const AUX_MU_LCR_REG: usize = 0x2021_504C;
const AUX_MU_MCR_REG: usize = 0x2021_5050;
const AUX_MU_LSR_REG: usize = 0x2021_5054;
const AUX_MU_CNTL_REG: usize = 0x2021_5060;
const AUX_MU_STAT_REG: usize = 0x2021_5064;
const AUX_MU_BAUD: usize = 0x2021_5068;

/// Initialize mini-UART to 8n1 115200 baud, no interrupts.
pub fn init() {
    unsafe {
        dmb();
        // set GPIO 14,15 to alt5 (mini-UART)
        gpio::gpio_set_function(14, GpioFunc::Alt5);
        gpio::gpio_set_function(15, GpioFunc::Alt5);
        dmb();

        // enable mini-UART (AUX bit 0)
        let enables = memory::get32(AUX_ENABLES);
        memory::put32(AUX_ENABLES, enables | 1);
        dmb();

        // disable tx/rx while we configure
        memory::put32(AUX_MU_CNTL_REG, 0);
        // disable interrupts
        memory::put32(AUX_MU_IER_REG, 0);
        // clear FIFOs
        memory::put32(AUX_MU_IIR_REG, 0b110);
        // clear MCR
        memory::put32(AUX_MU_MCR_REG, 0);
        // 8-bit mode
        memory::put32(AUX_MU_LCR_REG, 0b11);
        // 115200 baud: baudrate = system_clock / (8 * (reg + 1))
        // 250MHz / (8 * 271) = ~115200
        memory::put32(AUX_MU_BAUD, 270);
        // enable tx and rx
        memory::put32(AUX_MU_CNTL_REG, 0b11);
        dmb();
    }
}

/// Disable the mini-UART. Flushes TX first.
pub fn disable() {
    flush_tx();
    unsafe {
        let enables = memory::get32(AUX_ENABLES);
        memory::put32(AUX_ENABLES, enables & !1);
    }
}

/// Block until at least one byte is available, then return it.
pub fn get8() -> u8 {
    unsafe {
        dmb();
        // bit 0 of STAT = symbol available
        while (memory::get32(AUX_MU_STAT_REG) & 1) == 0 {}
        let r = memory::get32(AUX_MU_IO_REG) & 0xFF;
        dmb();
        r as u8
    }
}

/// Non-blocking read: returns Some(byte) if data available, None otherwise.
pub fn get8_async() -> Option<u8> {
    if has_data() { Some(get8()) } else { None }
}

/// Returns true if the RX FIFO has at least one byte.
pub fn has_data() -> bool {
    unsafe { (memory::get32(AUX_MU_STAT_REG) & 1) != 0 }
}

/// Returns true if the TX FIFO has room for at least one byte.
pub fn can_put8() -> bool {
    unsafe { (memory::get32(AUX_MU_STAT_REG) & 0b10) != 0 }
}

/// Write one byte to the TX FIFO, blocking until space is available.
pub fn put8(c: u8) {
    unsafe {
        dmb();
        while !can_put8() {}
        memory::put32(AUX_MU_IO_REG, c as u32);
        dmb();
    }
}

/// Returns true if the TX FIFO is empty AND the transmitter is idle.
pub fn tx_is_empty() -> bool {
    unsafe { (memory::get32(AUX_MU_LSR_REG) & (1 << 6)) != 0 }
}

/// Block until all TX bytes have been transmitted.
pub fn flush_tx() {
    while !tx_is_empty() {
        core::hint::black_box(0);
    }
}

// ---- compatibility wrappers used by print.rs ----

pub fn flush() {
    flush_tx();
}

pub fn write_bytes(bytes: &[u8]) {
    for &b in bytes {
        put8(b);
    }
}

pub fn read_bytes(buf: &mut [u8]) {
    for b in buf.iter_mut() {
        *b = get8();
    }
}

pub fn put32(v: u32) {
    for &b in &v.to_le_bytes() {
        put8(b);
    }
}

pub fn get32() -> u32 {
    let mut bytes = [0u8; 4];
    for b in &mut bytes {
        *b = get8();
    }
    u32::from_le_bytes(bytes)
}
