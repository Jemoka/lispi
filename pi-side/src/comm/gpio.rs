use crate::utils::memory::{get32, put32};

const GPIO_MAX_PIN: u32 = 53;
const GPIO_BASE: usize = 0x2020_0000;

const GPIO_FSEL: [usize; 6] = [
    GPIO_BASE + 0x00,
    GPIO_BASE + 0x04,
    GPIO_BASE + 0x08,
    GPIO_BASE + 0x0C,
    GPIO_BASE + 0x10,
    GPIO_BASE + 0x14,
];

const GPIO_SET0: usize = GPIO_BASE + 0x1C;
const GPIO_CLR0: usize = GPIO_BASE + 0x28;
const GPIO_LEV0: usize = GPIO_BASE + 0x34;

const GPIO_PUD: usize = GPIO_BASE + 0x94;
const GPIO_PUDCLK0: usize = GPIO_BASE + 0x98;

#[repr(u32)]
#[derive(Copy, Clone)]
pub enum GpioFunc {
    Input = 0b000,
    Output = 0b001,
    Alt0 = 0b100,
    Alt1 = 0b101,
    Alt2 = 0b110,
    Alt3 = 0b111,
    Alt4 = 0b011,
    Alt5 = 0b010,
}

fn assert_valid_pin(pin: u32) {
    assert!(pin <= GPIO_MAX_PIN, "illegal pin={}", pin);
}

pub fn gpio_set_function(pin: u32, function: GpioFunc) {
    assert_valid_pin(pin);

    let group = (pin / 10) as usize;
    let offset = (pin % 10) * 3;

    let mut sel = unsafe { get32(GPIO_FSEL[group]) };
    sel &= !(0b111 << offset);
    sel |= (function as u32) << offset;
    unsafe { put32(GPIO_FSEL[group], sel) };
}

pub fn gpio_set_output(pin: u32) {
    gpio_set_function(pin, GpioFunc::Output);
}

pub fn gpio_set_input(pin: u32) {
    gpio_set_function(pin, GpioFunc::Input);
}

pub fn gpio_set_on(pin: u32) {
    assert_valid_pin(pin);
    // SET0 and SET1 are contiguous
    unsafe { put32(GPIO_SET0 + (pin / 32) as usize * 4, 1 << (pin % 32)) };
}

pub fn gpio_set_off(pin: u32) {
    assert_valid_pin(pin);
    // CLR0 and CLR1 are contiguous
    unsafe { put32(GPIO_CLR0 + (pin / 32) as usize * 4, 1 << (pin % 32)) };
}

pub fn gpio_write(pin: u32, v: bool) {
    if v {
        gpio_set_on(pin);
    } else {
        gpio_set_off(pin);
    }
}

#[repr(u32)]
#[derive(Copy, Clone)]
pub enum GpioPud {
    Off = 0b00,
    PullDown = 0b01,
    PullUp = 0b10,
}

/// Set pull-up/pull-down for <pin> using the BCM2835 sequence:
///   1. Write to GPPUD to set the desired control signal
///   2. Wait 150 cycles
///   3. Write to GPPUDCLK0/1 to clock the signal into the pin
///   4. Wait 150 cycles
///   5. Clear GPPUD and GPPUDCLK
pub fn gpio_set_pullup(pin: u32, pud: GpioPud) {
    assert_valid_pin(pin);

    unsafe {
        put32(GPIO_PUD, pud as u32);
    }
    // ~150 cycle delay
    for _ in 0..150 {
        core::hint::black_box(0);
    }
    unsafe {
        put32(
            GPIO_PUDCLK0 + (pin / 32) as usize * 4,
            1 << (pin % 32),
        );
    }
    for _ in 0..150 {
        core::hint::black_box(0);
    }
    unsafe {
        put32(GPIO_PUD, 0);
        put32(GPIO_PUDCLK0 + (pin / 32) as usize * 4, 0);
    }
}

pub fn gpio_read(pin: u32) -> bool {
    assert_valid_pin(pin);
    // LEV0 and LEV1 are contiguous
    let val = unsafe { get32(GPIO_LEV0 + (pin / 32) as usize * 4) };
    (val & (1 << (pin % 32))) != 0
}
