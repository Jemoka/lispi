#![no_std]
#![no_main]
#![feature(sync_unsafe_cell)]

// os stuff
mod boot;
mod comm;
mod threading;
#[macro_use]
mod syscalls;

mod utils;

// lisp stuff
mod language;

extern crate alloc;

use alloc::vec::Vec;
use shared::{Framer, Transport};

/// Adapter so pi-side uart (module-level fns) implements Transport.
struct PiUart;

impl Transport for PiUart {
    fn put8(&mut self, b: u8) { comm::uart::put8(b); }
    fn get8(&mut self) -> u8 { comm::uart::get8() }
    fn put32(&mut self, v: u32) { comm::uart::put32(v); }
    fn get32(&mut self) -> u32 { comm::uart::get32() }
    fn flush(&mut self) { comm::uart::flush_tx(); }
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 13;
    x = x.wrapping_mul(0xc2b2_ae35);
    x ^= x >> 16;
    x
}

fn main() {
    println!("pi-side: echo server ready\n");

    let mut framer = Framer::pi_side(PiUart);
    let mut buf = [0u8; 1024];
    let mut seed: u32 = 0xCAFE;

    loop {
        let payload = framer.recv(&mut buf);
        seed = hash_u32(seed);

        // build response: original payload + " [rand=0xXXXXXXXX]"
        let mut response: Vec<u8> = Vec::new();
        response.extend_from_slice(payload);
        let suffix = alloc::format!(" [rand=0x{:08x}]", seed);
        response.extend_from_slice(suffix.as_bytes());

        framer.send(&response);
    }
}
