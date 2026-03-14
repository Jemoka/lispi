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

use alloc::format;
use alloc::string::String;
use comm::uart::PiUart;
use shared::Framer;

fn main() {

    let mut img = language::Image::new();
    let mut framer = Framer::pi_side(PiUart);

    let _ = framer.recv();
    framer.send("PI_READY".as_bytes());

    loop {
        let payload_str = match String::from_utf8(framer.recv()) {
            Ok(s) => s,
            Err(_) => {
                framer.send("pi-side: received non-UTF8 message\n".as_bytes());
                continue;
            }
        };
        match language::parse(&payload_str) {
            Ok(expr) => {
                match language::evaluate(expr.into(), &mut img) {
                    Ok(result) => {
                        let response = format!("{}", result);
                        framer.send(response.as_bytes());
                    }
                    Err(e) => {
                        let response = format!("{}", e);
                        framer.send(response.as_bytes());
                    }
                }
            }
            Err(e) => {
                let response = format!("PARSE ERROR: {}", e);
                framer.send(response.as_bytes());
            }
        }
    }
}
