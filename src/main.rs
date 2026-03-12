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

fn main() {
    let test = language::parse("(+ 1 (- 12 @dsb))").unwrap();
    println!("Goodbye, world! {}\n", test);
}
