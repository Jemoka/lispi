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
    // (begin (set a 3) (let (a 2) (set a 5)) a) 
    println!("Goodbye, world!");
}
