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
    let mut img = language::Image::new();
    let test = language::parse("(+ 1 (- 12 8))").unwrap();
    let (result, _) = language::evaluate(test.into(), &mut img).unwrap();

    println!("Goodbye, world! {}\n", result);
}
