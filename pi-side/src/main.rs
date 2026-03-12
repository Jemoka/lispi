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
    let _ = language::evaluate(
        language::parse("(set chicken (+ 12 (- 8 3)))")
            .unwrap()
            .into(),
        &mut img,
    )
    .unwrap();
    let _ = language::evaluate(
        language::parse("(@put32 #0x80000 chicken)").unwrap().into(),
        &mut img,
    )
    .unwrap();
    let (result3, _) = language::evaluate(
        language::parse("(@get32 #0x80000)").unwrap().into(),
        &mut img,
    )
    .unwrap();

    println!("Goodbye, world! {}\n", result3);
}
