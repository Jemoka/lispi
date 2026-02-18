#![no_std]
#![no_main]

// os stuff
mod boot;
mod comm;
mod cs;
mod regs;
mod utils;

// lisp stuff
mod language;

extern crate alloc;
use alloc::collections::BTreeMap;

fn main() {
    let mut v = BTreeMap::new();
    v.insert(2, 3);
    println!("eyo {:?}", v[&2]);
}
