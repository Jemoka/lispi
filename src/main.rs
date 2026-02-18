#![no_std]
#![no_main]

mod comm;
mod boot;
mod utils;
mod cs;
mod regs;

extern crate alloc;
use alloc::collections::BTreeMap;

fn main() {
    let mut v = BTreeMap::new();
    v.insert(2,3);
    println!("eyo {:?}", v[&2]);
}

