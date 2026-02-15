#![no_std]
#![no_main]

mod comm;
mod boot;
mod utils;

use heapless::Vec;
use utils::bits::bit_set;

fn main() {
    let mut xs: Vec<u8, 8> = Vec::new();
    xs.push(8).unwrap();

    println!("eyo {}", bit_set(0, 2));
}
