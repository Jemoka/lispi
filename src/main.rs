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
// mod language; // UNCOMMENT WHEN BACK TO LISPI

extern crate alloc;

#[inline]
pub fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 13;
    x = x.wrapping_mul(0xc2b2_ae35);
    x ^= x >> 16;
    x
}

extern "C" fn thread1() {
    let mut x = 0;
    let mut k = 0;
    loop {
        k += 1;
        x += 1;
        println!("thread 1: {} {}", x, hash_u32(x));
        if x % 7 == 0 {
            threading::thread_yield();
        }
        x += hash_u32(x);
        x %= 1000;

        if k >= 100 {
            break;
        }
    }
}

extern "C" fn thread2() {
    let mut x = 0;
    let mut k = 0;
    loop {
        k += 1;
        x += 1;
        println!("thread 2: {} {}", x, hash_u32(x));
        if x % 7 == 0 {
            threading::thread_yield();
        }
        x += hash_u32(x);
        x %= 1000;

        if k >= 100 {
            break;
        }
    }
}

extern "C" fn thread3() {
    let mut x = 0;
    let mut k = 0;
    loop {
        k += 1;
        x += 1;
        println!("thread 3: {} {}", x, hash_u32(x));
        if x % 7 == 0 {
            threading::thread_yield();
        }
        x += hash_u32(x);
        x %= 1000;
        if k >= 100 {
            break;
        }
    }
}

fn main() {
    println!("Hello, world!");

    // create two threads
    threading::thread_push(thread1);
    threading::thread_push(thread2);
    threading::thread_push(thread3);

    // start the scheduler
    threading::thread_join();

    println!("Goodbye, world!");
}
