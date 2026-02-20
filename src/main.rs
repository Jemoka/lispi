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

extern "C" fn thread1() {
    let mut x = 0;
    loop {
        x += 1;
        println!("thread 1: {}", x);
        threading::thread_yield();
    }
}

extern "C" fn thread2() {
    let mut x = 0;
    loop {
        x += 1;
        println!("thread 2: {}", x);
        threading::thread_yield();
    }
}

fn main() {
    println!("Hello, world!");

    // create two threads
    threading::thread_push(thread1);
    threading::thread_push(thread2);

    // start the scheduler
    threading::thread_join();
}
