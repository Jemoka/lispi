#![no_std]
#![no_main]

mod arch;
mod llvm_infra;
mod panic_infra;
mod print;
mod start;
mod uart;
mod watchdog;

fn main() {
    uart::init();
    println!("Hello from Rust on the Pi!");
}
