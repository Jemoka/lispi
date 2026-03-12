use crate::println;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    if let Some(loc) = info.location() {
        println!(
            "Panic occurred at file '{}' line {}:\n",
            loc.file(),
            loc.line()
        );
    } else {
        println!("Panic occurred at unknown location.\n");
    }
    let msg = info.message();
    use ::core::fmt::Write as _;
    let _ = ::core::writeln!(crate::comm::print::UartProxy, "{}\n", msg);
    crate::comm::uart::flush();
    crate::utils::memory::dsb();
    crate::utils::watchdog::restart();
}
