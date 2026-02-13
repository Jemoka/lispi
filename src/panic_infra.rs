#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    if let Some(loc) = info.location() {
        crate::println!(
            "Panic occurred at file '{}' line {}:\n",
            loc.file(),
            loc.line()
        );
    } else {
        crate::println!("Panic occurred at unknown location.\n");
    }
    let msg = info.message();
    use ::core::fmt::Write as _;
    let _ = ::core::writeln!(crate::print::UartProxy, "{}\n", msg);
    crate::uart::flush();

    crate::arch::dsb();

    crate::watchdog::restart();
}
