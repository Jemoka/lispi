pub struct UartProxy;

impl ::core::fmt::Write for UartProxy {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        crate::comm::uart::write_bytes(s.as_bytes());
        Ok(())
    }
}

#[macro_export]
macro_rules! print {
    ($($args:tt)*) => {
        {
            #[allow(unused)]
            use ::core::fmt::Write;
            let _ = ::core::write!(&mut $crate::comm::print::UartProxy, $($args)*);
            $crate::comm::uart::flush();
        }
    }
}
#[macro_export]
macro_rules! println {
    ($($args:tt)*) => {
        {
            #[allow(unused)]
            use ::core::fmt::Write;
            let _ = ::core::writeln!(&mut $crate::comm::print::UartProxy, $($args)*);
            $crate::comm::uart::flush();
        }
    }
}
