unsafe extern "C" {
    fn staff_uart_init(baud_rate: u32);
    fn staff_uart_write_bytes(bytes: *const u8, len: usize);
    fn staff_uart_flush();
}

pub fn init() {
    unsafe { staff_uart_init(115200) }
}

pub fn flush() {
    unsafe { staff_uart_flush() }
}

pub fn write_bytes(bytes: &[u8]) {
    unsafe { staff_uart_write_bytes(bytes.as_ptr(), bytes.len()) }
}
