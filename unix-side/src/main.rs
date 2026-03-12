mod tty;

fn main() {
    let mut uart = tty::Tty::open(None, 115200);

    // example: read bytes from pi and print them
    loop {
        let b = uart.get8();
        print!("{}", b as char);
    }
}
