mod tty;

use shared::{BAUD_RATE, Framer};
use std::io::{self, BufRead, Write};

fn main() {
    let uart = tty::Tty::open(None, BAUD_RATE);
    let mut framer = Framer::unix_side(uart);

    framer.send("UNIX_READY".as_bytes());
    let _ = framer.recv();

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("read error: {}", e);
                break;
            }
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        framer.send(line.as_bytes());
        let response = framer.recv();

        match std::str::from_utf8(&response) {
            Ok(s) => println!("{}", s),
            Err(_) => println!("Received non-UTF8 response: {:?}", response),
        }
    }
}
