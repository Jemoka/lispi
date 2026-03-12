mod tty;

use shared::{BAUD_RATE, Framer};
use std::io::Write as _;

fn main() {
    let uart = tty::Tty::open(None, BAUD_RATE);
    let mut framer = Framer::unix_side(uart);

    let messages = ["hello pi!", "how are you?", "lispi is cool", "goodbye"];

    for msg in &messages {
        print!("unix -> pi: {}", msg);
        std::io::stdout().flush().unwrap();

        framer.send(msg.as_bytes());
        let response = framer.recv();

        match std::str::from_utf8(&response) {
            Ok(s) => println!("  pi -> unix: {}", s),
            Err(_) => println!("  pi -> unix: {:?}", response),
        }
    }

    println!("done.");
}
