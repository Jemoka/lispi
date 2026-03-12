use std::fs;
use std::io::{self, Read, Write};
use std::os::unix::io::{AsRawFd, RawFd};
use std::thread;
use std::time::Duration;

const TTY_PREFIXES: &[&str] = &[
    "ttyUSB",      // linux
    "ttyACM",      // linux
    "cu.SLAB_USB", // mac os
];

/// Find a /dev/ttyUSB* (or equivalent) device.
/// Panics if zero or more than one match.
pub fn find_ttyusb() -> String {
    let entries: Vec<_> = fs::read_dir("/dev")
        .expect("failed to read /dev")
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            TTY_PREFIXES.iter().any(|p| name.starts_with(p))
        })
        .collect();

    match entries.len() {
        0 => panic!("no ttyUSB device found in /dev"),
        1 => format!("/dev/{}", entries[0].file_name().to_string_lossy()),
        n => panic!("found {} ttyUSB devices, expected exactly 1", n),
    }
}

pub struct Tty {
    file: fs::File,
}

impl Tty {
    /// Open a TTY device. If `path` is None, auto-detect via `find_ttyusb()`.
    /// Retries up to `max_attempts` times (1 second between attempts).
    /// Configures 8N1 at the given baud rate.
    pub fn open(path: Option<&str>, speed: u32) -> Self {
        let device = match path {
            Some(p) => p.to_string(),
            None => find_ttyusb(),
        };

        let file = open_with_retries(&device, 10);
        let fd = file.as_raw_fd();
        set_8n1(fd, speed);

        eprintln!("opened tty port <{}>", device);
        Tty { file }
    }
}

impl shared::Transport for Tty {
    fn put8(&mut self, b: u8) {
        self.file.write_all(&[b]).expect("tty write failed");
    }
    fn get8(&mut self) -> u8 {
        let mut buf = [0u8; 1];
        self.file.read_exact(&mut buf).expect("tty read failed");
        buf[0]
    }
}

fn open_with_retries(device: &str, max_attempts: u32) -> fs::File {
    for i in 0..max_attempts {
        match fs::OpenOptions::new().read(true).write(true).open(device) {
            Ok(f) => return f,
            Err(_) if i + 1 < max_attempts => {
                eprintln!("couldn't open <{}>, retrying...", device);
                thread::sleep(Duration::from_secs(1));
            }
            Err(e) => panic!("couldn't open <{}>: {}", device, e),
        }
    }
    unreachable!()
}

fn set_8n1(fd: RawFd, speed: u32) {
    use libc::*;

    let baud = match speed {
        115200 => B115200,
        9600 => B9600,
        57600 => B57600,
        38400 => B38400,
        19200 => B19200,
        _ => panic!("unsupported baud rate: {}", speed),
    };

    unsafe {
        let mut tty: termios = std::mem::zeroed();

        cfsetispeed(&mut tty, baud);
        cfsetospeed(&mut tty, baud);

        // 8N1
        tty.c_cflag &= !PARENB;
        tty.c_cflag &= !CSTOPB;
        tty.c_cflag &= !CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag &= !CRTSCTS;
        tty.c_cflag |= CREAD | CLOCAL;

        // raw input
        tty.c_iflag &= !(IGNBRK | IXON | IXOFF | IXANY);
        tty.c_iflag &= !(ICANON | ECHO | ECHOE | ISIG);

        // raw output
        tty.c_oflag &= !OPOST;

        // blocking: wait for at least 1 byte, no timeout
        tty.c_cc[VMIN] = 1;
        tty.c_cc[VTIME] = 0;

        if tcsetattr(fd, TCSANOW, &tty) != 0 {
            panic!("tcsetattr failed: {}", io::Error::last_os_error());
        }
    }
}
