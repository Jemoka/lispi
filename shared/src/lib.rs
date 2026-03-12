//! Shared constants and protocol definitions for pi-side <-> unix-side
//! UART communication.
//!
//! # Framing Protocol
//!
//! Communication is synchronous and half-duplex. Each message is wrapped
//! in a frame with sync barriers and delimiters.
//!
//! ## Pi-side frame (pi -> unix):
//!
//! ```text
//! [DEADBEEF]*n          sync barrier (repeated; discarded)
//! [00000000 00000000    frame header: four u32 zeros
//!  00000000 00000000]
//! [n: u32]              payload length in bytes
//! [payload: u8 * n]     payload data
//! [00000000 00000000    frame footer: four u32 zeros
//!  00000000 00000000]
//! [FACEFEED]*n          footer padding (repeated; other side should also discard)
//! ```
//!
//! ## Unix-side frame (unix -> pi):
//!
//! Same structure, but the sync/footer magic words are **swapped**:
//!
//! ```text
//! [FACEFEED]*n          sync barrier
//! [00000000 * 4]        frame header
//! [n: u32]              payload length
//! [payload: u8 * n]     payload data
//! [00000000 * 4]        frame footer
//! [DEADBEEF]*n          footer padding
//! ```

#![no_std]

extern crate alloc;
use alloc::vec::Vec;

pub const BAUD_RATE: u32 = 115200;

/// Number of sync words to send before a frame.
pub const SYNC_COUNT: u32 = 8;

/// Number of footer words to send after a frame.
pub const FOOTER_COUNT: u32 = 4;

/// Number of zero u32s in the header/footer delimiter.
pub const ZERO_DELIMITER_COUNT: u32 = 4;

// Pi-side framing constants
pub const PI_SYNC_WORD: u32 = 0xDEAD_BEEF;
pub const PI_FOOTER_WORD: u32 = 0xFACE_FEED;

// Unix-side framing constants (swapped)
pub const UNIX_SYNC_WORD: u32 = 0xFACE_FEED;
pub const UNIX_FOOTER_WORD: u32 = 0xDEAD_BEEF;

/// Transport trait — anything that can send/receive bytes and u32s.
pub trait Transport {
    fn put8(&mut self, b: u8);
    fn get8(&mut self) -> u8;

    fn put32(&mut self, v: u32) {
        for &b in &v.to_le_bytes() {
            self.put8(b);
        }
    }

    fn get32(&mut self) -> u32 {
        let mut bytes = [0u8; 4];
        for b in &mut bytes {
            *b = self.get8();
        }
        u32::from_le_bytes(bytes)
    }

    fn flush(&mut self) {}
}

/// A framer wraps a transport and handles send/recv with a specific
/// sync_word (sent before) and footer_word (sent after).
pub struct Framer<T: Transport> {
    pub transport: T,
    pub sync_word: u32,
    pub footer_word: u32,
    /// The sync word we expect to *receive* from the other side.
    pub peer_sync_word: u32,
}

impl<T: Transport> Framer<T> {
    /// Create a pi-side framer (sends DEADBEEF, receives FACEFEED).
    pub fn pi_side(transport: T) -> Self {
        Self {
            transport,
            sync_word: PI_SYNC_WORD,
            footer_word: PI_FOOTER_WORD,
            peer_sync_word: UNIX_SYNC_WORD,
        }
    }

    /// Create a unix-side framer (sends FACEFEED, receives DEADBEEF).
    pub fn unix_side(transport: T) -> Self {
        Self {
            transport,
            sync_word: UNIX_SYNC_WORD,
            footer_word: UNIX_FOOTER_WORD,
            peer_sync_word: PI_SYNC_WORD,
        }
    }

    /// Send a framed message.
    pub fn send(&mut self, payload: &[u8]) {
        // sync barrier
        for _ in 0..SYNC_COUNT {
            self.transport.put32(self.sync_word);
        }
        // header: four zero u32s
        for _ in 0..ZERO_DELIMITER_COUNT {
            self.transport.put32(0);
        }
        // payload length
        self.transport.put32(payload.len() as u32);
        // payload bytes
        for &b in payload {
            self.transport.put8(b);
        }
        // footer: four zero u32s
        for _ in 0..ZERO_DELIMITER_COUNT {
            self.transport.put32(0);
        }
        // footer padding
        for _ in 0..FOOTER_COUNT {
            self.transport.put32(self.footer_word);
        }
        self.transport.flush();
    }

    /// Receive a framed message. Blocks until a complete frame arrives.
    /// Returns the payload as an owned Vec.
    pub fn recv(&mut self) -> Vec<u8> {
        // scan for header: skip sync words, find four consecutive zero u32s
        let mut zero_count: u32 = 0;
        loop {
            let word = self.transport.get32();
            if word == 0 {
                zero_count += 1;
                if zero_count >= ZERO_DELIMITER_COUNT {
                    break;
                }
            } else {
                zero_count = 0;
            }
        }

        // payload length
        let len = self.transport.get32() as usize;

        // payload
        let mut buf = Vec::with_capacity(len);
        for _ in 0..len {
            buf.push(self.transport.get8());
        }

        // consume footer zeros
        for _ in 0..ZERO_DELIMITER_COUNT {
            self.transport.get32();
        }
        // consume footer words
        for _ in 0..FOOTER_COUNT {
            self.transport.get32();
        }

        buf
    }
}
