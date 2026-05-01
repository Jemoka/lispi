//! LISP Syscall Infrastructure

use alloc::rc::Rc;
use proc_bitfield::{bitfield, ConvRaw};

use core::alloc::Layout;

use super::ast::Value;
use super::environment::Image;
use super::execute::evaluate;

use crate::comm::uart;
use crate::utils::memory::{dsb, get32, prefetch_flush, put32};

static BAD_APPLE: &[u8] = include_bytes!("apple.bin");

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Syscall {
    Get32,
    Put32,
    DSB,
    PrefetchFlush,
    UartInit,
    UartPut8,
    UartGet8,
    Delay,
    Alloc32,
    Free32,
    Read32,
    Zero32,
    Fill32,
    Full32,
    Ldr,
    Str,
    Apple,
    Unpack1to16,
    ClearSetMonitor,
    GetMonitor,
    StopMonitor
}

impl Syscall {
    /// Look up a syscall by name (case-insensitive).
    /// Returns None for unknown names.
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("apple") {
            return Some(Self::Apple);
        }
        if name.eq_ignore_ascii_case("get32") {
            return Some(Self::Get32);
        }
        if name.eq_ignore_ascii_case("put32") {
            return Some(Self::Put32);
        }
        if name.eq_ignore_ascii_case("dsb") {
            return Some(Self::DSB);
        }
        if name.eq_ignore_ascii_case("prefetch_flush") {
            return Some(Self::PrefetchFlush);
        }
        if name.eq_ignore_ascii_case("uart/init") {
            return Some(Self::UartInit);
        }
        if name.eq_ignore_ascii_case("uart/put8") {
            return Some(Self::UartPut8);
        }
        if name.eq_ignore_ascii_case("uart/get8") {
            return Some(Self::UartGet8);
        }
        if name.eq_ignore_ascii_case("delay") {
            return Some(Self::Delay);
        }
        if name.eq_ignore_ascii_case("alloc32") {
            return Some(Self::Alloc32);
        }
        if name.eq_ignore_ascii_case("free32") {
            return Some(Self::Free32);
        }
        if name.eq_ignore_ascii_case("read32") {
            return Some(Self::Read32);
        }
        if name.eq_ignore_ascii_case("zero32") {
            return Some(Self::Zero32);
        }
        if name.eq_ignore_ascii_case("fill32") {
            return Some(Self::Fill32);
        }
        if name.eq_ignore_ascii_case("full32") {
            return Some(Self::Full32);
        }
        if name.eq_ignore_ascii_case("ldr") {
            return Some(Self::Ldr);
        }
        if name.eq_ignore_ascii_case("str") {
            return Some(Self::Str);
        }
        if name.eq_ignore_ascii_case("unpack1to16") {
            return Some(Self::Unpack1to16);
        }
        if name.eq_ignore_ascii_case("monitor/clear") {
            return Some(Self::ClearSetMonitor);
        }
        if name.eq_ignore_ascii_case("monitor/get") {
            return Some(Self::GetMonitor);
        }
        if name.eq_ignore_ascii_case("monitor/stop") {
            return Some(Self::StopMonitor);
        }
        None
    }
}


#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, ConvRaw)]
pub enum PMUEvent {
    ICacheMiss = 0x00,
    IMicroTLBMiss = 0x3,
    DMicroTLBMiss = 0x4,
    BranchExecuted = 0x5,
    BranchMiss = 0x6,
    DCacheMiss = 0xB,
    MainTLBMiss = 0xF,
    
}

bitfield! {
    #[derive(Clone, PartialEq, Eq)]
    pub struct PMUControlReg(pub u32): Debug, FromStorage, IntoStorage {
        pub enabled: bool @ 0, // enable all counters
        pub resetbufs: bool @ 1, // reset event counter buffers
        pub resetcycle: bool @ 2, // reset cycle counter
        pub divider: bool @ 3, // cycle counter counts every 64 cycles if set
        pub ctr0interrupt: bool @ 4, // interrupt reporting ctr0 
        pub ctr1interrupt: bool @ 5, // interrupt reporting ctr0 
        pub cycleinterrupt: bool @ 6, // interrupt reporting cycle counter
        // sbz @ 7
        pub ctr0overflow: bool @ 8, // ctr0 overflow flag
        pub ctr1overflow: bool @ 9, // ctr1 overflow flag
        pub cycleoverflow: bool @ 10, // cycle counter overflow flag
        pub export: bool @ 11, // export events to external pin
        pub event0: u8 [unwrap PMUEvent] @ 12..=19, // event type for ctr0
        pub event1: u8 [unwrap PMUEvent] @ 20..=27, // event type for
    }
}

pub fn execute_syscall(
    syscall: Syscall,
    sexp: Rc<Value>,
    image: &mut Image,
) -> Result<Value, &'static str> {
    match syscall {
        Syscall::GetMonitor => {
            let clocks = unsafe {
                let val: u32;
                ::core::arch::asm!("mrc p15, 0, {val}, c15, c12, 1", val=out(reg) val);
                val
            };

            let icachemiss = unsafe {
                // this is hardcoded event 0 for now...
                let val: u32;
                ::core::arch::asm!("mrc p15, 0, {val}, c15, c12, 2", val=out(reg) val);
                val
            };

            let branchmiss = unsafe {
                // this is hardcoded event 1 for now...
                let val: u32;
                ::core::arch::asm!("mrc p15, 0, {val}, c15, c12, 3", val=out(reg) val);
                val
            };

            Ok(Value::cons(
                Value::Number(super::number::Number::Unsigned(clocks as u32)),
                Value::cons (
                    Value::Number(super::number::Number::Unsigned(icachemiss as u32)),
                    Value::cons (
                        Value::Number(super::number::Number::Unsigned(branchmiss as u32)),
                        Value::Nil
                    )
                )
            ))
        }
        Syscall::ClearSetMonitor => {
            unsafe {
                let pmu_config = PMUControlReg::from(0)
                    .with_enabled(true)
                    .with_resetbufs(true)
                    .with_resetcycle(true)
                    .with_divider(false)
                    .with_event0(PMUEvent::ICacheMiss)
                    .with_event1(PMUEvent::BranchMiss);
                let val: u32 = pmu_config.into();
                ::core::arch::asm!("mcr p15, 0, {val}, c15, c12, 0", val = in(reg) val);
            };
            Ok(Value::Nil)
        }
        Syscall::StopMonitor => {
            unsafe {
                let pmu_config = PMUControlReg::from(0)
                    .with_enabled(false);
                let val: u32 = pmu_config.into();
                ::core::arch::asm!("mcr p15, 0, {val}, c15, c12, 0", val = in(reg) val);
            };
            Ok(Value::Nil)
        }
        Syscall::Apple => {
            // Returns (addr nframes) pointing to the raw 1bpp Bad Apple data.
            // Each frame is 320x240 pixels at 1bpp = 9600 bytes.
            let bytes_per_frame = 320 / 8 * 240; // 9600
            let num_frames = BAD_APPLE.len() / bytes_per_frame;
            let ptr = BAD_APPLE.as_ptr() as usize;
            Ok(Value::cons(
                Value::Number(super::number::Number::Addr(ptr)),
                Value::cons(
                    Value::Number(super::number::Number::Integer(num_frames as i32)),
                    Value::Nil,
                ),
            ))
        }
        Syscall::Get32 => {
            let addr = evaluate(sexp.nth(1), image)?;
            if let Value::Number(n) = &addr {
                let raw_addr = n
                    .as_addr()
                    .map_err(|_| "GET32: argument must be an address or non-negative integer.")?;
                if let super::number::Number::Addr(a) = raw_addr {
                    if a % 4 != 0 {
                        return Err("GET32: address must be 4-byte aligned.");
                    }
                    let val = unsafe { get32(a) };
                    Ok(Value::Number(super::number::Number::Unsigned(val)))
                } else {
                    unreachable!()
                }
            } else {
                Err("GET32 requires a number argument.")
            }
        }
        Syscall::Put32 => {
            let addr_val = evaluate(sexp.nth(1), image)?;
            let val_val = evaluate(sexp.nth(2), image)?;
            if let (Value::Number(n_addr), Value::Number(n_val)) = (&addr_val, &val_val) {
                let raw_addr = n_addr.as_addr().map_err(
                    |_| "PUT32: first argument must be an address or non-negative integer.",
                )?;
                let raw_val = n_val
                    .as_u32()
                    .map_err(|_| "PUT32: second argument must be an integer or unsigned.")?;
                if let super::number::Number::Addr(a) = raw_addr {
                    if a % 4 != 0 {
                        return Err("PUT32: address must be 4-byte aligned.");
                    }
                    unsafe { put32(a, raw_val) };
                    prefetch_flush();
                    dsb();
                    Ok(Value::Nil)
                } else {
                    unreachable!()
                }
            } else {
                Err("PUT32 requires two number arguments.")
            }
        }
        Syscall::DSB => {
            dsb();
            Ok(Value::Nil)
        }
        Syscall::PrefetchFlush => {
            prefetch_flush();
            Ok(Value::Nil)
        }
        Syscall::UartInit => {
            uart::init();
            Ok(Value::Nil)
        }
        Syscall::UartPut8 => {
            let val = evaluate(sexp.nth(1), image)?;
            if let Value::Number(n) = &val {
                let byte = n.as_i32().map_err(|_| "uart/put8: argument must be an integer.")?;
                uart::put8(byte as u8);
                Ok(Value::Nil)
            } else {
                Err("uart/put8 requires a number argument.")
            }
        }
        Syscall::UartGet8 => {
            let byte = uart::get8();
            Ok(Value::Number(super::number::Number::Integer(byte as i32)))
        }
        Syscall::Delay => {
            let val = evaluate(sexp.nth(1), image)?;
            if let Value::Number(n) = &val {
                let count = n
                    .as_i32()
                    .map_err(|_| "delay: argument must be an integer.")?;
                if count < 0 {
                    return Err("delay: argument must be non-negative.");
                }
                for _ in 0..count {
                    unsafe {
                        core::arch::asm!("add r1, r1, #0", out("r1") _);
                    }
                }
                Ok(Value::Nil)
            } else {
                Err("delay requires a number argument.")
            }
        }
        Syscall::Alloc32 => {
            let val = evaluate(sexp.nth(1), image)?;
            let count = if let Value::Number(n) = &val {
                let c = n
                    .as_i32()
                    .map_err(|_| "alloc32: first argument must be an integer.")?;
                if c < 0 {
                    return Err("alloc32: count must be non-negative.");
                }
                c as usize
            } else {
                return Err("alloc32 requires a number argument.");
            };
            let align = if sexp.nth_exists(2) {
                let a_val = evaluate(sexp.nth(2), image)?;
                if let Value::Number(n) = &a_val {
                    let a = n
                        .as_i32()
                        .map_err(|_| "alloc32: alignment must be an integer.")?;
                    if a < 4 || (a as u32) & (a as u32 - 1) != 0 {
                        return Err("alloc32: alignment must be a power of 2 >= 4.");
                    }
                    a as usize
                } else {
                    return Err("alloc32: alignment must be a number.");
                }
            } else {
                4
            };
            let size = count * 4;
            if size == 0 {
                return Ok(Value::Number(super::number::Number::Addr(align)));
            }
            let layout =
                Layout::from_size_align(size, align).map_err(|_| "alloc32: invalid layout.")?;
            let ptr = unsafe { alloc::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                return Err("alloc32: allocation failed.");
            }
            Ok(Value::Number(super::number::Number::Addr(ptr as usize)))
        }
        // (@read32 addr offset n) — read n u32 slots starting at addr + offset*4,
        // return as a lisp list.
        Syscall::Read32 => {
            let addr_val = evaluate(sexp.nth(1), image)?;
            let off_val = evaluate(sexp.nth(2), image)?;
            let n_val = evaluate(sexp.nth(3), image)?;
            let base = if let Value::Number(n) = &addr_val {
                let a = n.as_addr().map_err(|_| "read32: first arg must be an address.")?;
                if let super::number::Number::Addr(a) = a { a } else { unreachable!() }
            } else {
                return Err("read32: first arg must be an address.");
            };
            let offset = if let Value::Number(n) = &off_val {
                n.as_i32().map_err(|_| "read32: offset must be an integer.")? as usize
            } else {
                return Err("read32: offset must be a number.");
            };
            let count = if let Value::Number(n) = &n_val {
                n.as_i32().map_err(|_| "read32: count must be an integer.")? as usize
            } else {
                return Err("read32: count must be a number.");
            };
            let mut result = Value::Nil;
            for i in (0..count).rev() {
                let addr = base + (offset + i) * 4;
                let val = unsafe { get32(addr) };
                result = Value::cons(
                    Value::Number(super::number::Number::Unsigned(val)),
                    result,
                );
            }
            Ok(result)
        }

        // (@zero32 addr offset n) — zero n u32 slots starting at addr + offset*4.
        Syscall::Zero32 => {
            let addr_val = evaluate(sexp.nth(1), image)?;
            let off_val = evaluate(sexp.nth(2), image)?;
            let n_val = evaluate(sexp.nth(3), image)?;
            let base = if let Value::Number(n) = &addr_val {
                let a = n.as_addr().map_err(|_| "zero32: first arg must be an address.")?;
                if let super::number::Number::Addr(a) = a { a } else { unreachable!() }
            } else {
                return Err("zero32: first arg must be an address.");
            };
            let offset = if let Value::Number(n) = &off_val {
                n.as_i32().map_err(|_| "zero32: offset must be an integer.")? as usize
            } else {
                return Err("zero32: offset must be a number.");
            };
            let count = if let Value::Number(n) = &n_val {
                n.as_i32().map_err(|_| "zero32: count must be an integer.")? as usize
            } else {
                return Err("zero32: count must be a number.");
            };
            for i in 0..count {
                let addr = base + (offset + i) * 4;
                unsafe { put32(addr, 0) };
            }
            Ok(Value::Nil)
        }

        // (@fill32 addr offset list) — write each u32 in list to consecutive
        // slots starting at addr + offset*4.
        Syscall::Fill32 => {
            let addr_val = evaluate(sexp.nth(1), image)?;
            let off_val = evaluate(sexp.nth(2), image)?;
            let list_val = evaluate(sexp.nth(3), image)?;
            let base = if let Value::Number(n) = &addr_val {
                let a = n.as_addr().map_err(|_| "fill32: first arg must be an address.")?;
                if let super::number::Number::Addr(a) = a { a } else { unreachable!() }
            } else {
                return Err("fill32: first arg must be an address.");
            };
            let offset = if let Value::Number(n) = &off_val {
                n.as_i32().map_err(|_| "fill32: offset must be an integer.")? as usize
            } else {
                return Err("fill32: offset must be a number.");
            };
            let mut cur = list_val;
            let mut i = 0;
            loop {
                match &cur {
                    Value::Nil => break,
                    Value::Cons(head, tail) => {
                        if let Value::Number(n) = head.as_ref() {
                            let val = n.as_u32().map_err(|_| "fill32: list element must be a u32.")?;
                            let addr = base + (offset + i) * 4;
                            unsafe { put32(addr, val) };
                            i += 1;
                            cur = tail.as_ref().clone();
                        } else {
                            return Err("fill32: list elements must be numbers.");
                        }
                    }
                    _ => return Err("fill32: third arg must be a list."),
                }
            }
            Ok(Value::Nil)
        }

        Syscall::Free32 => {
            let ptr_val = evaluate(sexp.nth(1), image)?;
            let len_val = evaluate(sexp.nth(2), image)?;
            let raw_addr = if let Value::Number(n) = &ptr_val {
                let a = n
                    .as_addr()
                    .map_err(|_| "free32: first argument must be an address.")?;
                if let super::number::Number::Addr(a) = a {
                    a
                } else {
                    unreachable!()
                }
            } else {
                return Err("free32: first argument must be an address.");
            };
            let count = if let Value::Number(n) = &len_val {
                let c = n
                    .as_i32()
                    .map_err(|_| "free32: second argument must be an integer.")?;
                if c < 0 {
                    return Err("free32: length must be non-negative.");
                }
                c as usize
            } else {
                return Err("free32 requires a number as second argument.");
            };
            let align = if sexp.nth_exists(3) {
                let a_val = evaluate(sexp.nth(3), image)?;
                if let Value::Number(n) = &a_val {
                    let a = n
                        .as_i32()
                        .map_err(|_| "free32: alignment must be an integer.")?;
                    if a < 4 || (a as u32) & (a as u32 - 1) != 0 {
                        return Err("free32: alignment must be a power of 2 >= 4.");
                    }
                    a as usize
                } else {
                    return Err("free32: alignment must be a number.");
                }
            } else {
                4
            };
            let size = count * 4;
            if size == 0 {
                return Ok(Value::Nil);
            }
            let layout =
                Layout::from_size_align(size, align).map_err(|_| "free32: invalid layout.")?;
            unsafe { alloc::alloc::dealloc(raw_addr as *mut u8, layout) };
            Ok(Value::Nil)
        }

        // (@full32 addr offset n value) — fill n consecutive u32 slots
        // starting at addr + offset*4 with value. Pure Rust loop, no list needed.
        Syscall::Full32 => {
            let addr_val = evaluate(sexp.nth(1), image)?;
            let off_val = evaluate(sexp.nth(2), image)?;
            let n_val = evaluate(sexp.nth(3), image)?;
            let val_val = evaluate(sexp.nth(4), image)?;
            let base = if let Value::Number(n) = &addr_val {
                let a = n.as_addr().map_err(|_| "full32: first arg must be an address.")?;
                if let super::number::Number::Addr(a) = a { a } else { unreachable!() }
            } else {
                return Err("full32: first arg must be an address.");
            };
            let offset = if let Value::Number(n) = &off_val {
                n.as_i32().map_err(|_| "full32: offset must be an integer.")? as usize
            } else {
                return Err("full32: offset must be a number.");
            };
            let count = if let Value::Number(n) = &n_val {
                n.as_i32().map_err(|_| "full32: count must be an integer.")? as usize
            } else {
                return Err("full32: count must be a number.");
            };
            let val = if let Value::Number(n) = &val_val {
                n.as_u32().map_err(|_| "full32: value must be a u32.")?
            } else {
                return Err("full32: value must be a number.");
            };
            let dst = (base + offset * 4) as *mut u32;
            let slice = unsafe { core::slice::from_raw_parts_mut(dst, count) };
            slice.fill(val);
            Ok(Value::Nil)
        }

        // (@ldr addr-or-array offset n) — load n u32 slots into a new array.
        // Source can be an address or an array (uses backing pointer).
        Syscall::Ldr => {
            let addr_val = evaluate(sexp.nth(1), image)?;
            let off_val = evaluate(sexp.nth(2), image)?;
            let n_val = evaluate(sexp.nth(3), image)?;
            let base = match &addr_val {
                Value::Number(n) => {
                    let a = n.as_addr().map_err(|_| "ldr: first arg must be an address or array.")?;
                    if let super::number::Number::Addr(a) = a { a } else { unreachable!() }
                }
                Value::Array(a) => a.borrow().as_ptr() as usize,
                _ => return Err("ldr: first arg must be an address or array."),
            };
            let offset = if let Value::Number(n) = &off_val {
                n.as_i32().map_err(|_| "ldr: offset must be an integer.")? as usize
            } else {
                return Err("ldr: offset must be a number.");
            };
            let count = if let Value::Number(n) = &n_val {
                n.as_i32().map_err(|_| "ldr: count must be an integer.")? as usize
            } else {
                return Err("ldr: count must be a number.");
            };
            let src = (base + offset * 4) as *const u8;
            let mut v = alloc::vec![0u32; count];
            unsafe { core::ptr::copy_nonoverlapping(src, v.as_mut_ptr() as *mut u8, count * 4) };
            Ok(Value::array(v))
        }

        // (@str addr-or-array offset array) — copy array contents to dest + offset*4
        // dest can be an address or an array (uses backing pointer).
        Syscall::Str => {
            let addr_val = evaluate(sexp.nth(1), image)?;
            let off_val = evaluate(sexp.nth(2), image)?;
            let arr_val = evaluate(sexp.nth(3), image)?;
            let base = match &addr_val {
                Value::Number(n) => {
                    let a = n.as_addr().map_err(|_| "str: first arg must be an address or array.")?;
                    if let super::number::Number::Addr(a) = a { a } else { unreachable!() }
                }
                Value::Array(a) => a.borrow().as_ptr() as usize,
                _ => return Err("str: first arg must be an address or array."),
            };
            let offset = if let Value::Number(n) = &off_val {
                n.as_i32().map_err(|_| "str: offset must be an integer.")? as usize
            } else {
                return Err("str: offset must be a number.");
            };
            if let Value::Array(a) = &arr_val {
                let borrowed = a.borrow();
                let dst = (base + offset * 4) as *mut u32;
                unsafe {
                    core::ptr::copy_nonoverlapping(borrowed.as_ptr(), dst, borrowed.len());
                }
                Ok(Value::Nil)
            } else {
                Err("str: third arg must be an array.")
            }
        }

        // (@unpack1to16 src dst) — expand 1bpp packed array into 16bpp.
        // Each bit in src becomes one u16 in dst (0x0000 or 0xFFFF).
        // src has N u32 words (N*32 pixels), dst needs N*16 u32 words.
        // Reads source bytes directly to avoid endian confusion.
        Syscall::Unpack1to16 => {
            let src_val = evaluate(sexp.nth(1), image)?;
            let dst_val = evaluate(sexp.nth(2), image)?;
            if let (Value::Array(src), Value::Array(dst)) = (&src_val, &dst_val) {
                let src_b = src.borrow();
                let mut dst_b = dst.borrow_mut();
                if dst_b.len() < src_b.len() * 16 {
                    return Err("unpack1to16: dst array too small.");
                }
                // Reinterpret src as bytes to avoid endian issues
                let src_bytes: &[u8] = unsafe {
                    core::slice::from_raw_parts(
                        src_b.as_ptr() as *const u8,
                        src_b.len() * 4,
                    )
                };
                let dst_u16: &mut [u16] = unsafe {
                    core::slice::from_raw_parts_mut(
                        dst_b.as_mut_ptr() as *mut u16,
                        dst_b.len() * 2,
                    )
                };
                let mut pixel = 0usize;
                for &byte in src_bytes.iter() {
                    for bit in (0..8).rev() {
                        dst_u16[pixel] = if (byte >> bit) & 1 != 0 { 0xFFFF } else { 0x0000 };
                        pixel += 1;
                    }
                }
                Ok(Value::Nil)
            } else {
                Err("unpack1to16: both arguments must be arrays.")
            }
        }
    }
}
