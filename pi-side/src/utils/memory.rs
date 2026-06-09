use embedded_alloc::LlffHeap as Heap;

//// heap stuff ////

#[unsafe(no_mangle)]
pub extern "C" fn __aeabi_unwind_cpp_pr0() {}

unsafe extern "C" {
    safe static __heap_start__: [u32; 0];
    safe static __heap_end__: [u32; 0];
}

#[inline(always)]
pub fn heap_start() -> usize {
    __heap_start__.as_ptr() as usize
}
#[inline(always)]
pub fn heap_end() -> usize {
    __heap_end__.as_ptr() as usize
}

/// set our global heap
#[global_allocator]
pub static HEAP: Heap = Heap::empty();

/// initialize heap using the linker symbols we defined in the linker script
pub fn init_heap() {
    let size = heap_end() - heap_start();

    // cast a region of maybeuninit from heap_start to heap_end as a slice of MaybeUninit<u8>
    let heap_start = heap_start() as *mut core::mem::MaybeUninit<u8>;
    let heap = unsafe { core::slice::from_raw_parts_mut(heap_start, size) };

    unsafe { HEAP.init(heap.as_mut_ptr() as usize, heap.len()) }
}

//// ===================== Identity-map MMU ====================
//
// Full first-level page table identity-mapping the entire 4GB
// address space as 1MB sections. Every entry is full R/W
// (AP[2:0]=0b011), domain 0; DACR is set so domain 0 is Manager,
// which makes the AP bits a non-check (everything is permitted).
//
// DRAM regions get write-back cacheable attrs so the D-cache helps
// every load/store. The Pi MMIO range at [0x2000_0000, 0x2200_0000)
// is mapped as *device shared* (uncached) so volatile peripheral
// writes (UART, watchdog, …) bypass the cache and reach the actual
// registers — no per-access cache maintenance needed.

/// First-level descriptor for a 1MB section at base address `base`.
/// `cacheable = true` → write-back normal memory;
/// `cacheable = false` → device shared (uncached). All other knobs
/// fixed: full R/W (AP[2:0] = 0b011), domain 0, executable.
#[inline(always)]
const fn section_descriptor(base: u32, cacheable: bool) -> u32 {
    // ARM1176 first-level section format (TRM B4-31):
    //   bits[1:0]   = 0b10                (section)
    //   bit 2       = B
    //   bit 3       = C
    //   bit 4       = XN (0 = executable)
    //   bits[8:5]   = domain (0)
    //   bit 9       = IMP (0)
    //   bits[11:10] = AP[1:0]              (0b11 = R/W user)
    //   bits[14:12] = TEX
    //   bit 15      = AP[2]                (0 → AP[2:0]=0b011)
    //   bit 16      = S (0)
    //   bit 17      = nG (0 = global)
    //   bit 18      = 0                    (section, not supersection)
    //   bit 19      = NS (0)
    //   bits[31:20] = section base
    let ap = 0b11u32 << 10; // AP[1:0] = 11
    let (c, b) = if cacheable {
        // TEX=000, C=1, B=1: outer/inner write-back, no write-allocate.
        (1u32 << 3, 1u32 << 2)
    } else {
        // TEX=000, C=0, B=1: device shared, uncached.
        (0u32, 1u32 << 2)
    };
    (base & 0xFFF00000) | ap | c | b | 0b10
}

/// First-level page table. 4096 × 1MB sections = 4GB. 16KB total.
/// Must be 16KB-aligned for ARM1176 TTBR0.
#[repr(C, align(16384))]
struct L1Table {
    entries: [u32; 4096],
}

static mut L1_PAGE_TABLE: L1Table = L1Table { entries: [0; 4096] };

/// Bring up the MMU with a full identity-mapped first-level page
/// table and enable the D-cache. Must be called *before* any non-
/// trivial Rust code that depends on cacheability or MMIO.
///
/// After this returns:
///   * Every 1MB section is identity-mapped: VA == PA.
///   * `[0, 0x2000_0000)` and `[0x2200_0000, 0x1_0000_0000)` are
///     write-back cacheable normal memory.
///   * `[0x2000_0000, 0x2200_0000)` is device shared (uncached) so
///     existing volatile MMIO writes keep reaching the peripherals.
///   * All 16 domains are Manager → AP bits are not a filter →
///     every access is permitted (read and write).
pub unsafe fn mmu_init() {
    // 1. Fill the identity-mapping L1 table.
    let tbl = unsafe { &mut *core::ptr::addr_of_mut!(L1_PAGE_TABLE) };
    let mut i: u32 = 0;
    while i < 4096 {
        let base = i << 20;
        let cacheable = !(0x2000_0000..0x2200_0000).contains(&base);
        tbl.entries[i as usize] = section_descriptor(base, cacheable);
        i += 1;
    }
    let table_addr = tbl.entries.as_ptr() as u32;

    // 2. Point TTBR0 at the table.
    unsafe { core::arch::asm!("mcr p15, 0, {0}, c2, c0, 0", in(reg) table_addr); }

    // 3. TTBCR=0 → always use TTBR0 (no split address space).
    unsafe { core::arch::asm!("mcr p15, 0, {0}, c2, c0, 2", in(reg) 0u32); }

    // 4. DACR: all 16 domains = Manager (0b11 × 16 = 0xFFFFFFFF) so
    //    AP bits are effectively bypassed.
    unsafe { core::arch::asm!("mcr p15, 0, {0}, c3, c0, 0", in(reg) 0xFFFFFFFFu32); }

    // 5. Invalidate the entire TLB so any stale boot entries can't
    //    shadow our identity map.
    unsafe { core::arch::asm!("mcr p15, 0, {0}, c8, c7, 0", in(reg) 0u32); }

    // 6. SCTLR: enable MMU (bit 0) and D-cache (bit 2). I-cache (bit
    //    12) was already turned on in boot.
    let mut sctlr: u32;
    unsafe { core::arch::asm!("mrc p15, 0, {0}, c1, c0, 0", out(reg) sctlr); }
    sctlr |= 1 << 0;
    sctlr |= 1 << 2;
    unsafe { core::arch::asm!("mcr p15, 0, {0}, c1, c0, 0", in(reg) sctlr); }

    // 7. Prefetch flush so subsequent instruction fetches use the new
    //    translation regime.
    unsafe { core::arch::asm!("mcr p15, 0, {0}, c7, c5, 4", in(reg) 0u32); }
}

//// barriers ////

/// Data Storage Barrier
#[inline(always)]
pub fn dsb() {
    unsafe { ::core::arch::asm!("mcr p15, 0, {t}, c7, c10, 4", t = in(reg) 0) }
}

/// Data Memory Barrier
#[inline(always)]
#[allow(dead_code)]
pub fn dmb() {
    unsafe { ::core::arch::asm!("mcr p15, 0, {t}, c7, c10, 5", t = in(reg) 0) }
}

//// writing ////

/// Stick Something Somewhere
/// Remember to potentially use DMB if you write across devices
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn put32(addr: usize, value: u32) {
    unsafe { ::core::ptr::write_volatile(addr as *mut u32, value) }
}

/// Get Something  from Somewhere
/// Remember to potentially use DMB if you write across devices
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn get32(addr: usize) -> u32 {
    unsafe { ::core::ptr::read_volatile(addr as *const u32) }
}

//// flushes ////
#[allow(unused)]
pub fn prefetch_flush() {
    unsafe {
        core::arch::asm!("mcr p15, 0, {t}, c7, c5, 4", t = in(reg) 0);
    }
}

#[macro_export]
macro_rules! prefetch_flush {
    ($donate:tt) => {
        concat!(
            "mov ",
            stringify!($donate),
            ", #0\n",
            "mcr p15, 0, ",
            stringify!($donate),
            ", c7, c5, 4"
        )
    };
}
