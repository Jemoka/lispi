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
pub fn put32(addr: usize, value: u32) {
    unsafe { ::core::ptr::write_volatile(addr as *mut u32, value) }
}

/// Get Something  from Somewhere
/// Remember to potentially use DMB if you write across devices
#[inline(always)]
#[allow(dead_code)]
pub fn get32(addr: usize) -> u32 {
    unsafe { ::core::ptr::read_volatile(addr as *const u32) }
}

