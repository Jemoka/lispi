//! JIT executor: lower a post-regalloc `LIRSegment` to ARMv6 machine
//! words, install a trampoline, run it.
//!
//! Single file by design — file-wide `static mut` globals stitch the
//! emitted code together with the Rust-side helpers. `JitExecutor::run`
//! is the only legitimate setter of those globals (RAII guard on
//! function exit).
//!
//! # The shadow-slot model
//!
//! Rust's `Rc<Value>` has unstable internal layout, so emitted code
//! cannot directly poke offsets inside boxed values. Instead, every
//! heap-flavored ARM register holds a `slot_id: u32` — an index into a
//! `repr(C)` `[ShadowSlot]` table. The emitted code reads/writes slot
//! fields with stable offsets (tag@0, payload@4, extra@8, src@12).
//! Helpers (`bind_local`, `cons_alloc`, ...) operate on slot ids and
//! mediate any interaction with the real `Image` / `Rc<Value>` graph.
//!
//! Convention: a JIT register holds *either* a raw 32-bit imm payload
//! (after `MovImm`, `Unbox`, arithmetic) *or* a slot id (after
//! `LoadLocal`, `Box`, `Cons`, ...). The IR4 opcode tells us which
//! universe each operand lives in — no runtime tag needed.

#![allow(dead_code)]

use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::language::ast::Value;
use crate::language::environment::{Binding, Image};
use crate::language::execute::evaluate;
use crate::language::number::Number;

use super::encodings::*;
use super::ir::Name;
use super::ir3::ImmNumber;
use super::ir4::{Cond, Instr, Instruction, LIRSegment, Register};

// =================== shadow slot ABI ===================

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ShadowSlot {
    /// off 0 — TAG_* discriminant.
    pub tag: u32,
    /// off 4 — imm payload, OR Cons car slot id, OR Array vec ptr.
    pub payload: u32,
    /// off 8 — Cons cdr slot id, OR Array len.
    pub extra: u32,
    /// off 12 — back-pointer for slots that mirror an `Rc<Value>` from
    /// the Image (Binding inner cell, or owned-by-this-executor Rc).
    pub src: u32,
}

const TAG_EMPTY: u32 = 0xFF;
const TAG_NIL: u32 = 0;
const TAG_INT: u32 = 1;
const TAG_UNSIGNED: u32 = 2;
const TAG_ADDR: u32 = 3;
const TAG_BOOL: u32 = 4;
const TAG_CONS: u32 = 5;
const TAG_CLOSURE: u32 = 6;
const TAG_ARRAY: u32 = 7;
const TAG_EXTERN: u32 = 8;
const TAG_STRING: u32 = 9;
const TAG_SYMBOL: u32 = 10;
const TAG_SPECIAL: u32 = 11;
const TAG_SYSCALL: u32 = 12;
const TAG_MACRO: u32 = 13;

/// Fixed slot capacity — sized large enough that long programs don't
/// run out. Lives in `.bss`-equivalent storage allocated at executor
/// construction.
const SLOT_CAPACITY: usize = 4096;

// =================== file-wide globals ===================

static mut IMAGE: *mut Image = core::ptr::null_mut();
static mut SLOTS_BASE: *mut ShadowSlot = core::ptr::null_mut();
static mut SLOTS_LEN: usize = 0;
static mut SLOT_BUMP: u32 = 0;
/// Pinned storage of `Rc<Value>` keep-alives mirroring slots that hold
/// reified runtime values (captures, closure args, cons cells when we
/// keep references back to source).
static mut SLOT_VALUES: *mut Vec<Option<Rc<Value>>> = core::ptr::null_mut();
/// Per-call `LocalId → Binding` map. `BindLocal` inserts; `LoadLocal`
/// / `StoreLocal` / `UnboxLocal` read/write through it. Grows lazily.
static mut LOCALS: *mut Vec<Option<Binding>> = core::ptr::null_mut();
/// Debug flag — when true, the next `JitExecutor::new()` dumps its LIR.
pub static mut JIT_DUMP_NEXT: bool = false;

unsafe fn slot_at(id: u32) -> *mut ShadowSlot {
    debug_assert!((id as usize) < unsafe { SLOTS_LEN });
    unsafe { SLOTS_BASE.add(id as usize) }
}

unsafe fn alloc_slot() -> u32 {
    let id = unsafe { SLOT_BUMP };
    unsafe { SLOT_BUMP += 1; }
    if (id as usize) >= unsafe { SLOTS_LEN } {
        panic!("JIT: slot table exhausted");
    }
    id
}

unsafe fn image_ref() -> &'static mut Image {
    unsafe { &mut *IMAGE }
}

unsafe fn slot_values() -> &'static mut Vec<Option<Rc<Value>>> {
    unsafe { &mut *SLOT_VALUES }
}

unsafe fn locals() -> &'static mut Vec<Option<Binding>> {
    unsafe { &mut *LOCALS }
}

fn locals_set(id: usize, b: Binding) {
    unsafe {
        let v = locals();
        if id >= v.len() { v.resize(id + 1, None); }
        v[id] = Some(b);
    }
}

fn locals_get(id: usize) -> Option<Binding> {
    unsafe {
        let v = locals();
        if id < v.len() { v[id].clone() } else { None }
    }
}

// =================== slot <-> Value bridge ===================

/// Materialize a `Value` into a freshly-allocated slot. Used when the
/// JIT receives a runtime `Value` (e.g. a capture) it must surface to
/// emitted code as a slot.
fn intern_value(v: &Value) -> u32 {
    unsafe {
        let id = alloc_slot();
        let slot = slot_at(id);
        match v {
            Value::Nil => {
                (*slot).tag = TAG_NIL;
            }
            Value::Bool(b) => {
                (*slot).tag = TAG_BOOL;
                (*slot).payload = if *b { 1 } else { 0 };
            }
            Value::Number(Number::Integer(i)) => {
                (*slot).tag = TAG_INT;
                (*slot).payload = *i as u32;
            }
            Value::Number(Number::Unsigned(u)) => {
                (*slot).tag = TAG_UNSIGNED;
                (*slot).payload = *u;
            }
            Value::Number(Number::Addr(a)) => {
                (*slot).tag = TAG_ADDR;
                (*slot).payload = *a as u32;
            }
            Value::Cons(_, _) => {
                (*slot).tag = TAG_EXTERN;
                slot_values().push(Some(Rc::new(v.clone())));
                (*slot).src = slot_values().len() as u32 - 1;
            }
            Value::Array(_) => {
                (*slot).tag = TAG_EXTERN;
                slot_values().push(Some(Rc::new(v.clone())));
                (*slot).src = slot_values().len() as u32 - 1;
            }
            _ => {
                (*slot).tag = TAG_EXTERN;
                slot_values().push(Some(Rc::new(v.clone())));
                (*slot).src = slot_values().len() as u32 - 1;
            }
        }
        id
    }
}

/// Reify a slot back into a `Value`. Used at exit to convert the
/// return slot, and inside helpers that need to surface a slot as an
/// `Rc<Value>` to interpreter-facing code.
fn reify_slot(id: u32) -> Value {
    unsafe {
        if id == 0 {
            return Value::Nil;
        }
        let slot = &*slot_at(id);
        match slot.tag {
            TAG_NIL => Value::Nil,
            TAG_BOOL => Value::Bool(slot.payload != 0),
            TAG_INT => Value::Number(Number::Integer(slot.payload as i32)),
            TAG_UNSIGNED => Value::Number(Number::Unsigned(slot.payload)),
            TAG_ADDR => Value::Number(Number::Addr(slot.payload as usize)),
            TAG_CONS => {
                let car = reify_slot(slot.payload);
                let cdr = reify_slot(slot.extra);
                Value::cons(car, cdr)
            }
            TAG_EXTERN => {
                let idx = slot.src as usize;
                slot_values()[idx]
                    .as_ref()
                    .map(|rc| (**rc).clone())
                    .unwrap_or(Value::Nil)
            }
            TAG_ARRAY => {
                let idx = slot.src as usize;
                slot_values()[idx]
                    .as_ref()
                    .map(|rc| (**rc).clone())
                    .unwrap_or(Value::Nil)
            }
            _ => {
                let idx = slot.src as usize;
                if idx < slot_values().len() {
                    slot_values()[idx]
                        .as_ref()
                        .map(|rc| (**rc).clone())
                        .unwrap_or(Value::Nil)
                } else {
                    Value::Nil
                }
            }
        }
    }
}

// =================== helper functions (extern "C") ===================
//
// Each helper takes its args in r0..r3 and returns its result in r0,
// AAPCS. The JIT lowers every helper-call LIR variant to:
//   ldr r12, =&helper_fn   ; literal pool
//   blx r12

unsafe extern "C" fn h_push_frame() -> u32 {
    unsafe { image_ref().push_frame(); }
    0
}

unsafe extern "C" fn h_pop_frame() -> u32 {
    unsafe { image_ref().pop_frame(); }
    0
}

unsafe extern "C" fn h_bind_local(slot_id: u32, local_id: u32, name_ptr: *const Name) -> u32 {
    unsafe {
        let name: Name = (*name_ptr).clone();
        let val = reify_slot(slot_id);
        let img = image_ref();
        img.insert(name.clone(), Rc::new(val));
        if let Some(b) = img.binding(&name) {
            locals_set(local_id as usize, b.clone());
        }
    }
    slot_id
}

unsafe extern "C" fn h_bind_immediate(imm_payload: u32, kind: u32, name_ptr: *const Name) -> u32 {
    // kind encodes the ImmNumber variant: 0=Integer, 1=Unsigned, 2=Addr, 3=Bool
    let v = match kind {
        0 => Value::Number(Number::Integer(imm_payload as i32)),
        1 => Value::Number(Number::Unsigned(imm_payload)),
        2 => Value::Number(Number::Addr(imm_payload as usize)),
        3 => Value::Bool(imm_payload != 0),
        _ => Value::Nil,
    };
    unsafe {
        let name: Name = (*name_ptr).clone();
        image_ref().insert(name.clone(), Rc::new(v.clone()));
        if let Some(b) = image_ref().binding(&name) {
            // local_id was implicitly the first arg of BindImmediate
            // in the IR — but our calling convention places kind in
            // r1 and name_ptr in r2 (a 3-arg call). We don't have
            // local_id here. Fall back to keying by name lookup at
            // load time; bind_immediate is used only via the IR3
            // codepath where it's paired with a separate LoadLocal.
            let _ = b;
        }
        intern_value(&v)
    }
}

unsafe extern "C" fn h_load_local(local_id: u32) -> u32 {
    if let Some(b) = locals_get(local_id as usize) {
        let v = b.borrow().as_ref().clone();
        return intern_value(&v);
    }
    0
}

unsafe extern "C" fn h_store_local(local_id: u32, slot_id: u32) -> u32 {
    if let Some(b) = locals_get(local_id as usize) {
        let v = reify_slot(slot_id);
        *b.borrow_mut() = Rc::new(v);
    }
    slot_id
}

unsafe extern "C" fn h_unbox_local(local_id: u32) -> u32 {
    if let Some(b) = locals_get(local_id as usize) {
        let val = b.borrow().as_ref().clone();
        match &val {
            Value::Number(Number::Integer(i)) => return *i as u32,
            Value::Number(Number::Unsigned(u)) => return *u,
            Value::Number(Number::Addr(a)) => return *a as u32,
            Value::Bool(b) => return if *b { 1 } else { 0 },
            _ => return 0,
        }
    }
    0
}

/// `binding_ptr` is what `Rc::as_ptr(&Binding)` returned at JIT-emit
/// time — i.e. a pointer to a `RefCell<Rc<Value>>`. We **must** access
/// it through the `RefCell` API; the inner `Rc<Value>` sits at offset
/// 4 inside the cell (offset 0 is the borrow counter), so treating
/// the pointer as `*const Rc<Value>` directly reads the borrow flag
/// and produces nonsense.
unsafe extern "C" fn h_store_capture(binding_ptr: *mut RefCell<Rc<Value>>, slot_id: u32) -> u32 {
    unsafe {
        let v = reify_slot(slot_id);
        *(*binding_ptr).borrow_mut() = Rc::new(v);
    }
    slot_id
}

unsafe extern "C" fn h_load_capture_slot(binding_ptr: *const RefCell<Rc<Value>>) -> u32 {
    unsafe {
        let cell_ref = (*binding_ptr).borrow();
        let v: Value = (**cell_ref).clone();
        intern_value(&v)
    }
}

/// Convert a baked `*const Value` (literal pool entry from
/// `LdrValuePtr`) into a fresh shadow slot. Used so the JIT can keep
/// the convention "heap regs hold slot ids", even for opcodes whose
/// IR-level form is "load the address of this Value literal."
unsafe extern "C" fn h_intern_value_ptr(value_ptr: *const Value) -> u32 {
    unsafe {
        let v: Value = (*value_ptr).clone();
        intern_value(&v)
    }
}

unsafe extern "C" fn h_cons(car_slot: u32, cdr_slot: u32) -> u32 {
    unsafe {
        let id = alloc_slot();
        let s = slot_at(id);
        (*s).tag = TAG_CONS;
        (*s).payload = car_slot;
        (*s).extra = cdr_slot;
        id
    }
}

unsafe extern "C" fn h_box_int(payload: u32) -> u32 {
    unsafe {
        let id = alloc_slot();
        let s = slot_at(id);
        (*s).tag = TAG_INT;
        (*s).payload = payload;
        id
    }
}

unsafe extern "C" fn h_box_uns(payload: u32) -> u32 {
    unsafe {
        let id = alloc_slot();
        let s = slot_at(id);
        (*s).tag = TAG_UNSIGNED;
        (*s).payload = payload;
        id
    }
}

unsafe extern "C" fn h_truthy(slot_id: u32) -> u32 {
    unsafe {
        if slot_id == 0 { return 0; }
        let s = &*slot_at(slot_id);
        let truthy = match s.tag {
            TAG_NIL => false,
            TAG_BOOL => s.payload != 0,
            TAG_INT | TAG_UNSIGNED => s.payload != 0,
            _ => true,
        };
        if truthy { 1 } else { 0 }
    }
}

unsafe extern "C" fn h_lognot(slot_id: u32) -> u32 {
    unsafe {
        if slot_id == 0 { return h_box_int(1); }
        let s = &*slot_at(slot_id);
        let truthy = match s.tag {
            TAG_NIL => false,
            TAG_BOOL => s.payload != 0,
            TAG_INT | TAG_UNSIGNED => s.payload != 0,
            _ => true,
        };
        h_box_int(if truthy { 0 } else { 1 })
    }
}

unsafe extern "C" fn h_xor(a: u32, b: u32) -> u32 {
    unsafe {
        let ta = h_truthy(a);
        let tb = h_truthy(b);
        h_box_int(((ta != 0) ^ (tb != 0)) as u32)
    }
}

unsafe extern "C" fn h_divsi3(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a.wrapping_div(b) }
}

unsafe extern "C" fn h_modsi3(a: i32, b: i32) -> i32 {
    if b == 0 { 0 } else { a.wrapping_rem(b) }
}

unsafe extern "C" fn h_nullp(slot_id: u32) -> u32 {
    unsafe {
        if slot_id == 0 { return 1; }
        let s = &*slot_at(slot_id);
        if s.tag == TAG_NIL { 1 } else { 0 }
    }
}

unsafe extern "C" fn h_car(slot_id: u32) -> u32 {
    unsafe {
        if slot_id == 0 { return 0; }
        let s = &*slot_at(slot_id);
        if s.tag == TAG_CONS { return s.payload; }
        if s.tag == TAG_EXTERN {
            let v = reify_slot(slot_id);
            return intern_value(&v.car());
        }
        0
    }
}

unsafe extern "C" fn h_cdr(slot_id: u32) -> u32 {
    unsafe {
        if slot_id == 0 { return 0; }
        let s = &*slot_at(slot_id);
        if s.tag == TAG_CONS { return s.extra; }
        if s.tag == TAG_EXTERN {
            let v = reify_slot(slot_id);
            return intern_value(&v.cdr());
        }
        0
    }
}

unsafe extern "C" fn h_escape(slot_id: u32) -> u32 {
    unsafe {
        let v = reify_slot(slot_id);
        match evaluate(Rc::new(v), image_ref()) {
            Ok(result) => intern_value(&result),
            Err(_) => 0,
        }
    }
}

unsafe extern "C" fn h_hits(slot_id: u32) -> u32 {
    unsafe {
        let v = reify_slot(slot_id);
        let count: u64 = match &v {
            Value::Closure(c) => c.hits.get(),
            Value::Macro(m) => m.closure.hits.get(),
            _ => 0,
        };
        h_box_uns(count as u32)
    }
}

unsafe extern "C" fn h_unbox(slot_id: u32) -> u32 {
    unsafe {
        if slot_id == 0 { return 0; }
        let s = &*slot_at(slot_id);
        s.payload
    }
}

// array helpers — operate via reify_slot / intern_value boundary.

unsafe extern "C" fn h_array_pack(list_slot: u32) -> u32 {
    let list = reify_slot(list_slot);
    let mut v: alloc::vec::Vec<u32> = alloc::vec::Vec::new();
    let mut cur = list;
    loop {
        match cur {
            Value::Nil => break,
            Value::Cons(head, tail) => {
                if let Value::Number(n) = head.as_ref() {
                    let u = n.as_u32().unwrap_or(0);
                    v.push(u);
                }
                cur = tail.as_ref().clone();
            }
            _ => break,
        }
    }
    intern_value(&Value::array(v))
}

unsafe extern "C" fn h_array_full(n_slot: u32, val_slot: u32) -> u32 {
    // Per ir2, both args are HeapReg (slot ids). Reify to u32.
    let n = unsafe { reify_to_u32(n_slot) } as usize;
    let v = unsafe { reify_to_u32(val_slot) };
    intern_value(&Value::array_fill(n, v))
}

unsafe extern "C" fn h_array_unpack(arr_slot: u32) -> u32 {
    let v = reify_slot(arr_slot);
    if let Value::Array(a) = &v {
        let b = a.borrow();
        let mut r = Value::Nil;
        for u in b.iter().rev() {
            r = Value::cons(Value::Number(Number::Unsigned(*u)), r);
        }
        intern_value(&r)
    } else {
        0
    }
}

// Per ir2, all array-op inputs (arr, idx, val, n, off, list) are
// HeapReg slot ids. GetIdx returns an ImmReg (raw u32 in r0); all
// others return HeapReg (slot id).

unsafe extern "C" fn h_array_getidx(arr_slot: u32, i_slot: u32) -> u32 {
    let v = reify_slot(arr_slot);
    let i = unsafe { reify_to_u32(i_slot) } as i32 as usize;
    let r = match &v {
        Value::Array(a) => {
            let b = a.borrow();
            if i < b.len() { b[i] } else { 0 }
        }
        Value::Number(Number::Addr(base)) => unsafe {
            let p = *base as *const u32;
            *p.add(i)
        },
        _ => 0,
    };
    // Match the interpreter: getidx returns Number::Unsigned(...).
    intern_value(&Value::Number(Number::Unsigned(r)))
}

unsafe extern "C" fn h_array_putidx(arr_slot: u32, i_slot: u32, val_slot: u32) -> u32 {
    let v = reify_slot(arr_slot);
    let i = unsafe { reify_to_u32(i_slot) } as i32 as usize;
    let val = unsafe { reify_to_u32(val_slot) };
    match &v {
        Value::Array(a) => {
            let mut b = a.borrow_mut();
            if i < b.len() { b[i] = val; }
        }
        Value::Number(Number::Addr(base)) => unsafe {
            let p = *base as *mut u32;
            *p.add(i) = val;
        },
        _ => {}
    }
    0
}

unsafe extern "C" fn h_array_readidx(arr_slot: u32, off_slot: u32, n_slot: u32) -> u32 {
    let v = reify_slot(arr_slot);
    let off = unsafe { reify_to_u32(off_slot) } as i32 as usize;
    let n = unsafe { reify_to_u32(n_slot) } as i32 as usize;
    let mut r = Value::Nil;
    if let Value::Array(a) = &v {
        let b = a.borrow();
        for i in (0..n).rev() {
            if off + i < b.len() {
                r = Value::cons(Value::Number(Number::Unsigned(b[off + i])), r);
            }
        }
    }
    intern_value(&r)
}

unsafe extern "C" fn h_array_fillidx(arr_slot: u32, off_slot: u32, list_slot: u32) -> u32 {
    let v = reify_slot(arr_slot);
    let off = unsafe { reify_to_u32(off_slot) } as i32 as usize;
    let list = reify_slot(list_slot);
    if let Value::Array(a) = &v {
        let mut b = a.borrow_mut();
        let mut cur = list;
        let mut i = 0usize;
        loop {
            match cur {
                Value::Nil => break,
                Value::Cons(head, tail) => {
                    if let Value::Number(n) = head.as_ref() {
                        let u = n.as_u32().unwrap_or(0);
                        if off + i < b.len() { b[off + i] = u; }
                    }
                    i += 1;
                    cur = tail.as_ref().clone();
                }
                _ => break,
            }
        }
    }
    0
}

unsafe extern "C" fn h_array_fullidx(arr_slot: u32, off_slot: u32, n_slot: u32, val_slot: u32) -> u32 {
    let v = reify_slot(arr_slot);
    let off = unsafe { reify_to_u32(off_slot) } as i32 as usize;
    let n = unsafe { reify_to_u32(n_slot) } as i32 as usize;
    let val = unsafe { reify_to_u32(val_slot) };
    if let Value::Array(a) = &v {
        let mut b = a.borrow_mut();
        for i in 0..n {
            if off + i < b.len() { b[off + i] = val; }
        }
    }
    0
}

/// Extract u32 payload from a slot (assumes imm/box tag).
unsafe fn reify_to_u32(slot_id: u32) -> u32 {
    unsafe {
        if slot_id == 0 { return 0; }
        let s = &*slot_at(slot_id);
        match s.tag {
            TAG_INT | TAG_UNSIGNED | TAG_ADDR | TAG_BOOL => s.payload,
            _ => 0,
        }
    }
}

// syscall helpers — stubs; populated by phase 6.
unsafe extern "C" fn h_uart_init() -> u32 { 0 }
unsafe extern "C" fn h_uart_get8() -> u32 { 0 }
unsafe extern "C" fn h_uart_put8(_b: u32) -> u32 { 0 }
unsafe extern "C" fn h_delay(_n: u32) -> u32 { 0 }
unsafe extern "C" fn h_monitor_clear() -> u32 { 0 }
unsafe extern "C" fn h_monitor_get() -> u32 { 0 }
unsafe extern "C" fn h_monitor_stop() -> u32 { 0 }
unsafe extern "C" fn h_zero32(_a: u32, _b: u32, _c: u32) -> u32 { 0 }
unsafe extern "C" fn h_full32(_a: u32, _b: u32, _c: u32, _v: u32) -> u32 { 0 }

// =================== JitExecutor ===================

pub(crate) struct JitExecutor {
    /// Emitted ARM machine code (one u32 per instruction word, plus
    /// the literal pool at the tail).
    code: Vec<u32>,
    /// Pinned `Binding`s for captures used by the LIR. Keeps the
    /// underlying `RefCell` allocations alive while raw pointers from
    /// `Binding::as_ptr()` are baked into the code.
    captures: Vec<Binding>,
    /// Pinned `Value` literals so `LdrValuePtr(Value)` can bake a
    /// stable address.
    value_literals: Vec<Rc<Value>>,
    /// Pinned Name (Symbol) literals.
    name_literals: Vec<Rc<Name>>,
    /// Backing storage for `slot_values()`.
    slot_value_storage: Vec<Option<Rc<Value>>>,
    /// Backing storage for the shadow-slot table.
    slot_storage: Vec<ShadowSlot>,
    /// Backing storage for the LocalId → Binding side table.
    locals_storage: Vec<Option<Binding>>,
    /// Byte offset of the entry point in `code` (always 0 for now).
    entry_offset: usize,
}

impl JitExecutor {
    pub(crate) fn new(seg: LIRSegment) -> Self {
        let (code, captures, value_literals, name_literals) = emit_segment(&seg);

        let slot_storage = vec![
            ShadowSlot { tag: TAG_NIL, payload: 0, extra: 0, src: 0 };
            SLOT_CAPACITY
        ];

        JitExecutor {
            code,
            captures,
            value_literals,
            name_literals,
            slot_value_storage: Vec::with_capacity(64),
            slot_storage,
            locals_storage: Vec::with_capacity(64),
            entry_offset: 0,
        }
    }

    /// Run the JIT'd code with `image` as the live runtime image.
    /// Returns the reified value of the return slot.
    /// Pretty-print the emitted machine code as hex words. Useful when
    /// the JIT-emitted function misbehaves and we want to inspect what
    /// was actually written into the buffer.
    pub(crate) fn dump_code(&self) {
        crate::println!("--- jit code dump ({} words) ---", self.code.len());
        for (i, w) in self.code.iter().enumerate() {
            crate::println!("  [{:3}] {:#010x}", i, w);
        }
        crate::println!("--- end dump ---");
    }

    pub(crate) fn run(&mut self, image: &mut Image) -> Result<Value, &'static str> {
        // Re-entrancy support. A JIT body that Escapes back into the
        // interpreter can ultimately reach another `JitExecutor::run()`
        // (e.g. when one jitted closure calls another). Each
        // executor's globals (slot table, locals side-table, …) are
        // private to its single call, so we save the previous values
        // on entry and restore them on exit.
        let saved_image  = unsafe { IMAGE };
        let saved_base   = unsafe { SLOTS_BASE };
        let saved_len    = unsafe { SLOTS_LEN };
        let saved_bump   = unsafe { SLOT_BUMP };
        let saved_values = unsafe { SLOT_VALUES };
        let saved_locals = unsafe { LOCALS };

        unsafe {
            IMAGE = image as *mut Image;
            SLOTS_BASE = self.slot_storage.as_mut_ptr();
            SLOTS_LEN = self.slot_storage.len();
            SLOT_BUMP = 1; // slot 0 reserved for nil
            SLOT_VALUES = &mut self.slot_value_storage as *mut _;
            LOCALS = &mut self.locals_storage as *mut _;
        }

        // Cache-coherency dance before jumping into freshly-written
        // code. On ARM1176 with caches enabled, the JIT writes go
        // into the D-cache; the I-cache fetch path can return stale
        // (or zero) data unless we explicitly:
        //   1. Clean+invalidate the entire D-cache to PoU
        //   2. DSB
        //   3. Invalidate the entire I-cache
        //   4. DSB
        //   5. Flush prefetch buffer (ISB)
        // The "test, clean, invalidate D-cache" loop (c7, c14, 3) is
        // the ARMv6 idiom — it returns with Z=0 while dirty lines
        // remain.
        unsafe {
            // ARM1176-specific cache coherency dance: clean+invalidate
            // the entire D-cache (single-shot, not a loop), DSB, then
            // invalidate the I-cache and flush the prefetch buffer.
            core::arch::asm!(
                "mcr p15, 0, {z}, c7, c14, 0",  // Clean+Invalidate D-cache
                "mcr p15, 0, {z}, c7, c10, 4",  // DSB
                "mcr p15, 0, {z}, c7, c5, 0",   // Invalidate I-cache
                "mcr p15, 0, {z}, c7, c10, 4",  // DSB
                "mcr p15, 0, {z}, c7, c5, 4",   // Prefetch flush (ISB)
                z = in(reg) 0u32,
                options(nostack, preserves_flags),
            );
        }

        let code_ptr = self.code.as_ptr();
        let entry_addr = unsafe { code_ptr.add(self.entry_offset / 4) };
        let entry: unsafe extern "C" fn() -> u32 = unsafe {
            core::mem::transmute(entry_addr)
        };

        let return_slot = unsafe { entry() };
        let result = if (return_slot as usize) < unsafe { SLOTS_LEN } {
            reify_slot(return_slot)
        } else {
            Value::Nil
        };

        // Tear down our locals/slot-value storage, then restore the
        // previous globals so the enclosing JIT call (if any) resumes
        // with its own state.
        unsafe {
            self.slot_value_storage.clear();
            self.locals_storage.clear();
            IMAGE = saved_image;
            SLOTS_BASE = saved_base;
            SLOTS_LEN = saved_len;
            SLOT_BUMP = saved_bump;
            SLOT_VALUES = saved_values;
            LOCALS = saved_locals;
        }

        Ok(result)
    }
}

// =================== code emission ===================
//
// Two passes:
//   1. Size each instruction (count machine words) to compute per-
//      block byte offsets and the literal-pool starting offset.
//   2. Emit every instruction, resolving branch targets and literal-
//      pool offsets.

/// What kind of word the literal pool entry holds. Each variant
/// resolves to a 32-bit absolute address at executor build time.
#[derive(Clone)]
enum LiteralKind {
    Value(Rc<Value>),
    Name(Rc<Name>),
    Binding(Binding),
    HelperFn(u32), // helper fn address as u32
    SlotsBase,
    Imm32(u32),
    /// Address of the file-wide `SLOTS_BASE` static — emitted code
    /// reads it at runtime to get the current shadow-slot table base.
    /// Used by all inline shadow-slot ops.
    SlotsBaseStaticAddr,
    /// Address of the file-wide `SLOT_BUMP` static — emitted code
    /// reads/writes it for inline slot allocation.
    SlotBumpStaticAddr,
}

struct Emitter {
    code: Vec<u32>,
    block_offsets: Vec<usize>, // byte offset of each block's first instr
    pool_entries: Vec<LiteralKind>,
    /// Map (instr_index_in_code, pool_entry_index) for fixups during
    /// the actual emit pass — each `ldr rd, =LIT` is encoded with a
    /// placeholder pc-relative offset, then patched after we know
    /// where the pool lives.
    pool_fixups: Vec<(usize, usize)>,
    captures: Vec<Binding>,
    value_literals: Vec<Rc<Value>>,
    name_literals: Vec<Rc<Name>>,
    /// Total bytes of spill area reserved by the prologue. Epilogue
    /// uses this to restore SP.
    spill_slots_bytes: u32,
    /// Register-list bitmask for the epilogue's `pop {…, pc}` —
    /// callee-saves used + PC.
    epilogue_mask: u16,
}

impl Emitter {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            block_offsets: Vec::new(),
            pool_entries: Vec::new(),
            pool_fixups: Vec::new(),
            captures: Vec::new(),
            value_literals: Vec::new(),
            name_literals: Vec::new(),
            spill_slots_bytes: 0,
            epilogue_mask: 0,
        }
    }

    fn intern_pool(&mut self, k: LiteralKind) -> usize {
        let idx = self.pool_entries.len();
        self.pool_entries.push(k);
        idx
    }

    fn push(&mut self, w: u32) {
        self.code.push(w);
    }

    /// `ldr rd, [pc, #off]` to a pool entry. The offset is patched
    /// after the layout pass.
    fn emit_pool_load(&mut self, rd: Register, pool_idx: usize) {
        // placeholder — patched later
        let instr_idx = self.code.len();
        self.code.push(ldr_pc_rel(rd, 0));
        self.pool_fixups.push((instr_idx, pool_idx));
    }

    fn emit_load_imm32(&mut self, rd: Register, v: u32) {
        load_imm32(&mut self.code, rd, v);
    }

    /// Emit a `ldr r12, =&helper; blx r12` pair.
    fn emit_helper_call(&mut self, fn_addr: u32) {
        let pool_idx = self.intern_pool(LiteralKind::HelperFn(fn_addr));
        self.emit_pool_load(Register::R12, pool_idx);
        self.code.push(blx(Register::R12));
    }
}

/// Helper to convert an `unsafe extern "C" fn` pointer to a `u32`
/// address for the literal pool. Rust's `function_item_references`
/// lint complains about `as usize` on a fn item; we route through a
/// fn pointer (already a pointer-sized scalar) to keep the cast
/// explicit. All extern "C" fn pointers are pointer-sized on armv6
/// so `transmute_copy` is safe.
#[inline]
fn fn_addr_ptr(p: *const ()) -> u32 {
    p as usize as u32
}

// ===================== inline shadow-slot ops =====================
//
// The regalloc places the source slot id in r0 and reads the result
// from r0. We keep that ABI but replace the `bl helper_fn` with
// inline machine code that does the tag/field load without a call.
// Saves the AAPCS handshake and the helper's prologue/epilogue
// — typically ~30 cycles vs ~5 inline.

/// Emit a load of the current `SLOTS_BASE` value into `dst`.
///   ldr dst, =&SLOTS_BASE   ; pool
///   ldr dst, [dst]
fn emit_load_slots_base(e: &mut Emitter, dst: Register) {
    let pool_idx = e.intern_pool(LiteralKind::SlotsBaseStaticAddr);
    e.emit_pool_load(dst, pool_idx);
    e.push(ldr_off12(dst, dst, 0));
}

/// Compute &slots[slot_id_reg] into `dst_ptr` using `tmp` as scratch.
/// `dst_ptr` may alias `tmp` but not `slot_id_reg`.
fn emit_slot_ptr(e: &mut Emitter, dst_ptr: Register, slot_id: Register, _tmp: Register) {
    emit_load_slots_base(e, dst_ptr);
    e.push(add_lsl_imm(dst_ptr, dst_ptr, slot_id, 4));
}

/// Inline `nullp`: r0 = (slot[r0].tag == TAG_NIL) ? 1 : 0.
///   ldr r12, =&SLOTS_BASE       (pool)
///   ldr r12, [r12]
///   add r12, r12, r0, lsl #4
///   ldr r12, [r12]              ; tag
///   cmp r12, #TAG_NIL
///   mov r0, #0
///   moveq r0, #1
/// 7 words. Replaces a 2-word bl + the helper's prologue/body/epilogue.
fn emit_inline_nullp(e: &mut Emitter) {
    emit_slot_ptr(e, Register::R12, Register::R0, Register::R12);
    e.push(ldr_off12(Register::R12, Register::R12, 0));   // tag
    e.push(cmp_imm8(Register::R12, TAG_NIL));
    e.push(mov_imm8(Register::R0, 0));
    e.push(mov_imm8_cond_rot(COND_EQ, Register::R0, 1, 0));
}

/// Inline `truthy`: r0 = falsy(slot) ? 0 : 1.
///   Falsy iff: tag == NIL  OR  (tag < TAG_CONS && payload == 0).
///   (TAG_NIL=0, TAG_INT=1, TAG_UNSIGNED=2, TAG_BOOL=4 are the
///    payload-bearing tags; anything ≥ TAG_CONS=5 is always truthy.)
///
///   ldr r12, =&SLOTS_BASE       ; pool
///   ldr r12, [r12]
///   add r12, r12, r0, lsl #4    ; r12 = &slot
///   ldr r1, [r12]               ; r1 = tag
///   ldr r2, [r12, #4]           ; r2 = payload
///   cmp r1, #TAG_CONS           ; tag ≥ CONS?
///   movhs r0, #1                ;   yes → truthy
///   bhs done
///   cmp r1, #TAG_NIL            ; tag == NIL?
///   moveq r0, #0                ;   yes → falsy
///   beq done
///   cmp r2, #0                  ; else: payload nonzero?
///   movne r0, #1
///   moveq r0, #0
///   done:
fn emit_inline_truthy(e: &mut Emitter) {
    emit_slot_ptr(e, Register::R12, Register::R0, Register::R12);
    e.push(ldr_off12(Register::R1, Register::R12, 0));        // tag
    e.push(ldr_off12(Register::R2, Register::R12, 4));        // payload
    // Fully predicated to avoid branches:
    //   r0 := 1  (default truthy)
    //   if tag == NIL    : r0 := 0
    //   else if tag < 5  : r0 := (payload != 0) ? 1 : 0
    //   else             : r0 := 1
    e.push(mov_imm8(Register::R0, 1));                         // default truthy
    e.push(cmp_imm8(Register::R1, TAG_NIL));
    e.push(mov_imm8_cond_rot(COND_EQ, Register::R0, 0, 0));    // NIL → 0
    // Reset flags for next check
    e.push(cmp_imm8(Register::R1, 5));
    // tag < 5 (LO) AND non-NIL handled above: now zero r0 iff payload==0
    // Set r0 = (tag<5 && payload==0) ? 0 : r0_current
    // We'll: if tag<5 (LO), cmp payload, #0; if eq → r0 = 0.
    // Emit two predicated ops with LO condition:
    //   cmplo r2, #0    (only sets flags if tag<5)
    //   moveq r0, #0    (set 0 only if eq flag set AND we hit cmplo)
    // ARM allows cmp<cond> via dp_imm with set_flags=1 and a cond.
    // We open-code it:
    let cond_lo = COND_CC; // unsigned lower
    let s = 1u32 << 20;
    let i = 1u32 << 25;
    let op_cmp = 0b1010 << 21;
    let rn_r2 = (Register::R2 as u32) << 16;
    let cmplo_r2_0 = (cond_lo << 28) | i | op_cmp | s | rn_r2 | 0;
    e.push(cmplo_r2_0);
    e.push(mov_imm8_cond_rot(COND_EQ, Register::R0, 0, 0));
}

/// Inline `lognot`: r0 = !truthy(slot). Compute truthy into r0, then
/// xor with 1.
fn emit_inline_lognot(e: &mut Emitter) {
    emit_inline_truthy(e);
    // r0 ^= 1
    e.push(eor(Register::R0, Register::R0, Register::R0)); // placeholder; need eor_imm
    // Simpler: rsb r0, r0, #1  → r0 = 1 - r0
    let cond = COND_AL << 28;
    let i = 1u32 << 25;
    let op_rsb = 0b0011 << 21;
    let rn_r0 = (Register::R0 as u32) << 16;
    let rd_r0 = (Register::R0 as u32) << 12;
    let imm = 1u32;
    // overwrite the eor placeholder
    let last = e.code.len() - 1;
    e.code[last] = cond | i | op_rsb | rn_r0 | rd_r0 | imm;
}

/// Inline `unbox`: r0 = slot[r0].payload (raw 32-bit value).
///   3 words via shadow-slot pointer compute + ldr.
fn emit_inline_unbox(e: &mut Emitter) {
    emit_slot_ptr(e, Register::R12, Register::R0, Register::R12);
    e.push(ldr_off12(Register::R0, Register::R12, 4));
}

/// Inline `box_number` (int-tagged): allocate a fresh slot, set
/// tag=TAG_INT, payload=r0, return the new slot id in r0.
///   ldr r12, =&SLOT_BUMP         ; pool
///   ldr r1, [r12]                ; r1 = SLOT_BUMP (new slot id)
///   add r2, r1, #1
///   str r2, [r12]                ; SLOT_BUMP++
///   ldr r12, =&SLOTS_BASE
///   ldr r12, [r12]
///   add r12, r12, r1, lsl #4     ; r12 = &slot[r1]
///   mov r2, #TAG_INT
///   str r2, [r12]                ; slot.tag = TAG_INT
///   str r0, [r12, #4]            ; slot.payload = r0
///   mov r0, r1                   ; r0 = slot id
fn emit_inline_box_int(e: &mut Emitter) {
    let bump_pool = e.intern_pool(LiteralKind::SlotBumpStaticAddr);
    e.emit_pool_load(Register::R12, bump_pool);
    e.push(ldr_off12(Register::R1, Register::R12, 0));
    e.push(add_imm8(Register::R2, Register::R1, 1));
    e.push(str_off12(Register::R2, Register::R12, 0));
    emit_load_slots_base(e, Register::R12);
    e.push(add_lsl_imm(Register::R12, Register::R12, Register::R1, 4));
    e.push(mov_imm8(Register::R2, TAG_INT));
    e.push(str_off12(Register::R2, Register::R12, 0));
    e.push(str_off12(Register::R0, Register::R12, 4));
    e.push(mov(Register::R0, Register::R1));
}

/// Build the prologue save mask: callee-saves used + LR. The pool
/// bit-positions follow ARM's `stm` register-list encoding.
fn build_save_mask(callee_saves_used: u32) -> u16 {
    let mut m: u16 = 0;
    for r in Register::POOL {
        if let Some(i) = r.pool_index() {
            if (callee_saves_used >> i) & 1 == 1 {
                m |= 1u16 << r.bit();
            }
        }
    }
    m |= 1u16 << Register::LR.bit();
    m
}

/// Number of machine words the prologue occupies. `push {regs}` is one
/// word; `sub sp, sp, #imm` is one word if the immediate fits an
/// 8-bit rotated literal, otherwise four (load_imm32 + sub).
fn compute_prologue_size(spill_slots: u32, _mask: u16) -> usize {
    let mut n = 1; // push
    if spill_slots > 0 {
        let bytes = spill_slots * 4;
        if bytes < 0x100 || encode_rotated_imm8(bytes).is_some() {
            n += 1;
        } else {
            n += 5; // load_imm32 (4) + sub rN, sp, rN (1)
        }
    }
    n
}

fn emit_prologue(e: &mut Emitter, spill_slots: u32, mask: u16) {
    e.push(push(mask));
    if spill_slots > 0 {
        let bytes = spill_slots * 4;
        if bytes < 0x100 {
            e.push(sub_imm8(Register::SP, Register::SP, bytes));
        } else if let Some((imm8, rot4)) = encode_rotated_imm8(bytes) {
            // sub_imm8 with rotation — emit raw via dp_imm path. We
            // open-code it: COND_AL | I=1 | OP_SUB | Rd=SP | Rn=SP
            let cond = COND_AL << 28;
            let i = 1u32 << 25;
            let op = 0b0010 << 21;
            let s = 0u32 << 20;
            let rn_f = (13u32) << 16; // SP=13
            let rd_f = (13u32) << 12;
            let imm = (rot4 << 8) | imm8;
            e.push(cond | i | op | s | rn_f | rd_f | imm);
        } else {
            // Materialize bytes in r12, then sub sp, sp, r12.
            e.emit_load_imm32(Register::R12, bytes);
            // sub sp, sp, r12
            e.push(sub(Register::SP, Register::SP, Register::R12));
        }
    }
}

fn size_instr_with_epilogue(i: &Instruction, spill_slots: u32, mask: u16) -> usize {
    if matches!(i, Instr::Ret) {
        // Replace with: (add sp ...) + pop {…, pc}
        let mut n = 1; // pop
        if spill_slots > 0 {
            let bytes = spill_slots * 4;
            if bytes < 0x100 || encode_rotated_imm8(bytes).is_some() {
                n += 1;
            } else {
                n += 5;
            }
        }
        let _ = mask;
        n
    } else {
        size_instr(i)
    }
}

fn cond_to_u32(c: Cond) -> u32 {
    match c {
        Cond::Eq => COND_EQ,
        Cond::Ne => COND_NE,
        Cond::Lt => COND_LT,
        Cond::Le => COND_LE,
        Cond::Gt => COND_GT,
        Cond::Ge => COND_GE,
    }
}

/// Encode a MovImm — uses the rotated-imm8 encoding when possible,
/// otherwise falls back to the 4-instruction load_imm32 sequence.
fn emit_mov_imm(e: &mut Emitter, rd: Register, n: ImmNumber) {
    let bits: u32 = match n {
        ImmNumber::Integer(i) => i as u32,
        ImmNumber::Unsigned(u) => u,
        ImmNumber::Addr(a) => a as u32,
    };
    if bits < 0x100 {
        e.push(mov_imm8(rd, bits));
    } else if let Some((imm8, rot4)) = encode_rotated_imm8(bits) {
        e.push(mov_imm8_rot(rd, imm8, rot4));
    } else {
        e.emit_load_imm32(rd, bits);
    }
}

/// Try to fit `v` as `imm8 ROR (2*rot4)` for some `(imm8, rot4)`.
fn encode_rotated_imm8(v: u32) -> Option<(u32, u32)> {
    for rot4 in 0..16u32 {
        let r = 2 * rot4;
        let rotated = v.rotate_left(r);
        if rotated < 0x100 {
            return Some((rotated, rot4));
        }
    }
    None
}

fn emit_mov(e: &mut Emitter, rd: Register, rm: Register) {
    // Always emit one word — even for identity movs — so block offsets
    // computed in the sizing pass match what's emitted.
    e.push(mov(rd, rm));
}

/// First pass: compute per-block byte offsets by sizing each
/// instruction. Branch instructions are 1 word; MovImm may be 1 or 4;
/// helper-calls are 2 (ldr+blx); pool loads are 1.
fn size_instr(i: &Instruction) -> usize {
    use Instr::*;
    match i {
        MovImm(_, n) => {
            let bits: u32 = match n {
                ImmNumber::Integer(i) => *i as u32,
                ImmNumber::Unsigned(u) => *u,
                ImmNumber::Addr(a) => *a as u32,
            };
            if bits < 0x100 || encode_rotated_imm8(bits).is_some() { 1 } else { 4 }
        }
        // LdrValuePtr is: ldr r0,=val + ldr r12,=fn + blx r12 + mov d,r0 = 4
        LdrValuePtr(..) => 4,
        // LdrNamePtr is still a raw pool load (the consumer is BindLocal
        // which takes a *const Name in r2)
        LdrNamePtr(..) => 1,
        // LdrCapture is: ldr r0,=bind + ldr r12,=fn + blx r12 + maybe mov d,r0
        // We over-budget at 4 words (the mov is always added; if d==r0 we
        // pad with a nop). Simpler to always emit 4 words.
        LdrCapture(..) => 4,
        StrCapture(..) => 4, // ldr r0,=bind + ldr r12,=fn + blx + mov
        // helper calls: ldr r12, =&fn (1) + blx r12 (1) = 2
        BindLocal | LoadLocal | StoreLocal | UnboxLocal
        | PushFrame | PopFrame | Xor
        | Div | Mod | Cons
        | Array | Full | Unpack | GetIdx | PutIdx | ReadIdx | FillIdx | FullIdx
        | Hits | Escape
        | UartInit | UartGet8 | UartPut8 | Delay
        | ClearMonitor | GetMonitor | StopMonitor | Zero32 | Full32 => 2,
        // inline ops — sized to match the bodies above.
        Nullp     => 7,
        Truthy    => 11,
        LogNot    => 12,
        Box       => 11,
        // cset is mov + cond-mov = 2
        Cset(..) => 2,
        // LdrOffset: off=0 is a direct ldr (1); off!=0 is a shadow-
        // slot read (slot_ptr=3 + ldr=1 = 4).
        LdrOffset(_, _, 0) => 1,
        LdrOffset(..) => 4,
        // CondBr/Phi should not appear post-regalloc; if they do they're
        // 1 word placeholders to keep sizing total correct.
        CondBr { .. } | Phi(..) => 1,
        // Identity moves filtered by regalloc; safe to size 1.
        Mov(..) => 1,
        // Everything else: 1 word.
        _ => 1,
    }
}

fn emit_segment(seg: &LIRSegment)
    -> (Vec<u32>, Vec<Binding>, Vec<Rc<Value>>, Vec<Rc<Name>>)
{
    let mut e = Emitter::new();

    // --- Prologue: save callee-saves used + LR, reserve spill slots ---
    let prologue_mask = build_save_mask(seg.callee_saves_used);
    let prologue_words = compute_prologue_size(seg.spill_slots, prologue_mask);

    // --- Pass 1: per-block byte offsets ---
    let mut offset_words: usize = prologue_words;
    e.block_offsets = Vec::with_capacity(seg.blocks.len());
    for blk in &seg.blocks {
        e.block_offsets.push(offset_words * 4);
        if blk.dead { continue; }
        for instr in &blk.instructions {
            offset_words += size_instr_with_epilogue(instr, seg.spill_slots, prologue_mask);
        }
    }
    let _total_code_words = offset_words;

    // Pre-set the segment-wide epilogue info on the emitter.
    // Epilogue pops the same set of registers we pushed, but with LR
    // swapped for PC (so the popped saved-LR value lands in PC — that
    // executes the return). NOT setting both — that would overcount.
    e.spill_slots_bytes = seg.spill_slots * 4;
    e.epilogue_mask = (prologue_mask & !(1u16 << Register::LR.bit()))
        | (1u16 << Register::PC.bit());

    // Emit prologue.
    emit_prologue(&mut e, seg.spill_slots, prologue_mask);

    // --- Pass 2: emit ---
    for (bi, blk) in seg.blocks.iter().enumerate() {
        if blk.dead { continue; }
        // sanity: code length matches the offset we predicted
        debug_assert_eq!(e.code.len(), e.block_offsets[bi] / 4,
            "block {} offset mismatch", bi);
        for instr in &blk.instructions {
            emit_instr(&mut e, instr);
        }
    }

    // --- Pass 3: patch pool loads ---
    let pool_start_words = e.code.len();
    // emit pool entries as placeholder words first (we'll fill below).
    for _ in 0..e.pool_entries.len() {
        e.code.push(0);
    }

    // Fill pool words. For Value/Name/Binding entries, store the
    // pinned address. For HelperFn, the function pointer.
    for (i, k) in e.pool_entries.iter().enumerate() {
        let word: u32 = match k {
            LiteralKind::Value(rc) => Rc::as_ptr(rc) as u32,
            LiteralKind::Name(rc) => Rc::as_ptr(rc) as u32,
            LiteralKind::Binding(b) => Rc::as_ptr(b) as u32,
            LiteralKind::HelperFn(addr) => *addr,
            LiteralKind::SlotsBase => unsafe { SLOTS_BASE as u32 },
            LiteralKind::Imm32(v) => *v,
            LiteralKind::SlotsBaseStaticAddr =>
                core::ptr::addr_of!(SLOTS_BASE) as u32,
            LiteralKind::SlotBumpStaticAddr =>
                core::ptr::addr_of!(SLOT_BUMP) as u32,
        };
        e.code[pool_start_words + i] = word;
    }

    // Patch each pc-relative ldr to point at its pool entry.
    let fixups = core::mem::take(&mut e.pool_fixups);
    for (instr_idx, pool_idx) in fixups {
        let pool_word_idx = pool_start_words + pool_idx;
        // PC at execution is instr_addr + 8, so byte offset from PC
        // to pool word is (pool_word_idx - instr_idx) * 4 - 8.
        let off = ((pool_word_idx as i32) - (instr_idx as i32)) * 4 - 8;
        let rd_bits = (e.code[instr_idx] >> 12) & 0xF;
        let rd: Register = decode_reg(rd_bits);
        e.code[instr_idx] = ldr_pc_rel(rd, off);
    }

    // Move pinned objects out to the executor.
    let JitOwned { captures, value_literals, name_literals } =
        collect_pinned(&e.pool_entries);

    (e.code, captures, value_literals, name_literals)
}

struct JitOwned {
    captures: Vec<Binding>,
    value_literals: Vec<Rc<Value>>,
    name_literals: Vec<Rc<Name>>,
}

fn collect_pinned(pool: &[LiteralKind]) -> JitOwned {
    let mut captures = Vec::new();
    let mut value_literals = Vec::new();
    let mut name_literals = Vec::new();
    for k in pool {
        match k {
            LiteralKind::Value(rc) => value_literals.push(rc.clone()),
            LiteralKind::Name(rc) => name_literals.push(rc.clone()),
            LiteralKind::Binding(b) => captures.push(b.clone()),
            _ => {}
        }
    }
    JitOwned { captures, value_literals, name_literals }
}

fn decode_reg(n: u32) -> Register {
    match n {
        0 => Register::R0, 1 => Register::R1, 2 => Register::R2, 3 => Register::R3,
        4 => Register::R4, 5 => Register::R5, 6 => Register::R6, 7 => Register::R7,
        8 => Register::R8, 9 => Register::R9, 10 => Register::R10, 11 => Register::R11,
        12 => Register::R12, 13 => Register::SP, 14 => Register::LR, 15 => Register::PC,
        _ => unreachable!(),
    }
}

fn emit_instr(e: &mut Emitter, instr: &Instruction) {
    use Instr::*;
    match instr {
        Mov(d, s) => emit_mov(e, *d, *s),
        MovImm(d, n) => emit_mov_imm(e, *d, *n),
        MovId(d, id) => {
            // LocalId encoded as an integer immediate. Most LocalIds
            // fit in 8 bits; fall back to load_imm32 otherwise.
            if id.0 < 0x100 {
                e.push(mov_imm8(*d, id.0));
            } else {
                e.emit_load_imm32(*d, id.0);
            }
        }

        LdrValuePtr(d, v) => {
            // Sequence: r0 ← &value-literal ; r12 ← &h_intern_value_ptr ;
            //           blx r12 ; mov d, r0.
            // Yields a fresh slot id in d. Sized to 4 words.
            let val_pool = e.intern_pool(LiteralKind::Value(Rc::new(v.clone())));
            e.emit_pool_load(Register::R0, val_pool);
            let fn_pool = e.intern_pool(LiteralKind::HelperFn(
                fn_addr_ptr(h_intern_value_ptr as unsafe extern "C" fn(*const Value) -> u32 as *const ())));
            e.emit_pool_load(Register::R12, fn_pool);
            e.push(blx(Register::R12));
            if *d != Register::R0 {
                e.push(mov(*d, Register::R0));
            } else {
                e.push(mov(Register::R0, Register::R0));
            }
        }
        LdrNamePtr(d, n) => {
            let pool_idx = e.intern_pool(LiteralKind::Name(Rc::new(n.clone())));
            e.emit_pool_load(*d, pool_idx);
        }
        LdrCapture(d, b) => {
            // Materialize the capture's current value into a fresh
            // shadow slot. Sequence:
            //   ldr r0, =&binding         ; pool ldr (1)
            //   ldr r12, =&h_load_capture_slot ; pool ldr (1)
            //   blx r12                   ; (1)
            //   mov d, r0   (if d != r0)  ; (1, only if needed)
            let bind_pool = e.intern_pool(LiteralKind::Binding(b.clone()));
            e.emit_pool_load(Register::R0, bind_pool);
            let fn_pool = e.intern_pool(LiteralKind::HelperFn(
                fn_addr_ptr(h_load_capture_slot as unsafe extern "C" fn(*const RefCell<Rc<Value>>) -> u32 as *const ())));
            e.emit_pool_load(Register::R12, fn_pool);
            e.push(blx(Register::R12));
            if *d != Register::R0 {
                e.push(mov(*d, Register::R0));
            } else {
                // padding nop to keep instr size constant at 4 words
                e.push(mov(Register::R0, Register::R0));
            }
        }
        StrCapture(b, src) => {
            // Sequence: mov r1, src ; ldr r0, =&binding ;
            //           ldr r12, =&h_store_capture ; blx r12
            e.push(mov(Register::R1, *src));
            let bind_pool = e.intern_pool(LiteralKind::Binding(b.clone()));
            e.emit_pool_load(Register::R0, bind_pool);
            let fn_pool = e.intern_pool(LiteralKind::HelperFn(
                fn_addr_ptr(h_store_capture as unsafe extern "C" fn(*mut RefCell<Rc<Value>>, u32) -> u32 as *const ())));
            e.emit_pool_load(Register::R12, fn_pool);
            e.push(blx(Register::R12));
        }
        LdrOffset(d, b, o) => {
            // SysGet32 uses offset 0 with `b` holding a raw MMIO
            // address — emit a direct ldr.
            // Car/Cdr/Unbox/Hits use offset 4 or 8 with `b` holding
            // a *slot id* — compute the slot pointer then ldr.
            if *o == 0 {
                e.push(ldr_off12(*d, *b, 0));
            } else {
                emit_slot_ptr(e, Register::R12, *b, Register::R12);
                e.push(ldr_off12(*d, Register::R12, *o));
            }
        }
        StrOffset(s, b, o) => {
            // Symmetric to LdrOffset. Currently only SysPut32 with off=0
            // uses this; the shadow-slot writes happen via dedicated
            // inline emitters above.
            e.push(str_off12(*s, *b, *o));
        }

        LoadSpill(d, s) => {
            let off = (s.0 as i32) * 4;
            e.push(ldr_off12(*d, Register::SP, off));
        }
        StoreSpill(s, r) => {
            let off = (s.0 as i32) * 4;
            e.push(str_off12(*r, Register::SP, off));
        }

        Add(d, a, b)    => e.push(add(*d, *a, *b)),
        Sub(d, a, b)    => e.push(sub(*d, *a, *b)),
        Mul(d, a, b)    => e.push(mul(*d, *a, *b)),
        Lshift(d, a, b) => e.push(lsl(*d, *a, *b)),
        Rshift(d, a, b) => e.push(lsr(*d, *a, *b)),
        BinOr(d, a, b)  => e.push(orr(*d, *a, *b)),
        BinAnd(d, a, b) => e.push(and(*d, *a, *b)),
        Mvn(d, a)       => e.push(mvn(*d, *a)),

        Cmp(a, b)    => e.push(cmp(*a, *b)),
        CmpImm(a, n) => {
            let bits: u32 = match n {
                ImmNumber::Integer(i) => *i as u32,
                ImmNumber::Unsigned(u) => *u,
                ImmNumber::Addr(a) => *a as u32,
            };
            if bits < 0x100 {
                e.push(cmp_imm8(*a, bits));
            } else {
                // load into r12 and compare register-register
                e.emit_load_imm32(Register::R12, bits);
                e.push(cmp(*a, Register::R12));
            }
        }
        Cset(d, c) => {
            let pair = cset(*d, cond_to_u32(*c));
            e.push(pair[0]);
            e.push(pair[1]);
        }

        // ---- helper calls ----
        PushFrame => e.emit_helper_call(fn_addr_ptr(h_push_frame as *const ())),
        PopFrame  => e.emit_helper_call(fn_addr_ptr(h_pop_frame as *const ())),
        BindLocal => e.emit_helper_call(fn_addr_ptr(h_bind_local as *const ())),
        LoadLocal => e.emit_helper_call(fn_addr_ptr(h_load_local as *const ())),
        StoreLocal => e.emit_helper_call(fn_addr_ptr(h_store_local as *const ())),
        UnboxLocal => e.emit_helper_call(fn_addr_ptr(h_unbox_local as *const ())),
        Box       => emit_inline_box_int(e),
        Truthy    => emit_inline_truthy(e),
        LogNot    => emit_inline_lognot(e),
        Xor       => e.emit_helper_call(fn_addr_ptr(h_xor as *const ())),
        Div       => e.emit_helper_call(fn_addr_ptr(h_divsi3 as *const ())),
        Mod       => e.emit_helper_call(fn_addr_ptr(h_modsi3 as *const ())),
        Cons      => e.emit_helper_call(fn_addr_ptr(h_cons as *const ())),
        Nullp     => emit_inline_nullp(e),
        Array     => e.emit_helper_call(fn_addr_ptr(h_array_pack as *const ())),
        Full      => e.emit_helper_call(fn_addr_ptr(h_array_full as *const ())),
        Unpack    => e.emit_helper_call(fn_addr_ptr(h_array_unpack as *const ())),
        GetIdx    => e.emit_helper_call(fn_addr_ptr(h_array_getidx as *const ())),
        PutIdx    => e.emit_helper_call(fn_addr_ptr(h_array_putidx as *const ())),
        ReadIdx   => e.emit_helper_call(fn_addr_ptr(h_array_readidx as *const ())),
        FillIdx   => e.emit_helper_call(fn_addr_ptr(h_array_fillidx as *const ())),
        FullIdx   => e.emit_helper_call(fn_addr_ptr(h_array_fullidx as *const ())),
        Hits      => e.emit_helper_call(fn_addr_ptr(h_hits as *const ())),
        Escape    => e.emit_helper_call(fn_addr_ptr(h_escape as *const ())),
        UartInit  => e.emit_helper_call(fn_addr_ptr(h_uart_init as *const ())),
        UartGet8  => e.emit_helper_call(fn_addr_ptr(h_uart_get8 as *const ())),
        UartPut8  => e.emit_helper_call(fn_addr_ptr(h_uart_put8 as *const ())),
        Delay     => e.emit_helper_call(fn_addr_ptr(h_delay as *const ())),
        ClearMonitor => e.emit_helper_call(fn_addr_ptr(h_monitor_clear as *const ())),
        GetMonitor   => e.emit_helper_call(fn_addr_ptr(h_monitor_get as *const ())),
        StopMonitor  => e.emit_helper_call(fn_addr_ptr(h_monitor_stop as *const ())),
        Zero32    => e.emit_helper_call(fn_addr_ptr(h_zero32 as *const ())),
        Full32    => e.emit_helper_call(fn_addr_ptr(h_full32 as *const ())),

        Dsb => e.push(DSB_SY),
        PrefetchFlush => {
            // mcr p15, 0, r0, c7, c5, 4  — flush prefetch buffer
            e.push(0xee07_0f95);
        }

        StackPush(mask) => e.push(push(*mask)),
        StackPop(mask)  => e.push(pop(*mask)),

        B(target) => {
            let here = e.code.len() * 4;
            let tgt = e.block_offsets.get(*target).copied().unwrap_or(0) as i32;
            let off = tgt - (here as i32) - 8;
            e.push(b(off));
        }
        Beq(target) => {
            let here = e.code.len() * 4;
            let tgt = e.block_offsets.get(*target).copied().unwrap_or(0) as i32;
            let off = tgt - (here as i32) - 8;
            e.push(b_cond(COND_EQ, off));
        }
        Bne(target) => {
            let here = e.code.len() * 4;
            let tgt = e.block_offsets.get(*target).copied().unwrap_or(0) as i32;
            let off = tgt - (here as i32) - 8;
            e.push(b_cond(COND_NE, off));
        }
        Ret => {
            // Restore SP, then pop callee-saves + PC (combined return).
            let bytes = e.spill_slots_bytes;
            if bytes > 0 {
                if bytes < 0x100 {
                    e.push(add_imm8(Register::SP, Register::SP, bytes));
                } else if let Some((imm8, rot4)) = encode_rotated_imm8(bytes) {
                    let cond = COND_AL << 28;
                    let i = 1u32 << 25;
                    let op = 0b0100 << 21; // ADD
                    let s = 0u32 << 20;
                    let rn_f = (13u32) << 16;
                    let rd_f = (13u32) << 12;
                    let imm = (rot4 << 8) | imm8;
                    e.push(cond | i | op | s | rn_f | rd_f | imm);
                } else {
                    e.emit_load_imm32(Register::R12, bytes);
                    e.push(add(Register::SP, Register::SP, Register::R12));
                }
            }
            e.push(pop(e.epilogue_mask));
        }

        CondBr { .. } | Phi(..) => {
            // Post-regalloc should have eliminated these. Emit a nop.
            e.push(mov(Register::R0, Register::R0));
        }
    }
}
