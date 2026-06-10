//! Compile-time scope tracking for the JIT.
//!
//! `JitImage` is to compilation what `Image` is to interpretation: a
//! stack of named scopes with push/pop lifecycle and top-to-bottom
//! lookup. API names deliberately mirror `Image` so the parallel is
//! obvious — `push_frame`/`pop_frame`/`insert`/`binding`.
//!
//! The differences are intentional and minimal:
//! * `insert(name) -> LocalId` mints a fresh local instead of storing a
//!   `Value` — locals are the compile-time stand-in for per-invocation
//!   `Binding` slots that don't exist yet.
//! * `binding(name) -> Resolution` returns `Local` for names found in
//!   the compile-time stack, falling through to the wrapped `Image` for
//!   captures and globals. `Unbound` mirrors `Image::binding`'s `None`.
//!
//! `JitImage` borrows the runtime `Image` immutably for the lifetime of
//! compilation — cgen does not (and cannot) mutate runtime state.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use heapless::String;

use crate::language::constants::SYMB_NAME_LEN;
use crate::language::environment::{Binding, Image};

/// Stable identifier for a lexically-scoped local (param or `let` binding).
/// Assigned by `JitImage::insert` at IR-gen time. The JIT wrapper maps
/// each `LocalId` to the call's actual register / `Binding` at
/// invocation entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct LocalId(pub(super) u32);

/// Result of resolving a name during IR generation.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) enum Resolution {
    /// Name binds to a lexical local introduced inside the body.
    Local(LocalId),
    /// Name binds to a slot in the wrapped runtime `Image` — a capture
    /// from the closure's snapshot or a global. The `Binding` is stable
    /// across invocations and safe to bake into the IR.
    Capture(Binding),
    /// Name is not bound anywhere visible.
    Unbound,
}

/// Compile-time scope wrapping a runtime `Image`. Mirrors `Image`'s
/// frame-stack API.
#[allow(dead_code)]
pub(crate) struct JitImage<'a> {
    /// Wrapped runtime image. Borrowed immutably — cgen does not mutate.
    image: &'a Image,
    /// Stack of compile-only frames. Bottom = outermost compile scope,
    /// top = innermost. Lookups walk top-to-bottom.
    frames: Vec<BTreeMap<String<SYMB_NAME_LEN>, LocalId>>,
    /// Monotonic counter for minting fresh `LocalId`s.
    next_local: u32,
}

#[allow(dead_code)]
impl<'a> JitImage<'a> {
    pub fn new(image: &'a Image) -> Self {
        JitImage {
            image,
            frames: vec![BTreeMap::new()],
            next_local: 0,
        }
    }

    /// Push an empty compile-time scope. Mirrors `Image::push_frame`.
    pub fn push_frame(&mut self) {
        self.frames.push(BTreeMap::new());
    }

    /// Pop the top compile-time scope. Mirrors `Image::pop_frame`.
    pub fn pop_frame(&mut self) {
        debug_assert!(
            self.frames.len() > 1,
            "cannot pop the outermost compile frame"
        );
        self.frames.pop();
    }

    /// Mint a fresh `LocalId` for `name` in the top scope and return it.
    /// Mirrors `Image::insert` (which creates a fresh `Binding`).
    pub fn insert(&mut self, name: String<SYMB_NAME_LEN>) -> LocalId {
        let id = LocalId(self.next_local);
        self.next_local += 1;
        self.frames.last_mut().unwrap().insert(name, id);
        id
    }

    /// Total number of `LocalId`s minted during this compilation.
    /// The call-entry trampoline uses this to pre-allocate the per-call
    /// `Vec<Binding>` indexed by `LocalId`.
    pub fn num_locals(&self) -> u32 {
        self.next_local
    }

    /// Resolve `name`, searching top-to-bottom through compile-time
    /// frames first, then falling through to the wrapped runtime image
    /// for captures / globals. Mirrors `Image::binding`.
    pub fn binding(&self, name: &str) -> Resolution {
        for frame in self.frames.iter().rev() {
            if let Some(&id) = frame.get(name) {
                return Resolution::Local(id);
            }
        }
        match self.image.binding(name) {
            Some(b) => Resolution::Capture(b.clone()),
            None => Resolution::Unbound,
        }
    }
}
