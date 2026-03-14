//! LISP interpreter environment infrastructure
//!
//! Uses a frame stack (scope chain) instead of a single flat BTreeMap.
//! Each function call / let-block pushes a frame; returning pops it.
//! This avoids cloning the entire environment on every call.
//!
//! Frames are either Owned (mutable, for params/let-bindings) or
//! Shared (Rc, for captured closure environments — push is O(1)).
//!
//! Bindings still use Rc<RefCell<..>> so that `set` mutations propagate
//! through shared references (e.g. a closure and its defining scope).

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use heapless::String;

use super::ast;
use super::constants::SYMB_NAME_LEN;

/// A single mutable binding slot, shareable across multiple environments.
pub type Binding = Rc<RefCell<Rc<ast::Value>>>;

/// A flat map of symbol names to bindings.
/// Used by closures to store their captured environment snapshot.
pub type Environment = BTreeMap<String<SYMB_NAME_LEN>, Binding>;

/// A scope frame: either owned (mutable) or shared (read-only, cheap to push).
#[derive(Debug, Clone)]
enum Frame {
    /// Mutable frame for params, let-bindings, top-level set.
    Owned(Environment),
    /// Shared frame from a closure's captured env. Push is O(1) (Rc::clone).
    Shared(Rc<Environment>),
}

impl Frame {
    fn get(&self, name: &str) -> Option<&Binding> {
        match self {
            Frame::Owned(m) => m.get(name),
            Frame::Shared(m) => m.get(name),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Image {
    /// Stack of scope frames. Bottom = global, top = innermost scope.
    /// Lookups search top-to-bottom; inserts go into the top (Owned) frame.
    frames: Vec<Frame>,
}

impl Image {
    pub fn new() -> Self {
        Image {
            frames: vec![Frame::Owned(BTreeMap::new())],
        }
    }

    /// Push an empty scope frame (for params, let-bindings, etc.).
    pub fn push_frame(&mut self) {
        self.frames.push(Frame::Owned(BTreeMap::new()));
    }

    /// Push a captured environment as a shared read-only frame.
    /// O(1) — just bumps the Rc refcount.
    pub fn push_env(&mut self, env: &Rc<Environment>) {
        self.frames.push(Frame::Shared(Rc::clone(env)));
    }

    /// Pop the top scope frame.
    pub fn pop_frame(&mut self) {
        debug_assert!(self.frames.len() > 1, "cannot pop the global frame");
        self.frames.pop();
    }

    /// Create a **fresh** binding for `name` with `value` in the top frame.
    /// The top frame must be Owned.
    pub fn insert(&mut self, name: String<SYMB_NAME_LEN>, value: Rc<ast::Value>) {
        match self.frames.last_mut().unwrap() {
            Frame::Owned(m) => {
                m.insert(name, Rc::new(RefCell::new(value)));
            }
            Frame::Shared(_) => panic!("insert into shared frame"),
        }
    }

    /// Look up the current value of `name`, searching top-to-bottom.
    pub fn get(&self, name: &str) -> Option<Rc<ast::Value>> {
        for frame in self.frames.iter().rev() {
            if let Some(binding) = frame.get(name) {
                return Some(binding.borrow().clone());
            }
        }
        None
    }

    /// Get the raw Binding for `name` (the shared mutable slot),
    /// searching top-to-bottom.
    pub fn binding(&self, name: &str) -> Option<&Binding> {
        for frame in self.frames.iter().rev() {
            if let Some(binding) = frame.get(name) {
                return Some(binding);
            }
        }
        None
    }

    /// Flatten all frames into a single Environment for closure capture,
    /// wrapped in Rc for cheap sharing.
    pub fn snapshot(&self) -> Rc<Environment> {
        let mut env = BTreeMap::new();
        for frame in &self.frames {
            match frame {
                Frame::Owned(m) => {
                    for (k, v) in m {
                        env.insert(k.clone(), Rc::clone(v));
                    }
                }
                Frame::Shared(m) => {
                    for (k, v) in m.as_ref() {
                        env.insert(k.clone(), Rc::clone(v));
                    }
                }
            }
        }
        Rc::new(env)
    }
}
