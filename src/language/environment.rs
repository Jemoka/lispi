//! LISP interpreter environment infrastructure
//!
//! Uses Rc<RefCell<..>> as a poor man's garbage collector: when the last
//! environment referencing a Binding is dropped, Rc's refcount hits zero
//! and the value is freed automatically.
//!
//! The old design used a central State table (StateID -> Rc<Value>) that
//! held an Rc to every value ever allocated, so refcounts never reached
//! zero and values leaked for the lifetime of the interpreter.

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use core::cell::RefCell;
use heapless::String;

use super::ast;
use super::constants::SYMB_NAME_LEN;

/// A single mutable binding slot, shareable across multiple environments.
///
///   Rc<          -- outer: lets multiple environments (e.g. a closure and
///                   its defining scope) point to the *same* binding.
///                   Cloning just bumps a refcount; when the last clone
///                   drops, the binding is freed.
///     RefCell<   -- middle: interior mutability so `set!` can mutate a
///                   binding in-place without &mut on every environment
///                   that shares it.
///       Rc<Value>-- inner: the actual LISP value, already Rc-wrapped
///                   because Values appear inside cons cells, closures, etc.
///     >
///   >
pub type Binding = Rc<RefCell<Rc<ast::Value>>>;

/// Maps symbol names to their bindings.
///
/// Cloning an Environment is cheap: it clones the Rc pointers (bumps
/// refcounts), not the underlying RefCells or Values. Two cloned
/// environments that share a Binding will see each other's `set!`
/// mutations — this is what gives closures access to mutable state.
pub type Environment = BTreeMap<String<SYMB_NAME_LEN>, Binding>;

#[derive(Debug, Clone)]
pub struct Image {
    /// the current environment (symbol names -> shared mutable bindings)
    pub e: Environment,
}

impl Image {
    pub fn new() -> Self {
        Image { e: BTreeMap::new() }
    }

    /// Create a **fresh** binding for `name` with `value`.
    ///
    /// Always allocates a new RefCell, so this binding is independent of
    /// any prior binding for the same name. Use this for `define` and
    /// for binding closure parameters.
    pub fn insert(&mut self, name: String<SYMB_NAME_LEN>, value: Rc<ast::Value>) {
        self.e.insert(name, Rc::new(RefCell::new(value)));
    }

    /// Look up the current value of `name`.
    ///
    /// Borrows the RefCell momentarily to clone the inner Rc<Value>.
    /// Returns None if `name` is unbound.
    pub fn get(&self, name: &str) -> Option<Rc<ast::Value>> {
        self.e.get(name).map(|binding| binding.borrow().clone())
    }

    /// Get the raw Binding for `name` (the shared mutable slot).
    ///
    /// Needed for `set!`: mutating *through* the Binding ensures every
    /// environment sharing it (e.g. a closure that captured it) sees
    /// the new value.
    ///
    ///   image.binding("x").unwrap().replace(new_val);
    pub fn binding(&self, name: &str) -> Option<&Binding> {
        self.e.get(name)
    }
}
