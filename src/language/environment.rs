//! LISP interpreter environment infrastructure

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use heapless::String;

use super::ast::{self, Symbol};
use super::constants::SYMB_NAME_LEN;

/// a good key type for the environment
pub type StateID = usize;
pub type Environment = BTreeMap<String<SYMB_NAME_LEN>, StateID>;
pub type State = BTreeMap<StateID, Rc<ast::Value>>;

/// a global image (i.e. the "state"); which maps ids to symbol values
#[derive(Debug, Clone)]
pub struct Image {
    /// the current environment table (symbol names to state ids)
    pub e: Environment,
    /// the current state table (state ids to Rc<Value>)
    pub s: State,
    /// the next state id to be allocated
    pub next_id: StateID,
}

impl Image {
    pub fn new() -> Self {
        Image {
            e: BTreeMap::new(),
            s: BTreeMap::new(),
            next_id: 0,
        }
    }

    pub fn lookup(&self, name: &str) -> Option<StateID> {
        self.e.get(name).cloned()
    }
    pub fn value(&self, id: StateID) -> Option<Rc<ast::Value>> {
        self.s.get(&id).cloned()
    }
    pub fn insert(&mut self, name: String<SYMB_NAME_LEN>, value: Rc<ast::Value>) -> StateID {
        let id = self.next_id;
        self.next_id += 1;
        self.e.insert(name, id);
        self.s.insert(id, value);
        id
    }
    pub fn get(&self, name: &str) -> Option<Rc<ast::Value>> {
        self.lookup(name).and_then(|id| self.value(id))
    }
}
