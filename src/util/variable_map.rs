use crate::prelude::*;
use crate::util::id_map::IdMap;
use std::cell::{Cell, RefCell};

#[derive(Default)]
pub struct VariableMap {
    map: RefCell<IdMap<Variable, usize>>,
    count: Cell<usize>,
}

impl VariableMap {
    pub fn get(&self, variable: Id<Variable>) -> usize {
        if let Some(count) = self.map.borrow().get(variable) {
            return *count;
        }
        let count = self.count.get();
        self.count.set(count + 1);
        self.map.borrow_mut().set(variable, count);
        count
    }
}
