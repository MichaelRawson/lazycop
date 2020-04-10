use crate::prelude::*;
use crate::util::id_map::IdMap;
use std::cell::{Cell, RefCell};

#[derive(Default)]
pub(crate) struct VariableMap {
    map: RefCell<IdMap<Variable, Option<usize>>>,
    count: Cell<usize>,
}

impl VariableMap {
    pub(crate) fn get(&self, variable: Id<Variable>) -> usize {
        self.map.borrow_mut().ensure_capacity(variable);
        if let Some(count) = self.map.borrow()[variable] {
            return count;
        }
        let count = self.count.get();
        self.count.set(count + 1);
        self.map.borrow_mut()[variable] = Some(count);
        count
    }
}
