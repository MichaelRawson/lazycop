use crate::prelude::*;
use fnv::FnvHashMap;

#[derive(Default)]
pub(crate) struct Fresh {
    map: FnvHashMap<Id<Variable>, usize>,
    count: usize,
}

impl Fresh {
    pub(crate) fn get(&mut self, variable: Id<Variable>) -> usize {
        if let Some(count) = self.map.get(&variable) {
            return *count;
        }
        let count = self.count;
        self.count += 1;
        self.map.insert(variable, count);
        count
    }
}
