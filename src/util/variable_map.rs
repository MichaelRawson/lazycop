use crate::prelude::*;
use crate::util::id_map::IdMap;

#[derive(Default)]
pub(crate) struct VariableMap {
    map: IdMap<Variable, Option<usize>>,
    count: usize,
}

impl VariableMap {
    pub(crate) fn get(&mut self, variable: Id<Variable>) -> usize {
        self.map.ensure_capacity(variable);
        if let Some(count) = self.map[variable] {
            return count;
        }
        let count = self.count;
        self.count += 1;
        self.map[variable] = Some(count);
        count
    }
}
