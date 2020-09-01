use crate::prelude::*;

#[derive(Default)]
pub(crate) struct Fresh {
    map: LUT<Variable, Option<usize>>,
    count: usize,
}

impl Fresh {
    pub(crate) fn get(&mut self, variable: Id<Variable>) -> usize {
        if variable >= self.map.len() {
            self.map.resize(variable + Offset::new(1));
        }
        if let Some(count) = self.map[variable] {
            return count;
        }
        let count = self.count;
        self.count += 1;
        self.map[variable] = Some(count);
        count
    }
}
