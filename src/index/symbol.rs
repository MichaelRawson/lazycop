use crate::prelude::*;

#[derive(Default)]
pub struct Index<T> {
    map: Vec<T>,
}

impl<T: Default> Index<T> {
    pub fn make_entry(&mut self, symbol: Id<Symbol>) -> &mut T {
        if self.map.len() <= symbol.index() {
            self.map.resize_with(symbol.index() + 1, Default::default);
        }
        &mut self.map[symbol.index()]
    }

    pub fn query(&self, query: Id<Symbol>) -> &T {
        &self.map[query.index()]
    }
}
