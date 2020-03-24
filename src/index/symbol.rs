use crate::prelude::*;

pub struct Index<T> {
    map: Vec<Vec<T>>,
}

impl<T> Default for Index<T> {
    fn default() -> Self {
        let map = vec![];
        Self { map }
    }
}

impl<T> Index<T> {
    pub fn make_entry(&mut self, symbol: Id<Symbol>) -> &mut Vec<T> {
        if symbol.index() >= self.map.len() {
            self.map.resize_with(symbol.index() + 1, Default::default);
        }
        &mut self.map[symbol.index()]
    }

    pub fn query(&self, query: Id<Symbol>) -> &[T] {
        self.map
            .get(query.index())
            .map(|vec| vec.as_slice())
            .unwrap_or(&[])
    }
}
