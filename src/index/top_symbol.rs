use crate::prelude::*;
use std::collections::HashMap;

#[derive(Default)]
pub struct Index<T> {
    map: HashMap<Id<Symbol>, T>,
}

impl<T: Default> Index<T> {
    pub fn make_entry(&mut self, symbol: Id<Symbol>) -> &mut T {
        self.map.entry(symbol).or_default()
    }

    pub fn query(&self, query: Id<Symbol>) -> Option<&T> {
        self.map.get(&query)
    }
}
