use crate::prelude::*;
use std::collections::HashMap;

pub struct Index<T> {
    map: HashMap<Id<Symbol>, Vec<T>>,
}

impl<T> Default for Index<T> {
    fn default() -> Self {
        let map = HashMap::new();
        Self { map }
    }
}

impl<T> Index<T> {
    pub fn make_entry(&mut self, symbol: Id<Symbol>) -> &mut Vec<T> {
        self.map.entry(symbol).or_default()
    }

    pub fn query(&self, query: Id<Symbol>) -> &[T] {
        self.map
            .get(&query)
            .map(|vec| vec.as_slice())
            .unwrap_or(&[])
    }
}
