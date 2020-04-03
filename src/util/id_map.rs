use crate::prelude::*;
use std::marker::PhantomData;

pub struct IdMap<K, V> {
    map: Vec<V>,
    _phantom: PhantomData<Id<K>>,
}

impl<K, V> Default for IdMap<K, V> {
    fn default() -> Self {
        let map = vec![];
        let _phantom = PhantomData;
        Self { map, _phantom }
    }
}

impl<K, V> IdMap<K, V> {
    pub fn get(&self, id: Id<K>) -> Option<&V> {
        self.map.get(id.index())
    }
}

impl<K, V: Default> IdMap<K, V> {
    pub fn entry(&mut self, id: Id<K>) -> &mut V {
        let index = id.index();
        if index >= self.map.len() {
            self.map.resize_with(index + 1, Default::default);
        }
        &mut self.map[index]
    }
}
