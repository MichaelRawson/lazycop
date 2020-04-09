use crate::prelude::*;
use std::marker::PhantomData;

pub struct IdMap<K, V> {
    map: Vec<Option<V>>,
    _phantom: PhantomData<K>,
}

impl<K, V> Default for IdMap<K, V> {
    fn default() -> Self {
        let map = vec![];
        let _phantom = PhantomData;
        Self { map, _phantom }
    }
}

impl<K, V> IdMap<K, V> {
    pub fn clear(&mut self) {
        let len = self.map.len();
        self.map.clear();
        self.map.resize_with(len, Default::default);
    }

    pub fn get(&self, id: Id<K>) -> Option<&V> {
        self.map.get(id.as_usize())?.as_ref()
    }

    pub fn set(&mut self, id: Id<K>, value: V) {
        let index = id.as_usize();
        if index >= self.map.len() {
            self.map.resize_with(index + 1, Default::default);
        }
        self.map[index] = Some(value)
    }
}

impl<K, V: Default> IdMap<K, V> {
    pub fn get_mut_default(&mut self, id: Id<K>) -> &mut V {
        let index = id.as_usize();
        if index >= self.map.len() {
            self.set(id, V::default());
        }
        if self.map[index].is_none() {
            self.map[index] = Some(V::default());
        }
        match &mut self.map[index] {
            Some(value) => value,
            None => unreachable!("just set this index"),
        }
    }
}
