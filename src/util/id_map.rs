use crate::prelude::*;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub(crate) struct IdMap<K, V> {
    map: Vec<V>,
    _phantom: PhantomData<K>,
}

impl<K, V: Default> IdMap<K, V> {
    pub(crate) fn wipe(&mut self) {
        for item in &mut self.map {
            *item = Default::default();
        }
    }

    pub(crate) fn ensure_capacity(&mut self, enough: Id<K>) {
        let required = enough.as_usize();
        if required >= self.map.len() {
            self.map.resize_with(required + 1, Default::default);
        }
    }
}

impl<K, V> Default for IdMap<K, V> {
    fn default() -> Self {
        let map = vec![];
        let _phantom = PhantomData;
        Self { map, _phantom }
    }
}

impl<K, V> Index<Id<K>> for IdMap<K, V> {
    type Output = V;

    fn index(&self, id: Id<K>) -> &Self::Output {
        unsafe { self.map.get_unchecked(id.as_usize()) }
        //&self.map[id.as_usize()]
    }
}

impl<K, V> IndexMut<Id<K>> for IdMap<K, V> {
    fn index_mut(&mut self, id: Id<K>) -> &mut Self::Output {
        unsafe { self.map.get_unchecked_mut(id.as_usize()) }
        //&mut self.map[id.as_usize()]
    }
}

impl<'a, K, V> IntoIterator for &'a IdMap<K, V> {
    type Item = Id<K>;
    type IntoIter = IdRange<K>;

    fn into_iter(self) -> Self::IntoIter {
        IdRange::new_including(Id::default(), self.map.len() as u32)
    }
}
