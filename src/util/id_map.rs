use crate::prelude::*;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub(crate) trait Reset {
    fn reset(&mut self);
}

impl<T> Reset for Option<T> {
    fn reset(&mut self) {
        *self = None;
    }
}

impl<T> Reset for Block<T> {
    fn reset(&mut self) {
        self.clear();
    }
}

pub(crate) struct IdMap<K, V> {
    items: Block<V>,
    _phantom: PhantomData<K>,
}

impl<K, V: Default> IdMap<K, V> {
    pub(crate) fn ensure_capacity(&mut self, required: Id<K>) {
        let required = required.transmute();
        if required >= self.items.len() {
            self.items.resize_default(required);
        }
    }
}

impl<K, V: Reset> IdMap<K, V> {
    pub(crate) fn reset(&mut self) {
        for id in self.items.range() {
            self.items[id].reset();
        }
    }
}

impl<K, V: Clone> IdMap<K, V> {
    pub(crate) fn copy_from(&mut self, other: &Self) {
        self.items.clear();
        self.items.extend_from_slice(&other.items.as_ref());
    }
}

impl<K, V> Default for IdMap<K, V> {
    fn default() -> Self {
        let items = Block::default();
        let _phantom = PhantomData;
        Self { items, _phantom }
    }
}

impl<K, V> Index<Id<K>> for IdMap<K, V> {
    type Output = V;

    fn index(&self, id: Id<K>) -> &Self::Output {
        &self.items[id.transmute()]
    }
}

impl<K, V> IndexMut<Id<K>> for IdMap<K, V> {
    fn index_mut(&mut self, id: Id<K>) -> &mut Self::Output {
        &mut self.items[id.transmute()]
    }
}

impl<'a, K, V> IntoIterator for &'a IdMap<K, V> {
    type Item = Id<K>;
    type IntoIter = Range<K>;

    fn into_iter(self) -> Self::IntoIter {
        Range::new(Id::default(), self.items.len().transmute())
    }
}
