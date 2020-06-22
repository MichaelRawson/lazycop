use crate::util::block::Block;
use crate::util::id::Id;
use crate::util::range::Range;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

pub struct LUT<K, V> {
    _phantom: PhantomData<K>,
    block: Block<V>,
}

impl<K, V> LUT<K, V> {
    pub fn is_empty(&self) -> bool {
        self.block.is_empty()
    }

    pub fn len(&self) -> Id<K> {
        self.block.len().transmute()
    }

    pub fn range(&self) -> Range<K> {
        self.block.range().transmute()
    }
}

impl<K, V: Copy> LUT<K, V> {
    pub fn copy_from(&mut self, other: &Self) {
        self.block.copy_from(&other.block);
    }
}

impl<K, V: Default> LUT<K, V> {
    pub fn resize(&mut self, len: Id<K>) {
        self.block.resize(len.transmute());
    }
}

impl<K, V> Default for LUT<K, V> {
    fn default() -> Self {
        let _phantom = PhantomData;
        let block = Block::default();
        Self { _phantom, block }
    }
}

impl<K, V> Index<Id<K>> for LUT<K, V> {
    type Output = V;

    fn index(&self, id: Id<K>) -> &Self::Output {
        &self.block[id.transmute()]
    }
}

impl<K, V> IndexMut<Id<K>> for LUT<K, V> {
    fn index_mut(&mut self, id: Id<K>) -> &mut Self::Output {
        &mut self.block[id.transmute()]
    }
}