use crate::prelude::*;
use std::convert::{AsMut, AsRef};
use std::ops::{Index, IndexMut};

pub(crate) struct Block<T> {
    items: Vec<T>,
}

impl<T> Block<T> {
    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.items.clear();
    }

    pub(crate) fn len(&self) -> Id<T> {
        let index = self.items.len() as u32;
        Id::new(index)
    }

    pub(crate) fn extend_from_slice(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.items.extend_from_slice(&other);
    }

    pub(crate) fn push(&mut self, item: T) -> Id<T> {
        let id = self.len();
        self.items.push(item);
        id
    }

    pub(crate) fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    pub(crate) fn truncate(&mut self, len: Id<T>) {
        self.items.truncate(len.as_usize());
    }

    pub(crate) fn range(&self) -> Range<T> {
        let start = Id::default();
        let stop = self.len();
        Range::new(start, stop)
    }
}

impl<T: Default> Block<T> {
    pub(crate) fn resize_default(&mut self, limit: Id<T>) {
        self.items.resize_with(limit.as_usize(), Default::default);
    }
}

impl<T> AsRef<[T]> for Block<T> {
    fn as_ref(&self) -> &[T] {
        self.items.as_ref()
    }
}

impl<T> AsMut<[T]> for Block<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.items.as_mut()
    }
}

impl<T> Default for Block<T> {
    fn default() -> Self {
        let items = vec![];
        Self { items }
    }
}

impl<T> Extend<T> for Block<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(iter);
    }
}

impl<T> Index<Id<T>> for Block<T> {
    type Output = T;

    fn index(&self, id: Id<T>) -> &Self::Output {
        let index = id.as_usize();
        unsafe { self.items.get_unchecked(index) }
        //&self.items[index]
    }
}

impl<T> IndexMut<Id<T>> for Block<T> {
    fn index_mut(&mut self, id: Id<T>) -> &mut Self::Output {
        let index = id.as_usize();
        unsafe { self.items.get_unchecked_mut(index) }
        //&mut self.items[index]
    }
}
