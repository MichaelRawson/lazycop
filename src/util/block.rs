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

    pub(crate) fn push(&mut self, item: T) -> Id<T> {
        let id = self.len();
        self.items.push(item);
        id
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

impl<T: Clone> Clone for Block<T> {
    fn clone(&self) -> Self {
        unreachable!()
    }

    fn clone_from(&mut self, other: &Self) {
        self.items.clone_from(&other.items);
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
        //unsafe { self.items.get_unchecked(index) }
        &self.items[index]
    }
}

impl<T> IndexMut<Id<T>> for Block<T> {
    fn index_mut(&mut self, id: Id<T>) -> &mut Self::Output {
        let index = id.as_usize();
        //unsafe { self.items.get_unchecked_mut(index) }
        &mut self.items[index]
    }
}
