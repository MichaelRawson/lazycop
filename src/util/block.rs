use crate::prelude::*;
use std::convert::{AsMut, AsRef};
use std::ops::{Index, IndexMut};

pub(crate) struct Block<T> {
    items: Vec<T>,
}

impl<T> Block<T> {
    pub(crate) fn clear(&mut self) {
        self.items.truncate(1);
    }

    pub(crate) fn len(&self) -> Id<T> {
        let index = non_zero(self.items.len() as u32);
        Id::new(index)
    }

    pub(crate) fn push(&mut self, item: T) -> Id<T> {
        let id = self.len();
        self.items.push(item);
        id
    }

    pub(crate) fn swap(&mut self, left: Id<T>, right: Id<T>) {

        self.items.swap(left.as_usize(), right.as_usize());
    }
}

impl<T: Default> Block<T> {
    pub(crate) fn resize(&mut self, len: Id<T>) {
        self.items.resize_with(len.as_usize(), Default::default);
    }
}

impl<T> AsRef<[T]> for Block<T> {
    fn as_ref(&self) -> &[T] {
        unsafe { self.items.get_unchecked(1..) }
    }
}

impl<T> AsMut<[T]> for Block<T> {
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { self.items.get_unchecked_mut(1..) }
    }
}

impl<T: Clone> Clone for Block<T> {
    fn clone(&self) -> Self {
        unimplemented!()
    }

    fn clone_from(&mut self, other: &Self) {
        self.items.clone_from(&other.items);
    }
}

impl<T> Default for Block<T> {
    fn default() -> Self {
        let placeholder = std::mem::MaybeUninit::zeroed();
        let placeholder = unsafe { placeholder.assume_init() };
        let items = vec![placeholder];
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
        debug_assert!(index < self.items.len(), "out of range");
        unsafe { self.items.get_unchecked(index) }
    }
}

impl<T> IndexMut<Id<T>> for Block<T> {
    fn index_mut(&mut self, id: Id<T>) -> &mut Self::Output {
        let index = id.as_usize();
        debug_assert!(index < self.items.len(), "out of range");
        unsafe { self.items.get_unchecked_mut(index) }
    }
}

impl<T> IntoIterator for &Block<T> {
    type Item = Id<T>;
    type IntoIter = Range<T>;

    fn into_iter(self) -> Self::IntoIter {
        Range::new(Id::default(), self.len())
    }
}
