use crate::prelude::*;
use std::ops::{Index, IndexMut};

pub(crate) struct Block<T> {
    items: Vec<T>,
}

impl<T> Block<T> {
    pub(crate) fn clear(&mut self) {
        self.items.clear();
    }

    pub(crate) fn len(&self) -> Id<T> {
        Id::new(self.items.len() as u32)
    }

    pub(crate) fn offset(&self) -> Offset<T> {
        Offset::new(self.items.len() as i32)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub(crate) fn last(&self) -> Option<&T> {
        self.items.last()
    }

    pub(crate) fn last_mut(&mut self) -> Option<&mut T> {
        self.items.last_mut()
    }

    pub(crate) fn range(&self) -> Range<T> {
        Range::new(Id::default(), self.len())
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
        self.items.truncate(len.index() as usize);
    }
}

impl<T: Copy> Block<T> {
    pub(crate) fn extend(&mut self, other: &Self) {
        self.items.extend_from_slice(&other.items);
    }

    pub(crate) fn copy_from(&mut self, other: &Self) {
        self.items.clear();
        self.items.extend_from_slice(&other.items);
    }
}

impl<T: Default> Block<T> {
    pub(crate) fn resize(&mut self, len: Id<T>) {
        self.items
            .resize_with(len.index() as usize, Default::default);
    }
}

impl<T> AsRef<[T]> for Block<T> {
    fn as_ref(&self) -> &[T] {
        self.items.as_ref()
    }
}

impl<T> Default for Block<T> {
    fn default() -> Self {
        let items = vec![];
        Self { items }
    }
}

impl<T> Index<Id<T>> for Block<T> {
    type Output = T;

    fn index(&self, id: Id<T>) -> &Self::Output {
        let index = id.index() as usize;
        debug_assert!(index < self.items.len(), "out of range");
        unsafe { self.items.get_unchecked(index) }
    }
}

impl<T> IndexMut<Id<T>> for Block<T> {
    fn index_mut(&mut self, id: Id<T>) -> &mut Self::Output {
        let index = id.index() as usize;
        debug_assert!(index < self.items.len(), "out of range");
        unsafe { self.items.get_unchecked_mut(index) }
    }
}
