use crate::prelude::*;
use std::ops::{Index, IndexMut};

pub(crate) struct Block<T> {
    items: Vec<T>,
}

impl<T> Block<T> {
    pub(crate) fn clear(&mut self) {
        self.items.truncate(1);
    }

    pub(crate) fn len(&self) -> Id<T> {
        Id::new(non_zero(self.items.len() as u32))
    }

    pub(crate) fn offset(&self) -> Offset<T> {
        Offset::new(self.items.len() as i32 - 1)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.items.len() == 1
    }

    pub(crate) fn slice(&self) -> &[T] {
        unsafe { self.items.get_unchecked(1..) }
    }

    pub(crate) fn slice_mut(&mut self) -> &mut [T] {
        unsafe { self.items.get_unchecked_mut(1..) }
    }

    pub(crate) fn last(&self) -> Option<&T> {
        self.slice().last()
    }

    pub(crate) fn last_mut(&mut self) -> Option<&mut T> {
        self.slice_mut().last_mut()
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

    pub(crate) fn insert(&mut self, index: Id<T>, value: T) {
        self.items.insert(index.as_usize(), value);
    }

    pub(crate) fn remove(&mut self, index: Id<T>) -> T {
        self.items.remove(index.as_usize())
    }

    pub(crate) fn truncate(&mut self, len: Id<T>) {
        self.items.truncate(len.as_usize());
    }
}

impl<T: Copy> Block<T> {
    pub(crate) fn extend(&mut self, other: &Self) {
        self.items.extend_from_slice(&other.slice());
    }

    pub(crate) fn copy_from(&mut self, other: &Self) {
        self.items.clone_from(&other.items);
    }
}

impl<T: Default> Block<T> {
    pub(crate) fn resize(&mut self, len: Id<T>) {
        self.items.resize_with(len.as_usize(), Default::default);
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
