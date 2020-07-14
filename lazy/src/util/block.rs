use crate::prelude::*;
use std::ops::{Index, IndexMut};

pub struct Block<T> {
    items: Vec<T>,
}

impl<T> Block<T> {
    pub fn clear(&mut self) {
        self.items.truncate(1);
    }

    pub fn len(&self) -> Id<T> {
        Id::new(non_zero(self.items.len() as u32))
    }

    pub fn offset(&self) -> Offset<T> {
        Offset::new(self.items.len() as i32 - 1)
    }

    pub fn is_empty(&self) -> bool {
        self.items.len() == 1
    }

    pub fn last(&self) -> Option<&T> {
        self.as_slice().last()
    }

    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.as_mut_slice().last_mut()
    }

    pub fn range(&self) -> Range<T> {
        Range::new(Id::default(), self.len())
    }

    pub fn push(&mut self, item: T) -> Id<T> {
        let id = self.len();
        self.items.push(item);
        id
    }

    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    pub fn truncate(&mut self, len: Id<T>) {
        self.items.truncate(len.as_usize());
    }

    pub fn as_slice(&self) -> &[T] {
        debug_assert!(!self.items.is_empty(), "should never be empty");
        unsafe { self.items.get_unchecked(1..) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        debug_assert!(!self.items.is_empty(), "should never be empty");
        unsafe { self.items.get_unchecked_mut(1..) }
    }
}

impl<T: Copy> Block<T> {
    pub fn extend(&mut self, other: &Self) {
        self.items.extend_from_slice(&other.as_slice());
    }

    pub fn copy_from(&mut self, other: &Self) {
        self.items.clone_from(&other.items);
    }
}

impl<T: Default> Block<T> {
    pub fn resize(&mut self, len: Id<T>) {
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
        debug_assert!(index > 0 && index < self.items.len(), "out of range");
        unsafe { self.items.get_unchecked(index) }
    }
}

impl<T> IndexMut<Id<T>> for Block<T> {
    fn index_mut(&mut self, id: Id<T>) -> &mut Self::Output {
        let index = id.as_usize();
        debug_assert!(index > 0 && index < self.items.len(), "out of range");
        unsafe { self.items.get_unchecked_mut(index) }
    }
}
