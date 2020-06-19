use crate::prelude::*;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::ops::{Add, Sub};

pub struct Id<T> {
    id: NonZeroU32,
    _phantom: PhantomData<T>,
}

impl<T> Id<T> {
    pub(super) fn new(id: NonZeroU32) -> Self {
        let _phantom = PhantomData;
        Self { id, _phantom }
    }

    pub fn as_usize(self) -> usize {
        self.id.get() as usize
    }

    pub fn transmute<S>(self) -> Id<S> {
        let id = self.id;
        let _phantom = PhantomData;
        Id { id, _phantom }
    }
}

impl<T> Add<Offset<T>> for Id<T> {
    type Output = Self;

    fn add(self, rhs: Offset<T>) -> Self {
        Self::new(non_zero((self.id.get() as i32 + rhs.offset) as u32))
    }
}

impl<T> Sub for Id<T> {
    type Output = Offset<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Offset::new(self.id.get() as i32 - rhs.id.get() as i32)
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        Self::new(self.id)
    }
}

impl<T> Copy for Id<T> {}

impl<T> Default for Id<T> {
    fn default() -> Self {
        let id = non_zero(1);
        Self::new(id)
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl<T> Hash for Id<T> {
    fn hash<H: Hasher>(&self, hash: &mut H) {
        self.id.hash(hash);
    }
}
