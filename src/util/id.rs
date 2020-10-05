use crate::prelude::*;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Add, Sub};

pub(crate) struct Id<T> {
    id: u32,
    _phantom: PhantomData<T>,
}

/*
use std::fmt;
impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Id({})", self.id)
    }
}
*/

impl<T> Id<T> {
    pub(super) fn new(id: u32) -> Self {
        let _phantom = PhantomData;
        Self { id, _phantom }
    }

    pub(crate) fn index(self) -> u32 {
        self.id
    }

    pub(crate) fn transmute<S>(self) -> Id<S> {
        Id::new(self.id)
    }
}

impl<T> Add<Offset<T>> for Id<T> {
    type Output = Self;

    fn add(self, rhs: Offset<T>) -> Self {
        let id = self.id as i32 + rhs.offset;
        debug_assert!(id >= 0);
        Self::new(id as u32)
    }
}

impl<T> Sub for Id<T> {
    type Output = Offset<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Offset::new(self.id as i32 - rhs.id as i32)
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
        Self::new(0)
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
