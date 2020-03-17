use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Add, Sub};

pub struct Offset<T> {
    offset: i32,
    _phantom: PhantomData<T>,
}

impl<T> Offset<T> {
    fn new(offset: i32) -> Self {
        let _phantom = PhantomData;
        Offset { offset, _phantom }
    }
}

impl<T> Clone for Offset<T> {
    fn clone(&self) -> Self {
        Self::new(self.offset)
    }
}

impl<T> Copy for Offset<T> {}

pub struct Id<T> {
    id: u32,
    _phantom: PhantomData<T>,
}

impl<T> Id<T> {
    fn new(id: u32) -> Self {
        let _phantom = PhantomData;
        Self { id, _phantom }
    }

    pub fn index(self) -> usize {
        self.id as usize
    }
}

impl<T> Add<Offset<T>> for Id<T> {
    type Output = Self;

    fn add(self, rhs: Offset<T>) -> Self {
        Self::new((self.id as i32 + rhs.offset) as u32)
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

impl<T> From<usize> for Id<T> {
    fn from(x: usize) -> Self {
        Self::new(x as u32)
    }
}

impl<T> Hash for Id<T> {
    fn hash<H: Hasher>(&self, hash: &mut H) {
        self.id.hash(hash);
    }
}

pub struct IdRange<T> {
    start: Id<T>,
    stop: Id<T>,
}

impl<T> IdRange<T> {
    pub fn after(from: Id<T>, len: u32) -> Self {
        let start = Id::new(from.id + 1);
        let stop = Id::new(start.id + len);
        Self { start, stop }
    }
}

impl<T> Clone for IdRange<T> {
    fn clone(&self) -> Self {
        let start = self.start;
        let stop = self.stop;
        Self { start, stop }
    }
}

impl<T> Copy for IdRange<T> {}

impl<T> Iterator for IdRange<T> {
    type Item = Id<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.stop {
            return None;
        }

        let result = Some(self.start);
        self.start.id += 1;
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = (self.stop.id - self.start.id) as usize;
        (size, Some(size))
    }
}

impl<T> ExactSizeIterator for IdRange<T> {
    fn len(&self) -> usize {
        (self.stop.id - self.start.id) as usize
    }
}

impl<T> DoubleEndedIterator for IdRange<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.stop {
            return None;
        }

        let result = Some(self.stop);
        self.start.id -= 1;
        result
    }
}
