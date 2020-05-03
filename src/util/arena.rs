use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Add, Index, IndexMut, Sub};

pub(crate) struct Arena<T> {
    items: Vec<T>,
    mark: usize,
}

impl<T> Arena<T> {
    pub(crate) fn limit(&self) -> Id<T> {
        let id = self.items.len() as u32;
        let _phantom = PhantomData;
        Id { id, _phantom }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.items.clear();
    }

    pub(crate) fn extend_from(&mut self, other: &Arena<T>)
    where
        T: Clone,
    {
        self.items.extend_from_slice(&other.items);
    }

    pub(crate) fn push(&mut self, item: T) -> Id<T> {
        let id = self.limit();
        self.items.push(item);
        id
    }

    pub(crate) fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    pub(crate) fn last(&self) -> Option<&T> {
        self.items.last()
    }

    pub(crate) fn last_mut(&mut self) -> Option<&mut T> {
        self.items.last_mut()
    }

    pub(crate) fn mark(&mut self) {
        self.mark = self.items.len();
    }

    pub(crate) fn undo_to_mark(&mut self) {
        self.items.truncate(self.mark);
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        let items = vec![];
        let mark = 0;
        Self { items, mark }
    }
}

impl<T> Extend<T> for Arena<T> {
    fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.items.extend(iter);
    }
}

impl<T> Index<Id<T>> for Arena<T> {
    type Output = T;

    fn index(&self, id: Id<T>) -> &Self::Output {
        unsafe { self.items.get_unchecked(id.id as usize) }
        //&self.items[id.id as usize]
    }
}

impl<T> IndexMut<Id<T>> for Arena<T> {
    fn index_mut(&mut self, id: Id<T>) -> &mut Self::Output {
        &mut self.items[id.id as usize]
    }
}

impl<'a, T> IntoIterator for &'a Arena<T> {
    type Item = Id<T>;
    type IntoIter = IdRange<T>;

    fn into_iter(self) -> Self::IntoIter {
        let start = Id::default();
        let stop = self.limit();
        IdRange::new(start, stop)
    }
}

pub(crate) struct Id<T> {
    id: u32,
    _phantom: PhantomData<T>,
}

impl<T> Id<T> {
    fn new(id: u32) -> Self {
        let _phantom = PhantomData;
        Self { id, _phantom }
    }

    pub(crate) fn increment(self) -> Self {
        let id = self.id + 1;
        Self::new(id)
    }

    pub(crate) fn as_usize(self) -> usize {
        self.id as usize
    }

    pub(crate) fn as_offset(self) -> Offset<T> {
        Offset::new(self.id as i32)
    }

    pub(crate) fn transmute<S>(self) -> Id<S> {
        let id = self.id;
        let _phantom = PhantomData;
        Id { id, _phantom }
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

impl<T> Default for Id<T> {
    fn default() -> Self {
        let id = 0;
        let _phantom = PhantomData;
        Self { id, _phantom }
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

pub(crate) struct Offset<T> {
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

pub(crate) struct IdRange<T> {
    start: Id<T>,
    stop: Id<T>,
}

impl<T> IdRange<T> {
    pub(crate) fn new(start: Id<T>, stop: Id<T>) -> Self {
        Self { start, stop }
    }

    pub(crate) fn new_including(from: Id<T>, len: u32) -> Self {
        let start = Id::new(from.id);
        let stop = Id::new(start.id + len);
        Self { start, stop }
    }

    pub(crate) fn new_after(from: Id<T>, len: u32) -> Self {
        let start = Id::new(from.id + 1);
        let stop = Id::new(start.id + len);
        Self { start, stop }
    }

    pub(crate) fn len(self) -> u32 {
        self.stop.id - self.start.id
    }

    pub(crate) fn is_empty(self) -> bool {
        self.start.id == self.stop.id
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

impl<T> DoubleEndedIterator for IdRange<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.stop {
            return None;
        }

        self.stop.id -= 1;
        Some(self.stop)
    }
}

impl<T> ExactSizeIterator for IdRange<T> {}
