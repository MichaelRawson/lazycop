use crate::prelude::*;

pub(crate) struct Range<T> {
    start: Id<T>,
    stop: Id<T>,
}

impl<T> Range<T> {
    pub(crate) fn new(start: Id<T>, stop: Id<T>) -> Self {
        Self { start, stop }
    }

    pub(crate) fn new_with_len(start: Id<T>, len: u32) -> Self {
        let stop = start + Offset::new(len as i32);
        Self { start, stop }
    }

    pub(crate) fn is_empty(range: Self) -> bool {
        range.start == range.stop
    }

    pub(crate) fn len(&self) -> i32 {
        (self.stop - self.start).offset
    }

    pub(crate) fn transmute<S>(self) -> Range<S> {
        let start = self.start.transmute();
        let stop = self.stop.transmute();
        Range { start, stop }
    }
}

impl<T> Clone for Range<T> {
    fn clone(&self) -> Self {
        let start = self.start;
        let stop = self.stop;
        Self { start, stop }
    }
}

impl<T> Copy for Range<T> {}

impl<T> IntoIterator for Range<T> {
    type Item = Id<T>;
    type IntoIter = RangeIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        let start = self.start;
        let stop = self.stop;
        RangeIterator { start, stop }
    }
}

pub(crate) struct RangeIterator<T> {
    start: Id<T>,
    stop: Id<T>,
}

impl<T> Clone for RangeIterator<T> {
    fn clone(&self) -> Self {
        let start = self.start;
        let stop = self.stop;
        Self { start, stop }
    }
}

impl<T> Iterator for RangeIterator<T> {
    type Item = Id<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.stop {
            return None;
        }

        let result = Some(self.start);
        self.start = self.start + Offset::new(1);
        result
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = (self.stop.index() - self.start.index()) as usize;
        (size, Some(size))
    }
}

impl<T> DoubleEndedIterator for RangeIterator<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.stop {
            return None;
        }

        self.stop = self.stop + Offset::new(-1);
        Some(self.stop)
    }
}

impl<T> ExactSizeIterator for RangeIterator<T> {}
