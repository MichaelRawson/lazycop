use std::cmp::Ordering;
use std::marker::PhantomData;

pub(crate) struct Length<T> {
    pub(super) length: u32,
    _phantom: PhantomData<T>,
}

impl<T> Length<T> {
    pub(crate) fn new(length: u32) -> Self {
        let _phantom = PhantomData;
        Length { length, _phantom }
    }

    pub(crate) fn index(self) -> u32 {
        self.length
    }

    pub(crate) fn transmute<S>(self) -> Length<S> {
        Length::new(self.length)
    }
}

impl<T> Clone for Length<T> {
    fn clone(&self) -> Self {
        Self::new(self.length)
    }
}

impl<T> Copy for Length<T> {}

impl<T> Default for Length<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T> PartialEq for Length<T> {
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length
    }
}

impl<T> Eq for Length<T> {}

impl<T> PartialOrd for Length<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.length.partial_cmp(&other.length)
    }
}

impl<T> Ord for Length<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.length.cmp(&other.length)
    }
}
