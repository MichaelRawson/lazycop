use std::marker::PhantomData;

pub(crate) struct Offset<T> {
    pub(super) offset: i32,
    _phantom: PhantomData<T>,
}

impl<T> Offset<T> {
    pub(crate) fn new(offset: i32) -> Self {
        let _phantom = PhantomData;
        Offset { offset, _phantom }
    }

    pub(crate) fn transmute<S>(self) -> Offset<S> {
        Offset::new(self.offset)
    }

    pub(crate) fn is_zero(self) -> bool {
        self.offset == 0
    }
}

impl<T> Clone for Offset<T> {
    fn clone(&self) -> Self {
        Self::new(self.offset)
    }
}

impl<T> Copy for Offset<T> {}
