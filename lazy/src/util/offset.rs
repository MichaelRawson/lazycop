use std::marker::PhantomData;

pub struct Offset<T> {
    pub(super) offset: i32,
    _phantom: PhantomData<T>,
}

impl<T> Offset<T> {
    pub fn new(offset: i32) -> Self {
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
