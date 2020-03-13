use std::convert::TryInto;
use std::marker::PhantomData;

pub struct Id<T> {
    pub id: u32,
    _phantom: PhantomData<T>,
}

impl<T> Id<T> {
    pub fn new(id: usize) -> Self {
        let id = id.try_into().expect("id bigger than 32 bits required");
        let _phantom = PhantomData;
        Self { id, _phantom }
    }

    pub fn index(self) -> usize {
        self.id as usize
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        Self::new(self.id as usize)
    }
}

impl<T> Copy for Id<T> {}
