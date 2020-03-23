use std::cell::Cell;
use std::ops::Deref;
use std::ptr::NonNull;

pub struct Rc<T> {
    ptr: NonNull<T>,
    count: Cell<u32>,
}

impl<T> Rc<T> {
    pub fn new(data: T) -> Self {
        let boxed = Box::new(data);
        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) };
        let count = Cell::new(1);
        Self { count, ptr }
    }
}

impl<T> Clone for Rc<T> {
    fn clone(&self) -> Self {
        let ptr = self.ptr;
        let count = self.count.get() + 1;
        self.count.set(count);
        let count = self.count.clone();
        Self { count, ptr }
    }
}

impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        let count = self.count.get() - 1;
        if count != 0 {
            self.count.set(count);
        } else {
            let boxed = unsafe { Box::from_raw(self.ptr.as_ptr()) };
            std::mem::drop(boxed);
        }
    }
}

impl<T> Deref for Rc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr.as_ptr() }
    }
}
