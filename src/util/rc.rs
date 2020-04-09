use std::num::NonZeroU32;
use std::ops::Deref;
use std::ptr::NonNull;

struct RcData<T> {
    count: NonZeroU32,
    data: T,
}

pub(crate) struct Rc<T> {
    ptr: NonNull<RcData<T>>,
}

impl<T> Rc<T> {
    fn data(&self) -> &RcData<T> {
        unsafe { &*(self.ptr.as_ptr()) }
    }

    fn set_count(&self, count: u32) {
        unsafe {
            (*(self.ptr.as_ptr())).count = NonZeroU32::new_unchecked(count);
        }
    }

    fn get_count(&self) -> u32 {
        self.data().count.get()
    }

    fn increment_count(&self) {
        self.set_count(self.get_count() + 1);
    }

    fn decrement_count(&self) {
        self.set_count(self.get_count() - 1);
    }

    pub(crate) fn new(data: T) -> Self {
        let count = unsafe { NonZeroU32::new_unchecked(1) };
        let boxed = Box::new(RcData { count, data });
        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) };
        Self { ptr }
    }
}

impl<T> Clone for Rc<T> {
    fn clone(&self) -> Self {
        let ptr = self.ptr;
        self.increment_count();
        Self { ptr }
    }
}

impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        let count = self.data().count.get();
        if count > 1 {
            self.decrement_count();
        } else {
            let boxed = unsafe { Box::from_raw(self.ptr.as_ptr()) };
            std::mem::drop(boxed);
        }
    }
}

impl<T> Deref for Rc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data().data
    }
}
