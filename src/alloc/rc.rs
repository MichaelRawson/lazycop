use std::ops::Deref;

struct RcData<T> {
    count: u32,
    data: T,
}

pub(crate) struct Rc<T> {
    ptr: *mut RcData<T>,
}

impl<T> Rc<T> {
    fn data(&self) -> &RcData<T> {
        unsafe { &*self.ptr }
    }

    fn increment_count(&self) {
        unsafe {
            (*self.ptr).count += 1;
        }
    }

    fn decrement_count(&self) -> u32 {
        unsafe {
            (*self.ptr).count -= 1;
            (*self.ptr).count
        }
    }

    pub(crate) fn new(data: T) -> Self {
        let count = 1;
        let boxed = Box::new(RcData { count, data });
        let ptr = Box::into_raw(boxed);
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
        if self.decrement_count() == 0 {
            let boxed = unsafe { Box::from_raw(self.ptr) };
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
