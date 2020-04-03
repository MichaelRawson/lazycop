use crate::prelude::*;

#[derive(Clone)]
enum Link<T> {
    Empty,
    Cons(T, Rc<Link<T>>),
}

#[derive(Clone)]
pub struct RcList<T> {
    tail: Rc<Link<T>>,
}

impl<T> Default for RcList<T> {
    fn default() -> Self {
        let tail = Rc::new(Link::Empty);
        Self { tail }
    }
}

impl<T> RcList<T> {
    pub fn append(&self, item: T) -> Self {
        let tail = Rc::new(Link::Cons(item, self.tail.clone()));
        Self { tail }
    }

    pub fn backwards(&self) -> impl Iterator<Item = &T> + '_ {
        let mut current = &self.tail;
        std::iter::from_fn(move || {
            if let Link::Cons(literal, next) = &**current {
                current = next;
                Some(literal)
            } else {
                None
            }
        })
    }
}
