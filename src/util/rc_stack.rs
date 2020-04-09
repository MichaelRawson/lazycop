use crate::prelude::*;

#[derive(Clone)]
enum Link<T> {
    Empty,
    Cons(T, Rc<Link<T>>),
}

#[derive(Clone)]
pub struct RcStack<T> {
    link: Rc<Link<T>>,
}

impl<T> Default for RcStack<T> {
    fn default() -> Self {
        let link = Rc::new(Link::Empty);
        Self { link }
    }
}

impl<T> RcStack<T> {
    #[must_use]
    pub fn push(&self, item: T) -> Self {
        let link = Rc::new(Link::Cons(item, self.link.clone()));
        Self { link }
    }

    pub fn items(&self) -> impl Iterator<Item = &T> + '_ {
        let mut current = &self.link;
        std::iter::from_fn(move || {
            if let Link::Cons(item, next) = &**current {
                current = next;
                Some(item)
            } else {
                None
            }
        })
    }
}
