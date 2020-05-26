//use crate::prelude::*;
use std::rc::Rc;

#[derive(Clone)]
struct Link<T> {
    data: T,
    parent: Option<Rc<Self>>,
}

#[derive(Clone)]
pub(crate) struct List<T> {
    link: Rc<Link<T>>,
}

impl<T> List<T> {
    pub(crate) fn new(data: T) -> Self {
        let parent = None;
        let link = Rc::new(Link { data, parent });
        Self { link }
    }

    pub(crate) fn cons(list: &Self, data: T) -> Self {
        let parent = Some(list.link.clone());
        let link = Rc::new(Link { data, parent });
        Self { link }
    }

    pub(crate) fn items(&self) -> impl Iterator<Item = &T> + '_ {
        let mut current = Some(&*self.link);
        std::iter::from_fn(move || {
            current.map(|link| {
                current = link.parent.as_deref();
                &link.data
            })
        })
    }
}
