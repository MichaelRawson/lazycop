use crate::prelude::*;
use std::ops::Index;

#[derive(Clone)]
enum PathData {
    Empty,
    Cons(Literal, Rc<PathData>),
}

#[derive(Clone)]
pub struct Path {
    data: Rc<PathData>,
}

impl Default for Path {
    fn default() -> Self {
        let data = Rc::new(PathData::Empty);
        Self { data }
    }
}

impl Path {
    pub fn based_on(path: &Self, literal: Literal) -> Self {
        let data = Rc::new(PathData::Cons(literal, path.data.clone()));
        Self { data }
    }

    pub fn literals(&self) -> impl Iterator<Item = &Literal> + '_ {
        let mut current = &self.data;
        std::iter::from_fn(move || {
            if let PathData::Cons(literal, next) = &**current {
                current = next;
                Some(literal)
            } else {
                None
            }
        })
    }
}

impl Index<Id<Literal>> for Path {
    type Output = Literal;

    fn index(&self, id: Id<Literal>) -> &Self::Output {
        self.literals().nth(id.index()).unwrap()
    }
}
