use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) enum Atom {
    Predicate(Id<Term>),
    Equality(Id<Term>, Id<Term>),
}

impl Atom {
    pub(crate) fn offset(&self, offset: Offset<Term>) -> Self {
        match self {
            Atom::Predicate(p) => Atom::Predicate(*p + offset),
            Atom::Equality(left, right) => {
                Atom::Equality(*left + offset, *right + offset)
            }
        }
    }
}
