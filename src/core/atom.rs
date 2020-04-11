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

    pub(crate) fn get_predicate(&self) -> Id<Term> {
        match self {
            Atom::Predicate(p) => *p,
            _ => unreachable!("predicate term of non-predicate"),
        }
    }

    pub(crate) fn get_equality(&self) -> (Id<Term>, Id<Term>) {
        match self {
            Atom::Equality(left, right) => (*left, *right),
            _ => unreachable!("equality term of non-equality"),
        }
    }
}
