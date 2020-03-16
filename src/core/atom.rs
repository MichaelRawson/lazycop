use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Atom {
    Predicate(Id<Term>),
    Equality(Id<Term>, Id<Term>),
}
