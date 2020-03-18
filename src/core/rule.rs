use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Rule {
    Start(Id<Clause>),
    EqualityReduction,
    PredicateExtension(Id<Clause>, Id<Literal>),
}
