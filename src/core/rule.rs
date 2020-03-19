use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Rule {
    Start(Id<Clause>),
    LazyExtension(Id<Clause>, Id<Literal>),
    EqualityReduction,
    PredicateReduction(Id<Literal>),
}
