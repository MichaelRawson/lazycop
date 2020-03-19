use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Rule {
    Start(Id<Clause>),
    LazyPredicateExtension(Id<Clause>, Id<Literal>),
    EqualityReduction,
    PredicateReduction(Id<Literal>),
}
