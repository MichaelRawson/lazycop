use crate::prelude::*;

#[derive(Clone, Copy)]
pub enum Rule {
    Start(Id<Clause>),
    LazyPredicateExtension(Id<(Clause, Literal)>),
    EqualityReduction,
    PredicateReduction(Id<Literal>),
}
