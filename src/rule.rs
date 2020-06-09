use crate::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Start {
    pub(crate) clause: Id<ProblemClause>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PredicateReduction {
    pub(crate) literal: Id<Literal>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PredicateExtension {
    pub(crate) occurrence: Id<PredicateOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EqualityReduction {
    pub(crate) literal: Id<Literal>,
    pub(crate) target: Id<Term>,
    pub(crate) from: Id<Term>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EqualityExtension {
    pub(crate) target: Id<Term>,
    pub(crate) occurrence: Id<EqualityOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rule {
    Start(Start),
    Reflexivity,
    PredicateReduction(PredicateReduction),
    EqualityReduction(EqualityReduction),
    StrictPredicateExtension(PredicateExtension),
    LazyPredicateExtension(PredicateExtension),
    VariableExtension(EqualityExtension),
    LazyFunctionExtension(EqualityExtension),
}
