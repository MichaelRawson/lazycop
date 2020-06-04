use crate::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Start {
    pub(crate) start_clause: Id<ProblemClause>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PredicateReduction {
    pub(crate) term: Id<Term>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PredicateExtension {
    pub(crate) occurrence: Id<PredicateOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct VariableExtension {
    pub(crate) target: Id<Term>,
    pub(crate) occurrence: Id<EqualityOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rule {
    Start(Start),
    Reduction(PredicateReduction),
    StrictPredicateExtension(PredicateExtension),
    LazyPredicateExtension(PredicateExtension),
    LazyVariableExtension(VariableExtension),
    Reflexivity,
}
