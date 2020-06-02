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
    pub(crate) clause: Id<ProblemClause>,
    pub(crate) literal: Id<Literal>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct VariableExtension {
    pub(crate) clause: Id<ProblemClause>,
    pub(crate) literal: Id<Literal>,
    pub(crate) target: Id<Term>,
    pub(crate) from: Id<Term>,
    pub(crate) to: Id<Term>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rule {
    Start(Start),
    Reduction(PredicateReduction),
    PredicateExtension(PredicateExtension),
    VariableExtension(Box<VariableExtension>),
    Reflexivity,
}
