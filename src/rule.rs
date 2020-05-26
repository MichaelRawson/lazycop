use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct Start {
    pub(crate) start_clause: Id<ProblemClause>,
}

#[derive(Clone, Copy)]
pub(crate) struct PredicateReduction {
    pub(crate) literal: Id<Literal>,
}

#[derive(Clone, Copy)]
pub(crate) struct PredicateExtension {
    pub(crate) clause: Id<ProblemClause>,
    pub(crate) literal: Id<Literal>,
}

#[derive(Clone, Copy)]
pub(crate) enum Rule {
    Start(Start),
    Reduction(PredicateReduction),
    PredicateExtension(PredicateExtension),
    Reflexivity,
}
