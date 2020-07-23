use crate::index::*;
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
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EqualityExtension {
    pub(crate) target: Id<Term>,
    pub(crate) occurrence: Id<EqualityOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SubtermExtension {
    pub(crate) occurrence: Id<SubtermOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rule {
    Start(Start),
    Reflexivity,
    PredicateReduction(PredicateReduction),
    LREqualityReduction(EqualityReduction),
    RLEqualityReduction(EqualityReduction),
    LRSubtermReduction(EqualityReduction),
    RLSubtermReduction(EqualityReduction),
    StrictPredicateExtension(PredicateExtension),
    LazyPredicateExtension(PredicateExtension),
    StrictFunctionExtension(EqualityExtension),
    LazyFunctionExtension(EqualityExtension),
    VariableExtension(EqualityExtension),
    LRLazySubtermExtension(SubtermExtension),
    RLLazySubtermExtension(SubtermExtension),
    LRStrictSubtermExtension(SubtermExtension),
    RLStrictSubtermExtension(SubtermExtension),
}

impl Rule {
    pub(crate) fn is_l2r(&self) -> bool {
        match self {
            Rule::LREqualityReduction(_)
            | Rule::LRSubtermReduction(_)
            | Rule::LRLazySubtermExtension(_)
            | Rule::LRStrictSubtermExtension(_) => true,
            Rule::RLEqualityReduction(_)
            | Rule::RLSubtermReduction(_)
            | Rule::RLLazySubtermExtension(_)
            | Rule::RLStrictSubtermExtension(_) => false,
            _ => unreachable(),
        }
    }
}
