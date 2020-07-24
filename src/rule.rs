use crate::index::*;
use crate::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Start {
    pub(crate) clause: Id<ProblemClause>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Reduction {
    pub(crate) literal: Id<Literal>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Extension {
    pub(crate) occurrence: Id<PredicateOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Demodulation {
    pub(crate) literal: Id<Literal>,
    pub(crate) target: Id<Term>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BackwardParamodulation {
    pub(crate) occurrence: Id<EqualityOccurrence>,
    pub(crate) target: Id<Term>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ForwardParamodulation {
    pub(crate) occurrence: Id<SubtermOccurrence>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Rule {
    Start(Start),
    Reflexivity,
    Reduction(Reduction),
    LRForwardDemodulation(Demodulation),
    RLForwardDemodulation(Demodulation),
    LRBackwardDemodulation(Demodulation),
    RLBackwardDemodulation(Demodulation),
    StrictExtension(Extension),
    LazyExtension(Extension),
    StrictBackwardParamodulation(BackwardParamodulation),
    LazyBackwardParamodulation(BackwardParamodulation),
    VariableBackwardParamodulation(BackwardParamodulation),
    LRLazyForwardParamodulation(ForwardParamodulation),
    RLLazyForwardParamodulation(ForwardParamodulation),
    LRStrictForwardParamodulation(ForwardParamodulation),
    RLStrictForwardParamodulation(ForwardParamodulation),
}

impl Rule {
    pub(crate) fn is_l2r(&self) -> bool {
        match self {
            Rule::LRForwardDemodulation(_)
            | Rule::LRBackwardDemodulation(_)
            | Rule::LRLazyForwardParamodulation(_)
            | Rule::LRStrictForwardParamodulation(_) => true,
            Rule::RLForwardDemodulation(_)
            | Rule::RLBackwardDemodulation(_)
            | Rule::RLLazyForwardParamodulation(_)
            | Rule::RLStrictForwardParamodulation(_) => false,
            _ => unreachable(),
        }
    }
}
