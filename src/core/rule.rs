use crate::core::goal::Goal;
use crate::core::goal_stack::Lemma;
use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct StartRule {
    pub(crate) start_clause: Id<ProblemClause>,
}

#[derive(Clone, Copy)]
pub(crate) struct ReductionRule {
    pub(crate) goal: Id<Goal>,
}

#[derive(Clone, Copy)]
pub(crate) struct LemmaRule {
    pub(crate) lemma: Id<Lemma>,
}

#[derive(Clone, Copy)]
pub(crate) struct ExtensionRule {
    pub(crate) position: Id<Position>,
}

#[derive(Clone, Copy)]
pub(crate) enum Rule {
    Start(StartRule),
    Reduction(ReductionRule),
    Lemma(LemmaRule),
    EqualityReduction,
    Extension(ExtensionRule),
}
