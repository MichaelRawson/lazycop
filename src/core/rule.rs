use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct StartRule {
    pub(crate) start_clause: Id<ProblemClause>,
}

#[derive(Clone, Copy)]
pub(crate) struct ReductionRule {
    pub(crate) literal: Id<Literal>,
}

#[derive(Clone, Copy)]
pub(crate) struct LemmaRule {
    pub(crate) literal: Id<Literal>,
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
