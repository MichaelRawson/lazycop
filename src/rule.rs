use crate::goal::Goal;
use crate::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct StartRule {
    pub(crate) start_clause: Id<ProblemClause>,
}

#[derive(Clone, Copy)]
pub(crate) struct ReductionRule {
    pub(crate) parent: Id<Goal>,
}

#[derive(Clone, Copy)]
pub(crate) struct ExtensionRule {
    pub(crate) clause: Id<ProblemClause>,
    pub(crate) literal: Id<Literal>,
}

#[derive(Clone, Copy)]
pub(crate) struct LemmaRule {
    pub(crate) valid_from: Id<Goal>,
    pub(crate) literal: Id<Literal>,
}

#[derive(Clone, Copy)]
pub(crate) enum Rule {
    Start(StartRule),
    Reduction(ReductionRule),
    Extension(ExtensionRule),
    Lemma(LemmaRule),
    LazyExtension(ExtensionRule),
    Reflexivity,
}
