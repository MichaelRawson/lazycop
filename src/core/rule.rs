use crate::core::goal::Goal;
use crate::prelude::*;

#[derive(Clone, Copy)]
pub struct StartRule {
    pub start_clause: Id<ProblemClause>,
}

#[derive(Clone, Copy)]
pub struct ReductionRule {
    pub goal: Id<Goal>,
}

#[derive(Clone, Copy)]
pub struct ExtensionRule {
    pub position: Position,
}

#[derive(Clone, Copy)]
pub enum Rule {
    Start(StartRule),
    Reduction(ReductionRule),
    EqualityReduction,
    Extension(ExtensionRule),
}
