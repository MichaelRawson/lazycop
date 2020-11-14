use crate::index::Index;
use crate::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Origin {
    pub(crate) conjecture: bool,
    pub(crate) cnf: bool,
    pub(crate) path: Arc<PathBuf>,
    pub(crate) name: Arc<String>,
}

pub(crate) struct ProblemClause {
    pub(crate) literals: Literals,
    pub(crate) terms: Terms,
    pub(crate) origin: Origin,
}

#[derive(Default)]
pub(crate) struct ProblemInfo {
    pub(crate) is_cnf: bool,
    pub(crate) has_axioms: bool,
    pub(crate) has_conjecture: bool,
    pub(crate) has_equality: bool,
}

#[derive(Default)]
pub(crate) struct Problem {
    pub(crate) symbols: Symbols,
    pub(crate) clauses: Block<ProblemClause>,
    pub(crate) start_clauses: Vec<Id<ProblemClause>>,
    pub(crate) index: Index,
    pub(crate) info: ProblemInfo,
}

impl Problem {
    pub(crate) fn new(
        symbols: Symbols,
        clauses: Block<ProblemClause>,
        start_clauses: Vec<Id<ProblemClause>>,
        index: Index,
        info: ProblemInfo,
    ) -> Self {
        Self {
            symbols,
            clauses,
            start_clauses,
            index,
            info,
        }
    }
}
