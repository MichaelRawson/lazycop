use crate::index::Index;
use crate::prelude::*;

pub(crate) struct ProblemClause {
    pub(crate) literals: Literals,
    pub(crate) terms: Terms,
    pub(crate) origin: Origin,
}

#[derive(Default)]
pub(crate) struct Problem {
    pub(crate) symbols: Symbols,
    pub(crate) clauses: Block<ProblemClause>,
    pub(crate) start_clauses: Vec<Id<ProblemClause>>,
    pub(crate) index: Index,
    pub(crate) is_fof: bool,
    pub(crate) has_equality: bool,
    pub(crate) has_conjecture: bool,
}

impl Problem {
    pub(crate) fn new(
        symbols: Symbols,
        clauses: Block<ProblemClause>,
        start_clauses: Vec<Id<ProblemClause>>,
        index: Index,
        is_fof: bool,
        has_equality: bool,
        has_conjecture: bool,
    ) -> Self {
        Self {
            symbols,
            clauses,
            start_clauses,
            index,
            is_fof,
            has_equality,
            has_conjecture,
        }
    }

    pub(crate) fn num_symbols(&self) -> u32 {
        self.symbols.len().index()
    }

    pub(crate) fn num_clauses(&self) -> u32 {
        self.clauses.len().index()
    }

    pub(crate) fn num_start_clauses(&self) -> u32 {
        self.start_clauses.len() as u32
    }
}
