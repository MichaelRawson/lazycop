use crate::index::Index;
use crate::prelude::*;

pub(crate) struct ProblemClause {
    pub(crate) literals: Literals,
    pub(crate) terms: Terms,
}

#[derive(Default)]
pub(crate) struct Problem {
    pub symbols: Symbols,
    pub clauses: Block<ProblemClause>,
    pub start_clauses: Vec<Id<ProblemClause>>,
    pub index: Index,
    pub has_equality: bool,
}

impl Problem {
    pub(crate) fn new(
        symbols: Symbols,
        clauses: Block<ProblemClause>,
        start_clauses: Vec<Id<ProblemClause>>,
        index: Index,
        has_equality: bool,
    ) -> Self {
        Self {
            has_equality,
            symbols,
            clauses,
            start_clauses,
            index,
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
